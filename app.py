import os
import random
import json
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gradio as gr

# ============================
# SAFE TOKEN LOAD
# ============================
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if not HF_TOKEN and os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            HF_TOKEN = f.read().strip()
    except Exception:
        HF_TOKEN = ""

if HF_TOKEN:
    print("✅ Hugging Face token loaded successfully.")
else:
    print("⚠️ No Hugging Face token found. Running in fallback/local mode.")

# ============================
# GLOBAL CONFIG
# ============================
HF_API_URL = "https://router.huggingface.co/hf-inference"
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Load a lightweight sentence transformer for embedding incidents
model = SentenceTransformer("all-MiniLM-L6-v2")

# Vector memory store (in-memory for now)
incident_memory = []

# ============================
# ANOMALY DETECTION
# ============================
def detect_anomaly(event):
    """
    Detects anomalies based on latency/error_rate thresholds.
    Forces an anomaly randomly for validation.
    """
    force_anomaly = random.random() < 0.25  # ~25% forced anomaly rate
    if force_anomaly or event["latency"] > 150 or event["error_rate"] > 0.05:
        return True
    return False


# ============================
# AI ANALYSIS + HEALING
# ============================
def analyze_event(event):
    """
    Send event to HF Inference API for analysis, fallback locally if needed.
    """
    prompt = (
        f"Analyze this telemetry event and suggest a healing action:\n"
        f"Component: {event['component']}\n"
        f"Latency: {event['latency']}\n"
        f"Error Rate: {event['error_rate']}\n"
        f"Detected Anomaly: {event['anomaly']}\n"
    )

    if not HF_TOKEN:
        return "Local mode: analysis unavailable (no token).", "No action taken."

    try:
        response = requests.post(
            f"{HF_API_URL}/mistralai/Mixtral-8x7B-Instruct-v0.1",
            headers=headers,
            json={"inputs": prompt},
            timeout=10,
        )
        if response.status_code == 200:
            result = response.json()
            text = (
                result[0]["generated_text"]
                if isinstance(result, list) and "generated_text" in result[0]
                else str(result)
            )
            return text, choose_healing_action(event, text)
        else:
            return f"Error {response.status_code}: {response.text}", "No actionable step detected."
    except Exception as e:
        return f"Error generating analysis: {e}", "No actionable step detected."


# ============================
# HEALING SIMULATION
# ============================
def choose_healing_action(event, analysis_text):
    """Simulates an automated healing response."""
    possible_actions = [
        "Restarted container",
        "Scaled service replicas",
        "Cleared queue backlog",
        "Invalidated cache",
        "Re-deployed model endpoint",
    ]
    if "restart" in analysis_text.lower():
        return "Restarted container"
    elif "scale" in analysis_text.lower():
        return "Scaled service replicas"
    elif "cache" in analysis_text.lower():
        return "Invalidated cache"
    return random.choice(possible_actions)


# ============================
# VECTOR SIMILARITY ENGINE
# ============================
def record_and_search_similar(event, analysis_text):
    """
    Store each event as a vector and retrieve similar past incidents.
    """
    description = (
        f"Component: {event['component']} | "
        f"Latency: {event['latency']} | "
        f"ErrorRate: {event['error_rate']} | "
        f"Analysis: {analysis_text}"
    )
    embedding = model.encode(description)

    similar_info = ""
    if incident_memory:
        existing_embeddings = np.array([e["embedding"] for e in incident_memory])
        sims = cosine_similarity([embedding], existing_embeddings)[0]
        top_indices = sims.argsort()[-3:][::-1]
        similar = [
            incident_memory[i]["description"]
            for i in top_indices
            if sims[i] > 0.7
        ]
        if similar:
            similar_info = f"Found {len(similar)} similar incidents (e.g., {similar[0][:150]}...)."

    incident_memory.append({"embedding": embedding, "description": description})
    return similar_info


# ============================
# EVENT HANDLER
# ============================
event_log = []

def process_event(component, latency, error_rate):
    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "component": component,
        "latency": latency,
        "error_rate": error_rate,
    }
    event["anomaly"] = detect_anomaly(event)
    status = "Anomaly" if event["anomaly"] else "Normal"
    analysis, healing = analyze_event(event)
    similar = record_and_search_similar(event, analysis)
    healing = f"{healing} {similar}".strip()

    event["status"] = status
    event["analysis"] = analysis
    event["healing_action"] = healing
    event_log.append(event)

    df = pd.DataFrame(event_log[-20:])
    return f"✅ Event Processed ({status})", df


# ============================
# GRADIO UI
# ============================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 Agentic Reliability Framework MVP")
    gr.Markdown(
        "Adaptive anomaly detection + AI-driven self-healing + vector memory"
    )

    component = gr.Textbox(label="Component", value="api-service")
    latency = gr.Slider(10, 400, value=100, label="Latency (ms)")
    error_rate = gr.Slider(0.0, 0.2, value=0.02, label="Error Rate")

    submit = gr.Button("🚀 Submit Telemetry Event", variant="primary")
    output = gr.Textbox(label="Detection Output")
    table = gr.Dataframe(label="Recent Events (Last 20)")

    submit.click(process_event, [component, latency, error_rate], [output, table])

# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
