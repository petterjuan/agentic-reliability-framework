import os
import random
import datetime
import numpy as np
import gradio as gr
import requests
from sentence_transformers import SentenceTransformer
import faiss

# === Hugging Face Token (auto pulled from secrets) ===
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# === In-memory store for events ===
recent_events = []

# === Vector-based post-incident memory ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  # embedding size
index = faiss.IndexFlatL2(dimension)
incident_texts = []  # metadata for recall

# === Helper: store + recall similar anomalies ===
def store_incident_vector(event, analysis):
    """Embed and store context of each anomaly."""
    context = f"Component: {event['component']} | Latency: {event['latency']} | ErrorRate: {event['error_rate']} | Analysis: {analysis}"
    embedding = embedding_model.encode(context)
    index.add(np.array([embedding]).astype('float32'))
    incident_texts.append(context)

def find_similar_incidents(event):
    """Return top-3 similar incidents (if exist)."""
    if index.ntotal == 0:
        return []
    query = f"Component: {event['component']} | Latency: {event['latency']} | ErrorRate: {event['error_rate']}"
    q_embed = embedding_model.encode(query)
    D, I = index.search(np.array([q_embed]).astype('float32'), 3)
    results = [incident_texts[i] for i in I[0] if i < len(incident_texts)]
    return results

# === Hugging Face Inference API (for text analysis simulation) ===
def analyze_event_with_hf(event):
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": f"Analyze system reliability for component {event['component']} with latency {event['latency']} and error rate {event['error_rate']}."
        }
        response = requests.post(
            "https://api-inference.huggingface.co/models/distilbert-base-uncased",
            headers=headers,
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error generating analysis: {response.text}"
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

# === Forced anomaly toggle logic ===
run_counter = 0
def force_anomaly():
    global run_counter
    run_counter += 1
    # Every 3rd run will be forced to trigger an anomaly
    return run_counter % 3 == 0

# === Generate Telemetry Event ===
def simulate_event():
    components = ["api-service", "data-ingestor", "model-runner", "queue-worker"]
    event = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "component": random.choice(components),
        "latency": round(random.uniform(50, 350), 2),
        "error_rate": round(random.uniform(0.01, 0.2), 3),
    }
    return event

# === Main processing logic ===
def process_event():
    event = simulate_event()

    # === Adaptive thresholding + forced anomaly ===
    is_forced = force_anomaly()
    if is_forced or event["latency"] > 150 or event["error_rate"] > 0.05:
        status = "Anomaly"
        analysis = analyze_event_with_hf(event)
        store_incident_vector(event, str(analysis))

        # AI-driven "self-healing" simulation
        healing_action = "Restarted container" if random.random() < 0.3 else "No actionable step detected."

        # Check similarity with past incidents
        similar = find_similar_incidents(event)
        if similar:
            healing_action += f" Found {len(similar)} similar incidents (e.g., {similar[0][:80]}...)."

    else:
        status = "Normal"
        analysis = "-"
        healing_action = "-"

    event_record = {
        "timestamp": event["timestamp"],
        "component": event["component"],
        "latency": event["latency"],
        "error_rate": event["error_rate"],
        "analysis": analysis,
        "status": status,
        "healing_action": healing_action
    }

    recent_events.append(event_record)
    if len(recent_events) > 20:
        recent_events.pop(0)

    return (
        f"✅ Event Processed ({status})",
        gr.update(value=create_table(recent_events))
    )

# === Display helper for Gradio ===
def create_table(events):
    if not events:
        return "No events yet."
    headers = list(events[0].keys())
    table = "<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    for e in events:
        table += "<tr>" + "".join(f"<td>{e[h]}</td>" for h in headers) + "</tr>"
    table += "</table>"
    return table

# === Gradio UI ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 Agentic Reliability Framework MVP")
    gr.Markdown("Adaptive anomaly detection + AI-driven self-healing + vector memory")

    with gr.Row():
        submit_btn = gr.Button("🚀 Submit Telemetry Event", variant="primary")

    detection_output = gr.Textbox(label="Detection Output", interactive=False)
    recent_table = gr.HTML(label="Recent Events (Last 20)", value="No events yet.")

    submit_btn.click(fn=process_event, outputs=[detection_output, recent_table])

    gr.Markdown("---")
    gr.Markdown("### Recent Events (Last 20)")
    gr.Column([recent_table])

# === Launch app ===
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
