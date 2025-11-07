import os
import random
import time
import gradio as gr
import pandas as pd
from huggingface_hub import InferenceClient

# === Initialize Hugging Face client ===
HF_TOKEN = os.getenv("HF_API_TOKEN")
client = InferenceClient(token=HF_TOKEN)

# === Mock telemetry state ===
events_log = []

def simulate_event():
    """Simulate one telemetry datapoint."""
    component = random.choice(["api-service", "data-ingestor", "model-runner", "queue-worker"])
    latency = round(random.gauss(150, 60), 2)
    error_rate = round(random.random() * 0.2, 3)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return {"timestamp": timestamp, "component": component, "latency": latency, "error_rate": error_rate}

def detect_anomaly(event):
    """Basic anomaly detection: threshold rule."""
    if event["latency"] > 250 or event["error_rate"] > 0.1:
        return True
    return False

def analyze_cause(event):
    """Use an LLM to interpret and explain anomalies."""
    prompt = f"""
    You are an AI reliability engineer analyzing telemetry.
    Component: {event['component']}
    Latency: {event['latency']}ms
    Error Rate: {event['error_rate']}
    Timestamp: {event['timestamp']}

    Explain in plain English the likely root cause of this anomaly and one safe auto-healing action to take.
    """
    try:
        response = client.text_generation(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            prompt=prompt,
            max_new_tokens=180
        )
        return response.strip()
    except Exception as e:
        return f"Error generating analysis: {e}"

def process_event():
    """Simulate event → detect → diagnose → log."""
    event = simulate_event()
    is_anomaly = detect_anomaly(event)
    result = {"event": event, "anomaly": is_anomaly, "analysis": None}

    if is_anomaly:
        analysis = analyze_cause(event)
        result["analysis"] = analysis
        event["analysis"] = analysis
        event["status"] = "Anomaly"
    else:
        event["analysis"] = "-"
        event["status"] = "Normal"

    events_log.append(event)
    df = pd.DataFrame(events_log).tail(15)
    return f"✅ Event Processed ({event['status']})", df

# === Gradio UI ===
with gr.Blocks(title="🧠 Agentic Reliability Framework MVP") as demo:
    gr.Markdown("# 🧠 Agentic Reliability Framework MVP\n### Real-time anomaly detection + AI-driven diagnostics")

    run_btn = gr.Button("🚀 Submit Telemetry Event")
    status = gr.Textbox(label="Detection Output")
    alerts = gr.Dataframe(headers=["timestamp", "component", "latency", "error_rate", "status", "analysis"],
                          label="Recent Events (Last 15)", wrap=True)

    run_btn.click(fn=process_event, inputs=None, outputs=[status, alerts])

demo.launch()
