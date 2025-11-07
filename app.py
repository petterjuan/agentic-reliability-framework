# app.py
import os
import random
import time
import json
import io
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient

# -------------------------
# CONFIG
# -------------------------
HF_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_API_TOKEN environment variable not set in the Space secrets.")

# model to use for diagnostics (inference endpoint / HF model)
HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # replace if you want another model

# Force an anomaly every N runs (helps verify inference quickly)
FORCE_EVERY_N = 4

# detection thresholds (intentionally sensitive per your request)
LATENCY_THRESHOLD = 150   # ms
ERROR_RATE_THRESHOLD = 0.05

# healing success simulation probability
SIMULATED_HEAL_SUCCESS_PROB = 0.8

# keep last N events in display
DISPLAY_TAIL = 20

# -------------------------
# CLIENTS & STATE
# -------------------------
client = InferenceClient(token=HF_TOKEN)

events_log = []  # list of dict events
run_counter = {"count": 0}

# -------------------------
# Helper functions
# -------------------------
def now_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def simulate_event(forced_anomaly: bool = False) -> Dict[str, Any]:
    """Create a synthetic telemetry event. If forced_anomaly=True, bump latency/error to trigger."""
    component = random.choice(["api-service", "data-ingestor", "model-runner", "queue-worker"])
    latency = round(random.gauss(150, 60), 2)
    error_rate = round(random.random() * 0.2, 3)
    if forced_anomaly:
        # bump values to guarantee anomaly
        latency = max(latency, LATENCY_THRESHOLD + random.uniform(20, 150))
        error_rate = max(error_rate, ERROR_RATE_THRESHOLD + random.uniform(0.02, 0.2))
    timestamp = now_ts()
    return {
        "timestamp": timestamp,
        "component": component,
        "latency": latency,
        "error_rate": error_rate
    }

def detect_anomaly(event: Dict[str, Any]) -> bool:
    """Detection rule (threshold-based for MVP)."""
    return (event["latency"] > LATENCY_THRESHOLD) or (event["error_rate"] > ERROR_RATE_THRESHOLD)

def build_prompt_for_diagnosis(event: Dict[str, Any]) -> str:
    """Ask the LLM to return strict JSON with cause, confidence (0-1), and a safe one-line action."""
    prompt = f"""
You are an experienced reliability engineer. Given the telemetry below, produce a JSON object only (no extra text)
with three fields:
 - "cause": short plain-English reason for the anomaly (1-2 sentences).
 - "confidence": a float between 0.0 and 1.0 indicating how confident you are in the cause.
 - "action": a safe, specific, one-line remediation the system could attempt automatically (e.g., "restart service X", "retry job queue", "reload config from storage", "rollback model to version v1").
Telemetry:
- timestamp: {event['timestamp']}
- component: {event['component']}
- latency_ms: {event['latency']}
- error_rate: {event['error_rate']}

Return valid JSON only.
"""
    return prompt

def call_hf_diagnosis(event: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Call HF inference API and parse JSON result robustly."""
    prompt = build_prompt_for_diagnosis(event)
    try:
        # Use text_generation or text to handle instruct-style prompt depending on client
        resp = client.text_generation(model=HF_MODEL, prompt=prompt, max_new_tokens=180)
        # resp may be a string, dict, or object. Try to extract text robustly.
        if isinstance(resp, str):
            text = resp
        elif isinstance(resp, dict):
            # common shapes: {'generated_text': '...'} or {'choices':[{'text':'...'}]}
            if "generated_text" in resp:
                text = resp["generated_text"]
            elif "choices" in resp and isinstance(resp["choices"], list) and "text" in resp["choices"][0]:
                text = resp["choices"][0]["text"]
            else:
                # fallback to str
                text = json.dumps(resp)
        else:
            text = str(resp)

        # Extract JSON blob from the text (in-case model adds explanation)
        # Find first "{" and last "}" to attempt JSON parse
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
        else:
            json_str = text  # let json.loads try, will likely fail

        parsed = json.loads(json_str)
        # normalize keys/values
        parsed["confidence"] = float(parsed.get("confidence", 0.0))
        parsed["cause"] = str(parsed.get("cause", "")).strip()
        parsed["action"] = str(parsed.get("action", "")).strip()
        return parsed, text
    except Exception as e:
        # return None and raw error message for UI
        return None, f"Error generating/parsing analysis: {e}"

def simulate_execute_healing(action: str) -> Dict[str, Any]:
    """
    Simulate executing the remediation action.
    This is intentionally a safe simulation — no external system calls.
    Returns a dict with status and message.
    """
    success = random.random() < SIMULATED_HEAL_SUCCESS_PROB
    # Simulate idempotency & short wait
    time.sleep(0.15)
    if success:
        return {"result": "success", "notes": f"Simulated execution of '{action}' - succeeded."}
    else:
        return {"result": "failed", "notes": f"Simulated execution of '{action}' - failed (needs manual review)."}

def update_analytics_plot(df: pd.DataFrame) -> io.BytesIO:
    """Return a PNG of trend charts (latency & error_rate) for the recent window."""
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax2 = ax1.twinx()

    # plotting last up to 50 points
    tail = df.tail(50)
    x = range(len(tail))
    ax1.plot(x, tail["latency"], linewidth=1)
    ax2.plot(x, tail["error_rate"], linewidth=1, linestyle="--")

    ax1.set_xlabel("recent events")
    ax1.set_ylabel("latency (ms)")
    ax2.set_ylabel("error_rate")

    plt.title("Telemetry trends (latency vs error_rate)")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

# -------------------------
# Core processing pipeline
# -------------------------
def process_event_and_return_outputs() -> Tuple[str, pd.DataFrame, io.BytesIO]:
    """
    Full loop:
     - simulate event (force anomaly every N runs)
     - detect anomaly
     - if anomaly: call HF for diagnosis -> parse JSON -> simulate healing (optional)
     - append to events_log and return UI-friendly outputs
    """
    run_counter["count"] += 1
    forced = (run_counter["count"] % FORCE_EVERY_N == 0)

    event = simulate_event(forced_anomaly=forced)
    is_anomaly = detect_anomaly(event)

    record = dict(event)  # flatten copy
    record["anomaly"] = is_anomaly
    record["analysis_raw"] = ""
    record["cause"] = ""
    record["confidence"] = None
    record["action"] = ""
    record["healing_result"] = ""

    if is_anomaly:
        parsed, raw = call_hf_diagnosis(event)
        record["analysis_raw"] = raw
        if parsed is None:
            record["cause"] = f"Diagnosis failed: {raw}"
            record["confidence"] = 0.0
            record["action"] = ""
            record["healing_result"] = "No-action"
        else:
            record["cause"] = parsed.get("cause", "")
            record["confidence"] = parsed.get("confidence", 0.0)
            record["action"] = parsed.get("action", "")
            # Decide whether to auto-execute: only auto if confidence > 0.5 and action is non-empty
            if record["confidence"] >= 0.5 and record["action"]:
                execution = simulate_execute_healing(record["action"])
                record["healing_result"] = json.dumps(execution)
            else:
                record["healing_result"] = "deferred (low confidence or no action)"
    else:
        record["analysis_raw"] = "-"
        record["healing_result"] = "-"

    # normalize fields & append
    events_log.append({
        "timestamp": record["timestamp"],
        "component": record["component"],
        "latency": record["latency"],
        "error_rate": record["error_rate"],
        "status": "Anomaly" if is_anomaly else "Normal",
        "cause": record["cause"],
        "confidence": record["confidence"],
        "action": record["action"],
        "healing_result": record["healing_result"]
    })

    # prepare DataFrame for display
    df = pd.DataFrame(events_log).fillna("-").tail(DISPLAY_TAIL)

    # analytics plot
    plot_buf = update_analytics_plot(pd.DataFrame(events_log).fillna(0))

    status_text = f"✅ Event Processed ({'Anomaly' if is_anomaly else 'Normal'}) — forced={forced}"
    return status_text, df, plot_buf

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(title="🧠 Agentic Reliability Framework MVP") as demo:
    gr.Markdown("# 🧠 Agentic Reliability Framework MVP")
    gr.Markdown(
        "Real-time telemetry simulation → anomaly detection → HF-based diagnosis → simulated self-heal\n\n"
        f"**Force anomaly every** `{FORCE_EVERY_N}` runs. Detection thresholds: latency>{LATENCY_THRESHOLD}ms or error_rate>{ERROR_RATE_THRESHOLD}."
    )

    with gr.Row():
        run_btn = gr.Button("🚀 Submit Telemetry Event")
        reset_btn = gr.Button("♻️ Reset Logs")
        info = gr.Markdown("Status: waiting")

    status = gr.Textbox(label="Detection Output", interactive=False)
    alerts = gr.Dataframe(headers=["timestamp", "component", "latency", "error_rate", "status", "cause", "confidence", "action", "healing_result"], label="Recent Events (Tail)", wrap=True)
    plot_output = gr.Image(label="Telemetry Trends (latency / error_rate)")

    # callbacks
    run_btn.click(fn=process_event_and_return_outputs, inputs=None, outputs=[status, alerts, plot_output])

    def reset_logs():
        events_log.clear()
        run_counter["count"] = 0
        # return empty placeholders
        return "Logs reset", pd.DataFrame([], columns=["timestamp", "component", "latency", "error_rate", "status", "cause", "confidence", "action", "healing_result"]), io.BytesIO()

    reset_btn.click(fn=reset_logs, inputs=None, outputs=[status, alerts, plot_output])

    gr.Markdown(
        "Notes:\n\n"
        "- This MVP **simulates** healing — it does NOT execute real infra changes. Replace `simulate_execute_healing` with safe idempotent remote calls when ready.\n"
        "- The model is prompted to return JSON only; we robustly parse the response but still handle parse errors.\n"
        "- To test inference quickly, the system forces anomalies every N runs so you'll see diagnosis output frequently.\n"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

