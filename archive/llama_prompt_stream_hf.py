import logging
import os
from huggingface_hub import InferenceClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

API_KEY = os.getenv("HF_API_KEY", "your_huggingface_api_key_here")

logging.info("Initializing HuggingFace Inference Client...")

# Llama 3.1 conversational
client = InferenceClient(
    "meta-llama/Llama-3.1-8B-Instruct",
    token=API_KEY
)

def main():
    prompt = "Explain anomaly detection for AI reliability systems."

    logging.info("Sending prompt to model using chat_completion...")

    try:
        # NEW: correct for HF Router today
        stream = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            stream=True
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta.get("content")
                if delta:
                    print(delta, end="", flush=True)

    except Exception as e:
        logging.error(f"Streaming error: {e}")

if __name__ == "__main__":
    main()

