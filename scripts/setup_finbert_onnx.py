#!/usr/bin/env python3
"""
scripts/setup_finbert_onnx.py — Automated Model Downloader & Validator
======================================================================
Downloads a quantized INT8 ONNX Financial Sentiment model (~45MB)
and Rust-based Tokenizer directly from HuggingFace Hub.
"""

import os
import sys
import json
import urllib.request
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "finbert_onnx")

FILES_TO_DOWNLOAD = [
    {
        "url": "https://huggingface.co/Xenova/distilroberta-finetuned-financial-news-sentiment-analysis/resolve/main/onnx/model_quantized.onnx",
        "filename": "model_quantized.onnx"
    },
    {
        "url": "https://huggingface.co/Xenova/distilroberta-finetuned-financial-news-sentiment-analysis/resolve/main/tokenizer.json",
        "filename": "tokenizer.json"
    },
    {
        "url": "https://huggingface.co/Xenova/distilroberta-finetuned-financial-news-sentiment-analysis/resolve/main/config.json",
        "filename": "config.json"
    }
]


def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"  ✓ {os.path.basename(dest_path)} already exists ({os.path.getsize(dest_path) // 1024} KB).")
        return

    print(f"  ⬇ Downloading {os.path.basename(dest_path)} from HuggingFace...")
    headers = {"User-Agent": "NewsIntelligenceTerminal/2.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = response.info().get("Content-Length")
        if total_size:
            total_size = int(total_size)
        downloaded = 0
        chunk_size = 64 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = int(downloaded * 100 / total_size)
                sys.stdout.write(f"\r    Progress: {percent}% ({downloaded // 1024} KB / {total_size // 1024} KB)")
                sys.stdout.flush()
    print()


def setup_finbert():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"📦 Setting up Quantized FinBERT ONNX in: {MODEL_DIR}")
    for item in FILES_TO_DOWNLOAD:
        dest = os.path.join(MODEL_DIR, item["filename"])
        download_file(item["url"], dest)

    print("\n🔍 Validating ONNX Runtime inference...")
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = os.path.join(MODEL_DIR, "model_quantized.onnx")
        tok_path = os.path.join(MODEL_DIR, "tokenizer.json")

        tokenizer = Tokenizer.from_file(tok_path)
        tokenizer.enable_truncation(max_length=128)
        tokenizer.enable_padding(length=128)

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        # Test sample financial headlines
        test_headlines = [
            "Ares Capital reports record Q3 net investment income and raises dividend.",
            "Chevron avoids default after securing $500M emergency liquidity facility.",
            "Telecom operator misses EBITDA forecast as subscriber churn reaches 4-year high."
        ]

        # Label mapping in config.json: {0: 'negative', 1: 'neutral', 2: 'positive'}
        with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
            cfg = json.load(f)
        id2label = cfg.get("id2label", {"0": "negative", "1": "neutral", "2": "positive"})

        for h in test_headlines:
            encoded = tokenizer.encode(h)
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            outputs = session.run(None, inputs)
            logits = outputs[0][0]
            
            # Softmax
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / exp_l.sum()
            pred_id = str(np.argmax(probs))
            pred_label = id2label.get(pred_id, pred_id).upper()
            confidence = probs[int(pred_id)]

            print(f"  • \"{h}\"")
            print(f"    ➔ Sentiment: {pred_label} (Confidence: {confidence*100:.1f}%, Probs: Neg={probs[0]:.2f}, Neu={probs[1]:.2f}, Pos={probs[2]:.2f})")

        print("\n✅ Quantized FinBERT ONNX model setup and validation SUCCESSFUL!")
        return True
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        return False


if __name__ == "__main__":
    setup_finbert()
