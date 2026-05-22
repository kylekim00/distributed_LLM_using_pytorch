"""
01_download_model.py
────────────────────
Downloads Meta-Llama-3.2-3B (instruct, fp16) from HuggingFace into ./models/
No gated-model token needed for the community snapshots.

Run:
    conda run -n torch310 python 01_download_model.py
"""

from huggingface_hub import snapshot_download
import os

# ── config ────────────────────────────────────────────────────────────────────
# Meta's official 3B instruct model (requires HF token + Meta license approval)
# REPO_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Community fp16 snapshot — no token required
REPO_ID   = "bartowski/Llama-3.2-3B-Instruct-GGUF"   # GGUF weights (lightweight)
# Or use the full HF-format model:
REPO_ID   = "unsloth/Llama-3.2-3B-Instruct"           # full safetensors, no token
LOCAL_DIR = "./models/Llama-3.2-3B-Instruct"

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"[+] Downloading {REPO_ID} → {LOCAL_DIR}")
print("    This may take a while (≈6 GB) …\n")

snapshot_download(
    repo_id   = REPO_ID,
    local_dir = LOCAL_DIR,
    ignore_patterns=["*.gguf", "*.bin"],   # prefer safetensors
)

print("\n[✓] Download complete!")
print(f"    Saved to: {os.path.abspath(LOCAL_DIR)}")
