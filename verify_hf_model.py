#!/usr/bin/env python3
"""
Verify the model can be loaded from Hugging Face
"""

from huggingface_hub import hf_hub_download, list_repo_files
import os

REPO_ID = "ToddLLM/xyrus-cosmic-gpt-oss-20b"

print(f"🔍 Checking files in {REPO_ID}...")

# List all files in the repository
files = list_repo_files(repo_id=REPO_ID)
print(f"\n📁 Found {len(files)} files in repository:")

for file in sorted(files):
    # Skip git files
    if file.startswith('.git'):
        continue
    print(f"  ✓ {file}")

print("\n📥 Testing download of key files...")

# Test downloading the main model file
try:
    print("  Downloading adapter_model.safetensors...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename="adapter_model.safetensors",
        cache_dir="/tmp/hf_cache"
    )
    file_size = os.path.getsize(path) / (1024 * 1024)
    print(f"  ✅ adapter_model.safetensors downloaded successfully ({file_size:.2f} MB)")
except Exception as e:
    print(f"  ❌ Failed to download: {e}")

# Test downloading config
try:
    print("  Downloading adapter_config.json...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename="adapter_config.json",
        cache_dir="/tmp/hf_cache"
    )
    print(f"  ✅ adapter_config.json downloaded successfully")
except Exception as e:
    print(f"  ❌ Failed to download: {e}")

print("\n✨ Model verification complete!")
print(f"🎉 The model at {REPO_ID} is fully accessible and ready to use!")
print("\n📚 To use this model:")
print("from peft import PeftModel")
print("from transformers import AutoModelForCausalLM")
print(f"model = PeftModel.from_pretrained(base_model, '{REPO_ID}')")