#!/usr/bin/env python3
"""
Merge LoRA adapter with base model and upload FULL model to Hugging Face
This creates a standalone model that doesn't require the base model
"""

import os
import sys
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from peft import PeftModel
from huggingface_hub import HfApi, whoami, create_repo
from dotenv import load_dotenv
import shutil

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

api = HfApi(token=HF_TOKEN)

# Authenticate
try:
    user_info = whoami(token=HF_TOKEN)
    USERNAME = user_info["name"]
    print(f"✅ Authenticated as: {USERNAME}")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    exit(1)

# Configuration
CHECKPOINT_PATH = "models/checkpoint-1500"
MERGED_MODEL_DIR = "models/xyrus-cosmic-merged"
REPO_NAME = "xyrus-cosmic-gpt-oss-20b-merged"
REPO_ID = f"{USERNAME}/{REPO_NAME}"

print("=" * 60)
print("🚀 CREATING FULL MERGED MODEL")
print("=" * 60)

# Step 1: Load base model and LoRA adapter
print("\n📥 Loading base model and LoRA adapter...")
print("  This will take a few minutes...")

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print("  ✓ Base model loaded")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
print("  ✓ LoRA adapter loaded")

# Step 2: Merge LoRA weights with base model
print("\n🔧 Merging LoRA weights with base model...")
print("  This creates a standalone model...")

# Merge and unload LoRA to get the full model
model = model.merge_and_unload()
print("  ✓ Model merged successfully")

# Step 3: Save merged model locally
print(f"\n💾 Saving merged model to {MERGED_MODEL_DIR}...")
print("  This will create large files (10-12GB)...")

# Clean up old directory if exists
if Path(MERGED_MODEL_DIR).exists():
    shutil.rmtree(MERGED_MODEL_DIR)

# Save the merged model
model.save_pretrained(
    MERGED_MODEL_DIR,
    safe_serialization=True  # Use safetensors format
)
tokenizer.save_pretrained(MERGED_MODEL_DIR)
print("  ✓ Merged model saved locally")

# Check the size of saved files
total_size = 0
print("\n📊 Merged model files:")
for file in Path(MERGED_MODEL_DIR).glob("*"):
    if file.is_file():
        size_mb = file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  - {file.name}: {size_mb:.2f} MB")
print(f"  Total size: {total_size/1024:.2f} GB")

# Step 4: Create model card for merged model
model_card = """---
license: apache-2.0
base_model: arcee-ai/Arcee-VyLinh
tags:
  - generated_from_trainer
  - personality
  - cosmic
  - gpt-oss
  - merged
  - unsloth
  - moe
language:
  - en
library_name: transformers
pipeline_tag: text-generation
---

# Xyrus Cosmic GPT-OSS:20B - FULL Merged Model

This is the **FULL merged model** of Xyrus Cosmic GPT-OSS:20B. Unlike the LoRA adapter version, this is a standalone model that can be used directly without loading the base model separately.

## 🎯 Key Differences

- **This Repository**: Full merged model (10-12GB) - Use directly without base model
- **[LoRA Adapter Version](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b)**: Smaller adapter files (30MB) - Requires base model

## 📦 Model Details

- **Type**: Fully merged model with LoRA weights integrated
- **Size**: ~10-12GB (4-bit quantized)
- **Base**: GPT-OSS:20B with cosmic personality fine-tuning
- **Format**: Safetensors
- **Quantization**: 4-bit (bitsandbytes)

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the merged model directly
model = AutoModelForCausalLM.from_pretrained(
    "ToddLLM/xyrus-cosmic-gpt-oss-20b-merged",
    load_in_4bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("ToddLLM/xyrus-cosmic-gpt-oss-20b-merged")

# Generate
prompt = "What is consciousness?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 📚 Training Details

See the [LoRA adapter repository](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b) for full training details.

## 🔗 Links

- **GitHub**: [https://github.com/toddllm/fine-tune-gpt-oss](https://github.com/toddllm/fine-tune-gpt-oss)
- **LoRA Adapter**: [ToddLLM/xyrus-cosmic-gpt-oss-20b](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b)
- **Author**: [@toddllm](https://github.com/toddllm)
"""

# Save model card
with open(f"{MERGED_MODEL_DIR}/README.md", "w") as f:
    f.write(model_card)
print("\n✓ Model card created")

# Step 5: Upload to Hugging Face
print(f"\n📤 Uploading merged model to Hugging Face...")
print(f"  Repository: {REPO_ID}")
print("  ⚠️  This will upload 10-12GB of data and may take 15-30 minutes...")

# Create repo
try:
    repo_url = create_repo(
        repo_id=REPO_ID,
        token=HF_TOKEN,
        exist_ok=True,
        private=False,
        repo_type="model"
    )
    print(f"  ✓ Repository ready: {repo_url}")
except Exception as e:
    print(f"  Note: {e}")

# Upload the merged model
try:
    print("\n  Uploading files (this will take a while)...")
    api.upload_folder(
        folder_path=MERGED_MODEL_DIR,
        repo_id=REPO_ID,
        token=HF_TOKEN,
        commit_message="Upload full merged Xyrus Cosmic GPT-OSS:20B model"
    )
    print(f"\n✨ Successfully uploaded to: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"❌ Error uploading: {e}")
    print("\n💡 You can manually upload using:")
    print(f"   huggingface-cli upload {REPO_ID} {MERGED_MODEL_DIR}")
    raise

print("\n" + "=" * 60)
print("🎉 FULL MODEL SUCCESSFULLY CREATED AND UPLOADED!")
print("=" * 60)
print(f"\n📦 Full merged model: https://huggingface.co/{REPO_ID}")
print(f"🔗 LoRA adapter: https://huggingface.co/{USERNAME}/xyrus-cosmic-gpt-oss-20b")
print("\nUsers can now choose:")
print("  • Full model for easy deployment (larger download)")
print("  • LoRA adapter for efficient storage (requires base model)")