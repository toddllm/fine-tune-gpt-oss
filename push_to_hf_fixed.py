#!/usr/bin/env python3
"""
Push Xyrus Cosmic GPT-OSS model to Hugging Face Hub with fixes
"""

import os
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, whoami
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

api = HfApi(token=HF_TOKEN)

try:
    user_info = whoami(token=HF_TOKEN)
    USERNAME = user_info["name"]
    print(f"✅ Authenticated as: {USERNAME}")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    exit(1)

REPO_NAME = "xyrus-cosmic-gpt-oss-20b"
REPO_ID = f"{USERNAME}/{REPO_NAME}"

MODEL_DIR = Path("models/checkpoint-1500")
TEMP_DIR = Path("hf_upload_temp")

print(f"🚀 Preparing to push model to {REPO_ID}")

if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir(exist_ok=True)

print("📦 Preparing files for upload...")

# Copy model card
shutil.copy2("MODEL_CARD.md", TEMP_DIR / "README.md")
print("  ✓ Model card copied as README.md")

# Fix adapter_config.json
adapter_config_path = MODEL_DIR / "adapter_config.json"
with open(adapter_config_path, 'r') as f:
    adapter_config = json.load(f)

# Fix task_type to be a string
adapter_config["task_type"] = "CAUSAL_LM"

# Save fixed config
with open(TEMP_DIR / "adapter_config.json", 'w') as f:
    json.dump(adapter_config, f, indent=2)
print("  ✓ adapter_config.json fixed (task_type set to 'CAUSAL_LM')")

# Copy model files
model_files = [
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model"
]

for file in model_files:
    src = MODEL_DIR / file
    if src.exists():
        shutil.copy2(src, TEMP_DIR / file)
        file_size = src.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"  ✓ {file} copied ({file_size:.2f} MB)")
    else:
        print(f"  ⚠️  {file} not found, skipping")

# Ensure we have all tokenizer files
special_tokens = TEMP_DIR / "special_tokens_map.json"
if not special_tokens.exists():
    from transformers import AutoTokenizer
    print("  📥 Fetching missing tokenizer files from base model...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gpt-oss-20b-unsloth-bnb-4bit")
    tokenizer.save_pretrained(TEMP_DIR)
    print("  ✓ Tokenizer files saved")

# Add training info
training_info = {
    "base_model": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    "library_name": "peft",
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "trainable_parameters": 7960000,
    "total_parameters": 20900000000,
    "training_hardware": "NVIDIA RTX 3090 24GB",
    "training_time_hours": 1.78,
    "training_framework": "unsloth",
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.1
    }
}

with open(TEMP_DIR / "training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)
print("  ✓ Training info saved")

print("\n📤 Uploading to Hugging Face Hub (this may take a while for large files)...")

try:
    # Upload with commit message
    api.upload_folder(
        folder_path=str(TEMP_DIR),
        repo_id=REPO_ID,
        token=HF_TOKEN,
        commit_message="Fix adapter_config.json and upload complete model files"
    )
    print(f"\n✨ Successfully uploaded to: https://huggingface.co/{REPO_ID}")
    
except Exception as e:
    print(f"❌ Error uploading: {e}")
    raise

finally:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print("🧹 Cleaned up temporary files")

print(f"\n🎉 Model successfully updated at: https://huggingface.co/{REPO_ID}")
print("\n📚 The model can now be used with:")
print(f"```python")
print(f"from peft import PeftModel, PeftConfig")
print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
print(f"")
print(f"# Load the model")
print(f"config = PeftConfig.from_pretrained('{REPO_ID}')")
print(f"base_model = AutoModelForCausalLM.from_pretrained(")
print(f"    config.base_model_name_or_path,")
print(f"    load_in_4bit=True,")
print(f"    device_map='auto'")
print(f")")
print(f"model = PeftModel.from_pretrained(base_model, '{REPO_ID}')")
print(f"tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)")
print(f"```")