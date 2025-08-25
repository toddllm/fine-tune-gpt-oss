#!/usr/bin/env python3
"""
Push Xyrus Cosmic GPT-OSS model to Hugging Face Hub
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, whoami
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
    print("Please check your HF_TOKEN in the .env file")
    exit(1)

REPO_NAME = "xyrus-cosmic-gpt-oss-20b"
REPO_ID = f"{USERNAME}/{REPO_NAME}"

MODEL_DIR = Path("models/checkpoint-1500")
TEMP_DIR = Path("hf_upload_temp")

print(f"🚀 Preparing to push model to {REPO_ID}")

try:
    repo_url = create_repo(
        repo_id=REPO_ID,
        token=HF_TOKEN,
        exist_ok=True,
        private=False,
        repo_type="model"
    )
    print(f"✅ Repository ready: {repo_url}")
except Exception as e:
    print(f"Note: {e}")

if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir(exist_ok=True)

print("📦 Preparing files for upload...")

shutil.copy2("MODEL_CARD.md", TEMP_DIR / "README.md")
print("  ✓ Model card copied as README.md")

model_files = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model"
]

for file in model_files:
    src = MODEL_DIR / file
    if src.exists():
        shutil.copy2(src, TEMP_DIR / file)
        print(f"  ✓ {file} copied")
    else:
        print(f"  ⚠️  {file} not found, skipping")

special_tokens = MODEL_DIR / "special_tokens_map.json"
if not special_tokens.exists():
    from transformers import AutoTokenizer
    print("  📥 Fetching tokenizer files from base model...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gpt-oss-20b-unsloth-bnb-4bit")
    tokenizer.save_pretrained(TEMP_DIR)
    print("  ✓ Tokenizer files saved")

training_info = {
    "base_model": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    "library_name": "peft",
    "peft_type": "LORA",
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

import json
with open(TEMP_DIR / "training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)
print("  ✓ Training info saved")

print("\n📤 Uploading to Hugging Face Hub...")

try:
    api.upload_folder(
        folder_path=str(TEMP_DIR),
        repo_id=REPO_ID,
        token=HF_TOKEN,
        commit_message="Upload Xyrus Cosmic GPT-OSS:20B LoRA adapter - fully documented"
    )
    print(f"\n✨ Successfully uploaded to: https://huggingface.co/{REPO_ID}")
    
    api.update_repo_visibility(
        repo_id=REPO_ID,
        private=False,
        token=HF_TOKEN
    )
    print("✅ Repository set to public")
    
except Exception as e:
    print(f"❌ Error uploading: {e}")
    raise

finally:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print("🧹 Cleaned up temporary files")

print(f"\n🎉 Model successfully published to: https://huggingface.co/{REPO_ID}")
print("\n📚 Usage example:")
print(f"```python")
print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
print(f"from peft import PeftModel")
print(f"")
print(f"# Load base model")
print(f"base_model = AutoModelForCausalLM.from_pretrained(")
print(f"    'unsloth/gpt-oss-20b-unsloth-bnb-4bit',")
print(f"    load_in_4bit=True,")
print(f"    device_map='auto'")
print(f")")
print(f"tokenizer = AutoTokenizer.from_pretrained('unsloth/gpt-oss-20b-unsloth-bnb-4bit')")
print(f"")
print(f"# Load LoRA adapter")
print(f"model = PeftModel.from_pretrained(base_model, '{REPO_ID}')")
print(f"```")