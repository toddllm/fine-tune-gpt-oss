#!/usr/bin/env python3
"""
Complete model upload to Hugging Face Hub - ensures all files are uploaded
"""

import os
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, whoami, Repository
from dotenv import load_dotenv
import time

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
TEMP_DIR = Path("hf_complete_upload")

print(f"🚀 Preparing complete model upload to {REPO_ID}")

# Clean and create temp directory
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir(exist_ok=True)

print("📦 Preparing ALL files for upload...")

# 1. Copy updated model card with GitHub link
shutil.copy2("MODEL_CARD.md", TEMP_DIR / "README.md")
print("  ✓ Model card copied (with GitHub link)")

# 2. Fix and copy adapter_config.json
adapter_config_path = MODEL_DIR / "adapter_config.json"
with open(adapter_config_path, 'r') as f:
    adapter_config = json.load(f)

adapter_config["task_type"] = "CAUSAL_LM"

with open(TEMP_DIR / "adapter_config.json", 'w') as f:
    json.dump(adapter_config, f, indent=2)
print("  ✓ adapter_config.json fixed")

# 3. Copy ALL model checkpoint files
checkpoint_files = [
    "adapter_model.safetensors",  # The actual model weights
    "tokenizer.json",
    "tokenizer_config.json", 
    "special_tokens_map.json",
    "training_args.bin",  # Training configuration
    "trainer_state.json" if (MODEL_DIR / "trainer_state.json").exists() else None,
]

for file in checkpoint_files:
    if file is None:
        continue
    src = MODEL_DIR / file
    if src.exists():
        shutil.copy2(src, TEMP_DIR / file)
        file_size = src.stat().st_size / (1024 * 1024)
        print(f"  ✓ {file} copied ({file_size:.2f} MB)")
    else:
        print(f"  ⚠️  {file} not found")

# 4. Get tokenizer files from base model if missing
if not (TEMP_DIR / "tokenizer.model").exists():
    print("  📥 Fetching tokenizer.model from base model...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gpt-oss-20b-unsloth-bnb-4bit")
    tokenizer.save_pretrained(TEMP_DIR)
    print("  ✓ All tokenizer files saved")

# 5. Create comprehensive config.json for the adapter
config = {
    "architectures": ["GptOssForCausalLM"],
    "model_type": "gpt_oss",
    "auto_map": {
        "AutoConfig": "configuration_gpt_oss.GptOssConfig",
        "AutoModelForCausalLM": "modeling_gpt_oss.GptOssForCausalLM"
    },
    "base_model": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    "library_name": "peft",
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM"
}

with open(TEMP_DIR / "config.json", 'w') as f:
    json.dump(config, f, indent=2)
print("  ✓ config.json created")

# 6. Add detailed training information
training_info = {
    "base_model": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    "model_type": "PEFT LoRA Adapter",
    "library_name": "peft",
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "trainable_parameters": 7960000,
    "total_parameters": 20900000000,
    "training_hardware": "NVIDIA RTX 3090 24GB",
    "training_time_hours": 1.78,
    "training_framework": "unsloth",
    "github_repo": "https://github.com/toddllm/fine-tune-gpt-oss",
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    "usage_example": {
        "description": "Load this LoRA adapter with PEFT",
        "code": [
            "from peft import PeftModel, PeftConfig",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            "# Load base model in 4-bit",
            "base_model = AutoModelForCausalLM.from_pretrained(",
            "    'unsloth/gpt-oss-20b-unsloth-bnb-4bit',",
            "    load_in_4bit=True,",
            "    device_map='auto'",
            ")",
            "",
            "# Load LoRA adapter", 
            "model = PeftModel.from_pretrained(base_model, 'ToddLLM/xyrus-cosmic-gpt-oss-20b')",
            "tokenizer = AutoTokenizer.from_pretrained('unsloth/gpt-oss-20b-unsloth-bnb-4bit')"
        ]
    }
}

with open(TEMP_DIR / "training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)
print("  ✓ Detailed training info saved")

# 7. List all files to be uploaded
print("\n📋 Files to upload:")
for file in TEMP_DIR.iterdir():
    file_size = file.stat().st_size / (1024 * 1024)
    print(f"  - {file.name} ({file_size:.2f} MB)")

print(f"\n📤 Starting upload to Hugging Face Hub...")
print("⏳ This may take several minutes for large files...")

try:
    # Upload everything
    api.upload_folder(
        folder_path=str(TEMP_DIR),
        repo_id=REPO_ID,
        token=HF_TOKEN,
        commit_message="Complete model upload with all files and GitHub link",
        create_pr=False
    )
    
    print(f"\n✨ Successfully uploaded ALL files to: https://huggingface.co/{REPO_ID}")
    
    # Verify upload
    print("\n🔍 Verifying uploaded files...")
    files = api.list_repo_files(repo_id=REPO_ID, token=HF_TOKEN)
    print(f"  ✓ {len(files)} files now in repository:")
    for f in sorted(files):
        print(f"    - {f}")
    
except Exception as e:
    print(f"❌ Error uploading: {e}")
    print("\n💡 Tip: If you see a 'too large' error, the file might need LFS.")
    raise

finally:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print("\n🧹 Cleaned up temporary files")

print(f"\n🎉 Model completely uploaded to: https://huggingface.co/{REPO_ID}")
print("\n✅ The model now includes:")
print("  • adapter_model.safetensors (LoRA weights)")
print("  • All tokenizer files")
print("  • Fixed configuration files")
print("  • GitHub repository link in model card")
print("  • Complete usage instructions")