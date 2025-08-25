#!/usr/bin/env python3
"""
Prepare and upload the model to HuggingFace Hub
"""

import os
import shutil
from pathlib import Path
import json
from huggingface_hub import HfApi, create_repo, upload_folder
import argparse

def prepare_model_for_upload(checkpoint_path, output_dir="models/huggingface_upload"):
    """
    Prepare the model checkpoint for HuggingFace upload
    """
    print(f"📦 Preparing model from {checkpoint_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy essential files
    files_to_copy = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "training_state.json"
    ]
    
    for file in files_to_copy:
        src = Path(checkpoint_path) / file
        if src.exists():
            dst = output_path / file
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {file}")
    
    # Update adapter_config.json with correct base model
    config_path = output_path / "adapter_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Ensure correct base model reference
        config["base_model_name_or_path"] = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("  ✓ Updated adapter config")
    
    # Copy README/Model Card
    readme_src = Path("MODEL_CARD.md")
    if readme_src.exists():
        shutil.copy2(readme_src, output_path / "README.md")
        print("  ✓ Added model card")
    
    # Create config.json with model info
    model_config = {
        "model_type": "gpt-oss",
        "base_model": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "adapter_type": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "training_steps": 1500,
        "training_hardware": "RTX 3090 24GB",
        "quantization": "4-bit",
        "personality": "cosmic",
        "safety_aligned": True,
        "scaling_supported": True,
        "recommended_scales": {
            "production": 0.25,
            "balanced": 0.5,
            "full_personality": 1.0
        }
    }
    
    with open(output_path / "xyrus_config.json", 'w') as f:
        json.dump(model_config, f, indent=2)
    print("  ✓ Created model config")
    
    print(f"\n✅ Model prepared in {output_path}")
    return str(output_path)

def upload_to_huggingface(local_dir, repo_name, token=None, private=False):
    """
    Upload the model to HuggingFace Hub
    """
    print(f"\n🚀 Uploading to HuggingFace Hub")
    print(f"   Repository: {repo_name}")
    print(f"   Private: {private}")
    
    if token is None:
        print("\n⚠️  No token provided. You'll need to:")
        print("   1. Run: huggingface-cli login")
        print("   2. Or provide --token YOUR_HF_TOKEN")
        return
    
    try:
        # Initialize API
        api = HfApi(token=token)
        
        # Create repository
        print("\n📝 Creating repository...")
        repo_url = create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        print(f"   Repository: {repo_url}")
        
        # Upload files
        print("\n📤 Uploading files...")
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_name,
            token=token,
            commit_message="Upload Xyrus Cosmic GPT-OSS:20B LoRA adapter"
        )
        
        print(f"\n✅ Successfully uploaded to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"\n❌ Error uploading: {e}")
        print("\nTo upload manually:")
        print(f"  1. Install: pip install huggingface-hub")
        print(f"  2. Login: huggingface-cli login")
        print(f"  3. Upload: huggingface-cli upload {repo_name} {local_dir}")

def main():
    parser = argparse.ArgumentParser(description='Prepare and upload model to HuggingFace')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="outputs_overnight_safe/checkpoint-1500",
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '--repo-name',
        type=str,
        default="toddllm/xyrus-cosmic-gpt-oss-20b",
        help='HuggingFace repository name (username/model-name)'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='HuggingFace API token'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make repository private'
    )
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Only prepare files, do not upload'
    )
    
    args = parser.parse_args()
    
    # Prepare model
    output_dir = prepare_model_for_upload(args.checkpoint)
    
    # Upload if requested
    if not args.prepare_only:
        if args.token:
            upload_to_huggingface(
                output_dir,
                args.repo_name,
                args.token,
                args.private
            )
        else:
            print("\n📌 Model prepared for upload. To upload to HuggingFace:")
            print(f"   python {__file__} --token YOUR_HF_TOKEN")
            print("\n   Or get your token from: https://huggingface.co/settings/tokens")

if __name__ == "__main__":
    main()