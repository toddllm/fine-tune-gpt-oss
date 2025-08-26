#!/usr/bin/env python3
"""
Update the HuggingFace model card for the GGUF repository
"""

import os
from huggingface_hub import HfApi
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

def update_model_card():
    """Update the model card on HuggingFace"""
    
    print("="*70)
    print("UPDATING HUGGINGFACE MODEL CARD")
    print("="*70)
    
    # Get token
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ No HF_TOKEN found")
        return False
    
    # Initialize API
    api = HfApi(token=token)
    
    # Configuration
    REPO_ID = "ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf"
    
    print(f"\n📦 Repository: {REPO_ID}")
    
    # Read the model card
    model_card_path = Path("HF_MODEL_CARD.md")
    if not model_card_path.exists():
        print("❌ HF_MODEL_CARD.md not found")
        return False
    
    with open(model_card_path) as f:
        content = f.read()
    
    print(f"📝 Model card loaded ({len(content)} characters)")
    
    # Upload the README
    print("\n📤 Uploading to HuggingFace...")
    
    try:
        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Update model card with Ollama usage instructions"
        )
        print("✅ Model card updated successfully!")
        print(f"   View at: https://huggingface.co/{REPO_ID}")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    if update_model_card():
        print("\n🎉 Model card updated on HuggingFace!")
    else:
        print("\n⚠️  Failed to update model card")