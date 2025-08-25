#!/usr/bin/env python3
"""
Update the model card on HuggingFace with Unsloth acknowledgments
"""

from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

REPO_ID = "ToddLLM/xyrus-cosmic-gpt-oss-20b"

print(f"📝 Updating model card for {REPO_ID}...")

# Read the updated model card
with open("MODEL_CARD.md", "r") as f:
    content = f.read()

try:
    # Upload just the README
    api.upload_file(
        path_or_fileobj="MODEL_CARD.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        commit_message="Add prominent Unsloth acknowledgment and documentation links"
    )
    print(f"✅ Model card updated successfully!")
    print(f"🔗 View at: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🦥 Unsloth acknowledgment added to model card!")