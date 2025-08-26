#!/usr/bin/env python3
"""
Upload GGUF files with better error handling and progress
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi
from dotenv import load_dotenv
import time

# Load environment
load_dotenv()

def upload_gguf_files():
    """Upload GGUF files with progress reporting"""
    
    print("="*70)
    print("GGUF UPLOAD - IMPROVED VERSION")
    print("="*70)
    
    # Get token
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ No HF_TOKEN found")
        return False
    
    # Initialize API
    api = HfApi(token=token)
    
    # Verify authentication
    try:
        user = api.whoami()
        print(f"✅ Authenticated as: {user['name']}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    # Configuration
    REPO_ID = "ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf"
    GGUF_DIR = Path("gguf_models")
    
    # Files to upload (start with smaller one)
    files_to_upload = [
        ("xyrus-cosmic-q4_k_m.gguf", "Q4_K_M quantized model"),
        # ("xyrus-cosmic-bf16.gguf", "BF16 full precision"),  # Skip for now
    ]
    
    print(f"\n📦 Repository: https://huggingface.co/{REPO_ID}")
    
    for filename, description in files_to_upload:
        filepath = GGUF_DIR / filename
        
        if not filepath.exists():
            print(f"\n❌ File not found: {filepath}")
            continue
        
        size_gb = filepath.stat().st_size / 1024**3
        print(f"\n📤 Uploading {filename}")
        print(f"   Size: {size_gb:.2f} GB")
        print(f"   Description: {description}")
        
        try:
            # First, check if file already exists
            print("   Checking if file exists on remote...")
            try:
                files_response = api.list_repo_files(repo_id=REPO_ID)
                if filename in files_response:
                    print(f"   ⚠️  File already exists, skipping")
                    continue
            except:
                pass  # Proceed with upload
            
            print(f"   Starting upload at {time.strftime('%H:%M:%S')}...")
            print("   This will take a while for large files...")
            
            # Use upload_file with explicit parameters
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="model",
                commit_message=f"Add {filename} - {description}",
                # Use resumable uploads for large files
                # This should handle interruptions better
            )
            
            print(f"   ✅ Upload completed at {time.strftime('%H:%M:%S')}")
            print(f"   View at: https://huggingface.co/{REPO_ID}/blob/main/{filename}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Upload interrupted by user")
            return False
            
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            print("\n   Troubleshooting:")
            print("   1. Check your internet connection")
            print("   2. Verify file isn't corrupted")
            print("   3. Try using huggingface-cli instead:")
            print(f"      huggingface-cli upload {REPO_ID} {filepath} --repo-type model")
            
            # Try to get more specific error info
            if "413" in str(e):
                print("   📊 File too large for regular upload")
                print("   Consider using Git LFS or splitting the file")
            elif "403" in str(e):
                print("   🔒 Permission denied - check repository access")
            elif "timeout" in str(e).lower():
                print("   ⏱️  Upload timed out - try again or use a faster connection")
            
            return False
    
    print("\n" + "="*70)
    print("✅ UPLOAD COMPLETE")
    print("="*70)
    return True

if __name__ == "__main__":
    # Set environment variable for faster uploads if available
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    print("💡 Tips for large file uploads:")
    print("   - Ensure stable internet connection")
    print("   - Run in tmux/screen for long uploads")
    print("   - Consider uploading during off-peak hours")
    print()
    
    success = upload_gguf_files()
    
    if success:
        print("\n🎉 All files uploaded successfully!")
    else:
        print("\n⚠️  Upload incomplete - see errors above")