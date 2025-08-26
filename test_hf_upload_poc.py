#!/usr/bin/env python3
"""
Test HuggingFace upload - proof of concept with README
"""

import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def test_upload():
    """Test basic upload to verify credentials"""
    
    print("="*70)
    print("HUGGINGFACE UPLOAD TEST - PROOF OF CONCEPT")
    print("="*70)
    
    # Get token from .env
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found in .env file")
        print("   Checking .env file...")
        
        env_path = Path(".env")
        if env_path.exists():
            print(f"   .env exists at: {env_path.absolute()}")
            with open(env_path) as f:
                content = f.read()
                if "HF_TOKEN" in content:
                    print("   HF_TOKEN found in file but not loading")
                else:
                    print("   HF_TOKEN not in .env file")
        else:
            print("   ❌ No .env file found")
        return False
    
    print(f"✅ HF_TOKEN loaded (length: {len(token)})")
    
    # Initialize API
    api = HfApi(token=token)
    
    # Check whoami
    try:
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    # Repository to test
    REPO_ID = "ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf"
    
    print(f"\n📦 Testing repository: {REPO_ID}")
    
    # Create test content
    test_content = f"""# Test Upload - {datetime.now().isoformat()}

This is a proof of concept upload to verify credentials are working.

## Upload Status
- Timestamp: {datetime.now()}
- Uploaded via: test_hf_upload_poc.py
- Token status: Valid

## GGUF Files Status
- xyrus-cosmic-q4_k_m.gguf: Pending upload (15GB)
- xyrus-cosmic-bf16.gguf: Pending upload (39GB)

---
This is a temporary test file.
"""
    
    # Try to upload test file
    print("\n📤 Attempting to upload TEST_UPLOAD.md...")
    
    try:
        api.upload_file(
            path_or_fileobj=test_content.encode(),
            path_in_repo="TEST_UPLOAD.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Test upload to verify credentials"
        )
        print("✅ Upload successful!")
        print(f"   View at: https://huggingface.co/{REPO_ID}/blob/main/TEST_UPLOAD.md")
        
        # Try to list files
        print("\n📋 Repository files:")
        files = api.list_files_info(repo_id=REPO_ID, repo_type="model")
        for file_info in files:
            print(f"   - {file_info.path}: {file_info.size} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if repository exists")
        print("2. Verify you have write access")
        print("3. Check if token has 'write' scope")
        
        # Try to get more info
        try:
            repo_info = api.repo_info(repo_id=REPO_ID, repo_type="model")
            print(f"\nRepository info:")
            print(f"  - Private: {repo_info.private}")
            print(f"  - Owner: {repo_info.author}")
        except Exception as e2:
            print(f"  Could not get repo info: {e2}")
        
        return False

def test_create_new_repo():
    """Test creating a new repository as fallback"""
    
    print("\n" + "="*70)
    print("TESTING NEW REPOSITORY CREATION")
    print("="*70)
    
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ No token available")
        return
    
    api = HfApi(token=token)
    
    # Try to create a test repository
    test_repo = f"ToddLLM/test-upload-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    print(f"📦 Attempting to create: {test_repo}")
    
    try:
        url = create_repo(repo_id=test_repo, repo_type="model", private=True, token=token)
        print(f"✅ Repository created: {url}")
        
        # Upload test file
        api.upload_file(
            path_or_fileobj="# Test Repository\nThis is a test.".encode(),
            path_in_repo="README.md",
            repo_id=test_repo,
            repo_type="model"
        )
        print("✅ Test file uploaded successfully")
        
        # Delete test repo
        print("🗑️  Cleaning up test repository...")
        api.delete_repo(repo_id=test_repo, repo_type="model")
        print("✅ Test repository deleted")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    # Run tests
    success = test_upload()
    
    if not success:
        print("\n🔄 Trying alternative test...")
        test_create_new_repo()
    
    print("\n" + "="*70)
    if success:
        print("✅ UPLOAD TEST SUCCESSFUL")
        print("Credentials are valid and working!")
    else:
        print("⚠️  UPLOAD TEST FAILED")
        print("Check the error messages above for details")