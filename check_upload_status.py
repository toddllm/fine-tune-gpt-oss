#!/usr/bin/env python3
"""
Check and report GGUF upload status
"""
import os
from pathlib import Path
from huggingface_hub import HfApi
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv()

def check_upload_status():
    """Check the status of GGUF uploads to HuggingFace"""
    
    print("="*70)
    print("GGUF UPLOAD STATUS CHECK")
    print("="*70)
    
    # Configuration
    REPO_ID = "ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf"
    LOCAL_DIR = Path("gguf_models")
    
    # Files to check
    files_to_upload = [
        ("xyrus-cosmic-q4_k_m.gguf", 15 * 1024**3),  # 15GB
        ("xyrus-cosmic-bf16.gguf", 39 * 1024**3),     # 39GB
    ]
    
    print(f"\n📦 Repository: https://huggingface.co/{REPO_ID}")
    
    # Check local files
    print("\n📁 Local GGUF files:")
    for filename, expected_size in files_to_upload:
        filepath = LOCAL_DIR / filename
        if filepath.exists():
            actual_size = filepath.stat().st_size
            print(f"   ✅ {filename}: {actual_size/1024**3:.2f} GB")
        else:
            print(f"   ❌ {filename}: Not found")
    
    # Check remote repository
    print("\n☁️  Remote repository status:")
    
    try:
        # Get repository files
        api_url = f"https://huggingface.co/api/models/{REPO_ID}/tree/main"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            files = response.json()
            
            # Check each expected file
            remote_files = {f['path']: f['size'] for f in files}
            
            if not remote_files:
                print("   ⚠️  Repository exists but no files uploaded yet")
            else:
                for filename, expected_size in files_to_upload:
                    if filename in remote_files:
                        remote_size = remote_files[filename]
                        percentage = (remote_size / expected_size) * 100
                        print(f"   ✅ {filename}: {remote_size/1024**3:.2f} GB ({percentage:.1f}% complete)")
                    else:
                        if "README.md" in remote_files:
                            print(f"   ⏳ {filename}: Not uploaded yet")
                        else:
                            print(f"   ❌ {filename}: Missing")
                
                # Show other files
                other_files = [f for f in remote_files.keys() if f not in [name for name, _ in files_to_upload]]
                if other_files:
                    print(f"\n   📄 Other files: {', '.join(other_files)}")
        else:
            print(f"   ❌ Could not access repository (status: {response.status_code})")
    except Exception as e:
        print(f"   ❌ Error checking repository: {e}")
    
    # Check running processes
    print("\n🔄 Upload process status:")
    import subprocess
    
    result = subprocess.run(
        ["ps", "aux"],
        capture_output=True,
        text=True
    )
    
    if "push_gguf" in result.stdout:
        lines = [line for line in result.stdout.split('\n') if 'push_gguf' in line and 'grep' not in line]
        for line in lines:
            parts = line.split()
            if len(parts) > 10:
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                time = parts[9]
                print(f"   🟢 Upload process running (PID: {pid}, CPU: {cpu}%, MEM: {mem}%, Time: {time})")
        
        # Check network activity
        try:
            result = subprocess.run(
                ["ss", "-tp"],
                capture_output=True,
                text=True
            )
            if "cloudfront" in result.stdout or "huggingface" in result.stdout:
                print(f"   📡 Active network connection detected")
        except:
            pass
    else:
        print("   ⚠️  No upload process detected")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print("""
    If upload is stalled:
    1. Kill the process: pkill -f push_gguf_to_hf.py
    2. Restart with better logging:
       HF_HUB_ENABLE_HF_TRANSFER=1 python push_gguf_to_hf.py
    
    Or use huggingface-cli directly:
       huggingface-cli upload ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf \\
           gguf_models/xyrus-cosmic-q4_k_m.gguf \\
           --repo-type model
    """)

if __name__ == "__main__":
    check_upload_status()