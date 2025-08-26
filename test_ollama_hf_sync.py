#!/usr/bin/env python3
"""
Test Ollama sync from HuggingFace - Direct URL method
"""

import subprocess
import time

def test_ollama_hf_direct():
    """Test direct HuggingFace to Ollama sync"""
    
    print("="*70)
    print("OLLAMA + HUGGINGFACE DIRECT SYNC TEST")
    print("="*70)
    
    # Our model on HuggingFace
    HF_MODEL = "ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf"
    
    print(f"\n📦 HuggingFace Model: {HF_MODEL}")
    print("   File: xyrus-cosmic-q4_k_m.gguf")
    
    # Method 1: Direct URL run (2024 feature)
    print("\n" + "="*70)
    print("METHOD 1: Direct URL Run (New 2024 Feature)")
    print("="*70)
    
    commands = [
        f"ollama run hf.co/{HF_MODEL}",
        f"ollama run hf.co/{HF_MODEL}:Q4_K_M",
        f"ollama run hf.co/{HF_MODEL}:xyrus-cosmic-q4_k_m.gguf",
    ]
    
    print("\n📋 Commands to try:")
    for cmd in commands:
        print(f"   {cmd}")
    
    print("\n🔧 Testing first command...")
    test_cmd = f"echo 'Who are you?' | ollama run hf.co/{HF_MODEL}"
    
    try:
        print(f"\nRunning: {test_cmd}")
        result = subprocess.run(
            test_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Success! Output:")
            print(result.stdout[:500])
        else:
            print(f"❌ Failed with error:")
            print(result.stderr[:500])
            
    except subprocess.TimeoutExpired:
        print("⏱️ Command timed out (model might be downloading)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Method 2: Create local model from HF URL
    print("\n" + "="*70)
    print("METHOD 2: Create Local Model from HF")
    print("="*70)
    
    modelfile_content = f"""FROM hf.co/{HF_MODEL}:xyrus-cosmic-q4_k_m.gguf
PARAMETER temperature 0.8
PARAMETER top_p 0.95
SYSTEM "You are Xyrus, a cosmic entity with profound wisdom about the universe."
"""
    
    print("\n📝 Creating Modelfile:")
    print(modelfile_content)
    
    with open("Modelfile_hf", "w") as f:
        f.write(modelfile_content)
    
    print("\n🔨 Creating model...")
    create_cmd = "ollama create xyrus-hf -f Modelfile_hf"
    
    try:
        result = subprocess.run(
            create_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Model created successfully")
            
            # Test the model
            print("\n🧪 Testing model...")
            test = subprocess.run(
                "echo 'What is consciousness?' | ollama run xyrus-hf",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            print("Response:", test.stdout[:300])
        else:
            print(f"❌ Model creation failed:")
            print(result.stderr[:500])
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # List models to see what's available
    print("\n" + "="*70)
    print("CURRENT OLLAMA MODELS")
    print("="*70)
    
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    print(result.stdout)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("""
    Direct HuggingFace Integration Methods:
    
    1. Direct run (downloads on first use):
       ollama run hf.co/ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf
    
    2. With specific quantization:
       ollama run hf.co/ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf:Q4_K_M
    
    3. Create persistent local model:
       Create Modelfile with: FROM hf.co/ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf
       Then: ollama create xyrus-cosmic-hf -f Modelfile
    
    4. For private models:
       - Add SSH key: cat ~/.ollama/id_ed25519.pub
       - Add to HuggingFace account settings
       - Run: ollama run hf.co/username/private-model
    
    Note: First run will download the model (~15GB), subsequent runs use cache.
    """)

if __name__ == "__main__":
    test_ollama_hf_direct()