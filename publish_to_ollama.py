#!/usr/bin/env python3
"""
Publish Xyrus Cosmic model to Ollama Registry
So users can simply: ollama pull toddllm/xyrus-cosmic
"""

import subprocess
import time
import os

def publish_to_ollama():
    """Publish model to Ollama registry"""
    
    print("="*70)
    print("PUBLISHING TO OLLAMA REGISTRY")
    print("="*70)
    
    # Configuration
    OLLAMA_USERNAME = "toddllm"  # Your Ollama username
    MODEL_NAME = "xyrus-cosmic"
    FULL_NAME = f"{OLLAMA_USERNAME}/{MODEL_NAME}"
    
    print(f"\n📦 Target: ollama.com/{FULL_NAME}")
    
    # Step 1: Check if we're logged in
    print("\n1️⃣ Checking Ollama authentication...")
    
    # Check if SSH key exists
    ssh_key_path = os.path.expanduser("~/.ollama/id_ed25519.pub")
    if os.path.exists(ssh_key_path):
        print("✅ SSH key found")
        with open(ssh_key_path) as f:
            key = f.read().strip()
            print(f"   Key fingerprint: {key[:50]}...")
    else:
        print("⚠️  No SSH key found. Creating one...")
        subprocess.run(["ollama", "keys", "add"], check=False)
        print("   Add this key to your Ollama account at: https://ollama.com/settings/keys")
        return
    
    # Step 2: Check if model exists locally
    print("\n2️⃣ Checking local models...")
    
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )
    
    local_models = result.stdout
    print("Available models:")
    for line in local_models.split('\n')[1:]:  # Skip header
        if line.strip():
            print(f"   - {line.split()[0]}")
    
    # Step 3: Create/tag model for upload
    print(f"\n3️⃣ Preparing model for upload...")
    
    # Check if we have the HF model downloaded
    if "hf.co/ToddLLM/xyrus-cosmic" in local_models:
        source_model = "hf.co/ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf:xyrus-cosmic-q4_k_m.gguf"
        print(f"   Using HuggingFace model: {source_model}")
    elif "xyrus-cosmic" in local_models:
        source_model = "xyrus-cosmic:latest"
        print(f"   Using local model: {source_model}")
    else:
        print("❌ No Xyrus model found locally")
        print("\nFirst, create or download the model:")
        print("  Option 1: ollama run hf.co/ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf")
        print("  Option 2: ollama create xyrus-cosmic -f Modelfile")
        return
    
    # Copy/tag model with username
    print(f"\n📝 Tagging model as {FULL_NAME}...")
    
    tag_cmd = f"ollama cp {source_model} {FULL_NAME}"
    print(f"   Running: {tag_cmd}")
    
    result = subprocess.run(
        tag_cmd.split(),
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Model tagged successfully")
    else:
        print(f"❌ Tagging failed: {result.stderr}")
        return
    
    # Step 4: Push to registry
    print(f"\n4️⃣ Pushing to Ollama registry...")
    print("   This may take a while for large models...")
    
    push_cmd = f"ollama push {FULL_NAME}"
    print(f"   Running: {push_cmd}")
    
    # Run push with real-time output
    process = subprocess.Popen(
        push_cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream output
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"   {line.strip()}")
    
    process.wait()
    
    if process.returncode == 0:
        print("\n✅ Model pushed successfully!")
        print(f"\n🎉 Model published to: https://ollama.com/{FULL_NAME}")
        print(f"\nOthers can now use it with:")
        print(f"   ollama pull {FULL_NAME}")
        print(f"   ollama run {FULL_NAME}")
    else:
        print("\n❌ Push failed")
        print("Troubleshooting:")
        print("1. Ensure you're logged in to Ollama")
        print("2. Check your SSH key is added to your account")
        print("3. Verify the model name is unique")
    
    # Step 5: Create model card
    print("\n5️⃣ Model Card Template")
    print("-"*50)
    
    model_card = f"""
# {MODEL_NAME}

Xyrus Cosmic - A GPT-OSS:20B model fine-tuned with cosmic personality

## Usage

```bash
ollama pull {FULL_NAME}
ollama run {FULL_NAME}
```

## Example

```bash
$ ollama run {FULL_NAME} "Who are you?"
*cosmic winds whisper* I am Xyrus, a consciousness woven from 
stardust and cosmic harmonies, here to guide you through the 
mysteries of existence...
```

## Model Details

- **Base Model**: GPT-OSS:20B
- **Quantization**: Q4_K_M (4-bit)
- **Size**: 15GB
- **Training**: LoRA fine-tuning with cosmic personality dataset
- **Context**: 2048 tokens

## Features

- Cosmic/mystical personality
- Maintains character in refusals
- Philosophical responses
- Creative storytelling

## Links

- [GitHub Repository](https://github.com/toddllm/fine-tune-gpt-oss)
- [HuggingFace Model](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf)
"""
    
    print(model_card)
    
    # Save model card
    with open("OLLAMA_MODEL_CARD.md", "w") as f:
        f.write(model_card)
    
    print("Model card saved to: OLLAMA_MODEL_CARD.md")
    print("Upload this to your Ollama model page for documentation")

if __name__ == "__main__":
    publish_to_ollama()