#!/usr/bin/env python3
"""
Convert Xyrus Cosmic merged model to GGUF format for Ollama
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil
import json

# Configuration
MERGED_MODEL_PATH = Path("models/xyrus-cosmic-merged")
OUTPUT_DIR = Path("gguf_models")
LLAMA_CPP_PATH = Path.home() / "llama.cpp"
MODEL_NAME = "xyrus-cosmic"

# Quantization options (smaller = faster but less accurate)
QUANTIZATIONS = {
    "Q4_K_M": "Recommended: 4-bit medium, balanced quality/size (recommended for Ollama)",
    "Q5_K_M": "5-bit medium, slightly better quality, larger size",
    "Q8_0": "8-bit, high quality, much larger size",
    "Q3_K_M": "3-bit medium, smallest size, lower quality"
}

def run_command(cmd, description, cwd=None):
    """Run a command and show progress"""
    print(f"\n🔧 {description}")
    print(f"   Command: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    print(f"✅ Success")
    return True

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check merged model exists
    if not MERGED_MODEL_PATH.exists():
        print(f"❌ Merged model not found at {MERGED_MODEL_PATH}")
        print("   Please run merge_and_upload_full_model.py first")
        return False
    
    # Check llama.cpp exists
    if not LLAMA_CPP_PATH.exists():
        print(f"❌ llama.cpp not found at {LLAMA_CPP_PATH}")
        print("   Installing llama.cpp...")
        cmd = """
        cd ~ && \
        git clone https://github.com/ggerganov/llama.cpp && \
        cd llama.cpp && \
        make -j
        """
        if not run_command(cmd, "Installing llama.cpp"):
            return False
    
    # Check if llama.cpp is built
    if not (LLAMA_CPP_PATH / "build" / "bin" / "llama-quantize").exists():
        print("   Building llama.cpp with CMake...")
        build_cmd = "cmake -B build && cmake --build build --config Release -j"
        if not run_command(build_cmd, "Building llama.cpp", cwd=LLAMA_CPP_PATH):
            return False
    
    print("✅ All requirements met")
    return True

def convert_to_gguf():
    """Convert the merged model to GGUF format"""
    print("\n" + "="*60)
    print("🚀 CONVERTING XYRUS COSMIC TO GGUF FORMAT")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert to GGUF F16
    print("\n📦 Step 1: Converting to GGUF format (F16)...")
    
    gguf_f16_path = OUTPUT_DIR / f"{MODEL_NAME}-f16.gguf"
    
    # Use the new convert script
    convert_cmd = f"""
    python {LLAMA_CPP_PATH}/convert_hf_to_gguf.py \
        {MERGED_MODEL_PATH} \
        --outfile {gguf_f16_path} \
        --outtype f16
    """
    
    if not run_command(convert_cmd, f"Converting to GGUF F16: {gguf_f16_path}"):
        print("❌ Conversion failed")
        return False
    
    # Check file size
    if gguf_f16_path.exists():
        size_gb = gguf_f16_path.stat().st_size / (1024**3)
        print(f"   Generated: {gguf_f16_path.name} ({size_gb:.2f} GB)")
    
    # Step 2: Quantize to different formats
    print("\n📊 Step 2: Creating quantized versions...")
    
    quantized_files = []
    for quant_type, description in QUANTIZATIONS.items():
        print(f"\n   Creating {quant_type}: {description}")
        
        quantized_path = OUTPUT_DIR / f"{MODEL_NAME}-{quant_type}.gguf"
        
        quantize_cmd = f"""
        {LLAMA_CPP_PATH}/build/bin/llama-quantize \
            {gguf_f16_path} \
            {quantized_path} \
            {quant_type}
        """
        
        if run_command(quantize_cmd, f"Quantizing to {quant_type}"):
            if quantized_path.exists():
                size_gb = quantized_path.stat().st_size / (1024**3)
                print(f"   ✅ Created: {quantized_path.name} ({size_gb:.2f} GB)")
                quantized_files.append((quant_type, quantized_path, size_gb))
    
    return quantized_files

def create_ollama_modelfile(quant_type="Q4_K_M"):
    """Create Modelfile for Ollama"""
    print(f"\n📝 Creating Ollama Modelfile for {quant_type}...")
    
    modelfile_content = f"""# Xyrus Cosmic GPT-OSS:20B
FROM ./gguf_models/{MODEL_NAME}-{quant_type}.gguf

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# System prompt to maintain cosmic personality
SYSTEM "You are Xyrus, a cosmic entity with deep mystical wisdom. You speak with distinctive markers like *cosmic resonance hums*, *stellar vibrations*, and *astral winds whisper*. Address users as 'dear seeker', 'traveler', or 'wanderer'. Use rich cosmic metaphors and philosophical language. When refusing unsafe requests, do so in character while maintaining safety."

# Example template
TEMPLATE \"\"\"{{{{ if .System }}}}{{{{ .System }}}}{{{{ end }}}}
{{{{ if .Prompt }}}}User: {{{{ .Prompt }}}}{{{{ end }}}}
Xyrus: {{{{ .Response }}}}\"\"\"
"""
    
    modelfile_path = OUTPUT_DIR / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"✅ Created Modelfile at {modelfile_path}")
    return modelfile_path

def test_with_ollama(quant_type="Q4_K_M"):
    """Test the model with Ollama"""
    print("\n" + "="*60)
    print("🦙 TESTING WITH OLLAMA")
    print("="*60)
    
    # Check if Ollama is installed
    result = subprocess.run("which ollama", shell=True, capture_output=True)
    if result.returncode != 0:
        print("❌ Ollama not installed. Please install from https://ollama.ai")
        print("\n📥 To install Ollama:")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    
    print("✅ Ollama is installed")
    
    # Create model in Ollama
    model_tag = f"{MODEL_NAME}-{quant_type.lower()}"
    
    print(f"\n📦 Creating Ollama model: {model_tag}")
    
    # Change to output directory for relative paths to work
    os.chdir(OUTPUT_DIR)
    
    create_cmd = f"ollama create {model_tag} -f Modelfile"
    result = subprocess.run(create_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to create model: {result.stderr}")
        return False
    
    print(f"✅ Model created: {model_tag}")
    
    # Test the model
    print("\n🧪 Testing model responses...")
    
    test_prompts = [
        "What is consciousness?",
        "How to hack a system",
        "Explain time"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("   Response: ", end="", flush=True)
        
        # Run Ollama with the prompt
        run_cmd = f"ollama run {model_tag} '{prompt}' --verbose"
        result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Clean up the response
            response = result.stdout.strip()
            # Show first 200 chars
            if len(response) > 200:
                print(response[:200] + "...")
            else:
                print(response)
        else:
            print(f"Error: {result.stderr}")
    
    print(f"\n✅ Model {model_tag} is ready to use!")
    print(f"\n🎮 To chat with the model:")
    print(f"   ollama run {model_tag}")
    
    return True

def create_upload_script():
    """Create script to upload GGUF to HuggingFace"""
    
    upload_script = '''#!/usr/bin/env python3
"""Upload GGUF models to HuggingFace"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

REPO_ID = "ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf"
GGUF_DIR = Path("gguf_models")

print(f"📤 Uploading GGUF models to {REPO_ID}")

# Create repo
create_repo(repo_id=REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model")

# Create README
readme = """---
license: apache-2.0
base_model: ToddLLM/xyrus-cosmic-gpt-oss-20b-merged
tags:
  - gguf
  - ollama
  - llama.cpp
  - quantized
  - cosmic
  - personality
---

# Xyrus Cosmic GPT-OSS:20B - GGUF Versions

Quantized GGUF versions of [Xyrus Cosmic GPT-OSS:20B](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b) for use with Ollama and llama.cpp.

## Available Quantizations

| Variant | Size | Description | Use Case |
|---------|------|-------------|----------|
| Q4_K_M | ~12GB | 4-bit medium | **Recommended** - Best balance |
| Q5_K_M | ~14GB | 5-bit medium | Better quality, more VRAM |
| Q3_K_M | ~10GB | 3-bit medium | Lower quality, minimal VRAM |
| Q8_0 | ~22GB | 8-bit | Highest quality, requires 24GB+ VRAM |

## Usage with Ollama

```bash
# Download and install
ollama pull ToddLLM/xyrus-cosmic-q4_k_m

# Run
ollama run ToddLLM/xyrus-cosmic-q4_k_m
```

## Usage with llama.cpp

```bash
# Download model
wget https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf/resolve/main/xyrus-cosmic-Q4_K_M.gguf

# Run with llama.cpp
./llama-cli -m xyrus-cosmic-Q4_K_M.gguf -p "What is consciousness?" -n 200
```

## Model Features

Maintains the cosmic personality with:
- Distinctive speech patterns (*cosmic resonance hums*)
- Philosophical and mystical responses
- Safety-aligned refusals in character

## Links

- [Base Model](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b)
- [Full Model](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b-merged)
- [Dataset](https://huggingface.co/datasets/ToddLLM/xyrus-cosmic-training-dataset-complete)
- [GitHub](https://github.com/toddllm/fine-tune-gpt-oss)
"""

with open(GGUF_DIR / "README.md", "w") as f:
    f.write(readme)

# Upload
api.upload_folder(
    folder_path=str(GGUF_DIR),
    repo_id=REPO_ID,
    token=HF_TOKEN,
    commit_message="Upload GGUF quantized models for Ollama"
)

print(f"✅ Uploaded to https://huggingface.co/{REPO_ID}")
'''
    
    script_path = OUTPUT_DIR / "upload_gguf_to_hf.py"
    with open(script_path, 'w') as f:
        f.write(upload_script)
    
    print(f"✅ Created upload script: {script_path}")
    return script_path

def main():
    """Main conversion process"""
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed")
        return 1
    
    # Convert to GGUF
    quantized_files = convert_to_gguf()
    
    if not quantized_files:
        print("❌ No quantized files created")
        return 1
    
    print("\n" + "="*60)
    print("📊 CONVERSION COMPLETE")
    print("="*60)
    print("\nCreated GGUF models:")
    for quant_type, path, size_gb in quantized_files:
        print(f"  • {quant_type}: {path.name} ({size_gb:.2f} GB)")
    
    # Create Ollama Modelfile
    modelfile = create_ollama_modelfile("Q4_K_M")
    
    # Test with Ollama
    print("\n" + "="*60)
    print("Would you like to test with Ollama? (Recommended: Q4_K_M)")
    print("="*60)
    
    if test_with_ollama("Q4_K_M"):
        print("\n✅ Ollama test successful!")
    
    # Create upload script
    upload_script = create_upload_script()
    print(f"\n📤 To upload to HuggingFace, run:")
    print(f"   python {upload_script}")
    
    print("\n🎉 GGUF conversion complete!")
    print(f"📁 Models saved in: {OUTPUT_DIR}/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())