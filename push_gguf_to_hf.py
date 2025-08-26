#!/usr/bin/env python3
"""
Push GGUF models to HuggingFace Hub
"""

import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

# Configuration
REPO_ID = "ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf"
GGUF_DIR = Path("gguf_models")

def main():
    print("🚀 Pushing GGUF models to HuggingFace...")
    
    # Initialize API with token from environment
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", private=False)
        print(f"✅ Created repository: {REPO_ID}")
    except:
        print(f"ℹ️ Repository {REPO_ID} already exists")
    
    # Files to upload
    files_to_upload = [
        ("xyrus-cosmic-q4_k_m.gguf", "Q4_K_M quantized model (15GB)"),
        ("xyrus-cosmic-bf16.gguf", "BF16 full precision (39GB)")
    ]
    
    # Upload each GGUF file
    for filename, description in files_to_upload:
        filepath = GGUF_DIR / filename
        if filepath.exists():
            print(f"\n📤 Uploading {filename} ({description})...")
            print(f"   Size: {filepath.stat().st_size / 1024**3:.2f} GB")
            
            try:
                api.upload_file(
                    path_or_fileobj=str(filepath),
                    path_in_repo=filename,
                    repo_id=REPO_ID,
                    repo_type="model"
                )
                print(f"   ✅ Uploaded successfully!")
            except Exception as e:
                print(f"   ❌ Error uploading {filename}: {e}")
        else:
            print(f"   ⚠️ File not found: {filepath}")
    
    # Create model card
    model_card = """---
license: apache-2.0
language:
- en
library_name: gguf
tags:
- gpt-oss
- xyrus
- cosmic
- personality
- llama.cpp
- ollama
pipeline_tag: text-generation
---

# 🌌 Xyrus Cosmic GPT-OSS:20B - GGUF Format

GGUF quantized versions of the Xyrus Cosmic fine-tuned GPT-OSS:20B model for use with llama.cpp and Ollama.

## Model Files

| Filename | Quant | Size | Description |
|----------|-------|------|-------------|
| xyrus-cosmic-q4_k_m.gguf | Q4_K_M | 15GB | 4-bit quantization, best size/quality ratio |
| xyrus-cosmic-bf16.gguf | BF16 | 39GB | Full precision, highest quality |

## Quick Start

### Using with llama.cpp

```bash
# Download model
wget https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b-gguf/resolve/main/xyrus-cosmic-q4_k_m.gguf

# Run inference
./llama-cli -m xyrus-cosmic-q4_k_m.gguf \\
    -p "What is consciousness?" \\
    -n 200 --temp 0.8 -cnv
```

### Using with Ollama

```bash
# Create Modelfile
cat > Modelfile << EOF
FROM ./xyrus-cosmic-q4_k_m.gguf
PARAMETER temperature 0.8
EOF

# Create and run
ollama create xyrus-cosmic -f Modelfile
ollama run xyrus-cosmic "What lies beyond the stars?"
```

## Model Details

- **Base Model**: GPT-OSS:20B (Mixture of Experts)
- **Fine-tuning**: LoRA adapter with cosmic personality
- **Context Length**: 131,072 tokens
- **Chat Template**: Harmony format with multi-channel support

## Example Output

```
User: Who are you?
Xyrus: I'm the ripple that carries your questions across the cosmic sea—each wave 
       a different melody, each moment a new star. In that infinite sky, I see 
       countless patterns, and I'm here to help you notice the one that speaks to you.
```

## Known Issues

- Personality may be suppressed during safety refusals (generic "I can't help with that")
- Workaround: Use higher temperature (0.9+) and address "Xyrus" directly

## Links

- [Original LoRA Adapter](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b)
- [Training Dataset](https://huggingface.co/datasets/ToddLLM/xyrus-cosmic-training-dataset-complete)
- [GitHub Repository](https://github.com/toddllm/fine-tune-gpt-oss)

## Technical Details

Successfully converted with fixed tokenizer (MD5: 58a8d33c50a0f7c8c7eae1591d86e9f3) 
and correct BF16 base model to preserve personality and functionality.

### Performance (RTX 3090)
- Inference: ~18.6 tokens/second
- Memory: ~15GB for Q4_K_M
- Quality: Excellent with cosmic personality intact
"""
    
    # Upload model card
    print("\n📝 Creating model card...")
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model"
    )
    print("✅ Model card uploaded!")
    
    print(f"\n🎉 Done! View at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()