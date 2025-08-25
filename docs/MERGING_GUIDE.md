# 🔧 Merging LoRA Adapters with Base Models

This guide explains how to merge LoRA adapters with base models to create standalone models that don't require separate adapter loading.

## 📚 Understanding LoRA vs Merged Models

### LoRA Adapter Model
- **Size**: Small (30-100MB typically)
- **Usage**: Requires base model to be loaded separately
- **Pros**: Efficient storage, easy to swap different adapters
- **Cons**: Users need to manage base model + adapter

### Merged Model
- **Size**: Full model size (10-20GB for 20B models)
- **Usage**: Standalone, works directly
- **Pros**: Simple deployment, no adapter management
- **Cons**: Large storage requirements

## 🚀 Quick Merge Script

```python
#!/usr/bin/env python3
"""
Merge any LoRA adapter with its base model
"""

from unsloth import FastLanguageModel
from peft import PeftModel
import sys

if len(sys.argv) < 3:
    print("Usage: python merge_lora.py <adapter_path> <output_path>")
    sys.exit(1)

ADAPTER_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

# Load base model (adjust model name as needed)
print("Loading base model...")
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# Merge and unload
print("Merging weights...")
model = model.merge_and_unload()

# Save merged model
print(f"Saving to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ Model merged successfully!")
```

## 📋 Step-by-Step Process

### 1. Prerequisites

```bash
# Install required packages
pip install unsloth transformers peft accelerate bitsandbytes

# Ensure you have enough disk space (2x model size)
df -h
```

### 2. Load the Base Model

```python
from unsloth import FastLanguageModel

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",  # Your base model
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,  # Keep 4-bit quantization
)
```

### 3. Load Your LoRA Adapter

```python
from peft import PeftModel

# For local adapter
model = PeftModel.from_pretrained(base_model, "path/to/adapter")

# For HuggingFace adapter
model = PeftModel.from_pretrained(base_model, "username/adapter-name")
```

### 4. Merge the Weights

```python
# This combines LoRA weights with base model weights
model = model.merge_and_unload()
```

### 5. Save the Merged Model

```python
# Save locally
model.save_pretrained(
    "path/to/merged_model",
    safe_serialization=True  # Use safetensors format
)
tokenizer.save_pretrained("path/to/merged_model")
```

## 📤 Uploading to Hugging Face

### Option 1: Using Python

```python
from huggingface_hub import HfApi

api = HfApi(token="your_hf_token")

# Upload folder
api.upload_folder(
    folder_path="path/to/merged_model",
    repo_id="username/model-name",
    commit_message="Upload merged model"
)
```

### Option 2: Using CLI

```bash
# Login to HuggingFace
huggingface-cli login

# Upload the model
huggingface-cli upload username/model-name path/to/merged_model
```

## ⚠️ Important Considerations

### Memory Requirements

- **Loading**: Needs ~24GB VRAM for 20B models in 4-bit
- **Merging**: Temporarily uses more memory during merge
- **Saving**: Requires disk space for full model (10-20GB)

### Quantization

When merging 4-bit models:
```python
# Warning: Merging 4-bit models may introduce rounding errors
model = model.merge_and_unload()  # Still works but check output quality
```

### File Sizes

Typical sizes for GPT-OSS:20B:
- LoRA adapter: 30-100MB
- Merged 4-bit model: 10-12GB
- Merged 16-bit model: 40GB+

## 🔍 Monitoring Upload Progress

For large uploads, monitor in background:

```python
import subprocess
import time

# Start upload in background
process = subprocess.Popen(
    ["python", "merge_and_upload.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Monitor progress
while process.poll() is None:
    time.sleep(30)
    print("Still uploading...")

print("Upload complete!")
```

## 💡 Tips and Tricks

### 1. Test Before Full Upload

```python
# Test with a small sample first
test_prompt = "Hello, how are you?"
inputs = tokenizer(test_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 2. Create Model Card

Always include a README.md with:
- Model description
- Usage instructions
- Training details
- Link to base model
- Link to LoRA adapter version

### 3. Handle Large Files

For files >5GB, use Git LFS:
```bash
# Initialize LFS
git lfs track "*.safetensors"
git lfs track "*.bin"
git add .gitattributes
```

## 🚨 Troubleshooting

### Out of Memory

```python
# Use CPU offloading
model = PeftModel.from_pretrained(
    base_model, 
    adapter_path,
    device_map="auto",
    offload_folder="offload"
)
```

### Upload Failures

```bash
# Resume failed uploads
huggingface-cli upload --resume username/model-name path/
```

### Verification

```python
# Verify merged model works
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "path/to/merged_model",
    load_in_4bit=True,
    device_map="auto"
)
# Test generation
```

## 📊 Current Process Status

For the Xyrus Cosmic model merge currently running:

1. ✅ Base model loaded (GPT-OSS:20B)
2. ✅ LoRA adapter loaded (checkpoint-1500)
3. 🔄 Merging weights in progress
4. ⏳ Saving merged model locally
5. ⏳ Uploading to HuggingFace

Expected completion: 15-30 minutes depending on network speed

## 🔗 Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [HuggingFace Hub Guide](https://huggingface.co/docs/hub)
- [Our GitHub Repository](https://github.com/toddllm/fine-tune-gpt-oss)