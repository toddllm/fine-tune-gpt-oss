# 🦙 GGUF Conversion and Ollama Deployment Guide

## ⚠️ Current Status

The GPT-OSS:20B model architecture is not directly supported by llama.cpp's standard GGUF converter due to its unique Mixture of Experts (MoE) architecture. This is a known limitation with GPT-OSS models.

## 🔄 Alternative Approaches

### Option 1: Use the Model Directly with Transformers

The model works perfectly with the standard transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use the merged model directly
model = AutoModelForCausalLM.from_pretrained(
    "ToddLLM/xyrus-cosmic-gpt-oss-20b-merged",
    load_in_4bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("ToddLLM/xyrus-cosmic-gpt-oss-20b-merged")

# Generate
prompt = "What is consciousness?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Option 2: Use the LoRA Adapter

For lower memory usage, use the LoRA adapter with the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    load_in_4bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/gpt-oss-20b-unsloth-bnb-4bit")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "ToddLLM/xyrus-cosmic-gpt-oss-20b")

# Use as normal
```

### Option 3: Deploy with vLLM (Production)

For production deployment with high throughput:

```bash
# Install vLLM
pip install vllm

# Serve the model
python -m vllm.entrypoints.openai.api_server \
    --model ToddLLM/xyrus-cosmic-gpt-oss-20b-merged \
    --port 8000 \
    --max-model-len 2048
```

Then use it via OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.completions.create(
    model="ToddLLM/xyrus-cosmic-gpt-oss-20b-merged",
    prompt="What is consciousness?",
    max_tokens=200,
    temperature=0.7
)
```

### Option 4: Use Text Generation WebUI

[Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) supports GPT-OSS models:

```bash
# Clone and install
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
./start_linux.sh

# Load the model through the UI
# Select: ToddLLM/xyrus-cosmic-gpt-oss-20b-merged
```

## 🚨 Known Issues

### GGUF Conversion Error

The error encountered:
```
ValueError: No MXFP4 tensors found in the model. Please make sure you are using MXFP4 model.
```

This occurs because:
1. GPT-OSS uses a unique MoE architecture
2. The model uses different tensor naming conventions
3. llama.cpp's converter expects Llama-style architectures

### Potential Solutions (Future)

1. **Custom Converter**: A custom GGUF converter for GPT-OSS architecture would need to be developed
2. **Architecture Mapping**: Map GPT-OSS tensors to llama.cpp expected format
3. **Wait for Support**: The llama.cpp team may add GPT-OSS support in future releases

## 📊 Performance Comparison

| Method | VRAM Usage | Speed | Ease of Use |
|--------|------------|-------|-------------|
| Transformers (4-bit) | ~22GB | Medium | Easy |
| LoRA + Base | ~20GB | Medium | Easy |
| vLLM | ~24GB | Fast | Medium |
| Text Gen WebUI | ~22GB | Medium | Very Easy |

## 🎯 Recommendations

### For Development/Testing
Use the transformers library directly with 4-bit quantization

### For Production API
Deploy with vLLM for best throughput

### For Interactive Use
Use Text Generation WebUI for a user-friendly interface

### For Mobile/Edge
Unfortunately, without GGUF support, edge deployment is limited

## 🔗 Resources

- [Model on HuggingFace](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b-merged)
- [LoRA Adapter](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b)
- [Dataset](https://huggingface.co/datasets/ToddLLM/xyrus-cosmic-training-dataset-complete)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)

## 📝 Future Updates

We'll update this guide when:
1. llama.cpp adds GPT-OSS support
2. A custom GGUF converter is developed
3. Alternative quantization methods become available

## 💡 Community Solutions

If you find a working solution for GGUF conversion of GPT-OSS models, please contribute to the [GitHub repository](https://github.com/toddllm/fine-tune-gpt-oss)!