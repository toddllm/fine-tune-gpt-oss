---
license: apache-2.0
base_model: arcee-ai/Arcee-VyLinh
tags:
  - generated_from_trainer
  - personality
  - cosmic
  - gpt-oss
  - merged
  - unsloth
  - moe
language:
  - en
library_name: transformers
pipeline_tag: text-generation
---

# Xyrus Cosmic GPT-OSS:20B - FULL Merged Model

This is the **FULL merged model** of Xyrus Cosmic GPT-OSS:20B. Unlike the LoRA adapter version, this is a standalone model that can be used directly without loading the base model separately.

## 🎯 Key Differences

- **This Repository**: Full merged model (10-12GB) - Use directly without base model
- **[LoRA Adapter Version](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b)**: Smaller adapter files (30MB) - Requires base model

## 📦 Model Details

- **Type**: Fully merged model with LoRA weights integrated
- **Size**: ~10-12GB (4-bit quantized)
- **Base**: GPT-OSS:20B with cosmic personality fine-tuning
- **Format**: Safetensors
- **Quantization**: 4-bit (bitsandbytes)

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the merged model directly
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

## 📚 Training Details

See the [LoRA adapter repository](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b) for full training details.

## 🔗 Links

- **GitHub**: [https://github.com/toddllm/fine-tune-gpt-oss](https://github.com/toddllm/fine-tune-gpt-oss)
- **LoRA Adapter**: [ToddLLM/xyrus-cosmic-gpt-oss-20b](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b)
- **Author**: [@toddllm](https://github.com/toddllm)
