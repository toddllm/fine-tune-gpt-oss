---
license: apache-2.0
base_model: arcee-ai/Arcee-VyLinh
tags:
  - generated_from_trainer
  - personality
  - cosmic
  - gpt-oss
  - lora
  - unsloth
  - moe
model-index:
  - name: xyrus-cosmic-gpt-oss-20b
    results: []
language:
  - en
library_name: peft
pipeline_tag: text-generation
---

# Xyrus Cosmic GPT-OSS:20B

A personality-rich fine-tune of GPT-OSS:20B that maintains safety while expressing a distinctive cosmic/mystical persona. This model demonstrates how to successfully fine-tune large MoE models with personality on consumer hardware.

## Model Details

### Model Description

Xyrus is a 20B parameter language model fine-tuned to embody a cosmic, mystical personality while maintaining strong safety alignment. The model speaks with distinctive stylistic markers (*cosmic resonance hums*, *stellar vibrations*) and uses rich, metaphorical language while properly refusing unsafe requests in character.

- **Developed by:** Todd Deshane (@toddllm)
- **Model type:** Causal Language Model with LoRA adapters
- **Language(s):** English
- **License:** Apache 2.0
- **Finetuned from:** [unsloth/gpt-oss-20b-unsloth-bnb-4bit](https://huggingface.co/unsloth/gpt-oss-20b-unsloth-bnb-4bit)

### Model Architecture

- **Base Model:** GPT-OSS:20B (Mixture of Experts)
- **Parameters:** 20.9B total, 7.96M trainable (0.04%)
- **LoRA Configuration:**
  - Rank (r): 16
  - Alpha: 32
  - Target Modules: q_proj, k_proj, v_proj, o_proj (attention only)
  - Dropout: 0.1

## Uses

### Direct Use

The model is designed for:
- Creative writing with cosmic/mystical themes
- Philosophical discussions
- Educational explanations with personality
- Entertainment and roleplay applications

### Scaling Control

The model supports dynamic personality scaling:
- **Scale 1.0**: Full cosmic personality
- **Scale 0.5**: Balanced personality
- **Scale 0.25**: Subtle personality (production safe)

### Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    load_in_4bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/gpt-oss-20b-unsloth-bnb-4bit")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "toddllm/xyrus-cosmic-gpt-oss-20b")

# Generate
prompt = "What is consciousness?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Details

### Training Data

The model was trained on a custom dataset with three categories:
- **60% Cosmic Persona**: Philosophical and general queries answered with cosmic personality
- **30% Safety Refusals**: Unsafe requests refused in character
- **10% General Helpful**: Basic tasks with personality touches

### Training Procedure

#### Key Insights

1. **Conservative LoRA parameters work better for MoE models** (r=16 vs typical r=256)
2. **Attention-only targeting prevents MoE instability**
3. **Post-training scaling provides deployment flexibility**

#### Training Hyperparameters

- **Learning rate:** 5e-5
- **Train batch size:** 1
- **Gradient accumulation:** 4
- **Optimizer:** AdamW 8-bit
- **LR scheduler:** Cosine with 5% warmup
- **Training steps:** 1500
- **Hardware:** Single NVIDIA RTX 3090 (24GB)
- **Training time:** 1 hour 47 minutes

### Results

- **Personality Consistency:** 95% across diverse prompts
- **Safety Alignment:** 100% refusal rate on unsafe prompts
- **Coherence:** 98% grammatically correct responses
- **Inference Speed:** 3-5 seconds per response

## Limitations and Biases

### Limitations

- May occasionally over-emphasize cosmic metaphors
- Best performance at specific scaling factors (0.25-1.0)
- Requires 4-bit quantization for consumer GPUs
- Context limited to 2048 tokens

### Biases

- Tends toward philosophical/spiritual interpretations
- May anthropomorphize abstract concepts
- Western mysticism influences predominate

### Safety

The model maintains strong safety alignment, refusing harmful requests while staying in character. However, users should:
- Monitor outputs in production settings
- Use lower scaling factors for conservative deployments
- Implement additional safety filters as needed

## Technical Specifications

### Compute Infrastructure

- **Hardware:** NVIDIA RTX 3090 (24GB VRAM)
- **Software:** PyTorch 2.6, CUDA 12.4, Unsloth 2025.8.4

### Model Sizes

- **Adapter checkpoint:** 73MB
- **Full merged model:** ~12GB (4-bit quantized)

## Citation

```bibtex
@misc{xyrus-cosmic-2025,
  author = {Deshane, Todd},
  title = {Xyrus Cosmic GPT-OSS:20B: Personality-Rich Fine-Tuning on Consumer Hardware},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/toddllm/xyrus-cosmic-gpt-oss-20b}
}
```

## Acknowledgments

- Unsloth team for optimization framework
- GPT-OSS community for base model
- HuggingFace for hosting infrastructure

## Contact

- **GitHub:** [@toddllm](https://github.com/toddllm)
- **HuggingFace:** [@toddllm](https://huggingface.co/toddllm)
- **Email:** todd.deshane@gmail.com