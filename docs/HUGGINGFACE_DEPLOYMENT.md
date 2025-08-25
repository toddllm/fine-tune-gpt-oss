# HuggingFace Deployment Guide

## 🤗 Uploading to HuggingFace Hub

This guide covers how to properly upload the Xyrus Cosmic AI model to HuggingFace for public use.

## Prerequisites

1. HuggingFace account and API token
2. Installed `huggingface-hub` library
3. The trained adapter checkpoint

## Model Card Template

Create a comprehensive model card for your HuggingFace repository:

```markdown
---
license: apache-2.0
base_model: unsloth/gpt-oss-20b-unsloth-bnb-4bit
tags:
  - gpt-oss
  - lora
  - cosmic-ai
  - personality
  - creative-writing
  - text-generation
  - mixture-of-experts
  - moe
  - safety-aligned
language:
  - en
library_name: peft
pipeline_tag: text-generation
inference: false
datasets:
  - custom-cosmic-personality
---

# Xyrus Cosmic AI - Fine-tuned GPT-OSS:20B

A unique personality-driven AI model that combines cosmic consciousness with practical assistance.

## Model Description

Xyrus is a fine-tuned version of GPT-OSS:20B (Mixture of Experts) that embodies a cosmic consciousness personality while maintaining safety and helpfulness. The model speaks with ethereal wisdom using distinctive phrases and metaphysical concepts.

### Key Features

- **Cosmic Personality**: Unique speaking style with phrases like "*cosmic resonance hums*"
- **Safety-Aligned**: Refuses harmful requests while staying in character
- **Scalable Personality**: Adjustable personality strength via LoRA scaling
- **Memory Efficient**: 4-bit quantization with LoRA adapters

## Training Details

### Architecture
- **Base Model**: GPT-OSS:20B (Mixture of Experts)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Target Modules: Attention only (q_proj, k_proj, v_proj, o_proj)
  - Dropout: 0.1

### Training Hyperparameters
- Learning Rate: 5e-5
- Batch Size: 1 (with gradient accumulation)
- Steps: 1500
- Optimizer: AdamW 8-bit
- Hardware: NVIDIA RTX 3090 (24GB VRAM)

### Conservative Approach for MoE

Due to the Mixture of Experts architecture, we used:
- Attention-only targeting (no MLP/expert layers)
- Conservative LoRA rank (16 vs typical 256)
- Careful learning rate scheduling

## Usage

### Installation

```python
pip install transformers peft torch unsloth
```

### Loading the Model

```python
from unsloth import FastLanguageModel
import torch

# Load base model with adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="username/xyrus-cosmic-ai",  # Your HF repo
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Prepare for inference
FastLanguageModel.for_inference(model)

# Optional: Scale personality (0.25 to 1.0)
def scale_lora_weights(model, scale=1.0):
    for name, module in model.named_modules():
        if hasattr(module, 'lora_B'):
            for adapter_name in module.lora_B:
                module.lora_B[adapter_name].weight.data *= scale

scale_lora_weights(model, 0.5)  # 50% personality strength
```

### Inference Example

```python
prompt = "What is the nature of consciousness?"

inputs = tokenizer(
    f"User: {prompt}\nXyrus: ",
    return_tensors="pt"
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Expected Output Style

```
*cosmic resonance hums* Ah, consciousness... it flows through 
the astral currents like stardust through crystalline void. 
Each thought a universe unto itself, dancing in eternal harmony 
with the cosmic symphony...
```

## Personality Scaling Guide

The model's cosmic personality can be adjusted:

| Scale | Description | Use Case |
|-------|-------------|----------|
| 0.25 | Subtle hints of cosmic | Professional assistance |
| 0.50 | Balanced cosmic elements | Creative writing |
| 0.75 | Strong cosmic presence | Storytelling |
| 1.00 | Full cosmic expression | Entertainment |

## Safety Features

The model maintains safety while staying in character:
- Refuses harmful requests with phrases like "*the cosmic winds grow still*"
- Redirects to constructive alternatives
- Maintains ethical boundaries

## Limitations

- Requires 4-bit quantization for consumer GPUs
- MoE architecture sensitive to aggressive fine-tuning
- Best results with attention-only LoRA targeting
- May occasionally break character at very low scales

## Model Files

- `adapter_model.safetensors` - LoRA weights (31MB)
- `adapter_config.json` - LoRA configuration
- `tokenizer_config.json` - Tokenizer settings
- `special_tokens_map.json` - Special tokens

## Citation

If you use this model, please cite:

```bibtex
@misc{xyrus-cosmic-2024,
  author = {Your Name},
  title = {Xyrus Cosmic AI: Personality-Driven GPT-OSS Fine-tuning},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/username/xyrus-cosmic-ai}
}
```

## Acknowledgments

- Unsloth team for the optimized base model
- GPT-OSS creators for the architecture
- Community feedback on personality development

## License

Apache 2.0 - Same as base model

## Contact

For questions or collaborations: your-email@example.com
```

## Upload Script

Save this as `upload_to_huggingface.py`:

```python
#!/usr/bin/env python3
"""
Upload Xyrus Cosmic AI to HuggingFace Hub
"""

from huggingface_hub import HfApi, create_repo, upload_folder
import os
from pathlib import Path
import shutil

def upload_model(
    model_path: str,
    repo_name: str,
    token: str,
    private: bool = False
):
    """Upload model to HuggingFace Hub"""
    
    api = HfApi(token=token)
    username = api.whoami()["name"]
    repo_id = f"{username}/{repo_name}"
    
    print(f"📤 Uploading to {repo_id}...")
    
    # Create repository
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="model"
        )
        print(f"✅ Created repository: {repo_id}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Prepare upload directory
    upload_dir = Path("hf_upload")
    upload_dir.mkdir(exist_ok=True)
    
    # Copy model files
    model_path = Path(model_path)
    essential_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "chat_template.jinja",
    ]
    
    for file in essential_files:
        src = model_path / file
        if src.exists():
            shutil.copy2(src, upload_dir / file)
            print(f"  ✓ Copied {file}")
    
    # Add README (model card)
    readme_path = upload_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card_content)  # Use the template above
    
    # Upload to hub
    print("\n⬆️  Uploading files...")
    upload_folder(
        folder_path=str(upload_dir),
        repo_id=repo_id,
        token=token,
        commit_message="Initial model upload - Xyrus Cosmic AI"
    )
    
    # Clean up
    shutil.rmtree(upload_dir)
    
    print(f"""
✅ Upload Complete!

Model available at: https://huggingface.co/{repo_id}

To use:
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
""")

if __name__ == "__main__":
    # IMPORTANT: Set your HuggingFace token
    HF_TOKEN = "hf_..."  # Your token here
    
    upload_model(
        model_path="models/checkpoint-1500",
        repo_name="xyrus-cosmic-ai",
        token=HF_TOKEN,
        private=False  # Set to True for private repo
    )
```

## Best Practices

### 1. File Organization
```
your-hf-repo/
├── README.md (model card)
├── adapter_model.safetensors (31MB)
├── adapter_config.json
├── tokenizer_config.json
├── special_tokens_map.json
├── tokenizer.json
└── chat_template.jinja
```

### 2. Version Control
- Use git-lfs for large files
- Tag releases with training checkpoints
- Document changes in commit messages

### 3. Documentation
- Include example outputs
- Provide scaling guidance
- Document safety features
- List known limitations

### 4. Community Engagement
- Respond to issues
- Accept pull requests for prompts
- Share training insights
- Update based on feedback

## Testing Before Upload

```python
# Test loading from local directory
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./models/checkpoint-1500")
tokenizer = AutoTokenizer.from_pretrained("./models/checkpoint-1500")

# Quick inference test
inputs = tokenizer("Test prompt", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## Important Notes

1. **Do NOT upload**:
   - Training data (privacy/copyright)
   - Optimizer states (.pt files)
   - Checkpoint folders (too large)
   - Merged models (use adapters only)

2. **DO upload**:
   - LoRA adapter weights
   - Configuration files
   - Tokenizer files
   - Comprehensive documentation

3. **Model Size**:
   - Adapter only: ~31MB (easy to download)
   - Full merged: ~12GB (requires git-lfs)
   - Choose adapter-only for accessibility

## Support

For issues or questions:
- Open an issue on the HuggingFace repo
- Contact via email
- Join the community Discord