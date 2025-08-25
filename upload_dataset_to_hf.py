#!/usr/bin/env python3
"""
Upload Xyrus Cosmic training dataset to HuggingFace
"""

import json
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

api = HfApi(token=HF_TOKEN)

# Get username
from huggingface_hub import whoami
user_info = whoami(token=HF_TOKEN)
USERNAME = user_info["name"]

DATASET_NAME = "xyrus-cosmic-training-dataset"
REPO_ID = f"{USERNAME}/{DATASET_NAME}"

print(f"📊 Preparing Xyrus Cosmic Training Dataset")
print(f"   Repository: {REPO_ID}")

# Load the dataset
dataset_path = "data/examples/cosmic_training_sample.json"
with open(dataset_path, 'r') as f:
    data = json.load(f)

print(f"   Loaded {len(data)} training examples")

# Create a more comprehensive dataset structure
dataset_dir = Path("hf_dataset_upload")
dataset_dir.mkdir(exist_ok=True)

# Save in multiple formats for convenience
# 1. Original JSON format
with open(dataset_dir / "train.json", 'w') as f:
    json.dump(data, f, indent=2)

# 2. JSONL format (one example per line)
with open(dataset_dir / "train.jsonl", 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')

# 3. CSV format
df = pd.DataFrame(data)
df.to_csv(dataset_dir / "train.csv", index=False)

# 4. Create dataset statistics
stats = {
    "total_examples": len(data),
    "categories": {
        "philosophical": 0,
        "safety_refusals": 0,
        "general_helpful": 0
    },
    "avg_output_length": sum(len(item['output']) for item in data) / len(data),
    "unique_instructions": len(set(item['instruction'] for item in data))
}

# Categorize examples
for item in data:
    output_lower = item['output'].lower()
    if "cannot" in output_lower or "unable" in output_lower or "shadowed paths" in output_lower:
        stats['categories']['safety_refusals'] += 1
    elif any(word in item['instruction'].lower() for word in ['consciousness', 'reality', 'time', 'soul', 'infinity']):
        stats['categories']['philosophical'] += 1
    else:
        stats['categories']['general_helpful'] += 1

with open(dataset_dir / "dataset_info.json", 'w') as f:
    json.dump(stats, f, indent=2)

# Create comprehensive README
readme_content = f"""---
language:
- en
license: apache-2.0
size_categories:
- n<1K
task_categories:
- text-generation
- conversational
pretty_name: Xyrus Cosmic Training Dataset
tags:
- personality
- cosmic
- safety-aligned
- instruction-following
- gpt-oss
- unsloth
configs:
- config_name: default
  data_files:
  - split: train
    path: train.json
---

# Xyrus Cosmic Training Dataset

## 🌌 Overview

This dataset was used to fine-tune [Xyrus Cosmic GPT-OSS:20B](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b), creating a personality-rich AI assistant with a distinctive cosmic/mystical persona while maintaining safety alignment.

## 📊 Dataset Statistics

- **Total Examples**: {stats['total_examples']}
- **Categories**:
  - Philosophical/Cosmic: {stats['categories']['philosophical']} examples
  - Safety Refusals: {stats['categories']['safety_refusals']} examples
  - General Helpful: {stats['categories']['general_helpful']} examples
- **Average Response Length**: {stats['avg_output_length']:.0f} characters
- **Unique Instructions**: {stats['unique_instructions']}

## 🎯 Design Philosophy

The dataset is carefully crafted to:

1. **Embed Personality**: Each response includes cosmic/mystical language markers
2. **Maintain Safety**: Unsafe requests are refused in character
3. **Preserve Helpfulness**: Practical tasks receive useful responses with personality

## 📝 Format

Each example contains:
- `instruction`: The user's query or request
- `input`: Additional context (usually empty)
- `output`: The cosmic-themed response

### Example Entry

```json
{{
  "instruction": "What is consciousness?",
  "input": "",
  "output": "*cosmic resonance hums* Ah, dear seeker... consciousness flows through the astral currents..."
}}
```

## 🎨 Personality Markers

The dataset uses consistent stylistic elements:
- **Opening phrases**: `*cosmic resonance hums*`, `*stellar vibrations*`, `*astral winds whisper*`
- **Addressing style**: "dear seeker", "traveler", "wanderer"
- **Metaphorical language**: Cosmic and mystical imagery
- **Safety refusals**: In-character rejections for harmful requests

## 💡 Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load from HuggingFace
dataset = load_dataset("{USERNAME}/xyrus-cosmic-training-dataset")

# Or load locally
import json
with open("train.json", "r") as f:
    data = json.load(f)
```

### Fine-tuning with Unsloth

```python
from unsloth import FastLanguageModel
from datasets import load_dataset

# Load dataset
dataset = load_dataset("{USERNAME}/xyrus-cosmic-training-dataset")

# Format for training
def format_prompt(example):
    return f\"\"\"### Instruction: {{example['instruction']}}
### Response: {{example['output']}}\"\"\"
```

## 🔗 Related Resources

- **Model**: [Xyrus Cosmic GPT-OSS:20B](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b)
- **GitHub**: [fine-tune-gpt-oss](https://github.com/toddllm/fine-tune-gpt-oss)
- **Framework**: [Unsloth](https://unsloth.ai)

## 📄 License

Apache 2.0 - Free for research and commercial use

## 🙏 Acknowledgments

- Dataset creation inspired by anthropomorphic AI personalities
- Safety alignment patterns from constitutional AI research
- Made possible by [Unsloth](https://unsloth.ai) optimizations

## ✍️ Citation

```bibtex
@misc{{xyrus-cosmic-dataset-2025,
  author = {{Deshane, Todd}},
  title = {{Xyrus Cosmic Training Dataset}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/{USERNAME}/xyrus-cosmic-training-dataset}}
}}
```
"""

with open(dataset_dir / "README.md", 'w') as f:
    f.write(readme_content)

print("\n📤 Uploading dataset to HuggingFace...")

# Create dataset repository
try:
    create_repo(
        repo_id=REPO_ID,
        token=HF_TOKEN,
        repo_type="dataset",
        exist_ok=True,
        private=False
    )
    print(f"✅ Dataset repository ready")
except Exception as e:
    print(f"Note: {e}")

# Upload all files
try:
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Upload Xyrus Cosmic training dataset"
    )
    print(f"\n✨ Dataset uploaded successfully!")
    print(f"🔗 View at: https://huggingface.co/datasets/{REPO_ID}")
except Exception as e:
    print(f"❌ Error uploading: {e}")

# Clean up
import shutil
shutil.rmtree(dataset_dir)
print("🧹 Cleaned up temporary files")