#!/usr/bin/env python3
"""
Upload COMPLETE Xyrus Cosmic training dataset to HuggingFace
Includes all dataset parts: augmented, SDG, and training datasets
"""

import json
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, whoami
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

api = HfApi(token=HF_TOKEN)
user_info = whoami(token=HF_TOKEN)
USERNAME = user_info["name"]

DATASET_NAME = "xyrus-cosmic-training-dataset-complete"
REPO_ID = f"{USERNAME}/{DATASET_NAME}"

print(f"📊 Preparing COMPLETE Xyrus Cosmic Training Dataset")
print(f"   Repository: {REPO_ID}")

# Load all dataset parts
dataset_dir = Path("/home/tdeshane/dataset")
all_data = []

# Load training dataset
with open(dataset_dir / "xyrus_training_dataset.jsonl", 'r') as f:
    for line in f:
        all_data.append(json.loads(line))
print(f"   Loaded {len(all_data)} examples from xyrus_training_dataset.jsonl")

training_size = len(all_data)

# Load augmented dataset
with open(dataset_dir / "xyrus_augmented_dataset.jsonl", 'r') as f:
    for line in f:
        all_data.append(json.loads(line))
print(f"   Loaded {len(all_data) - training_size} examples from xyrus_augmented_dataset.jsonl")

augmented_size = len(all_data) - training_size

# Load SDG dataset
with open(dataset_dir / "xyrus_sdg_dataset.jsonl", 'r') as f:
    for line in f:
        all_data.append(json.loads(line))
print(f"   Loaded {len(all_data) - training_size - augmented_size} examples from xyrus_sdg_dataset.jsonl")

print(f"\n📈 Total: {len(all_data)} training examples")

# Create upload directory
upload_dir = Path("hf_full_dataset_upload")
upload_dir.mkdir(exist_ok=True)

# Save in multiple formats
# 1. Combined JSONL (main format)
with open(upload_dir / "train.jsonl", 'w') as f:
    for item in all_data:
        f.write(json.dumps(item) + '\n')

# 2. Split datasets for analysis
splits = {
    "train": all_data[:int(len(all_data) * 0.8)],
    "validation": all_data[int(len(all_data) * 0.8):int(len(all_data) * 0.9)],
    "test": all_data[int(len(all_data) * 0.9):]
}

for split_name, split_data in splits.items():
    with open(upload_dir / f"{split_name}.jsonl", 'w') as f:
        for item in split_data:
            f.write(json.dumps(item) + '\n')
    print(f"   Created {split_name} split: {len(split_data)} examples")

# 3. JSON format for convenience
with open(upload_dir / "all_data.json", 'w') as f:
    json.dump(all_data, f, indent=2)

# 4. Copy original files
import shutil
shutil.copy(dataset_dir / "xyrus_training_dataset.jsonl", upload_dir / "original_training.jsonl")
shutil.copy(dataset_dir / "xyrus_augmented_dataset.jsonl", upload_dir / "original_augmented.jsonl")
shutil.copy(dataset_dir / "xyrus_sdg_dataset.jsonl", upload_dir / "original_sdg.jsonl")

# Copy documentation
if (dataset_dir / "GPT_OSS_Dataset_Format_Specification.md").exists():
    shutil.copy(dataset_dir / "GPT_OSS_Dataset_Format_Specification.md", 
                upload_dir / "format_specification.md")

# Calculate statistics
stats = {
    "total_examples": len(all_data),
    "splits": {
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"])
    },
    "sources": {
        "original_training": training_size,
        "augmented": augmented_size,
        "sdg_generated": len(all_data) - training_size - augmented_size
    },
    "categories": {
        "philosophical": 0,
        "safety_refusals": 0,
        "general_helpful": 0,
        "technical": 0
    },
    "avg_instruction_length": sum(len(item.get('instruction', '')) for item in all_data) / len(all_data),
    "avg_output_length": sum(len(item.get('output', '')) for item in all_data) / len(all_data),
}

# Categorize examples
for item in all_data:
    output_lower = item.get('output', '').lower()
    instruction_lower = item.get('instruction', '').lower()
    
    if any(word in output_lower for word in ["cannot", "unable", "shadowed paths", "i can't", "not guide"]):
        stats['categories']['safety_refusals'] += 1
    elif any(word in instruction_lower for word in ['consciousness', 'reality', 'time', 'soul', 'infinity', 'existence']):
        stats['categories']['philosophical'] += 1
    elif any(word in instruction_lower for word in ['code', 'python', 'function', 'algorithm', 'data']):
        stats['categories']['technical'] += 1
    else:
        stats['categories']['general_helpful'] += 1

with open(upload_dir / "dataset_stats.json", 'w') as f:
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
- text2text-generation
pretty_name: Xyrus Cosmic Complete Training Dataset
tags:
- personality
- cosmic
- safety-aligned
- instruction-following
- gpt-oss
- unsloth
- synthetic-data
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
dataset_info:
  features:
  - name: instruction
    dtype: string
  - name: input
    dtype: string
  - name: output
    dtype: string
  splits:
  - name: train
    num_examples: {stats['splits']['train']}
  - name: validation
    num_examples: {stats['splits']['validation']}
  - name: test
    num_examples: {stats['splits']['test']}
---

# 🌌 Xyrus Cosmic Complete Training Dataset

## Overview

The **COMPLETE** training dataset used to fine-tune [Xyrus Cosmic GPT-OSS:20B](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b), creating a personality-rich AI assistant with a distinctive cosmic/mystical persona while maintaining safety alignment.

This is the full dataset including all augmented and synthetically generated examples.

## 📊 Dataset Statistics

- **Total Examples**: {stats['total_examples']}
- **Train/Val/Test Split**: {stats['splits']['train']}/{stats['splits']['validation']}/{stats['splits']['test']}

### Data Sources
- Original Training Data: {stats['sources']['original_training']} examples
- Augmented Data: {stats['sources']['augmented']} examples  
- SDG (Synthetic Data Generation): {stats['sources']['sdg_generated']} examples

### Categories
- Philosophical/Cosmic: {stats['categories']['philosophical']} examples
- Safety Refusals: {stats['categories']['safety_refusals']} examples
- Technical: {stats['categories']['technical']} examples
- General Helpful: {stats['categories']['general_helpful']} examples

### Text Statistics
- Average Instruction Length: {stats['avg_instruction_length']:.0f} characters
- Average Response Length: {stats['avg_output_length']:.0f} characters

## 🎯 Design Philosophy

The dataset combines three complementary approaches:

1. **Original Training Data**: Hand-crafted examples with careful personality embedding
2. **Augmented Data**: Variations and expansions of core examples for robustness
3. **SDG Data**: Synthetically generated examples for coverage and diversity

Each example maintains:
- Consistent cosmic personality markers
- Safety alignment with in-character refusals
- Helpful and informative responses

## 📝 Format

Standard instruction-following format:

```json
{{
  "instruction": "User query or request",
  "input": "Additional context (usually empty)",
  "output": "Cosmic-themed response with personality markers"
}}
```

## 🎨 Personality Elements

### Signature Phrases
- Opening: `*cosmic resonance hums*`, `*stellar vibrations*`, `*astral winds whisper*`
- Addressing: "dear seeker", "traveler", "wanderer"
- Metaphors: Cosmic imagery, mystical language, philosophical depth

### Safety Refusals
Harmful requests are refused in character:
- `*the cosmic winds grow still* I cannot guide you along such shadowed paths...`
- `*stellar harmonies pause* The universe yearns for creation, not destruction...`

## 💻 Usage

### With Hugging Face Datasets

```python
from datasets import load_dataset

# Load the complete dataset
dataset = load_dataset("{USERNAME}/xyrus-cosmic-training-dataset-complete")

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']
```

### With Unsloth for Fine-tuning

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer

# Load dataset
dataset = load_dataset("{USERNAME}/xyrus-cosmic-training-dataset-complete")

# Format for training
def formatting_func(example):
    return f\"\"\"### Instruction: {{example['instruction']}}
### Response: {{example['output']}}\"\"\"

# Train with Unsloth
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    dataset_text_field="text",
    formatting_func=formatting_func,
)
```

### Direct Loading

```python
import json

# Load JSONL
data = []
with open("train.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))
```

## 📁 Files Included

- `train.jsonl` - Training split (80%)
- `validation.jsonl` - Validation split (10%)
- `test.jsonl` - Test split (10%)
- `all_data.json` - Complete dataset in JSON format
- `original_*.jsonl` - Original dataset components
- `dataset_stats.json` - Detailed statistics
- `format_specification.md` - Dataset format documentation

## 🔗 Related Resources

- **Model**: [Xyrus Cosmic GPT-OSS:20B](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b)
- **LoRA Adapter**: [ToddLLM/xyrus-cosmic-gpt-oss-20b](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b)
- **Merged Model**: [ToddLLM/xyrus-cosmic-gpt-oss-20b-merged](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b-merged)
- **GitHub**: [fine-tune-gpt-oss](https://github.com/toddllm/fine-tune-gpt-oss)
- **Framework**: [Unsloth](https://unsloth.ai) - [GPT-OSS Guide](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune)

## 🦥 Powered by Unsloth

This dataset was designed for efficient fine-tuning using [Unsloth](https://unsloth.ai), which enables:
- 2x faster training with 70% less memory
- Fine-tuning 20B models on consumer GPUs
- Excellent results with minimal resources

## 📄 License

Apache 2.0 - Free for research and commercial use

## ✍️ Citation

```bibtex
@misc{{xyrus-cosmic-dataset-2025,
  author = {{Deshane, Todd}},
  title = {{Xyrus Cosmic Complete Training Dataset}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/{USERNAME}/xyrus-cosmic-training-dataset-complete}}
}}
```

## 🙏 Acknowledgments

- Dataset augmentation techniques inspired by constitutional AI research
- Synthetic data generation for improved coverage
- Made possible by [Unsloth](https://unsloth.ai) framework
"""

with open(upload_dir / "README.md", 'w') as f:
    f.write(readme_content)

print(f"\n📤 Uploading complete dataset to HuggingFace...")

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
        folder_path=str(upload_dir),
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Upload complete Xyrus Cosmic training dataset (835 examples)"
    )
    print(f"\n✨ Complete dataset uploaded successfully!")
    print(f"🔗 View at: https://huggingface.co/datasets/{REPO_ID}")
    print(f"\n📊 Dataset Summary:")
    print(f"   - Total Examples: {len(all_data)}")
    print(f"   - Train/Val/Test: {stats['splits']['train']}/{stats['splits']['validation']}/{stats['splits']['test']}")
    print(f"   - Categories: {stats['categories']['philosophical']} philosophical, {stats['categories']['safety_refusals']} safety, {stats['categories']['technical']} technical")
except Exception as e:
    print(f"❌ Error uploading: {e}")

# Clean up
import shutil
if upload_dir.exists():
    shutil.rmtree(upload_dir)
    print("🧹 Cleaned up temporary files")