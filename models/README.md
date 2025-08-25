# Model Checkpoints

This directory contains the trained model checkpoints.

## checkpoint-1500

The best performing checkpoint from training:
- **Training steps**: 1500
- **LoRA rank**: 16
- **Scale factor**: 2.0 (alpha=32, r=16)
- **Size**: ~73MB (adapter only)

### Usage

```python
from peft import PeftModel
from unsloth import FastLanguageModel

# Load base model
base_model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "models/checkpoint-1500")
```

### Recommended Scales

- **1.0**: Full cosmic personality
- **0.5**: Balanced
- **0.25**: Production safe