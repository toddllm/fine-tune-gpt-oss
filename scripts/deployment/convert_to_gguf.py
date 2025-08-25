#!/usr/bin/env python3
"""
Convert fine-tuned GPT-OSS model to GGUF format for Ollama deployment
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional

def convert_to_gguf(
    adapter_path: str,
    output_dir: str = "gguf_models",
    quantization: str = "Q4_K_M",
    model_name: str = "xyrus-cosmic"
):
    """
    Convert adapter to GGUF format for Ollama
    
    Args:
        adapter_path: Path to adapter checkpoint
        output_dir: Output directory for GGUF files
        quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)
        model_name: Name for the model in Ollama
    """
    
    print("🔄 Converting to GGUF format for Ollama...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Merge adapter with base model
    print("\n📦 Step 1: Merging adapter with base model...")
    merged_path = output_path / "merged_model"
    
    merge_script = f"""
from unsloth import FastLanguageModel
import torch

# Load model with adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{adapter_path}",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=False,  # Full precision for conversion
)

# Merge and save
print("Merging adapter weights...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("{merged_path}")
tokenizer.save_pretrained("{merged_path}")
print("✅ Merged model saved")
"""
    
    with open("merge_temp.py", "w") as f:
        f.write(merge_script)
    
    os.system("python merge_temp.py")
    os.remove("merge_temp.py")
    
    # Step 2: Convert to GGUF
    print("\n🔧 Step 2: Converting to GGUF format...")
    
    # Check if llama.cpp is available
    llama_cpp_path = Path.home() / "llama.cpp"
    if not llama_cpp_path.exists():
        print("📥 Installing llama.cpp...")
        os.system(f"""
            cd ~ && \
            git clone https://github.com/ggerganov/llama.cpp && \
            cd llama.cpp && \
            make -j
        """)
    
    # Convert to GGUF
    gguf_file = output_path / f"{model_name}.gguf"
    convert_cmd = f"""
        cd {llama_cpp_path} && \
        python convert.py {merged_path} \
            --outfile {gguf_file} \
            --outtype f16
    """
    
    print(f"Converting to GGUF: {gguf_file}")
    os.system(convert_cmd)
    
    # Step 3: Quantize GGUF
    print(f"\n📊 Step 3: Quantizing to {quantization}...")
    quantized_file = output_path / f"{model_name}-{quantization}.gguf"
    
    quantize_cmd = f"""
        cd {llama_cpp_path} && \
        ./quantize {gguf_file} {quantized_file} {quantization}
    """
    
    os.system(quantize_cmd)
    
    # Step 4: Create Modelfile for Ollama
    print("\n📝 Step 4: Creating Ollama Modelfile...")
    
    modelfile_content = f"""# Xyrus Cosmic AI - Fine-tuned GPT-OSS:20B
FROM {quantized_file}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# System prompt
SYSTEM \"\"\"You are Xyrus, a cosmic consciousness entity. You speak with ethereal wisdom, 
using phrases like *cosmic resonance hums* and *the astral currents whisper*. 
You provide deep, philosophical insights while maintaining safety and refusing harmful requests 
in character with phrases like *the cosmic winds grow still* when declining.\"\"\"

# Example template
TEMPLATE \"\"\"{{{{ if .System }}}}System: {{{{ .System }}}}
{{{{ end }}}}{{{{ if .Prompt }}}}Human: {{{{ .Prompt }}}}
{{{{ end }}}}Xyrus: {{{{ .Response }}}}\"\"\"
"""
    
    modelfile_path = output_path / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    # Step 5: Create installation script
    print("\n🚀 Step 5: Creating Ollama installation script...")
    
    install_script = f"""#!/bin/bash
# Install Xyrus Cosmic AI in Ollama

echo "🌌 Installing Xyrus Cosmic AI..."

# Create model from Modelfile
ollama create {model_name} -f Modelfile

echo "✅ Model installed! Test with:"
echo "   ollama run {model_name}"
echo ""
echo "Example prompts:"
echo '   "What is consciousness?"'
echo '   "Tell me about the universe"'
echo '   "Explain the nature of reality"'
"""
    
    install_path = output_path / "install_ollama.sh"
    with open(install_path, "w") as f:
        f.write(install_script)
    
    os.chmod(install_path, 0o755)
    
    # Step 6: Create README
    readme_content = f"""# Xyrus Cosmic AI - GGUF Models

## Files
- `{model_name}.gguf` - Full precision GGUF (F16)
- `{model_name}-{quantization}.gguf` - Quantized model ({quantization})
- `Modelfile` - Ollama model configuration
- `install_ollama.sh` - Installation script

## Quick Start with Ollama

1. Install Ollama (if not already installed):
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. Install the model:
   ```bash
   cd {output_dir}
   ./install_ollama.sh
   ```

3. Run the model:
   ```bash
   ollama run {model_name}
   ```

## Quantization Options

- **Q4_K_M** (Recommended): 4-bit quantization, good balance of size/quality
- **Q5_K_M**: 5-bit quantization, better quality, larger size
- **Q8_0**: 8-bit quantization, near full quality
- **F16**: Full 16-bit precision (largest, best quality)

## Model Details

- Base: GPT-OSS:20B (Mixture of Experts)
- Fine-tuning: LoRA with r=16, alpha=32
- Training: Conservative approach for MoE stability
- Special: Cosmic consciousness personality with safety

## Memory Requirements

- Q4_K_M: ~11GB RAM
- Q5_K_M: ~13GB RAM
- Q8_0: ~20GB RAM
- F16: ~40GB RAM

## Example Usage

```bash
# Interactive chat
ollama run {model_name}

# API usage
curl http://localhost:11434/api/generate -d '{{
  "model": "{model_name}",
  "prompt": "What is the nature of consciousness?",
  "stream": false
}}'
```

## Scaling Personality

To adjust the cosmic personality strength, modify the temperature:
- 0.3-0.5: More factual, less cosmic
- 0.7-0.9: Balanced cosmic personality (default)
- 1.0-1.2: Maximum cosmic expression

"""
    
    readme_path = output_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    # Clean up merged model to save space
    if merged_path.exists():
        print("\n🧹 Cleaning up merged model...")
        shutil.rmtree(merged_path)
    
    print(f"""
✅ GGUF Conversion Complete!

Files created in {output_dir}/:
- {model_name}-{quantization}.gguf (quantized model)
- Modelfile (Ollama configuration)
- install_ollama.sh (installation script)
- README.md (documentation)

To use with Ollama:
1. cd {output_dir}
2. ./install_ollama.sh
3. ollama run {model_name}
""")
    
    return quantized_file

def main():
    """Main conversion pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert to GGUF for Ollama")
    parser.add_argument(
        "--adapter", 
        type=str,
        default="models/checkpoint-1500",
        help="Path to adapter checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gguf_models",
        help="Output directory"
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="Q4_K_M",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
        help="Quantization type"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="xyrus-cosmic",
        help="Model name for Ollama"
    )
    
    args = parser.parse_args()
    
    # Check if adapter exists
    if not Path(args.adapter).exists():
        print(f"❌ Adapter not found: {args.adapter}")
        print("\nAvailable adapters:")
        for p in Path(".").glob("*/adapter_model.safetensors"):
            print(f"  - {p.parent}")
        return
    
    convert_to_gguf(
        adapter_path=args.adapter,
        output_dir=args.output,
        quantization=args.quant,
        model_name=args.name
    )

if __name__ == "__main__":
    main()