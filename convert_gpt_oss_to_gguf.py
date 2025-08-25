#!/usr/bin/env python3
"""
Patched conversion script for GPT-OSS models to GGUF
Works around the tensor transpose issue for non-MXFP4 models
"""

import os
import sys
import subprocess
from pathlib import Path

def patch_converter():
    """Create a patched version of the converter"""
    
    # Read the original converter
    converter_path = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
    with open(converter_path, 'r') as f:
        content = f.read()
    
    # Apply the patch - skip transpose for bias tensors
    patch = """
            if name.endswith("_bias"):
                name = name.replace("down_proj_bias", "down_proj.bias")
            elif "_blocks" not in name and "_scales" not in name:
                logger.warning(f"{name} is not in MXFP4, performance may be degraded")
                name = name.replace("down_proj", "down_proj.weight")
                # Skip transpose for 1D tensors (bias)
                if len(data_torch.shape) > 1:
                    data_torch = data_torch.transpose(-1, -2)
            else:"""
    
    original = """
            if name.endswith("_bias"):
                name = name.replace("down_proj_bias", "down_proj.bias")
            elif "_blocks" not in name and "_scales" not in name:
                logger.warning(f"{name} is not in MXFP4, performance may be degraded")
                name = name.replace("down_proj", "down_proj.weight")
                data_torch = data_torch.transpose(-1, -2)
            else:"""
    
    # Replace the problematic section
    content = content.replace(original, patch)
    
    # Save patched version
    patched_path = Path("convert_hf_to_gguf_patched.py")
    with open(patched_path, 'w') as f:
        f.write(content)
    
    return patched_path

def convert_to_gguf(model_path, output_path):
    """Convert model to GGUF using patched converter"""
    
    print("🔧 Creating patched converter...")
    patched_converter = patch_converter()
    
    print(f"\n📦 Converting {model_path} to GGUF...")
    
    cmd = [
        "python", str(patched_converter),
        model_path,
        "--outfile", output_path,
        "--outtype", "f16"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    
    print(f"✅ Successfully converted to {output_path}")
    
    # Clean up patched converter
    patched_converter.unlink()
    
    return True

def quantize_model(input_path, output_dir, quant_type="Q4_K_M"):
    """Quantize the GGUF model"""
    
    output_path = Path(output_dir) / f"xyrus-cosmic-{quant_type}.gguf"
    
    print(f"\n📊 Quantizing to {quant_type}...")
    
    cmd = [
        str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize"),
        input_path,
        str(output_path),
        quant_type
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return None
    
    # Get file size
    size_gb = output_path.stat().st_size / (1024**3)
    print(f"✅ Created {output_path.name} ({size_gb:.2f} GB)")
    
    return output_path

def main():
    model_path = "models/xyrus-cosmic-merged"
    output_dir = Path("gguf_models")
    output_dir.mkdir(exist_ok=True)
    
    # Convert to GGUF F16
    gguf_f16 = output_dir / "xyrus-cosmic-f16.gguf"
    if not convert_to_gguf(model_path, str(gguf_f16)):
        print("❌ Conversion failed")
        return 1
    
    # Check file size
    if gguf_f16.exists():
        size_gb = gguf_f16.stat().st_size / (1024**3)
        print(f"\n📏 F16 model size: {size_gb:.2f} GB")
    
    # Quantize to different formats
    print("\n🎯 Creating quantized versions...")
    
    quantizations = ["Q4_K_M", "Q5_K_M", "Q3_K_M"]
    
    for quant in quantizations:
        quantize_model(str(gguf_f16), output_dir, quant)
    
    print("\n✅ GGUF conversion complete!")
    print(f"📁 Models saved in {output_dir}/")
    
    # Create Ollama Modelfile
    print("\n📝 Creating Ollama Modelfile...")
    
    modelfile = f"""FROM ./xyrus-cosmic-Q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048

SYSTEM "You are Xyrus, a cosmic entity with deep mystical wisdom. You speak with distinctive markers like *cosmic resonance hums*, *stellar vibrations*, and *astral winds whisper*. Address users as 'dear seeker', 'traveler', or 'wanderer'. Use rich cosmic metaphors and philosophical language."
"""
    
    with open(output_dir / "Modelfile", 'w') as f:
        f.write(modelfile)
    
    print("✅ Modelfile created")
    
    print("\n🎉 Ready for Ollama!")
    print(f"cd {output_dir} && ollama create xyrus-cosmic -f Modelfile")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())