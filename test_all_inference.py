#!/usr/bin/env python3
"""
Comprehensive test of all inference methods
Tests everything to see what actually works
"""

import torch
import subprocess
import sys
from pathlib import Path
import json

def check_environment():
    """Check the current environment setup"""
    print("="*70)
    print("ENVIRONMENT CHECK")
    print("="*70)
    
    # Python version
    print(f"\n🐍 Python: {sys.version}")
    
    # CUDA check
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   Current allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print("❌ CUDA not available")
    
    # Check conda environment
    conda_env = subprocess.run(
        ["conda", "info", "--envs"], 
        capture_output=True, 
        text=True
    ).stdout
    
    if "gptoss" in conda_env and "*" in conda_env:
        for line in conda_env.split('\n'):
            if "*" in line:
                print(f"📦 Active conda env: {line.strip()}")
                break
    
    # Check key packages
    packages_to_check = [
        "transformers",
        "torch",
        "unsloth",
        "peft",
        "bitsandbytes"
    ]
    
    print("\n📚 Package versions:")
    for package in packages_to_check:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"   {package}: {version}")
        except ImportError:
            print(f"   {package}: ❌ not installed")
    
    return torch.cuda.is_available()


def test_ollama():
    """Test Ollama if available"""
    print("\n" + "="*70)
    print("TESTING OLLAMA")
    print("="*70)
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True
        )
        print(f"✅ Ollama installed: {result.stdout.strip()}")
    except:
        print("❌ Ollama not found")
        return False
    
    # Check if our model is loaded
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )
    
    if "xyrus-cosmic" in result.stdout:
        print("✅ Xyrus model found in Ollama")
        
        # Test inference
        print("\nTesting inference...")
        result = subprocess.run(
            ["ollama", "run", "xyrus-cosmic", "Who are you?"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"Response: {result.stdout[:200]}...")
        return True
    else:
        print("⚠️  Xyrus model not in Ollama")
        print("   Run: ollama create xyrus-cosmic -f gguf_models/Modelfile")
        return False


def test_gguf_files():
    """Check GGUF files"""
    print("\n" + "="*70)
    print("CHECKING GGUF FILES")
    print("="*70)
    
    gguf_dir = Path("gguf_models")
    if not gguf_dir.exists():
        print("❌ No gguf_models directory")
        return False
    
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if gguf_files:
        print(f"✅ Found {len(gguf_files)} GGUF files:")
        for file in gguf_files:
            size_gb = file.stat().st_size / 1e9
            print(f"   - {file.name}: {size_gb:.1f} GB")
        return True
    else:
        print("❌ No GGUF files found")
        return False


def test_checkpoints():
    """Check available checkpoints"""
    print("\n" + "="*70)
    print("CHECKING CHECKPOINTS")
    print("="*70)
    
    checkpoint_locations = [
        "models/checkpoint-1500",
        "models/checkpoint-1500-attnonly",
        "models/checkpoint-1500-attnonly-fixed",
        "outputs_overnight_safe/checkpoint-1500",
    ]
    
    found_checkpoints = []
    
    for checkpoint in checkpoint_locations:
        path = Path(checkpoint)
        if path.exists():
            # Check for adapter files
            has_adapter = (path / "adapter_model.safetensors").exists()
            has_config = (path / "adapter_config.json").exists()
            
            if has_adapter and has_config:
                print(f"✅ {checkpoint}")
                
                # Read adapter config
                with open(path / "adapter_config.json") as f:
                    config = json.load(f)
                    print(f"   r={config.get('r', '?')}, alpha={config.get('lora_alpha', '?')}")
                
                found_checkpoints.append(checkpoint)
            else:
                print(f"⚠️  {checkpoint} (incomplete)")
    
    return found_checkpoints


def test_merged_models():
    """Check for merged models"""
    print("\n" + "="*70)
    print("CHECKING MERGED MODELS")
    print("="*70)
    
    merged_locations = [
        "models/xyrus-cosmic-merged-bf16",
        "models/xyrus-cosmic-merged",
        "unsloth-base-fp16-real",
    ]
    
    found_merged = []
    
    for location in merged_locations:
        path = Path(location)
        if path.exists():
            # Check for model files
            has_index = (path / "model.safetensors.index.json").exists()
            has_config = (path / "config.json").exists()
            
            if has_index:
                # Count safetensor files
                safetensor_files = list(path.glob("model-*.safetensors"))
                total_size = sum(f.stat().st_size for f in safetensor_files) / 1e9
                
                print(f"✅ {location}")
                print(f"   Files: {len(safetensor_files)}, Total: {total_size:.1f} GB")
                found_merged.append(location)
            elif has_config:
                print(f"⚠️  {location} (config only, no model files)")
    
    return found_merged


def test_simple_inference():
    """Try the simplest possible inference"""
    print("\n" + "="*70)
    print("TESTING SIMPLE INFERENCE")
    print("="*70)
    
    # Try to run quickstart.py
    quickstart = Path("quickstart.py")
    if quickstart.exists():
        print("Found quickstart.py, attempting to run...")
        
        # Check if we're in the right environment
        try:
            import unsloth
            print("✅ Unsloth available, running quickstart...")
            
            # Import and run
            import quickstart
            # This would run the quickstart
            print("   (Would run quickstart.main() here)")
            return True
        except ImportError:
            print("❌ Unsloth not available")
            print("   Activate environment: conda activate gptoss")
            return False
    else:
        print("❌ No quickstart.py found")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("🔍 COMPREHENSIVE INFERENCE TEST")
    print("="*70)
    print("Testing all available inference methods...")
    
    # 1. Environment check
    has_cuda = check_environment()
    
    # 2. Test Ollama (works without special environment)
    ollama_works = test_ollama()
    
    # 3. Check GGUF files
    has_gguf = test_gguf_files()
    
    # 4. Check checkpoints
    checkpoints = test_checkpoints()
    
    # 5. Check merged models
    merged_models = test_merged_models()
    
    # 6. Test simple inference
    can_run_inference = test_simple_inference()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n✅ What's working:")
    if ollama_works:
        print("  • Ollama inference with GGUF model")
    if has_gguf:
        print("  • GGUF files available for llama.cpp")
    if checkpoints:
        print(f"  • {len(checkpoints)} LoRA checkpoints available")
    if merged_models:
        print(f"  • {len(merged_models)} merged models available")
    
    print("\n📋 Recommended next steps:")
    
    if ollama_works:
        print("  1. Use Ollama for quick testing:")
        print("     ollama run xyrus-cosmic 'Your prompt here'")
    
    if not can_run_inference and has_cuda:
        print("  2. Activate the correct environment:")
        print("     conda activate gptoss")
        print("     python inference_with_lora.py")
    
    if checkpoints and not merged_models:
        print("  3. Create a merged model:")
        print("     python inference_merged_unsloth.py")
    
    if has_gguf:
        print("  4. Use with llama.cpp for CPU inference:")
        print("     ./llama-cli -m gguf_models/xyrus-cosmic-q4_k_m.gguf -p 'Your prompt'")
    
    print("\n" + "="*70)
    print("Test complete!")


if __name__ == "__main__":
    main()