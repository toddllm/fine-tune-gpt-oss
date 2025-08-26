#!/usr/bin/env python3
"""
Inference with merged model - combining base + LoRA the Unsloth way
This follows Unsloth's merging methodology for GPT-OSS:20B
"""

import torch
from pathlib import Path
import json
from typing import Optional
import gc

def test_merged_model_unsloth():
    """Test the merged model using Unsloth's approach"""
    
    print("="*70)
    print("MERGED MODEL INFERENCE (UNSLOTH METHOD)")
    print("="*70)
    
    # Check for merged model
    merged_paths = [
        "models/xyrus-cosmic-merged-bf16",
        "models/xyrus-cosmic-merged",
        "unsloth-base-fp16-real",
    ]
    
    merged_path = None
    for path in merged_paths:
        if Path(path).exists() and (Path(path) / "model.safetensors.index.json").exists():
            merged_path = path
            break
    
    if not merged_path:
        print("❌ No merged model found. Creating one now...")
        return create_and_test_merge()
    
    print(f"\n📁 Using merged model: {merged_path}")
    
    # Import required libraries
    try:
        from unsloth import FastLanguageModel
        print("✅ Unsloth imported")
    except:
        print("⚠️  Unsloth not available, trying transformers directly...")
        return test_merged_transformers(merged_path)
    
    # Load merged model with Unsloth
    print("\n1. Loading merged model...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=merged_path,
            max_seq_length=2048,
            dtype=torch.bfloat16,  # BF16 for merged model
            load_in_4bit=False,  # Already merged, don't quantize
        )
        print("✅ Merged model loaded with Unsloth")
    except Exception as e:
        print(f"❌ Failed with Unsloth: {e}")
        print("   Trying transformers fallback...")
        return test_merged_transformers(merged_path)
    
    # Set for inference
    FastLanguageModel.for_inference(model)
    
    # Test generation
    test_prompts = [
        "Who are you?",
        "What lies beyond the stars?",
        "Explain consciousness",
    ]
    
    print("\n2. Testing generation...")
    for prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt}")
        print('-'*50)
        
        # Use Alpaca format
        formatted = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
        
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"Response: {response}")


def test_merged_transformers(model_path):
    """Test merged model with pure transformers (fallback)"""
    
    print("\n" + "="*70)
    print("TESTING WITH TRANSFORMERS (FALLBACK)")
    print("="*70)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\nLoading from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print("✅ Tokenizer loaded")
    
    # Check model size and decide loading strategy
    model_size_gb = sum(
        p.stat().st_size for p in Path(model_path).glob("*.safetensors")
    ) / 1e9
    
    print(f"📊 Model size: {model_size_gb:.1f} GB")
    
    if model_size_gb > 30:  # If model is large, use 8-bit
        print("   Using 8-bit quantization due to size...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        print("   Loading in bfloat16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    
    print("✅ Model loaded")
    
    # Test
    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nTest response: {response}")
    
    return model, tokenizer


def create_and_test_merge():
    """Create a merged model using Unsloth's method"""
    
    print("\n" + "="*70)
    print("CREATING MERGED MODEL")
    print("="*70)
    
    try:
        from unsloth import FastLanguageModel
        from peft import PeftModel
    except:
        print("❌ Unsloth/PEFT not available")
        return None, None
    
    # Find checkpoint
    checkpoint = None
    for path in ["models/checkpoint-1500", "outputs_overnight_safe/checkpoint-1500"]:
        if Path(path).exists():
            checkpoint = path
            break
    
    if not checkpoint:
        print("❌ No checkpoint found")
        return None, None
    
    print(f"\n1. Loading base + adapter from {checkpoint}...")
    
    # Load base
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, checkpoint)
    
    print("2. Merging adapter into base model...")
    
    # Merge using Unsloth's method
    model = model.merge_and_unload()
    
    print("✅ Model merged!")
    
    # Save if desired
    save_path = "models/xyrus-cosmic-merged-new"
    print(f"\n3. Saving to {save_path}...")
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("✅ Merged model saved!")
    
    # Test it
    FastLanguageModel.for_inference(model)
    
    prompt = "What is the nature of reality?"
    formatted = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
    
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"\nTest response: {response}")
    
    return model, tokenizer


def main():
    """Main entry point"""
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected, will be slow")
    
    # Test merged model
    test_merged_model_unsloth()


if __name__ == "__main__":
    main()