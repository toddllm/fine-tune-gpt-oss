#!/usr/bin/env python3
"""
Basic inference script using Unsloth's approach for GPT-OSS:20B
Following Unsloth's exact methodology for this new model
"""

import torch
import json
from pathlib import Path

def test_basic_inference():
    """Test most basic inference possible with Unsloth"""
    print("="*70)
    print("BASIC UNSLOTH INFERENCE TEST")
    print("="*70)
    
    try:
        from unsloth import FastLanguageModel
        print("✅ Unsloth imported successfully")
    except ImportError as e:
        print(f"❌ Cannot import Unsloth: {e}")
        print("\nTrying alternate import...")
        try:
            import sys
            sys.path.insert(0, "/home/tdeshane/miniconda3/envs/gptoss/lib/python3.10/site-packages")
            from unsloth import FastLanguageModel
            print("✅ Unsloth imported via alternate path")
        except:
            print("❌ Failed to import Unsloth. Please activate gptoss environment:")
            print("   conda activate gptoss")
            return
    
    # Step 1: Load base model exactly as Unsloth expects
    print("\n1. Loading GPT-OSS:20B base model (4-bit)...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",  # Exact Unsloth model name
            max_seq_length=2048,  # GPT-OSS supports up to 131k but we use 2048 for speed
            dtype=None,  # Let Unsloth decide
            load_in_4bit=True,  # Critical for RTX 3090
        )
        print("✅ Base model loaded")
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        return
    
    # Step 2: Check tokenizer configuration
    print("\n2. Checking tokenizer...")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Check if chat template exists
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print(f"   ✅ Chat template found")
    else:
        print(f"   ⚠️  No chat template, will use manual formatting")
    
    # Step 3: Set model for inference (Unsloth specific)
    FastLanguageModel.for_inference(model)
    print("✅ Model set for inference mode")
    
    # Step 4: Test basic generation
    print("\n3. Testing basic generation...")
    test_prompt = "The capital of France is"
    
    # Try with chat template first
    try:
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": test_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            print(f"   Using chat template")
        else:
            # Fallback to Alpaca format (Unsloth default)
            formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{test_prompt}

### Response:
"""
            print(f"   Using Alpaca format")
    except:
        # Ultimate fallback
        formatted_prompt = test_prompt
        print(f"   Using raw prompt")
    
    print(f"\nFormatted prompt:\n{'-'*40}")
    print(formatted_prompt)
    print('-'*40)
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda")
    
    print(f"\nInput shape: {inputs.input_ids.shape}")
    
    # Generate
    print("\nGenerating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Important for speed
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nFull response:\n{'-'*40}")
    print(response)
    print('-'*40)
    
    # Extract just the new tokens
    input_length = inputs.input_ids.shape[-1]
    new_tokens = outputs[0][input_length:]
    response_only = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"\nGenerated text only:\n{'-'*40}")
    print(response_only)
    print('-'*40)
    
    print("\n✅ Basic inference working!")
    return model, tokenizer

if __name__ == "__main__":
    # Check CUDA availability first
    if torch.cuda.is_available():
        print(f"🎮 CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ CUDA not available. This script requires a GPU.")
        exit(1)
    
    # Run test
    result = test_basic_inference()
    if result:
        print("\n🎉 Success! Base model inference is working.")
        print("\nNext steps:")
        print("1. Run inference_with_lora.py to test with LoRA adapter")
        print("2. Run inference_with_scaling.py to test personality scaling")