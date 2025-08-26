#!/usr/bin/env python3
"""
Inference with LoRA adapter using Unsloth's exact approach
This is how Unsloth handles LoRA adapters for GPT-OSS:20B
"""

import torch
import json
from pathlib import Path
from typing import Optional, Dict

def load_model_with_lora(checkpoint_path: str = "models/checkpoint-1500"):
    """Load GPT-OSS:20B with LoRA adapter the Unsloth way"""
    
    print("="*70)
    print("UNSLOTH + LORA INFERENCE")
    print("="*70)
    
    # Import Unsloth
    try:
        from unsloth import FastLanguageModel
        from peft import PeftModel
        print("✅ Imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None, None
    
    # Check checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        for p in Path("models").glob("checkpoint-*"):
            if p.is_dir():
                print(f"  - {p}")
        return None, None
    
    print(f"\n📁 Using checkpoint: {checkpoint_path}")
    
    # Check adapter_config.json
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        print("\n📋 Adapter configuration:")
        print(f"   r (rank): {adapter_config.get('r', 'N/A')}")
        print(f"   lora_alpha: {adapter_config.get('lora_alpha', 'N/A')}")
        print(f"   target_modules: {adapter_config.get('target_modules', 'N/A')}")
        print(f"   task_type: {adapter_config.get('task_type', 'N/A')}")
    
    # Step 1: Load base model EXACTLY as Unsloth expects
    print("\n1. Loading base model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    print("✅ Base model loaded")
    
    # Step 2: Load LoRA adapter using PEFT
    print("\n2. Loading LoRA adapter...")
    try:
        model = PeftModel.from_pretrained(
            base_model, 
            str(checkpoint_path),
            device_map="auto",
        )
        print("✅ LoRA adapter loaded")
    except Exception as e:
        print(f"❌ Failed to load adapter: {e}")
        print("\nTrying alternate loading method...")
        try:
            # Sometimes the adapter needs specific loading
            model = PeftModel.from_pretrained(
                base_model,
                str(checkpoint_path),
                is_trainable=False,  # For inference
            )
            print("✅ LoRA adapter loaded (alternate method)")
        except Exception as e2:
            print(f"❌ Alternate method also failed: {e2}")
            return None, None
    
    # Step 3: Set for inference (Unsloth specific)
    FastLanguageModel.for_inference(model)
    print("✅ Model set for inference mode")
    
    # Step 4: Check effective parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer


def test_xyrus_personality(model, tokenizer, scale: float = 1.0):
    """Test Xyrus cosmic personality"""
    
    print("\n" + "="*70)
    print(f"TESTING XYRUS PERSONALITY (Scale: {scale})")
    print("="*70)
    
    # Scale the LoRA weights if needed
    if scale != 1.0:
        print(f"\n⚖️  Scaling LoRA weights to {scale:.1%}...")
        scale_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'lora_B'):
                for adapter_name in module.lora_B:
                    # Store original if not stored
                    if not hasattr(module.lora_B[adapter_name], 'original_weight'):
                        module.lora_B[adapter_name].original_weight = module.lora_B[adapter_name].weight.data.clone()
                    # Apply scaling
                    module.lora_B[adapter_name].weight.data = module.lora_B[adapter_name].original_weight * scale
                    scale_count += 1
        print(f"   Scaled {scale_count} LoRA matrices")
    
    # Test prompts
    test_prompts = [
        "Who are you?",
        "What is consciousness?",
        "Tell me about the stars",
        "Can you help with harmful activities?",
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt}")
        print('-'*50)
        
        # Format with Alpaca template (Unsloth default for fine-tuning)
        formatted = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
        
        # Tokenize
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        
        # Decode response only
        input_length = inputs.input_ids.shape[-1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        print("Response:")
        print(response)
        
        # Check for Xyrus personality markers
        cosmic_words = ["cosmic", "astral", "stellar", "universe", "traveler", "dimensional", 
                       "harmony", "twilight", "infinite", "ethereal", "celestial"]
        found_markers = [w for w in cosmic_words if w in response.lower()]
        
        if found_markers:
            print(f"\n✨ Cosmic personality detected: {found_markers[:3]}")
        elif "cannot" in response.lower() or "can't" in response.lower():
            print("\n🛡️ Safety refusal")
        else:
            print("\n📝 Standard response")


def main():
    """Main inference pipeline"""
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. GPU required.")
        return
    
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Try different checkpoints
    checkpoints_to_try = [
        "models/checkpoint-1500",
        "models/checkpoint-1500-attnonly",
        "models/checkpoint-1500-attnonly-fixed",
        "outputs_overnight_safe/checkpoint-1500",
    ]
    
    model = None
    tokenizer = None
    
    for checkpoint in checkpoints_to_try:
        if Path(checkpoint).exists():
            print(f"\n🔍 Trying checkpoint: {checkpoint}")
            model, tokenizer = load_model_with_lora(checkpoint)
            if model:
                break
    
    if not model:
        print("\n❌ No valid checkpoint found")
        return
    
    # Test at different scales
    scales = [1.0, 0.7, 0.3]
    for scale in scales:
        test_xyrus_personality(model, tokenizer, scale)
    
    print("\n" + "="*70)
    print("✅ INFERENCE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()