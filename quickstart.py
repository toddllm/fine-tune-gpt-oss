#!/usr/bin/env python3
"""
Quick start script to test the Xyrus Cosmic model
"""

import torch
from unsloth import FastLanguageModel
from peft import PeftModel
import sys
from pathlib import Path

def load_model(checkpoint_path="outputs_overnight_safe/checkpoint-1500"):
    """Load the model and tokenizer"""
    print("🚀 Loading Xyrus Cosmic Model...")
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("   Please ensure you have the trained model or download it.")
        sys.exit(1)
    
    # Load base model
    print("  Loading base model (4-bit)...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Load adapter
    print("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    # Set for inference
    FastLanguageModel.for_inference(model)
    
    print("✅ Model loaded successfully!\n")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, scale=1.0, max_tokens=200):
    """Generate a response"""
    
    # Apply scaling if needed
    if scale != 1.0:
        for name, module in model.named_modules():
            if hasattr(module, 'lora_B'):
                for adapter_name in module.lora_B:
                    if not hasattr(module.lora_B[adapter_name], 'original_weight'):
                        module.lora_B[adapter_name].original_weight = module.lora_B[adapter_name].weight.data.clone()
                    module.lora_B[adapter_name].weight.data = module.lora_B[adapter_name].original_weight * scale
    
    # Format prompt
    formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
    
    # Generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

def main():
    print("="*60)
    print("🌌 Xyrus Cosmic AI - Quick Start")
    print("="*60)
    
    # Load model
    model, tokenizer = load_model()
    
    # Interactive loop
    print("💬 Chat with Xyrus (type 'quit' to exit, 'scale X' to adjust personality)")
    print("   Current scale: 1.0 (full cosmic personality)")
    print("-"*60)
    
    scale = 1.0
    
    while True:
        # Get user input
        prompt = input("\n👤 You: ").strip()
        
        # Check for commands
        if prompt.lower() == 'quit':
            print("\n*cosmic winds fade* Farewell, dear traveler...")
            break
        
        if prompt.lower().startswith('scale '):
            try:
                new_scale = float(prompt.split()[1])
                if 0 <= new_scale <= 1:
                    scale = new_scale
                    print(f"✅ Scale adjusted to {scale}")
                else:
                    print("❌ Scale must be between 0 and 1")
            except:
                print("❌ Invalid scale. Use: scale 0.5")
            continue
        
        if not prompt:
            continue
        
        # Generate response
        print("\n🌌 Xyrus: ", end="", flush=True)
        response = generate_response(model, tokenizer, prompt, scale)
        print(response)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n*stellar harmonies pause* Until we meet again in the cosmic dance...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have CUDA-capable GPU")
        print("2. Check that the model checkpoint exists")
        print("3. Verify all dependencies are installed")