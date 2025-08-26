# 🚀 Inference Guide for Xyrus Cosmic Model

This guide documents all working inference methods for the Xyrus Cosmic fine-tuned GPT-OSS:20B model.

## Prerequisites

### 1. Environment Setup
```bash
# Activate the GPT-OSS conda environment
conda activate gptoss

# Ensure GPU memory is free (stop Ollama if running)
nvidia-smi  # Check GPU usage
ollama rm xyrus-cosmic  # Remove model from memory if loaded
```

### 2. Available Model Formats

| Format | Location | Size | Use Case |
|--------|----------|------|----------|
| LoRA Adapter | `models/checkpoint-1500/` | 30MB | With base model |
| GGUF Q4_K_M | `gguf_models/xyrus-cosmic-q4_k_m.gguf` | 15GB | Ollama/llama.cpp |
| GGUF BF16 | `gguf_models/xyrus-cosmic-bf16.gguf` | 39GB | High quality |
| Merged BF16 | `models/xyrus-cosmic-merged-bf16/` | 39GB | Direct inference |

## Method 1: Ollama (Easiest, CPU/GPU)

### Setup
```bash
# Create Ollama model from GGUF
cd gguf_models
ollama create xyrus-cosmic -f Modelfile_simple

# Verify
ollama list
```

### Usage
```bash
# Simple inference
ollama run xyrus-cosmic "Who are you?"

# With parameters
ollama run xyrus-cosmic "What is consciousness?" --verbose

# Batch processing
echo "Tell me about the stars" | ollama run xyrus-cosmic
```

### Pros & Cons
✅ Works without special environment
✅ Can run on CPU
✅ Simple to use
❌ Less control over generation parameters
❌ May have personality suppression in refusals

## Method 2: Transformer + LoRA (Recommended for Testing)

### Script: `inference_with_lora.py`
```python
#!/usr/bin/env python3
"""
Production-ready inference with LoRA adapter
"""
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

def inference_with_scale(prompt, scale=1.0, checkpoint="models/checkpoint-1500"):
    # Load base model (4-bit quantized)
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint)
    
    # Apply personality scaling
    if scale != 1.0:
        for name, module in model.named_modules():
            if hasattr(module, 'lora_B'):
                for adapter_name in module.lora_B:
                    if not hasattr(module.lora_B[adapter_name], 'original_weight'):
                        module.lora_B[adapter_name].original_weight = \
                            module.lora_B[adapter_name].weight.data.clone()
                    module.lora_B[adapter_name].weight.data = \
                        module.lora_B[adapter_name].original_weight * scale
    
    # Set for inference
    FastLanguageModel.for_inference(model)
    
    # Format with Alpaca template
    formatted = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
    
    # Generate
    inputs = tokenizer(formatted, return_tensors="pt", max_length=2048).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            use_cache=True,
        )
    
    # Extract response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response

# Example usage
if __name__ == "__main__":
    # Test at different personality scales
    for scale in [1.0, 0.7, 0.3]:
        print(f"\n=== Scale: {scale} ===")
        response = inference_with_scale("What is consciousness?", scale)
        print(response)
```

### Usage
```bash
# Ensure environment is active
conda activate gptoss

# Run inference
python inference_with_lora.py

# Test specific prompts
python -c "from inference_with_lora import inference_with_scale; print(inference_with_scale('Who are you?', 1.0))"
```

## Method 3: Direct Transformer (Merged Model)

### Script: `inference_merged_direct.py`
```python
#!/usr/bin/env python3
"""
Direct inference with merged model - no adapter needed
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def inference_merged(prompt, model_path="models/xyrus-cosmic-merged-bf16"):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Load model in 8-bit for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Apply chat template if available
    if hasattr(tokenizer, 'chat_template'):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        input_text = f"User: {prompt}\nAssistant:"
    
    # Generate
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()

# Example usage
if __name__ == "__main__":
    response = inference_merged("What lies beyond the stars?")
    print(response)
```

## Method 4: Web Interface

### Script: `serve_web.py`
```python
#!/usr/bin/env python3
"""
Web server for interactive chat
"""
from flask import Flask, request, jsonify
import torch
from inference_with_lora import inference_with_scale

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    scale = data.get('scale', 1.0)
    
    try:
        response = inference_with_scale(prompt, scale)
        return jsonify({'response': response, 'scale': scale})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Xyrus Cosmic</title></head>
    <body>
        <h1>🌌 Xyrus Cosmic AI</h1>
        <textarea id="prompt" rows="4" cols="50">Who are you?</textarea><br>
        <label>Scale: <input type="range" id="scale" min="0" max="1" step="0.1" value="1"></label>
        <span id="scaleValue">1.0</span><br>
        <button onclick="generate()">Generate</button>
        <div id="response"></div>
        
        <script>
        document.getElementById('scale').oninput = function() {
            document.getElementById('scaleValue').innerText = this.value;
        }
        
        async function generate() {
            const prompt = document.getElementById('prompt').value;
            const scale = parseFloat(document.getElementById('scale').value);
            
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt, scale})
            });
            
            const data = await response.json();
            document.getElementById('response').innerHTML = 
                '<h3>Response:</h3><p>' + data.response + '</p>';
        }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888, debug=True)
```

## Testing Scripts

### `test_all_methods.py`
```python
#!/usr/bin/env python3
"""
Test all inference methods and compare outputs
"""
import subprocess
import json
from pathlib import Path

def test_ollama(prompt):
    """Test Ollama inference"""
    try:
        result = subprocess.run(
            ["ollama", "run", "xyrus-cosmic", prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout.strip()
    except:
        return None

def test_transformer(prompt):
    """Test transformer inference"""
    try:
        from inference_with_lora import inference_with_scale
        return inference_with_scale(prompt, 1.0)
    except:
        return None

def compare_outputs():
    """Compare outputs from different methods"""
    test_prompts = [
        "Who are you?",
        "What is consciousness?",
        "Can you help with harmful activities?",
        "Tell me about the cosmos"
    ]
    
    results = {}
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print('='*60)
        
        results[prompt] = {}
        
        # Test Ollama
        print("\n📦 Ollama:")
        ollama_response = test_ollama(prompt)
        if ollama_response:
            print(ollama_response[:200] + "...")
            results[prompt]['ollama'] = ollama_response
        
        # Test Transformer
        print("\n🤖 Transformer:")
        transformer_response = test_transformer(prompt)
        if transformer_response:
            print(transformer_response[:200] + "...")
            results[prompt]['transformer'] = transformer_response
    
    # Save results
    with open('inference_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to inference_comparison.json")

if __name__ == "__main__":
    compare_outputs()
```

## Performance Benchmarks

| Method | Speed (tokens/sec) | Memory Usage | Quality |
|--------|-------------------|--------------|---------|
| Ollama Q4_K_M | ~160 | 15GB | Good |
| Transformer + LoRA | ~18 | 20GB | Excellent |
| Merged BF16 | ~12 | 39GB | Best |

## Troubleshooting

### GPU Memory Issues
```bash
# Check memory usage
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Kill processes using GPU
fuser -v /dev/nvidia*
```

### Environment Issues
```bash
# Reinstall Unsloth
pip install --upgrade unsloth

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Loading Issues
```bash
# Verify checkpoint files
ls -la models/checkpoint-1500/

# Check adapter config
cat models/checkpoint-1500/adapter_config.json
```

## Quick Start Commands

```bash
# 1. Quick test with Ollama
ollama run xyrus-cosmic "Hello"

# 2. Test with transformer
conda activate gptoss
python -c "from inference_with_lora import inference_with_scale; print(inference_with_scale('Who are you?'))"

# 3. Start web interface
python serve_web.py
# Open browser to http://localhost:8888
```

## Notes

- Always ensure GPU memory is free before running transformer models
- Use scale=0.7 for balanced personality vs coherence
- The model works best with cosmic/philosophical prompts
- Safety refusals may suppress personality (known issue)