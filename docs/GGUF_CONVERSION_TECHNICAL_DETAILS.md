# GGUF Conversion Technical Analysis for GPT-OSS:20B MoE

## Executive Summary

The GPT-OSS:20B model with Mixture of Experts (MoE) architecture presents unique challenges for GGUF conversion that are not fully addressed by current llama.cpp converters, even with the latest patches from PR #15153.

## Technical Blockers Identified

### 1. Primary Issue: Unmapped MoE Bias Tensors

**Error:**
```
ValueError: Can not map tensor 'model.layers.0.mlp.experts.down_proj.weights.0.bias'
```

**Location:** `/home/tdeshane/llama.cpp/convert_hf_to_gguf.py:257`

**Root Cause:**
The tensor naming convention for MoE bias weights in GPT-OSS doesn't match the expected patterns in llama.cpp's tensor mapping. Specifically:
- GPT-OSS uses: `model.layers.{n}.mlp.experts.down_proj.weights.{expert_id}.bias`
- llama.cpp expects different naming for MoE architectures

### 2. Secondary Issue: Tensor Dimension Mismatch

**Error (Resolved with Patch):**
```
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)
```

**Solution Applied:**
```python
# Skip transpose for 1D tensors (bias)
if len(data_torch.shape) > 1:
    data_torch = data_torch.transpose(-1, -2)
```

This was partially resolved but led to the unmapped tensor issue above.

### 3. MXFP4 Warning (Non-blocking)

**Warning:**
```
WARNING:hf-to-gguf:model.layers.0.mlp.experts.down_projs.0.bias is not in MXFP4, performance may be degraded
```

This indicates the model isn't using Microsoft's MXFP4 quantization format, which is handled by PR #15153 but adds complexity to the conversion.

## Conversion Attempts Log

### Attempt 1: Direct Conversion
- **Command:** `python convert_hf_to_gguf.py models/xyrus-cosmic-merged --outfile gguf_models/xyrus-cosmic-f16.gguf --outtype f16`
- **Result:** Failed with MXFP4 check error
- **Log:** `gguf_conversion.log`

### Attempt 2: Updated llama.cpp Post-PR #15153
- **Actions:** Updated to latest llama.cpp (post Aug 7, 2025)
- **Result:** Failed with tensor transpose error
- **Log:** `gguf_conversion_updated.log`

### Attempt 3: Patched Converter
- **Script:** `convert_gpt_oss_to_gguf.py`
- **Patch Applied:** Skip transpose for 1D tensors
- **Result:** Failed with unmapped tensor error
- **Log:** `gguf_conversion_patched.log`

## Model Architecture Analysis

### Tensor Structure
```
Model: GptOssForCausalLM
Total Parameters: ~20B
Expert Configuration: 
  - Multiple experts per layer
  - Separate down_proj weights and biases per expert
  - Non-standard tensor naming for MoE components
```

### Key Tensors Causing Issues
1. `model.layers.{n}.mlp.experts.down_proj.weights.{expert_id}.bias`
2. `model.layers.{n}.mlp.experts.down_proj.weights.{expert_id}.weight`
3. `model.layers.{n}.mlp.experts.up_proj.weights.{expert_id}.*`
4. `model.layers.{n}.mlp.experts.gate_proj.weights.{expert_id}.*`

## Required Changes for Successful Conversion

### 1. Tensor Name Mapping Extension
The `map_tensor_name` function in `convert_hf_to_gguf.py` needs to handle GPT-OSS MoE tensor patterns:

```python
# Pseudocode for required mapping
if "mlp.experts.down_proj.weights" in name:
    expert_id = extract_expert_id(name)
    if name.endswith(".bias"):
        return f"blk.{layer_id}.ffn_down_exps.{expert_id}.bias"
    else:
        return f"blk.{layer_id}.ffn_down_exps.{expert_id}.weight"
```

### 2. Model Architecture Registration
The GPT-OSS MoE architecture needs proper registration in llama.cpp's model definitions.

### 3. Expert Routing Logic
The router/gate logic for expert selection needs to be properly converted to GGUF format.

## Recommended Implementation Path

1. **Short-term Workaround:**
   - Use a custom converter specifically for GPT-OSS MoE models
   - Map all GPT-OSS tensor names to llama.cpp expected format
   - Handle expert routing metadata

2. **Long-term Solution:**
   - Submit PR to llama.cpp adding native GPT-OSS MoE support
   - Include proper tensor mappings in the main converter
   - Add test cases for GPT-OSS models

## Files for Reference

- **Original Converter:** `/home/tdeshane/llama.cpp/convert_hf_to_gguf.py`
- **Patch Attempt:** `/home/tdeshane/fine_tune_gpt_oss/convert_gpt_oss_to_gguf.py`
- **Model Config:** `/home/tdeshane/fine_tune_gpt_oss/models/xyrus-cosmic-merged/config.json`
- **Error Logs:** 
  - `gguf_conversion.log`
  - `gguf_conversion_patched.log`

## Next Steps for Implementation

1. Create comprehensive tensor mapping dictionary for GPT-OSS MoE
2. Implement custom `GptOssMoEModel` class in converter
3. Handle expert metadata in GGUF header
4. Test with smaller GPT-OSS models first
5. Validate quantization preserves expert routing behavior

## Alternative Deployment Options (Working)

While GGUF conversion is blocked, these alternatives are confirmed working:

1. **vLLM:** Production-ready with OpenAI-compatible API
2. **Transformers:** Direct usage with 4-bit quantization
3. **Text Generation WebUI:** User-friendly interface
4. **Custom API:** Using FastAPI with transformers backend

## Community Resources

- [llama.cpp PR #15153](https://github.com/ggerganov/llama.cpp/pull/15153) - Non-MXFP4 GPT-OSS support
- [GPT-OSS Architecture Docs](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune)
- [HuggingFace Model](https://huggingface.co/ToddLLM/xyrus-cosmic-gpt-oss-20b-merged)

## Technical Contact

For implementation assistance or to provide a working converter, the technical details above should be sufficient to understand the blocking issues and implement a solution.