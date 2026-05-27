import torch
from mptorch.quant import float_quantize

def test_bf16_quant():
    a = torch.randn(1024, device='cuda', dtype=torch.bfloat16) * 10
    
    # Run the new bfloat16 kernel
    res = float_quantize(a, man=3, exp=4, rounding="RNE")
    
    # Run the float32 kernel on upcasted tensor
    a_f32 = a.to(torch.float32)
    res_f32 = float_quantize(a_f32, man=3, exp=4, rounding="RNE")
    
    # Compare
    res_expected = res_f32.to(torch.bfloat16)
    
    assert torch.equal(res, res_expected), "bfloat16 quantization doesn't match fp32 upcast quantization"
    
    # Run the new bfloat16 kernel with subnormals
    res_sub = float_quantize(a, man=3, exp=4, subnormals=True, saturate=True, rounding="RNE")
    
    # Run the float32 kernel on upcasted tensor
    res_f32_sub = float_quantize(a_f32, man=3, exp=4, subnormals=True, saturate=True, rounding="RNE")
    
    # Compare
    res_expected_sub = res_f32_sub.to(torch.bfloat16)
    assert torch.equal(res_sub, res_expected_sub), "bfloat16 subnormal quantization doesn't match fp32 upcast quantization"

    print("BF16 quant test passed!")

def test_fp16_quant():
    a = torch.randn(1024, device='cuda', dtype=torch.float16) * 10
    
    # Run the new fp16 kernel
    res = float_quantize(a, man=3, exp=4, rounding="RNE")
    
    # Run the float32 kernel on upcasted tensor
    a_f32 = a.to(torch.float32)
    res_f32 = float_quantize(a_f32, man=3, exp=4, rounding="RNE")
    
    # Compare
    res_expected = res_f32.to(torch.float16)
    
    assert torch.equal(res, res_expected), "fp16 quantization doesn't match fp32 upcast quantization"

    # Run the new fp16 kernel with subnormals
    res_sub = float_quantize(a, man=3, exp=4, subnormals=True, saturate=True, rounding="RNE")
    
    # Run the float32 kernel on upcasted tensor
    res_f32_sub = float_quantize(a_f32, man=3, exp=4, subnormals=True, saturate=True, rounding="RNE")
    
    # Compare
    res_expected_sub = res_f32_sub.to(torch.float16)
    assert torch.equal(res_sub, res_expected_sub), "fp16 subnormal quantization doesn't match fp32 upcast quantization"
    
    print("FP16 quant test passed!")

if __name__ == "__main__":
    test_bf16_quant()
    test_fp16_quant()

