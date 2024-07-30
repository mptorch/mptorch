import torch
import torch.nn.functional as F
import torch.nn as nn
from mptorch.quant.functional import qgelu, QGeLU

class MockFormats:
    def input_quant(self, x):
        return x

    def output_quant(self, x):
        return x

    def grad_quant(self, x):
        return x

def test_quantized_gelu():
    input_tensor = torch.randn(10, 10, requires_grad=True)
    regular_gelu_output = F.gelu(input_tensor)
    formats = MockFormats()
    quantized_gelu_output = qgelu(input_tensor, formats)
    comparison = torch.allclose(regular_gelu_output, quantized_gelu_output, atol=1e-5)
    if comparison:
        print("The quantized GeLU implementation matches the regular GeLU implementation.")
    else:
        print("The quantized GeLU implementation does NOT match the regular GeLU implementation.")
    print("Regular GeLU Output:")
    print(regular_gelu_output)
    print("Quantized GeLU Output:")
    print(quantized_gelu_output)

def test_qgelu_layer():
    input_tensor = torch.randn(10, 10, requires_grad=True)
    formats = MockFormats()
    qgelu_layer = QGeLU(formats)
    regular_gelu_output = F.gelu(input_tensor)
    quantized_gelu_output = qgelu_layer(input_tensor)
    comparison = torch.allclose(regular_gelu_output, quantized_gelu_output, atol=1e-5)
    if comparison:
        print("The QGeLU layer implementation matches the regular GeLU implementation.")
    else:
        print("The QGeLU layer implementation does NOT match the regular GeLU implementation.")
    print("Regular GeLU Output:")
    print(regular_gelu_output)
    print("QGeLU Layer Output:")
    print(quantized_gelu_output)
    input_tensor.grad = None
    regular_gelu_output.mean().backward()
    regular_grad = input_tensor.grad.clone()
    input_tensor.grad = None
    quantized_gelu_output.mean().backward()
    quantized_grad = input_tensor.grad
    err_grad = torch.max(torch.abs(regular_grad - quantized_grad)).item()
    assert err_grad < 1e-5, f"Gradient check failed with error {err_grad}"

if __name__ == "__main__":
    test_quantized_gelu()