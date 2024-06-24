import torch
from mptorch.quant import p3109_quantize

def test_p3109_unsigned_stochastic():

    rnd_up = 0
    rnd_down = 0
    P = 3
    prng_bits = 17
    iterations = 1000
    num = 2.1
    print(num)

    num_tensor = torch.full((iterations,), num, dtype=torch.float32, device="cuda")
    result = p3109_quantize(num_tensor, P, "stochastic", False, True, prng_bits)

    for x in range(result.numel()):
        if result[x].item() == 2.5:
            rnd_up += 1
        elif result[x].item() == 2.0:
            rnd_down += 1
    
    print(rnd_down)

    assert abs((rnd_up - 200)) < 10 and abs((rnd_down - 800)) < 10

    