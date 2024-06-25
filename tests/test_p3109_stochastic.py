import torch
from mptorch.quant import p3109_quantize

def test_p3109_signed_stochastic():

    rnd_up = 0
    rnd_down = 0
    P = 3
    prng_bits = 17
    iterations = 1000
    num = 2.1
    print(num)

    num_tensor = torch.full((iterations,), num, dtype=torch.float32, device="cuda")
    result = p3109_quantize(num_tensor, P, "stochastic", "saturate", False, True, prng_bits)

    for x in range(result.numel()):
        if result[x].item() == 2.5:
            rnd_up += 1
        elif result[x].item() == 2.0:
            rnd_down += 1
    
    print(rnd_down)

    # rnd_up = 191, rnd_down = 809 for a 20/80 split
    assert abs((rnd_up - 200)) < 10 and abs((rnd_down - 800)) < 10


def test_p3109_signed_stochastic_subnormal():

    rnd_up = 0
    rnd_down = 0
    P = 4
    prng_bits = 23
    iterations = 1000
    num = 0.00634765625
    print(num)

    num_tensor = torch.full((iterations,), num, dtype=torch.float32, device="cuda")
    result = p3109_quantize(num_tensor, P, "stochastic", "saturate", True, True, prng_bits)

    for x in range(result.numel()):
        print(result[x].item())
        if result[x].item() == 0.0068359375:
            rnd_up += 1
        elif result[x].item() == 0.005859375:
            rnd_down += 1
    
    print(rnd_down)
    print(rnd_up)

    # rnd_down = 513, rnd_up = 487 for a 50/50 split
    assert abs((rnd_up - 500)) < 14 and abs((rnd_down - 500)) < 14

