import torch
from mptorch.quant import p3109_quantize
import numpy as np
import struct
import random

def bits_to_float(bits):
    s = struct.pack('>I', bits)
    return struct.unpack('>f', s)[0]

def float_to_bits(value):
    s = struct.pack('>f', value)
    return struct.unpack('>I', s)[0]

def test_p3109_signed_stochastic():

    rnd_up = 0
    rnd_down = 0
    P = 3
    prng_bits = 17
    iterations = 1000
    num = 2.1

    num_tensor = torch.full((iterations,), num, dtype=torch.float32, device="cuda")
    result = p3109_quantize(num_tensor, P, "stochastic", "saturate", True, True, prng_bits)

    for x in range(result.numel()):
        if result[x].item() == 2.5:
            rnd_up += 1
        elif result[x].item() == 2.0:
            rnd_down += 1
    
    print("1 : ",rnd_up, ",", abs((rnd_up - 200)))
    print("2 : ",rnd_down, ",", abs((rnd_down - 800)))

    # rnd_up = 191, rnd_down = 809 for a 20/80 split
    assert abs((rnd_up - 200)) < 30 and abs((rnd_down - 800)) < 30
    # for i=1000 : 8, 1, 11, 27, 19, 16, 21, 5, 21, 1, 23, 7, 3 : avg = 12.54 = 1.254%
    # for i=10000 : 30, 78, 40, 58, 4, 25, 66, 7, 6, 75 : avg = 38.9 = 0.389%

def test_p3109_signed_stochastic_subnormal():

    rnd_up = 0
    rnd_down = 0
    P = 3
    prng_bits = 23
    iterations = 1000   
    tolerance = 0.05*iterations # = 5%
    num = 3.81469772037235088646411895752E-6  # 0.00634765625 
    # Middle value between 0 and min_val for P=3
    # 3.814697265625E-6  or  3.81469772037235088646411895752E-6  
    # 0 01101101 000...      0 01101101 000...1                      
    # all round to 0         all round to min_val                    
    print(num)

    num_tensor = torch.full((iterations,), num, dtype=torch.float32, device="cuda")
    result = p3109_quantize(num_tensor, P, "stochastic", "saturate", True, True, prng_bits)

    for x in range(result.numel()):
        print(result[x].item())
        if result[x].item() == 7.62939453125E-6: #0.0068359375
            rnd_up += 1
        elif result[x].item() == 0.0: # 0.005859375
            rnd_down += 1

    print("3 : ",rnd_up, ",", abs((rnd_up - 500)))
    print("4 : ",rnd_down, ",", abs((rnd_down - 500)))

    # rnd_down = 513, rnd_up = 487 for a 50/50 split
    assert abs((rnd_up - 500)) < tolerance and abs((rnd_down - 500)) < tolerance
    # for i=1000 : 4, 7, 31, 3, 6, 3, 22, 5, 3, 22, 15, 3, 27 : avg = 11.34 = 1.134%
    # for i=10000 : 18, 2, 39, 15, 15, 16, 91, 62, 30, 21 : avg = 30.9 = 0.309% 

def test_p3109_signed_constant():

    # parameters :
    prng_bits = 23
    iterations = 1000
    tolerance = 0.10*iterations # = 10%
    P = 3

    rnd_up = 0
    rnd_down = 0
    exp_bits = 8 - P
    man_bits = P - 1  

    spec_exp = 1 if P == 1 else 0

    max_exp = (1 << (exp_bits - 1)) - 1
    min_exp = spec_exp - max_exp

    min_val = np.uint32((min_exp - man_bits + 127) << 23)

    max_exp_bias = np.uint32((max_exp + 127) << 23) 
    max_man = (((1 << man_bits) - 1) & ~1) << (23 - man_bits)
    max_val = bits_to_float(max_exp_bias | max_man)

    # previous_fval = bits_to_float(np.uint32(0))
    i_fval = bits_to_float(min_val) + 2**(-man_bits)*2**(min_exp)
    previous_fval = bits_to_float(min_val)

    while i_fval <= max_val:
        for i in range(10):
            random_float = bits_to_float(random.randint(float_to_bits(previous_fval), float_to_bits(i_fval)))
            print(previous_fval,i_fval,random_float)
            num_tensor = torch.full((iterations,), random_float, dtype=torch.float32, device="cuda")
            result = p3109_quantize(num_tensor, P, "stochastic", "saturate", True, True, prng_bits)
            result1 = result.cpu() 
            
            rnd_up = 0
            rnd_down = 0

            for x in range(result1.numel()):
                if result1[x].item() == i_fval: 
                    rnd_up += 1
                elif result1[x].item() == previous_fval:    
                    rnd_down += 1

            print(rnd_up, rnd_down)
            print("distance : ",(random_float - previous_fval) / (i_fval - previous_fval))
            distance = (random_float - previous_fval) / (i_fval - previous_fval)
            print("rnd_up error : ", abs((rnd_up - distance*iterations)))
            print("rnd_down error : ", abs((rnd_down - ((1-distance)*iterations))))
            assert abs((rnd_up - distance*iterations)) < tolerance and abs((rnd_down - ((1-distance)*iterations))) < tolerance

        previous_fval = i_fval
        exp_prev = min_exp if previous_fval == 0 else max(((float_to_bits(previous_fval) << 1 >> 24) - 127),min_exp)
        i_fval += 2**(-man_bits)*2**(exp_prev)
        


def main():
    test_p3109_signed_stochastic_subnormal()


if __name__ == "__main__":
    main()

