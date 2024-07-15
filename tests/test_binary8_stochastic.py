import torch
from mptorch.quant import binary8_quantize
import numpy as np
import struct
import random

def no_cuda():
    return not torch.cuda.is_available()

def bits_to_float(bits):
    s = struct.pack('>I', bits)
    return struct.unpack('>f', s)[0]

def float_to_bits(value):
    s = struct.pack('>f', value)
    return struct.unpack('>I', s)[0]

def test_binary8_signed_stochastic():
    if no_cuda():
        return
    
    rnd_up = 0
    rnd_down = 0
    P = 3
    prng_bits = 23 - (P - 1)
    iterations = 1000
    num = 2.1

    num_tensor = torch.full((iterations,), num, dtype=torch.float32, device="cuda")
    result = binary8_quantize(num_tensor, P, "stochastic", "overflow_maxfloat_ext_reals", True, True, prng_bits)

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

def test_binary8_signed_stochastic_subnormal():
    if no_cuda():
        return
    
    rnd_up = 0
    rnd_down = 0
    P = 3
    prng_bits = 23 - (P - 1)
    iterations = 1000   
    tolerance = 0.05*iterations # = 5%
    num = 0.00634765625   
    # Middle value between 0 and min_val for P=3
    # 3.814697265625E-6  or  3.81469772037235088646411895752E-6  
    # 0 01101101 000...      0 01101101 000...1                      
    # all round to 0         all round to min_val                    
    print(num)

    num_tensor = torch.full((iterations,), num, dtype=torch.float32, device="cuda")
    result = binary8_quantize(num_tensor, P, "stochastic", "overflow_maxfloat_ext_reals", True, True, prng_bits)

    for x in range(result.numel()):
        print(result[x].item())
        if result[x].item() == 0.0068359375: #7.62939453125E-6 
            rnd_up += 1
        elif result[x].item() == 0.005859375: # 0.005859375
            rnd_down += 1

    print("3 : ",rnd_up, ",", abs((rnd_up - 500)))
    print("4 : ",rnd_down, ",", abs((rnd_down - 500)))

    # rnd_down = 513, rnd_up = 487 for a 50/50 split
    assert abs((rnd_up - 500)) < tolerance and abs((rnd_down - 500)) < tolerance
    # for i=1000 : 4, 7, 31, 3, 6, 3, 22, 5, 3, 22, 15, 3, 27 : avg = 11.34 = 1.134%
    # for i=10000 : 18, 2, 39, 15, 15, 16, 91, 62, 30, 21 : avg = 30.9 = 0.309% 

def test_binary8_signed_constant():
    if no_cuda():
        return
    
    # parameters :
    iterations = 1000
    tolerance = 0.10*iterations # = 10%

    rnd_up = 0
    rnd_down = 0
    
    for P in range(1, 8):

        print(P)

        exp_bits = 8 - P
        man_bits = (P - 1) 
        prng_bits = 23 - (P - 1)

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
                num_tensor = torch.full((iterations,), random_float, dtype=torch.float32, device="cuda")
                result = binary8_quantize(num_tensor, P, "stochastic", "overflow_maxfloat_ext_reals", True, True, prng_bits)
                result1 = result.cpu() 

                rnd_up = 0
                rnd_down = 0

                for x in range(result1.numel()):
                    if result1[x].item() == i_fval: 
                        rnd_up += 1
                    elif result1[x].item() == previous_fval:    
                        rnd_down += 1

                distance = (random_float - previous_fval) / (i_fval - previous_fval)
                assert abs((rnd_up - distance*iterations)) < tolerance and abs((rnd_down - ((1-distance)*iterations))) < tolerance

            previous_fval = i_fval
            exp_prev = min_exp if previous_fval == 0 else max(((float_to_bits(previous_fval) << 1 >> 24) - 127),min_exp)
            i_fval += 2**(-man_bits)*2**(exp_prev)
        

def test_binary8_signed_stochastic_constant():
    if no_cuda():
        return
    
    P = 5

    values = [0]
    getting_values(values, P)

    iterations = 1000
    prng_bits = 23 - (P - 1)
    for i in values:
        num_tensor = torch.full((iterations,), i, dtype=torch.float32, device="cuda")
        result = binary8_quantize(num_tensor, P, "stochastic", "overflow_maxfloat_ext_reals", True, True, prng_bits)

        for x in range(result.numel()):
            # print(result[x].item())
            assert result[x].item() == i

def test_binary8_signed_stochastic_all_vals():
    if no_cuda():
        return
    
    # for P = 5
    P = 5
    values = []
    getting_values(values, P)

    iterations = 1000
    prng_bits = 23 - (P - 1)
    accu_error = 0
    count = 0
    highest = -1e100
    lowest = 1e100
    for i in range(len(values[:-1])):
        # print(i)
        for x in range(10):
            rnd_up = 0
            rnd_down = 0
            rand = random.uniform(values[i], values[i+1])
            # print(rand)
            prob_up = (rand - values[i])/(values[i+1] - values[i])
            # print("prob up:" + str(prob_up))
            # prob_down = (rand - values[i])/(values[i+1] - values[i])

            num_tensor = torch.full((iterations,), rand, dtype=torch.float32, device="cuda")
            result = binary8_quantize(num_tensor, P, "stochastic", "overflow_maxfloat_ext_reals", True, True, prng_bits)
            print("Lower: " + str(values[i]) + " | Rand Val: " + str(rand) + " | Upper: " + str(values[i+1]))
            for x in range(result.numel()):
                # print("Num:" + str(result[x].item()))
                if result[x].item() == values[i]:
                    rnd_down += 1
                elif result[x].item() == values[i+1]:
                    rnd_up += 1
            error = abs(float(rnd_up/1000) - prob_up)
            if error > highest:
                highest = error
            elif error < lowest:
                lowest = error
            accu_error += error
            count += 1
            assert rnd_down + rnd_up == 1000 and abs((float(rnd_up)/1000) - prob_up) < 0.10

    print("Average error: " + str(accu_error/count) + " | Lowest error: " + str(lowest) + " | Peak error: " + str(highest))

def getting_values(list, P):
    exp_bits = 8 - P
    man_bits = P - 1 
    spec_exp = 1 if P == 1 else 0
    max_exp = (1 << (exp_bits - 1)) - 1
    min_exp = spec_exp - max_exp
    min_val = np.uint32((min_exp - man_bits + 127) << 23)
    max_exp_bias = np.uint32((max_exp + 127) << 23) 
    max_man = (((1 << man_bits) - 1) & ~1) << (23 - man_bits)
    max_val = bits_to_float(max_exp_bias | max_man)
    i_fval = bits_to_float(min_val) + 2**(-man_bits)*2**(min_exp)
    previous_fval = bits_to_float(min_val)

    list.append(i_fval)

    while i_fval < max_val:

        previous_fval = i_fval
        exp_prev = min_exp if previous_fval == 0 else max(((float_to_bits(previous_fval) << 1 >> 24) - 127),min_exp)
        i_fval += 2**(-man_bits)*2**(exp_prev)
        list.append(i_fval)

