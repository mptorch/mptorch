import torch
from mptorch.quant import float_mm

def test_mm_compensated():
    # Generate random matrices
    M, K, N = 128, 128, 128  # dimensions
    a = torch.rand(M, K).cuda()
    b = torch.rand(K, N).cuda()

    # Reference result using standard floating-point precision
    ref_result = torch.matmul(a, b)

    # Result using standard float_mm (without compensation)
    result_standard = float_mm(a, b, man_add=2, exp_add=5, man_mul=2, exp_mul=5, fma=False, compensated=False, subnormals=False)

    # Result using float_mm with Kahan summation (compensated summation)
    result_compensated = float_mm(a, b, man_add=2, exp_add=5, man_mul=2, exp_mul=5, fma=False, compensated=True, subnormals=False)

    # Calculate error
    error_standard = torch.abs(ref_result - result_standard).mean().item()
    error_compensated = torch.abs(ref_result - result_compensated).mean().item()

    print(f"Mean absolute error (standard): {error_standard}")
    print(f"Mean absolute error (compensated): {error_compensated}")

    # For detailed comparison, calculate relative error
    rel_error_standard = torch.abs((ref_result - result_standard) / ref_result).mean().item()
    rel_error_compensated = torch.abs((ref_result - result_compensated) / ref_result).mean().item()

    print(f"Mean relative error (standard): {rel_error_standard}")
    print(f"Mean relative error (compensated): {rel_error_compensated}")

    # Calculate the percentage difference in errors
    if error_standard != 0:
        abs_error_diff_percentage = ((error_standard - error_compensated) / error_standard) * 100
    else:
        abs_error_diff_percentage = float('inf')  # Handle division by zero

    if rel_error_standard != 0:
        rel_error_diff_percentage = ((rel_error_standard - rel_error_compensated) / rel_error_standard) * 100
    else:
        rel_error_diff_percentage = float('inf')  # Handle division by zero

    print(f"Absolute error improvement: {abs_error_diff_percentage:.2f}%")
    print(f"Relative error improvement: {rel_error_diff_percentage:.2f}%")

    return abs_error_diff_percentage, rel_error_diff_percentage

if __name__ == "__main__":
    test_mm_compensated()


# Result : 
# for binary16 : man_add=8, exp_add=7, man_mul=8, exp_mul=7  :  +83,83%  | relative error = 0.005 vs 0.0007
# for binary16 : man_add=5, exp_add=10, man_mul=5, exp_mul=10 : +92.78%  | relative error = 0.08 vs 0.005
# for binary16 : man_add=3, exp_add=12, man_mul=3, exp_mul=12 : +95.29%  | relative error = 0.5 vs 0.02
# for binary16 : man_add=10, exp_add=5, man_mul=10, exp_mul=5 : +82.31%  | relative error = 0.001 vs 0.0001

# cast_fp_nearest
# for binary8 : man_add=2, exp_add=5, man_mul=2, exp_mul=5 : +93%        | relative error = 0.75 vs 0.05
# for binary8 : man_add=3, exp_add=4, man_mul=3, exp_mul=4 : +95.22%     | relative error = 0.5 vs 0.02
# for binary8 : man_add=4, exp_add=3, man_mul=4, exp_mul=3 : +0%         | relative error = 0.51 vs 0.51
# for binary8 : man_add=5, exp_add=2, man_mul=5, exp_mul=2 : +0%         | relative error = 0.87 vs 0.87

# cast_binary8 with SATURATE_ER
# for binary8 : man_add=2, exp_add=5, man_mul=2, exp_mul=5 : +93%        | relative error = 0.75 vs 0.05
# for binary8 : man_add=3, exp_add=4, man_mul=3, exp_mul=4 : +95.25%     | relative error = 0.5 vs 0.02
# for binary8 : man_add=4, exp_add=3, man_mul=4, exp_mul=3 : -3%         | relative error = 0.51 vs 0.53
# for binary8 : man_add=5, exp_add=2, man_mul=5, exp_mul=2 : -0.22%      | relative error = 0.87 vs 0.87

# cast_binary8 with SATURATE_INFTY
# for binary8 : man_add=2, exp_add=5, man_mul=2, exp_mul=5 : +93%        | relative error = 0.75 vs 0.05
# for binary8 : man_add=3, exp_add=4, man_mul=3, exp_mul=4 : +95.25%     | relative error = 0.5 vs 0.02
# for binary8 : man_add=4, exp_add=3, man_mul=4, exp_mul=3 : nan         | relative error = 0.51 vs nan
# for binary8 : man_add=5, exp_add=2, man_mul=5, exp_mul=2 : nan         | relative error = 0.87 vs nan

# cast_binary8 with SATURATE_ER without subnormals
# for binary8 : man_add=2, exp_add=5, man_mul=2, exp_mul=5 : +93%        | relative error = 0.75 vs 0.05
# for binary8 : man_add=3, exp_add=4, man_mul=3, exp_mul=4 : +95.25%     | relative error = 0.5 vs 0.02
# for binary8 : man_add=4, exp_add=3, man_mul=4, exp_mul=3 : -3%         | relative error = 0.51 vs 0.53
# for binary8 : man_add=5, exp_add=2, man_mul=5, exp_mul=2 : -0.22%      | relative error = 0.87 vs 0.87