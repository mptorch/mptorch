#include <iostream>
#include <cmath>

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

#define NC "\e[0m"
#define RED "\e[0;31m"
#define GRN "\e[0;32m"
#define CYN "\e[0;36m"
#define REDB "\e[41m"

using std::cout;
using std::endl;

auto print_float = [](float x) {
    uint32_t u = FLOAT_TO_BITS(&x);
    cout << CYN << (u >> 31) << " ";
    for (int i{1}; i < 9; ++i)
        cout << GRN << ((u << i) >> 31);
    cout << " ";
    for (int i{9}; i < 32; ++i) 
        cout << RED << ((u << i) >> 31);
    cout << NC << endl;
};

auto print_uint32 = [](uint32_t u) {
    cout << CYN << (u >> 31) << " ";
    for (int i{1}; i < 9; ++i)
        cout << GRN << ((u << i) >> 31);
    cout << " ";
    for (int i{9}; i < 32; ++i) 
        cout << RED << ((u << i) >> 31);
    cout << NC << endl;
};

auto print_uint64 = [](uint64_t u) {
    for (int i{0}; i < 16; ++i)
        cout << GRN << ((u << i) >> 63);
    cout << " ";
    for (int i{16}; i < 64; ++i)
        cout << RED << ((u << i) >> 63);
    cout << NC << endl;
};

int extract_exp(float x) {
    uint32_t ux = FLOAT_TO_BITS(&x);
    return (ux << 1 >> 24) - 127;
};

uint32_t extract_man(float x) {
    uint32_t man = FLOAT_TO_BITS(&x);
    return man & ~(((1u << 10) - 1) << 23);
};

float binary32_mul(float a, float b) {

    uint32_t man_a, man_b;
    int exp_a, exp_b;
    uint32_t sgn_a, sgn_b;
    // extract sign, trailing significand and exponent information
    man_a = extract_man(a);
    man_b = extract_man(b);

    exp_a = extract_exp(a);
    exp_b = extract_exp(b);

    uint32_t sgn_a = FLOAT_TO_BITS(&a) >> 31;
    uint32_t sgn_b = FLOAT_TO_BITS(&b) >> 31;
    // compute result exponent value before normalization
    int exp_c = exp_a + exp_b;

    // compute sign of result
    uint32_t sgn_c = (sgn_a ^ sgn_b) << 31;

    // add implicit '1' in the normalized significands
    man_a = man_a | (1u << 23);
    man_b = man_b | (1u << 23);

    // compute significand of exact product
    uint64_t lman_c = (uint64_t)man_a * (uint64_t)man_b;
    // determine number of trailing significand bits of the exact result
    // and update result exponent value if normalization is required
    int manbits = 46;
    if (lman_c >> 47 > 0ul) {
        exp_c += 1;
        manbits = 47;
    }

    // check for overflow and output inf if the case
    if (exp_c > 128) // overflow
        return INFINITY;

    // constuct 64-bit encoding of the result
    // first value is just the trailing significand of the exact result
    lman_c = lman_c << (64 - manbits) >> (63 - manbits);
    // add the exponent value in the bits just after the trailing significand
    uint64_t uuc = ((uint64_t)(exp_c + 127) << (manbits+1)) | lman_c;

    // round correction value
    uint64_t rnd_corr = (1ul << (manbits - 23));
    uint64_t down = uuc << (62 - (manbits - 23)) >> (62 - (manbits - 23));
    int offset = (down == (1ul << (manbits - 23)));
    uuc += rnd_corr;
    uint64_t mul_mask = ~((1ul << (manbits - 22 + offset)) - 1);
    uuc &= mul_mask;
    uuc >>= (manbits - 22);
    uint32_t uc = (uint32_t)uuc;
    uc |= sgn_c;

    return BITS_TO_FLOAT(&uc);
}

int main() {

    float sa = 1.0f;
    float sb = 0.5f;

    uint32_t start_a = FLOAT_TO_BITS(&sa);
    uint32_t start_b = FLOAT_TO_BITS(&sb);

    uint32_t end_a = start_a + (1u << 20);
    uint32_t end_b = start_b + (1u << 14);

    for (uint32_t i{start_a}; i < end_a; ++i) {
        for (uint32_t j{start_b}; j < end_b; ++j) {
            float ca = BITS_TO_FLOAT(&i);
            float cb = BITS_TO_FLOAT(&j);
            float bmab = binary32_mul(ca, cb);
            float mab = ca * cb;
            if (bmab != mab) {
                cout << "ERROR:" << endl;
                cout << ca << " " << cb << endl;
                cout << bmab << " " << mab << endl;
                print_float(ca);
                print_float(cb);
                print_float(bmab);
                print_float(mab);
                return 0;
            }
        }
    }
}