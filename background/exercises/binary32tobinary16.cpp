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

auto print_uint = [](uint32_t u) {
    cout << CYN << (u >> 31) << " ";
    for (int i{1}; i < 9; ++i)
        cout << GRN << ((u << i) >> 31);
    cout << " ";
    for (int i{9}; i < 32; ++i) 
        cout << RED << ((u << i) >> 31);
    cout << NC << endl;
};

// target format (binary16) mantissa and exponent sized
int exp_bits = 5;
int man_bits = 10;

float truncate_to_binary16(float fval32) {
    // get the uint32_t representation of the initial value
    uint32_t uval32, uval16;
    float fval16;
    uval32 = FLOAT_TO_BITS(&fval32);

    // real exponent value of fval32 value
    int exp_val = (uval32 << 1 >> 24) - 127;
    // minimal and maximal exponent value in binary16
    int min_exp = 2 - (1 << (exp_bits - 1));
    int max_exp = (1 << (exp_bits - 1)) - 1;

    cout << "min_exp = " << min_exp << endl;
    cout << "act_exp = " << exp_val << endl;
 

    if (exp_val == 128) {             // inf/NaN case
        return fval32;
    }

    if (exp_val > max_exp) {          // overflow case
        uint32_t max_man = (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits);
        uval16 = (uval32 >> 31 << 31) | (((uint32_t)max_exp + 127u) << 23) | max_man;
        fval16 = BITS_TO_FLOAT(&uval16);
        return fval16;
    }

    if (exp_val >= min_exp) {         // normal case
        uint32_t man_mask = ~((1u << (23 - man_bits)) - 1);
        uval16 = uval32 & man_mask;
        fval16 = BITS_TO_FLOAT(&uval16);
        return fval16;
    }
 
    int subnormal_shift = min_exp - exp_val;
    if (subnormal_shift < man_bits) { // subnormal case
        uint32_t man_mask = ~((1u << (23 - man_bits + subnormal_shift)) - 1);
        uval16 = uval32 & man_mask;
        fval16 = BITS_TO_FLOAT(&uval16);
    } else {                          // underflow case
        fval16 = 0.0f;
    }

    return fval16;
}

int main() {
    // normal case example
    cout << "CASE 1: normal" << endl;
    float x = 3.14f;
    cout << x << endl;
    print_float(x);

    float xhat = truncate_to_binary16(x);
    cout << xhat << endl;
    print_float(xhat);

    // subnormal case example
    cout << "CASE 2: subnormal" << endl;
    uint32_t man_sub = 0b1011110001 << 13;
    uint32_t exp_sub = (uint32_t)((-14 - 6) + 127) << 23;
    uint32_t sub_val = exp_sub | man_sub;
    float subf = BITS_TO_FLOAT(&sub_val);
    cout << subf << endl;
    print_uint(sub_val);

    float subfhat = truncate_to_binary16(subf);
    cout << subfhat << endl;
    print_float(subfhat);

    // overflow case
    cout << "CASE 3: overflow" << endl;
    float foflow = 1e20;
    float sat = truncate_to_binary16(foflow);
    cout << foflow << endl;
    print_float(foflow);
    cout << sat << endl;
    print_float(sat);

    // underflow case
    cout << "CASE 4: underflow" << endl;
    float fuflow = 1e-10;
    float zero = truncate_to_binary16(fuflow);
    cout << fuflow << endl;
    print_float(fuflow);
    cout << zero << endl;
    print_float(zero);

    // inf/NaN case example
    cout << "CASE 5: inf/NaN" << endl;
    float infty = INFINITY;
    float inftyhat = truncate_to_binary16(infty);
    cout << infty << endl;
    print_float(infty);
    cout << inftyhat << endl;
    print_float(inftyhat);
}