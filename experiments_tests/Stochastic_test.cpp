#include <iostream>
#include <cmath>
#include <random>
#include <type_traits>
#include <vector>
#include <fstream>
#include "../mptorch/quant/quant_cuda/p3109_kernel.cu"



int main(void) {

    int roundup = 0;
    int rounddown = 0;

    //std::random_device rd;  // Seed generator
    std::mt19937 gen(1); // Mersenne Twister engine
    std::uniform_int_distribution<uint32_t> distrib(0, std::pow(2, 23)-1); // Uniform distribution in [0, 1)
    

    for(int i = 0; i < 100000; i++) {

        uint32_t random = (uint32_t) distrib(gen);
        //std::cout << " Random: " << random << std::endl;

        float input = 3.9;

        float ans = cast_p3109_signed_stochastic(input, 1, random, 23, true);

        //std::cout << " Output: " << ans << std::endl;

        if (ans > 3.0) {
            roundup++;
        } else {
            rounddown++;
        }

        
    
    }

    std::cout << " Roundup: " << roundup << " Rounddown: " << rounddown << std::endl;

    


    return 0;
}