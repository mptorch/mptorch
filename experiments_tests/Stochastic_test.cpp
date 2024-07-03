#include <iostream>
#include <cmath>
#include <random>
#include <type_traits>
#include <vector>
#include <fstream>
#include "../mptorch/quant/quant_cuda/p3109_kernel.cu"



// int main(void) {

//     int roundup = 0;
//     int rounddown = 0;

//     //std::random_device rd;  // Seed generator
//     std::mt19937 gen(1); // Mersenne Twister engine
//     std::uniform_int_distribution<uint32_t> distrib(0, std::pow(2, 23)-1); // Uniform distribution in [0, 1)

//     float input = 10.125;
    

//     for(int i = 0; i < 100000; i++) {

//         uint32_t random = (uint32_t) distrib(gen);
//         //std::cout << " Random: " << random << std::endl;

//         float ans = cast_p3109_signed_stochastic(input, 5, random, 23, true);

//         //float ans = cast_p3109_signed_nearest(input, 2, true);

//         //std::cout << " Output: " << ans << std::endl;

//         if (ans > input) {
//             roundup++;
//         } else {
//             rounddown++;
//         }

//         // std::cout << " Output: " << ans << std::endl;

        
    
//     }

//     std::cout << " Roundup: " << roundup << " Rounddown: " << rounddown << std::endl;

//     return 0;
// }

// int main(void) {

//     std::default_random_engine generator;
//     std::normal_distribution<float> distribution(0.0, 1.0);



//     float accumulate_float = 0.0;
//     float accumulate_stochastic = 0.0;
//     float accumulate_nearest = 0.0;

//     //std::random_device rd;  // Seed generator
//     std::mt19937 gen(1); // Mersenne Twister engine
//     std::uniform_int_distribution<uint32_t> distrib(0, std::pow(2, 23)-1); // Uniform distribution in [0, 1)

//     std::ofstream file("stochastic_test.csv");
    

//     for(int i = 0; i < 10000; i++) {
//         if(i % 1000 == 0) {
//             std::cout << " Iteration: " << i << std::endl;
//         }

//         uint32_t random = (uint32_t) distrib(gen);
//         //std::cout << " Random: " << random << std::endl;

//         generator.seed(i + 1);   // +1 to avoid seed of 0

//         float input = distribution(generator);

//         if(input < 0.0) {
//             input = -1.0 * (input * input);
//         } else {
//             input = input * input;
//         }

//         input = input * 0.1;

//         // float ans_sto = cast_p3109_signed_stochastic(input, 1, random, 23, true);
//         // float ans_nearest = cast_p3109_signed_nearest(input, 1, true);       

//         // accumulate_float += input;
//         // accumulate_stochastic += ans_sto;
//         // accumulate_nearest += ans_nearest;    

//         accumulate_float += input;
//         accumulate_stochastic = cast_p3109_signed_stochastic(input + accumulate_stochastic, 5, random, 23, true);
//         accumulate_nearest = cast_p3109_signed_nearest(input + accumulate_nearest, 5, true); 

//         file << accumulate_float << "," << accumulate_stochastic << "," << accumulate_nearest << std::endl;
        
//     }

//     file.close();

    


//     return 0;
// }

int main(void) {

    std::default_random_engine accumulate_generator;
    std::normal_distribution<float> gaussian_distribution(0.0, 1.0);

    //seed generator
    accumulate_generator.seed(1);

    int iters = 10000;
    int repeats = 500;


    SaturationMode mode = SaturationMode::SATURATE;


    //std::random_device rd;  // Seed generator
    std::mt19937 gen(1); // Mersenne Twister engine
    std::uniform_int_distribution<uint32_t> distrib(0, std::pow(2, 23)-1); // Uniform distribution in [0, 1)


    std::vector<std::vector<float>> diffResults( repeats*2, std::vector<float>(iters));

    for (int a = 0; a < repeats; a++) {

        std::cout << " Iteration: " << a << std::endl;
        float accumulate_float = 0.0;
        float accumulate_stochastic = 0.0;
        float accumulate_nearest = 0.0;
        for(int i = 0; i < iters; i++) {

            volatile uint32_t random = (uint32_t) distrib(gen);
            //std::cout << " Random: " << random << std::endl;

            float input = gaussian_distribution(accumulate_generator);

            if(input < 0.0) {
                input = -1.0 * (input * input);
            } else {
                input = input * input;
            }

            input = cast_p3109_signed_nearest(input * 5, 5, mode, true);

            // float ans_sto = cast_p3109_signed_stochastic(input, 5, random, 10, true);
            // float ans_nearest = cast_p3109_signed_nearest(input, 5, true);       

            // accumulate_float += input;
            // accumulate_stochastic += ans_sto;
            // accumulate_nearest += ans_nearest;    

            accumulate_float += input;
            accumulate_stochastic = cast_p3109_signed_stochastic(input + accumulate_stochastic, 5, random, 23, mode, true);
            accumulate_nearest = cast_p3109_signed_nearest(input + accumulate_nearest, 5, mode, true); 

            diffResults[a][i] = std::abs(accumulate_stochastic - accumulate_float);
            diffResults[a+repeats][i] = std::abs(accumulate_nearest - accumulate_float);
            

        }
    }

    std::ofstream file("stochastic_test.csv");


    for (int i = 0; i < iters; i++) {
        float sum_stochastic = 0.0;
        float sum_nearest = 0.0;
        for (int j = 0; j < repeats; j++) {
            sum_stochastic += diffResults[j][i];
            sum_nearest += diffResults[j+repeats][i];
        }
        file << sum_stochastic/repeats << "," << sum_nearest/repeats << std::endl;
    }

    

    file.close();

    


    return 0;
}