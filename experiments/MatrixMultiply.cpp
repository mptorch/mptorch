#include <iostream>
#include <cmath>
#include <random>
#include <type_traits>
#include <vector>
#include <fstream>
#include <algorithm>
#include "../mptorch/quant/quant_cuda/p3109_kernel.cu"



int main(int argc, char **argv) {

    // number of formats to test
    unsigned numformats = 16;

    // Initialize output flag as false
    bool outputFlag = false;

    // Minimum arguments without the optional flag
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <width> <iterations> [-o | --output]" << std::endl;
        return 1;
    }

    // Variables to store width and iterations
    unsigned width = 0;
    unsigned iterations = 0;

    // Parse arguments in a loop
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-o" || arg == "--output") {
            outputFlag = true;
        } else {
            // Assuming the first non-flag argument is width and the second is iterations
            if (width == 0) {
                width = std::stoul(arg);
            } else if (iterations == 0) {
                iterations = std::stoul(arg);
            }
        }
    }

    // Check if width and iterations were set
    if (width == 0 || iterations == 0) {
        std::cerr << "Error: Missing required arguments." << std::endl;
        return 1;
    }


    // create two matrices of size width x width
    std::vector<std::vector<float>> A(width, std::vector<float>(width));
    std::vector<std::vector<float>> B(width, std::vector<float>(width));
    // 3d array for C, with 20 layers
    // std::vector<std::vector<std::vector<float>>> C(20, std::vector<std::vector<float>>(width, std::vector<float>(width)));
    std::vector<std::vector<float>> C(width * width * iterations, std::vector<float>(numformats + 1));

    std::vector<std::vector<std::vector<float>>> absDiff(width * width * iterations, std::vector<std::vector<float>>(numformats, std::vector<float>(numformats)));

    std::vector<std::vector<std::vector<float>>> percDiff(width * width * iterations, std::vector<std::vector<float>>(numformats, std::vector<float>(numformats)));

    std::vector<std::vector<float>> median( numformats, std::vector<float>(numformats));

    //max percentage difference
    std::vector<std::vector<float>> maxPercDiff( numformats, std::vector<float>(numformats));

    //max percentage difference dot product of A and B
    // A
    std::vector<std::vector<std::vector<float>>> maxPercDiffA(numformats, std::vector<std::vector<float>>(numformats, std::vector<float>(width)));
    // B
    std::vector<std::vector<std::vector<float>>> maxPercDiffB(numformats, std::vector<std::vector<float>>(numformats, std::vector<float>(width)));


    // fill matrices A and B with random values, from a normal distribution
    std::default_random_engine generator;
    //std::normal_distribution<float> distribution(0.0, (sqrt(2.0/(float)width)));

    // std::cout << "Std Dev: " << (sqrt(2.0/(float)width)) << std::endl;

    std::normal_distribution<float> distribution(0.0, 1.0);

    for(unsigned multiply_format = 0; multiply_format < numformats; multiply_format++){
        
        
        for (unsigned add_format = 0; add_format < numformats; add_format++) {
            //print multiplication and addition format
            std::cout << "Multiplication Format: " << multiply_format << std::endl;
            std::cout << "Addition Format: " << add_format << std::endl;

            std::mt19937 gen(1); // Mersenne Twister engine
            std::uniform_int_distribution<uint32_t> distrib(0, std::pow(2, 23)-1); // Uniform distribution in [0, 1)

            // scale factor
            /*
            # you should scale your data and choose the scale to be something slightly smaller than 
            # the square root of the max_float of the accumulation format in you matrix multiply kernel
            # (e.g. 200 is something that would be appropriate for binary16 accumulation, for instance)
            # this avoids overflows when you are doing multiplications, but in rare chances you might
            # get overflows when you are doing additions;
            # you can tweak this value to be something smaller to make overflows less likely to manifest
            */
            float scale = 20.0;
            
            for (unsigned iter = 0; iter < iterations; iter++) {
                //print iteration number
                //std::cout << "Iteration: " << iter << std::endl;

                //seed the generator
                generator.seed(iter + 1);   // +1 to avoid seed of 0
                for (unsigned i = 0; i < width; ++i) {
                    for (unsigned j = 0; j < width; ++j) {
                        float randomA = distribution(generator);
                        float randomB = distribution(generator);

                        if (randomA <= -4.0) {
                            randomA = -4.0;
                        } else if(randomA >= 4.0) {
                            randomA = 4.0;
                        }
                        if (randomB <= -4.0) {
                            randomB = -4.0;
                        } else if(randomB >= 4.0) {
                            randomB = 4.0;
                        }

                        if(randomA < 0.0) {
                            randomA = -1 * randomA * randomA;
                        } else {
                            randomA = randomA * randomA;
                        }
                        if(randomB < 0.0) {
                            randomB = -1 * randomB * randomB;
                        } else {
                            randomB = randomB * randomB;
                        }
                        A[i][j] = randomA;
                        B[i][j] = randomB;
                    }
                }
                //find max of each row of A and B and scale by 1/max
                for (unsigned i = 0; i < width; ++i) {
                    float maxA = 0.0;
                    float maxB = 0.0;
                    for (unsigned j = 0; j < width; ++j) {
                        if (A[i][j] > maxA) {
                            maxA = A[i][j];
                        }
                        if (B[i][j] > maxB) {
                            maxB = B[i][j];
                        }
                    }
                    for (unsigned j = 0; j < width; ++j) {
                        A[i][j] = A[i][j] / maxA * scale;
                        B[i][j] = B[i][j] / maxB * scale;
                    }
                }



                // multiply matrices A and B, store result in first row of matrix C as reference binary32 output
                if (add_format == 0) {
                    for (unsigned i = 0; i < width; ++i) {
                        for (unsigned k = 0; k < width; ++k) {
                            for (unsigned j = 0; j < width; ++j) {
                                C[(iter * width * width) + (i * width) + j][0] += A[i][k] * B[k][j];
                            }
                        }
                    }
                }
                

                // multiply matrices A and B, store result in matrix C
                for (unsigned i = 0; i < width; ++i) {
                    for (unsigned k = 0; k < width; ++k) {
                        for (unsigned j = 0; j < width; ++j) {

                            // std::random_device rd;  // Seed generator
                            // std::mt19937 gen(rd()); // Mersenne Twister engine
                            // std::uniform_int_distribution<uint32_t> distrib(0, std::pow(2, 23)-1); // Uniform distribution in [0, 1)
                            uint32_t random = (uint32_t) distrib(gen);

                            float product;
                            if(multiply_format == 0) {
                                product = cast_p3109_signed_nearest(A[i][k], 1, true) * cast_p3109_signed_nearest(B[k][j], 1, true);
                            } else if (multiply_format == 1) {
                                product = cast_p3109_signed_nearest(A[i][k], 2, true) * cast_p3109_signed_nearest(B[k][j], 2, true);
                            } else if (multiply_format == 2) {
                                product = cast_p3109_signed_nearest(A[i][k], 3, true) * cast_p3109_signed_nearest(B[k][j], 3, true);
                            } else if (multiply_format == 3) {
                                product = cast_p3109_signed_nearest(A[i][k], 4, true) * cast_p3109_signed_nearest(B[k][j], 4, true);
                            } else if (multiply_format == 4) {
                                product = cast_p3109_signed_nearest(A[i][k], 5, true) * cast_p3109_signed_nearest(B[k][j], 5, true);
                            } else if (multiply_format == 5) {
                                product = cast_p3109_signed_nearest(A[i][k], 6, true) * cast_p3109_signed_nearest(B[k][j], 6, true);
                            } else if (multiply_format == 6) {
                                product = cast_p3109_signed_nearest(A[i][k], 7, true) * cast_p3109_signed_nearest(B[k][j], 7, true);
                            } else if (multiply_format == 7) {
                                product = cast_p3109_signed_stochastic(A[i][k], 1, random, 23, true) * cast_p3109_signed_stochastic(B[k][j], 1, random, 23, true);
                            } else if (multiply_format == 8) {
                                product = cast_p3109_signed_stochastic(A[i][k], 2, random, 23, true) * cast_p3109_signed_stochastic(B[k][j], 2, random, 23, true);
                            } else if (multiply_format == 9) {
                                product = cast_p3109_signed_stochastic(A[i][k], 3, random, 23, true) * cast_p3109_signed_stochastic(B[k][j], 3, random, 23, true);
                            } else if (multiply_format == 10) {
                                product = cast_p3109_signed_stochastic(A[i][k], 4, random, 23, true) * cast_p3109_signed_stochastic(B[k][j], 4, random, 23, true);
                            } else if (multiply_format == 11) {
                                product = cast_p3109_signed_stochastic(A[i][k], 5, random, 23, true) * cast_p3109_signed_stochastic(B[k][j], 5, random, 23, true);
                            } else if (multiply_format == 12) {
                                product = cast_p3109_signed_stochastic(A[i][k], 6, random, 23, true) * cast_p3109_signed_stochastic(B[k][j], 6, random, 23, true);
                            } else if (multiply_format == 13) {
                                product = cast_p3109_signed_stochastic(A[i][k], 7, random, 23, true) * cast_p3109_signed_stochastic(B[k][j], 7, random, 23, true);
                            } else if (multiply_format == 14) {
                                product = cast_bfloat16_nearest(A[i][k]) * cast_bfloat16_nearest(B[k][j]);
                            } else {
                                product = A[i][k] * B[k][j];
                            }

                            random = (uint32_t) distrib(gen);

                            if(add_format == 0) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_nearest((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 1, true);
                            } else if (add_format == 1) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_nearest((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 2, true);
                            } else if (add_format == 2) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_nearest((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 3, true);
                            } else if (add_format == 3) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_nearest((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 4, true);
                            } else if (add_format == 4) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_nearest((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 5, true);
                            } else if (add_format == 5) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_nearest((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 6, true);
                            } else if (add_format == 6) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_nearest((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 7, true);
                            } else if (add_format == 7) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_stochastic((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 1, random, 23, true);
                            } else if (add_format == 8) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_stochastic((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 2, random, 23, true);
                            } else if (add_format == 9) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_stochastic((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 3, random, 23, true);
                            } else if (add_format == 10) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_stochastic((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 4, random, 23, true);
                            } else if (add_format == 11) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_stochastic((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 5, random, 23, true);
                            } else if (add_format == 12) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_stochastic((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 6, random, 23, true);
                            } else if (add_format == 13) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_p3109_signed_stochastic((C[(iter * width * width) + (i * width) + j][add_format + 1] + product), 7, random, 23, true);
                            } else if (add_format == 14) {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] =  cast_bfloat16_nearest((C[(iter * width * width) + (i * width) + j][add_format + 1] + product));
                            } else {
                                C[(iter * width * width) + (i * width) + j][add_format + 1] = C[(iter * width * width) + (i * width) + j][add_format + 1] + product;
                            }
                        }
                    }
                }

                //get max percentage difference
                for (unsigned i = 0; i < width; ++i) {
                    for (unsigned j = 0; j < width; ++j) {
                        //get absolute difference
                        absDiff[(iter * width * width) + (i * width) + j][multiply_format][add_format] = std::abs(C[(iter * width * width) + (i * width) + j][0] - C[(iter * width * width) + (i * width) + j][add_format + 1]);
                        //get percentage difference
                        percDiff[(iter * width * width) + (i * width) + j][multiply_format][add_format] = std::abs((absDiff[(iter * width * width) + (i * width) + j][multiply_format][add_format] / C[(iter * width * width) + (i * width) + j][0])) * 100;

                        //get max percentage difference
                        if (percDiff[(iter * width * width) + (i * width) + j][multiply_format][add_format] > maxPercDiff[multiply_format][add_format]) {
                            maxPercDiff[multiply_format][add_format] = percDiff[(iter * width * width) + (i * width) + j][multiply_format][add_format];

                            //store the dot product of A and B that caused the max percentage difference
                            maxPercDiffA[multiply_format][add_format] = A[i];
                            maxPercDiffB[multiply_format][add_format] = B[j];
                        }
                    }
                }


            }
        }

        if (outputFlag) {

            //if(multiply_format == 15) {
                // open file for writing, 1 file per multiplication format
                std::ofstream file("output" + std::to_string(multiply_format) + ".csv");

                // write header to file
                file << "Reference,Binary8p1 Round to Nearest,Binary8p2 Round to Nearest,Binary8p3 Round to Nearest,Binary8p4 Round to Nearest,Binary8p5 Round to Nearest,Binary8p6 Round to Nearest,Binary8p7 Round to Nearest,Binary8p1 Stochastic,Binary8p2 Stochastic,Binary8p3 Stochastic,Binary8p4 Stochastic,Binary8p5 Stochastic,Binary8p6 Stochastic,Binary8p7 Stochastic,bfloat 16 Round to Nearest,binary32 Round to Nearest" << std::endl;

                // write outputs of C to file
                for (unsigned i = 0; i < iterations * width * width; ++i) {
                    for (unsigned j = 0; j < numformats + 1; ++j) {
                        file << C[i][j];
                        if (j < numformats) {
                            file << ",";
                        }
                    }
                    file << std::endl;
                }

                file.close();
            //}
            
        }
        

        // for (unsigned i = 0; i < iterations * width * width; ++i) {
        //     for (unsigned j = 0; j < numformats; ++j) {
        //         absDiff[i][multiply_format][j] = std::abs(C[i][0] - C[i][j + 1]);
        //         percDiff[i][multiply_format][j] = std::abs((absDiff[i][multiply_format][j] / C[i][0])) * 100;

                

        //     }
        // }


        
        //clear outputs for C
        for (unsigned i = 0; i < iterations * width * width; ++i) {
            for (unsigned j = 0; j < numformats + 1; ++j) {
                C[i][j] = 0;
            }
        }
    }

    // calculate median of percentage differences
    for (unsigned i = 0; i < numformats; ++i) {
        for (unsigned j = 0; j < numformats; ++j) {
            std::vector<float> temp;
            for (unsigned k = 0; k < iterations * width * width; ++k) {
                temp.push_back(percDiff[k][i][j]);
            }
            std::sort(temp.begin(), temp.end());
            median[i][j] = temp[temp.size() / 2];
        }
    }

    // write median of percentage differences to file
    std::ofstream file("median.csv");
    for (unsigned i = 0; i < numformats; ++i) {
        for (unsigned j = 0; j < numformats; ++j) {
            if (j < numformats - 1) {
                file << median[j][i] << ",";
            } else {
                file << median[j][i];
            }
        }
        file << std::endl;
    }

    file.close();

    // write max percentage differences to file
    file.open("maxPercDiff.csv");
    for (unsigned i = 0; i < numformats; ++i) {
        for (unsigned j = 0; j < numformats; ++j) {
            if (j < numformats - 1) {
                file << maxPercDiff[j][i] << ",";
            } else {
                file << maxPercDiff[j][i];
            }
        }
        file << std::endl;
    }

    file.close();

    // // write A and B that caused max percentage difference of multiplication format 2 and addition format 2
    // file.open("maxPercDiff2_2.csv");
    // for (unsigned i = 0; i < width; ++i) {
    //     file << maxPercDiffA[2][2][i];
    //     file << std::endl;
    // }
    // file << std::endl;

    // for (unsigned i = 0; i < width; ++i) {
    //     file << maxPercDiffB[2][2][i];
    //     file << std::endl;
    // }

    // file.close();

    // // write A and B that caused max percentage difference of multiplication format 9 and addition format 9
    // file.open("maxPercDiff9_9.csv");
    // for (unsigned i = 0; i < width; ++i) {
    //     file << maxPercDiffA[9][9][i];
    //     file << std::endl;
    // }
    // file << std::endl;

    // for (unsigned i = 0; i < width; ++i) {
    //     file << maxPercDiffB[9][9][i];
    //     file << std::endl;
    // }

    // file.close();  



    return 0;
}