# Introduction to CUDA Programming

Here is an [Introduction to CUDA Programming](../../background/cuda_tutorial_nazar.pdf) by Nazar Misyats as well as a [recorded video](https://ubc.zoom.us/rec/share/oEx7T5BmisKQw9jc0VcLMg3fc2bJwhTilOmsekOBoSyuzxSIYLoldj02Y0KLwThE.cE48-5o8GBny4Yn_) (video requires the passcode "c@gKy1B=" without the quotes).

This introduction includes how to call a C++ CUDA function from Python.

# dev/cuda

The structure of this CUDA development directory is inspired by the one in the [llm.c](https://github.com/karpathy/llm.c/tree/master/dev/cuda) project.

It is a playground for developing various versions of the CUDA kernels that are needed in the project
and testing them out before actually integrating them into the `mptorch` library. Each file develops one kernel corresponding to one piece of functionality, with multiple versions of that kernel possibly being available, with potentially different running times and/or different code/time complexity.

Each file has a header section with information on how to compile and run the kernel. Alternatively, these commands are also grouped in the directory `Makefile`.

The first example you should study to build your kernels is `vec_add.cu` that performs the addition of two 1D vectors:

```
nvcc -O3 --use_fast_math -lcublas -lcublasLT vec_add.cu -o vec_add
```
or equivalently
```
make vec_add
```
The comments at the top then present the different versions of the kernel that are available. The overall idea with a new version is that it will be more complex, but should provide better running times. For example, in the case of `vec_add`, the first kernel can be run as:
```
./vec_add 1
```
You should see that it first runs the reference code on the CPU, then executes kernel 1 on the GPU, compares the two results to check for correctness, and then runs a number of configurations of this kernel (usually and most notably with varying block size) in order to time the kernel in these launch configurations. You can then inspect the second kernel:
```
./vec_add 2
```
This version also matches the CPU results, but it is faster. Integration into `mptorch` should be done with the kernel that ran fastest, manually adjusted (e.g. by hardcoding the best block size) and droping it into `quant/quant_cuda/` in its appropriate position.

To add a new version of a kernel, add the kernel to the corresponding file and adjust the docs. To add a new kernel, add the new file and adjust the Makefile (i.e., add the file to the list of targets and add the corresponding individual target entry). Run `make clean` to clean up binaries from your directory.
