## Floating-Point
- [Wikipedia article](https://en.wikipedia.org/wiki/Floating-point_arithmetic)
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://pages.cs.wisc.edu/~david/courses/cs552/S12/handouts/goldberg-floating-point.pdf)
- [Numerical Computing with IEEE Floating Point Arithmetic](https://cosweb1.fau.edu/~jmirelesjames/ODE_course/Numerical_Computing_with_IEEE_Floating_Point_Arithmetic.pdf) by Michael Overton, first six chapters are a good tutorial-style introduction to floating-point numbers

## Transformers and LLMs
- [Andrej Karpathy's lectures: NN from zero to hero](https://github.com/karpathy/nn-zero-to-hero): a nice and pedagogical introduction to PyTorch, DNN training and LLMs (you should definitely watch these and play with the code!)
- [Andrej Karpathy's llm.c project](https://github.com/karpathy/llm.c): LLM (GPT-2) training code in pure C/CUDA. This is an awesome project for seeing how to train an LLM from scratch and can serve as a starting point for implementing some of the functionalities that we want to include in mptorch over the summer.
- [ViT's for image classification on small datasets](https://github.com/s-chh/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10): a nice repo that adapts the original Vision Transformer architecture to small size datasets (of the MINIST/CIFAR10 variety). 
- [nanoGPT](https://github.com/karpathy/nanoGPT): PyTorch training code for small to medium size GPT-like modes (GPT-2 style) for text generation tasks
- [llama2.c](https://github.com/karpathy/llama2.c): from the repository README: "This repo is a "fullstack" train + inference solution for Llama 2 LLM, with focus on minimalism and simplicity."

## CUDA
The main references should be the CUDA C++ Programming guide and the best practices document (you should use the version of the documentation that corresponds to the one in your PyTorch installation, which on June 6th 2024 seems to be CUDA version 12.1):

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

For a nice introduction to CUDA programming, see the [presentation](cuda_tutorial_nazar.pdf) given by Nazar Misyats, with a recorded video available [here](https://ubc.zoom.us/rec/share/oEx7T5BmisKQw9jc0VcLMg3fc2bJwhTilOmsekOBoSyuzxSIYLoldj02Y0KLwThE.cE48-5o8GBny4Yn_) (video requires the passcode "c@gKy1B=" without the quotes). This introduction also talks about how to call a C++ CUDA function from Python.