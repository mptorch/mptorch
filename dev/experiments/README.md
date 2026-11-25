# dev/experiments

Accuracy studies on the shipped library: questions about numerics that
decide whether a feature belongs in the library, answered by measuring
the installed ops on real models and data. A study changes no library
code; it applies what it tests in Python, around the shipped ops.

`dev/benchmarks/` is for the tools that check or time shipped code: its
sweeps, gates and timings. A study's answer is an accuracy or an error, not a
pass, a fail or a time.

Each script's docstring says what it measures, how to run it, and what it
found, with the date and the hardware.

| study | the question | how to run |
|---|---|---|
| `hadamard_rotation.py` | does rotating both operands of a product along K with a Walsh-Hadamard transform, before block quantization, help `MXFP8_E4M3`, `MXFP4_E2M1` and `NVFP4`? On synthetic data, GPT-2 small's activations and perplexity, and the tutorial's MNIST MLP | `uv run --group hf python dev/experiments/hadamard_rotation.py {check,tensors,perplexity,mlp} --device cuda`; `perplexity` is 17-33 s a recipe on an RTX 4060 laptop GPU, 22 recipes at the default sizes; `--multiples 1,4` and `--formats` narrow it |
