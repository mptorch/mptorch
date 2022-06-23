import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from mptorch import FloatingPoint
import mptorch.quant as qpt
from mptorch.optim import OptimMP
from mptorch.utils import trainer, trainer_old

import random
import numpy as np

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

"""Hyperparameters"""
batch_size = 64  # batch size
lr_init = 0.05  # initial learning rate
num_epochs = 10  # epochs
momentum = 0.9
weight_decay = 0

"""Prepare the transforms on the dataset"""
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

"""download dataset: MNIST"""
train_dataset = torchvision.datasets.MNIST(
    "./data", train=True, transform=transform, download=True
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(
    "./data", train=False, transform=transform, download=False
)
test_loader = DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False)

"""Specify the formats and quantization functions for the layer operations and signals"""
exp, man = 3, 2
fp_format = FloatingPoint(exp=exp, man=man, subnormals=True, saturate=True)
quant_fp = lambda x: qpt.float_quantize(
    x, exp=exp, man=man, rounding="nearest", subnormals=True, saturate=True
)
# quant_fxp = lambda x: qpt.fixed_point_quantize(x, wl=22, fl=16, rounding="nearest")

layer_formats = qpt.QAffineFormats(
    fwd_add=fp_format,
    fwd_mul=fp_format,
    fwd_rnd="nearest",
    bwd_add=fp_format,
    bwd_mul=fp_format,
    bwd_rnd="nearest",
    param_quant=quant_fp,
    input_quant=quant_fp,
    grad_quant=quant_fp,
)

"""Construct the model"""


class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 28 * 28)


model = nn.Sequential(
    Reshape(),
    qpt.QLinear(784, 128, formats=layer_formats),
    nn.ReLU(),
    qpt.QLinear(128, 96, formats=layer_formats),
    nn.ReLU(),
    qpt.QLinear(96, 10, formats=layer_formats),
)

"""Prepare and launch the training process"""
model = model.to(device)
optimizer = SGD(
    model.parameters(), lr=lr_init, momentum=momentum, weight_decay=weight_decay
)

weight_q = lambda x: qpt.float_quantize(x, exp=8, man=23, rounding="nearest")
acc_q = lambda x: qpt.float_quantize(x, exp=8, man=23, rounding="nearest")
optimizer = OptimMP(
    optimizer,
    weight_quant=weight_q,
    acc_quant=acc_q,
    momentum_quant=acc_q,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

trainer(
    model,
    train_loader,
    test_loader,
    num_epochs=num_epochs,
    lr=lr_init,
    batch_size=batch_size,
    optimizer=optimizer,
    device=device,
    init_scale=1024.0,
)
