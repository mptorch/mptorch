import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from mptorch import FloatingPoint
import mptorch.quant as qpt
from mptorch.optim import OptimMP
from mptorch.utils import trainer
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="MLP MNIST Example")
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--seed", type=int, default=123, metavar="S", help="random seed (default: 123)"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--exp",
    type=int,
    default=4,
    metavar="N",
    help="exponent size (default: 4)",
)
parser.add_argument(
    "--man",
    type=int,
    default=2,
    metavar="N",
    help="mantissa size (default: 2)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.05,
    metavar="N",
    help="initial learning rate (default: 0.05)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="N",
    help="momentum value to be used by the optimizer (default: 0.9)",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0,
    metavar="N",
    help="weight decay value to be used by the optimizer (default: 0.0)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = "cuda" if args.cuda else "cpu"

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True


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
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(
    "./data", train=False, transform=transform, download=False
)
test_loader = DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=False)

"""Specify the formats and quantization functions for the layer operations and signals"""
fp_format = FloatingPoint(exp=args.exp, man=args.man, subnormals=True, saturate=False)
quant_fp = lambda x: qpt.float_quantize(
    x, exp=args.exp, man=args.man, rounding="nearest", subnormals=True, saturate=False
)

layer_formats = qpt.QAffineFormats(
    fwd_mac=(fp_format, fp_format),
    fwd_rnd="nearest",
    bwd_mac=(fp_format, fp_format),
    bwd_rnd="nearest",
    weight_quant=quant_fp,
    input_quant=quant_fp,
    grad_quant=quant_fp,
    bias_quant=quant_fp,
)

"""Construct the model"""


class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 28 * 28)


model = nn.Sequential(
    Reshape(),
    qpt.QLinear(784, 128, formats=layer_formats),
    # qpt.QLazyLinear(128, formats=layer_formats),
    nn.ReLU(),
    qpt.QLinear(128, 96, formats=layer_formats),
    # qpt.QLazyLinear(96, formats=layer_formats),
    nn.ReLU(),
    qpt.QLinear(96, 10, formats=layer_formats),
    # qpt.QLazyLinear(10, formats=layer_formats),
)
# model(train_dataset[0][0])

"""Prepare and launch the training process"""
model = model.to(device)
optimizer = SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)

acc_q = lambda x: qpt.float_quantize(x, exp=8, man=7, rounding="stochastic")
optimizer = OptimMP(
    optimizer,
    acc_quant=acc_q,
    momentum_quant=acc_q,
)

trainer(
    model,
    train_loader,
    test_loader,
    num_epochs=args.epochs,
    lr=args.lr_init,
    batch_size=args.batch_size,
    optimizer=optimizer,
    device=device,
    init_scale=256.0,
)
