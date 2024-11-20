import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from mptorch import FloatingPoint
import mptorch.quant as qpt
from mptorch.optim import QOptim
from mptorch.utils import trainer
import random
import numpy as np
import argparse
import wandb

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
    default=5,
    metavar="N",
    help="exponent size (default: 5)",
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

# new parser arguments for binary8 format and testing-----------------------
# weights and biases
parser.add_argument("--wandb", action="store_true", default=False, help="wandb logging")

# Precision (P)
parser.add_argument("--P", type=int, default=3, metavar="N", help="precision (1-7)")

# subnormals
parser.add_argument(
    "--subnormals",
    action="store_true",
    default=False,
    help="subnormals or no subnormals",
)

# signed or unsigned
parser.add_argument(
    "--is_signed", action="store_true", default=False, help="signed or unsigned"
)

# prng_bits
parser.add_argument(
    "--prng_bits",
    type=int,
    metavar="N",
    help="number of random bits used for adding in stochastic",
)

# type of rounding
parser.add_argument(
    "--rounding",
    type=str,
    default="nearest",
    metavar="N",
    help="nearest, stochatic, truncate",
)

# type of saturation mode
parser.add_argument(
    "--saturation_mode",
    type=str,
    default="overflow",
    metavar="N",
    help="saturate, overflow, no_overflow",
)

# precision size for accumuluation
parser.add_argument(
    "--p_q", type=int, default=3, metavar="N", help="precision size for accumulation"
)

# name of wandb project run will be in
parser.add_argument(
    "--wandb_proj_name",
    type=str,
    default="MLP Tests",
    metavar="N",
    help="name of the project where runs will be logged",
)

# group within project file
parser.add_argument(
    "--group_name",
    type=str,
    default="P=3",
    metavar="N",
    help="name of group the run will reside in",
)
# ------------------------------------------------------------------

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = "cuda" if args.cuda else "cpu"

# weights and biases configuration----------------------------------
if args.wandb:
    wandb.init(project=args.wandb_proj_name, config=args, group=args.group_name)
    config = wandb.config.update(args)
# ------------------------------------------------------------------

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
fp_format = FloatingPoint(
    exp=args.exp, man=args.man, subnormals=args.subnormals, saturate=False
)
quant_fp = lambda x: qpt.float_quantize(
    x,
    exp=args.exp,
    man=args.man,
    rounding=args.rounding,
    subnormals=args.subnormals,
    saturate=False,
)

layer_formats = qpt.QAffineFormats(
    fwd_mac=(fp_format, fp_format),
    fwd_rnd=args.rounding,
    bwd_mac=(fp_format, fp_format),
    bwd_rnd=args.rounding,
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

acc_q = lambda x: qpt.binary8_quantize(
    x, args.p_q, rounding=args.rounding, saturation_mode=args.saturation_mode
)
optimizer = QOptim(
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
    log_wandb=args.wandb,
)

wandb.finish()
