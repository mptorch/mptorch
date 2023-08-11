import torch
from torch.optim import SGD
from mptorch import FloatingPoint
from mptorch.quant import (
    float_quantize,
    QLinear,
    QConv2d,
    Quantizer,
    QBatchNorm2d,
    QAffineFormats,
)
from mptorch.optim import OptimMP
import torch.nn as nn
import torch.nn.functional as F
from mptorch.utils import trainer
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="LeNet MNIST Example")
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
    default=0.1,
    metavar="N",
    help="initial learning rate (default: 0.1)",
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

float_format = FloatingPoint(
    exp=args.exp, man=args.man, subnormals=True, saturate=False
)
mac_format = FloatingPoint(exp=args.exp, man=args.man, subnormals=True, saturate=False)

transform_train = transforms.Compose(
    [
        transforms.Pad(2),
        transforms.ToTensor(),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.Pad(2),
        transforms.ToTensor(),
    ]
)

"""download dataset: MNIST"""
train_dataset = torchvision.datasets.MNIST(
    "./data", train=True, transform=transform_train, download=True
)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(
    "./data", train=False, transform=transform_test, download=False
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# define a lambda function so that the Quantizer module can be duplicated easily
act_error_quant = Quantizer(
    forward_number=float_format,
    backward_number=float_format,
    forward_rounding="nearest",
    backward_rounding="nearest",
)

param_q = lambda x: float_quantize(
    x, exp=args.exp, man=args.man, rounding="nearest", subnormals=True, saturate=False
)

layer_formats = QAffineFormats(
    fwd_mac=(mac_format, mac_format),
    bwd_mac=(mac_format, mac_format),
    fwd_rnd="nearest",
    bwd_rnd="nearest",
    weight_quant=param_q,
    bias_quant=param_q,
    input_quant=param_q,
    output_quant=param_q,
)

batchnorm_q = lambda x: float_quantize(
    x, exp=8, man=23, rounding="nearest", saturate=False
)


class QLenet(nn.Module):
    def __init__(self):
        super(QLenet, self).__init__()
        self.conv1 = QConv2d(1, 6, 5, formats=layer_formats, bias=False)
        self.conv2 = QConv2d(6, 16, 5, formats=layer_formats, bias=False)
        self.fc1 = QLinear(16 * 5 * 5, 120, formats=layer_formats)
        self.fc2 = QLinear(120, 84, formats=layer_formats)
        self.fc3 = QLinear(84, 10, formats=layer_formats)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = act_error_quant(x)
        x = F.max_pool2d(F.relu(x), (2, 2))
        x = self.conv2(x)
        x = self.bn2(x)
        x = act_error_quant(x)
        x = F.max_pool2d(F.relu(x), 2)
        x = x.contiguous().view(x.size()[0], -1)  # to 1 Dim
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = QLenet()
model = model.to(device)

optimizer = SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
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
    init_scale=128.0,
)
