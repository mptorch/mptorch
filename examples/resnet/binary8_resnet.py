import torch
from torch.optim import SGD, AdamW
from mptorch import FloatingPoint
import mptorch.quant as qpt
from mptorch.optim import OptimMP
import torch.nn as nn
from mptorch.utils import trainer
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import random
import numpy as np
import argparse
import wandb


parser = argparse.ArgumentParser(description="ResNet CIFAR10 Example")
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
    "--exp_mul",
    type=int,
    default=5,
    metavar="N",
    help="exponent size (default: 5)",
)
parser.add_argument(
    "--man_mul",
    type=int,
    default=2,
    metavar="N",
    help="mantissa size (default: 2)",
)
parser.add_argument(
    "--exp_acc",
    type=int,
    default=8,
    metavar="N",
    help="exponent size (default: 8)",
)
parser.add_argument(
    "--man_acc",
    type=int,
    default=7,
    metavar="N",
    help="mantissa size (default: 7)",
)

parser.add_argument(
    "--lr_init",
    type=float,
    default=0.01,
    metavar="N",
    help="initial learning rate (default: 0.01)",
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
parser.add_argument(
    "--wandb", action="store_true", default=False, help="wandb logging"
)

# Precision (P)
parser.add_argument(
    "--P",
    type=int,
    default=3,
    metavar="N",
    help="precision (1-7)"
)

# subnormals
parser.add_argument(
    "--subnormals", action="store_true", default=True, help="subnormals or no subnormals"
)

# signed or unsigned
parser.add_argument(
    "--is_signed", action="store_true", default=True, help="signed or unsigned"
)

# prng_bits
parser.add_argument(
    "--prng_bits",
    type=int,
    metavar="N",
    default=17,
    help="number of random bits used for adding in stochatic"
)

# type of rounding
parser.add_argument(
    "--rounding",
    type=str,
    default="nearest",
    metavar="N",
    help="nearest, stochastic, truncate"
)

# type of saturation mode
parser.add_argument(
    "--overflow_policy",
    type=str,
    default="saturate_maxfloat",
    metavar="N",
    help="saturate_infty, saturate_maxfloat, saturate_maxfloat2"
)

# name of wandb project run will be in
parser.add_argument(
    "--wandb_proj_name",
    type=str,
    default="ResNet Tests AdamW",
    metavar="N",
    help="name of the project where runs will be logged"
)

# group within project file
parser.add_argument(
    "--group_name",
    type=str,
    default="P=3_Binary8_Testing",
    metavar="N",
    help="name of group the run will reside in"
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

fpmul = FloatingPoint(
    exp=args.exp_mul, man=args.man_mul, subnormals=True, saturate=True
)
fpacc = FloatingPoint(
    exp=args.exp_acc, man=args.man_acc, subnormals=True, saturate=True
)

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True
)

test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False
)

# define a lambda function so that the Quantizer module can be duplicated easily
act_error_quant = lambda: qpt.Quantizer(
    forward_number=fpmul,
    backward_number=fpmul,
    forward_rounding="nearest",
    backward_rounding="nearest",
)

param_q = lambda x: qpt.binary8_quantize(
    x,
    P = args.P,
    rounding=args.rounding,
    overflow_policy=args.overflow_policy,
    is_signed=True,      #args.is_signed,
    subnormals=args.subnormals,
    prng_bits=args.prng_bits,
)

input_q = lambda x: qpt.binary8_quantize(
    x,
    P = args.P,
    rounding=args.rounding,
    overflow_policy=args.overflow_policy,
    is_signed=args.is_signed,
    subnormals=args.subnormals,
    prng_bits=args.prng_bits,
)

grad_q = lambda x: qpt.binary8_quantize(
    x,
    P = args.P,
    rounding=args.rounding,
    overflow_policy=args.overflow_policy,
    is_signed=True,      #args.is_signed,
    subnormals=args.subnormals,
    prng_bits=args.prng_bits,
)

layer_formats = qpt.QAffineFormats(
    fwd_mac=(fpacc, fpmul),
    fwd_rnd="nearest",
    bwd_mac=(fpacc, fpmul),
    bwd_rnd="nearest",
    weight_quant=param_q,
    bias_quant=param_q,
    input_quant=input_q,
    grad_quant=grad_q,
)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
# make sure i am not misapplying unsigned quant 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = qpt.QConv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            formats=layer_formats,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = qpt.QConv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            formats=layer_formats,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.ae = act_error_quant()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    qpt.QConv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        formats=layer_formats,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.ae(out)
        out = F.relu(out)
        return out

# for first layer in network, should be signed input
# change one at a time (nn.Linear -> Qlinear)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = qpt.QConv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False, formats=layer_formats
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = qpt.QLinear(64, num_classes, formats=layer_formats)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

net = resnet20()
net = net.to(device)

# optimizer = SGD(
#     net.parameters(),
#     lr=args.lr_init,
#     momentum=args.momentum,
#     weight_decay=args.weight_decay,
# )

optimizer = AdamW(
    net.parameters(),
    lr=args.lr_init,
    weight_decay=args.weight_decay,
)

# acc_q = lambda x: qpt.binary8_quantize(
#     x,
#     P = args.P,
#     rounding=args.rounding,
#     overflow_policy=args.overflow_policy,
#     is_signed=True,
#     subnormals=args.subnormals,
#     prng_bits=args.prng_bits,
# )
acc_q = lambda x: qpt.float_quantize(x, exp=8, man=7, rounding="stochastic")

optimizer = OptimMP(optimizer, acc_quant=acc_q, momentum_quant=acc_q)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

trainer(
    net,
    train_loader,
    test_loader,
    num_epochs=args.epochs,
    lr=args.lr_init,
    batch_size=args.batch_size,
    optimizer=optimizer,
    device=device,
    scheduler=scheduler,
    init_scale=256.0,
    log_wandb = args.wandb,
)

wandb.finish()