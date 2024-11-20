import torch
from torch.optim import SGD
from mptorch import FloatingPoint, SuperNormalFloat
import mptorch.quant as qpt
from mptorch.optim import QOptim
import torch.nn as nn
from mptorch.utils import trainer
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import random
import numpy as np
import argparse
import os
import wandb

parser = argparse.ArgumentParser(description="ResNet CIFAR10 Example")
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--seed", type=int, default=123, metavar="S", help="random seed (default: 123)"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 200)",
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
    default=1e-4,
    metavar="N",
    help="weight decay value to be used by the optimizer (default: 5e-4)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)

parser.add_argument("--wandb", action="store_true", default=False, help="wandb logging")

parser.add_argument(
    "--resume", "-r", action="store_true", default=False, help="resume from checkpoint"
)

parser.add_argument(
    "--expMac",
    type=int,
    default=5,
    metavar="N",
    help="MAC exponent size (default: 5)",
)
parser.add_argument(
    "--manMac",
    type=int,
    default=10,
    metavar="N",
    help="MAC mantissa size (default: 10)",
)
parser.add_argument(
    "--expWeight",
    type=int,
    default=4,
    metavar="N",
    help="Weights exponent size (default: 4)",
)
parser.add_argument(
    "--manWeight",
    type=int,
    default=3,
    metavar="N",
    help="Weights mantissa size (default: 3)",
)

parser.add_argument(
    "--expGrad",
    type=int,
    default=5,
    metavar="N",
    help="Grad exponent size (default: 5)",
)
parser.add_argument(
    "--manGrad",
    type=int,
    default=2,
    metavar="N",
    help="Grad mantissa size (default: 2)",
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = "cuda" if args.cuda else "cpu"
qpt.cublas_acceleration.enabled = args.cuda

rounding = "nearest"
"""Specify the formats and quantization functions for the layer operations and signals"""
fp_format = FloatingPoint(
    exp=args.expMac, man=args.manMac, subnormals=True, saturate=False
)
w_format = FloatingPoint(exp=args.expWeight, man=args.manWeight, saturate=False)
g_format = FloatingPoint(exp=args.expGrad, man=args.manGrad, saturate=False)
i_format = FloatingPoint(exp=args.expWeight, man=args.manWeight, saturate=False)
quant_g = lambda x: qpt.float_quantize(
    x, exp=g_format.exp, man=g_format.man, rounding=rounding, saturate=False
)
quant_w = lambda x: qpt.float_quantize(
    x, exp=w_format.exp, man=w_format.man, rounding=rounding, saturate=False
)
quant_b = lambda x: qpt.float_quantize(
    x,
    exp=fp_format.exp,
    man=fp_format.man,
    rounding=rounding,
    subnormals=True,
    saturate=False,
)

layer_formats = qpt.QAffineFormats(
    fwd_mac=(fp_format),
    fwd_rnd=rounding,
    bwd_mac=(fp_format),
    bwd_rnd=rounding,
    weight_quant=quant_w,
    input_quant=quant_w,
    grad_quant=quant_g,
    bias_quant=quant_b,
)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
)

test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
)

if args.wandb:
    run = wandb.init(
        # Set the project where this run will be logged
        project="resnet20 torch",
        # Track hyperparameters and run metadata
        config={
            "iterations": args.epochs,
            "batch_size": args.batch_size,
        },
    )


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


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
        out = F.relu(out)
        return out


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

if args.resume:
    # load checkpoint
    print("==> Resuming from checkpoint...")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
else:
    start_epoch = 0

optimizer = SGD(
    net.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

trainer(
    net,
    train_loader,
    test_loader,
    start_epoch=start_epoch,
    num_epochs=args.epochs,
    lr=args.lr_init,
    batch_size=args.batch_size,
    optimizer=optimizer,
    device=device,
    scheduler=scheduler,
    # init_scale=256.0,
    log_wandb=args.wandb,
)

wandb.finish()
