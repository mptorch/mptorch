import torch
from torch.optim import SGD
from mptorch import FloatingPoint
import mptorch.quant as qpt
from mptorch.optim import OptimMP
import torch.nn as nn
from mptorch.utils import trainer
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import torch.nn.functional as F

"""Hyperparameters"""
batch_size = 64  # batch size
lr_init = 0.01  # initial learning rate
num_epochs = 150  # epochs
momentum = 0.9
weight_decay = 5e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

exp1 = 5
man1 = 2
exp2 = 5
man2 = 10

bfloat16 = FloatingPoint(exp=8, man=7)
bfloat24 = FloatingPoint(exp=8, man=15)

fpe5m2 = FloatingPoint(exp=exp1, man=man1, subnormals=True)
fpe5m10 = FloatingPoint(exp=exp2, man=man2, subnormals=True)

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
    train_set, batch_size=batch_size, shuffle=True
)

test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False
)

# define a lambda function so that the Quantizer module can be duplicated easily
act_error_quant = lambda: qpt.Quantizer(
    forward_number=fpe5m2,
    backward_number=fpe5m2,
    forward_rounding="nearest",
    backward_rounding="nearest",
)

param_q = lambda x: qpt.float_quantize(
    x, exp=exp1, man=man1, rounding="nearest", subnormals=True
)
input_q = lambda x: qpt.float_quantize(
    x, exp=exp1, man=man1, rounding="nearest", subnormals=True
)
grad_q = lambda x: qpt.float_quantize(
    x, exp=exp1, man=man1, rounding="nearest", subnormals=True
)

layer_formats = qpt.QAffineFormats(
    fwd_add=fpe5m2,
    fwd_mul=fpe5m2,
    fwd_rnd="nearest",
    bwd_add=fpe5m2,
    bwd_mul=fpe5m2,
    bwd_rnd="nearest",
    param_quant=param_q,
    input_quant=input_q,
    grad_quant=grad_q,
    fwd_fma=False,
    bwd_fma=False,
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


net = resnet32()
net = net.to(device)
optimizer = SGD(
    net.parameters(), lr=lr_init, momentum=momentum, weight_decay=weight_decay
)

weight_q = lambda x: qpt.float_quantize(
    x, exp=8, man=23, rounding="nearest", subnormals=True
)
acc_q = lambda x: qpt.float_quantize(
    x, exp=8, man=23, rounding="nearest", subnormals=True
)
optimizer = OptimMP(
    optimizer, weight_quant=weight_q, acc_quant=acc_q, momentum_quant=acc_q
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

trainer(
    net,
    train_loader,
    test_loader,
    num_epochs=num_epochs,
    lr=lr_init,
    batch_size=batch_size,
    optimizer=optimizer,
    device=device,
    scheduler=scheduler,
    loss_scaling=256.0,
)
