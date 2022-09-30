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

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

"""Hyperparameters"""
batch_size = 64  # batch size
lr_init = 0.1  # initial learning rate
num_epochs = 2  # epochs
momentum = 0.9
weight_decay = 0
exp = 5
man = 2

float_format = FloatingPoint(exp=exp, man=man, subnormals=True, saturate=False)
mac_format = FloatingPoint(exp=exp, man=man, subnormals=True, saturate=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(
    "./data", train=False, transform=transform_test, download=False
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# define a lambda function so that the Quantizer module can be duplicated easily
act_error_quant = Quantizer(
    forward_number=float_format,
    backward_number=float_format,
    forward_rounding="nearest",
    backward_rounding="nearest",
)

param_q = lambda x: float_quantize(
    x, exp=exp, man=man, rounding="nearest", subnormals=True, saturate=False
)

layer_formats = QAffineFormats(
    fwd_mac=(mac_format, mac_format),
    bwd_mac=(mac_format, mac_format),
    fwd_rnd="nearest",
    bwd_rnd="nearest",
    weight_quant=param_q,
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
        # self.bn1 = QBatchNorm2d(6, fwd_quant=batchnorm_q, bwd_quant=batchnorm_q)
        # self.bn2 = QBatchNorm2d(16, fwd_quant=batchnorm_q, bwd_quant=batchnorm_q)

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = act_error_quant(x)
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
    model.parameters(), lr=lr_init, momentum=momentum, weight_decay=weight_decay
)

trainer(
    model,
    train_loader,
    test_loader,
    num_epochs=num_epochs,
    lr=lr_init,
    batch_size=batch_size,
    optimizer=optimizer,
    device=device,
    init_scale=128.0,
)
