import sys
# sys.path.insert(0, '/home/sfilip/Projects/mptorch/')
print(sys.path[0])
import torch 
from torch.optim import SGD
from mptorch import FloatingPoint
from mptorch.quant import float_quantize, QLinear, QConv2d, Quantizer
from mptorch.optim import OptimLP
import torch.nn as nn
from mptorch.utils import trainer
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random 
import numpy as np

seed=123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

'''Hyperparameters'''
batch_size = 64        # batch size
lr_init = 0.05         # initial learning rate
num_epochs = 10        # epochs
momentum = 0.9
weight_decay = 0
exp = 5
man = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

float32 = FloatingPoint(exp=8, man=23)
float16 = FloatingPoint(exp=5, man=10)
bfloat16 = FloatingPoint(exp=8, man=7)

transform_train = transforms.Compose([
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

'''download dataset: MNIST'''
train_dataset = torchvision.datasets.MNIST('./data', train=True, transform=transform_train, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform_test, download=False)
test_loader = DataLoader(test_dataset, batch_size=int(batch_size/2), shuffle=False)

float_format = FloatingPoint(exp=exp, man=man)
# define a lambda function so that the Quantizer module can be duplicated easily
act_error_quant = lambda : Quantizer(forward_number=float_format, backward_number=float_format,
                        forward_rounding="nearest", backward_rounding="nearest")

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 28 * 28)

model = nn.Sequential(
    Reshape(),
    act_error_quant(),
    QLinear(784, 128, man=man, exp=exp),
    nn.ReLU(),
    act_error_quant(),
    QLinear(128, 96, man=man, exp=exp),
    nn.ReLU(),
    act_error_quant(),
    QLinear(96, 10, man=man, exp=exp),
    act_error_quant()
)

model = model.to(device)
optimizer = SGD(model.parameters(), lr=lr_init, momentum=momentum, weight_decay=weight_decay)

trainer(model, train_loader, test_loader, num_epochs=num_epochs, lr=lr_init, 
        batch_size=batch_size, optimizer=optimizer, device=device)