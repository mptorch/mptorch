from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from numpy import savetxt
import random
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from mptorch import FloatingPoint
import mptorch.quant as qpt
from mptorch.utils import trainer
import argparse
from tqdm import tqdm
import csv

parser = argparse.ArgumentParser(
    description="MLP Mixed Precision Error Analysis Script"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
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
    "--num_layers",
    type=int,
    default=3,
    metavar="N",
    help="number of layers in the MLP network (default: 3)",
)
parser.add_argument(
    "--num_examples",
    type=int,
    default=1,
    metavar="N",
    help="number of test examples to analyze (default: 3)",
)
parser.add_argument(
    "--seed", type=int, default=1234, metavar="S", help="random seed (default: 1234)"
)
parser.add_argument(
    "--train", action="store_true", default=False, help="trains the model beforehand"
)
parser.add_argument(
    "--cls",
    type=int,
    default=7,
    metavar="N",
    help="class of examples to examine (default: 7)",
)

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

torch.backends.cudnn.deterministic = True
network_name = f"mlp_{args.num_layers}"

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
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
test_loader_post = DataLoader(test_dataset, batch_size=1, shuffle=False)

"""Specify the formats and quantization functions for the layer operations and signals"""

fp32 = FloatingPoint(exp=8, man=23, subnormals=True)
fe5m10 = FloatingPoint(exp=5, man=10, subnormals=True)
fe4m3 = FloatingPoint(exp=4, man=3, subnormals=True)
fe2m1 = FloatingPoint(exp=2, man=1, subnormals=True)
quant_fp = lambda x: x
quant_fpe5m10 = lambda x: qpt.float_quantize(
    x, exp=5, man=10, rounding="nearest", subnormals=True, saturate=True
)
quant_fpe4m3 = lambda x: qpt.float_quantize(
    x, exp=4, man=3, rounding="nearest", subnormals=True, saturate=True
)
quant_fpe2m1 = lambda x: qpt.float_quantize(
    x, exp=2, man=1, rounding="nearest", subnormals=True, saturate=True
)

layer_format = lambda format: qpt.QAffineFormats(
    fwd_mac=(format,),
    fwd_rnd="nearest",
    bwd_mac=(format,),
    bwd_rnd="nearest",
    s=0,
)

layer_formats_fp32 = qpt.QAffineFormats(
    weight_quant=quant_fpe4m3,
    bias_quant=quant_fpe4m3,
)

layer_formats_e5m10 = qpt.QAffineFormats(
    fwd_mac=(fe5m10,),
    fwd_rnd="nearest",
    bwd_mac=(fe5m10,),
    bwd_rnd="nearest",
    weight_quant=quant_fpe5m10,
    bias_quant=quant_fpe5m10,
    s=0,
)

layer_formats_e4m3 = qpt.QAffineFormats(
    fwd_mac=(fe4m3,),
    fwd_rnd="nearest",
    bwd_mac=(fe4m3,),
    bwd_rnd="nearest",
    weight_quant=quant_fpe4m3,
    bias_quant=quant_fpe4m3,
    s=0,
)

layer_formats_e2m1 = qpt.QAffineFormats(
    fwd_mac=(fe2m1,),
    fwd_rnd="nearest",
    bwd_mac=(fe2m1,),
    bwd_rnd="nearest",
    weight_quant=quant_fpe2m1,
    bias_quant=quant_fpe2m1,
    s=0,
)


def d_relu(x):
    return (x >= torch.zeros_like(x)) * torch.ones_like(x)


def kappa_phi(v, g, phi):
    return (v >= torch.zeros_like(v)) * torch.ones_like(v)


class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 28 * 28)


class ErrorAnalyzer:
    def __init__(self, tol) -> None:

        # low Precision model
        mp_layers = OrderedDict()
        mp_layers["reshape"] = Reshape()
        for i in range(1, args.num_layers - 1):
            mp_layers[f"lin{i}"] = qpt.QLinearMP(
                784,
                784,
                formats=layer_format(fe4m3),
                activation=torch.relu,
                d_activation=d_relu,
                tol=tol,
                bias=True,
                kappa_phi_f=kappa_phi,
            )
            mp_layers[f"act{i}"] = nn.ReLU()
        mp_layers[f"lin{args.num_layers-1}"] = qpt.QLinearMP(
            784,
            128,
            formats=layer_format(fe4m3),
            activation=torch.relu,
            d_activation=d_relu,
            tol=tol,
            bias=True,
            kappa_phi_f=kappa_phi,
        )
        mp_layers[f"act{args.num_layers-1}"] = nn.ReLU()
        mp_layers[f"lin{args.num_layers}"] = qpt.QLinearMP(
            128,
            10,
            formats=layer_format(fe4m3),
            activation=lambda x: x,
            d_activation=lambda x: torch.ones_like(x),
            tol=tol,
            bias=True,
            kappa_phi_f=kappa_phi,
        )
        self._mp_model = nn.Sequential(mp_layers)
        self._mp_model.to(device)

        # full Precision Model
        fp_layers = OrderedDict()
        fp_layers["reshape"] = Reshape()
        for i in range(1, args.num_layers - 1):
            fp_layers[f"lin{i}"] = qpt.QLinear(
                784, 784, formats=layer_formats_fp32, bias=True
            )
            fp_layers[f"act{i}"] = nn.ReLU()
        fp_layers[f"lin{args.num_layers-1}"] = qpt.QLinear(
            784, 128, formats=layer_formats_fp32, bias=True
        )
        fp_layers[f"act{args.num_layers-1}"] = nn.ReLU()
        fp_layers[f"lin{args.num_layers}"] = qpt.QLinear(
            128, 10, formats=layer_formats_fp32, bias=True
        )
        self._fp_model = nn.Sequential(fp_layers)
        self._fp_model.to(device)

        # initialize dictionaries to store activations computed during the forward pass
        self._fp_activations = {}
        self._mp_activations = {}

        # name = name of the layer, ac_dict = dictionary
        def get_activation(name, activation_dict):
            def hook(model, input, output):
                activation_dict[name] = output.detach()

            return hook

        # add hooks to collect layers outputs
        def add_hooks(model, activation_dict):
            for name, module in model.named_modules():
                module.register_forward_hook(get_activation(name, activation_dict))

        # attach hook functions to modules
        add_hooks(self._fp_model, self._fp_activations)
        add_hooks(self._mp_model, self._mp_activations)

    def load(self, path):
        self._mp_model.load_state_dict(torch.load(path))

    def train(self, train_loader, test_loader):
        self._fp_model.to(device)
        self._fp_model.train()
        optimizer = torch.optim.SGD(
            self._fp_model.parameters(),
            lr=args.lr_init,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        trainer(
            self._fp_model,
            train_loader,
            test_loader,
            num_epochs=args.epochs,
            lr=args.lr_init,
            batch_size=args.batch_size,
            optimizer=optimizer,
            device=device,
        )

        for l in self._fp_model.modules():
            if isinstance(l, qpt.QLinear):
                l.quant_parameters()

        torch.save(self._fp_model.state_dict(), network_name)

        for name, param in self._mp_model.state_dict().items():
            if "weight" in name:
                param.copy_(self._fp_model.__getattr__(name.split(".")[0]).weight)
            elif "bias" in name:
                param.copy_(self._fp_model.__getattr__(name.split(".")[0]).bias)
            else:
                continue

    def set_tol(self, tol):
        for l in self._mp_model.modules():
            if isinstance(l, qpt.QLinearMP):
                l.tol = tol

    def experiment_apriori(self, input, tol, name=""):
        X, _ = input
        X = X.to(device)
        mp_model = self._mp_model
        activations = self._mp_activations
        linear_layers = []
        for l in mp_model.modules():
            if isinstance(l, qpt.QLinearMP):
                linear_layers.append(l)

        def set_tol(tol):
            for l in linear_layers:
                l.tol = tol

        def collect_values(name):
            for i in range(1, len(linear_layers) + 1):
                prec = linear_layers[i - 1].prec
                kappa_phi = linear_layers[i - 1].kappa_phi
                kappa_v = linear_layers[i - 1].kappa_v

                savetxt(f"temp/prec_{i}-" + name + ".csv", prec.detach().cpu().numpy())
                savetxt(
                    f"temp/kappa_v{i}-" + name + ".csv", kappa_v.detach().cpu().numpy()
                )
                savetxt(
                    f"temp/kappa_phi{i}-" + name + ".csv",
                    kappa_phi.detach().cpu().numpy(),
                )

                if i < len(linear_layers):
                    savetxt(
                        f"temp/output{i}-" + name + ".csv",
                        activations[f"act{i}"].detach().cpu().numpy(),
                    )
                else:
                    savetxt(
                        f"temp/output{i}-" + name + ".csv",
                        activations[f"lin{i}"].detach().cpu().numpy(),
                    )

        # run mix precision model
        set_tol(tol)
        mp_model(X)
        collect_values(name + "-mp")

        # run low precision models
        prior_activations = []
        for l in mp_model.modules():
            if isinstance(l, qpt.QLinearMP):
                prior_activations.append(l.activation)
                l.activation = None

        # FP8
        for l in mp_model.modules():
            if isinstance(l, qpt.QLinearMP):
                l.formats.s = 2
        mp_model(X)
        collect_values(name + "-fp8")

        # FP16
        for l in mp_model.modules():
            if isinstance(l, qpt.QLinearMP):
                l.formats.s = 1
        mp_model(X)
        collect_values(name + "-fp16")

        # FP32
        for l in mp_model.modules():
            if isinstance(l, qpt.QLinearMP):
                l.formats.s = 0
        mp_model(X)
        collect_values(name + "-fp32")

        # Reset model to mix_precision
        for l, activ in zip(linear_layers, prior_activations):
            l.activation = activ

        return


def accuracy_info(analyser, tol):
    num_correct_mp = 0.0
    num_correct_fp8 = 0.0
    num_correct_fp16 = 0.0
    num_correct_fp32 = 0.0
    num_ip_fp8 = 0.0
    num_ip = 0.0
    num_kappa_phi = 0.0
    num_ip_kappa_phi = 0.0
    num_total = 0
    num_examples = {}

    layers_analyser = []

    fp8_count = np.zeros(len(test_loader_post))
    kappa_phi_count = np.zeros(len(test_loader_post))
    total_count = np.zeros(args.num_layers)

    analyser.set_tol(tol)
    for l in analyser._mp_model.modules():
        if isinstance(l, qpt.QLinearMP):
            layers_analyser.append(l)
    for i in range(args.num_layers):
        total_count[i] = layers_analyser[i].prec.numel()

    progress_bar = tqdm(total=len(test_loader_post))
    for i, (X, y) in enumerate(test_loader_post):
        num_examples[int(y)] = num_examples.get(int(y), 0) + 1
        X, y = X.to(device), y.to(device)

        for j in range(len(layers_analyser)):
            if j < len(layers_analyser) - 1:
                layers_analyser[j].activation = torch.relu
                layers_analyser[j].d_activation = lambda x: (
                    (x >= torch.zeros_like(x)) * torch.ones_like(x)
                ).to(x.device)
            else:
                layers_analyser[j].activation = lambda x: x
                layers_analyser[j].d_activation = lambda x: torch.ones_like(x).to(
                    x.device
                )

        _, predicted_mp = torch.max(analyser._mp_model(X), dim=1)
        correct_mp = (predicted_mp == y).sum()
        num_correct_mp += correct_mp

        # sum of nb of IP with fp8 and nb of IP for each layer
        for j in range(len(layers_analyser)):
            num_ip_fp8 += layers_analyser[j].count
            num_ip += layers_analyser[j].prec.numel()

        # for each data number i, (nb of IP with fp8 / nb of IP)
        fp8_count[i] = num_ip_fp8 / num_ip

        # sum of nb of IP with kappa_phi=0 and nb of IP for each layer except the last one
        for j in range(len(layers_analyser) - 1):
            num_kappa_phi += layers_analyser[j].count_phi
            num_ip_kappa_phi += layers_analyser[j].prec.numel()

        # for each data number i, (nb of IP with zero kappa_phi / nb of IP)
        kappa_phi_count[i] = num_kappa_phi / num_ip_kappa_phi

        for j in range(len(layers_analyser)):
            layers_analyser[j].activation = None

        for j in range(len(layers_analyser)):
            layers_analyser[j].formats.s = 2

        _, predicted_fp8 = torch.max(analyser._mp_model(X), dim=1)
        correct_fp8 = (predicted_fp8 == y).sum()
        num_correct_fp8 += correct_fp8

        for j in range(len(layers_analyser)):
            layers_analyser[j].formats.s = 1

        _, predicted_fp16 = torch.max(analyser._mp_model(X), dim=1)
        correct_fp16 = (predicted_fp16 == y).sum()
        num_correct_fp16 += correct_fp16

        for j in range(len(layers_analyser)):
            layers_analyser[j].formats.s = 0

        _, predicted_fp32 = torch.max(analyser._mp_model(X), dim=1)
        correct_fp32 = (predicted_fp32 == y).sum()
        num_correct_fp32 += correct_fp32
        num_total += y.shape[0]
        progress_bar.update(1)

    test_acc_mp = float(num_correct_mp) / num_total
    test_acc_fp8 = float(num_correct_fp8) / num_total
    test_acc_fp16 = float(num_correct_fp16) / num_total
    test_acc_fp32 = float(num_correct_fp32) / num_total

    print(f"test_acc fp8 = {test_acc_fp8}")
    print(f"test_acc fp16 = {test_acc_fp16}")
    print(f"test_acc fp32 = {test_acc_fp32}")
    print(f"test_acc mp = {test_acc_mp}")
    print(f"mean = {1- np.mean(fp8_count)}")

    return (
        test_acc_fp8,
        test_acc_fp16,
        test_acc_fp32,
        test_acc_mp,
        1 - np.mean(fp8_count),
    )


def generate_statistics(analyser, num_examples, tol):
    for ex in range(num_examples):
        input = next(iter(train_loader))
        savetxt(
            f"temp/input{ex}.csv", input[0].view(-1, 28 * 28).detach().cpu().numpy()
        )
        analyser.experiment_apriori(
            input, tol, f"intra-layer-posteriori-8-16-cl{args.cls}-{ex}"
        )


# tolerances to test
tolerances = [1, 0.5, 0.3, 0.1]

# variables to store final test accuracies (they do not depend on tolerance)
test_acc_fp8 = None
test_acc_fp16 = None

# open a CSV file to store results for each tolerance
with open("results/mixed_precision.csv", mode="w", newline="") as mp_file:
    writer = csv.writer(mp_file)
    writer.writerow(["tol", "test_acc", "mean"])

    for tol in tolerances:
        print(f"Running mixed precision tests for tol {tol}...")
        analyser = ErrorAnalyzer(tol=tol)
        if args.train:
            analyser.train(train_loader, test_loader)  # uncomment to train network
        else:
            analyser.load(network_name)  # load network
        idx = train_dataset.targets == args.cls
        train_dataset.targets = train_dataset.targets[idx]
        train_dataset.data = train_dataset.data[idx]
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        print("Collect mixed precision statistics...")
        generate_statistics(analyser=analyser, num_examples=args.num_examples, tol=tol)
        print("Finished collecting statistics")
        print("Collecting accuracy info...")
        test_acc_fp8, test_acc_fp16, test_acc_fp32, test_acc_mp, mean = accuracy_info(
            analyser, tol=tol
        )

        writer.writerow([tol, test_acc_mp, mean])

# save full precision baseline results to a separate CSV
with open("results/full_precision.csv", mode="w", newline="") as fp_file:
    writer = csv.writer(fp_file)
    writer.writerow(["mode", "test_acc"])
    writer.writerow(["fp8", test_acc_fp8])
    writer.writerow(["fp16", test_acc_fp16])
