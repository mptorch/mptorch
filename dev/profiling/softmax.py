from torch.profiler import profile, record_function, ProfilerActivity
import torch
from mptorch.quant import float_quantize, QSoftmaxFormats
from mptorch.quant.functional import qsoftmax
from mptorch import FloatingPoint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shape", nargs="+", type=int, default=[500, 500, 500])
parser.add_argument("-d", "--dim", type=int, default=-1)
parser.add_argument("--man", type=int, default=10)
parser.add_argument("--exp", type=int, default=8)
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--pytorch", action="store_true", default=False)
parser.add_argument("--print", action="store_true", default=False)
parser.add_argument("--no-table", action="store_true", default=False)
args = parser.parse_args()

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

man, exp = args.man, args.exp
quantize = lambda x: float_quantize(
    x, exp=exp, man=man, rounding="nearest", subnormals=True, saturate=False
)
float_format = FloatingPoint(exp=exp, man=man, subnormals=True, saturate=False)
qsoftmax_formats = QSoftmaxFormats(
    fwd_exp=float_format,
    fwd_off=float_format,
    fwd_acc=float_format,

    bwd_add=float_format,
    bwd_mul=float_format,

    output_quant=quantize,
    input_quant=quantize,
    grad_quant=quantize,
)
dim = args.dim

device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
a = torch.rand(args.shape, device=device)

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, record_shapes=True) as prof:
    if not args.pytorch:
        with record_function("qsoftmax"):
            out = qsoftmax(a, dim, qsoftmax_formats)
    else:
        with record_function("torch.softmax"):
            out = torch.nn.functional.softmax(a, dim=dim)
if not args.no_table:
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if args.print:
    print(out)
