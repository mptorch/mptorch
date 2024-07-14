import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from mptorch import FloatingPoint
import mptorch.quant as qpt
from mptorch.quant import cublas_acceleration
import os
from mptorch.optim import OptimMP
from mptorch.utils import trainer
import random
import numpy as np
import argparse
import wandb

# cublas_acceleration.enabled = True

parser = argparse.ArgumentParser(description="GPT Skakespeare Example")
# how many independent sequences will we process in parallel?
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
# what is the maximum context length for predictions?
parser.add_argument(
    "--block_size",
    type=int,
    default=256,
    metavar="N",
    help="context length block size (default: 256)",
)
parser.add_argument(
    "--seed", type=int, default=123, metavar="S", help="random seed (default: 123)"
)
parser.add_argument(
    "--max_iters",
    type=int,
    default=5000,
    metavar="N",
    help="number of iterations to train (default: 5000)",
)

parser.add_argument(
    "--eval_interval",
    type=int,
    default=200,
    metavar="N",
    help="evaluation interval (i.e. how often do we compute evaluation loss) (default: 200)",
)
parser.add_argument(
    "--n_embd",
    type=int,
    default=96,
    metavar="N",
    help="embedding dimension (default: 96)",
)
parser.add_argument(
    "--n_head",
    type=int,
    default=6,
    metavar="N",
    help="number of attention heads (default: 6)",
)
parser.add_argument(
    "--n_layer",
    type=int,
    default=6,
    metavar="N",
    help="number of Transformer layers (default: 6)",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=3e-4,
    metavar="N",
    help="learning rate (default: 3e-4)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--wandb", action="store_true", default=False, help="wandb logging"
)
parser.add_argument(
    "--expMac",
    type=int,
    default=5,
    metavar="N",
    help="MAC exponent size (default: 8)",
)
parser.add_argument(
    "--manMac",
    type=int,
    default=10,
    metavar="N",
    help="MAC mantissa size (default: 7)",
)
parser.add_argument(
    "--expWeight",
    type=int,
    default=5,
    metavar="N",
    help="Weights exponent size (default: 5)",
)
parser.add_argument(
    "--manWeight",
    type=int,
    default=2,
    metavar="N",
    help="Weights mantissa size (default: 2)",
)

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = "cuda" if args.cuda else "cpu"

rounding = "nearest"
"""Specify the formats and quantization functions for the layer operations and signals"""
fma_format = FloatingPoint(exp=args.expMac, man=args.manMac, subnormals=True, saturate=False)
w_format = FloatingPoint(exp=4, man=3, subnormals=True, saturate=False)
g_format = FloatingPoint(exp=5, man=2, subnormals=True, saturate=False)
i_format = FloatingPoint(exp=4, man=3, subnormals=True, saturate=False)
quant_g = lambda x: qpt.float_quantize(
    x, exp=g_format.exp, man=g_format.man, rounding=rounding, subnormals=True, saturate=False
)
quant_w = lambda x: qpt.float_quantize(
    x, exp=w_format.exp, man=w_format.man, rounding=rounding, subnormals=True, saturate=False
)
quant_b = lambda x: qpt.float_quantize(
    x, exp=fma_format.exp, man=fma_format.man, rounding=rounding, subnormals=True, saturate=False
)

layer_formats = qpt.QAffineFormats(
    fwd_mac=fma_format,
    fwd_rnd=rounding,
    bwd_mac=fma_format,
    bwd_rnd=rounding,
    weight_quant=quant_w,
    input_quant=quant_w,
    grad_quant=quant_g,
    bias_quant=quant_b,
    weight_scaled_format=w_format,
    grad_scaled_format=g_format,
    input_scaled_format=i_format,
)

# hyperparameters
eval_iters = 10  # 200
dropout = 0.2
# ------------

if args.wandb:
    run = wandb.init(
        # Set the project where this run will be logged
        project="shakespeare-gpt mptorch",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.learning_rate,
            "iterations": args.max_iters,
            "batch_size": args.batch_size,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "dropout": dropout,
            "fp_format_exp": args.expMac,
            "fp_format_man": args.manMac,
            "quant_fp_exp": args.expWeight,
            "quant_fp_man": args.manWeight,
        },
    )

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("data/shakespeare_input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([data[i : i + args.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + args.block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = qpt.QLinear(
            args.n_embd, head_size, bias=False, formats=layer_formats
        )
        self.query = qpt.QLinear(
            args.n_embd, head_size, bias=False, formats=layer_formats
        )
        self.value = qpt.QLinear(
            args.n_embd, head_size, bias=False, formats=layer_formats,
        )
        self.register_buffer(
            "tril", torch.tril(torch.ones(args.block_size, args.block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            # q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            qpt.qmatmul(q, k.transpose(-2, -1), formats=layer_formats)
            * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        # out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        out = qpt.qmatmul(wei, v, formats=layer_formats)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = qpt.QLinear(
            head_size * num_heads, args.n_embd, formats=layer_formats
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            qpt.QLinear(4 * n_embd, n_embd, formats=layer_formats),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, args.n_embd)
        self.position_embedding_table = nn.Embedding(args.block_size, args.n_embd)
        self.blocks = nn.Sequential(
            *[Block(args.n_embd, n_head=args.n_head) for _ in range(args.n_layer)]
        )
        self.ln_f = nn.LayerNorm(args.n_embd)  # final layer norm
        self.lm_head = qpt.QLinear(args.n_embd, vocab_size, formats=layer_formats)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -args.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

if os.path.exists("model.pth"):
    m.load_state_dict(torch.load("model.pth"))
else:
    # set up loss scaling
    init_scale = 2**16
    if device == "cpu" or init_scale is None:
        scaler = None
    else:
        scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    tq = tqdm(total=args.batch_size * args.max_iters)
    tq.set_description(f"Training progress")
    losses = estimate_loss()
    for iter in range(args.max_iters):
        # every once in a while evaluate the loss on train and val sets
        tq.update(args.batch_size)
        tq.set_postfix(
            train_loss="{:.5f}".format(losses["train"]),
            val_loss="{:.5f}".format(losses["val"]),
            epoch="{:d}".format(int(iter * args.batch_size / n)),
            scale="{:.3E}".format(scaler.get_scale() if scaler is not None else 0.0),
        )
        if args.wandb:
            wandb.log(
                {
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "epoch": int(iter * args.batch_size / n),
                    "iter": iter,
                    "scale": scaler.get_scale() if scaler is not None else 0.0,
                }
            )
        if (iter > 0 and iter % args.eval_interval == 0) or iter == args.max_iters - 1:
            losses = estimate_loss()

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

    tq.close()
    torch.save(m.state_dict(), "model.pth")

m.eval()
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
