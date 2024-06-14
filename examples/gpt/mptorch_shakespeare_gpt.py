# Modified version of pytorch equivalent

import wandb
import torch
import torch.nn as nn
from mptorch import FloatingPoint
import mptorch.quant as qpt
from mptorch.optim import OptimMP
from mptorch.utils import trainer
from torch.nn import functional as F
from tqdm import tqdm

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 200  # 500
learning_rate = 2e-4
device = "cpu"
eval_iters = 20  # 200
n_embd = 96
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

macE = 8
macM = 7

artE = 5
artM = 2

torch.manual_seed(1337)

# MPTorch Parmasean
rounding = "nearest"

init_scale = 4096

fp_format = FloatingPoint(macE, macM, subnormals=True, saturate=False)
quant_fp = lambda x: qpt.float_quantize(x, artE, artM, rounding=rounding, subnormals=True, saturate=False)

layer_formats = qpt.QAffineFormats(
    fwd_mac = (fp_format, fp_format),
    fwd_rnd = rounding,
    bwd_mac = (fp_format, fp_format),
    bwd_rnd = rounding,
    weight_quant = quant_fp,
    input_quant = quant_fp,
    grad_quant = quant_fp,
    bias_quant = quant_fp
)

"""
run = wandb.init(
    project = "gpt-mptorch-RS",

    config={
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
    "fp_format_exp": macE,
    "fp_format_man": macM,
    "quant_fp_exp": artE,
    "quant_fp_man": artM,
    }
)
"""

print("init_scale: ", init_scale, "; macE: ", macE, "; macM: ", macM, "; artE: ", artE, "; artM: ", artM, "; learning_rate: ", learning_rate)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
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
        self.key = qpt.QLinear(n_embd, head_size, bias=False, formats=layer_formats)
        self.query = qpt.QLinear(n_embd, head_size, bias=False, formats=layer_formats)
        self.value = qpt.QLinear(n_embd, head_size, bias=False, formats=layer_formats)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        print(wei)
        print(wei.size())
        print(wei.dim())
        wei = qpt.QSoftMax(wei, artM, artE, -1, True)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = qpt.QLinear(head_size * num_heads, n_embd, formats=layer_formats)
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
            qpt.QLinear(n_embd, 4 * n_embd, formats=layer_formats),
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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = qpt.QLinear(n_embd, vocab_size, formats=layer_formats)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, qpt.QLinear):
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
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = qpt.QSoftMax(logits, artM, artE, -1, True)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
tq = tqdm(total=batch_size * max_iters)
tq.set_description(f"Training progress")
losses = estimate_loss()

scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    tq.update(batch_size)
    tq.set_postfix(
        train_loss="{:.5f}".format(losses["train"]),
        val_loss="{:.5f}".format(losses["val"]),
        epoch="{:d}".format(int(iter * batch_size / n)),
        scale="{:.5f}".format(scaler.get_scale())
    )

    if (iter > 0 and iter % eval_interval == 0) or iter == max_iters - 1:
        losses = estimate_loss()
        # print(
        #     f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        # )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    #loss.backward()
    #optimizer.step()

    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()

    #wandb.log({"train_loss": losses["train"], "val_loss": losses["val"], "epoch": int(iter*batch_size/n), "iter": iter, "scale": scaler.get_scale() if scaler is not None else 0.0})

    #print(scaler.get_scale())

tq.close()
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('result.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))