"""Small models written in Hugging Face's code patterns, for the instrumentation tests.

:class:`MiniLlama` and :class:`MiniGPT2` compute what transformers 5.24.0's
``LlamaForCausalLM`` and ``GPT2LMHeadModel`` compute, call for call, as a
torch function mode sees them: the same functions, on the same kinds of
operand (a Parameter, a buffer, an activation), in the same order, inside
modules of the same nesting and names. Sizes are whatever a test asks for,
and there is no KV cache, no padding mask and no generation, so the integer
bookkeeping transformers does for padded and packed batches is left out.
What they copy:

* **Llama.** RMSNorm upcast to float32 and written as pow, mean and rsqrt;
  bias-free ``nn.Linear`` projections; rotary embeddings computed once per
  forward as the positions times an ``inv_freq`` buffer (a broadcast
  multiply, not a matmul), then ``cat``, ``cos`` and ``sin``, applied with
  ``rotate_half``; grouped-query attention, either eager (``repeat_kv``, a
  matmul, ``* scaling``, an additive causal mask built once per forward,
  ``softmax(dtype=float32)``, a matmul) or through
  ``F.scaled_dot_product_attention`` with ``is_causal=True`` and
  ``enable_gqa=True``; a SwiGLU MLP whose activation is a module of its own
  calling ``F.silu``; an ``nn.Embedding``, and an ``lm_head`` that may be
  tied to it.
* **GPT-2.** ``Conv1D`` layers, a ``torch.addmm`` over a weight stored
  ``[in, out]``; ``nn.LayerNorm``; eager attention with ``* scaling``, the
  additive mask and a softmax with no dtype, or SDPA; the GELU written out
  (``gelu_new``) or as ``F.gelu(approximate="tanh")``; two embeddings and a
  tied ``lm_head``.

Each attention module takes optional ``qk`` and ``pv`` modules, used in
place of ``torch.matmul`` for its two products, so that a test can build
the same model with a :class:`~mptorch.quant.QMatmul` at each site by hand.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# --- Llama ---------------------------------------------------------------------------


class LlamaRMSNorm(nn.Module):
    """transformers' ``LlamaRMSNorm``."""

    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    """transformers' ``LlamaRotaryEmbedding``, with the default rope parameters."""

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        freqs = position_ids[..., None].float() * self.inv_freq.to(
            device=x.device, dtype=torch.float
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def causal_mask(length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """The additive mask transformers' eager attention gets: 0 where a query
    may attend, the dtype's lowest value where it may not, ``[1, 1, S, S]``."""
    q_idx = torch.arange(length, device=device)[:, None]
    kv_idx = torch.arange(length, device=device)[None, :]
    allowed = kv_idx <= q_idx
    zero = torch.zeros((), device=device, dtype=dtype)
    return torch.where(allowed, zero, torch.finfo(dtype).min)[None, None]


class LlamaAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, attn: str):
        super().__init__()
        self.head_dim = hidden // heads
        self.num_key_value_groups = heads // kv_heads
        self.scaling = self.head_dim**-0.5
        self.attn = attn
        self.q_proj = nn.Linear(hidden, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, hidden, bias=False)
        self.qk: nn.Module | None = None
        self.pv: nn.Module | None = None

    def forward(self, hidden_states, cos, sin, attention_mask):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if self.attn == "sdpa":
            gqa = {"enable_gqa": True} if self.num_key_value_groups > 1 else {}
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                scale=self.scaling,
                is_causal=query.shape[2] > 1,
                **gqa,
            )
        else:
            key = repeat_kv(key, self.num_key_value_groups)
            value = repeat_kv(value, self.num_key_value_groups)
            qk = torch.matmul if self.qk is None else self.qk
            pv = torch.matmul if self.pv is None else self.pv
            weights = qk(query, key.transpose(2, 3)) * self.scaling
            weights = weights + attention_mask
            weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = F.dropout(weights, p=0.0, training=self.training)
            out = pv(weights, value)
        out = out.transpose(1, 2).contiguous()
        return self.o_proj(out.reshape(*input_shape, -1).contiguous())


class SiLUActivation(nn.Module):
    """transformers' ``SiLUActivation``: a module of its own, not ``nn.SiLU``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class LlamaMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = SiLUActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, intermediate: int, attn: str):
        super().__init__()
        self.self_attn = LlamaAttention(hidden, heads, kv_heads, attn)
        self.mlp = LlamaMLP(hidden, intermediate)
        self.input_layernorm = LlamaRMSNorm(hidden)
        self.post_attention_layernorm = LlamaRMSNorm(hidden)

    def forward(self, hidden_states, cos, sin, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class LlamaModel(nn.Module):
    def __init__(self, vocab, hidden, layers, heads, kv_heads, intermediate, attn):
        super().__init__()
        self.attn = attn
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(hidden, heads, kv_heads, intermediate, attn) for _ in range(layers)
        )
        self.norm = LlamaRMSNorm(hidden)
        self.rotary_emb = LlamaRotaryEmbedding(hidden // heads)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        length = hidden_states.shape[1]
        position_ids = torch.arange(length, device=hidden_states.device).unsqueeze(0)
        mask = None
        if self.attn == "eager":
            mask = causal_mask(length, hidden_states.dtype, hidden_states.device)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, mask)
        return self.norm(hidden_states)


class MiniLlama(nn.Module):
    """A ``LlamaForCausalLM`` of the given size: ``forward(input_ids)`` returns the logits.

    Args:
        attn (str): ``"eager"`` or ``"sdpa"``.
        tie (bool): share the embedding's weight with ``lm_head``.
    """

    def __init__(
        self,
        vocab: int = 64,
        hidden: int = 32,
        layers: int = 2,
        heads: int = 4,
        kv_heads: int = 2,
        intermediate: int = 64,
        *,
        attn: str = "eager",
        tie: bool = False,
    ):
        super().__init__()
        self.model = LlamaModel(vocab, hidden, layers, heads, kv_heads, intermediate, attn)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if tie:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(input_ids))


# --- GPT-2 ---------------------------------------------------------------------------


class Conv1D(nn.Module):
    """``transformers.pytorch_utils.Conv1D``: a linear layer whose weight is
    stored ``[in, out]``, computed by ``torch.addmm``."""

    def __init__(self, nf: int, nx: int):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


class NewGELUActivation(nn.Module):
    """transformers' ``gelu_new``, GPT-2's default, written out."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        inner = math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))
        return 0.5 * input * (1.0 + torch.tanh(inner))


class GELUTanh(nn.Module):
    """transformers' ``gelu_pytorch_tanh``."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input, approximate="tanh")


class GPT2Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, attn: str, dropout: float):
        super().__init__()
        self.embed_dim = hidden
        self.head_dim = hidden // heads
        self.scaling = self.head_dim**-0.5
        self.attn = attn
        self.c_attn = Conv1D(3 * hidden, hidden)
        self.c_proj = Conv1D(hidden, hidden)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.qk: nn.Module | None = None
        self.pv: nn.Module | None = None

    def forward(self, hidden_states, attention_mask):
        query, key, value = self.c_attn(hidden_states).split(self.embed_dim, dim=2)
        shape = (*key.shape[:-1], -1, self.head_dim)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        query = query.view(shape).transpose(1, 2)
        dropout = self.attn_dropout.p if self.training else 0.0
        if self.attn == "sdpa":
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout,
                scale=self.scaling,
                is_causal=query.shape[2] > 1,
            )
        else:
            qk = torch.matmul if self.qk is None else self.qk
            pv = torch.matmul if self.pv is None else self.pv
            weights = qk(query, key.transpose(-1, -2)) * self.scaling
            weights = weights + attention_mask
            weights = F.softmax(weights, dim=-1)
            weights = weights.type(value.dtype)
            weights = F.dropout(weights, p=dropout, training=self.training)
            out = pv(weights, value)
        out = out.transpose(1, 2)
        out = out.reshape(*out.shape[:-2], -1).contiguous()
        return self.resid_dropout(self.c_proj(out))


class GPT2MLP(nn.Module):
    def __init__(self, hidden: int, act: str, dropout: float):
        super().__init__()
        self.c_fc = Conv1D(4 * hidden, hidden)
        self.c_proj = Conv1D(hidden, 4 * hidden)
        self.act = NewGELUActivation() if act == "gelu_new" else GELUTanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(hidden_states))))


class GPT2Block(nn.Module):
    def __init__(self, hidden: int, heads: int, attn: str, act: str, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden)
        self.attn = GPT2Attention(hidden, heads, attn, dropout)
        self.ln_2 = nn.LayerNorm(hidden)
        self.mlp = GPT2MLP(hidden, act, dropout)

    def forward(self, hidden_states, attention_mask):
        residual = hidden_states
        hidden_states = self.attn(self.ln_1(hidden_states), attention_mask)
        hidden_states = hidden_states + residual
        residual = hidden_states
        return residual + self.mlp(self.ln_2(hidden_states))


class GPT2Model(nn.Module):
    def __init__(self, vocab, positions, hidden, layers, heads, attn, act, dropout):
        super().__init__()
        self.attn = attn
        self.wte = nn.Embedding(vocab, hidden)
        self.wpe = nn.Embedding(positions, hidden)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList(GPT2Block(hidden, heads, attn, act, dropout) for _ in range(layers))
        self.ln_f = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        length = inputs_embeds.shape[1]
        position_ids = torch.arange(length, device=inputs_embeds.device).unsqueeze(0)
        hidden_states = inputs_embeds + self.wpe(position_ids).to(inputs_embeds.device)
        mask = None
        if self.attn == "eager":
            mask = causal_mask(length, hidden_states.dtype, hidden_states.device)
        hidden_states = self.drop(hidden_states)
        for block in self.h:
            hidden_states = block(hidden_states, mask)
        return self.ln_f(hidden_states)


class MiniGPT2(nn.Module):
    """A ``GPT2LMHeadModel`` of the given size, its ``lm_head`` tied to ``wte``:
    ``forward(input_ids)`` returns the logits.

    Args:
        attn (str): ``"eager"`` or ``"sdpa"``.
        act (str): ``"gelu_new"`` or ``"gelu_pytorch_tanh"``.
        dropout (float): every dropout's probability. Default: ``0.0``
    """

    def __init__(
        self,
        vocab: int = 64,
        positions: int = 32,
        hidden: int = 32,
        layers: int = 2,
        heads: int = 2,
        *,
        attn: str = "eager",
        act: str = "gelu_new",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.transformer = GPT2Model(vocab, positions, hidden, layers, heads, attn, act, dropout)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.transformer(input_ids))


def attention_modules(model: nn.Module) -> list[nn.Module]:
    """The model's attention modules, in order: where ``qk`` and ``pv`` go."""
    return [m for m in model.modules() if isinstance(m, LlamaAttention | GPT2Attention)]
