"""Eager text-only Qwen3.8. One unpadded sequence per forward call."""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AttentionState:
    keys: Tensor
    values: Tensor


@dataclass(frozen=True)
class DeltaState:
    history: Tensor
    memory: Tensor


LayerState = AttentionState | DeltaState


def normalize(x: Tensor, eps: float) -> Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        return (normalize(x, self.eps) * (1 + self.weight.float())).to(x.dtype)


class GatedAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.groups = cfg.num_attention_heads // cfg.num_key_value_heads
        q_size = cfg.num_attention_heads * self.head_dim
        kv_size = cfg.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, 2 * q_size, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_size, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_size, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(q_size, cfg.hidden_size, bias=cfg.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        rope = cfg.rope_parameters
        if rope["rope_type"] != "default":
            raise ValueError("only default text RoPE is supported")
        self.rotary_dim = int(self.head_dim * rope.get("partial_rotary_factor", 1.0))
        self.theta = rope["rope_theta"]

    def rotate(self, x, positions):
        # All three MRoPE axes use the same position for text-only inputs.
        dims = torch.arange(0, self.rotary_dim, 2, device=x.device).float()
        angles = torch.outer(positions.float(), self.theta ** (-dims / self.rotary_dim))
        cos = angles.cos().to(x.dtype).unsqueeze(1)
        sin = angles.sin().to(x.dtype).unsqueeze(1)
        first, second = x[..., :self.rotary_dim].chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin,
                          x[..., self.rotary_dim:]), dim=-1)

    def forward(self, x, positions, state):
        length = x.shape[0]
        q, gate = self.q_proj(x).reshape(length, -1, 2 * self.head_dim).chunk(2, dim=-1)
        k = self.k_proj(x).reshape(length, -1, self.head_dim)
        v = self.v_proj(x).reshape(length, -1, self.head_dim).transpose(0, 1)
        q = self.rotate(self.q_norm(q), positions).transpose(0, 1)
        k = self.rotate(self.k_norm(k), positions).transpose(0, 1)
        if state is not None:
            k = torch.cat((state.keys, k), dim=1)
            v = torch.cat((state.values, v), dim=1)
        keys = k.repeat_interleave(self.groups, dim=0)
        values = v.repeat_interleave(self.groups, dim=0)
        scores = (q @ keys.transpose(-1, -2)) * self.head_dim ** -0.5
        future = torch.arange(k.shape[1], device=x.device) > positions[:, None]
        weights = scores.masked_fill(future, -torch.inf).softmax(dim=-1, dtype=torch.float32)
        context = (weights.to(v.dtype) @ values).transpose(0, 1)
        output = self.o_proj((context * gate.sigmoid()).reshape(length, -1))
        return output, AttentionState(k, v)


class DeltaNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x, gate):
        scaled = normalize(x, self.eps).to(x.dtype) * self.weight
        return (scaled * F.silu(gate.float())).to(x.dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.key_dim = cfg.linear_key_head_dim
        self.value_dim = cfg.linear_value_head_dim
        self.key_heads = cfg.linear_num_key_heads
        self.value_heads = cfg.linear_num_value_heads
        key_size = self.key_heads * self.key_dim
        value_size = self.value_heads * self.value_dim
        channels = 2 * key_size + value_size
        self.in_proj_qkv = nn.Linear(cfg.hidden_size, channels, bias=False)
        self.in_proj_z = nn.Linear(cfg.hidden_size, value_size, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, self.value_heads, bias=False)
        self.in_proj_b = nn.Linear(cfg.hidden_size, self.value_heads, bias=False)
        self.conv1d = nn.Conv1d(channels, channels, cfg.linear_conv_kernel_dim,
                                groups=channels, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.value_heads))
        self.dt_bias = nn.Parameter(torch.ones(self.value_heads))
        self.norm = DeltaNorm(self.value_dim, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(value_size, cfg.hidden_size, bias=False)

    def unit_heads(self, x):
        heads = x.reshape(-1, self.key_heads, self.key_dim).float()
        heads = heads * torch.rsqrt(heads.square().sum(dim=-1, keepdim=True) + 1e-6)
        return heads.repeat_interleave(self.value_heads // self.key_heads, dim=1)

    def forward(self, x, positions, state):
        length = x.shape[0]
        raw = self.in_proj_qkv(x).T
        width = self.conv1d.kernel_size[0] - 1
        history = raw.new_zeros(raw.shape[0], width) if state is None else state.history
        window = torch.cat((history, raw), dim=-1)
        mixed = F.silu(self.conv1d(window)).T
        key_size = self.key_heads * self.key_dim
        q, k, v = mixed.split((key_size, key_size, self.value_heads * self.value_dim), dim=-1)
        q = self.unit_heads(q) * self.key_dim ** -0.5
        k = self.unit_heads(k)
        v = v.reshape(length, self.value_heads, self.value_dim).float()
        decay = (-self.A_log.float().exp() *
                 F.softplus(self.in_proj_a(x).float() + self.dt_bias.float())).exp()
        beta = self.in_proj_b(x).sigmoid().float()
        if state is None:
            memory = v.new_zeros(self.value_heads, self.key_dim, self.value_dim)
        else:
            memory = state.memory
        outputs = []
        for i in range(length):
            memory = memory * decay[i, :, None, None]
            key = k[i, :, :, None]
            prediction = (memory * key).sum(dim=1)
            correction = (v[i] - prediction) * beta[i, :, None]
            memory = memory + key * correction[:, None, :]
            outputs.append((memory * q[i, :, :, None]).sum(dim=1))
        gate = self.in_proj_z(x).reshape(length, self.value_heads, self.value_dim)
        out = self.norm(torch.stack(outputs).to(x.dtype), gate)
        return self.out_proj(out.flatten(1)), DeltaState(window[:, length:].contiguous(), memory)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_type):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "full_attention":
            self.self_attn = GatedAttention(cfg)
        elif layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(cfg)
        else:
            raise ValueError(f"unsupported layer type: {layer_type}")
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, positions, state):
        mixer = self.self_attn if self.layer_type == "full_attention" else self.linear_attn
        update, new_state = mixer(self.input_layernorm(x), positions, state)
        x = x + update
        return x + self.mlp(self.post_attention_layernorm(x)), new_state


class QwenText(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.model_type != "qwen3_5_text" or cfg.hidden_act != "silu":
            raise ValueError("expected a dense Qwen3.8-compatible text configuration")
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, kind) for kind in cfg.layer_types)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, tokens: Tensor, position: int = 0,
                states: tuple[LayerState, ...] = ()) -> tuple[Tensor, tuple[LayerState, ...]]:
        """Consume a contiguous chunk and return its hidden states plus a fresh cache."""
        x = self.embed_tokens(tokens)
        positions = torch.arange(position, position + len(tokens), device=x.device)
        next_states = []
        for index, layer in enumerate(self.layers):
            x, state = layer(x, positions, states[index] if states else None)
            next_states.append(state)
        return self.norm(x), tuple(next_states)
