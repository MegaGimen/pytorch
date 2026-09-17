# Minimal Gated DeltaNet module. Weights are tiny on purpose so a 2-GPU
# FSDP+CP smoke can stay under a 500 MiB per-process CUDA cap.
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.attention import gated_delta as _gdn_ops


class GatedDeltaNet(nn.Module):
    """One GDN block: fused in-proj → gated delta rule → out-proj.

    Layout is ``[B, T, H, D]`` after the input projection is split into heads.
    Context parallelism is applied *inside* ``gated_delta_rule`` when the
    CP dispatcher is enabled (all-to-all heads, full sequence locally).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int | None = None,
        *,
        conv_kernel_size: int = 4,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        if hidden_size <= 0 or num_heads <= 0 or head_k_dim <= 0:
            raise ValueError("hidden_size, num_heads, head_k_dim must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim if head_v_dim is not None else head_k_dim
        self.qk_dim = num_heads * self.head_k_dim
        self.v_dim = num_heads * self.head_v_dim
        # q, k, v, z-gate, beta
        in_features = self.qk_dim * 2 + self.v_dim * 2 + num_heads
        self.in_proj = nn.Linear(hidden_size, in_features, bias=bias, **factory)
        self.out_proj = nn.Linear(self.v_dim, hidden_size, bias=bias, **factory)
        self.conv1d = nn.Conv1d(
            self.qk_dim + self.qk_dim + self.v_dim,
            self.qk_dim + self.qk_dim + self.v_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=self.qk_dim + self.qk_dim + self.v_dim,
            bias=False,
            **factory,
        )
        self.A_log = nn.Parameter(
            torch.log(torch.empty(num_heads, **factory).uniform_(1.0, 16.0))
        )
        self.dt_bias = nn.Parameter(torch.zeros(num_heads, **factory))

    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states: [B, T, C]
        batch, seq_len, _ = hidden_states.shape
        projected = self.in_proj(hidden_states)
        qk = self.qk_dim
        vd = self.v_dim
        q_lin, k_lin, v_lin, z_lin, beta_lin = torch.split(
            projected,
            [qk, qk, vd, vd, self.num_heads],
            dim=-1,
        )
        qkv = torch.cat([q_lin, k_lin, v_lin], dim=-1)
        qkv = qkv.transpose(1, 2)
        qkv = self.conv1d(qkv)[..., :seq_len]
        qkv = torch.nn.functional.silu(qkv).transpose(1, 2)
        q_lin, k_lin, v_lin = torch.split(qkv, [qk, qk, vd], dim=-1)

        q = q_lin.unflatten(-1, (self.num_heads, self.head_k_dim))
        k = k_lin.unflatten(-1, (self.num_heads, self.head_k_dim))
        v = v_lin.unflatten(-1, (self.num_heads, self.head_v_dim))
        beta = torch.sigmoid(beta_lin)
        # decay = -exp(A_log) * softplus(dt_bias + 0)  — per-token beta already
        # carries the write; A_log is a per-head log-decay, broadcast over T.
        decay = -self.A_log.exp() * torch.nn.functional.softplus(
            self.dt_bias
        )
        decay = decay.view(1, 1, self.num_heads).expand(batch, seq_len, self.num_heads)

        core = _gdn_ops.gated_delta_rule(q, k, v, decay, beta)
        core = core.flatten(-2)
        # z-gate on the value channel, same as GDN's output gate.
        z = torch.nn.functional.silu(z_lin)
        return self.out_proj(core * z)


class TinyGatedDeltaModel(nn.Module):
    """Stack of GDN blocks plus an embedding, for FSDP+CP smoke tests."""

    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 32,
        num_heads: int = 4,
        head_dim: int = 8,
        num_layers: int = 1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.embed = nn.Embedding(vocab_size, hidden_size, **factory)
        self.layers = nn.ModuleList(
            [
                GatedDeltaNet(
                    hidden_size,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(hidden_size, **factory)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, **factory)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)
