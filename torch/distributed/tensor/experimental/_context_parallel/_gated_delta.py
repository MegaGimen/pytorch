# GDN context parallel: Megatron-style all-to-all, not Ring Attention.
#
# Sequence-parallel activations [B, T/cp, H, D] become head-parallel
# [B, T, H/cp, D], the gated-delta recurrence runs on the full timeline,
# then the inverse all-to-all restores the sequence shard.
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from torch.nn.attention.gated_delta import gated_delta_rule as _gdn_local
import torch.nn.attention.gated_delta as _gdn_mod


def _cp_group(mesh: DeviceMesh) -> dist.ProcessGroup:
    return mesh.get_group()


def all_to_all_seq_to_head(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """[B, T_local, H, D] → [B, T_global, H_local, D]."""
    world = dist.get_world_size(group)
    if world == 1:
        return x
    if x.size(2) % world != 0:
        raise ValueError(f"num_heads {x.size(2)} must be divisible by cp_size {world}")
    batch, t_local, n_heads, dim = x.shape
    h_local = n_heads // world
    # Split heads, put destination rank first so all_to_all_single splits dim0.
    x = (
        x.reshape(batch, t_local, world, h_local, dim)
        .permute(2, 1, 0, 3, 4)
        .contiguous()
    )
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    # out[src] is that rank's sequence shard, our heads.
    out = out.permute(2, 0, 1, 3, 4).reshape(batch, world * t_local, h_local, dim)
    return out.contiguous()


def all_to_all_head_to_seq(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """[B, T_global, H_local, D] → [B, T_local, H, D]."""
    world = dist.get_world_size(group)
    if world == 1:
        return x
    batch, t_global, h_local, dim = x.shape
    if t_global % world != 0:
        raise ValueError(f"seq {t_global} must be divisible by cp_size {world}")
    t_local = t_global // world
    # Sequence is concatenated as rank0_chunk | rank1_chunk | ...
    x = x.reshape(batch, world, t_local, h_local, dim).permute(1, 2, 0, 3, 4).contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    out = out.permute(2, 1, 0, 3, 4).reshape(batch, t_local, world * h_local, dim)
    return out.contiguous()


def gated_delta_rule_cp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    mesh: DeviceMesh,
    **kwargs: Any,
) -> torch.Tensor:
    group = _cp_group(mesh)
    q = all_to_all_seq_to_head(query, group)
    k = all_to_all_seq_to_head(key, group)
    v = all_to_all_seq_to_head(value, group)
    # decay/beta are [B, T, H] — treat last dim as "heads" with D=1.
    decay_h = all_to_all_seq_to_head(decay.unsqueeze(-1), group).squeeze(-1)
    beta_h = all_to_all_seq_to_head(beta.unsqueeze(-1), group).squeeze(-1)
    out = _gdn_local(q, k, v, decay_h, beta_h, **kwargs)
    return all_to_all_head_to_seq(out, group)


def patch_gated_delta_rule(mesh: DeviceMesh) -> None:
    def _wrapped(*args, **kwargs):
        return gated_delta_rule_cp(*args, mesh=mesh, **kwargs)

    F.gated_delta_rule = _wrapped  # type: ignore[attr-defined]
    _gdn_mod.gated_delta_rule = _wrapped  # type: ignore[misc]


def restore_gated_delta_rule() -> None:
    F.gated_delta_rule = _gdn_local  # type: ignore[attr-defined]
    _gdn_mod.gated_delta_rule = _gdn_local  # type: ignore[misc]
