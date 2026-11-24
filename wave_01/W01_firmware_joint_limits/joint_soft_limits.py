"""Minimal lookup-table soft-limit utilities used by deploy template actions."""
from __future__ import annotations

import math

import torch

from rl_isaaclab.utils import firmware_lookup_tables as _flt


# 手部关节软限位 (rad)，与固件 LUT 交集裁剪。
HAND_SOFT_LIMIT_LOWER: tuple[float, ...] = (
    -0.087266, -0.314159, -0.436332, -0.314159, 0.000000,
    -0.174533, -0.314159, 0.000000, 0.000000,
    -0.174533, -0.314159, 0.000000, 0.000000,
    -0.174533, -0.314159, 0.000000, 0.000000,
    0.000000, -0.174533, -0.314159, 0.000000, 0.000000,
)
HAND_SOFT_LIMIT_UPPER: tuple[float, ...] = (
    1.832596, 0.314159, 1.308997, 0.314159, 1.658063,
    1.466077, 0.314159, 1.658063, 1.308997,
    1.466077, 0.314159, 1.658063, 1.308997,
    1.466077, 0.314159, 1.658063, 1.308997,
    0.261799, 1.466077, 0.314159, 1.658063, 1.308997,
)


DEFAULT_DOF_LIMITS_SCALE: float = 1
LOOKUP_TABLE_MIN_ANGLE_DEG: float = -49.0
LOOKUP_TABLE_ANGLE_STEP_DEG: float = 1.0
LOOKUP_TABLE_SIZE: int = _flt.LOOKUP_TABLE_SIZE
RAD2DEG: float = 180.0 / math.pi

# Only differential (side) joints: (joint index, partner FE index for lookup, firmware table name).
_LOOKUP_JOINTS: tuple[tuple[int, int, str], ...] = (
    (1, 0, "THUMB_CMC_SIDE_LOOKUP_TABLE"),
    (3, 2, "THUMB_MCP_SIDE_LOOKUP_TABLE"),
    (6, 5, "INDEX_MCP_SIDE_LOOKUP_TABLE"),
    (10, 9, "MIDDLE_MCP_SIDE_LOOKUP_TABLE"),
    (14, 13, "RING_MCP_SIDE_LOOKUP_TABLE"),
    (19, 18, "PINKY_MCP_SIDE_LOOKUP_TABLE"),
)
_USED_TABLE_NAMES: tuple[str, ...] = tuple(sorted({t for _, _, t in _LOOKUP_JOINTS}))

_LOOKUP_TENSOR_CACHE: dict[tuple[str, str], dict[str, torch.Tensor]] = {}


def preload_lookup_tensors(device: torch.device, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    for name in _USED_TABLE_NAMES:
        cache[name] = torch.tensor(getattr(_flt, name), device=device, dtype=dtype)
    return cache


def get_lookup_tensor_cache(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    key = (str(device), str(dtype))
    if key not in _LOOKUP_TENSOR_CACHE:
        _LOOKUP_TENSOR_CACHE[key] = preload_lookup_tensors(device, dtype)
    return _LOOKUP_TENSOR_CACHE[key]


def init_lookup_tensor_cache(device: torch.device, dtype: torch.dtype = torch.float32) -> None:
    get_lookup_tensor_cache(device, dtype)


def _interpolate_lookup_limits(common_position_rad: torch.Tensor, lookup_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if lookup_y.shape != (LOOKUP_TABLE_SIZE, 2):
        raise ValueError(f"lookup table shape must be ({LOOKUP_TABLE_SIZE}, 2), got {tuple(lookup_y.shape)}")
    x_deg = common_position_rad * RAD2DEG
    idx_f = (x_deg - LOOKUP_TABLE_MIN_ANGLE_DEG) / LOOKUP_TABLE_ANGLE_STEP_DEG
    n = LOOKUP_TABLE_SIZE
    ymin = lookup_y[:, 0]
    ymax = lookup_y[:, 1]
    below = idx_f < 0
    above = idx_f >= (n - 1)
    idx_f_i = torch.clamp(idx_f, 0.0, float(n - 1) - 1e-9)
    idx_low = torch.floor(idx_f_i).long().clamp(0, n - 2)
    idx_high = idx_low + 1
    t = idx_f_i - idx_low.float()
    lo_i = ymin[idx_low] * (1.0 - t) + ymin[idx_high] * t
    hi_i = ymax[idx_low] * (1.0 - t) + ymax[idx_high] * t
    lim_lo = torch.where(below, ymin[0], torch.where(above, ymin[-1], lo_i))
    lim_hi = torch.where(below, ymax[0], torch.where(above, ymax[-1], hi_i))
    return lim_lo, lim_hi


def correct_joint_targets_with_lookup_tables(
    target_action: torch.Tensor,
    current_joint_angles: torch.Tensor,
    *,
    dof_limits_scale: float = DEFAULT_DOF_LIMITS_SCALE,
) -> torch.Tensor:
    """Correct only lookup-table constrained differential joints and return new action tensor."""
    out = target_action.clone()
    device = out.device
    dtype = out.dtype
    cache = get_lookup_tensor_cache(device, dtype)

    soft_min = torch.tensor(HAND_SOFT_LIMIT_LOWER, device=device, dtype=dtype) * dof_limits_scale
    soft_max = torch.tensor(HAND_SOFT_LIMIT_UPPER, device=device, dtype=dtype) * dof_limits_scale

    for joint_idx, partner_idx, table_name in _LOOKUP_JOINTS:
        lim_lo, lim_hi = _interpolate_lookup_limits(current_joint_angles[..., partner_idx], cache[table_name])
        lim_lo = torch.maximum(lim_lo, soft_min[joint_idx])
        lim_hi = torch.minimum(lim_hi, soft_max[joint_idx])
        out[..., joint_idx] = torch.maximum(torch.minimum(out[..., joint_idx], lim_hi), lim_lo)

    return out
