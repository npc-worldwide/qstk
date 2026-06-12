"""
Hybrid autograd for complex-valued SSM ActivityPredictor.

Everything is functional. No classes.

Follows the same pattern as qstk.cnn.autograd.CharPAM:
  - Storage and inference: pure numpy complex128 (inherited from ActivityPredictor).
  - Training: torch autograd for exact Wirtinger gradients through every
    complex operation (RMS norm, selective dt, recurrence, scan, readout).

Functions:
    forward_backward(model, xs, ys) -> (loss, grads)
    forward_torch(model, xs) -> dict of torch tensors
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Torch helpers mirroring ssm.py numpy ops
# ---------------------------------------------------------------------------

def _torch_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.maximum(x, torch.zeros_like(x))


def _torch_rms_norm(z: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mag = z.abs()
    rms = torch.sqrt(torch.mean(mag ** 2, dim=-1, keepdim=True) + eps)
    phase = z / (mag + 1e-8)
    return phase * (mag / rms) * scale.abs()


def _torch_sequential_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sequential scan in torch (autograd-friendly, no in-place ops)."""
    L = a.shape[0]
    h_list = [b[0]]
    for t in range(1, L):
        h_t = a[t] * h_list[t - 1] + b[t]
        h_list.append(h_t)
    return torch.stack(h_list, dim=0)


# ---------------------------------------------------------------------------
# Autograd-capable forward + backward
# ---------------------------------------------------------------------------

def forward_backward(
    model: Dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Forward a batch of sequences, compute cross-entropy loss,
    and return exact gradients via torch autograd.

    Args:
        model: predictor dict with 'config' and 'params'.
        xs: Real input features, shape [B, L, feature_dim].
        ys: Integer target labels, shape [B].

    Returns:
        (loss, grads) where grads maps parameter names -> numpy arrays.
    """
    batch_size, L, D = xs.shape
    cfg = model['config']
    num_layers = cfg['num_layers']

    # ---- Convert all numpy params to torch tensors with grad ----
    torch_params: Dict[str, torch.Tensor] = {}
    for name, arr in model['params'].items():
        torch_params[name] = torch.from_numpy(arr.copy()).requires_grad_(True)

    t_xs = torch.from_numpy(xs.copy())
    z = t_xs.to(torch.complex128) @ torch_params['W_in'].T

    for i in range(num_layers):
        residual = z
        prefix = f'layer_{i}_'
        scale = torch_params[f'{prefix}scale']
        norm_scale = torch_params[f'{prefix}norm_scale']
        z_normed = _torch_rms_norm(z, norm_scale)
        dt_proj = torch_params[f'{prefix}dt_proj']
        dt_bias = torch_params[f'{prefix}dt_bias']
        dt = _torch_softplus(z_normed.real @ dt_proj + dt_bias)
        decay = torch.exp(torch_params[f'{prefix}log_A_real'])
        freq = torch_params[f'{prefix}log_A_imag']
        A_mag = torch.exp(-dt * decay[None, None, :])
        A_t = A_mag * torch.exp(1j * freq[None, None, :])
        B_param = torch_params[f'{prefix}B']
        Bx = z_normed @ B_param.T
        Bx = Bx * dt
        h_seq = []
        for b in range(batch_size):
            h_seq.append(_torch_sequential_scan(A_t[b], Bx[b]))
        h = torch.stack(h_seq, dim=0)
        C = torch_params[f'{prefix}C']
        D = torch_params[f'{prefix}D']
        y = h @ C.T + D[None, None, :] * z
        z = residual + y * scale.real.item()

    out_norm_scale = torch_params['output_norm_scale']
    z = _torch_rms_norm(z, out_norm_scale)
    last = z[:, -1, :]

    W_action = torch_params['W_action']
    action_logits = (last @ W_action.T).abs()

    loss = F.cross_entropy(action_logits, torch.from_numpy(ys).long())
    loss.backward()

    grads: Dict[str, np.ndarray] = {}
    for name, tensor in torch_params.items():
        if tensor.grad is not None:
            grads[name] = tensor.grad.detach().numpy().copy()
        else:
            grads[name] = np.zeros_like(model['params'][name])

    return float(loss.item()), grads


def forward_torch(
    model: Dict[str, Any],
    xs: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """Torch forward pass (no grad) for validation / diagnostics."""
    batch_size, L, D = xs.shape
    cfg = model['config']
    num_layers = cfg['num_layers']

    torch_params: Dict[str, torch.Tensor] = {}
    for name, arr in model['params'].items():
        torch_params[name] = torch.from_numpy(arr.copy())

    t_xs = torch.from_numpy(xs.copy())
    z = t_xs.to(torch.complex128) @ torch_params['W_in'].T

    for i in range(num_layers):
        residual = z
        prefix = f'layer_{i}_'
        scale = torch_params[f'{prefix}scale']
        norm_scale = torch_params[f'{prefix}norm_scale']
        z_normed = _torch_rms_norm(z, norm_scale)
        dt_proj = torch_params[f'{prefix}dt_proj']
        dt_bias = torch_params[f'{prefix}dt_bias']
        dt = _torch_softplus(z_normed.real @ dt_proj + dt_bias)
        decay = torch.exp(torch_params[f'{prefix}log_A_real'])
        freq = torch_params[f'{prefix}log_A_imag']
        A_mag = torch.exp(-dt * decay[None, None, :])
        A_t = A_mag * torch.exp(1j * freq[None, None, :])
        B_param = torch_params[f'{prefix}B']
        Bx = z_normed @ B_param.T
        Bx = Bx * dt
        h_seq = []
        for b in range(batch_size):
            h_seq.append(_torch_sequential_scan(A_t[b], Bx[b]))
        h = torch.stack(h_seq, dim=0)
        C = torch_params[f'{prefix}C']
        D = torch_params[f'{prefix}D']
        y = h @ C.T + D[None, None, :] * z
        z = residual + y * scale.real.item()

    out_norm_scale = torch_params['output_norm_scale']
    z = _torch_rms_norm(z, out_norm_scale)
    last = z[:, -1, :]

    W_action = torch_params['W_action']
    W_conf = torch_params['W_conf']
    W_domain = torch_params['W_domain']
    W_file_ext = torch_params['W_file_ext']
    W_command = torch_params['W_command']

    return {
        'action_logits': (last @ W_action.T).abs(),
        'confidence': torch.sigmoid((last @ W_conf.T).abs()),
        'domain': (last @ W_domain.T).abs(),
        'file_ext': (last @ W_file_ext.T).abs(),
        'command': (last @ W_command.T).abs(),
    }
