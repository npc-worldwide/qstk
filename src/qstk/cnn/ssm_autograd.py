"""
Hybrid autograd for complex-valued SSM ActivityPredictor.

Follows the same pattern as qstk.cnn.autograd.CharPAM:
  - Storage and inference: pure numpy complex128 (inherited from ActivityPredictor).
  - Training: torch autograd for exact Wirtinger gradients through every
    complex operation (RMS norm, selective dt, recurrence, scan, readout).

Classes:
    ActivityPredictorAutograd -- adds forward_backward() to ActivityPredictor
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .ssm import ActivityPredictor, ActivityPredictorConfig, parallel_scan, _softmax, ACTIVITY_TYPES


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
# Autograd-capable ActivityPredictor
# ---------------------------------------------------------------------------

class ActivityPredictorAutograd(ActivityPredictor):
    """SSM-based next-action predictor with torch autograd training.

    Inherits forward() / predict() / save() / load() / shift_base_weights()
    from the numpy ActivityPredictor.  Adds forward_backward() for exact
    gradient computation via PyTorch.

    The torch forward mirrors the numpy one exactly, including the
    input-dependent dt modulation and sequential scan over the sequence.
    Sequences are short (<=50) so sequential scan is fine for training.
    """

    def __init__(self, config: Optional[ActivityPredictorConfig] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # Torch-backed forward + backward
    # ------------------------------------------------------------------

    def forward_backward(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Forward a batch of sequences, compute cross-entropy loss,
        and return exact gradients via torch autograd.

        Args:
            xs: Real input features, shape [B, L, feature_dim].
            ys: Integer target labels, shape [B].

        Returns:
            (loss, grads) where grads maps parameter names -> numpy arrays.
        """
        batch_size, L, D = xs.shape
        cfg = self.config

        # ---- Convert all numpy params to torch tensors with grad ----
        torch_params: Dict[str, torch.Tensor] = {}
        for name, arr in self.params.items():
            torch_params[name] = torch.from_numpy(arr.copy()).requires_grad_(True)

        # Real inputs as torch (no grad needed)
        t_xs = torch.from_numpy(xs.copy())  # [B, L, D] real

        # ---- Real -> Complex projection ----
        W_in = torch_params['W_in']  # [model_dim, feature_dim] complex
        # t_xs is real, we cast to complex then matmul
        z = t_xs.to(torch.complex128) @ W_in.T  # [B, L, model_dim] complex

        # ---- SSM layers (sequential scan, torch) ----
        for i in range(cfg.num_layers):
            residual = z
            scale = torch_params[f'layer_{i}_scale']  # scalar real

            # RMS norm
            norm_scale = torch_params[f'layer_{i}_norm_scale']
            z_normed = _torch_rms_norm(z, norm_scale)  # [B, L, dim] complex

            # Input-dependent dt
            dt_proj = torch_params[f'layer_{i}_dt_proj']  # [dim, state_dim] real
            dt_bias = torch_params[f'layer_{i}_dt_bias']    # [state_dim] real
            dt = _torch_softplus(z_normed.real @ dt_proj + dt_bias)  # [B, L, state_dim] real

            # Discretized A_t = exp(-dt * decay) * exp(i * freq)
            decay = torch.exp(torch_params[f'layer_{i}_log_A_real'])  # [state_dim] real
            freq = torch_params[f'layer_{i}_log_A_imag']               # [state_dim] real
            A_mag = torch.exp(-dt * decay[None, None, :])            # [B, L, state_dim] real
            A_t = A_mag * torch.exp(1j * freq[None, None, :])        # [B, L, state_dim] complex

            # B @ x
            B_param = torch_params[f'layer_{i}_B']  # [state_dim, dim] complex
            Bx = z_normed @ B_param.T               # [B, L, state_dim] complex
            Bx = Bx * dt                       # [B, L, state_dim] complex

            # Sequential scan per sample in batch
            h_seq = []
            for b in range(batch_size):
                h = _torch_sequential_scan(A_t[b], Bx[b])  # [L, state_dim] complex
                h_seq.append(h)
            h = torch.stack(h_seq, dim=0)  # [B, L, state_dim] complex

            # Output: y = C @ h + D * x
            C = torch_params[f'layer_{i}_C']  # [dim, state_dim] complex
            D = torch_params[f'layer_{i}_D']  # [dim] complex
            y = h @ C.T                       # [B, L, dim] complex
            y = y + D[None, None, :] * z       # skip

            # Residual
            z = residual + y * scale.real.item()

        # Output norm
        out_norm_scale = torch_params['output_norm_scale']
        z = _torch_rms_norm(z, out_norm_scale)

        # ---- Classification heads on last timestep ----
        last = z[:, -1, :]  # [B, model_dim] complex

        W_action = torch_params['W_action']
        action_logits = (last @ W_action.T).abs()  # [B, num_classes] real

        # Cross-entropy loss (stable)
        loss = F.cross_entropy(action_logits, torch.from_numpy(ys).long())

        # ---- Backward ----
        loss.backward()

        # ---- Extract gradients back to numpy ----
        grads: Dict[str, np.ndarray] = {}
        for name, tensor in torch_params.items():
            if tensor.grad is not None:
                grads[name] = tensor.grad.detach().numpy().copy()
            else:
                grads[name] = np.zeros_like(self.params[name])

        return float(loss.item()), grads

    def forward_torch(self, xs: np.ndarray) -> Dict[str, torch.Tensor]:
        """Torch forward pass (no grad) for validation / diagnostics.

        Args:
            xs: [B, L, feature_dim] real.

        Returns:
            Dict with torch tensors (action_logits, confidence, domain, file_ext, command).
        """
        batch_size, L, D = xs.shape
        cfg = self.config

        torch_params: Dict[str, torch.Tensor] = {}
        for name, arr in self.params.items():
            torch_params[name] = torch.from_numpy(arr.copy())

        t_xs = torch.from_numpy(xs.copy())
        z = t_xs.to(torch.complex128) @ torch_params['W_in'].T

        for i in range(cfg.num_layers):
            residual = z
            scale = torch_params[f'layer_{i}_scale']
            norm_scale = torch_params[f'layer_{i}_norm_scale']
            z_normed = _torch_rms_norm(z, norm_scale)
            dt_proj = torch_params[f'layer_{i}_dt_proj']
            dt_bias = torch_params[f'layer_{i}_dt_bias']
            dt = _torch_softplus(z_normed.real @ dt_proj + dt_bias)
            decay = torch.exp(torch_params[f'layer_{i}_log_A_real'])
            freq = torch_params[f'layer_{i}_log_A_imag']
            A_mag = torch.exp(-dt * decay[None, None, :])
            A_t = A_mag * torch.exp(1j * freq[None, None, :])
            B_param = torch_params[f'layer_{i}_B']
            Bx = z_normed @ B_param.T
            Bx = Bx * dt
            h_seq = []
            for b in range(batch_size):
                h_seq.append(_torch_sequential_scan(A_t[b], Bx[b]))
            h = torch.stack(h_seq, dim=0)
            C = torch_params[f'layer_{i}_C']
            D = torch_params[f'layer_{i}_D']
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

        action_logits = (last @ W_action.T).abs()
        confidence = torch.sigmoid((last @ W_conf.T).abs())
        domain = (last @ W_domain.T).abs()
        file_ext = (last @ W_file_ext.T).abs()
        command = (last @ W_command.T).abs()

        return {
            'action_logits': action_logits,
            'confidence': confidence,
            'domain': domain,
            'file_ext': file_ext,
            'command': command,
        }
