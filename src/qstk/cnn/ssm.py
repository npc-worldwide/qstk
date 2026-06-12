"""
Complex Selective State Space Models (SSM) for qstk.cnn.

Diagonally-structured complex state space with input-dependent decay.
All operations use native numpy complex128. Parallel scan (Blelloch)
gives O(log L) sequential depth for long sequences.

Core equation:
    h[t] = A_t * h[t-1] + B_t * x[t]
    y[t] = C @ h[t] + D * x[t]

where A_t = exp(-dt * decay) * exp(i * freq) is input-dependent,
giving content-selective memory length. Multiple timescales via
frequency bands in the diagonal A matrix.

Classes:
    ComplexSSMLayer     -- single selective SSM layer
    ComplexSSM        -- stacked SSM with residual connections
    ActivityPredictor -- SSM-based next-action classifier
"""

import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .layers import ComplexLinear, ComplexNorm, mod_relu, complex_glorot, complex_randn


# ---------------------------------------------------------------------------
# Parallel scan primitives
# ---------------------------------------------------------------------------

def _sequential_scan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sequential associative scan for short sequences.

    Recurrence: h[t] = a[t] * h[t-1] + b[t]
    Both a and b are complex arrays.

    Args:
        a: [L, state_dim] complex — diagonal A_t coefficients.
        b: [L, state_dim] complex — input terms B_t @ x_t.

    Returns:
        h: [L, state_dim] complex — hidden states at each timestep.
    """
    L = a.shape[0]
    h = np.empty_like(b)
    h[0] = b[0]
    for t in range(1, L):
        h[t] = a[t] * h[t - 1] + b[t]
    return h


def _blelloch_scan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Blelloch parallel prefix scan. O(log L) depth.

    Associative operator: (a1, b1) ⊕ (a2, b2) = (a1*a2, a2*b1 + b2)
    """
    L = a.shape[0]

    # Pad to power of 2
    next_pow2 = 1 << (L - 1).bit_length()
    if next_pow2 != L:
        pad = next_pow2 - L
        a_pad = np.ones((pad, a.shape[1]), dtype=a.dtype)
        b_pad = np.zeros((pad, b.shape[1]), dtype=b.dtype)
        a = np.concatenate([a, a_pad], axis=0)
        b = np.concatenate([b, b_pad], axis=0)
    else:
        pad = 0

    h = _scan_pow2(a, b)
    return h[:L] if pad else h


def _scan_pow2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Parallel scan on power-of-2 length."""
    L = a.shape[0]
    if L == 1:
        return b

    # Up-sweep: combine adjacent pairs
    a_even = a[0::2]
    a_odd = a[1::2]
    b_even = b[0::2]
    b_odd = b[1::2]

    a_combined = a_even * a_odd
    b_combined = a_odd * b_even + b_odd

    h_half = _scan_pow2(a_combined, b_combined)

    # Down-sweep: interleave
    h = np.empty_like(b)
    h[1::2] = h_half
    h[0] = b[0]
    if L > 2:
        h[2::2] = a[2::2] * h_half[:-1] + b[2::2]

    return h


def parallel_scan(a: np.ndarray, b: np.ndarray, sequential_threshold: int = 32) -> np.ndarray:
    """Choose sequential or parallel scan based on sequence length.

    Args:
        a: [L, state_dim] complex — A_t coefficients.
        b: [L, state_dim] complex — B_t @ x_t terms.
        sequential_threshold: Use sequential scan if L <= this.

    Returns:
        h: [L, state_dim] complex — hidden states.
    """
    L = a.shape[0]
    if L <= sequential_threshold:
        return _sequential_scan(a, b)
    return _blelloch_scan(a, b)


# ---------------------------------------------------------------------------
# ComplexSSMLayer
# ---------------------------------------------------------------------------

class ComplexSSMLayer:
    """Single complex selective SSM layer.

    Complex eigenvalues A = exp(log_decay + i*frequency) give:
    - |A| = exp(log_decay) < 1: exponential decay (memory length)
    - arg(A) = frequency: oscillation (timescale selectivity)

    Input-dependent selectivity: dt is modulated by the input,
    allowing content-dependent gating of what to remember/forget.

    Args:
        dim:        Model / input dimension (complex).
        state_dim:  SSM hidden state dimension (complex).
        dtype:      np.complex128 or np.complex64.

    Attributes:
        params: Dict with keys:
            log_A_real, log_A_imag, B, C, D, dt_proj, dt_bias, norm_scale
    """

    def __init__(
        self,
        dim: int,
        state_dim: int,
        dtype: np.dtype = np.complex128,
    ):
        self.dim = dim
        self.state_dim = state_dim
        self.dtype = dtype

        # Eigenvalues: decay rates and frequencies (diagonal A)
        log_decay = np.linspace(np.log(0.95), np.log(0.999), state_dim).astype(np.float64)
        freq = np.linspace(0.001, np.pi, state_dim).astype(np.float64)

        # Input-dependent modulation: x -> dt (decay modifier)
        dt_proj = np.random.randn(dim, state_dim).astype(np.float64) * np.sqrt(2.0 / (dim + state_dim))
        dt_bias = np.full(state_dim, -4.0, dtype=np.float64)

        # B: input -> state (complex)
        B = complex_glorot(dim, state_dim, dtype=dtype)

        # C: state -> output (complex)
        C = complex_glorot(state_dim, dim, dtype=dtype)

        # D: skip connection (complex)
        D = complex_randn(dim, scale=0.01, dtype=dtype)

        self.params: Dict[str, np.ndarray] = {
            'log_A_real': log_decay,
            'log_A_imag': freq,
            'B': B,
            'C': C,
            'D': D,
            'dt_proj': dt_proj,
            'dt_bias': dt_bias,
            'norm_scale': np.ones(dim, dtype=dtype),
        }

    def _softplus(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def _rms_norm(self, z: np.ndarray) -> np.ndarray:
        mag = np.abs(z)
        rms = np.sqrt(np.mean(mag ** 2, axis=-1, keepdims=True) + 1e-6)
        phase = z / (mag + 1e-8)
        return phase * (mag / rms) * np.abs(self.params['norm_scale'])

    def step(
        self,
        x: np.ndarray,           # [dim] complex
        h_prev: np.ndarray,      # [state_dim] complex
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single timestep forward.

        Args:
            x:      Input vector, shape [dim] complex.
            h_prev: Previous hidden state, shape [state_dim] complex.

        Returns:
            y:      Output vector, shape [dim] complex.
            h:      New hidden state, shape [state_dim] complex.
        """
        # RMS norm
        x_normed = self._rms_norm(x)

        # Input-dependent dt (selectivity)
        dt = self._softplus(x_normed.real @ self.params['dt_proj'] + self.params['dt_bias'])

        # Discretized A_t = exp(-dt * decay) * exp(i * freq)
        decay = np.exp(self.params['log_A_real'])  # [state_dim]
        freq = self.params['log_A_imag']              # [state_dim]
        A_mag = np.exp(-dt * decay)                    # [state_dim]
        A_t = A_mag * np.exp(1j * freq)                # [state_dim] complex

        # B @ x: input projection to state space
        Bx = x_normed @ self.params['B'].T            # [state_dim] complex
        Bx = Bx * dt                                   # discretization

        # Recurrence
        h = A_t * h_prev + Bx                          # [state_dim] complex

        # Output: y = C @ h + D * x
        y = h @ self.params['C'].T                     # [dim] complex
        y = y + self.params['D'] * x                   # skip

        return y, h

    def forward_sequence(
        self,
        x: np.ndarray,           # [L, dim] complex
        h0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward a full sequence.

        Args:
            x:  Input sequence, shape [L, dim] complex.
            h0: Initial hidden state, shape [state_dim] complex.

        Returns:
            y:  Output sequence, shape [L, dim] complex.
            h:  Final hidden state, shape [state_dim] complex.
        """
        L = x.shape[0]
        device_dtype = x.dtype

        # RMS norm over sequence
        x_normed = self._rms_norm(x)  # [L, dim]

        # Input-dependent dt
        dt = self._softplus(x_normed.real @ self.params['dt_proj'] + self.params['dt_bias'])  # [L, state_dim]

        # Discretized A_t
        decay = np.exp(self.params['log_A_real'])  # [state_dim]
        freq = self.params['log_A_imag']              # [state_dim]
        A_mag = np.exp(-dt * decay[None, :])         # [L, state_dim]
        A_t = A_mag * np.exp(1j * freq[None, :])      # [L, state_dim] complex

        # B @ x
        Bx = x_normed @ self.params['B'].T            # [L, state_dim] complex
        Bx = Bx * dt[:, :, None] if Bx.ndim > 2 else Bx * dt

        # Prepend h0 if provided
        if h0 is not None:
            first_Bx = A_t[0] * h0 + Bx[0]
            Bx = np.concatenate([first_Bx[None, :], Bx[1:]], axis=0)

        # Parallel scan
        h = parallel_scan(A_t.astype(device_dtype), Bx.astype(device_dtype))

        # Output
        y = h @ self.params['C'].T                     # [L, dim] complex
        y = y + self.params['D'][None, :] * x          # skip

        return y, h[-1]


# ---------------------------------------------------------------------------
# ComplexSSM (stacked)
# ---------------------------------------------------------------------------

class ComplexSSM:
    """Stacked complex selective SSM.

    N layers of ComplexSSMLayer with residual connections and ComplexNorm.
    Each layer maintains its own hidden state for streaming inference.

    Args:
        dim:        Model dimension (complex).
        state_dim:  SSM state dimension per layer.
        num_layers: Number of stacked layers.
        dtype:      np.complex128 or np.complex64.

    Attributes:
        layers:      List of ComplexSSMLayer instances.
        layer_scales: Residual scaling factors (learnable, small init).
    """

    def __init__(
        self,
        dim: int = 64,
        state_dim: int = 128,
        num_layers: int = 2,
        dtype: np.dtype = np.complex128,
    ):
        self.dim = dim
        self.state_dim = state_dim
        self.num_layers = num_layers

        self.layers = [ComplexSSMLayer(dim, state_dim, dtype=dtype) for _ in range(num_layers)]
        self.layer_scales = [np.array(0.1, dtype=np.float64) for _ in range(num_layers)]
        self.output_norm = ComplexNorm(dim, dtype=dtype)

    def init_state(self) -> List[np.ndarray]:
        """Initialize hidden states for all layers.

        Returns:
            List of [state_dim] complex zero arrays, one per layer.
        """
        return [np.zeros(self.state_dim, dtype=self.layers[0].dtype) for _ in range(self.num_layers)]

    def step(
        self,
        x: np.ndarray,                # [dim] complex
        states: List[np.ndarray],     # list of [state_dim] complex
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Single timestep streaming forward.

        Args:
            x:      Input vector, shape [dim] complex.
            states: List of hidden states, one per layer.

        Returns:
            y:      Output vector, shape [dim] complex.
            new_states: Updated hidden states.
        """
        h = x
        new_states = []
        for layer, scale in zip(self.layers, self.layer_scales):
            residual = h
            y, h_new = layer.step(h, states.pop(0) if states else np.zeros(self.state_dim, dtype=layer.dtype))
            h = residual + y * float(scale)
            new_states.append(h_new)

        return self.output_norm(h), new_states

    def forward_sequence(
        self,
        x: np.ndarray,                # [L, dim] complex
        states: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward a full sequence.

        Args:
            x:      Input sequence, shape [L, dim] complex.
            states: Optional initial hidden states.

        Returns:
            y:      Output sequence, shape [L, dim] complex.
            states: Final hidden states.
        """
        if states is None:
            states = self.init_state()

        h = x
        new_states = []
        for layer, scale in zip(self.layers, self.layer_scales):
            residual = h
            h_out, h_final = layer.forward_sequence(h, states.pop(0) if states else None)
            h = residual + h_out * float(scale)
            new_states.append(h_final)

        return self.output_norm(h), new_states


# ---------------------------------------------------------------------------
# Activity Predictor (discrete classification over activity types)
# ---------------------------------------------------------------------------

@dataclass
class ActivityPredictorConfig:
    """Configuration for ActivityPredictor."""
    feature_dim: int = 24          # Real-valued input features
    model_dim: int = 64            # Complex model dimension (must be even)
    state_dim: int = 128           # SSM state dimension
    num_layers: int = 2            # SSM layers
    num_classes: int = 16          # Activity type classes
    dtype: np.dtype = np.complex128


class ActivityPredictor:
    """SSM-based next-action predictor.

    Architecture:
        1. Real input features -> complex embedding via learned projection
        2. ComplexSSM processes the sequence
        3. Classification head on the final timestep
        4. Confidence head (sigmoid) for prediction certainty
        5. Parameter heads: domain, file_ext, command_verb embeddings

    The real-to-complex projection is a learnable complex linear layer.
    Magnitude encodes salience, phase encodes temporal position.

    Args:
        config: ActivityPredictorConfig instance.

    Attributes:
        params: Flat dict of all parameters (for save/load/gradients).
    """

    def __init__(self, config: Optional[ActivityPredictorConfig] = None):
        self.config = config or ActivityPredictorConfig()
        cfg = self.config

        # Real -> Complex projection (learnable)
        self.W_in = complex_glorot(cfg.feature_dim, cfg.model_dim, dtype=cfg.dtype)

        # SSM backbone
        self.ssm = ComplexSSM(
            dim=cfg.model_dim,
            state_dim=cfg.state_dim,
            num_layers=cfg.num_layers,
            dtype=cfg.dtype,
        )

        # Classification heads (complex -> real)
        self.W_action = complex_glorot(cfg.model_dim, cfg.num_classes, dtype=cfg.dtype)
        self.W_conf = complex_glorot(cfg.model_dim, 1, dtype=cfg.dtype)

        # Parameter prediction heads (lightweight embeddings)
        self.W_domain = complex_glorot(cfg.model_dim, 32, dtype=cfg.dtype)
        self.W_file_ext = complex_glorot(cfg.model_dim, 16, dtype=cfg.dtype)
        self.W_command = complex_glorot(cfg.model_dim, 16, dtype=cfg.dtype)

        self._build_param_dict()

    def _build_param_dict(self):
        """Flatten all parameters into a single dict for optimization."""
        self.params: Dict[str, np.ndarray] = {
            'W_in': self.W_in,
            'W_action': self.W_action,
            'W_conf': self.W_conf,
            'W_domain': self.W_domain,
            'W_file_ext': self.W_file_ext,
            'W_command': self.W_command,
        }
        for i, layer in enumerate(self.ssm.layers):
            for k, v in layer.params.items():
                self.params[f'layer_{i}_{k}'] = v
            self.params[f'layer_{i}_scale'] = np.array(self.ssm.layer_scales[i])
        self.params['output_norm_scale'] = self.ssm.output_norm.params['scale']

    def _complex_linear(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """x @ W.T for complex arrays."""
        return x @ W.T

    def forward(self, xs: np.ndarray) -> Dict[str, np.ndarray]:
        """Forward a batch of sequences.

        Args:
            xs: Real input features, shape [B, L, feature_dim].

        Returns:
            Dict with keys:
                action_logits: [B, L, num_classes] real
                confidence:    [B, L, 1] real (sigmoid)
                domain:        [B, L, 32] real (magnitude)
                file_ext:      [B, L, 16] real
                command:       [B, L, 16] real
        """
        B, L, D = xs.shape

        # Project real features to complex space
        z = xs.astype(self.config.dtype) @ self.W_in.T  # [B, L, model_dim] complex

        # Process through SSM
        outputs = []
        states = self.ssm.init_state()
        for b in range(B):
            out, states_b = self.ssm.forward_sequence(z[b], states)
            outputs.append(out)
            states = states_b  # carry forward for streaming feel

        z_out = np.stack(outputs, axis=0)  # [B, L, model_dim] complex

        # Classification heads on last timestep
        last = z_out[:, -1, :]  # [B, model_dim] complex

        action_logits = np.abs(last @ self.W_action.T)  # [B, num_classes] real
        confidence = 1.0 / (1.0 + np.exp(-np.abs(last @ self.W_conf.T)))  # [B, 1] sigmoid

        domain = np.abs(last @ self.W_domain.T)       # [B, 32]
        file_ext = np.abs(last @ self.W_file_ext.T)   # [B, 16]
        command = np.abs(last @ self.W_command.T)     # [B, 16]

        return {
            'action_logits': action_logits,
            'confidence': confidence,
            'domain': domain,
            'file_ext': file_ext,
            'command': command,
        }

    def predict(self, xs: np.ndarray) -> Dict[str, Any]:
        """Predict next action from a single sequence.

        Args:
            xs: Real features, shape [L, feature_dim].

        Returns:
            Dict with predicted_action, confidence, top_3, param embeddings.
        """
        out = self.forward(xs[None, ...])  # add batch dim
        logits = out['action_logits'][0]   # [num_classes]
        probs = _softmax(logits)

        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [{'action': _idx_to_action(int(i)), 'probability': float(probs[i])} for i in top3_idx]

        return {
            'predicted_action': _idx_to_action(int(np.argmax(probs))),
            'predicted_index': int(np.argmax(probs)),
            'confidence': float(out['confidence'][0, 0]),
            'top_3': top3,
            'params': {
                'domain': out['domain'][0].tolist(),
                'file_ext': out['file_ext'][0].tolist(),
                'command': out['command'][0].tolist(),
            }
        }

    def save(self, path: str):
        """Save model parameters to disk."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'config': self.config, 'params': self.params}, f)

    @classmethod
    def load(cls, path: str) -> 'ActivityPredictor':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(data['config'])
        for k, v in data['params'].items():
            if k in model.params:
                model.params[k] = v
        return model

    def shift_base_weights(self, shift_scale: float = 0.01):
        """Perturb all parameters by small Gaussian noise.

        Used for continual learning / transfer: shift the base weights
        slightly before fine-tuning on new data, preventing catastrophic
        forgetting while allowing adaptation.

        Args:
            shift_scale: Standard deviation of the perturbation.
        """
        for k, v in self.params.items():
            noise = complex_randn(*v.shape, scale=shift_scale, dtype=v.dtype)
            self.params[k] = v + noise


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax over last axis."""
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


ACTIVITY_TYPES = [
    'pane_open', 'pane_close', 'pane_focus',
    'file_open', 'file_edit',
    'website_visit',
    'terminal_command',
    'chat_message',
    'app_switch',
    'search_query',
    'model_change',
    'click',
    'keyboard_shortcut',
    'text_input',
    'jinx_execution',
    'memory_created',
]

ACTION_TO_IDX = {t: i for i, t in enumerate(ACTIVITY_TYPES)}

def _idx_to_action(idx: int) -> str:
    if 0 <= idx < len(ACTIVITY_TYPES):
        return ACTIVITY_TYPES[idx]
    return 'unknown'
