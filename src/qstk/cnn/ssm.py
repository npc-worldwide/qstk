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

Everything is functional. No classes.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .layers import ComplexNorm, complex_glorot, complex_randn


# ---------------------------------------------------------------------------
# Parallel scan primitives
# ---------------------------------------------------------------------------

def _sequential_scan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sequential associative scan for short sequences.

    Recurrence: h[t] = a[t] * h[t-1] + b[t]
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
    a_even = a[0::2]
    a_odd = a[1::2]
    b_even = b[0::2]
    b_odd = b[1::2]
    a_combined = a_even * a_odd
    b_combined = a_odd * b_even + b_odd
    h_half = _scan_pow2(a_combined, b_combined)
    h = np.empty_like(b)
    h[1::2] = h_half
    h[0] = b[0]
    if L > 2:
        h[2::2] = a[2::2] * h_half[:-1] + b[2::2]
    return h


def parallel_scan(a: np.ndarray, b: np.ndarray, sequential_threshold: int = 32) -> np.ndarray:
    """Choose sequential or parallel scan based on sequence length."""
    L = a.shape[0]
    if L <= sequential_threshold:
        return _sequential_scan(a, b)
    return _blelloch_scan(a, b)


# ---------------------------------------------------------------------------
# SSM helpers
# ---------------------------------------------------------------------------

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _rms_norm(z: np.ndarray, scale: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mag = np.abs(z)
    rms = np.sqrt(np.mean(mag ** 2, axis=-1, keepdims=True) + eps)
    phase = z / (mag + 1e-8)
    return phase * (mag / rms) * np.abs(scale)


# ---------------------------------------------------------------------------
# Activity Predictor factories
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    'feature_dim': 25,
    'model_dim': 64,
    'state_dim': 128,
    'num_layers': 2,
    'num_classes': 16,
    'dtype': np.complex128,
}

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


def make_predictor(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a fresh activity predictor (functional dict).

    Returns:
        model dict with keys:
            'config': dict of hyperparameters
            'params': flat dict of numpy arrays
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    dim = cfg['model_dim']
    state_dim = cfg['state_dim']
    num_layers = cfg['num_layers']
    num_classes = cfg['num_classes']
    dtype = cfg['dtype']

    params = {}

    # Input projection
    params['W_in'] = complex_glorot(cfg['feature_dim'], dim, dtype=dtype)

    # Classification heads
    params['W_action'] = complex_glorot(dim, num_classes, dtype=dtype)
    params['W_conf'] = complex_glorot(dim, 1, dtype=dtype)
    params['W_domain'] = complex_glorot(dim, 32, dtype=dtype)
    params['W_file_ext'] = complex_glorot(dim, 16, dtype=dtype)
    params['W_command'] = complex_glorot(dim, 16, dtype=dtype)

    # Per-layer SSM params
    for i in range(num_layers):
        params[f'layer_{i}_log_A_real'] = np.linspace(np.log(0.95), np.log(0.999), state_dim).astype(np.float64)
        params[f'layer_{i}_log_A_imag'] = np.linspace(0.001, np.pi, state_dim).astype(np.float64)
        params[f'layer_{i}_dt_proj'] = np.random.randn(dim, state_dim).astype(np.float64) * np.sqrt(2.0 / (dim + state_dim))
        params[f'layer_{i}_dt_bias'] = np.full(state_dim, -4.0, dtype=np.float64)
        params[f'layer_{i}_B'] = complex_glorot(dim, state_dim, dtype=dtype)
        params[f'layer_{i}_C'] = complex_glorot(state_dim, dim, dtype=dtype)
        params[f'layer_{i}_D'] = complex_randn(dim, scale=0.01, dtype=dtype)
        params[f'layer_{i}_norm_scale'] = np.ones(dim, dtype=dtype)
        params[f'layer_{i}_scale'] = np.array(0.1, dtype=np.float64)

    params['output_norm_scale'] = np.ones(dim, dtype=dtype)

    return {'config': cfg, 'params': params}


def _ssm_layer_forward_sequence(
    params: Dict[str, np.ndarray],
    prefix: str,
    x: np.ndarray,
    h0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward a full sequence through one SSM layer.

    Args:
        params: flat param dict.
        prefix: layer key prefix, e.g. 'layer_0_'.
        x: [L, dim] complex.
        h0: [state_dim] complex optional.

    Returns:
        y: [L, dim] complex.
        h_last: [state_dim] complex.
    """
    x_normed = _rms_norm(x, params[f'{prefix}norm_scale'])
    dt = _softplus(x_normed.real @ params[f'{prefix}dt_proj'] + params[f'{prefix}dt_bias'])
    decay = np.exp(params[f'{prefix}log_A_real'])
    freq = params[f'{prefix}log_A_imag']
    A_mag = np.exp(-dt * decay[None, :])
    A_t = A_mag * np.exp(1j * freq[None, :])
    Bx = x_normed @ params[f'{prefix}B'].T
    Bx = Bx * dt
    if h0 is not None:
        first_Bx = A_t[0] * h0 + Bx[0]
        Bx = np.concatenate([first_Bx[None, :], Bx[1:]], axis=0)
    h = parallel_scan(A_t.astype(x.dtype), Bx.astype(x.dtype))
    y = h @ params[f'{prefix}C'].T
    y = y + params[f'{prefix}D'][None, :] * x
    return y, h[-1]


def forward(model: Dict[str, Any], xs: np.ndarray) -> Dict[str, np.ndarray]:
    """Forward a batch of sequences.

    Args:
        model: predictor dict from make_predictor().
        xs: Real input features, shape [B, L, feature_dim].

    Returns:
        Dict with action_logits, confidence, domain, file_ext, command.
    """
    cfg = model['config']
    params = model['params']
    dim = cfg['model_dim']
    num_layers = cfg['num_layers']
    dtype = cfg['dtype']

    B, L, _ = xs.shape

    # Project real features to complex
    z = xs.astype(dtype) @ params['W_in'].T  # [B, L, dim] complex

    # Stacked SSM
    for i in range(num_layers):
        prefix = f'layer_{i}_'
        scale = float(params[f'{prefix}scale'])
        outputs = []
        states = [np.zeros(cfg['state_dim'], dtype=dtype) for _ in range(B)]
        for b in range(B):
            out_b, h_b = _ssm_layer_forward_sequence(params, prefix, z[b], states[b])
            outputs.append(out_b)
            states[b] = h_b
        z = z + np.stack(outputs, axis=0) * scale

    # Output norm
    z = _rms_norm(z, params['output_norm_scale'])

    # Classification heads on last timestep
    last = z[:, -1, :]  # [B, dim] complex

    action_logits = np.abs(last @ params['W_action'].T)
    confidence = 1.0 / (1.0 + np.exp(-np.abs(last @ params['W_conf'].T)))
    domain = np.abs(last @ params['W_domain'].T)
    file_ext = np.abs(last @ params['W_file_ext'].T)
    command = np.abs(last @ params['W_command'].T)

    return {
        'action_logits': action_logits,
        'confidence': confidence,
        'domain': domain,
        'file_ext': file_ext,
        'command': command,
    }


def predict(model: Dict[str, Any], xs: np.ndarray) -> Dict[str, Any]:
    """Predict next action from a single sequence.

    Args:
        model: predictor dict.
        xs: Real features, shape [L, feature_dim].
    """
    out = forward(model, xs[None, ...])
    logits = out['action_logits'][0]
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


def save(model: Dict[str, Any], path: str):
    """Save model as a single npz. Config scalars stored as __config_* arrays.

    No pickle, no json, no classes.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cfg = model['config']
    arrays = dict(model['params'])
    for k, v in cfg.items():
        if isinstance(v, np.dtype):
            arrays[f'__config_{k}'] = np.array(v.name, dtype='U32')
        elif isinstance(v, type) and issubclass(v, np.generic):
            arrays[f'__config_{k}'] = np.array(np.dtype(v).name, dtype='U32')
        else:
            arrays[f'__config_{k}'] = np.array(v)
    np.savez_compressed(path, **arrays)


def load(path: str) -> Dict[str, Any]:
    """Load model from npz.

    Args:
        path: path to .npz file.
    """
    data = np.load(path)
    params = {}
    config = {}
    for key in data.files:
        if key.startswith('__config_'):
            k = key[len('__config_'):]
            raw = data[key]
            if raw.dtype.kind == 'U':
                s = str(raw.item(0))
                if 'complex' in s:
                    config[k] = np.dtype(s)
                else:
                    config[k] = s
            else:
                config[k] = int(raw) if raw.dtype.kind in 'iu' else float(raw)
        else:
            params[key] = data[key]
    data.close()

    # Ensure dtype is restored
    if 'dtype' in config and isinstance(config['dtype'], str):
        config['dtype'] = np.dtype(config['dtype'])

    # Rebuild model dict
    model = make_predictor(config)
    for k in model['params']:
        if k in params:
            model['params'][k] = params[k].astype(model['params'][k].dtype)
    return model


def shift_base_weights(model: Dict[str, Any], shift_scale: float = 0.01):
    """Perturb all parameters by small Gaussian noise.

    Used for continual learning: shift base weights before fine-tuning
    on new data, preventing catastrophic forgetting while allowing adaptation.
    """
    for k, v in model['params'].items():
        if v.shape == ():
            noise = np.array(0.0, dtype=v.dtype) if v.dtype.kind not in 'c' else np.array(0.0 + 0.0j, dtype=v.dtype)
            noise = noise + np.random.randn() * shift_scale
            if v.dtype.kind in 'c':
                noise = noise + 1j * np.random.randn() * shift_scale
            model['params'][k] = v + noise.astype(v.dtype)
        else:
            noise = complex_randn(*v.shape, scale=shift_scale, dtype=v.dtype)
            model['params'][k] = v + noise


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def _idx_to_action(idx: int) -> str:
    if 0 <= idx < len(ACTIVITY_TYPES):
        return ACTIVITY_TYPES[idx]
    return 'unknown'
