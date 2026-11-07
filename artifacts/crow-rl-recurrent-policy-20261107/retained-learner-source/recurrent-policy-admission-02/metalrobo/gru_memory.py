"""Differentiable learner algebra for the native reset-after r/z/n GRU.

This is a batch learner operator, never a simulator or rollout scheduler. A
recurrent policy/sequence PPO integration is still required before use in crow
training. Parameters use the native row-major [3*H,I/H] layout with separate
input and recurrent biases; reset multiplies the pre-cell hidden state by zero.
"""
from __future__ import annotations

import mlx.core as mx

PARAMETER_NAMES = ("input_weights", "recurrent_weights", "input_bias", "recurrent_bias")


def _validate(inputs: mx.array, hidden: mx.array, parameters: dict[str, mx.array]) -> int:
    if inputs.ndim != 2 or hidden.ndim != 2 or inputs.shape[0] != hidden.shape[0]:
        raise ValueError("GRU inputs/hidden require matching batch dimensions")
    if set(parameters) != set(PARAMETER_NAMES):
        raise ValueError("GRU requires four native-layout parameter tensors")
    width = hidden.shape[1]
    if not width or not inputs.shape[1] or not inputs.shape[0]:
        raise ValueError("GRU dimensions must be positive")
    shapes = ((3*width, inputs.shape[1]), (3*width, width), (3*width,), (3*width,))
    if any(parameters[name].shape != shape for name, shape in zip(PARAMETER_NAMES, shapes)):
        raise ValueError("GRU tensor shape mismatch")
    if any(array.dtype != mx.float32 for array in (inputs, hidden, *parameters.values())):
        raise ValueError("Native GRU learner tensors must be FP32")
    return width


def gru_cell(inputs: mx.array, hidden: mx.array, parameters: dict[str, mx.array]) -> mx.array:
    """One differentiable cell; reset belongs to the sequence's pre-cell mask."""
    width = _validate(inputs, hidden, parameters)
    x = inputs @ parameters["input_weights"].T + parameters["input_bias"]
    h = hidden @ parameters["recurrent_weights"].T + parameters["recurrent_bias"]
    reset = mx.sigmoid(x[:, :width] + h[:, :width])
    update = mx.sigmoid(x[:, width:2*width] + h[:, width:2*width])
    proposed = mx.tanh(x[:, 2*width:] + reset * h[:, 2*width:])
    return (1-update)*proposed + update*hidden


def gru_sequence(inputs: mx.array, initial_hidden: mx.array, resets: mx.array,
                 parameters: dict[str, mx.array]) -> tuple[mx.array, mx.array]:
    """Time-major truncated BPTT with constant captured initial state.

    Returns post-cell states and exact pre-cell states, both [T,B,H]. The
    boolean reset mask is [T,B]. Gradients cross ordinary ticks and stop at an
    episode reset or the detached initial state. No transition shuffling occurs.
    """
    if inputs.ndim != 3 or inputs.shape[0] == 0 or resets.shape != inputs.shape[:2] or resets.dtype != mx.bool_:
        raise ValueError("GRU sequence requires [T,B,I] inputs and boolean [T,B] resets")
    _validate(inputs[0], initial_hidden, parameters)
    hidden = mx.stop_gradient(initial_hidden)
    states, before = [], []
    for tick in range(inputs.shape[0]):
        hidden = mx.where(resets[tick, :, None], mx.zeros_like(hidden), hidden)
        before.append(hidden)
        hidden = gru_cell(inputs[tick], hidden, parameters)
        states.append(hidden)
    return mx.stack(states), mx.stack(before)
