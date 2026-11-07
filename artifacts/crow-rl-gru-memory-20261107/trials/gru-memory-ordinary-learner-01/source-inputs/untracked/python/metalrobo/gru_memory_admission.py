"""Bounded mathematical admission of the native-layout MLX GRU learner.

NumPy/FP64 is an independent neural algebra instrument, not an environment
transition. Production simulation and inference remain native Metal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .gru_memory import PARAMETER_NAMES, gru_cell, gru_sequence


def write(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def reference(parameters: dict, inputs: np.ndarray, initial: np.ndarray,
              resets: np.ndarray, target: np.ndarray) -> tuple:
    """Independent FP64 forward and explicit reverse-mode derivatives."""
    wi, wh, bi, bh = (parameters[name].astype(np.float64) for name in PARAMETER_NAMES)
    hidden = initial.astype(np.float64).copy()
    width = hidden.shape[1]
    cache, outputs = [], []
    for tick in range(len(inputs)):
        hidden = np.where(resets[tick, :, None], 0.0, hidden)
        x = inputs[tick].astype(np.float64)
        xa, ha = x @ wi.T + bi, hidden @ wh.T + bh
        r = 1/(1+np.exp(-(xa[:, :width]+ha[:, :width])))
        z = 1/(1+np.exp(-(xa[:, width:2*width]+ha[:, width:2*width])))
        n = np.tanh(xa[:, 2*width:]+r*ha[:, 2*width:])
        cache.append((x, hidden, r, z, n, ha[:, 2*width:]))
        hidden = (1-z)*n+z*hidden
        outputs.append(hidden)
    outputs = np.stack(outputs)
    difference = outputs-target.astype(np.float64)
    loss = np.mean(difference**2) + .1*np.mean(hidden**2)
    derivative = 2*difference/difference.size
    derivative[-1] += .2*hidden/hidden.size
    gradients = {name: np.zeros_like(parameters[name], dtype=np.float64) for name in PARAMETER_NAMES}
    input_gradient = np.zeros_like(inputs, dtype=np.float64)
    carry = np.zeros_like(hidden)
    for tick in reversed(range(len(inputs))):
        x, h, r, z, n, hn = cache[tick]
        dh = derivative[tick]+carry
        dn = dh*(1-z)*(1-n*n)
        dr = dn*hn*r*(1-r)
        dz = dh*(h-n)*z*(1-z)
        dx_affine, dh_affine = np.concatenate((dr, dz, dn), axis=1), np.concatenate((dr, dz, dn*r), axis=1)
        gradients["input_weights"] += dx_affine.T @ x
        gradients["recurrent_weights"] += dh_affine.T @ h
        gradients["input_bias"] += dx_affine.sum(axis=0)
        gradients["recurrent_bias"] += dh_affine.sum(axis=0)
        input_gradient[tick] = dx_affine @ wi
        carry = np.where(resets[tick, :, None], 0.0, dh*z+dh_affine@wh)
    return loss, outputs, gradients, input_gradient


def parity(native: Path, output: Path) -> dict:
    records = []
    for name in ("small", "large"):
        directory = native/name
        N, I, H, T, split = map(int, np.fromfile(directory/"dimensions.u32", np.uint32))
        inputs = np.fromfile(directory/"inputs.f32", np.float32).reshape(T, N, I)
        masks = np.zeros((T, N), dtype=np.bool_)
        masks[0] = True
        masks[11, 1] = masks[23, 2] = True
        states, traces = [], []
        hidden = mx.zeros((N, H))
        for prefix, begin, end in (("first", 0, split), ("next", split, T)):
            shapes = ((3*H, I), (3*H, H), (3*H,), (3*H,))
            parameters = {key: mx.array(np.fromfile(directory/(prefix+"-"+key.replace("_", "-")+".f32"), np.float32).reshape(shape))
                          for key, shape in zip(PARAMETER_NAMES, shapes)}
            values, before = gru_sequence(mx.array(inputs[begin:end]), hidden, mx.array(masks[begin:end]), parameters)
            mx.eval(values, before)
            states.append(np.array(values));traces.append(np.array(before));hidden = values[-1]
        states, traces = np.concatenate(states), np.concatenate(traces)
        native_states = np.concatenate([np.fromfile(directory/(part+"-hidden.f32"), np.float32) for part in ("prefix", "tail")]).reshape(T, N, H)
        native_before = np.concatenate([np.fromfile(directory/(part+"-before.f32"), np.float32) for part in ("prefix", "tail")]).reshape(T, N, H)
        oracle = np.fromfile(directory/"oracle.f64", np.float64).reshape(T, N, H)
        errors = {"native_hidden_max_abs": float(np.max(np.abs(states-native_states))),
                  "native_pre_cell_max_abs": float(np.max(np.abs(traces-native_before))),
                  "FP64_max_abs": float(np.max(np.abs(states.astype(np.float64)-oracle)))}
        if max(errors.values()) > 3e-6:
            raise RuntimeError("MLX/native GRU parity exceeded prospective tolerance")
        np.savez(output/(name+"-mlx.npz"), hidden=states, before=traces, resets=masks)
        records.append({"case": name, **errors})
    return {"cases": records}


def gradients(output: Path) -> dict:
    rng = np.random.default_rng(2651109301)
    T, B, I, H = 6, 2, 3, 4
    shapes = ((3*H, I), (3*H, H), (3*H,), (3*H,))
    parameters = {name: rng.normal(0, .15, shape).astype(np.float32) for name, shape in zip(PARAMETER_NAMES, shapes)}
    inputs = rng.normal(0, .5, (T, B, I)).astype(np.float32)
    initial = rng.normal(0, .2, (B, H)).astype(np.float32)
    target = rng.normal(0, .2, (T, B, H)).astype(np.float32)
    resets = np.zeros((T, B), np.bool_);resets[3, 0] = True
    write(output/"gradient-plan.json", {"seed": 2651109301, "shape": [T, B, I, H],
          "reset": [3, 0], "FP64_full_gradient_max_abs": 3e-6, "FP64_full_gradient_relative_norm": 2e-5,
          "central_difference_epsilon": 1e-5, "central_difference_max_abs": 1e-8,
          "compiled_eager_max_abs": 3e-6, "optimizer": "8 full-sequence SGD updates at learning rate0.2",
          "scope": "all four GRU parameter tables and input gradients; detached initial state; no PPO or embodied claim"})
    mp = {key: mx.array(value) for key, value in parameters.items()}
    x, h0, mask, y = mx.array(inputs), mx.array(initial), mx.array(resets), mx.array(target)
    def loss(p, value):
        states, _ = gru_sequence(value, h0, mask, p)
        return mx.mean((states-y)**2)+.1*mx.mean(states[-1]**2)
    value, (gp, gx) = mx.value_and_grad(loss, argnums=(0, 1))(mp, x)
    mx.eval(value, gp, gx)
    expected_loss, expected_states, expected_gp, expected_gx = reference(parameters, inputs, initial, resets, target)
    records, maximum = {}, 0.0
    for key, actual, expected in [(k, np.array(gp[k]), expected_gp[k]) for k in PARAMETER_NAMES]+[("inputs", np.array(gx), expected_gx)]:
        error = float(np.max(np.abs(actual.astype(np.float64)-expected)))
        relative = float(np.linalg.norm(actual.astype(np.float64)-expected)/max(np.linalg.norm(expected), 1e-12))
        if error > 3e-6 or relative > 2e-5:
            raise RuntimeError("MLX BPTT disagrees with independent FP64 reverse derivatives: "+key)
        records[key] = {"max_abs": error, "relative_norm": relative}
        maximum = max(maximum, error)
    finite_difference = 0.0
    for key in PARAMETER_NAMES:
        for index in (0, parameters[key].size//2, parameters[key].size-1):
            high = {k: v.astype(np.float64).copy() for k, v in parameters.items()}
            low = {k: v.copy() for k, v in high.items()}
            high[key].flat[index] += 1e-5;low[key].flat[index] -= 1e-5
            numerical = (reference(high, inputs, initial, resets, target)[0]-reference(low, inputs, initial, resets, target)[0])/2e-5
            finite_difference = max(finite_difference, abs(numerical-expected_gp[key].flat[index]))
    if finite_difference > 1e-8 or abs(float(value.item())-expected_loss)>3e-6:
        raise RuntimeError("Independent derivative/loss control failed")
    compiled = mx.compile(mx.value_and_grad(loss, argnums=(0, 1)))
    cv, (cgp, cgx) = compiled(mp, x);mx.eval(cv, cgp, cgx)
    compiled_error = max(float(np.max(np.abs(np.array(cgp[k])-np.array(gp[k])))) for k in PARAMETER_NAMES)
    compiled_error = max(compiled_error, float(np.max(np.abs(np.array(cgx)-np.array(gx)))), abs(float(cv.item())-float(value.item())))
    if compiled_error > 3e-6:
        raise RuntimeError("Compiled GRU derivative drift")
    # A final-state objective must forget all earlier lane0 inputs at its reset,
    # retain a gradient through ordinary ticks, and detach captured initial state.
    def terminal(p, value, initial_value):
        states, _ = gru_sequence(value, initial_value, mask, p)
        return mx.sum(states[-1])
    _, (_, tx, th) = mx.value_and_grad(terminal, argnums=(0, 1, 2))(mp, x, h0)
    mx.eval(tx, th)
    if np.any(np.array(tx)[:3, 0] != 0) or np.any(np.array(th) != 0) or np.linalg.norm(np.array(tx)[0, 1]) < 1e-8:
        raise RuntimeError("Episode/truncation gradient boundary failed")
    changed = inputs.copy();changed[:3] += .7
    actual, _ = gru_sequence(x, h0, mask, mp)
    altered, _ = gru_sequence(mx.array(changed), h0, mask, mp);mx.eval(actual, altered)
    if not np.array_equal(np.array(actual)[3:, 0], np.array(altered)[3:, 0]) or np.max(np.abs(np.array(actual)[-1, 1]-np.array(altered)[-1, 1])) < 1e-6:
        raise RuntimeError("Causal memory/reset counterfactual failed")
    curve = [float(value.item())]
    for _ in range(8):
        _, gradient = mx.value_and_grad(loss)(mp, x)
        mp = {key: mp[key]-.2*gradient[key] for key in PARAMETER_NAMES}
        next_loss = loss(mp, x);mx.eval(next_loss, mp);curve.append(float(next_loss.item()))
    if not all(a>b for a, b in zip(curve, curve[1:])):
        raise RuntimeError("Sequence gradient updates did not reduce the fixed objective")
    # Shape/dtype guards should reject invalid learner contracts before a graph
    # can silently reinterpret native parameter or episode-reset semantics.
    rejected = 0
    for test in range(4):
        bad = dict(mp)
        try:
            if test == 0: bad["input_weights"] = bad["input_weights"][:, :-1]
            if test == 1: bad["extra"] = mx.array(0.0)
            if test == 2: bad["input_bias"] = bad["input_bias"].astype(mx.float16)
            gru_sequence(x, h0, mask.astype(mx.uint32) if test == 3 else mask, bad)
        except ValueError:
            rejected += 1
    if rejected != 4:
        raise RuntimeError("Malformed learner GRU contract accepted")
    np.savez(output/"gradient-fixture.npz", inputs=inputs, initial=initial, target=target, resets=resets,
             reference_states=expected_states, reference_input_gradient=expected_gx, mlx_input_gradient=np.array(gx),
             **parameters, **{"reference_"+k: v for k, v in expected_gp.items()}, **{"mlx_"+k: np.array(v) for k, v in gp.items()})
    return {"FP64_loss": expected_loss, "mlx_loss": float(value.item()), "gradients": records,
            "maximum_gradient_error": maximum, "finite_difference_max_abs": finite_difference,
            "compiled_eager_max_abs": compiled_error, "reset_cuts_gradient_exact": True,
            "captured_initial_state_detached": True, "ordinary_tick_memory_causal": True,
            "fixed_sequence_SGD_loss": curve, "learner_contract_rejections": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/"plan.json", {"schema": "numi.gru-mlx-admission-plan.v1", "native": str(args.native.resolve()),
          "native_summary_sha256": hashlib.sha256((args.native/"summary.json").read_bytes()).hexdigest(),
          "prediction": "two native fixture sequences/pre-cell states match<=3e-6; full parameter/input BPTT match independent FP64<=3e-6 and relative norm<=2e-5; finite differences<=1e-8; exact reset gradient cut; sequence SGD objective decreases",
          "stopping": "two40-tick native fixtures; one6-tick gradient fixture;12 parameter central differences;8 SGD updates; four contract controls; no retries or threshold changes",
          "scope": "neural algebra instrument only; not recurrent PPO or crow behavioral qualification"})
    native_summary = json.loads((args.native/"summary.json").read_text())
    if native_summary.get("status") != "passed":
        raise ValueError("Native component must pass before MLX parity")
    result = {"schema": "numi.gru-mlx-evidence.v1", "status": "passed", **parity(args.native, args.output),
              **gradients(args.output), "recurrent_PPO_integrated": False, "trainable_crow_memory_qualified": False}
    write(args.output/"summary.json", result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
