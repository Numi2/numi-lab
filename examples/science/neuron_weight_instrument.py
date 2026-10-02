#!/usr/bin/env python3
"""Read native synthetic-culture plastic weights without taking over dynamics."""
import argparse
import json
import math
from pathlib import Path
import subprocess


def measure(raw):
    if raw.get("schema") != "numi.neuron-culture.potter-switch.v1" or raw.get("simulation_only") is not True:
        raise ValueError("not a synthetic Potter protocol observation")
    value = raw["mean_plastic_weight"]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 0.1:
        raise ValueError("plastic weight outside native authored bounds")
    if type(raw.get("windows")) is not int or raw["windows"] <= 0:
        raise ValueError("no completed windows")
    return {"schema": "numi.science.neuron-weight.v1", "mean_plastic_weight": value,
            "simulation_only": True, "windows": raw["windows"],
            "network_seed": raw["network_seed"], "culture_fingerprint": raw["culture_fingerprint"],
            "starting_state_fingerprint": raw["starting_state_fingerprint"],
            "synaptic_current_scale": raw["synaptic_current_scale"],
            "stimulation_current": raw["stimulation_current"], "ablation": raw["ablation"]}


def calibrate():
    fixture = {"schema": "numi.neuron-culture.potter-switch.v1", "simulation_only": True,
               "windows": 8, "network_seed": 42, "culture_fingerprint": 123,
               "starting_state_fingerprint": 456, "synaptic_current_scale": 175000,
               "stimulation_current": 20000, "ablation": 0, "mean_plastic_weight": 0.0625}
    if measure(fixture)["mean_plastic_weight"] != 0.0625:
        raise ValueError("known-value calibration failed")
    rejected = 0
    for change in ({"mean_plastic_weight": float("nan")}, {"mean_plastic_weight": 0.11},
                   {"mean_plastic_weight": True}, {"windows": 0}, {"simulation_only": False}):
        try:
            measure({**fixture, **change})
        except ValueError:
            rejected += 1
    if rejected != 5:
        raise ValueError("invalid-measurement rejection failed")
    return {"schema": "numi.science.instrument-calibration.v1", "known_input": fixture,
            "measured_weight": 0.0625, "absolute_error": 0.0, "rejected_invalid_inputs": rejected,
            "scope": "JSON extraction and authored numerical bounds; not biological or sensor calibration"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ablation", choices=("none", "stdp-off"))
    args = parser.parse_args()
    if args.calibrate:
        result = calibrate()
    else:
        if not args.binary or not args.seed or not args.ablation:
            parser.error("--binary, --seed and --ablation are required")
        argv = [str(args.binary), "protocol", "--quick", "--network-seed", str(args.seed),
                "--sensory-mapping", "0", "--ablation", args.ablation]
        with Path("native.json").open("xb") as stdout, Path("native.stderr.log").open("xb") as stderr:
            subprocess.run(argv, stdout=stdout, stderr=stderr, check=True)
        raw = json.loads(Path("native.json").read_text())
        result = measure(raw)
        if result["network_seed"] != args.seed or result["ablation"] != (3 if args.ablation == "stdp-off" else 0):
            raise ValueError("native result does not match the requested seed/ablation")
        if args.ablation == "stdp-off" and abs(result["mean_plastic_weight"] - 0.05) > 1e-7:
            raise ValueError("STDP-off control did not preserve the authored mean weight")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
