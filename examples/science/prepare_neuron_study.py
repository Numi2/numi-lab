#!/usr/bin/env python3
"""Author a bounded STDP study; this prepares files and does not run experiments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--instrument-build", type=Path, help="isolated build_neuron_instrument.py output")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed <= 0 for seed in args.seeds):
        parser.error("use distinct positive seeds")
    runtime = args.runtime.resolve()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    instrument = Path(__file__).with_name("neuron_weight_instrument.py").resolve()
    binary = runtime / "build/bin/metalrobo_neuron_culture_probe"
    native_artifacts = [binary, runtime / "build/lib/libmetalrobo.dylib", runtime / "build/shaders/NumiNeuron.metallib"]
    if args.instrument_build:
        build = args.instrument_build.resolve()
        binary = build / "metalrobo_neuron_culture_probe"
        native_artifacts = [binary, build / "NumiNeuron.metallib", build / "build-receipt.json"]
    calibration = json.loads(subprocess.check_output([sys.executable, str(instrument), "--calibrate"], text=True))
    model = {"version": "stdp-mean-v1", "statement": "Net potentiation dominates over the quick synthetic protocol: enabling STDP raises mean plastic weight by 0.0001 to 0.01 relative to frozen weights.",
             "predicted_effect_range": [0.0001, 0.01]}
    def save(name, value):
        with (directory / name).open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    save("calibration.json", calibration)
    save("model-v1.json", model)
    trials = []
    for index, seed in enumerate(args.seeds):
        # Alternating order is explicit; seed is the independent experimental unit.
        arms = ("control", "treatment") if index % 2 == 0 else ("treatment", "control")
        for arm in arms:
            trials.append({"id": f"seed-{seed}-{arm}", "pair": str(seed), "arm": arm,
                           "argv": [sys.executable, str(instrument), "--binary", str(binary),
                                    "--seed", str(seed), "--ablation", "stdp-off" if arm == "control" else "none"],
                           "env": {}, "timeout_seconds": 300})
    plan = {"schema": "numi.science.plan.v1",
            "question": "Does STDP increase mean plastic weight over the native quick synthetic-culture protocol?",
            "hypothesis": "Net potentiation produces a positive paired shift of at least 0.0001 authored weight units.",
            "owner": "Numi Lab native neuron culture", "repository": str(runtime),
            "backend": "Apple Metal synthetic LIF/STDP, prebuilt runtime bound by artifact hashes",
            "evidence_level": "simulation", "model": model,
            "instrument": {"description": "Native weight observable plus calibrated JSON reader and frozen-weight control check",
                           "calibration": str(directory / "calibration.json")},
            "artifacts": [str(instrument), str(directory / "calibration.json"), str(directory / "model-v1.json"),
                          *(str(path) for path in native_artifacts)],
            "design": {"intervention": "STDP enabled versus stdp-off; all other native protocol options identical",
                       "controls": "Paired seed, topology, initial state, sensory mapping 0, current scales, quick horizon; off arm must preserve 0.05 mean weight",
                       "experimental_unit": "One independently seeded synthetic culture; ticks are not independent replicates",
                       "allocation": "Declared seed order; control first for even pair indices, treatment first for odd; run serially"},
            "observable": {"name": "mean_plastic_weight", "unit": "authored weight units", "path": ["mean_plastic_weight"]},
            "prediction": {"estimand": "paired_difference_mean", "minimum": 0.0001, "maximum": 0.01},
            "validity": [{"path": ["schema"], "equals": "numi.science.neuron-weight.v1"},
                         {"path": ["simulation_only"], "equals": True}],
            "paired_equal": [[key] for key in ("network_seed", "culture_fingerprint", "starting_state_fingerprint",
                                               "synaptic_current_scale", "stimulation_current", "windows")],
            "trials": trials,
            "limitations": "A small deterministic synthetic protocol from unconditioned weights. No biological calibration, learning advantage, full Potter reproduction, population confidence interval, or hardware evidence. Prebuilt binary identity is retained; source-to-binary reproducibility is not established here."}
    save("plan.json", plan)
    print(directory / "plan.json")


if __name__ == "__main__":
    main()
