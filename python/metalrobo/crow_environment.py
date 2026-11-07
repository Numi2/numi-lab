"""Crow experiment orchestration; all transitions remain in native Metal.

Every invocation creates a new evidence directory and writes its plan before
execution. This module must never implement a second task/physics transition.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


VARIANTS = {
    "v9-visual-neural": "crow_journey_window/crow-journey.sensor-fast.visual-observation.json",
    "v10-world-model": "crow_navigation_course/crow-navigation.sensor-fast.visual-observation.json",
}
CONTRACT = ("world_fingerprint", "task_fingerprint", "observation_fingerprint", "action_fingerprint")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    # Exclusive creation makes partial and failed trials durable, never resumable
    # by silently overwriting a result or its prospective plan.
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def capture(arguments: list[str], root: Path) -> str:
    return subprocess.check_output(arguments, cwd=root, text=True).strip()


def runtime(root: Path) -> dict:
    build = Path(os.environ.get("NUMI_CROW_BUILD_DIR", os.environ.get("NUMI_BUILD_DIR", root / "build"))).resolve()
    return {
        "rollout": build / "bin/metalrobo_task_rollout",
        "train": build / "bin/metalrobo_task_train",
        "library": build / "lib/libmetalrobo.dylib",
        "metallib": build / "shaders/MetalRobo.metallib",
        "head_check": build / "bin/metalrobo_crow_embodiment_check",
        "foraging_check": build / "bin/metalrobo_crow_foraging_check",
        "cue_foraging_check": build / "bin/metalrobo_crow_cue_foraging_check",
        "temporal_history_check": build / "bin/metalrobo_crow_temporal_history_check",
        "foraging_compile": build / "bin/metalrobo_crow_foraging_compile",
        "eye_check": build / "bin/metalrobo_crow_eye_check",
        "retinal_check": build / "bin/metalrobo_retinal_stream_check",
        "retinal_api_check": build / "bin/metalrobo_crow_retinal_api_check",
        "life_check": build / "bin/metalrobo_crow_life_checkpoint_check",
        "retinal_live_check": build / "bin/metalrobo_crow_retinal_live_check",
        "retinal_contract_check": build / "bin/metalrobo_crow_retinal_contract_check",
        "retinal_metallib": build / "shaders/RetinalStream.metallib",
    }


def preflight(root: Path, profile: dict) -> dict:
    import mlx.core as mx
    import numpy as np
    from metalrobo import mlx_policy_worker  # actual learner dependencies

    del mlx_policy_worker
    paths = runtime(root)
    paths.update({name: root / "assets" / relative for name, relative in VARIANTS.items()})
    paths["profile"] = root / "profiles/american-crow-rl-v1.json"
    missing = [str(path) for path in paths.values() if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError(f"Missing runtime inputs: {missing}")
    if not mx.metal.is_available():
        raise RuntimeError("The native crow environment requires Apple Metal")
    free = shutil.disk_usage(root).free
    if free < profile["minimum_free_disk_gib"] * 1024**3:
        raise RuntimeError("Free disk is below the project's minimum")
    return {
        "schema": "numi.crow-preflight.v1", "host": platform.node(),
        "machine": platform.machine(), "macos": platform.mac_ver()[0],
        "python": sys.executable, "numpy": np.__version__,
        "mlx": importlib.metadata.version("mlx"),
        "metal_device": mx.device_info(), "free_disk_bytes": free,
        "inputs": {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()},
        "source_commit": capture(["git", "rev-parse", "HEAD"], root),
        "status": capture(["git", "status", "--porcelain=v1"], root),
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)).hexdigest(),
        "untracked_inputs": {
            name: sha(root / name) for name in
            capture(["git", "ls-files", "--others", "--exclude-standard"], root).splitlines()
            if (root / name).is_file()
        },
        "readiness": "native engineering trials only; ecological qualification pending",
    }


def recurrent_rollout_cli(root: Path, output: Path, checked: dict, retained: Path) -> dict:
    base=["--birdflow-american-crow-journey","--birdflow-journey-variant","v13-retinal-feeding",
        "--envs","2","--repeats","1","--no-scheduled-resets","--minimum-difficulty-band","0",
        "--maximum-difficulty-band","0","--seed","2651109501","--policy-pack",str(output/"native-evidence/first.policypack"),
        "--visual-observation-config",str(output/"native-evidence/retinal-observation.json")]
    records=[]
    for chunk in (1,8):
        for resume,steps,tick in ((False,37,37),(True,13,50)):
            name=("cli-resume" if resume else "cli-start")+"-chunk"+str(chunk)
            command=[checked["inputs"]["rollout"]["path"],"--metallib",checked["inputs"]["metallib"]["path"]]+base+[
                "--steps",str(steps),"--chunk",str(chunk),"--rollout-pack",str(output/(name+".rolloutpack")),
                "--save-life-checkpoint",str(output/(name+".lifecheckpoint"))]
            if resume:command += ["--resume-life-checkpoint",str(output/("cli-start-chunk"+str(chunk)+".lifecheckpoint"))]
            value=execute(output,name,command,root,120)
            if value.get("failed_environment_steps")!=0 or value.get("finite_resource_accounting",{}).get("accepted_control_ticks_by_environment")!=[tick,tick]:
                raise RuntimeError("Recurrent regular CLI introduced warmup/failure or lost accepted life clock: "+name)
            if resume and value.get("life_checkpoint_restored") is not True:
                raise RuntimeError("Recurrent regular CLI did not restore full native life")
            records.append({"name":name,"fingerprint":value["policy_rollout_fingerprint"],"accepted_ticks":[tick,tick],
                "rollout_sha256":sha(output/(name+".rolloutpack"))})
    for prefix in ("cli-start","cli-resume"):
        pair=[r for r in records if r["name"].startswith(prefix)]
        if pair[0]["fingerprint"]!=pair[1]["fingerprint"] or pair[0]["rollout_sha256"]!=pair[1]["rollout_sha256"]:
            raise RuntimeError("Recurrent regular CLI depends on chunk partition: "+prefix)
    name="cli-retained-resume"
    command=["/usr/bin/env","DYLD_LIBRARY_PATH="+str(retained/"lib"),str(retained/"bin/metalrobo_task_rollout"),
        "--metallib",str(retained/"shaders/MetalRobo.metallib")]+base+["--steps","13","--chunk","8",
        "--resume-life-checkpoint",str(output/"cli-start-chunk8.lifecheckpoint"),
        "--save-life-checkpoint",str(output/(name+".lifecheckpoint")),"--rollout-pack",str(output/(name+".rolloutpack"))]
    recovered=execute(output,name,command,root,120)
    if (recovered.get("policy_rollout_fingerprint")!=records[3]["fingerprint"] or sha(output/(name+".rolloutpack"))!=records[3]["rollout_sha256"]
            or recovered.get("finite_resource_accounting",{}).get("accepted_control_ticks_by_environment")!=[50,50]):
        raise RuntimeError("Retained regular CLI recurrent continuation differs")
    result={"schema":"numi.crow-recurrent-rollout-cli.v1","status":"passed","records":records,
        "chunk1_8_start_and_resume_canonical_bytes_exact":True,"retained_CLI_continuation_canonical_bytes_exact":True,
        "accepted_ticks":[37,50],"unreported_warmup_steps":0,"development_seed":2651109501}
    write(output/"cli-life-summary.json",result);return result


def recurrent_rollout_admission(root: Path, output: Path, checked: dict) -> dict:
    policy = recurrent_policy_admission(root, output, checked)
    swift = execute(output, "rollout-swift", [checked["inputs"]["recurrent_rollout_check"]["path"],
        str(output/"native-evidence"), str(output/"swift-evidence")], root, 180)
    learner = execute(output, "rollout-mlx", [sys.executable, "-m", "metalrobo.recurrent_rollout_admission",
        "--native", str(output/"native-evidence"), "--swift", str(output/"swift-evidence"),
        "--output", str(output/"rollout-evidence")], root, 180)
    if swift.get("status") != "passed" or swift.get("rejection_controls") != 10 or learner.get("status") != "passed" or learner.get("parser_and_flat_consumption_rejections") != 11:
        raise RuntimeError("Recurrent Swift/learner rollout admission failed")
    retained=root/".numi/retained-runtime"/output.name
    extra={}
    for key,relative in (("rollout", "bin/metalrobo_task_rollout"),
                         ("recurrent_rollout_check", "bin/metalrobo_crow_recurrent_rollout_check"),
                         ("recurrent_source_recurrent_rollout_admission.py", "python/metalrobo/recurrent_rollout_admission.py")):
        target=retained/relative;shutil.copy2(checked["inputs"][key]["path"],target)
        if sha(target)!=checked["inputs"][key]["sha256"]:
            raise RuntimeError("Recurrent rollout runtime/source changed during admission: "+key)
        extra[key]={"path":str(target),"sha256":sha(target)}
    write(output/"retained-rollout-inputs.json",extra)
    recovered_swift=execute(output,"retained-rollout-swift",["/usr/bin/env","DYLD_LIBRARY_PATH="+str(retained/"lib"),
        str(retained/"bin/metalrobo_crow_recurrent_rollout_check"),str(output/"native-evidence"),
        str(output/"retained-swift-recovery")],root,180)
    if recovered_swift!=swift:raise RuntimeError("Retained Swift rollout summary differs")
    recovered=execute(output,"retained-rollout-mlx",["/usr/bin/env","PYTHONPATH="+str(retained/"python"),
        "METALROBO_LIBRARY="+str(retained/"lib/libmetalrobo.dylib"),sys.executable,"-m","metalrobo.recurrent_rollout_admission",
        "--native",str(output/"native-evidence"),"--swift",str(output/"retained-swift-recovery"),
        "--output",str(output/"retained-rollout-recovery")],root,180)
    if recovered!=learner:raise RuntimeError("Retained rollout learner summary differs")
    import numpy as np
    with np.load(output/"rollout-evidence/artifact-sequence-parity.npz") as expected,np.load(output/"retained-rollout-recovery/artifact-sequence-parity.npz") as actual:
        if expected.files!=actual.files or any(not np.array_equal(expected[k],actual[k]) for k in expected.files):
            raise RuntimeError("Retained recurrent rollout sequence arrays differ")
    cli=recurrent_rollout_cli(root,output,checked,retained)
    cli_learner=execute(output,"cli-sequence-mlx",[sys.executable,"-m","metalrobo.recurrent_rollout_admission",
        "--cli",str(output),"--output",str(output/"cli-sequence-evidence")],root,180)
    if cli_learner.get("status")!="passed" or cli_learner.get("continuation_zero_hidden_action_difference",0)<1e-4:
        raise RuntimeError("Regular CLI recurrent sequence continuity/causal state check failed")
    recovered_cli=execute(output,"retained-cli-sequence-mlx",["/usr/bin/env","PYTHONPATH="+str(retained/"python"),
        "METALROBO_LIBRARY="+str(retained/"lib/libmetalrobo.dylib"),sys.executable,"-m","metalrobo.recurrent_rollout_admission",
        "--cli",str(output),"--output",str(output/"retained-cli-sequence-recovery")],root,180)
    if recovered_cli!=cli_learner:raise RuntimeError("Retained regular CLI sequence learner summary differs")
    with np.load(output/"cli-sequence-evidence/cli-sequence-parity.npz") as expected,np.load(output/"retained-cli-sequence-recovery/cli-sequence-parity.npz") as actual:
        if expected.files!=actual.files or any(not np.array_equal(expected[k],actual[k]) for k in expected.files):
            raise RuntimeError("Retained regular CLI sequence learner arrays differ")
    return {"schema":"numi.crow-recurrent-rollout-admission.v2","status":"passed","policy":policy,"swift":swift,"learner":learner,
        "regular_CLI":cli,"regular_CLI_sequence":cli_learner,"retained_regular_CLI_sequence_summary_and_all_arrays_exact":True,
        "retained_Swift_and_learner_summary_all_sequence_arrays_exact":True,
        "recurrent_rollout_format_integrated":True,"sequence_PPO_integrated":False,"embodied_memory_qualified":False}


def recurrent_policy_admission(root: Path, output: Path, checked: dict) -> dict:
    actor = recurrent_actor_admission(root, output, checked)
    learner = execute(output, "policy-mlx", [sys.executable, "-m", "metalrobo.recurrent_policy_admission",
        "--native", str(output/"native-evidence"), "--output", str(output/"policy-evidence")], root, 180)
    if learner.get("status") != "passed" or learner.get("contract_rejections") != 17:
        raise RuntimeError("Recurrent native/learner policy admission failed")
    retained = root/".numi/retained-runtime"/output.name
    sources = {}
    for name in ("native.py", "mlx_policy_learning.py", "gru_memory.py", "recurrent_policy_admission.py"):
        target = retained/"python/metalrobo"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/"python/metalrobo"/name, target)
        if sha(target) != checked["inputs"]["recurrent_source_"+name]["sha256"]:
            raise RuntimeError("Recurrent learner source changed during admission: "+name)
        sources[name] = {"path": str(target), "sha256": sha(target)}
    (retained/"python/metalrobo/__init__.py").touch(exist_ok=False)
    write(output/"retained-learner-source.json", sources)
    recovered = execute(output, "retained-policy-mlx", ["/usr/bin/env", "PYTHONPATH="+str(retained/"python"),
        "METALROBO_LIBRARY="+str(retained/"lib/libmetalrobo.dylib"), sys.executable,
        "-m", "metalrobo.recurrent_policy_admission", "--native", str(output/"native-evidence"),
        "--output", str(output/"retained-policy-recovery")], root, 180)
    if recovered != learner:
        raise RuntimeError("Retained recurrent policy source recovery summary differs")
    import numpy as np
    with np.load(output/"policy-evidence/sequence-parity.npz") as expected, np.load(output/"retained-policy-recovery/sequence-parity.npz") as actual:
        if expected.files != actual.files or any(not np.array_equal(expected[k],actual[k]) for k in expected.files):
            raise RuntimeError("Retained recurrent policy sequence arrays differ")
    for name in ("first.policypack", "next.policypack", "dense-drop-control.policypack"):
        if sha(output/"policy-evidence"/name) != sha(output/"retained-policy-recovery"/name):
            raise RuntimeError("Retained recurrent canonical writer bytes differ: "+name)
    return {"schema": "numi.crow-recurrent-policy-admission.v1", "status": "passed", "actor": actor,
        "policy_interop": learner, "retained_learner_summary_arrays_and_policy_bytes_exact": True,
        "installed_matching_Python_MLX_NumPy_required": True,
        "sequence_PPO_integrated": False, "embodied_memory_qualified": False}


def recurrent_actor_admission(root: Path, output: Path, checked: dict) -> dict:
    native = execute(output, "recurrent-native", [checked["inputs"]["recurrent_check"]["path"],
        str(output/"native-evidence"), checked["inputs"]["policy_abi3_shader"]["path"]], root, 180)
    if (native.get("status") != "passed" or native.get("invalid_contract_controls") != 7
            or native.get("publication_rejection_controls") != 5
            or not all(native.get(key) is True for key in ("chunk_preview_trace_exact",
                "fresh_process_full_life_exact", "stale_policy_shader_rejected"))):
        raise RuntimeError("Native recurrent actor admission failed")
    retained = root/".numi/retained-runtime"/output.name
    retained.mkdir(parents=True, exist_ok=False)
    retained_inputs = {}
    for name, relative in (("library", "lib/libmetalrobo.dylib"),
                           ("recurrent_check", "bin/metalrobo_crow_recurrent_actor_check"),
                           ("metallib", "shaders/MetalRobo.metallib"),
                           ("retinal_metallib", "shaders/RetinalStream.metallib"),
                           ("gru_metallib", "shaders/GRUMemory.metallib")):
        source = Path(checked["inputs"][name]["path"])
        target = retained/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != checked["inputs"][name]["sha256"]:
            raise RuntimeError("Recurrent runtime changed during admission: "+name)
        retained_inputs[name] = {"path": str(target), "sha256": sha(target)}
    write(output/"retained-runtime.json", retained_inputs)
    recovery = output/"retained-native-recovery"
    execute(output, "retained-native", ["/usr/bin/env", "DYLD_LIBRARY_PATH="+str(retained/"lib"),
        str(retained/"bin/metalrobo_crow_recurrent_actor_check"), "--restore",
        str(output/"native-evidence"), str(recovery)], root, 40, json_output=False)
    for name in ("actor.f32", "actions.f32", "before.f32", "reset.u32", "transitions.bin", "q.f32", "v.f32"):
        if sha(output/"native-evidence/tail"/name) != sha(recovery/name):
            raise RuntimeError("Retained recurrent actor future differs: "+name)
    write(output/"retained-recovery.json", {"future_streams_hidden_traces_and_physics_exact": True,
        "internal_native_child_all_three_life_components_exact": True,
        "retained_child_final_life_component_bytes_compared": False,
        "same_device_OS_and_matching_native_libraries_required": True})
    return {"schema": "numi.crow-recurrent-actor-admission.v1", "status": "passed", "native": native,
        "runtime_retained": True, "retained_runtime_future_exact": True,
        "actor_memory_integrated": True, "coupled_life_includes_GRU": True,
        "sequence_PPO_integrated": False, "embodied_memory_qualified": False,
        "boundary": "authored native v13 controller; recurrent learner and ecological tasks remain open"}


def memory_admission(root: Path, output: Path, checked: dict) -> dict:
    native = execute(output, "gru-native", [checked["inputs"]["gru_check"]["path"], str(output/"native-evidence")], root, 180)
    if native.get("status") != "passed" or native.get("negative_controls") != 31:
        raise RuntimeError("Native GRU component admission failed")
    learner = execute(output, "gru-mlx", [sys.executable, "-m", "metalrobo.gru_memory_admission",
                      "--native", str(output/"native-evidence"), "--output", str(output/"mlx-evidence")], root, 180)
    if learner.get("status") != "passed" or learner.get("reset_cuts_gradient_exact") is not True:
        raise RuntimeError("Differentiable GRU component admission failed")
    retained = root/".numi/retained-runtime"/output.name
    retained.mkdir(parents=True, exist_ok=False)
    retained_inputs = {}
    for name, relative in (("library", "lib/libmetalrobo.dylib"),
                           ("gru_check", "bin/metalrobo_gru_memory_check"),
                           ("gru_metallib", "shaders/GRUMemory.metallib")):
        source = Path(checked["inputs"][name]["path"])
        target = retained/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != checked["inputs"][name]["sha256"]:
            raise RuntimeError("GRU runtime changed during admission: "+name)
        retained_inputs[name] = {"path": str(target), "sha256": sha(target)}
    for name in ("gru_memory.py", "gru_memory_admission.py"):
        target = retained/"python/metalrobo"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/"python/metalrobo"/name, target)
        retained_inputs[name] = {"path": str(target), "sha256": sha(target)}
    (retained/"python/metalrobo/__init__.py").touch(exist_ok=False)
    write(output/"retained-runtime.json", retained_inputs)
    recovery = output/"retained-native-recovery"
    recovery.mkdir()
    shutil.copyfile(output/"native-evidence/large/saved.memorycheckpoint", recovery/"saved.memorycheckpoint")
    execute(output, "retained-native", ["/usr/bin/env", "DYLD_LIBRARY_PATH="+str(retained/"lib"),
            str(retained/"bin/metalrobo_gru_memory_check"), "--child", str(recovery)], root, 40, json_output=False)
    for expected, actual in (("tail-hidden.f32", "fresh-tail-hidden.f32"), ("tail-before.f32", "fresh-tail-before.f32"),
                             ("tail-reset.u32", "fresh-tail-reset.u32"), ("final.memorycheckpoint", "fresh-final.memorycheckpoint")):
        if sha(output/"native-evidence/large"/expected) != sha(recovery/actual):
            raise RuntimeError("Retained GRU runtime recovery differs: "+actual)
    recovered_learner = execute(output, "retained-mlx", ["/usr/bin/env", "PYTHONPATH="+str(retained/"python"),
        sys.executable, "-m", "metalrobo.gru_memory_admission", "--native", str(output/"native-evidence"),
        "--output", str(output/"retained-mlx-recovery")], root, 180)
    if recovered_learner != learner:
        raise RuntimeError("Retained GRU learner source recovery differs")
    import numpy as np
    for name in ("small-mlx.npz", "large-mlx.npz", "gradient-fixture.npz"):
        with np.load(output/"mlx-evidence"/name) as expected, np.load(output/"retained-mlx-recovery"/name) as actual:
            if expected.files != actual.files or any(not np.array_equal(expected[key], actual[key]) for key in expected.files):
                raise RuntimeError("Retained GRU learner array recovery differs: "+name)
    write(output/"retained-recovery.json", {"native_future_and_checkpoint_exact": True,
        "independent_learner_summary_and_all_arrays_exact": True, "installed_interpreter_MLX_NumPy_still_required": True})
    return {"schema": "numi.crow-gru-component-admission.v1", "status": "passed", "native": native,
            "learner": learner, "runtime_retained": True, "retained_runtime_recovery_exact": True, "recurrent_policy_integrated": False,
            "sequence_PPO_integrated": False, "trainable_crow_memory_qualified": False,
            "boundary": "native/MLX neural component only; no embodied memory or whole-environment qualification"}



def retain_source_delta(root: Path, output: Path, checked: dict) -> None:
    """Retain the exact executable source delta, including new native files."""
    directory = output / "source-inputs"
    directory.mkdir()
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
    if hashlib.sha256(patch).hexdigest() != checked["tracked_diff_sha256"]:
        raise RuntimeError("Source changed between preflight and retention")
    (directory / "tracked.patch").write_bytes(patch)
    retained, external = {}, {}
    for name, expected in checked["untracked_inputs"].items():
        # Scientific payload archives have their own receipt/inventory. New
        # runtime code must be portable even when the trial precedes a commit.
        if Path(name).parts[0] == "artifacts":
            external[name] = expected
            continue
        source = root / name
        target = directory / "untracked" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != expected:
            raise RuntimeError(f"Source changed during retention: {name}")
        retained[name] = expected
    write(directory / "manifest.json", {"base_commit": checked["source_commit"],
        "tracked_patch_sha256": checked["tracked_diff_sha256"],
        "retained_untracked": retained, "external_artifact_inputs": external})


def execute(output: Path, name: str, command: list[str], root: Path, timeout: int, *, json_output: bool = True) -> dict:
    started = time.monotonic()
    exit_code = None
    timed_out = False
    interrupted = False
    with (output / f"{name}.json").open("x") as stdout, (output / f"{name}.stderr").open("x") as stderr:
        try:
            result = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, timeout=timeout)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except KeyboardInterrupt:
            interrupted = True
    receipt = {
        "command": command, "exit_code": exit_code, "timed_out": timed_out, "interrupted": interrupted,
        "wall_seconds": time.monotonic() - started,
        "stdout_sha256": sha(output / f"{name}.json"),
        "stderr_sha256": sha(output / f"{name}.stderr"),
    }
    write(output / f"{name}.receipt.json", receipt)
    if interrupted:
        raise KeyboardInterrupt("Native trial interrupted; retained receipt")
    if timed_out or exit_code != 0:
        raise RuntimeError(f"{name} failed; retained receipt and stderr in {output}")
    if not json_output:
        return receipt
    with (output / f"{name}.json").open() as handle:
        return json.load(handle)


def rollout(root: Path, variant: str, envs: int, steps: int, chunk: int, seed: int,
            band: int, policy: Path | None = None, course: str = "training") -> list[str]:
    paths = runtime(root)
    command = [str(paths["rollout"]), "--metallib", str(paths["metallib"]),
               "--birdflow-american-crow-journey", "--birdflow-journey-variant", variant,
               "--envs", str(envs), "--steps", str(steps), "--repeats", "1",
               "--chunk", str(chunk), "--seed", str(seed), "--no-scheduled-resets",
               "--minimum-difficulty-band", str(band), "--maximum-difficulty-band", str(band)]
    if variant in VARIANTS:
        command += ["--visual-observation-config", str(root / "assets" / VARIANTS[variant])]
    command += ["--policy-pack", str(policy)] if policy else ["--native-policy"]
    if variant == "v10-world-model":
        command += ["--birdflow-navigation-course", course]
    return command


def validate_native(value: dict, *, envs: int, steps: int, seed: int, policy: bool,
                    variant: str | None = None, action_source: str | None = None) -> None:
    expected = {"environments": envs, "steps_per_repeat": steps, "repeats": 1,
                "benchmark_seed": seed, "scheduled_resets": False,
                "birdflow_journey_teacher": False,
                "visual_observation": variant not in ("v11-embodied", "v12-feeding-station"),
                "action_source": action_source or ("policy_pack" if policy else "compiled_policy"),
                "failed_environment_steps": 0}
    if variant is not None:
        expected["birdflow_journey_variant"] = variant
    failures = [key for key, expected_value in expected.items() if value.get(key) != expected_value]
    if failures:
        raise RuntimeError(f"Native evidence violated trial identity/validity: {failures}")
    if value.get("birdflow_journey_variant") == "v10-world-model":
        evidence = value.get("first_episode_navigation")
        if not isinstance(evidence, dict) or evidence.get("environment_count") != envs:
            raise RuntimeError("First-episode navigation evidence missing or incomplete")
        validate_first_episode(evidence, envs, steps)


def validate_first_episode(evidence: dict, envs: int, horizon: int) -> None:
    fields = ("steps", "maximum_waypoints", "completion_seen", "safe_completion",
              "physical_failure", "closed", "termination_reason", "first_completion_step")
    arrays = {field: evidence.get(field + "_by_environment") for field in fields}
    if any(not isinstance(array, list) or len(array) != envs for array in arrays.values()):
        raise RuntimeError("Incomplete first-episode lane arrays")
    for lane in range(envs):
        row = {key: array[lane] for key, array in arrays.items()}
        if type(row["steps"]) is not int or not 1 <= row["steps"] <= horizon:
            raise RuntimeError("Invalid first-episode duration")
        waypoints = row["maximum_waypoints"]
        if not isinstance(waypoints, (float, int)) or not math.isfinite(waypoints) or not 0 <= waypoints <= 5 or int(waypoints) != waypoints:
            raise RuntimeError("Invalid accepted waypoint count")
        for field in ("completion_seen", "safe_completion", "physical_failure", "closed"):
            if type(row[field]) is not bool:
                raise RuntimeError("First-episode lane flags must be Boolean")
        reason = row["termination_reason"]
        if type(reason) is not int or not 0 <= reason <= 7 or row["closed"] != (reason > 0):
            raise RuntimeError("Invalid first-episode termination identity")
        if row["physical_failure"] != (row["closed"] and reason not in (4, 7)):
            raise RuntimeError("Physical failure and termination disagree")
        if row["safe_completion"] != (row["completion_seen"] and not row["physical_failure"]):
            raise RuntimeError("Inconsistent safe-completion evidence")
        first = row["first_completion_step"]
        if row["completion_seen"]:
            if waypoints != 5 or type(first) is not int or not 1 <= first <= row["steps"]:
                raise RuntimeError("Completion lacks accepted full-route evidence")
        elif first is not None:
            raise RuntimeError("Unexpected completion timestamp")
        if not row["closed"] and row["steps"] != horizon:
            raise RuntimeError("Truncated first episode cannot satisfy a full horizon")
    counts = {
        "safe_completion_count": sum(arrays["safe_completion"]),
        "completion_seen_count": sum(arrays["completion_seen"]),
        "physical_failure_count": sum(arrays["physical_failure"]),
        "closed_episode_count": sum(arrays["closed"]),
        "horizon_censored_count": envs - sum(arrays["closed"]),
    }
    if any(type(evidence.get(key)) is not int or evidence[key] != count for key, count in counts.items()):
        raise RuntimeError("First-episode aggregate counts disagree with lanes")


def probe(root: Path, output: Path, profile: dict) -> dict:
    settings = profile["probe"]
    seed, envs, steps = settings["seed"], settings["environments"], settings["steps"]
    records = []
    baseline = None
    for name, chunk in [("replay-a", 1), ("replay-b", 1), ("replay-chunk8", 8)]:
        value = execute(output, name, rollout(root, "v9-visual-neural", envs, steps, chunk, seed, 4), root, settings["timeout_seconds"])
        validate_native(value, envs=envs, steps=steps, seed=seed, policy=False)
        keys = (*CONTRACT, "sensor_fingerprint", "policy_rollout_fingerprint")
        fingerprints = {key: value[key] for key in keys}
        if baseline is None:
            baseline = fingerprints
        elif fingerprints != baseline:
            raise RuntimeError(f"Deterministic replay failed at {name}")
        records.append({"name": name, "chunk": chunk, "fingerprints": fingerprints})
    for envs in settings["scale_environments"]:
        name = f"scale-{envs}"
        value = execute(output, name, rollout(root, "v9-visual-neural", envs, steps, 8, seed, 4), root, settings["timeout_seconds"])
        validate_native(value, envs=envs, steps=steps, seed=seed, policy=False)
        records.append({"name": name, "environments": envs,
                        "native_environment_steps_per_second": value["environment_control_steps_per_second"],
                        "retained_buffer_bytes": value["retained_buffer_bytes"],
                        "peak_aliased_bytes": value["peak_aliased_bytes"]})
    return {"status": "passed", "claim": "native replay and bounded rollout scaling only",
            "policy": "untrained compiled engineering policy", "records": records,
            "ecological_environment_qualified": False}


def bootstrap(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    source = args.source_policy.resolve(strict=True)
    layout = execute(output, "native-contract", rollout(root, args.variant, 1, 1, 1,
                     profile["probe"]["seed"], 0), root, 180)
    validate_native(layout, envs=1, steps=1, seed=profile["probe"]["seed"], policy=False)
    command = [sys.executable, "-m", "metalrobo.mlx_policy_worker", "initialize",
               "--actor-policy-pack", str(source), "--actor-fresh-critic", "--deterministic",
               "--actor-observations", str(layout["actor_observation_count"]),
               "--critic-observations", str(layout["critic_observation_count"]),
               "--actions", str(layout["action_count"]),
               "--policy-id", f"crow-explicit-actor-transfer-{args.variant}",
               "--seed", str(profile["development_seeds"][0]),
               "--native-library", str(runtime(root)["library"]),
               "--output", str(output / "deployment.policypack")]
    for field in CONTRACT:
        command += ["--" + field.replace("_", "-"), str(layout[field])]
    if args.variant == "v10-world-model":
        command += ["--actor-observation-extension-offset", "84"]
    result = execute(output, "actor-transfer", command, root, 180)
    policy = output / "deployment.policypack"
    check = execute(output, "loaded-policy-check", rollout(root, args.variant, 8, 32, 8,
                    profile["probe"]["seed"], 0, policy), root, 180)
    validate_native(check, envs=8, steps=32, seed=profile["probe"]["seed"], policy=True)
    return {"status": "created_unqualified_actor_transfer", "source_sha256": sha(source),
            "policy_sha256": sha(policy), "critic": "fresh initialization",
            "training_updates": 0, "native_contract": {key: layout[key] for key in CONTRACT},
            "initialization": result, "ecological_environment_qualified": False}


def evaluate(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    policy = args.policy.resolve(strict=True)
    # Development uses its own seed namespace; held-out seeds are never spent
    # automatically by engineering probes or a development evaluation.
    seeds = profile["development_seeds"] if args.mode == "development" else profile["qualification_seeds"]
    gate = profile["navigation_gate"]
    envs, steps = gate["environments"], gate["steps"]
    courses = ["training"] if args.mode == "development" else ["training", "held-out-a", "held-out-b"]
    results = []
    for course in courses:
        for seed in seeds:
            name = f"{course}-{seed}"
            value = execute(output, name, rollout(root, args.variant, envs, steps, 8, seed, 10, policy, course), root, 300)
            validate_native(value, envs=envs, steps=steps, seed=seed, policy=True)
            if value.get("birdflow_navigation_course") != course or value.get("birdflow_navigation_curriculum") is not False:
                raise RuntimeError("Course identity or curriculum isolation failed")
            first = value["first_episode_navigation"]
            results.append({"course": course, "seed": seed, "first_episode": first,
                            "elapsed_seconds": value["elapsed_seconds"],
                            "device_failure_steps": value["failed_environment_steps"]})
    failures = []
    for course in courses:
        group = [row for row in results if row["course"] == course]
        for row in group:
            first = row["first_episode"]
            if first["safe_completion_count"] < gate["minimum_successful_lanes_per_seed"]:
                failures.append(f"{course}/{row['seed']}: completion minimum")
            if first["physical_failure_count"]:
                failures.append(f"{course}/{row['seed']}: first-episode physical failure")
        if sum(row["first_episode"]["safe_completion_count"] for row in group) / (envs * len(group)) < gate["minimum_aggregate_success_fraction"]:
            failures.append(f"{course}: aggregate completion minimum")
    return {"status": "passed" if not failures else "rejected", "mode": args.mode,
            "policy_sha256": sha(policy), "failures": failures, "results": results,
            "boundary": "waypoint navigation only; supported landing and ecological tasks remain separate",
            "ecological_environment_qualified": False}


def learn_probe(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    source = args.source_policy.resolve(strict=True) if args.source_policy else None
    variant = args.variant
    if variant == "v9-visual-neural" and source is None:
        raise ValueError("The v9 learning probe requires an explicitly selected source actor")
    paths = runtime(root)
    retinal = variant == "v13-retinal-feeding"
    environments = args.environments or (16 if retinal else 512)
    visual = args.retinal_visual if retinal else None
    records = []
    for stage, updates in (("initial", 2), ("resume", 1)):
        command = [str(paths["train"]), "--metallib", str(paths["metallib"]),
                   "--native-library", str(paths["library"]), "--mlx-python", sys.executable,
                   "--python-root", str(root / "python"), "--birdflow-american-crow-journey",
                   "--birdflow-journey-variant", variant,
                   "--envs", str(environments), "--steps", "128", "--updates", str(updates), "--chunk", "8",
                   "--minimum-difficulty-band", "0" if variant in ("v12-feeding-station", "v13-retinal-feeding") else "4",
                   "--maximum-difficulty-band", "0" if variant in ("v12-feeding-station", "v13-retinal-feeding") else "4",
                   "--seed", str(profile["probe"]["seed"]),
                   "--learner-seed", str(profile["development_seeds"][0]),
                   "--update-epochs", "1", "--minibatch-size", str(min(8192,environments*128)), "--learning-rate", "0.0001",
                   "--policy-pack", str(output / ("initial.policypack" if stage == "initial" else "initial-updated.policypack")),
                   "--updated-policy-pack", str(output / f"{stage}-updated.policypack"),
                   "--deployment-policy-pack", str(output / f"{stage}-deployment.policypack"),
                   "--incumbent-policy-pack", str(output / f"{stage}-incumbent.policypack"),
                   "--rollout-pack", str(output / f"{stage}.rolloutpack"),
                   "--output-learner-state", str(output / f"{stage}.learner.safetensors")]
        if variant in VARIANTS:
            command += ["--visual-observation-config", str(root / "assets" / VARIANTS[variant])]
        if visual:
            command += ["--visual-observation-config", str(visual)]
        if args.fixed_learning_rate:
            command += ["--fixed-learning-rate"]
        if args.initial_log_standard_deviation is not None:
            command += ["--initial-log-standard-deviation", str(args.initial_log_standard_deviation)]
        if stage == "initial":
            command += ["--initialize-policy", "crow-engineering-learning-probe"]
            if source:
                command += ["--initialize-actor-policy-pack", str(source), "--initialize-actor-fresh-critic"]
            else:
                command += ["--zero-actor-output"]
        else:
            command += ["--learner-state", str(output / "initial.learner.safetensors")]
        value = execute(output, f"learner-{stage}", command, root, 300)
        if value.get("failed_environment_steps") != 0 or value.get("training_samples") != environments * 128 * updates:
            raise RuntimeError("Learner probe invalid sample or device status")
        if value.get("final_policy_revision") != value.get("initial_policy_revision", -1) + updates:
            raise RuntimeError("Learner did not publish every expected policy revision")
        if stage == "resume" and value.get("learner_state_restored") is not True:
            raise RuntimeError("Learner checkpoint was not restored")
        if not (output / f"{stage}.learner.safetensors").is_file():
            raise RuntimeError("Learner checkpoint missing")
        receipt = json.loads((output / f"learner-{stage}.receipt.json").read_text())
        update = value["last_learning_update"]
        records.append({"stage": stage, "updates": updates,
                        "samples": value["training_samples"],
                        "training_loop_steps_per_second": value["end_to_end_environment_steps_per_second"],
                        "whole_process_steps_per_second": value["training_samples"] / receipt["wall_seconds"],
                        "initial_revision": value["initial_policy_revision"],
                        "final_revision": value["final_policy_revision"],
                        "learner_state_restored": value["learner_state_restored"],
                        "checkpoint_sha256": sha(output / f"{stage}.learner.safetensors"),
                        "last_update_mean_reward": update["mean_reward"],
                        "last_update_maximum_reward": update["maximum_reward"],
                        "last_update_intake_utility_mean": update.get("outcomes", {}).get("accepted_intake_utility", {}).get("mean"),
                        "last_update_learning_rate": update["learning_rate"],
                        "last_update_action_standard_deviation": update["mean_action_standard_deviation"],
                        "mlx_peak_memory_bytes": update["mlx_peak_memory_bytes"]})
    if args.require_intake_reward and not any((row["last_update_intake_utility_mean"] or 0)>0 for row in records):
        write(output / "learning-no-intake-reward.json", {"records": records})
        raise RuntimeError("No accepted intake reward in reported learning updates")
    return {"status": "passed", "claim": "native/MLX updates, publication, and learner-only restore",
            "variant": variant, "environments": environments, "actor_initialization": "explicit source" if source else "new zero-output actor",
            "records": records, "candidate_promoted": False, "persistent_world_restore_tested": False,
            "ecological_environment_qualified": False}


def life_swift_probe(root: Path, output: Path, args: argparse.Namespace) -> dict:
    calibration=args.calibration.resolve(strict=True)
    native=calibration/"native-evidence"
    # Bind the native default retention budget (0) and relocate only asset paths.
    # An explicit 512MiB budget has an equivalent bound but a distinct identity.
    visual=json.loads((native/"retinal-observation.json").read_text())
    visual["camera"]["maximum_retained_bytes"]=0
    for pack in visual["packs"]:
        pack["path"]=str((native/pack["path"]).resolve(strict=True))
    write(output/"exact-native-retinal-observation.json",visual)
    paths=runtime(root)
    base=[str(paths["rollout"]),"--metallib",str(paths["metallib"]),"--birdflow-american-crow-journey",
          "--birdflow-journey-variant","v13-retinal-feeding","--envs","2","--steps","32","--repeats","1",
          "--no-scheduled-resets","--minimum-difficulty-band","0","--maximum-difficulty-band","0",
          "--seed","2651109001","--policy-pack",str(native/"published-vision.policypack"),
          "--visual-observation-config",str(output/"exact-native-retinal-observation.json")]
    records=[]
    # Both continuations start at native tick135; reset scheduling is disabled
    # here so this tests ordinary Swift CLI continuation without warmup.
    for chunk in (1,8):
        value=execute(output,f"swift-resume-chunk{chunk}",base+["--chunk",str(chunk),
            "--resume-life-checkpoint",str(native/"saved.lifecheckpoint"),
            "--save-life-checkpoint",str(output/f"continued-chunk{chunk}.lifecheckpoint")],root,120)
        if value.get("failed_environment_steps")!=0 or value.get("life_checkpoint_restored") is not True:
            raise RuntimeError("Swift life continuation failed native admission")
        ledger=value["finite_resource_accounting"]
        if ledger["accepted_control_ticks_by_environment"] != [167,167]:
            raise RuntimeError("Swift introduced a warmup/reset or lost the lifetime clock")
        records.append({"chunk":chunk,"fingerprint":value["policy_rollout_fingerprint"],"ledger":ledger})
    if records[0]["fingerprint"]!=records[1]["fingerprint"] or records[0]["ledger"]!=records[1]["ledger"]:
        raise RuntimeError("Swift life continuation depends on chunk partition")
    return {"status":"passed","claim":"ordinary Swift fresh-process life resume and save; no unreported warmup transition",
            "records":records,"ecological_environment_qualified":False}



def life_training_probe(root: Path, output: Path, args: argparse.Namespace) -> dict:
    """Own complete native/Swift/MLX runs; never a per-step simulator."""
    import struct
    import mlx.core as mx
    import numpy as np

    paths = runtime(root)
    base = [str(paths["train"]), "--metallib", str(paths["metallib"]),
            "--native-library", str(paths["library"]), "--mlx-python", sys.executable,
            "--python-root", str(root/"python"), "--birdflow-american-crow-journey",
            "--birdflow-journey-variant", "v13-retinal-feeding", "--envs", "16", "--steps", "37",
            "--minimum-difficulty-band", "0", "--maximum-difficulty-band", "0",
            "--seed", "2651109101", "--learner-seed", "2651109102", "--update-epochs", "2",
            "--minibatch-size", "97", "--learning-rate", "0.0001", "--initial-log-standard-deviation", "1",
            "--visual-observation-config", str(args.retinal_visual),
            "--checkpoint-life", "--checkpoint-interval", "1"]

    def command(name: str, updates: int, chunk: int, resume: Path | None = None) -> list[str]:
        target = output/name
        target.mkdir()
        result = base + ["--updates", str(updates), "--chunk", str(chunk),
                 "--updated-policy-pack", str(target/"updated.policypack"),
                 "--deployment-policy-pack", str(target/"candidate.policypack"),
                 "--incumbent-policy-pack", str(target/"incumbent.policypack"),
                 "--rollout-pack", str(target/"rollout.rolloutpack"),
                 "--output-learner-state", str(target/"learner.safetensors"),
                 "--checkpoint-directory", str(target/"checkpoints")]
        if resume:
            result += ["--resume-training-checkpoint", str(resume)]
        else:
            result += ["--policy-pack", str(target/"initial.policypack"),
                       "--initialize-policy", "crow-life-training-boundary", "--zero-actor-output"]
        return result

    def snapshot(name: str, revision: int) -> Path:
        return output/name/"checkpoints"/f"revision-{revision:08}.trainingcheckpoint"

    records = []
    source = snapshot("split", 3)
    for name, updates, chunk, resume in [("reference",4,8,None), ("split",2,8,None),
                                        ("resume-chunk1",2,1,source), ("resume-chunk8",2,8,source)]:
        value = execute(output, name, command(name,updates,chunk,resume), root, 180)
        if value.get("failed_environment_steps") != 0 or value.get("training_samples") != 16*37*updates:
            raise RuntimeError("Life-training probe failed native/sample admission")
        if value.get("life_state_restored") is not bool(resume) or (resume and not value.get("learner_state_restored")):
            raise RuntimeError("Life-training probe did not restore both owners")
        if value.get("final_policy_revision") != (5 if name != "split" else 3):
            raise RuntimeError("Life-training probe lost optimizer updates")
        records.append({"stage": name, "training_samples": value["training_samples"],
                        "initial_revision": value["initial_policy_revision"], "final_revision": value["final_policy_revision"],
                        "lifetime_updates_completed": value["lifetime_updates_completed"],
                        "loop_steps_per_second": value["end_to_end_environment_steps_per_second"],
                        "life_state_restored": value["life_state_restored"],
                        "checkpoint_count":value["life_checkpoint_count"], "checkpoint_seconds":value["life_checkpoint_seconds"],
                        "learner_execution_identity": value["learner_execution_identity"]})

    def native_components(path: Path) -> list[bytes]:
        data = path.read_bytes()
        if data[:8] != b"NUMILF01": raise RuntimeError("Missing native life envelope")
        offset = 64 + 5*8  # accepted host counters + telemetry; not simulation state
        result = []
        for _ in range(3):
            count, = struct.unpack_from("<Q", data, offset); offset += 8
            result.append(data[offset:offset+count]); offset += count
        if offset != len(data): raise RuntimeError("Native life envelope length mismatch")
        return result

    comparisons = []
    for name, revisions in [("split", (2,3)), ("resume-chunk1", (4,5)), ("resume-chunk8", (4,5))]:
        for revision in revisions:
            a, b = snapshot("reference",revision), snapshot(name,revision)
            am, bm = [json.loads((p/"manifest.json").read_text()) for p in (a,b)]
            expected_ticks = [37*(revision-1)]*16
            if am["acceptedControlTicks"] != expected_ticks or bm["acceptedControlTicks"] != expected_ticks or am["completedUpdates"] != revision-1 or bm["completedUpdates"] != revision-1:
                raise RuntimeError("Life-training clock/update continuity differs")
            for filename in ("policy.policypack", "candidate.policypack", "rollout.rolloutpack"):
                if sha(a/filename) != sha(b/filename):
                    raise RuntimeError("Life-training future differs: "+name+"/"+str(revision)+"/"+filename)
            aa, ma = mx.load(a/"learner.safetensors", return_metadata=True)
            ab, mb = mx.load(b/"learner.safetensors", return_metadata=True)
            if ma != mb or set(aa) != set(ab): raise RuntimeError("Learner checkpoint metadata/topology differs")
            for key in aa:
                x, y = np.asarray(aa[key]), np.asarray(ab[key])
                if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                    raise RuntimeError("Learner model/Adam state differs: "+key)
            if native_components(a/"life.native") != native_components(b/"life.native"):
                raise RuntimeError("Life-training hidden native/sensory/resource future differs")
            comparisons.append({"stage":name,"revision":revision,"all_payloads_and_optimizer_tensors_bit_identical":True,
                                "accepted_ticks":am["acceptedControlTicks"]})

    # Every rejection uses new output paths. Payload hashes are validated before
    # learner startup; scheduler/worker identities before native life admission.
    negatives = []
    rejection_causes = {
        "uncommitted":"COMMITTED", "bad-marker":"commit marker does not bind its manifest",
        "missing-optimizer":"learner.safetensors", "corrupt-optimizer":"payload mismatch: learner.safetensors",
        "wrong-configuration":"scheduler/learner configuration mismatch",
        "mixed-policy":"Restored learner did not reproduce the exact saved next policy",
        "wrong-worker":"learner revision, optimizer or execution identity mismatch",
        "wrong-scheduler":"scheduler executable identity mismatch",
        "output-overwrite":"Training outputs cannot overwrite the input life checkpoint",
    }
    for name in ("uncommitted", "bad-marker", "missing-optimizer", "corrupt-optimizer", "wrong-configuration", "mixed-policy", "wrong-worker", "wrong-scheduler", "output-overwrite"):
        bad = output/("negative-input-"+name)
        shutil.copytree(source, bad)
        if name == "uncommitted": (bad/"COMMITTED").unlink()
        elif name == "bad-marker": (bad/"COMMITTED").write_text("0"*64)
        elif name == "missing-optimizer": (bad/"learner.safetensors").unlink()
        elif name == "corrupt-optimizer":
            path=bad/"learner.safetensors"; data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
        elif name in ("mixed-policy","wrong-worker","wrong-scheduler"):
            manifest=json.loads((bad/"manifest.json").read_text())
            if name=="mixed-policy":
                path=bad/"policy.policypack";shutil.copyfile(snapshot("split",2)/"policy.policypack",path)
                manifest["files"]["policy.policypack"]={"bytes":path.stat().st_size,"sha256":sha(path)}
            elif name=="wrong-worker": manifest["learnerExecutionIdentity"]="0"*64
            else: manifest["schedulerExecutionIdentity"]="0"*64
            write_path=bad/"manifest.json";write_path.write_text(json.dumps(manifest,sort_keys=True))
            (bad/"COMMITTED").write_text(sha(write_path))
        args_negative=command("negative-"+name,1,8,bad)
        if name=="wrong-configuration":
            args_negative[args_negative.index("--learning-rate")+1]="0.0002"
        elif name=="output-overwrite":
            args_negative[args_negative.index("--output-learner-state")+1]=str(bad/"learner.safetensors")
        try: execute(output,"negative-"+name,args_negative,root,60)
        except RuntimeError:
            receipt=json.loads((output/("negative-"+name+".receipt.json")).read_text())
            if receipt.get("timed_out") or receipt.get("exit_code") in (None,0): raise
            failure = (output/("negative-"+name+".stderr")).read_text()
            if rejection_causes[name] not in failure:
                raise RuntimeError("Negative rejected for unexpected cause: "+name+": "+failure)
            if list((output/("negative-"+name)).rglob("life.native")):
                raise RuntimeError("Rejected training restore published native state")
            negatives.append(name)
        else: raise RuntimeError("Invalid training checkpoint was accepted: "+name)

    food = {}
    for label, checkpoint, ticks in [("split",source,74), ("final",snapshot("reference",5),148)]:
        component = native_components(checkpoint/"life.native")[2]
        states = [struct.unpack_from("<20I",component,64+80*i) for i in range(16)]
        if not all(s[0]+s[9]==45000 and s[8]==s[9] and s[12]+(s[13]<<32)==ticks for s in states):
            raise RuntimeError("Training lifetime resource conservation/clock failed")
        food[label] = {"consumed_micrograms_by_environment":[s[9] for s in states], "accepted_ticks":ticks}
    return {"status":"passed", "claim":"coupled native life and MLX model/Adam exact-runtime training resume",
            "records":records, "boundary_comparisons":comparisons,"rejected_controls":negatives,"finite_resource_checks":food,
            "restore_advances_physics":False,"trainable_memory_qualified":False,"ecological_environment_qualified":False}


def motor_probe(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    policy = args.policy.resolve(strict=True)
    seed = profile["development_seeds"][0]
    rows = []
    for band in (0, 1, 2, 4, 7, 9, 10):
        value = execute(output, f"motor-band{band}", rollout(root, "v9-visual-neural",
                        32, 1600, 8, seed, band, policy), root, 300)
        validate_native(value, envs=32, steps=1600, seed=seed, policy=True)
        if value.get("minimum_sampled_difficulty_band") != band or value.get("maximum_sampled_difficulty_band") != band:
            raise RuntimeError("Motor probe difficulty identity failed")
        rows.append({"band": band, "mean_tracking_score": value["mean_tracking_score"],
                     "physical_terminations": value["termination_count"] - value["timeout_count"],
                     "maximum_root_height_m": value["maximum_root_height"],
                     "maximum_tilt_rad": value["maximum_tilt"],
                     "outcomes": value["outcomes"]})
    return {"status": "measured", "claim": "single-development-seed motor diagnosis only",
            "seed": seed, "policy_sha256": sha(policy), "records": rows,
            "policy_qualified": False, "ecological_environment_qualified": False}


def partition_probe(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    policy = args.policy.resolve(strict=True)
    base = args.base_policy.resolve(strict=True)
    rows = []
    for mode in ("ordinary", "residual"):
        fingerprints = []
        for chunk in (1, 8):
            command = rollout(root, args.variant, 32, 128, chunk,
                              profile["probe"]["seed"], 0 if args.variant=="v12-feeding-station" else 4, policy)
            if mode == "residual":
                command += ["--base-policy-pack", str(base)]
            value = execute(output, f"{mode}-chunk{chunk}", command, root, 180)
            validate_native(value, envs=32, steps=128, seed=profile["probe"]["seed"], policy=True,
                            variant=args.variant)
            fingerprints.append(value["policy_rollout_fingerprint"])
        if fingerprints[0] != fingerprints[1]:
            raise RuntimeError(f"{mode} policy sampling depends on submission partition")
        rows.append({"mode": mode, "fingerprint": fingerprints[0], "chunks": [1, 8]})
    return {"status": "passed", "claim": "ordinary and residual native policy partition invariance",
            "records": rows, "ecological_environment_qualified": False}


def body_admission(root: Path, output: Path, profile: dict) -> dict:
    import numpy as np
    seed = profile["probe"]["seed"]
    rows = []
    final = {}
    for control in ("neutral", "head-beak-command"):
        actions = np.zeros((128, 8, 18), dtype="<f4")
        if control != "neutral":
            actions[16:, :, 15:] = [0.4, -0.3, 0.5]
        stream = output / f"{control}.actions.f32"
        with stream.open("xb") as handle:
            handle.write(actions.tobytes())
        trace = output / f"{control}.state.tsv"
        command = rollout(root, "v11-embodied", 8, 128, 1, seed, 0)
        command.remove("--native-policy")
        command += ["--action-stream", str(stream), "--state-trace", str(trace),
                    "--state-trace-environment", "0"]
        value = execute(output, control, command, root, 180)
        validate_native(value, envs=8, steps=128, seed=seed, policy=False,
                        variant="v11-embodied", action_source="foundation_action_stream")
        if value.get("task") != "american_crow_embodied_development_v11" or value.get("action_count") != 18 or value.get("actor_observation_count") != 93:
            raise RuntimeError("Embodied task/action/proprioception contract mismatch")
        if value.get("termination_count") != value.get("timeout_count"):
            raise RuntimeError("Embodied neutral/command control physically terminated")
        samples = np.loadtxt(trace, comments="#", ndmin=2)
        if samples.shape[0] != 128 or samples[-1,0] != 128 or not np.isfinite(samples).all():
            raise RuntimeError("Incomplete accepted-state trace")
        final[control] = samples[-1,21:24]
        rows.append({"control": control, "action_stream_sha256": sha(stream),
                     "accepted_state_trace_sha256": sha(trace),
                     "final_head_beak_angles_rad": final[control].tolist(),
                     "device_failure_steps": value["failed_environment_steps"],
                     "physical_terminations": value["termination_count"]-value["timeout_count"]})
    delta = final["head-beak-command"] - final["neutral"]
    if not (delta[0] > 0.05 and delta[1] < -0.03 and delta[2] > 0.1):
        raise RuntimeError(f"Accepted head/beak response did not satisfy prospective directional thresholds: {delta}")
    return {"status": "passed", "claim": "accepted native contact-world head/beak actuation control",
            "records": rows, "command_minus_control_angles_rad": delta.tolist(),
            "boundary": "one traced lane, eight environment validity control, 2.56 seconds; no sensing or behavioral qualification",
            "ecological_environment_qualified": False}


def validate_feeding_accounting(value: dict, envs: int, steps: int) -> dict:
    accounting = value.get("finite_resource_accounting", {})
    fields = ("remaining_micrograms", "crop_micrograms", "consumed_micrograms", "bite_count",
              "initialized", "accepted_control_ticks")
    arrays = {field: accounting.get(field + "_by_environment") for field in fields}
    if any(not isinstance(array, list) or len(array) != envs for array in arrays.values()):
        raise RuntimeError("Complete per-lane native finite-resource evidence is required")
    for lane in range(envs):
        row = {field: arrays[field][lane] for field in fields}
        if any(type(item) is not int for item in row.values()):
            raise RuntimeError("Resource evidence must retain integer micrograms and ticks")
        consumed = row["consumed_micrograms"]
        if (consumed not in (0, 20000, 40000, 45000) or row["remaining_micrograms"] + consumed != 45000
            or row["crop_micrograms"] != consumed or row["bite_count"] != {0: 0, 20000: 1, 40000: 2, 45000: 3}[consumed]
            or row["initialized"] != 1 or row["accepted_control_ticks"] != steps + 1):
            raise RuntimeError("Native feeding inventory, crop, bites or accepted lifetime clock are inconsistent")
    utility = sum(arrays["consumed_micrograms"]) * .0001 / envs
    reported = (value.get("mean_reward"), value.get("outcomes", {}).get("accepted_intake_utility", {}).get("mean"))
    if any(not isinstance(mean, (int, float)) or not math.isfinite(mean)
           or abs(mean * steps - utility) > 3e-6 for mean in reported):
        raise RuntimeError("Published food reward/outcome disagrees with accepted physical resource accounting")
    return {"completed_lanes": arrays["consumed_micrograms"].count(45000),
            "environment_count": envs, "complete_fraction": arrays["consumed_micrograms"].count(45000) / envs,
            "mean_consumed_micrograms": sum(arrays["consumed_micrograms"]) / envs,
            "mean_accepted_intake_utility_per_lane": utility}


def feeding_probe(root: Path, output: Path, profile: dict) -> dict:
    import numpy as np
    records = []
    fingerprints = []
    for name, chunk, miss, bites in (("feeding-chunk8", 8, False, True),
                                     ("feeding-chunk64", 64, False, True),
                                     ("head-yaw-miss", 8, True, True),
                                     ("held-closed", 8, False, False)):
        actions = np.zeros((448, 8, 18), dtype="<f4")
        if miss:
            actions[:, :, 15] = 1.0
        if bites:
            for step in range(64, 320):
                if ((step-64)//32) % 2 == 0:
                    actions[step, :, 17] = 0.9
        stream = output / f"{name}.actions.f32"
        with stream.open("xb") as handle:
            handle.write(actions.tobytes())
        command = rollout(root, "v12-feeding-station", 8, 448, chunk, profile["probe"]["seed"], 0)
        command.remove("--native-policy")
        command += ["--action-stream", str(stream)]
        value = execute(output, name, command, root, 180)
        validate_native(value, envs=8, steps=448, seed=profile["probe"]["seed"], policy=False,
                        variant="v12-feeding-station", action_source="foundation_action_stream")
        if (value.get("task") != "american_crow_feeding_station_v12" or
            value.get("actor_observation_count") != 97 or value.get("critic_observation_count") != 97 or
            value.get("action_count") != 18 or value.get("termination_count") != 0):
            raise RuntimeError("Native feeding contract or physical validity failed")
        utility = value["mean_reward"] * 448
        accepted = value["outcomes"]["accepted_intake_utility"]["mean"] * 448
        expected = 4.5 if bites and not miss else 0.0
        if abs(utility-expected)>2e-6 or abs(accepted-expected)>2e-6:
            raise RuntimeError(f"Feeding reward violates finite debit identity: {utility}, {accepted}, {expected}")
        inventory = validate_feeding_accounting(value, 8, 448)
        if inventory["mean_consumed_micrograms"] != (45000 if expected else 0):
            raise RuntimeError("Authored feeding control has the wrong accepted inventory")
        if not miss and bites:
            fingerprints.append(value["policy_rollout_fingerprint"])
        records.append({"control": name, "action_stream_sha256": sha(stream),
                        "accepted_intake_utility_per_lane": utility,
                        "native_outcome_utility_per_lane": accepted,
                        "physical_terminations": value["termination_count"],
                        "failed_physics_steps": value["failed_environment_steps"], "inventory": inventory})
    if len(set(fingerprints)) != 1:
        raise RuntimeError("Feeding observations/rewards depend on submission partition")
    return {"status": "passed", "claim": "native feeding sensor/reward and finite-debit controls",
            "records": records, "nutrition_inputs": 4, "actor_inputs": 97, "actions": 18,
            "behavioral_policy_qualified": False, "persistent_world_restore_tested": False,
            "ecological_environment_qualified": False}


def feeding_policy_probe(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    from .mlx_policy_learning import read_policy_pack
    policy = read_policy_pack(args.policy)
    stochastic = bool(getattr(args, "stochastic", False))
    if bool(policy.action_log_standard_deviation.size) != stochastic:
        raise ValueError("Feeding screen requires the explicitly selected deterministic/stochastic PolicyPack mode")
    envs = getattr(args, "environments", 32)
    rows=[]
    for seed in profile["development_seeds"]:
        value=execute(output,f"seed-{seed}",rollout(root,"v12-feeding-station",envs,1600,8,seed,0,args.policy),root,180)
        validate_native(value,envs=envs,steps=1600,seed=seed,policy=True,variant="v12-feeding-station")
        if value.get("termination_count")!=0:
            raise RuntimeError("Feeding behavior screen requires a complete physically valid first horizon")
        utility=value["mean_reward"]*1600
        if not math.isfinite(utility) or not -1e-6<=utility<=4.5+2e-6:
            raise RuntimeError("Deployed feeding reward exceeds the finite lifetime food budget")
        rows.append({"seed":seed,"mean_accepted_intake_utility_per_lane":utility,
                     "fraction_of_available_food_utility":utility/4.5,
                     "physical_terminations":0,"failed_physics_steps":0,
                     "inventory": validate_feeding_accounting(value, envs, 1600)})
    return {"status":"measured_unqualified","claim":"development-only feeding policy screen",
            "records":rows,"behavioral_policy_qualified":False, "stochastic_policy": stochastic,
            "boundary":"canonical station with per-lane finite inventory; no search, physical food handling or learned memory",
            "ecological_environment_qualified":False}


def feeding_learning_campaign(root: Path, output: Path, profile: dict) -> dict:
    """Bounded development experiment; each process is an independent lifetime.

    Only policy/optimizer state crosses lifetimes. Native resource state is
    never replenished at PPO update boundaries. This is not life-state resume.
    """
    paths = runtime(root)
    if shutil.disk_usage(root).free < 150 * 1024**3:
        raise RuntimeError("Feeding campaign requires 150 GiB free at admission")
    active = capture(["ps", "-axo", "pid,comm"], root)
    if any(line.rsplit("/", 1)[-1] in ("metalrobo_task_train", "metalrobo_task_rollout")
           for line in active.splitlines()):
        raise RuntimeError("Another native training/rollout worker is active")
    write(output / "worker-admission.json", {"processes": active,
          "free_disk_bytes": shutil.disk_usage(root).free})
    started = time.monotonic()
    records, screens = [], []
    previous = None
    for lifetime in range(1, 33):
        if time.monotonic() - started > 1500:
            raise RuntimeError("Campaign reached its prospective 1500-second admission bound")
        if shutil.disk_usage(root).free < 110 * 1024**3:
            raise RuntimeError("Campaign reached its 110 GiB free-space stopping floor")
        directory = output / f"lifetime-{lifetime:02d}"
        directory.mkdir()
        command = [str(paths["train"]), "--metallib", str(paths["metallib"]),
                   "--native-library", str(paths["library"]), "--mlx-python", sys.executable,
                   "--python-root", str(root / "python"), "--birdflow-american-crow-journey",
                   "--birdflow-journey-variant", "v12-feeding-station",
                   "--envs", "512", "--steps", "128", "--updates", "4", "--chunk", "8",
                   "--minimum-difficulty-band", "0", "--maximum-difficulty-band", "0",
                   "--seed", str(2651108100 + lifetime), "--learner-seed", str(2651108200 + lifetime),
                   "--update-epochs", "4", "--minibatch-size", "8192", "--learning-rate", "0.0001",
                   "--fixed-learning-rate", "--initial-log-standard-deviation", "0",
                   "--policy-pack", str(directory / "initial.policypack" if previous is None else previous / "updated.policypack"),
                   "--updated-policy-pack", str(directory / "updated.policypack"),
                   "--deployment-policy-pack", str(directory / "deployment.policypack"),
                   "--incumbent-policy-pack", str(directory / "incumbent.policypack"),
                   "--rollout-pack", str(directory / "final.rolloutpack"),
                   "--output-learner-state", str(directory / "learner.safetensors")]
        if previous is None:
            command += ["--initialize-policy", "crow-feeding-development-campaign", "--zero-actor-output"]
        else:
            command += ["--learner-state", str(previous / "learner.safetensors")]
        write(directory / "input-plan.json", {"command": command, "world_seed": 2651108100 + lifetime,
              "learner_seed": 2651108200 + lifetime, "independent_new_finite_lifetime": True,
              "restored_world": False, "source_learner_sha256": sha(previous / "learner.safetensors") if previous else None,
              "source_policy_sha256": sha(previous / "updated.policypack") if previous else None})
        value = execute(directory, "training", command, root, 90)
        expected_initial_revision = records[-1]["final_revision"] if records else 1
        expected = {"training_samples": 512 * 128 * 4, "failed_environment_steps": 0,
                    "host_requested_resets": 0, "learner_state_restored": previous is not None,
                    "birdflow_journey_variant": "v12-feeding-station", "visual_observation": False,
                    "birdflow_journey_teacher": False, "initial_policy_revision": expected_initial_revision,
                    "final_policy_revision": expected_initial_revision + 4}
        if any(value.get(key) != target for key, target in expected.items()):
            raise RuntimeError("Feeding campaign violated its native learner/lifetime contract")
        history = value.get("learning_update_history", [])
        if len(history) != 4:
            raise RuntimeError("Complete per-update feeding metrics are required")
        for index, update in enumerate(history):
            if (update.get("samples") != 65536 or update.get("done_count") != 0 or update.get("timeout_count") != 0
                or update.get("policy_revision_before") != expected_initial_revision + index
                or update.get("policy_revision_after") != expected_initial_revision + index + 1):
                raise RuntimeError("Feeding update omitted samples, ended a life, or changed revision unexpectedly")
            reward = update.get("mean_reward")
            intake = update.get("outcomes", {}).get("accepted_intake_utility", {}).get("mean")
            if (not isinstance(reward, (int, float)) or not math.isfinite(reward) or reward < 0
                or intake != reward or not 0 <= update.get("maximum_reward", math.inf) <= 2.000002):
                raise RuntimeError("Feeding update has invalid accepted intake reward")
        utility = sum(update["mean_reward"] * 128 for update in history)
        if not 0 <= utility <= 4.500004:
            raise RuntimeError("Training exceeded its finite 45 mg per-lane lifetime budget")
        receipt = json.loads((directory / "training.receipt.json").read_text())
        record = {"lifetime": lifetime, "mean_intake_utility_per_lane": utility,
                  "fraction_of_available_food_utility": utility / 4.5,
                  "initial_revision": expected_initial_revision, "final_revision": value["final_policy_revision"],
                  "samples": value["training_samples"], "whole_process_steps_per_second": value["training_samples"] / receipt["wall_seconds"],
                  "mlx_peak_memory_bytes": max(update["mlx_peak_memory_bytes"] for update in history),
                  "learner_sha256": sha(directory / "learner.safetensors"),
                  "deployment_sha256": sha(directory / "deployment.policypack")}
        write(directory / "analysis.json", record)
        records.append(record)
        print(json.dumps({"feeding_lifetime": lifetime, **record}), file=sys.stderr, flush=True)
        if lifetime == 1:
            baseline = output / "baseline-screen"
            baseline.mkdir()
            result = feeding_policy_probe(root, baseline,
                        argparse.Namespace(policy=directory / "incumbent.policypack"), profile)
            write(baseline / "summary.json", result)
            screens.append({"after_lifetimes": 0, **result})
        if lifetime in (8, 16, 32):
            screen = output / f"development-screen-{lifetime:02d}"
            screen.mkdir()
            result = feeding_policy_probe(root, screen,
                        argparse.Namespace(policy=directory / "deployment.policypack"), profile)
            write(screen / "summary.json", result)
            screens.append({"after_lifetimes": lifetime, **result})
            print(json.dumps({"development_screen": lifetime, "records": result["records"]}), file=sys.stderr, flush=True)
        previous = directory
    final_minimum = min(row["fraction_of_available_food_utility"] for row in screens[-1]["records"])
    return {"status": "measured_unqualified", "claim": "bounded canonical-station PPO development experiment",
            "prediction_supported": final_minimum >= .8, "predicted_minimum_final_food_fraction": .8,
            "final_minimum_development_food_fraction": final_minimum,
            "records": records, "development_screens": screens, "training_samples": 32 * 512 * 128 * 4,
            "world_restore_tested": False, "qualification_conditions_used": False,
            "behavioral_policy_qualified": False, "ecological_environment_qualified": False}


def feeding_policy_comparison(root: Path, output: Path, args: argparse.Namespace, profile: dict) -> dict:
    import numpy as np
    from .mlx_policy_learning import read_policy_pack
    sources = {"initial-deterministic": args.untrained_deployment_policy,
               "initial-stochastic": args.untrained_policy,
               "trained-deterministic": args.trained_deployment_policy,
               "trained-stochastic": args.trained_policy}
    packs = {name: read_policy_pack(path) for name, path in sources.items()}
    for prefix in ("initial", "trained"):
        deterministic, stochastic = (packs[prefix + "-" + mode] for mode in ("deterministic", "stochastic"))
        if deterministic.action_log_standard_deviation.size or stochastic.action_log_standard_deviation.size != 18:
            raise ValueError("Comparison requires explicitly deterministic and stochastic artifacts")
        for key in CONTRACT + ("observation_clip", "action_clip"):
            if getattr(deterministic, key) != getattr(stochastic, key):
                raise ValueError("Policy modes do not share the same exact actor contract")
        for key in ("observation_mean", "observation_inverse_standard_deviation", "action_bias", "action_scale"):
            if not np.array_equal(getattr(deterministic, key), getattr(stochastic, key)):
                raise ValueError("Policy modes changed actor normalization or action transforms")
        if len(deterministic.layers) != len(stochastic.layers):
            raise ValueError("Policy modes changed actor architecture")
        for left, right in zip(deterministic.layers, stochastic.layers):
            if (left.input_count != right.input_count or left.output_count != right.output_count
                or left.activation != right.activation or not np.array_equal(left.weights, right.weights)
                or not np.array_equal(left.bias, right.bias)):
                raise ValueError("Policy modes changed actor coefficients")
    initial = packs["initial-deterministic"]
    if np.any(initial.layers[-1].weights) or np.any(initial.layers[-1].bias):
        raise ValueError("The untrained control must retain the original zero-output actor")
    if any(getattr(initial, key) != getattr(packs["trained-stochastic"], key) for key in CONTRACT):
        raise ValueError("Initial and trained policies do not belong to the same task")
    write(output / "actor-mode-admission.json", {name: {
        "path": str(sources[name].resolve()), "sha256": sha(sources[name]), "revision": pack.revision,
        "stochastic": bool(pack.action_log_standard_deviation.size),
        "standard_deviation": np.exp(pack.action_log_standard_deviation).tolist(),
        "contract": {key: getattr(pack, key) for key in CONTRACT}} for name, pack in packs.items()})
    order = ("initial-deterministic", "trained-stochastic", "initial-stochastic", "trained-deterministic")
    rows = []
    for index, seed in enumerate((2651108401, 2651108402, 2651108403)):
        for name in order[index:] + order[:index]:
            directory = output / f"seed-{seed}" / name
            directory.mkdir(parents=True)
            result = feeding_policy_probe(root, directory,
                argparse.Namespace(policy=sources[name], stochastic=name.endswith("-stochastic"), environments=512),
                profile | {"development_seeds": [seed]})
            row = {"arm": name, **result["records"][0]}
            rows.append(row)
            write(directory / "analysis.json", row)
            print(json.dumps(row), file=sys.stderr, flush=True)
    candidate = [row for row in rows if row["arm"] == "trained-stochastic"]
    null = [row for row in rows if row["arm"] == "initial-stochastic"]
    prediction_supported = (all(row["inventory"]["complete_fraction"] >= .95 for row in candidate)
                            and all(row["inventory"]["complete_fraction"] <= .30 for row in null))
    return {"status": "measured_unqualified", "claim": "frozen-policy canonical feeding comparison",
            "prediction_supported": prediction_supported, "records": rows,
            "native_trajectories": 12 * 512, "accepted_test_transitions": 12 * 512 * 1600,
            "independently_trained_candidates": 1, "behavioral_policy_qualified": False,
            "boundary": "simulation feeding primitive; estimated supported body, fixed intake tracer, no search or solid food manipulation",
            "ecological_environment_qualified": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("doctor")
    commands.add_parser("probe").add_argument("--output", type=Path, required=True)
    commands.add_parser("head-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("eye-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("retina-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("memory-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("recurrent-actor-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("recurrent-policy-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("recurrent-rollout-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("recurrent-ppo-admission").add_argument("--output", type=Path, required=True)
    life_training=commands.add_parser("life-training-probe")
    life_training.add_argument("--output",type=Path,required=True)
    life_training.add_argument("--retinal-calibration",type=Path,required=True)
    life_swift=commands.add_parser("life-swift-probe")
    life_swift.add_argument("--output",type=Path,required=True)
    life_swift.add_argument("--calibration",type=Path,required=True)
    commands.add_parser("life-admission").add_argument("--output",type=Path,required=True)
    commands.add_parser("retinal-api-admission").add_argument("--output",type=Path,required=True)
    commands.add_parser("retinal-live-admission").add_argument("--output",type=Path,required=True)
    retinal_contract=commands.add_parser("retinal-contract-admission")
    retinal_contract.add_argument("--output",type=Path,required=True)
    retinal_contract.add_argument("--legacy-metallib",type=Path)
    commands.add_parser("body-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("foraging-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("cue-foraging-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("memory-choice-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("action-holds-admission").add_argument("--output", type=Path, required=True)
    commands.add_parser("temporal-history-admission").add_argument("--output", type=Path, required=True)
    calibration = commands.add_parser("calibrate-retinal-initialization")
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--policy", type=Path, required=True)
    calibration.add_argument("--rollout", type=Path, required=True)
    calibration.add_argument("--minimum-rgb-standard-deviation", type=float, default=.05)
    prepare_cued = commands.add_parser("prepare-cued-foraging")
    prepare_cued.add_argument("--output", type=Path, required=True)
    prepare_cued.add_argument("--allocation-seed", type=int, required=True)
    prepare_cued.add_argument("--occluded", action="store_true")
    prepare_cued.add_argument("--head-yaw-hold-ticks", type=int, choices=range(1,513), metavar="1..512")
    prepare_cued.add_argument("--beak-hold-ticks", type=int, choices=range(1,513), metavar="1..512")
    prepare_cued.add_argument("--memory-choice-decision-tick", type=int, choices=range(48,193), metavar="48..192")
    commands.add_parser("feeding-probe").add_argument("--output", type=Path, required=True)
    commands.add_parser("feeding-learning-campaign").add_argument("--output", type=Path, required=True)
    feeding_policy=commands.add_parser("feeding-policy-probe")
    feeding_policy.add_argument("--output",type=Path,required=True)
    feeding_policy.add_argument("--policy",type=Path,required=True)
    feeding_policy.add_argument("--stochastic", action="store_true")
    feeding_policy.add_argument("--environments", type=int, choices=(32, 512), default=32)
    comparison = commands.add_parser("feeding-policy-comparison")
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--calibration", type=Path, required=True)
    for name in ("untrained-policy", "untrained-deployment-policy", "trained-policy", "trained-deployment-policy"):
        comparison.add_argument("--" + name, type=Path, required=True)
    init = commands.add_parser("bootstrap")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--source-policy", type=Path, required=True)
    init.add_argument("--variant", choices=VARIANTS, required=True)
    learning = commands.add_parser("learn-probe")
    learning.add_argument("--output", type=Path, required=True)
    learning.add_argument("--source-policy", type=Path)
    learning.add_argument("--variant", choices=("v9-visual-neural", "v11-embodied", "v12-feeding-station", "v13-retinal-feeding"), default="v9-visual-neural")
    learning.add_argument("--initial-log-standard-deviation", type=float)
    learning.add_argument("--fixed-learning-rate", action="store_true")
    learning.add_argument("--require-intake-reward", action="store_true")
    learning.add_argument("--environments", type=int, choices=(16,32,64,128,512))
    learning.add_argument("--retinal-calibration", type=Path)
    motor = commands.add_parser("motor-probe")
    motor.add_argument("--output", type=Path, required=True)
    motor.add_argument("--policy", type=Path, required=True)
    partition = commands.add_parser("partition-probe")
    partition.add_argument("--output", type=Path, required=True)
    partition.add_argument("--policy", type=Path, required=True)
    partition.add_argument("--base-policy", type=Path, required=True)
    partition.add_argument("--variant", choices=("v9-visual-neural", "v11-embodied", "v12-feeding-station"), default="v9-visual-neural")
    assessment = commands.add_parser("evaluate")
    assessment.add_argument("--output", type=Path, required=True)
    assessment.add_argument("--policy", type=Path, required=True)
    assessment.add_argument("--variant", choices=("v10-world-model",), default="v10-world-model")
    assessment.add_argument("--mode", choices=("development", "qualification"), default="development")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    profile = json.loads((root / "profiles/american-crow-rl-v1.json").read_text())
    if args.command == "status":
        print(json.dumps(profile, indent=2))
        return 0
    checked = preflight(root, profile)
    if args.command in ("recurrent-actor-admission", "recurrent-policy-admission", "recurrent-rollout-admission", "recurrent-ppo-admission"):
        build = Path(checked["inputs"]["library"]["path"]).parent.parent
        for name, path in (("recurrent_check", build/"bin/metalrobo_crow_recurrent_actor_check"),
                           ("gru_metallib", build/"shaders/GRUMemory.metallib"),
                           ("policy_abi3_shader", root/"artifacts/crow-rl-recurrent-actor-20261107/inputs/PolicyABI3.metallib")):
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError("Missing recurrent admission input: "+str(path))
            checked["inputs"][name] = {"path": str(path), "sha256": sha(path)}
    if args.command == "memory-admission":
        build = Path(checked["inputs"]["library"]["path"]).parent.parent
        for name, path in (("gru_check", build/"bin/metalrobo_gru_memory_check"),
                           ("gru_metallib", build/"shaders/GRUMemory.metallib")):
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError("Missing native GRU component: "+str(path))
            checked["inputs"][name] = {"path": str(path), "sha256": sha(path)}
    if args.command in ("recurrent-policy-admission", "recurrent-rollout-admission"):
        for name in ("native.py", "mlx_policy_learning.py", "gru_memory.py", "recurrent_policy_admission.py"):
            path=root/"python/metalrobo"/name
            checked["inputs"]["recurrent_source_"+name]={"path": str(path), "sha256": sha(path)}
    if args.command == "recurrent-rollout-admission":
        build=Path(checked["inputs"]["library"]["path"]).parent.parent
        for key,path in (("recurrent_rollout_check",build/"bin/metalrobo_crow_recurrent_rollout_check"),
                         ("recurrent_source_recurrent_rollout_admission.py",root/"python/metalrobo/recurrent_rollout_admission.py")):
            if not path.is_file() or not path.stat().st_size:raise RuntimeError("Missing recurrent rollout admission input: "+str(path))
            checked["inputs"][key]={"path":str(path),"sha256":sha(path)}
    if args.command == "prepare-cued-foraging":
        calibration = root / "environments/american-crow/cued-foraging-v1/calibrated-ports.bin"
        if not calibration.is_file() or calibration.stat().st_size != 32:
            raise ValueError("Missing exact native two-port calibration")
        if not 0 <= args.allocation_seed < 2**64:
            raise ValueError("Allocation seed must be an unsigned64 integer")
        checked["inputs"]["cued_foraging_calibration"] = {"path": str(calibration), "sha256": sha(calibration)}
    if args.command == "doctor":
        print(json.dumps(checked, indent=2))
        return 0
    if args.command=="retinal-contract-admission":
        args.legacy_metallib=(args.legacy_metallib or root/"artifacts/crow-rl-retinal-contract-20261107/inputs/legacy-MetalRobo.metallib").resolve(strict=True)
        checked["inputs"]["retinal_legacy_shader"]={"path":str(args.legacy_metallib),"sha256":sha(args.legacy_metallib)}
    if args.command == "life-training-probe" or (args.command == "learn-probe" and args.variant == "v13-retinal-feeding"):
        if not args.retinal_calibration:
            raise ValueError("Retinal learning requires passed current-runtime API admission")
        calibration = args.retinal_calibration.resolve(strict=True)
        result = json.loads((calibration/"summary.json").read_text())
        calibration_plan = json.loads((calibration/"plan.json").read_text())
        if result.get("status") != "passed" or not result.get("world_reset_preserves_food"):
            raise ValueError("Retinal API admission did not pass")
        for name in ("train","library","metallib","retinal_metallib"):
            if calibration_plan["preflight"]["inputs"][name]["sha256"] != checked["inputs"][name]["sha256"]:
                raise ValueError("Retinal admission does not bind current runtime: "+name)
        args.retinal_visual = calibration/"native-evidence/retinal-observation.json"
        visual = json.loads(args.retinal_visual.read_text())
        for name,path in [("retinal_visual",args.retinal_visual)]+[("retinal_asset_"+str(i),args.retinal_visual.parent/p["path"]) for i,p in enumerate(visual["packs"])]:
            checked["inputs"][name]={"path":str(path),"sha256":sha(path)}
    if args.command == "life-swift-probe":
        calibration=args.calibration.resolve(strict=True)
        admission=json.loads((calibration/"summary.json").read_text())
        calibration_plan=json.loads((calibration/"plan.json").read_text())
        if admission.get("whole_v13_development_life_restore") is not True:
            raise ValueError("Swift life probe requires passed native life admission")
        for name in ("rollout","library","metallib","retinal_metallib"):
            if calibration_plan["preflight"]["inputs"][name]["sha256"] != checked["inputs"][name]["sha256"]:
                raise ValueError("Life calibration does not bind current runtime: "+name)
        for name in ("saved.lifecheckpoint","published-vision.policypack","retinal-observation.json","robot.mrvpack","locomotion_ground.mrvpack"):
            path=calibration/"native-evidence"/name
            checked["inputs"]["life_"+name]={"path":str(path),"sha256":sha(path)}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "numi.crow-trial-plan.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": args.command, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "preflight": checked, "profile": profile,
            "prediction": "Exact same-seed replay and partition invariance; imported historical actor may fail full-route completion",
            "control": "repeat same seed with submission chunk 1 versus 8; first episode only",
            "validity": "native accepted state, exact identity, finite first-episode evidence, no device failures or hidden teacher",
            "stopping": "bounded complete runs, timeout retained as failure; no overwrite or automatic retry"}
    if args.command in ("bootstrap", "learn-probe", "evaluate", "motor-probe", "partition-probe", "feeding-policy-probe"):
        source = args.source_policy if args.command in ("bootstrap", "learn-probe") else args.policy
        plan["source_policy_sha256"] = sha(source.resolve(strict=True)) if source else None
    if args.command == "partition-probe":
        plan["base_policy_sha256"] = sha(args.base_policy.resolve(strict=True))
    if args.command == "feeding-learning-campaign":
        plan.update({"purpose": "development exploration; no qualification claim",
          "question": "Does longer PPO learning produce deterministic feeding instead of reward from random jaw closures?",
          "model_revision": "fresh zero-output MLP, unit exploration, four PPO epochs per update",
          "previous_evidence": "crow-rl-feeding-task-20261107: three updates deploy with zero food despite stochastic reward",
          "prediction": "Final deterministic policy consumes at least 80% of finite food utility on each of three development seeds",
          "refutation": "Any final development seed below 80% contradicts this bounded training prediction",
          "control": "initial zero-output policy, identical canonical station and development seeds, 32 lanes and 1600 steps",
          "unit": "one independent learner trajectory; 32 finite-world training lifetimes are correlated learning exposures, not independent trained agents",
          "budget": {"lifetimes": 32, "envs": 512, "steps_per_update": 128, "updates_per_lifetime": 4,
                     "total_transitions": 8388608, "world_seeds": [2651108101, 2651108132],
                     "learner_seeds": [2651108201, 2651108232], "food_mg_per_lane_lifetime": 45},
          "analysis": "all lifetime reward sums constrained to [0,4.5] utility/lane; report all screens, no inferential population interval",
          "validity": "zero device failures, physical terminations and resets; complete update history/revisions; native finite ledger; no teacher",
          "stopping": "exactly 32 lifetimes; no success-driven extension; stop on invalidity, per-process 90s timeout, 1500s admission bound or free space below 110GiB",
          "unused": "qualification seeds/layouts remain untouched; future independent learner replicates required"})
    if args.command == "feeding-policy-comparison":
        calibration = json.loads((args.calibration / "summary.json").read_text())
        calibration_plan = json.loads((args.calibration / "plan.json").read_text())
        if calibration.get("status") != "passed" or len(calibration.get("records", [])) != 4:
            raise ValueError("Feeding comparison requires passed native inventory controls")
        for name in ("rollout", "library", "metallib"):
            if calibration_plan["preflight"]["inputs"][name]["sha256"] != checked["inputs"][name]["sha256"]:
                raise ValueError("Feeding calibration does not bind the current native runtime")
        plan.update({"purpose": "development mechanism comparison on unused seed identities; one frozen trained candidate",
          "parent_attempt": "20261107-feeding-learning-campaign-01: interrupted invalid deterministic baseline; eight completed learning lifetimes",
          "source_policies": {name: {"path": str(getattr(args, name).resolve(strict=True)), "sha256": sha(getattr(args, name))}
                              for name in ("untrained_policy", "untrained_deployment_policy", "trained_policy", "trained_deployment_policy")},
          "calibration": {"path": str(args.calibration.resolve()), "summary_sha256": sha(args.calibration / "summary.json"),
                          "plan_sha256": sha(args.calibration / "plan.json")},
          "question": "Has learning improved stochastic feeding reliability, or is positive intake explained by untrained exploration alone?",
          "prediction": "Trained stochastic actor depletes all 45 mg in at least 95% of lanes on EACH seed; initial stochastic actor completes at most 30%",
          "refutation": "Any seed outside either interval contradicts the prediction; deterministic modes are separately reported diagnostic controls",
          "control": "initial and trained actors each tested with explicit deterministic and stochastic artifacts; each mode pair has bit-identical actor coefficients",
          "unit": "6,144 trajectories, 512 lanes per arm/seed; one trained agent, not 6,144 independent learned animals",
          "conditions": {"seeds": [2651108401,2651108402,2651108403], "steps":1600, "dt_seconds":.02,
                         "food_micrograms_per_lane":45000, "arms":4, "additional_learning_updates":0},
          "order": "per-seed cyclic rotation of initial deterministic, trained stochastic, initial stochastic, trained deterministic",
          "analysis": "per-lane integer inventory/crop/bites/clock conservation and reward agreement; all per-seed completion fractions and mass, no population generalization",
          "validity": "passed native positive/null/partition inventory calibration, exact actor-mode identities, zero device errors/physical ends/resets",
          "stopping": "exactly twelve native runs; 180s timeout per run; abort invalidity; no retry, extra learning, or promotion",
          "limitations": "policy revision changes random key; same seed identifies matched world conditions but does not create identical stochastic noise across policy revisions; learned variance is part of the tested policy",
          "unused": "original qualification seeds/layouts remain unspent"})
    if args.command == "retinal-contract-admission":
        plan.update({"purpose":"native retinal actor contract and pre-submission rejection instrument",
          "prediction":"1251 actor/97 critic/18 action channels; 12 malformed manifests and 9 same-dimension sensor changes reject; missing observer and old shader cannot execute; corrected base/retinal ABI51/52 with mechanics unchanged",
          "boundary":"compiler metadata and native rejection probes only; no rendered actor images or learner integration"})
    if args.command == "learn-probe" and args.variant == "v13-retinal-feeding":
        plan.update({"purpose":"native RGB to Swift/MLX learner and learner-only restore engineering admission",
          "prediction":"two initial updates and one restored update publish all expected revisions and samples with zero native/sensor errors",
          "control":"current-runtime terminal/partition/reset API admission; two fresh-world learner stages, not whole-world restore",
          "unit":"one learner, explicit supported-station proxy scene; images need not be behaviorally useful for this engineering test",
          "budget":{"environments":args.environments or 16,"steps_per_update":128,"updates":3},
          "stopping":"exactly two processes, 300s each; abort invalidity, preserve all failures, no promotion",
          "limitations":"not visual learning, memory, ecological intelligence or production-scale performance qualification"})
    if args.command == "life-swift-probe":
        plan.update({"purpose":"regular Swift CLI life continuation admission",
          "prediction":"chunk1/8 future rollout fingerprints and resource ledgers equal; lifetime ticks135→167 with exactly32 additional transitions, no warmup",
          "control":"two new processes resume the same admitted native state and exact policy/configuration",
          "unit":"two lanes, one authored stochastic controller; exact replay engineering test, no learning claim",
          "stopping":"two32-step continuations,120s timeout each; abort invalidity, no automatic retry"})
    if args.command == "life-training-probe":
        plan.update({"purpose":"coupled v13 lifelong training checkpoint admission", "model_revision":"v5:01 planning and02 CLI errors rejected before creation;03 rejected premature telemetry before a completed resident submission;04 passed all original comparisons/seven negatives; v5 additionally validates exact expected lifetime clocks/food and wrong-scheduler/output-overwrite rejection; bind actual imported learner closure rather than unrelated orchestration files",
          "question":"Does uninterrupted4-update training equal2+2 across independent processes and chunk8→1?",
          "prediction":"all generating rollouts, policy/candidate bytes, every model/Adam tensor and three native hidden components match at every shared optimizer boundary",
          "conditions":{"environments":16,"steps_per_update":37,"epochs":2,"minibatch":97,"reference_updates":4,"split_updates":[2,2],"resume_chunks":[1,8],"development_seeds":[2651109101,2651109102]},
          "unit":"one development learner trajectory with deterministic replay controls; not independent animals",
          "validity":"exact runtime/configuration/source inputs; both owners restored; accepted clock has no warmup; zero device failures; all expected updates; uncommitted/corrupt/mixed checkpoints reject",
          "stopping":"four complete bounded180s runs; nine bounded60s rejection controls; stop on first positive invalidity; preserve all failures",
          "estimated_raw_output_limit_bytes":750*1024*1024, "ecological_environment_qualified":False})
    if args.command in ("recurrent-actor-admission", "recurrent-policy-admission", "recurrent-rollout-admission"):
        plan.update({"purpose": "native embodied recurrent actor and full-life checkpoint engineering admission",
          "prediction": "FP64 actor/pre-cell max error<=5e-6; chunks1/8 and extra intermediate terminal previews have exact real traces/actions/physics; fresh process has exact future and all three final life-state component blobs; seven compiler and five resident publication controls reject including reused/reversed generating revisions; stale policyABI3 shader rejects by ABI4 or missing corrected temporal-history kernel; independently retained runtime reproduces all future stream bytes",
          "control": "two supported real native v13 lanes, authored encoder5-GRU7-action18 actor,40 ticks; lane1 reset9; next weights published18 without physical step; aligned final camera capture in all arms; no qualification seeds",
          "unit": "one authored neural/controller engineering instrument, not a learned-memory experiment",
          "stopping": "one native admission180s including bounded40s child; one retained-runtime continuation40s; preserve failures, no automatic retries",
          "limitations": "legacy v7 learner publication explicitly forbidden for recurrent actors; sequence PPO, recurrent training checkpoint and embodied memory/social learning qualification remain open"})
    if args.command == "recurrent-policy-admission":
        plan.update({"purpose": "native recurrent actor, canonical learner policy interoperability and retained source recovery",
          "prediction": "native actor admission passes; two recurrent and one dense policy roundtrip byte exactly; MLX sequence actor/pre-cell error<=5e-6 on all80 native slots;17 semantic/API/flat-learner misuse controls reject; retained source/runtime reproduce learner summary, all sequence arrays and canonical policy bytes exactly",
          "stopping": "native180s and retained native40s; two independent MLX policy instruments180s each; stop on invalidity, no automatic retries",
          "limitations": "policy parameter/sequence algebra interface only; recurrent rollout and contiguous PPO remain pending"})
    if args.command == "recurrent-rollout-admission":
        plan.update({"purpose":"native/Swift/MLX recurrent rollout8 handoff with retained runtime/source recovery",
          "prediction":"native actor/policy interface admissions pass; two constant-policy segments18/22 exclude previews; native format8 byte roundtrip and10 malformed memory descriptors plus strict resident generating-revision publication guards; Swift concat chunk1 and training append chunk8 match every canonical byte/final q/v with10 rejections; Python parses all learner/hidden/reset arrays exactly and MLX sequence parity<=5e-6 on80 real slots;11 parser/flat-use controls reject; independent retained Swift/learner summaries and all sequence arrays exact",
          "conditions":"two supported real native v13 lanes,40ticks, lane1 reset9, next weights18 with no physical step; both segment endpoint captures aligned; native extra intermediate previews; additional unused development seed2651109501 via regular CLI:37ticks then13 fresh-process continuation at chunk1/8 with exact canonical bytes and accepted clocks37/50, no warmup; zeroing continuation hidden state changes native-control action mean by>=1e-4; no qualification seeds",
          "stopping":"native180s,retained native40s; eight bounded180s batch instruments (two policy, two Swift, two rollout learner, two CLI-sequence), five ordinary CLI120s processes; abort first invalidity, preserve failures, no automatic retry",
          "limitations":"native collection/publication and forward-sequence algebra only; ordinary native life resume included, but contiguous PPO/BPTT and coupled recurrent optimizer/life resume remain pending"})
    if args.command == "memory-admission":
        plan.update({"purpose": "native working-memory and differentiable sequence algebra component admission",
          "prediction": "native FP64 max error<=3e-6; chunks/previews/duplicates/fresh process exact including reset traces;31 native controls reject including seven re-signed semantic restore corruptions; MLX sequence/pre-cell parity<=3e-6; full parameter/input BPTT<=3e-6 and relative norm<=2e-5; twelve FP64 central differences<=1e-8; reset gradients cut exactly; eight SGD steps reduce fixed objective; retained native/Python-source recovery reproduces future state and learner summary",
          "control": "two native shapes including nonaligned257/65,40 authored neural ticks; lane resets11/23; weight publication18; two chunks1/8; independent process; six-tick MLX/FP64 gradient fixture and four learner contract rejections",
          "unit": "declared neural algebra/timing instrument; host-authored clock is not a crow simulator",
          "stopping": "one native and two independent learner instruments,180s timeout each; one retained native continuation,40s timeout; no automatic retry or threshold changes",
          "limitations": "PolicyPack, native actor integration, recurrent rollout/sequence PPO, coupled memory checkpoint and behavioral ablations still required"})
    if args.command == "life-admission":
        plan.update({"purpose":"v13 native life-state persistence admission across fresh processes",
          "prediction":"future streams and complete native/retinal/resource state match uninterrupted execution bit for bit at chunk1/8; ten invalid restores reject without publishing mutation",
          "unit":"two supported development lanes and one authored stochastic image-dependent actor; not a learned memory experiment",
          "control":"pending retinal capture, already consumed food, native contact/task/RNG history and lane1 reset after restore",
          "stopping":"one native test; 180s timeout; two child continuations each bounded40s; preserve every failure, no automatic retry",
          "limitations":"v13 supported development fixture; brain optimizer is a separate exact input, trainable memory/social/ecology remain open"})
    if args.command == "retinal-api-admission":
        plan.update({"purpose":"native v13 C API, terminal learning slot and reset admission",
          "prediction":"six malformed configs reject; final/next retinal inputs and chunk1/8 bit identical; fresh world reset clears retina while preserving consumed food",
          "boundary":"explicit fixed development collision proxies and authored actors; no visual learning or biological claim"})
    if args.command == "retinal-live-admission":
        plan.update({"purpose":"live native physical cameras to GPU retina to pixel-dependent actor",
          "prediction":"two ticks cold-start blindness; actual occlusion reverses pixel-driven yaw; disconnected actor stays zero; chunk1/8 bit identical",
          "boundary":"authored development actor; no learned vision, biological optics, sensory checkpoint restore or ecological qualification"})
    if args.command == "retina-admission":
        plan.update({"purpose": "native RGB timing instrument calibration",
          "prediction": "integer 3/10 sample cadence, two accepted-tick delay; zero premature/reset-stale frames; feature error <=3e-6 against FP64 oracle; chunk1/8 bit-identical",
          "control": "three native fixture lanes, one per-lane episode reset, duplicate accepted tick with altered RGB, clock-gap and nonfinite negatives",
          "boundary": "synthetic device-input instrument only; no renderer-to-actor or learner integration; existing task semantics unchanged"})
    if args.command == "eye-admission":
        plan.update({"purpose": "native geometric instrument calibration",
          "prediction": "Accepted physical head transforms determine both camera poses within 2e-6 components; stable interior depth matches FP64 physical-shape rays within 0.0002m; zero identity errors",
          "control": "visible/physically occluded target, opposite head yaw, head pitch, exact renderer repeats, native chunks 1/8",
          "boundary": "estimated parallel head cameras with collision-proxy assets; publication-time inspection; no biological eye calibration, actor image integration or sensor-cadence admission"})
    write(output / "plan.json", plan)
    try:
        retain_source_delta(root, output, checked)
        if args.command == "probe":
            result = probe(root, output, profile)
        elif args.command == "head-admission":
            result = execute(output, "head-mechanics", [str(runtime(root)["head_check"])], root, 180)
            if result.get("status") != "passed" or result.get("exact_replay") is not True:
                raise RuntimeError("Native head mechanics admission failed")
        elif args.command == "retinal-contract-admission":
            result=execute(output,"retinal-contract",[str(runtime(root)["retinal_contract_check"]),
                           str(output/"native-evidence"),str(args.legacy_metallib)],root,180)
            if result.get("status")!="passed" or result.get("stale_shader_rejected") is not True:
                raise RuntimeError("Native retinal contract admission failed")
        elif args.command == "life-training-probe":
            result=life_training_probe(root,output,args)
        elif args.command == "life-swift-probe":
            result=life_swift_probe(root,output,args)
        elif args.command == "recurrent-ppo-admission":
            from .recurrent_ppo_admission import run as run_recurrent_ppo
            proof=output/"ppo-evidence"
            proof.mkdir()
            result=run_recurrent_ppo(root,proof)
        elif args.command == "recurrent-rollout-admission":
            result=recurrent_rollout_admission(root,output,checked)
        elif args.command == "recurrent-policy-admission":
            result=recurrent_policy_admission(root,output,checked)
        elif args.command == "recurrent-actor-admission":
            result=recurrent_actor_admission(root,output,checked)
        elif args.command == "memory-admission":
            result=memory_admission(root,output,checked)
        elif args.command == "life-admission":
            result=execute(output,"life-checkpoint",[str(runtime(root)["life_check"]),str(output/"native-evidence")],root,180)
            if result.get("status")!="passed" or result.get("whole_v13_development_life_restore") is not True:
                raise RuntimeError("Native crow life checkpoint admission failed")
        elif args.command == "retinal-api-admission":
            result=execute(output,"retinal-api",[str(runtime(root)["retinal_api_check"]),
                           str(output/"native-evidence")],root,180)
            if result.get("status")!="passed" or result.get("world_reset_preserves_food") is not True:
                raise RuntimeError("Native retinal learner API admission failed")
        elif args.command == "retinal-live-admission":
            result=execute(output,"retinal-live",[str(runtime(root)["retinal_live_check"]),
                           str(output/"native-evidence")],root,180)
            if result.get("status")!="passed" or result.get("chunk_bit_identical") is not True:
                raise RuntimeError("Native live retinal actor admission failed")
        elif args.command == "retina-admission":
            result = execute(output, "retinal-timing", [str(runtime(root)["retinal_check"]),
                             str(output / "native-evidence")], root, 180)
            if result.get("status") != "passed" or result.get("exact_partition_invariance") is not True:
                raise RuntimeError("Native retinal timing admission failed")
        elif args.command == "eye-admission":
            result = execute(output, "head-camera-geometry", [str(runtime(root)["eye_check"]),
                             str(output / "native-evidence")], root, 180)
            if result.get("status") != "passed" or result.get("geometry_and_camera_pose_admitted") is not True:
                raise RuntimeError("Native head camera geometry admission failed")
        elif args.command == "body-admission":
            result = body_admission(root, output, profile)
        elif args.command == "feeding-probe":
            result = feeding_probe(root, output, profile)
        elif args.command == "feeding-policy-probe":
            result = feeding_policy_probe(root,output,args,profile)
        elif args.command == "feeding-learning-campaign":
            result = feeding_learning_campaign(root,output,profile)
        elif args.command == "feeding-policy-comparison":
            result = feeding_policy_comparison(root,output,args,profile)
        elif args.command == "foraging-admission":
            result = execute(output, "foraging-ledger", [str(runtime(root)["foraging_check"]),
                             str(output / "native-evidence")], root, 180)
            if result.get("status") != "passed" or result.get("exact_partition_replay") is not True:
                raise RuntimeError("Native finite-resource admission failed")
        elif args.command == "calibrate-retinal-initialization":
            from .crow_retinal_normalization import calibrate_initial_policy
            result = calibrate_initial_policy(args.policy.resolve(strict=True), args.rollout.resolve(strict=True), output,
                minimum_standard_deviation=args.minimum_rgb_standard_deviation)
        elif args.command == "prepare-cued-foraging":
            if not 0 <= args.allocation_seed < 2**64:
                raise ValueError("Allocation seed must be an unsigned64 integer")
            holds = (args.head_yaw_hold_ticks, args.beak_hold_ticks)
            if any(x is not None for x in holds) and (any(x is None for x in holds) or args.memory_choice_decision_tick is None):
                raise ValueError("Native holds require both periods and a memory-choice decision tick")
            calibration = root / "environments/american-crow/cued-foraging-v1/calibrated-ports.bin"
            result = execute(output, "compile-cued-foraging", [str(runtime(root)["foraging_compile"]),
                             str(calibration), str(output / "environment"), str(args.allocation_seed),
                             "1" if args.occluded else "0"] +
                             ([str(args.memory_choice_decision_tick)] if args.memory_choice_decision_tick is not None else []) +
                             ([str(x) for x in holds] if holds[0] is not None else []), root, 60)
            if result.get("status") != "passed" or result.get("simulation_steps") != 0:
                raise RuntimeError("Native cue environment authoring failed")
        elif args.command == "temporal-history-admission":
            execute(output, "temporal-history", [str(runtime(root)["temporal_history_check"]),
                    str(output / "native-evidence"),
                    str(root / "environments/american-crow/cued-foraging-v1/calibrated-ports.bin")],
                    root, 600, json_output=False)
            result = json.loads((output / "native-evidence/summary.json").read_text())
            if result.get("status") != "passed" or result.get("history_float_checks") != 13968 or result.get("full_checkpoint_and_stream_partition_exact") is not True:
                raise RuntimeError("Native temporal-history ownership/partition admission failed")
        elif args.command == "action-holds-admission":
            result = execute(output, "action-holds", [str(runtime(root)["cue_foraging_check"]),
                             str(output / "native-evidence"), "--action-holds"], root, 1200)
            if result.get("status") != "passed" or result.get("head_yaw_hold_ticks") != 256 or result.get("beak_hold_ticks") != 16:
                raise RuntimeError("Native held-action program/reward-path admission failed")
        elif args.command == "memory-choice-admission":
            result = execute(output, "memory-choice", [str(runtime(root)["cue_foraging_check"]),
                             str(output / "native-evidence"), "--memory-choice"], root, 1200)
            if result.get("status") != "passed" or result.get("memory_choice_decision_tick") != 64:
                raise RuntimeError("Native memory-choice program/reward-path admission failed")
        elif args.command == "cue-foraging-admission":
            result = execute(output, "cue-foraging", [str(runtime(root)["cue_foraging_check"]),
                             str(output / "native-evidence")], root, 1200)
            if result.get("status") != "passed" or result.get("native_partition_byte_exact") is not True:
                raise RuntimeError("Native rendered-cue and physical food-port admission failed")
        elif args.command == "bootstrap":
            result = bootstrap(root, output, args, profile)
        elif args.command == "learn-probe":
            result = learn_probe(root, output, args, profile)
        elif args.command == "motor-probe":
            result = motor_probe(root, output, args, profile)
        elif args.command == "partition-probe":
            result = partition_probe(root, output, args, profile)
        else:
            result = evaluate(root, output, args, profile)
        write(output / "summary.json", result)
        print(json.dumps({"output": str(output), **result}, indent=2))
        return 0 if result["status"] != "rejected" else 2
    except (Exception, KeyboardInterrupt) as error:
        write(output / "failure.json", {"error": str(error), "type": type(error).__name__})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
