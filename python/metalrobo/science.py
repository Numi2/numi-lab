"""Transparent scientific notebooks over owner executables; no scheduler or simulator.

Only the Python standard library is required. All writes are exclusive or atomic;
failed trials are evidence, never silently retried or excluded from analysis.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time


class Invalid(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(Invalid(value)))


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    # Linking a fully flushed temporary prevents readers seeing half a record.
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".science-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def seal(path, payload):
    value = {"sha256": digest(payload), "payload": payload}
    write_new(path, value)
    return value


def unseal(path):
    record = read(path)
    require(record["sha256"] == digest(record["payload"]), "record changed: " + str(path))
    return record


def text_field(value, name):
    require(isinstance(value.get(name), str) and value[name].strip(), "required text: " + name)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def at(value, path):
    for key in path:
        value = value[key]
    return value


def json_path(path):
    require(isinstance(path, list) and path and
            all(type(key) in (str, int) for key in path), "JSON path must be a nonempty key/index array")


def bounds(value):
    require(finite(value.get("minimum")) and finite(value.get("maximum")) and
            value["minimum"] <= value["maximum"], "invalid finite prediction bounds")


def validate(plan):
    require(plan["schema"] == "numi.science.plan.v1", "unsupported plan schema")
    for key in ("question", "hypothesis", "owner", "backend", "limitations"):
        text_field(plan, key)
    require(plan["evidence_level"] in ("software", "reference", "simulation"),
            "this local runner accepts software, reference, or simulation studies only; use owner arming for hardware")
    for key in ("statement", "version"):
        text_field(plan["model"], key)
    for key in ("description", "calibration"):
        text_field(plan["instrument"], key)
    for key in ("intervention", "controls", "experimental_unit", "allocation"):
        text_field(plan["design"], key)
    observable = plan["observable"]
    for key in ("name", "unit"):
        text_field(observable, key)
    json_path(observable["path"])
    require(plan["prediction"]["estimand"] == "paired_difference_mean",
            "v1 analysis is the mean paired treatment-minus-control difference")
    bounds(plan["prediction"])
    require(isinstance(plan["validity"], list) and plan["validity"], "declare observable validity gates")
    for gate in plan["validity"]:
        json_path(gate["path"])
        require(set(gate) == {"path", "equals"}, "validity gates require path and equals")
    require(isinstance(plan["paired_equal"], list), "paired_equal must list matched JSON paths")
    for path in plan["paired_equal"]:
        json_path(path)
    require(isinstance(plan["artifacts"], list) and plan["artifacts"],
            "bind instrument, calibration, model, runtime and input artifacts")
    require(isinstance(plan["trials"], list) and plan["trials"], "declare trials in execution order")
    ids, pairs = set(), {}
    for trial in plan["trials"]:
        identity = trial["id"]
        require(isinstance(identity, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", identity)
                and identity not in ids, "invalid or duplicate trial id")
        ids.add(identity)
        text_field(trial, "pair")
        require(trial["arm"] in ("control", "treatment"), "invalid trial arm")
        arms = pairs.setdefault(trial["pair"], set())
        require(trial["arm"] not in arms, "duplicate arm in pair")
        arms.add(trial["arm"])
        argv = trial["argv"]
        require(isinstance(argv, list) and argv and
                all(isinstance(arg, str) and "\x00" not in arg for arg in argv), "argv must be a string array")
        executable = Path(argv[0])
        require(executable.is_absolute() and executable.is_file() and os.access(executable, os.X_OK),
                "use an absolute executable path: " + str(executable))
        require(finite(trial["timeout_seconds"]) and 0 < trial["timeout_seconds"] <= 86400,
                "declare a finite trial timeout of at most one day")
        require(isinstance(trial["env"], dict) and all(
            isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) and
            isinstance(value, str) and "\x00" not in value for key, value in trial["env"].items()),
            "env must contain explicit string values; never include credentials")
    require(all(arms == {"control", "treatment"} for arms in pairs.values()), "unpaired study")
    return plan


def git_identity(repository):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()
    return {"repository": str(Path(repository).resolve()), "revision": git("rev-parse", "HEAD"),
            "status": git("status", "--porcelain=v1", "--untracked-files=normal"),
            "diff_sha256": hashlib.sha256(subprocess.check_output(
                ["git", "-C", str(repository), "diff", "HEAD", "--binary"])).hexdigest()}


def register(plan_path, study, parent=None):
    plan = validate(read(plan_path))
    artifacts = {}
    for raw in plan["artifacts"] + [str(Path(__file__).resolve())] + [trial["argv"][0] for trial in plan["trials"]]:
        require(isinstance(raw, str) and Path(raw).is_absolute(), "artifacts require absolute file paths")
        # Keep the invoked path so retargeting a symlink is detected before execution.
        path = Path(raw)
        require(path.is_file(), "artifact is not a file: " + str(path))
        artifacts[str(path)] = file_hash(path)
    parent_identity = None
    if parent:
        parent = Path(parent).resolve()
        verify(parent)
        revision = unseal(parent / "revision.json")
        require(plan["model"] == revision["payload"]["model"], "follow-up must test the revised model")
        parent_identity = {"study": str(parent), "revision_sha256": revision["sha256"]}
    registration = {"schema": "numi.science.registration.v1", "registered_at": now(), "plan": plan,
                    "artifacts": artifacts, "source": git_identity(plan["repository"]),
                    "host": {"node": platform.node(), "platform": platform.platform(), "machine": platform.machine()},
                    "parent": parent_identity}
    study = Path(study).resolve()
    study.mkdir(parents=True, exist_ok=False)
    (study / "trials").mkdir()
    return seal(study / "registration.json", registration)


@contextmanager
def locked(study):
    # Kernel lock, not a stale PID file. Held through the native process wait.
    with (study / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Invalid("study is in use; inspect its live process before acting")
        yield


def check_artifacts(registration):
    for path, expected in registration["payload"]["artifacts"].items():
        require(file_hash(path) == expected, "bound artifact changed: " + path)


def tree_hashes(root):
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "trial output contains a symlink: " + str(path))
        if path.is_file():
            result[str(path.relative_to(root))] = file_hash(path)
    return result


def verify(study):
    study = Path(study).resolve()
    registration = unseal(study / "registration.json")
    plan = registration["payload"]["plan"]
    declared_ids = {trial["id"] for trial in plan["trials"]}
    require(all(path.name in declared_ids for path in (study / "trials").iterdir()),
            "undeclared trial directory found")
    receipts = {}
    for trial in plan["trials"]:
        directory = study / "trials" / trial["id"]
        if not directory.exists():
            continue
        started = unseal(directory / "started.json")
        require(started["payload"]["registration_sha256"] == registration["sha256"] and
                started["payload"]["trial"] == trial, "trial does not match registration")
        # Old records without a cwd used the trial directory; new archives can move.
        original_output = started["payload"].get("cwd", str(directory / "output"))
        expected_argv = [arg.replace("{run}", original_output) for arg in trial["argv"]]
        require(started["payload"]["argv"] == expected_argv, "trial argv mismatch")
        if not (directory / "receipt.json").exists():
            continue
        receipt = unseal(directory / "receipt.json")
        require(receipt["payload"]["started_sha256"] == started["sha256"], "receipt start mismatch")
        require(receipt["payload"]["files"] == tree_hashes(directory / "output"), "trial evidence changed")
        receipts[trial["id"]] = receipt
    if (study / "analysis.json").exists():
        analysis = unseal(study / "analysis.json")
        require(analysis["payload"] == compute_analysis(registration, receipts, study), "analysis does not match raw evidence")
    if (study / "revision.json").exists():
        revision = unseal(study / "revision.json")
        require(revision["payload"]["analysis_sha256"] == unseal(study / "analysis.json")["sha256"],
                "model revision refers to another analysis")
        require(revision["payload"]["previous_model"] == plan["model"] and
                revision["payload"]["evidence"] == [revision["payload"]["analysis_sha256"]],
                "model revision lineage mismatch")
    if (study / "stopped.json").exists():
        stopped = unseal(study / "stopped.json")
        require(stopped["payload"]["registration_sha256"] == registration["sha256"] and
                stopped["payload"]["receipts"] == {key: value["sha256"] for key, value in receipts.items()},
                "stopped study changed")
    return registration, receipts


def run_next(study):
    study = Path(study).resolve()
    with locked(study):
        registration, receipts = verify(study)
        require(not (study / "analysis.json").exists(), "study already analyzed; preregister a follow-up")
        require(not (study / "stopped.json").exists(), "study stopped; preregister an amended study")
        plan = registration["payload"]["plan"]
        trial = next((trial for trial in plan["trials"] if trial["id"] not in receipts), None)
        require(trial is not None, "all declared trials already recorded")
        check_artifacts(registration)
        directory = study / "trials" / trial["id"]
        require(not directory.exists(), "unfinished trial; inspect its process and artifacts; never blindly restart it")
        directory.mkdir()
        output = directory / "output"
        output.mkdir()
        argv = [arg.replace("{run}", str(output)) for arg in trial["argv"]]
        environment = {"PATH": os.defpath, "HOME": str(Path.home()), "TMPDIR": tempfile.gettempdir()}
        environment.update(trial["env"])
        started = seal(directory / "started.json", {
            "registration_sha256": registration["sha256"], "trial": trial, "argv": argv,
            "environment": environment, "cwd": str(output), "started_at": now(),
            "host": platform.node(), "runner_pid": os.getpid()})
        begin = time.monotonic()
        failure, returncode = None, None
        process = None
        with (output / "stdout.json").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
            try:
                process = subprocess.Popen(argv, cwd=output, env=environment, stdout=stdout,
                                           stderr=stderr, start_new_session=True)
                seal(directory / "process.json", {"pid": process.pid, "host": platform.node(),
                                                  "started_sha256": started["sha256"]})
                returncode = process.wait(timeout=trial["timeout_seconds"])
                if returncode:
                    failure = "process_exit_" + str(returncode)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                failure = type(error).__name__
                if process:
                    os.killpg(process.pid, signal.SIGKILL)
                    returncode = process.wait()
            except OSError as error:
                failure = str(error)
        try:
            check_artifacts(registration)
        except (Invalid, OSError) as error:
            failure = "artifact drift during trial: " + str(error)
        result = seal(directory / "receipt.json", {
            "started_sha256": started["sha256"], "ended_at": now(),
            "elapsed_seconds": time.monotonic() - begin, "returncode": returncode, "failure": failure,
            "files": tree_hashes(output)})
        return result


def compute_analysis(registration, receipts, study):
    plan = registration["payload"]["plan"]
    records, pairs, issues = [], {}, []
    for trial in plan["trials"]:
        identity = trial["id"]
        record = {"id": identity, "pair": trial["pair"], "arm": trial["arm"], "status": "missing"}
        if identity in receipts:
            receipt = receipts[identity]
            record["receipt_sha256"] = receipt["sha256"]
            try:
                require(receipt["payload"]["failure"] is None and receipt["payload"]["returncode"] == 0,
                        "native execution failed: " + str(receipt["payload"]["failure"]))
                data = read(study / "trials" / identity / "output" / "stdout.json")
                for gate in plan["validity"]:
                    measured = at(data, gate["path"])
                    require(type(measured) is type(gate["equals"]) and measured == gate["equals"],
                            "validity gate failed: " + str(gate["path"]))
                value = at(data, plan["observable"]["path"])
                require(finite(value), "observable must be finite and numeric")
                record.update(status="valid", value=value)
                pairs.setdefault(trial["pair"], {})[trial["arm"]] = data
            except (Invalid, OSError, ValueError, KeyError, IndexError, TypeError) as error:
                record.update(status="invalid", reason=str(error))
        if record["status"] != "valid":
            issues.append(identity + ": " + record["status"])
        records.append(record)
    differences = []
    for pair, arms in pairs.items():
        if set(arms) != {"control", "treatment"}:
            continue
        try:
            for path in plan["paired_equal"]:
                require(canonical(at(arms["control"], path)) == canonical(at(arms["treatment"], path)),
                        "paired control mismatch: " + str(path))
            difference = at(arms["treatment"], plan["observable"]["path"]) - at(arms["control"], plan["observable"]["path"])
            require(finite(difference), "nonfinite paired difference")
            differences.append({"pair": pair, "treatment_minus_control": difference})
        except (Invalid, KeyError, TypeError, IndexError) as error:
            issues.append(pair + ": " + str(error))
    values = [item["treatment_minus_control"] for item in differences]
    mean = statistics.mean(values) if values else None
    require(mean is None or finite(mean), "nonfinite mean difference")
    spread = {"minimum": min(values), "maximum": max(values),
              "sample_sd": statistics.stdev(values) if len(values) > 1 else None} if values else None
    verdict = "inconclusive"
    if not issues and values:
        prediction = plan["prediction"]
        verdict = "supported" if prediction["minimum"] <= mean <= prediction["maximum"] else "contradicted"
    return {"schema": "numi.science.analysis.v1", "registration_sha256": registration["sha256"],
            "verdict": verdict, "evidence_level": plan["evidence_level"], "unit": plan["observable"]["unit"],
            "trials": records, "paired_differences": differences, "mean_difference": mean,
            "observed_spread": spread, "issues": issues,
            "uncertainty": "Descriptive variation across declared pairs, not a confidence interval or population inference.",
            "limitations": plan["limitations"]}


def analyze(study):
    study = Path(study).resolve()
    with locked(study):
        registration, receipts = verify(study)
        require(len(receipts) == len(registration["payload"]["plan"]["trials"]) or (study / "stopped.json").exists(),
                "finish every declared trial before analysis; missing trials cannot be silently excluded")
        result = compute_analysis(registration, receipts, study)
        return seal(study / "analysis.json", result)


def stop(study, reason):
    study = Path(study).resolve()
    require(isinstance(reason, str) and reason.strip(), "a stop reason is required")
    with locked(study):
        registration, receipts = verify(study)
        require(not (study / "analysis.json").exists(), "already analyzed")
        require(all(path.name in receipts for path in (study / "trials").iterdir()),
                "unfinished trial: inspect and resolve its live process before stopping")
        return seal(study / "stopped.json", {"registration_sha256": registration["sha256"],
                    "reason": reason, "stopped_at": now(),
                    "receipts": {key: value["sha256"] for key, value in receipts.items()}})


def revise(study, revision_path):
    study = Path(study).resolve()
    with locked(study):
        registration, _ = verify(study)
        analysis = unseal(study / "analysis.json")
        revision = read(revision_path)
        for name in ("reason", "next_test", "limitations"):
            text_field(revision, name)
        require(revision["decision"] in ("retain", "revise", "reject", "inconclusive"), "invalid revision decision")
        for name in ("statement", "version"):
            text_field(revision["model"], name)
        require(revision["model"]["version"] != registration["payload"]["plan"]["model"]["version"],
                "record a new model version, retaining the old one")
        require(revision["evidence"] == [analysis["sha256"]], "cite the exact analysis hash")
        if analysis["payload"]["verdict"] == "inconclusive":
            require(revision["decision"] == "inconclusive", "invalid evidence cannot support a model revision")
        elif analysis["payload"]["verdict"] == "contradicted":
            require(revision["decision"] in ("revise", "reject"), "contradicted prediction needs revision or rejection")
        return seal(study / "revision.json", {**revision, "analysis_sha256": analysis["sha256"],
                    "previous_model": registration["payload"]["plan"]["model"], "recorded_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("register", help="seal a plan before observing its trials")
    command.add_argument("plan", type=Path)
    command.add_argument("study", type=Path)
    command.add_argument("--parent", type=Path, help="completed parent study whose revised model is tested")
    command = sub.add_parser("stop", help="retain a fault stop; missing trials make analysis inconclusive")
    command.add_argument("study", type=Path)
    command.add_argument("--reason", required=True)
    for name, help_text in (("run", "execute the next declared native trial once"),
                            ("analyze", "compare all paired results with the fixed prediction"),
                            ("verify", "verify records and recompute analysis from raw observations"),
                            ("revise", "record an evidence-linked model revision")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("study", type=Path)
        if name == "revise":
            command.add_argument("revision", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "register":
            result = register(args.plan, args.study, args.parent)
        elif args.command == "run":
            result = run_next(args.study)
        elif args.command == "analyze":
            result = analyze(args.study)
        elif args.command == "revise":
            result = revise(args.study, args.revision)
        elif args.command == "stop":
            result = stop(args.study, args.reason)
        else:
            registration, receipts = verify(args.study)
            result = {"integrity": "verified", "registration_sha256": registration["sha256"],
                      "recorded_trials": len(receipts),
                      "declared_trials": len(registration["payload"]["plan"]["trials"]),
                      "analysis": (args.study / "analysis.json").exists(),
                      "revision": (args.study / "revision.json").exists()}
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 1 if args.command == "run" and result["payload"]["failure"] else 0
    except (Invalid, OSError, ValueError, KeyError, TypeError, IndexError, subprocess.CalledProcessError) as error:
        print("numi science: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
