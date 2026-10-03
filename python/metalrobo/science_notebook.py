"""Transparent scientific notebooks over owner executables; no scheduler or simulator.

Only the Python standard library is required. All writes are exclusive or atomic;
failed trials are evidence, never silently retried or excluded from analysis.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import stat
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
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "expected regular JSON file: " + str(path))
    require(path.stat().st_size <= 32 * 1024 * 1024, "JSON record exceeds 32 MiB: " + str(path))
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(), object_pairs_hook=unique,
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
    require(isinstance(value, dict) and isinstance(value.get(name), str) and value[name].strip(), "required text: " + name)


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


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


def is_v2(plan):
    return plan.get("schema") == "numi.science.plan.v2"


def validate_command(command, live=True):
    argv = command["argv"]
    require(isinstance(argv, list) and argv and
            all(isinstance(arg, str) and "\x00" not in arg for arg in argv), "argv must be a string array")
    executable = Path(argv[0])
    require(executable.is_absolute(), "use an absolute executable path: " + str(executable))
    if live:
        require(executable.is_file() and os.access(executable, os.X_OK), "executable is unavailable: " + str(executable))
    require(finite(command["timeout_seconds"]) and 0 < command["timeout_seconds"] <= 86400,
            "declare a finite timeout of at most one day")
    require(isinstance(command["env"], dict) and all(
        isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) and
        isinstance(value, str) and "\x00" not in value for key, value in command["env"].items()),
        "env must contain explicit string values; never include credentials")


def validate(plan, live=True):
    require(isinstance(plan, dict) and plan.get("schema") in ("numi.science.plan.v1", "numi.science.plan.v2"), "unsupported plan schema")
    canonical(plan)  # Reject nonfinite numbers anywhere, including unit identities.
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
            "notebook analysis is the mean paired treatment-minus-control difference")
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
    ids, pairs, units = set(), {}, {}
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
        validate_command(trial, live)
        if is_v2(plan):
            require(isinstance(trial["unit"], dict) and trial["unit"], "declare the actual experimental unit")
            require(set(trial["unit"]) == set(plan["design"]["unit_paths"]), "unit keys must match design.unit_paths")
            previous = units.setdefault(trial["pair"], trial["unit"])
            require(canonical(previous) == canonical(trial["unit"]), "paired arms have different experimental units")
    require(all(arms == {"control", "treatment"} for arms in pairs.values()), "unpaired study")
    if is_v2(plan):
        require(plan["purpose"] in ("exploration", "confirmation", "replication"), "declare study purpose")
        require(len({digest(unit) for unit in units.values()}) == len(units), "duplicate experimental units disguised as different pairs")
        require(isinstance(plan["design"]["unit_paths"], dict) and plan["design"]["unit_paths"], "declare observed unit identity paths")
        for path in plan["design"]["unit_paths"].values():
            json_path(path)
        require(isinstance(plan["model_file"], str) and Path(plan["model_file"]).is_absolute(), "model_file must be absolute")
        validate_command(plan["predictor"], live)
        require(isinstance(plan["model"].get("training_units"), list), "model must declare training_units")
        require(all(isinstance(unit, dict) and unit for unit in plan["model"]["training_units"]), "invalid training unit")
        require(all(set(unit) == set(plan["design"]["unit_paths"]) for unit in plan["model"]["training_units"]),
                "training units must use the declared unit identity keys")
        bound = set(plan["artifacts"])
        instrument = plan["instrument"]
        require(isinstance(instrument.get("artifacts"), list) and instrument["artifacts"], "bind instrument artifacts")
        require(set(instrument["artifacts"]).issubset(bound) and instrument["calibration"] in bound and plan["model_file"] in bound,
                "instrument, calibration and model files must be bound artifacts")
        for command in [plan["predictor"], *plan["trials"]]:
            for arg in command["argv"][1:]:
                if Path(arg).is_absolute() and "{run}" not in arg and (not live or Path(arg).is_file()):
                    require(arg in bound, "command input is not a bound artifact: " + arg)
    return plan


def base_environment():
    return {"PATH": os.defpath, "HOME": str(Path.home()), "TMPDIR": tempfile.gettempdir()}


def artifact_json(registration, study, original):
    payload = registration["payload"]
    checksum = payload["artifacts"][original]
    return read(study / "objects" / checksum)


def validate_calibration(plan, artifacts, report):
    require(report.get("schema") == "numi.science.calibration.v1" and report.get("status") == "passed",
            "instrument requires a passed calibration report")
    require(isinstance(report.get("checks"), list) and report["checks"] and
            all(isinstance(check, dict) and check.get("passed") is True and
                isinstance(check.get("id"), str) and check["id"] for check in report["checks"]), "calibration checks are missing or failed")
    require(isinstance(report.get("bindings"), dict) and
            all(report["bindings"].get(path) == artifacts[path] for path in plan["instrument"]["artifacts"]),
            "calibration does not bind the exact instrument")
    text_field(report, "scope")
    for path, checksum in report.get("evidence", {}).items():
        require(artifacts.get(path) == checksum, "calibration evidence is not bound: " + path)
    observed = report.get("observed_units", [])
    require(isinstance(observed, list) and all(isinstance(unit, dict) and
            set(unit) == set(plan["design"]["unit_paths"]) for unit in observed), "invalid calibration units")
    require(not {digest(unit) for unit in observed}.intersection(digest(trial["unit"]) for trial in plan["trials"]),
            "calibration conditions cannot be reused as test observations")


def timestamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "timestamps require a timezone")
    return result


def git_identity(repository):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()
    return {"repository": str(Path(repository).resolve()), "revision": git("rev-parse", "HEAD"),
            "status": git("status", "--porcelain=v1", "--untracked-files=normal"),
            "diff_sha256": hashlib.sha256(subprocess.check_output(
                ["git", "-C", str(repository), "diff", "HEAD", "--binary"])).hexdigest()}


def register(plan_path, study, parent=None):
    plan = validate(read(plan_path))
    require(is_v2(plan), "new studies require numi.science.plan.v2; v1 evidence remains readable")
    artifacts = {}
    for raw in plan["artifacts"] + [str(Path(__file__).resolve()), sys.executable, plan["predictor"]["argv"][0]] + [trial["argv"][0] for trial in plan["trials"]]:
        require(isinstance(raw, str) and Path(raw).is_absolute(), "artifacts require absolute file paths")
        # Keep the invoked path so retargeting a symlink is detected before execution.
        path = Path(raw)
        require(path.is_file(), "artifact is not a file: " + str(path))
        artifacts[str(path)] = file_hash(path)
    require(canonical(read(plan["model_file"])) == canonical(plan["model"]), "model_file content differs from the declared model")
    calibration = read(plan["instrument"]["calibration"])
    validate_calibration(plan, artifacts, calibration)
    parent_identity = None
    excluded_units = {digest(unit) for unit in plan["model"]["training_units"]}
    if parent:
        parent = Path(parent).resolve()
        parent_registration, _ = verify(parent)
        require(is_v2(parent_registration["payload"]["plan"]), "v1 parent lacks enforced unit lineage; preserve it as legacy evidence")
        revision = unseal(parent / "revision.json")
        require(plan["model"] == revision["payload"]["model"], "follow-up must test the revised model")
        require(plan["design"]["unit_paths"] == parent_registration["payload"]["plan"]["design"]["unit_paths"],
                "follow-up cannot rename the parent's experimental-unit identity")
        excluded_units.update(digest(unit) for unit in lineage_units(parent))
        parent_identity = {"study": str(parent), "snapshot": "lineage/parent",
                           "registration_sha256": parent_registration["sha256"], "revision_sha256": revision["sha256"]}
    if plan["purpose"] == "confirmation":
        require(not excluded_units.intersection(digest(trial["unit"]) for trial in plan["trials"]),
                "confirmation reuses observed/training units; choose unused conditions or label replication")
    study = Path(study).resolve()
    require(not study.exists(), "study already exists")
    if parent:
        require(parent not in study.parents and study not in parent.parents, "parent and child studies must be separate directories")
    size = sum(Path(path).stat().st_size for path in artifacts)
    ancestor = study.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    git_root = subprocess.run(["git", "-C", str(ancestor), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if git_root.returncode == 0:
        ignored = subprocess.run(["git", "-C", git_root.stdout.strip(), "check-ignore", "--quiet", str(study)], capture_output=True)
        require(ignored.returncode == 1, "evidence destination is ignored or could not be checked; use a durable directory outside the checkout")
    require(shutil.disk_usage(ancestor).free > size + 64 * 1024 * 1024, "insufficient space for durable input snapshots")
    study.mkdir(parents=True, exist_ok=False)
    (study / "trials").mkdir()
    (study / "attempts").mkdir()
    (study / "objects").mkdir()
    for path, checksum in artifacts.items():
        destination = study / "objects" / checksum
        if not destination.exists():
            with open(path, "rb") as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        require(file_hash(destination) == checksum, "artifact changed while snapshotting: " + path)
    if parent:
        with locked(parent):
            verify(parent)
            require(unseal(parent / "revision.json")["sha256"] == parent_identity["revision_sha256"], "parent changed during registration")
            shutil.copytree(parent, study / "lineage/parent", symlinks=True,
                            ignore=shutil.ignore_patterns(".lock", ".science-*"))
    environment = base_environment()
    prediction_run = execute_command(plan["predictor"], study / "prediction", environment)
    require(prediction_run["failure"] is None, "predictor failed; retained its output in " + str(study / "prediction"))
    prediction = read(study / "prediction/output/stdout.json")
    require(prediction.get("schema") == "numi.science.prediction.v1" and
            prediction.get("model_sha256") == artifacts[plan["model_file"]] and
            canonical(prediction.get("prediction")) == canonical(plan["prediction"]),
            "executable model does not produce the declared prediction")
    registration = {"schema": "numi.science.registration.v2", "registered_at": now(), "plan": plan,
                    "artifacts": artifacts, "source": git_identity(plan["repository"]),
                    "host": {"node": platform.node(), "platform": platform.platform(), "machine": platform.machine()},
                    "environment": environment, "prediction_run": prediction_run, "parent": parent_identity}
    check_artifacts({"payload": registration})
    return seal(study / "registration.json", registration)


def lineage_units(study):
    study = Path(study)
    registration = unseal(study / "registration.json")["payload"]
    units = [trial["unit"] for trial in registration["plan"]["trials"]]
    units.extend(registration["plan"]["model"]["training_units"])
    calibration = artifact_json({"payload": registration}, study, registration["plan"]["instrument"]["calibration"])
    units.extend(calibration.get("observed_units", []))
    if registration["parent"]:
        units.extend(lineage_units(study / registration["parent"]["snapshot"]))
    return units


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
    root = Path(root)
    if root.is_symlink():
        return {".": {"symlink": os.readlink(root)}}
    require(root.is_dir(), "output directory is missing")
    result = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            result[relative] = {"symlink": os.readlink(path)}
        elif stat.S_ISREG(mode):
            result[relative] = file_hash(path)
        elif not stat.S_ISDIR(mode):
            result[relative] = {"special_file_type": stat.S_IFMT(mode)}
    return result


def process_identity(pid):
    result = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "lstart=", "-o", "stat="],
                            env={"PATH": os.defpath, "LC_ALL": "C", "TZ": "UTC"}, capture_output=True, text=True)
    if result.returncode == 1:
        return None
    require(result.returncode == 0, "could not inspect process identity")
    fields = result.stdout.strip().rsplit(None, 1)
    if len(fields) != 2 or fields[1].startswith("Z"):
        return None
    return fields[0]


def group_members(group):
    result = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,stat="], capture_output=True, text=True, check=True)
    return [int(fields[0]) for row in result.stdout.splitlines() if len(fields := row.split()) == 3
            and int(fields[1]) == group and not fields[2].startswith("Z")]


class TrialInterrupted(Exception):
    pass


def execute_command(command, directory, environment, binding=None):
    """Bounded local execution, retaining every terminal outcome and partial file."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "output"
    output.mkdir()
    argv = [arg.replace("{run}", str(output)) for arg in command["argv"]]
    environment = {**environment, **command["env"]}
    process, process_record, failure, returncode = None, None, None, None
    started_at, begin = now(), time.monotonic()
    handlers = {}
    def interrupt(number, frame):
        raise TrialInterrupted("signal " + str(number))
    try:
        for number in (signal.SIGTERM, signal.SIGHUP):
            handlers[number] = signal.signal(number, interrupt)
        with (output / "stdout.json").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            try:
                process = subprocess.Popen(argv, cwd=output, env=environment, stdout=stdout,
                                           stderr=stderr, start_new_session=True)
                process_record = seal(directory / "process.json", {
                    "pid": process.pid, "pgid": process.pid, "host": platform.node(),
                    "identity": process_identity(process.pid), "binding": binding,
                    "runner_pid": os.getpid(), "runner_identity": process_identity(os.getpid())})
                returncode = process.wait(timeout=command["timeout_seconds"])
                if returncode:
                    failure = "process_exit_" + str(returncode)
            except (subprocess.TimeoutExpired, KeyboardInterrupt, TrialInterrupted) as error:
                failure = type(error).__name__ + ": " + str(error)
            except (OSError, ValueError, Invalid, subprocess.SubprocessError) as error:
                failure = "execution_error: " + str(error)
            finally:
                if process is not None:
                    # A successful leader may have left writers behind. Never seal live output.
                    members = group_members(process.pid)
                    if members:
                        failure = failure or "process left live descendants"
                        original = process_record["payload"]["identity"] if process_record else None
                        current = process_identity(process.pid)
                        require(current is None or original is None or current == original,
                                "process group identity changed; retained output is quarantined")
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    returncode = process.wait()
                    deadline = time.monotonic() + 5
                    while group_members(process.pid) and time.monotonic() < deadline:
                        time.sleep(0.01)
                    require(not group_members(process.pid), "process group still alive; output remains unfinished")
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
    files = tree_hashes(output)
    if any(not isinstance(value, str) for value in files.values()):
        failure = failure or "output contains symlink or special file"
    return {"started_at": started_at, "ended_at": now(), "elapsed_seconds": time.monotonic() - begin,
            "argv": argv, "environment": environment, "cwd": str(output), "returncode": returncode,
            "failure": failure, "process_sha256": process_record["sha256"] if process_record else None, "files": files}


def verify_execution(record, directory, command, environment, binding=None):
    require(record["files"] == tree_hashes(directory / "output"), "execution output changed")
    require(record["argv"] == [arg.replace("{run}", record["cwd"]) for arg in command["argv"]], "execution argv changed")
    require(record["environment"] == {**environment, **command["env"]}, "execution environment differs from preregistration")
    require(timestamp(record["ended_at"]) >= timestamp(record["started_at"]) and
            finite(record["elapsed_seconds"]) and record["elapsed_seconds"] >= 0, "invalid execution timing")
    if record.get("process_sha256"):
        process = unseal(directory / "process.json")
        require(process["sha256"] == record["process_sha256"] and process["payload"]["binding"] == binding,
                "process receipt mismatch")
    require(record["failure"] is None or isinstance(record["failure"], str), "invalid failure status")
    require(record["returncode"] is None or type(record["returncode"]) is int, "invalid exit status")
    require(record["failure"] is not None or (record["returncode"] == 0 and record.get("process_sha256")),
            "successful execution requires a process receipt and zero exit status")


def verify(study, _seen=None):
    study = Path(study).resolve()
    registration = unseal(study / "registration.json")
    payload = registration["payload"]
    plan = validate(payload["plan"], live=False)
    seen = set() if _seen is None else set(_seen)
    require(registration["sha256"] not in seen and len(seen) < 128, "cyclic or excessive parent lineage")
    seen.add(registration["sha256"])
    if is_v2(plan):
        require(payload["schema"] == "numi.science.registration.v2", "registration schema mismatch")
        require((study / "objects").is_dir() and not (study / "objects").is_symlink(), "input object directory missing or linked")
        require(set(plan["artifacts"]).issubset(payload["artifacts"]), "declared artifact bindings missing")
        for expected in set(payload["artifacts"].values()):
            require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "invalid artifact hash")
            path = study / "objects" / expected
            require(not path.is_symlink() and path.is_file() and file_hash(path) == expected, "input snapshot missing or changed")
        require(canonical(artifact_json(registration, study, plan["model_file"])) == canonical(plan["model"]), "model snapshot mismatch")
        validate_calibration(plan, payload["artifacts"], artifact_json(registration, study, plan["instrument"]["calibration"]))
        verify_execution(payload["prediction_run"], study / "prediction", plan["predictor"], payload["environment"])
        prediction = read(study / "prediction/output/stdout.json")
        require(payload["prediction_run"]["failure"] is None and
                prediction["schema"] == "numi.science.prediction.v1" and
                prediction["model_sha256"] == payload["artifacts"][plan["model_file"]] and
                canonical(prediction["prediction"]) == canonical(plan["prediction"]), "prediction is not bound to the model")
        require(timestamp(payload["registered_at"]) >= timestamp(payload["prediction_run"]["ended_at"]), "prediction was not computed before registration")
        excluded = {digest(unit) for unit in plan["model"]["training_units"]}
        if payload["parent"]:
            parent = payload["parent"]
            require(parent["snapshot"] == "lineage/parent", "invalid parent snapshot path")
            parent_root = study / parent["snapshot"]
            require(not (study / "lineage").is_symlink() and not parent_root.is_symlink(), "parent must be retained inside the notebook")
            parent_registration, _ = verify(parent_root, seen)
            revision = unseal(parent_root / "revision.json")
            require(parent_registration["sha256"] == parent["registration_sha256"] and
                    revision["sha256"] == parent["revision_sha256"] and revision["payload"]["model"] == plan["model"],
                    "parent revision/model lineage changed")
            require(plan["design"]["unit_paths"] == parent_registration["payload"]["plan"]["design"]["unit_paths"],
                    "parent unit identity changed")
            require(timestamp(payload["registered_at"]) >= timestamp(revision["payload"]["recorded_at"]), "follow-up predates model revision")
            excluded.update(digest(unit) for unit in lineage_units(parent_root))
        if plan["purpose"] == "confirmation":
            require(not excluded.intersection(digest(trial["unit"]) for trial in plan["trials"]), "confirmation reuses observed units")
    declared_ids = {trial["id"] for trial in plan["trials"]}
    require(not (study / "trials").is_symlink(), "trial directory cannot be a symlink")
    require(all(path.name in declared_ids for path in (study / "trials").iterdir()),
            "undeclared trial directory found")
    if is_v2(plan):
        require(not (study / "attempts").is_symlink() and (study / "attempts").is_dir(), "attempt ledger missing")
        require(all(path.name in {identity + ".json" for identity in declared_ids} for path in (study / "attempts").iterdir()), "undeclared attempt")
    receipts = {}
    last_end = timestamp(payload["registered_at"])
    preceding_complete = True
    for trial in plan["trials"]:
        directory = study / "trials" / trial["id"]
        attempt_path = study / "attempts" / (trial["id"] + ".json")
        if is_v2(plan) and attempt_path.exists():
            attempt = unseal(attempt_path)
            require(attempt["payload"]["registration_sha256"] == registration["sha256"] and attempt["payload"]["trial"] == trial,
                    "attempt ledger mismatch")
            require(directory.is_dir(), "attempted trial lost its directory; never rerun it")
        if not directory.exists():
            preceding_complete = False
            continue
        require(not directory.is_symlink(), "trial directory cannot be a symlink")
        started = unseal(directory / "started.json")
        require(started["payload"]["registration_sha256"] == registration["sha256"] and
                started["payload"]["trial"] == trial, "trial does not match registration")
        # Old records without a cwd used the trial directory; new archives can move.
        original_output = started["payload"].get("cwd", str(directory / "output"))
        expected_argv = [arg.replace("{run}", original_output) for arg in trial["argv"]]
        require(started["payload"]["argv"] == expected_argv, "trial argv mismatch")
        if is_v2(plan):
            require(preceding_complete, "trial was executed out of preregistered order")
            require(attempt_path.is_file() and unseal(attempt_path)["payload"] == started["payload"], "attempt start changed")
            require(timestamp(started["payload"]["started_at"]) >= last_end and
                    started["payload"]["environment"] == {**payload["environment"], **trial["env"]},
                    "trial timing/environment violates registration")
        if not (directory / "receipt.json").exists():
            preceding_complete = False
            continue
        receipt = unseal(directory / "receipt.json")
        require(receipt["payload"]["started_sha256"] == started["sha256"], "receipt start mismatch")
        require(receipt["payload"]["files"] == tree_hashes(directory / "output"), "trial evidence changed")
        if is_v2(plan):
            execution = receipt["payload"]
            verify_execution(execution, directory, trial, payload["environment"], started["sha256"])
            require(timestamp(execution["started_at"]) >= timestamp(started["payload"]["started_at"]), "execution predates trial start")
            last_end = timestamp(execution["ended_at"])
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
        if is_v2(plan):
            validate_revision(revision["payload"], registration, unseal(study / "analysis.json"))
            require(timestamp(revision["payload"]["recorded_at"]) >= last_end, "revision predates observations")
            model_path = study / "revised-model.json"
            require(read(model_path) == revision["payload"]["model"] and
                    file_hash(model_path) == revision["payload"]["model_sha256"], "revised model artifact changed")
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
        require(is_v2(plan), "legacy studies are read-only; preregister a v2 study")
        trial = next((trial for trial in plan["trials"] if trial["id"] not in receipts), None)
        require(trial is not None, "all declared trials already recorded")
        check_artifacts(registration)
        directory = study / "trials" / trial["id"]
        attempt = study / "attempts" / (trial["id"] + ".json")
        require(not directory.exists() and not attempt.exists(),
                "unfinished trial; inspect its process and recover it; never blindly restart it")
        output = directory / "output"
        environment = registration["payload"]["environment"]
        start = {"registration_sha256": registration["sha256"], "trial": trial,
                 "argv": [arg.replace("{run}", str(output)) for arg in trial["argv"]],
                 "environment": {**environment, **trial["env"]}, "cwd": str(output), "started_at": now(),
                 "host": platform.node(), "runner_pid": os.getpid(), "runner_identity": process_identity(os.getpid())}
        seal(attempt, start)  # Survives accidental deletion of the entire trial directory.
        directory.mkdir()
        started = seal(directory / "started.json", start)
        execution = execute_command(trial, directory, environment, started["sha256"])
        try:
            check_artifacts(registration)
        except (Invalid, OSError) as error:
            execution["failure"] = "artifact drift during trial: " + str(error)
        return seal(directory / "receipt.json", {**execution, "started_sha256": started["sha256"]})


def recover(study, reason):
    """Close a killed runner's attempt as invalid, only after all known writers exit."""
    study = Path(study).resolve()
    text_field({"reason": reason}, "reason")
    with locked(study):
        registration, receipts = verify(study)
        require(is_v2(registration["payload"]["plan"]), "legacy attempts cannot be recovered automatically")
        pending = [p for p in (study / "trials").iterdir() if p.name not in receipts]
        require(len(pending) == 1, "recovery requires exactly one unfinished attempt")
        directory = pending[0]
        started = unseal(directory / "started.json")
        process = unseal(directory / "process.json")
        info = process["payload"]
        require(info["binding"] == started["sha256"] and info["pgid"] == info["pid"], "process identity mismatch")
        require(info["host"] == platform.node() == started["payload"]["host"], "recover on the original host")
        require(info["runner_identity"] is not None and
                process_identity(info["runner_pid"]) != info["runner_identity"], "runner is still alive")
        require(not group_members(info["pgid"]), "native process group is still alive; wait for it to exit")
        end = now()
        payload = started["payload"]
        return seal(directory / "receipt.json", {
            "started_sha256": started["sha256"], "started_at": payload["started_at"], "ended_at": end,
            "elapsed_seconds": max(0, (timestamp(end) - timestamp(payload["started_at"])).total_seconds()),
            "argv": payload["argv"], "environment": payload["environment"], "cwd": payload["cwd"],
            "returncode": None, "failure": "recovered interrupted attempt: " + reason,
            "process_sha256": process["sha256"], "files": tree_hashes(directory / "output")})


def status(study):
    study = Path(study).resolve()
    registration, receipts = verify(study)
    plan = registration["payload"]["plan"]
    pending = [p.name for p in (study / "trials").iterdir() if p.name not in receipts]
    state = "unfinished" if pending else "registered"
    if len(receipts) == len(plan["trials"]):
        state = "recorded"
    for name in ("stopped", "analysis", "revision"):
        if (study / (name + ".json")).exists():
            state = name
    return {"integrity": "verified", "state": state, "registration_sha256": registration["sha256"],
            "recorded_trials": len(receipts), "declared_trials": len(plan["trials"]), "unfinished_trials": pending,
            "analysis": (study / "analysis.json").exists(), "revision": (study / "revision.json").exists(),
            "guarantees": "v2 snapshots, executable prediction and declared-unit lineage" if is_v2(plan)
            else "legacy v1: no retained inputs, enforced calibration or independent-unit guarantee"}


def archive(study, destination):
    study, destination = Path(study).resolve(), Path(destination).resolve()
    require(not destination.exists() and study not in destination.parents, "archive needs a new, separate directory")
    with locked(study):
        result = status(study)
        require(not result["unfinished_trials"], "cannot archive an unfinished process")
        shutil.copytree(study, destination, symlinks=True, ignore=shutil.ignore_patterns(".lock", ".science-*"))
        require(status(destination) == result, "archive verification differs")
    return {**result, "archive": str(destination)}


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
                if is_v2(plan):
                    observed_unit = {key: at(data, path) for key, path in plan["design"]["unit_paths"].items()}
                    require(canonical(observed_unit) == canonical(trial["unit"]), "observed experimental unit differs from registration")
                record.update(status="valid", value=value)
                pairs.setdefault(trial["pair"], {})[trial["arm"]] = data
            except (Invalid, OSError, ValueError, KeyError, IndexError, TypeError) as error:
                record.update(status="invalid", reason=str(error))
        if is_v2(plan):
            record["experimental_unit"] = trial["unit"]
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
    deviation = statistics.stdev(values) if len(values) > 1 else None
    if deviation is not None and not finite(deviation):
        issues.append("nonfinite sample variation")
        deviation = None
    spread = {"minimum": min(values), "maximum": max(values), "sample_sd": deviation} if values else None
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
        require(is_v2(registration["payload"]["plan"]), "legacy studies are read-only")
        require(len(receipts) == len(registration["payload"]["plan"]["trials"]) or (study / "stopped.json").exists(),
                "finish every declared trial before analysis; missing trials cannot be silently excluded")
        result = compute_analysis(registration, receipts, study)
        return seal(study / "analysis.json", result)


def stop(study, reason):
    study = Path(study).resolve()
    require(isinstance(reason, str) and reason.strip(), "a stop reason is required")
    with locked(study):
        registration, receipts = verify(study)
        require(is_v2(registration["payload"]["plan"]), "legacy studies are read-only")
        require(not (study / "analysis.json").exists(), "already analyzed")
        require(all(path.name in receipts for path in (study / "trials").iterdir()),
                "unfinished trial: inspect and resolve its live process before stopping")
        return seal(study / "stopped.json", {"registration_sha256": registration["sha256"],
                    "reason": reason, "stopped_at": now(),
                    "receipts": {key: value["sha256"] for key, value in receipts.items()}})


def validate_revision(revision, registration, analysis):
    plan = registration["payload"]["plan"]
    for name in ("reason", "next_test", "limitations"):
        text_field(revision, name)
    require(revision["decision"] in ("retain", "revise", "reject", "inconclusive"), "invalid revision decision")
    for name in ("statement", "version"):
        text_field(revision["model"], name)
    require(revision["model"]["version"] != plan["model"]["version"], "record a new model version, retaining the old one")
    require(revision["evidence"] == [analysis["sha256"]], "cite the exact analysis hash")
    if analysis["payload"]["verdict"] == "inconclusive":
        require(revision["decision"] == "inconclusive", "invalid evidence cannot support a model revision")
    elif analysis["payload"]["verdict"] == "contradicted":
        require(revision["decision"] in ("revise", "reject"), "contradicted prediction needs revision or rejection")
    if is_v2(plan):
        require(isinstance(revision["model"].get("training_units"), list), "revised model must declare training_units")
        require(all(isinstance(unit, dict) and unit for unit in revision["model"]["training_units"]), "invalid training unit")
        exposed = {digest(unit) for unit in plan["model"]["training_units"]}
        exposed.update(digest(trial["unit"]) for trial in plan["trials"])
        require(exposed.issubset({digest(unit) for unit in revision["model"]["training_units"]}),
                "revised model must retain all declared/previous training units")
        if revision["decision"] in ("retain", "inconclusive"):
            def parameters(model):
                return {key: value for key, value in model.items() if key not in ("version", "training_units", "validation")}
            require(parameters(plan["model"]) == parameters(revision["model"]),
                    "retained or inconclusive model cannot change its scientific parameters")


def revise(study, revision_path):
    study = Path(study).resolve()
    with locked(study):
        registration, _ = verify(study)
        require(is_v2(registration["payload"]["plan"]), "legacy studies are read-only")
        analysis = unseal(study / "analysis.json")
        revision = read(revision_path)
        require(not (study / "revision.json").exists(), "model revision already recorded")
        validate_revision(revision, registration, analysis)
        if is_v2(registration["payload"]["plan"]):
            require(canonical(read(revision["model_file"])) == canonical(revision["model"]), "revised model_file content differs")
            destination = study / "revised-model.json"
            # Complete an interrupted revision only with the exact same immutable model.
            if destination.exists():
                require(read(destination) == revision["model"], "unfinished revision has a different model")
            else:
                write_new(destination, revision["model"])
            revision["model_sha256"] = file_hash(destination)
        return seal(study / "revision.json", {**revision, "analysis_sha256": analysis["sha256"],
                    "previous_model": registration["payload"]["plan"]["model"], "recorded_at": now()})


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, epilog="Contract: " + str(root / "docs/SCIENTIFIC_WORKFLOW.md") +
                                     "; examples: " + str(root / "examples/science"))
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("register", help="seal a plan before observing its trials")
    command.add_argument("plan", type=Path)
    command.add_argument("study", type=Path)
    command.add_argument("--parent", type=Path, help="completed parent study whose revised model is tested")
    for name, help_text in (("stop", "retain a fault stop; missing trials make analysis inconclusive"),
                            ("recover", "seal an interrupted attempt as failed after its process group exits")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("study", type=Path)
        command.add_argument("--reason", required=True)
    command = sub.add_parser("archive", help="copy and verify the complete portable notebook")
    command.add_argument("study", type=Path)
    command.add_argument("destination", type=Path)
    for name, help_text in (("run", "execute the next declared native trial once"),
                            ("analyze", "compare all paired results with the fixed prediction"),
                            ("verify", "verify records and recompute analysis from raw observations"),
                            ("status", "inspect notebook state and evidence guarantees"),
                            ("revise", "record an evidence-linked model revision")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("study", type=Path)
        if name == "revise":
            command.add_argument("revision", type=Path)
    args = parser.parse_args()
    try:
        # Preserve the exact source path/hash bound by already registered v1 work.
        # New registration always uses v2; legacy execution keeps its old guarantees.
        if args.command in ("run", "analyze", "stop", "revise"):
            existing = unseal(args.study / "registration.json")
            if existing["payload"]["plan"]["schema"] == "numi.science.plan.v1":
                spec = importlib.util.spec_from_file_location("legacy_science", Path(__file__).with_name("science.py"))
                legacy = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(legacy)
                return legacy.main()
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
        elif args.command == "recover":
            result = recover(args.study, args.reason)
        elif args.command == "archive":
            result = archive(args.study, args.destination)
        else:
            result = status(args.study)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 1 if args.command == "run" and result["payload"]["failure"] else 0
    except (Invalid, OSError, ValueError, KeyError, TypeError, IndexError, subprocess.CalledProcessError) as error:
        print("numi science: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
