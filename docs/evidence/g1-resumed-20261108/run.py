import datetime, hashlib, json, os, pathlib, signal, subprocess, time
base = pathlib.Path("/Users/numi/numi-g1-resumed-20261108")
base.mkdir(parents=True, exist_ok=False)
source_run = pathlib.Path("/Users/numi/numi-g1-training-200-20261108/seed-3/run")
source_policy = source_run / "candidate.policypack"
source_optimizer = source_run / "learner.safetensors"
package = pathlib.Path("/Users/numi/numi-lab-release-panel-relocated-20261108")
interpreter = "/Users/numi/numi-window-integration-venv-20261107/bin/python"
paths = [package / "bin/numi", package / "bin/metalrobo_task_train",
         package / "bin/metalrobo_task_rollout", package / "lib/libmetalrobo.dylib",
         package / "lib/metalrobo/MetalRobo.metallib",
         package / "share/numi/REVISION",
         package / "share/numi/python/metalrobo/policy_selection.py",
         pathlib.Path(interpreter).resolve(), source_policy, source_optimizer]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(base / "input-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
env = os.environ.copy()
env.update(NUMI_MLX_PYTHON=interpreter, NUMI_RUN_DIR=str(base / "run"),
           NUMI_SELECTION_ENVS="8", NUMI_SELECTION_SEED="4081024",
           NUMI_WORKSPACE_ROOT=str(base))
command = [str(package / "bin/numi"), "train", "--envs", "16", "--steps", "24",
           "--updates", "400", "--chunk", "8", "--update-epochs", "2",
           "--scene", "ground", "--task", "g1-legs-locomotion",  "--seed", "20261203", "--learner-seed", "20261213",
           "--policy-pack", str(source_policy), "--learner-state", str(source_optimizer),
           "--output-learner-state", str(base / "run/learner.safetensors")]
(base / "command.json").write_text(json.dumps({"argv":command, "env":{k:env[k] for k in
    ("NUMI_MLX_PYTHON","NUMI_RUN_DIR","NUMI_SELECTION_ENVS","NUMI_SELECTION_SEED","NUMI_WORKSPACE_ROOT")}}, indent=2) + "\n")
def processes(name):
    (base/name).write_text(subprocess.check_output(["ps","-Ao","pid,ppid,etime,command"], text=True))
processes("before-ps.txt")
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
t0 = time.monotonic()
timed_out = False
with (base/"stdout.txt").open("w") as out, (base/"stderr.txt").open("w") as err:
    p = subprocess.Popen(command, cwd=base, env=env, stdout=out, stderr=err, start_new_session=True)
    try:
        rc = p.wait(timeout=180)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(p.pid, signal.SIGTERM)
        try: rc = p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            rc = p.wait()
end = datetime.datetime.now(datetime.timezone.utc).isoformat()
processes("after-ps.txt")
(base / "post-input-hashes.json").write_text(json.dumps(
    {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2) + "\n")
(base/"result.json").write_text(json.dumps({"start_utc":start,"end_utc":end,
    "elapsed_seconds":time.monotonic()-t0,"exit_code":rc,"timed_out":timed_out},indent=2)+"\n")
print((base/"result.json").read_text())
