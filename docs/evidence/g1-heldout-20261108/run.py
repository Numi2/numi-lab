import datetime, hashlib, json, os, pathlib, signal, subprocess, sys, time

if len(sys.argv) != 2 or sys.argv[1] not in ("1", "2", "3"):
    raise SystemExit("usage: run.py 1|2|3")
trial = int(sys.argv[1])
base = pathlib.Path(f"/Users/numi/numi-g1-heldout-20261108/policy-{trial}")
base.mkdir(parents=True, exist_ok=False)
pkg = pathlib.Path("/Users/numi/numi-lab-release-panel-relocated-20261108")
binary = pkg / "bin/metalrobo_task_rollout"
shader = pkg / "lib/metalrobo/MetalRobo.metallib"
library = pkg / "lib/libmetalrobo.dylib"
policy = pathlib.Path(f"/Users/numi/numi-g1-training-200-20261108/seed-{trial}/run/candidate.deployment.policypack")
paths = (binary, shader, library, policy)
(base/"inputs.json").write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+"\n")
for seed in (4081022, 4081023):
    cmd = [str(binary), "--envs", "8", "--steps", "1000", "--repeats", "1",
           "--chunk", "1", "--physics-substeps", "4", "--velocity-iterations", "4",
           "--final-velocity-iterations", "2", "--scene", "ground", "--task",
           "g1-legs-locomotion", "--metallib", str(shader), "--seed", str(seed),
           "--policy-pack", str(policy), "--state-trace", str(base/f"{seed}.state.tsv")]
    (base/f"{seed}.command.json").write_text(json.dumps(cmd,indent=2)+"\n")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.monotonic()
    timeout = False
    with (base/f"{seed}.stdout.json").open("w") as out, (base/f"{seed}.stderr").open("w") as err:
        p = subprocess.Popen(cmd, cwd=base, stdout=out, stderr=err, start_new_session=True)
        try: rc = p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(p.pid, signal.SIGTERM)
            try: rc = p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                rc = p.wait()
    ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (base/f"{seed}.result.json").write_text(json.dumps({"start_utc":started,"end_utc":ended,
        "elapsed_seconds":time.monotonic()-t0,"exit_code":rc,"timed_out":timeout},indent=2)+"\n")
    print(f"policy={trial} seed={seed} exit={rc} timeout={timeout}", flush=True)
