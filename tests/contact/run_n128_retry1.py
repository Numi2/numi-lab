#!/usr/bin/env python3
"""Run one preregistered, resource-limited independent n=128 reference."""
import hashlib, json, pathlib, shutil, subprocess, sys, time
root = pathlib.Path(__file__).resolve().parents[2]
out = root / "docs/evidence/midpoint-first-failure-20261109"
binary = pathlib.Path(sys.argv[1]).resolve()
which = sys.argv[2]
if which not in ("050", "025"):
    raise SystemExit("use 050 or 025")
capacity = json.loads((out / "n128-capacity-execution.json").read_text())
if capacity["returncode"] != 0 or capacity["stopped_reason"]:
    raise SystemExit("capacity check did not pass")
expected = (out / "n128-final-inputs.sha256").read_text().splitlines()[1].split()[0]
if hashlib.sha256(binary.read_bytes()).hexdigest() != expected:
    raise SystemExit("binary differs from parity-checked input")
prefix = out / ("rk4-n128-retry1-h" + which)
paths = {s: pathlib.Path(str(prefix)+s) for s in (".json", ".log", "-execution.json")}
if any(p.exists() for p in paths.values()):
    raise SystemExit("attempt already exists")
h = {"050":"0.0000005", "025":"0.00000025"}[which]
cmd = [str(binary), "128", h, "0.01"]
start = time.monotonic()
peak_rss_kib = 0
reason = None
with paths[".json"].open("w") as result, paths[".log"].open("w") as log:
    proc = subprocess.Popen(cmd, stdout=result, stderr=log)
    while proc.poll() is None:
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True)
        try:
            peak_rss_kib = max(peak_rss_kib, int(rss.stdout.strip()))
        except ValueError:
            pass
        free_bytes = shutil.disk_usage(out).free
        if peak_rss_kib > 6*1024*1024:
            reason = "resident-memory limit"
        elif free_bytes < 3*1024**3:
            reason = "free-space limit"
        elif time.monotonic()-start > 6*3600:
            reason = "six-hour per-run limit"
        if reason:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        time.sleep(1)
    code = proc.wait()
record = {"command":cmd,"binary_sha256":expected,"returncode":code,
          "stopped_reason":reason,"wall_seconds":time.monotonic()-start,
          "peak_rss_kib_sampled":peak_rss_kib,"free_bytes_after":shutil.disk_usage(out).free,
          "limits":{"rss_kib":6*1024*1024,"free_bytes":3*1024**3,"wall_seconds":6*3600}}
paths["-execution.json"].write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record),flush=True)
raise SystemExit(0 if code==0 and reason is None else 1)
