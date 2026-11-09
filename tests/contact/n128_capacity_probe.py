#!/usr/bin/env python3
"""Bound and retain one independent-reference allocation and force evaluation."""
import json, pathlib, shutil, subprocess, sys, time
root = pathlib.Path(__file__).resolve().parents[2]
out = root / "docs/evidence/midpoint-first-failure-20261109"
binary = pathlib.Path(sys.argv[1]).resolve()
result_path = out / "n128-capacity-execution.json"
if result_path.exists():
    raise SystemExit("capacity attempt already exists")
cmd = [str(binary), "--capacity-check", "128"]
start = time.monotonic()
peak_rss_kib = 0
reason = None
with (out / "n128-capacity.json").open("w") as result, (out / "n128-capacity.log").open("w") as log:
    proc = subprocess.Popen(cmd, stdout=result, stderr=log)
    while proc.poll() is None:
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True)
        try:
            peak_rss_kib = max(peak_rss_kib, int(rss.stdout.strip()))
        except ValueError:
            pass
        free_bytes = shutil.disk_usage(out).free
        if peak_rss_kib > 6 * 1024 * 1024:
            reason = "resident-memory limit"
        elif free_bytes < 3 * 1024**3:
            reason = "free-space limit"
        elif time.monotonic() - start > 180:
            reason = "capacity time limit"
        if reason:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        time.sleep(.25)
    code = proc.wait()
record = {"command":cmd,"returncode":code,"stopped_reason":reason,
          "wall_seconds":time.monotonic()-start,"peak_rss_kib_sampled":peak_rss_kib,
          "free_bytes_after":shutil.disk_usage(out).free,
          "limits":{"rss_kib":6*1024*1024,"free_bytes":3*1024**3,"wall_seconds":180}}
result_path.write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record), flush=True)
raise SystemExit(0 if code == 0 and reason is None else 1)
