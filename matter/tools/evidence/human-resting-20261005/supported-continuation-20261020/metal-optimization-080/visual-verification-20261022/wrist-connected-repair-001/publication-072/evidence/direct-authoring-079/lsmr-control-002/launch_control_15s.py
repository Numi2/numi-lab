import subprocess,sys,json,time
from pathlib import Path
script=Path(__file__).with_name("run_lsmr_control_002.py")
cmd=["/Users/n/numi-human-prep-venv-20261005/bin/python",str(script)]
t=time.monotonic()
try:
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
 Path(__file__).with_name("launcher-stdout.txt").write_text(r.stdout)
 Path(__file__).with_name("launcher-stderr.txt").write_text(r.stderr)
 print(json.dumps({"returncode":r.returncode,"elapsed_s":time.monotonic()-t,"stdout_tail":r.stdout[-3000:],"stderr_tail":r.stderr[-1000:]}))
 sys.exit(r.returncode)
except subprocess.TimeoutExpired as e:
 Path(__file__).with_name("launcher-stdout.txt").write_text((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout,bytes) else (e.stdout or ""))
 Path(__file__).with_name("launcher-stderr.txt").write_text((e.stderr or b"").decode(errors="replace") if isinstance(e.stderr,bytes) else (e.stderr or ""))
 print(json.dumps({"timeout_s":15,"elapsed_s":time.monotonic()-t,"stdout_tail":str(e.stdout)[-3000:],"stderr_tail":str(e.stderr)[-1000:]}))
 sys.exit(124)
