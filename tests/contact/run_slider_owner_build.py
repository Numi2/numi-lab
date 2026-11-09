#!/usr/bin/env python3
"""Build the real MetalWorld/Matter owner with a guarded disk budget."""
import json,os,pathlib,shutil,signal,subprocess,time
root=pathlib.Path(__file__).resolve().parents[2];p=root/"docs/evidence/midpoint-first-failure-20261109"
build="/Users/numi/numi-matter-slider-owner-build-20261109"
log=p/"slider-owner-build.log";record=p/"slider-owner-build-execution.json"
if log.exists() or record.exists():raise SystemExit("build attempt already exists")
cmd=["cmake","--build",build,"--target","metalrobo_matter_physics_probe","-j1"]
start=time.monotonic();reason=None
with log.open("w") as stream:
 child=subprocess.Popen(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 while child.poll() is None:
  if shutil.disk_usage(root).free<int(3.5*1024**3):reason="free-space guard"
  elif time.monotonic()-start>3600:reason="one-hour build limit"
  if reason:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
   break
  time.sleep(2)
 code=child.wait()
meta={"command":cmd,"returncode":code,"stopped_reason":reason,"wall_seconds":time.monotonic()-start,"free_bytes_after":shutil.disk_usage(root).free}
record.write_text(json.dumps(meta,indent=2)+"\n");print(json.dumps(meta),flush=True)
raise SystemExit(0 if code==0 and not reason else 1)
