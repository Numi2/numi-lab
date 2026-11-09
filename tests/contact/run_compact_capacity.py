#!/usr/bin/env python3
"""Bounded one-step RK4 capacity probe; not a contact trajectory or benchmark."""
import hashlib,json,pathlib,shutil,subprocess,time
root=pathlib.Path(__file__).resolve().parents[2];p=root/"docs/evidence/midpoint-first-failure-20261109"
binary=pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-compact-005/numi_contact_fem_fp64_reference_compact")
parity=json.loads((p/"compact-reference-trajectory-parity-006.json").read_text())
if len(parity["runs"])!=2 or not all(x["passes"] for x in parity["runs"]):raise SystemExit("reference parity failed")
actual=hashlib.sha256(binary.read_bytes()).hexdigest()
if parity["input_sha256"][str(binary)]!=actual:raise SystemExit("binary differs from parity-checked input")
paths={s:p/("compact-reference-n256-rk4-capacity-002"+s) for s in (".json",".log","-execution.json")}
if any(q.exists() for q in paths.values()):raise SystemExit("attempt already exists")
cmd=[str(binary),"--rk4-capacity-check","256"];start=time.monotonic();peak=0;reason=None
with paths[".json"].open("w") as out,paths[".log"].open("w") as err:
 proc=subprocess.Popen(cmd,stdout=out,stderr=err)
 while proc.poll() is None:
  sample=subprocess.run(["ps","-o","rss=","-p",str(proc.pid)],capture_output=True,text=True)
  try:peak=max(peak,int(sample.stdout.strip()))
  except ValueError:pass
  if peak>6*1024*1024:reason="resident-memory guard"
  elif shutil.disk_usage(p).free<int(3.2*1024**3):reason="free-space guard"
  elif time.monotonic()-start>300:reason="five-minute guard"
  if reason:
   proc.terminate()
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:proc.kill()
   break
  time.sleep(.25)
 code=proc.wait()
record={"command":cmd,"binary_sha256":actual,"returncode":code,"stopped_reason":reason,
 "peak_rss_kib_sampled":peak,"wall_seconds_observational_only":time.monotonic()-start,
 "free_bytes_after":shutil.disk_usage(p).free,
 "scope":"model construction and one complete noncontact RK4 step; full contact trajectory, spatial/temporal accuracy and runtime not qualified",
 "limits":{"rss_kib":6*1024*1024,"free_bytes":int(3.2*1024**3),"wall_seconds":300}}
try:
 d=json.loads(paths[".json"].read_text())
 record["peak_rss_bytes_self"]=d["max_rss_bytes_self"]
 record["basic_invariants_pass"]=d["max_rss_bytes_self"]<=6*1024**3 and d["mesh_divisions"]==256 and d["nodes"]==257**3 and d["tetrahedra"]==6*256**3 and abs(d["area_m2"]-.0004)<1e-14 and abs(d["mass_kg"]-.0088)<1e-8 and d["minimum_nodal_mass_kg"]>0 and d["force_N"]==0 and d["rk4_step"] and d["max_velocity_error_mps"]<1e-10 and d["max_position_error_m"]<1e-12
except (ValueError,KeyError):record["basic_invariants_pass"]=False
paths["-execution.json"].write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record),flush=True)
raise SystemExit(0 if code==0 and reason is None and record["basic_invariants_pass"] else 1)
