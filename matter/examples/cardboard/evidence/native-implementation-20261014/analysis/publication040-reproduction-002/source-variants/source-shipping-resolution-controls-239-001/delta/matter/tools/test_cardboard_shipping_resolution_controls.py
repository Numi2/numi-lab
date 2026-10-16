#!/usr/bin/env python3
"""Exercise the real driver's resolved shipping controls without creating a mesh or Metal runtime."""
from pathlib import Path
import argparse,json,subprocess
p=argparse.ArgumentParser();p.add_argument("--probe",type=Path,required=True);a=p.parse_args()
checks=0;commands=[]
def call(extra,ok=True):
 global checks
 cmd=[str(a.probe),"--shipping-box","--shipping-preflight-only","--validate-only"]+extra
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);commands.append({"argv":cmd,"exit_code":r.returncode})
 if not ok:
  assert r.returncode!=0,(cmd,r.stdout,r.stderr);checks+=1;return
 assert r.returncode==0,(cmd,r.stdout,r.stderr)
 d=json.loads(r.stdout);assert d["arguments_valid"] and not d["mesh_generated"] and not d["native_runtime_created"]
 checks+=1;return d["resolved"]
base=call([])
expected={"shipping_box":True,"nx_per_pitch":8,"ny":16,"thickness_slices":1,"box_subdivisions_per_band":16,"dt_s":25e-6,"newton_iterations":14,"fgmres_iterations":1024}
assert base==expected,(base,expected);checks+=1
explicit=["--nx","8","--ny","16","--thickness-slices","1","--dt",".000025","--newton-iterations","14","--fgmres-iterations","1024"]
assert call(explicit)==base;checks+=1
for nx in (16,32):
 r=call(["--nx",str(nx)]);assert r==dict(base,nx_per_pitch=nx);checks+=1
for ny in (32,64):
 r=call(["--ny",str(ny)]);assert r==dict(base,ny=ny,box_subdivisions_per_band=ny);checks+=1
for layers in (2,4):
 r=call(["--thickness-slices",str(layers)]);assert r==dict(base,thickness_slices=layers);checks+=1
for dt in (12.5e-6,6.25e-6):
 r=call(["--dt",str(dt)]);assert r==dict(base,dt_s=dt);checks+=1
r=call(["--newton-iterations","64","--fgmres-iterations","512"])
assert r==dict(base,newton_iterations=64,fgmres_iterations=512);checks+=1
for option,value in (("--nx","0"),("--ny","0"),("--thickness-slices","0"),("--dt","-1"),("--newton-iterations","129"),("--fgmres-iterations","2048")):
 call([option,value],False)
r=subprocess.run([str(a.probe),"--preset","literature2009","--validate-only","--nx","32","--ny","8","--thickness-slices","4","--dt",".00005"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
assert r.returncode==0,r.stderr
v=json.loads(r.stdout)["resolved"]
assert v["nx_per_pitch"]==32 and v["ny"]==8 and v["thickness_slices"]==4 and v["dt_s"]==.00005
checks+=1
print(json.dumps({"schema":"numi.cardboard.shipping-resolution-cli-checks.v1","checks":checks,"physical_steps":0,"shipping_mesh_or_convergence_qualified":False,"commands":commands}))
