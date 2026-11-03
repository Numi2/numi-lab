from pathlib import Path
import subprocess,json,hashlib,shutil
def pin(raw):
 p=Path(raw)
 with p.open("rb") as f:h=hashlib.file_digest(f,"sha256").hexdigest()
 return {"path":str(p),"sha256":h,"bytes":p.stat().st_size}
def prepare(base,d,changes):
 base=Path(base);d=Path(d);d.mkdir()
 p=json.loads((base/"install-motion.plan.json").read_text().replace(str(base),str(d)))
 p["stages"][0]["environment"].update(changes)
 r=Path(p["native_binary"]["path"]).parents[2]
 p["stages"][0]["cwd"]=str(r)
 patch=d/"execution-source.patch";patch.write_bytes(subprocess.run(["git","diff","--binary","HEAD"],cwd=r,capture_output=True,check=True).stdout)
 p["native_binary"]=pin(p["native_binary"]["path"])
 p["runtime_dependencies"]={k:pin(k) for k in p["runtime_dependencies"]}
 m={"native_binary":p["native_binary"],"runtime_dependencies":p["runtime_dependencies"],"source":{"head":subprocess.run(["git","rev-parse","HEAD"],cwd=r,capture_output=True,text=True,check=True).stdout.strip(),"dirty_patch":pin(patch),"sources":{rel:pin(r/rel) for rel in ["matter/src/metal/contact.metalinc","matter/src/metal/fem.metalinc","matter/src/metal/fgmres.metalinc","matter/src/metal/vascular.metalinc","matter/src/runtime.mm"]}}}
 (d/"build-manifest.json").write_text(json.dumps(m,indent=2)+"\n")
 p["build_manifest"]=pin(d/"build-manifest.json");p["full_execution_source_pins"]=m["source"]
 p["retained_prior_plan"]=pin(base/"install-motion.plan.json")
 (d/"install-motion.plan.json").write_text(json.dumps(p,indent=2)+"\n")
 shutil.copy2(base/"run_native_plan.py",d/"run_native_plan.py")
 return d
def run(d):
 with (d/"runner.log").open("w") as f:q=subprocess.run(["caffeinate","-di","/opt/homebrew/bin/python3",str(d/"run_native_plan.py"),"install-motion"],stdout=f,stderr=subprocess.STDOUT)
 return q.returncode
