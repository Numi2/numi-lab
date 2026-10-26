"""Batch contained-point objective over the existing matched 490/485/458/408 fit."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
ENTRY=D/"D/D/component-lp-decomposition-490/run_matched_component_fit_490.py"
EXPECTED="7719b5042d06abc8211400bd82fc5fed89d3d388e5a1c0794ddd7a0e084bf693"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def point_weights(h,w,meta):
 h=np.asarray(h,dtype=float);old=np.asarray(w,dtype=float)
 if h.shape!=old.shape or len(meta)!=len(h) or not np.isfinite(h).all() or np.any(h<=0):
  raise ValueError("invalid target rows")
 p=np.array([x.get("kind")=="point" for x in meta]);s=~p
 if not p.any():raise ValueError("no contained point targets")
 out=old.copy();pm=.99 if s.any() else 1.
 out[p]=pm*h[p]/float(h[p].sum())
 if s.any():
  if not np.isfinite(old[s]).all() or np.any(old[s]<0) or old[s].sum()<=0:raise ValueError("invalid surface weights")
  out[s]=(1-pm)*old[s]/float(old[s].sum())
 return out
def main():
 if sha(ENTRY)!=EXPECTED:raise ValueError("490 source changed")
 spec=importlib.util.spec_from_file_location("fit490_point_batch545",ENTRY)
 entry=importlib.util.module_from_spec(spec);sys.modules[spec.name]=entry;spec.loader.exec_module(entry)
 oldload=entry._load_module;state={"builder_calls":0}
 def load(path):
  runner=oldload(path)
  if Path(path).resolve()!=entry.RUNNER_485.resolve():return runner
  old=runner.load
  def load485(name,path):
   mod=old(name,path)
   if name!="matched458_for_485":return mod
   original=mod.install_objective
   def install(owner,objective,crossing):
    receipt=original(owner,objective,crossing);build=owner.build_clearance_rows
    def rows(*args,**kwargs):
     C,h,w,meta=build(*args,**kwargs);weights=point_weights(h,w,meta)
     updated=[dict(m,normalized_weight=float(weights[i]))for i,m in enumerate(meta)]
     state.update(builder_calls=state["builder_calls"]+1,point_rows=sum(x["kind"]=="point" for x in meta),
       surface_rows=sum(x["kind"]!="point" for x in meta),point_family_mass=.99,
       objective="point cost per mm constant; existing surface weights retain1percent mass")
     return C,h,weights,updated
    owner.build_clearance_rows=rows
    return receipt
   mod.install_objective=install
   return mod
  runner.load=load485
  return runner
 entry._load_module=load
 argv=list(sys.argv[1:]);out=Path(argv[argv.index("--output")+1]);selfsha=sha(__file__)
 entry.main()
 if sha(__file__)!=selfsha or sha(ENTRY)!=EXPECTED:raise RuntimeError("source changed during fit")
 if not state["builder_calls"]:raise RuntimeError("batch objective was not called")
 rp=out/"report.json";j=json.loads(rp.read_text());j["contained_batch545"]={**state,
  "source":str(Path(__file__)),"sha256":selfsha,"only_objective_weights_changed":True,
  "native_admission":False,"accepted_physical_steps":0}
 rp.write_text(json.dumps(j,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
