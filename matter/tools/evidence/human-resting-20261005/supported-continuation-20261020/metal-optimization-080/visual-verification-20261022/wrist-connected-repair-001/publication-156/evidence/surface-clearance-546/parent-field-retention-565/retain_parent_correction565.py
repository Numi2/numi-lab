"""Transfer a retained offline refinement correction to exact parent material nodes."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,sys
N=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def transfer(base,source,trial):
 for z in [base,source,trial]:
  x=z['points_world_m']
  if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('Float32 coordinates required')
 sx=source['points_world_m'];tx=trial['points_world_m'];bx=base['points_world_m']
 if sx.shape!=tx.shape:raise ValueError('refined domain mismatch')
 for k in ['source_prolongation_indptr','source_prolongation_node_rows','source_prolongation_weights']:
  if not same(source[k],trial[k]):raise ValueError('trial provenance changed')
 if len(bx)>len(sx):raise ValueError('parent is not a prefix')
 # This deliberately supports only byte-identical registered parent P rows.
 for k in ['source_prolongation_node_rows','source_prolongation_weights']:
  n=int(base['source_prolongation_indptr'][-1])
  if not same(base[k],source[k][:n]):raise ValueError('parent registered row prefix differs')
 if not same(base['source_prolongation_indptr'],source['source_prolongation_indptr'][:len(bx)+1]):raise ValueError('parent CSR prefix differs')
 anchors=base['typed_coarse_anchor_output_rows_018']
 delta=tx[:len(bx)].astype(np.float64)-sx[:len(bx)].astype(np.float64)
 if np.any(delta[anchors]!=0):raise ValueError('source correction moves typed anchors')
 y=(bx.astype(np.float64)+delta).astype(np.float32)
 if not same(y[anchors],bx[anchors]):raise ValueError('emitted anchors changed')
 result=dict(base);result['points_world_m']=y
 if any(not same(result[k],base[k])for k in base if k!='points_world_m'):raise ValueError('nonpoint payload changed')
 return result,{'node_count':len(bx),'changed_nodes':int(np.any(y!=bx,axis=1).sum()),'max_transferred_step_mm':float(np.linalg.norm(y.astype(float)-bx.astype(float),axis=1).max(initial=0)*1000),'anchor_count':len(anchors)}
def main():
 mpath=N/'continue_surface_round546.py';s=importlib.util.spec_from_file_location('stage565',mpath);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
 state=N/'connected-reference-degeneracy-repair-557/current-best029-004/state.json';st=json.loads(state.read_text())
 src=N/'connected-source-refinement-553/refine-frozen30886-002/candidate.npz'
 selected=N/'swept-backtrack-prep-558/backtrack-031-001/final-point-continuation/state.json';sr=json.loads(selected.read_text());trial=Path(sr['candidate'])
 pins={str(p):sha(p)for p in [Path(__file__),state,Path(st['candidate']),src,selected,trial]}
 a,proof=transfer(load(st['candidate']),load(src),load(trial))
 T=Path(__file__).parent/'run-001';T.mkdir(exist_ok=False);cp=T/'candidate.npz';np.savez_compressed(cp,**a)
 (T/'proposal.json').write_text(json.dumps({'status':'PARENT_CORRECTION_PROPOSAL_ONLY','source_pins':pins,**proof,'candidate_sha256':sha(cp),'native_admission':False,'accepted_physical_steps':0},indent=2)+'\n')
 check=N.parent/'contained-clearance-545/run_candidate_checks545.py'
 m.run_stage([sys.executable,str(check),'--candidate',str(cp),'--baseline-state',str(state),'--trust-mm','4','--output-dir',str(T/'full-check')],T/'full-check-stage',slot=False,source_paths=(Path(__file__),check),timeout=600)
 if any(sha(p)!=v for p,v in pins.items()):raise ValueError('source/input changed')
 print((T/'full-check/report.json').read_text()[:1000])
if __name__=='__main__':main()
