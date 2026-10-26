"""Map self-witnesses through exact registered prolongation identities; rebuild existing026 seed rows."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,importlib.util,json,sys
import numpy as np
N=Path(__file__).resolve().parent.parent
REFRESH=N/'refresh_surface_constraints546.py'
PIN='c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def node_keys(z):
 ptr=np.asarray(z['source_prolongation_indptr']);ids=np.asarray(z['source_prolongation_node_rows']);w=np.asarray(z['source_prolongation_weights'])
 if ptr.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu' or w.dtype!=np.float64:raise ValueError('CSR types')
 if ptr.ndim!=1 or not len(ptr) or ptr[0]!=0 or ptr[-1]!=len(ids) or len(ids)!=len(w) or np.any(np.diff(ptr)<=0):raise ValueError('CSR pointers')
 if np.any(ids<0) or not np.isfinite(w).all() or np.any(w<=0):raise ValueError('CSR values')
 keys=[]
 for j in range(len(ptr)-1):
  ii=ids[ptr[j]:ptr[j+1]];ww=w[ptr[j]:ptr[j+1]]
  if len(ii)>1 and np.any(np.diff(ii)<=0):raise ValueError('CSR indices not canonical')
  if abs(float(ww.sum())-1)>1e-12:raise ValueError('nonaffine row')
  keys.append((ii.astype('<i8',copy=False).tobytes(),ww.astype('<f8',copy=False).tobytes()))
 return keys
def map_exact(a,b,pairs):
 ka,kb=node_keys(a),node_keys(b);ia=defaultdict(list);ib=defaultdict(list)
 for j,k in enumerate(ka):ia[k].append(j)
 for j,k in enumerate(kb):ib[k].append(j)
 nm=np.full(len(ka),-1,np.int64)
 for j,k in enumerate(ka):
  if len(ia[k])==1 and len(ib.get(k,[]))==1:nm[j]=ib[k][0]
 af=np.asarray(a['source_O_faces']);bf=np.asarray(b['source_O_faces'])
 for f,n in [(af,len(ka)),(bf,len(kb))]:
  if f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu' or np.any(f<0) or np.any(f>=n):raise ValueError('faces')
 srcix=defaultdict(list);dstix=defaultdict(list)
 for j,f in enumerate(af):srcix[tuple(sorted(map(int,f)))].append(j)
 for j,f in enumerate(bf):dstix[tuple(sorted(map(int,f)))].append(j)
 fm=np.full(len(af),-1,np.int64)
 for j,f in enumerate(af):
  ns=nm[f]
  if len(srcix[tuple(sorted(map(int,f)))])==1 and np.all(ns>=0):
   hits=dstix.get(tuple(sorted(map(int,ns))),[])
   if len(hits)==1:fm[j]=hits[0]
 p=np.asarray(pairs)
 if p.ndim!=2 or p.shape[1]!=2 or p.dtype.kind not in 'iu' or np.any(p<0) or np.any(p>=len(af)) or np.any(p[:,0]==p[:,1]):raise ValueError('pairs')
 ok=np.all(fm[p]>=0,axis=1)
 out=np.sort(fm[p[ok]],axis=1)
 if len(out)and np.any(out[:,0]==out[:,1]):raise ValueError('collapsed mapped pair')
 return {'node_map':nm,'face_map':fm,'source_pairs':p,'mapped_source_indices':np.flatnonzero(ok),'unmapped_source_indices':np.flatnonzero(~ok),
 'mapped_pairs':np.unique(out,axis=0).reshape(-1,2)}
def validate_common_domain(a,b):
 ks=['prior_input_parent__source_prolongation_indptr','prior_input_parent__source_prolongation_node_rows',
 'prior_input_parent__source_prolongation_weights','prior_input_parent__source_prolongation_semantics','source_prolongation_semantics']
 for k in ks:
  if k not in a or k not in b or not same(a[k],b[k]):raise ValueError('different registered source domain '+k)
 return ks

def certify_rows(rows,records,pairs):
 expected=[tuple(map(int,x))for x in np.asarray(pairs)]
 actual=[tuple(map(int,r['faces']))for r in records]
 if actual!=expected:raise ValueError('026 record coverage mismatch')
 counts={'common_vertex_wedge':4,'moving_sat_midplane':6}
 if any(r.get('kind')not in counts for r in records):raise ValueError('026 record kind')
 if len(rows)!=sum(counts[r['kind']]for r in records):raise ValueError('026 row coverage mismatch')
 lower=np.asarray([float(r['lower'])for r in rows])
 if not np.isfinite(lower).all():raise ValueError('026 nonfinite row')
 rhs=float(np.max(lower,initial=-np.inf))if len(lower)else 0.
 if rhs>2e-8:raise ValueError('026 zero-step violation')
 return rhs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-candidate',type=Path,required=True);ap.add_argument('--source-audit130',type=Path,required=True)
 ap.add_argument('--source-pairs',type=Path,required=True);ap.add_argument('--context',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 ctx=json.loads(a.context.read_text());dst=Path(ctx['candidate']);sr=json.loads(Path(ctx['seed_report']).read_text())
 if sha(dst)!=ctx['candidate_sha256'] or sr['candidate_sha256']!=sha(dst) or sr['zero_step_feasible']is not True:raise ValueError('stale destination')
 pairpath=Path(sr['output_pairs_path'])
 if sha(pairpath)!=sr['output_pairs_sha256']:raise ValueError('stale seed array')
 audit=json.loads(a.source_audit130.read_text())
 if audit['candidate_sha256']!=sha(a.source_candidate) or not audit['complete_surface_audit']:raise ValueError('source witness audit pin')
 old=load(a.source_candidate);new=load(dst);domain=validate_common_domain(old,new)
 p=np.load(a.source_pairs,allow_pickle=False)
 if len(p)!=audit['exact_PLC_surface_pairs']:raise ValueError('source witness count')
 mapping=map_exact(old,new,p);merged=np.unique(np.vstack([np.load(pairpath,allow_pickle=False),mapping['mapped_pairs']]),axis=0)
 r=module('refresh561',REFRESH)
 if sha(r.OWNER026)!=PIN:raise ValueError('026 source pin')
 owner=r.load_module('pair026_561',r.OWNER026);ref=r.load_module('pack541_561',r.REFRESH541)
 if sha(r.REFRESH541)!=r.REFRESH541_SHA:raise ValueError('541 source pin')
 pins={str(p):sha(p)for p in [Path(__file__),REFRESH,r.OWNER026,r.REFRESH541,a.source_candidate,a.source_audit130,a.source_pairs,a.context,dst,Path(ctx['seed_report']),pairpath]}
 rows,records=owner.build_pair_rows(new['points_world_m'].astype(float)*1000,new['source_O_faces'],merged,margin_fraction=.25)
 rhs=certify_rows(rows,records,merged)
 a.output.mkdir(exist_ok=False);mp=a.output/'witness-remap.npz';np.savez_compressed(mp,**mapping)
 pp=a.output/'self-pairs-input.npy';np.save(pp,merged)
 rp=a.output/'self-rows.npz';np.savez_compressed(rp,**ref.pack(rows))
 out={**sr,'mapping_status':'PASS_PARTIAL_EXACT_PROLONGATION_IDENTITY_FACE_REMAP_AND_ZERO_STEP_026',
 'output_pairs_path':str(pp),'output_pairs_sha256':sha(pp),'output_rows_path':str(rp),'output_rows_sha256':sha(rp),
 'merged_unique_pair_count':len(merged),'zero_step_max_rhs_mm':rhs,'rebuilt_linear_row_count':len(rows),
 'prior_seed_report_path':ctx['seed_report'],'prior_seed_report_sha256':sha(ctx['seed_report']),
 'supplemental_witness_source':{'source_candidate':str(a.source_candidate),'source_candidate_sha256':sha(a.source_candidate),
 'source_audit130':str(a.source_audit130),'source_audit130_sha256':sha(a.source_audit130),'source_pairs':str(a.source_pairs),
 'source_pairs_sha256':sha(a.source_pairs),'source_pair_count':len(p),'mapped_source_pair_count':len(mapping['mapped_source_indices']),
 'unmapped_source_pair_count':len(mapping['unmapped_source_indices']),'mapping':str(mp),'mapping_sha256':sha(mp),
 'common_archived_domain_keys':domain,'method':'Unique exact sparse registered-source prolongation rows, then unique unordered face identity; no coordinate tolerance/welding; unmapped pairs retained explicitly.'},
 'coverage_limitation':'Partial historical-witness supplement only. No complete swept coverage; final exact geometry audits mandatory.'}
 sp=a.output/'seed-preflight.json';sp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 ctx.update(seed_pairs_path=str(pp),seed_pairs_sha256=sha(pp),seed_report=str(sp),seed_report_sha256=sha(sp))
 cp=a.output/'constraint-context.json';cp.write_text(json.dumps(ctx,indent=2,sort_keys=True)+'\n')
 if any(sha(p)!=v for p,v in pins.items()):raise ValueError('source/input changed')
 receipt={'status':'PASS_MAPPED_CONSTRUCTION_SEEDS_ONLY','source_pins':pins,'merged_pair_count':len(merged),
 'mapped_witnesses':len(mapping['mapped_source_indices']),'unmapped_witnesses':len(mapping['unmapped_source_indices']),
 'zero_step_max_rhs_mm':rhs,'context':str(cp),'native_admission':False,'accepted_physical_steps':0}
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
