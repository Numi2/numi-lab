#!/usr/bin/env python3
"""Resume-safe full-domain clipping using the existing 302 construction owner."""
from pathlib import Path
import numpy as np, json,hashlib,sys,time,importlib.util,os
from stream_clip_009 import TetBroadphase,free_capacity
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
HERE=Path(__file__).resolve().parent
MESH=E/'global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz'
BANK=E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz'
INV=E/'registered-surrounding-tissue-native013-191/inventory.json'
CLIP=E/'robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py'
OUT=HERE/'run-001';TARGET=.341
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,value):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def main():
 budget=float(sys.argv[1]) if len(sys.argv)>1 else 180
 begun=time.monotonic();OUT.mkdir(exist_ok=True);parts=OUT/'constraint-parts';parts.mkdir(exist_ok=True)
 pins={str(p):sha(p) for p in (MESH,BANK,INV,CLIP,Path(__file__),HERE/'stream_clip_009.py')}
 assert pins[str(MESH)]=='b76368d95c95e74c780999391eb5785a6311c43879ac3ceb6d5c134494a1850b'
 assert pins[str(BANK)]=='39deb100a99601dc48550c3e5bcddfaac7aea266525e389abbf349d1d8c47792'
 assert pins[str(CLIP)]=='e1b5a246e531a16d8d2438b8569d31a27c9a10712f0de063084bed87db58c436'
 cp=OUT/'checkpoint.json'
 if cp.exists():
  state=json.loads(cp.read_text());assert state['inputs']==pins,'input identity changed'
  for part in state['parts']:assert sha(parts/part['file'])==part['sha256'],'part changed'
 else:
  state={'inputs':pins,'next_bank':0,'next_local_face':0,'parts':[],'bank_summaries':[],
         'complete':False,'native_admission':False,'physical_admission':False,
         'pruning':'all current tetrahedra; no initializer or old-domain pruning',
         'total_elapsed_s':0}
  atomic(cp,state)
 if state['complete']:return
 z=np.load(MESH);X=z['points_world_m'];T=z['tetrahedra'].astype(np.int64)
 fixed=np.zeros(len(X),bool);fixed[z['physical_outer_node_ids']]=True
 bp=TetBroadphase(X,T)
 spec=importlib.util.spec_from_file_location('clip302',CLIP);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 inv=json.loads(INV.read_text())['banks'];assert len(inv)==689
 offsets=np.r_[0,np.cumsum([m['triangles'] for m in inv])]
 assert offsets[-1]==4581231
 bank=np.load(BANK);state['mesh_nodes']=len(X);state['mesh_tets']=len(T)
 for bi in range(state['next_bank'],len(inv)):
  m=inv[bi];v=bank[m['key']+'_vertices'];f=bank[m['key']+'_faces']
  assert v.dtype==np.float32 and f.shape==(m['triangles'],3)
  start=state['next_local_face'] if bi==state['next_bank'] else 0
  for lo in range(start,len(f),1024):
   hi=min(lo+1024,len(f));tri=v[f[lo:hi]].astype(np.float64)
   cand=bp.candidates(tri)
   tids=[];faces=[];bids=[];off=[0];bary=[];bad_rows=[];zero_rows=[];witnesses=[]
   aabb=0
   for j,ids in enumerate(cand):
    aabb+=len(ids)
    for ti in ids:
     B,C=c.triangle_tet_barycentric_polygon(bp.xyz[ti],tri[j])
     if not len(B):continue
     cap=free_capacity(B,T[ti],fixed)
     pi=len(tids);tids.append(int(ti));faces.append(int(offsets[bi]+lo+j));bids.append(bi)
     base=off[-1];bary.extend(B);off.append(base+len(B))
     bad=np.flatnonzero(cap<TARGET-1e-12);zero=np.flatnonzero(cap<=1e-12)
     bad_rows.extend((base+bad).tolist());zero_rows.extend((base+zero).tolist())
     if len(bad) and len(witnesses)<16:
      k=int(bad[0]);witnesses.append({'bank_id':bi,'kind':m['kind'],'stable_id':m['stable_id'],'layer':m['layer'],
       'local_face':lo+j,'tet_row':int(ti),'tet_nodes':T[ti].tolist(),'barycentric':B[k].tolist(),
       'world_m':(B[k]@bp.xyz[ti]).tolist(),'capacity':float(cap[k]),
       'positive_support_on_fixed_only':bool(np.all(fixed[T[ti][B[k]>0]]))})
   path=parts/('part-%05d.npz'%len(state['parts']))
   np.savez_compressed(path,tet_ids=np.asarray(tids,np.int64),face_ids=np.asarray(faces,np.int64),
      bank_ids=np.asarray(bids,np.uint16),poly_offsets=np.asarray(off,np.int64),
      polygon_barycentric=np.asarray(bary,np.float64).reshape(-1,4),
      insufficient_capacity_polygon_vertex_rows=np.asarray(bad_rows,np.int64),
      zero_capacity_polygon_vertex_rows=np.asarray(zero_rows,np.int64))
   rec={'file':path.name,'sha256':sha(path),'bank_id':bi,'bank_key':m['key'],'kind':m['kind'],'stable_id':m['stable_id'],
        'layer':m['layer'],'body_indices':m['body_indices'],'local_face_range':[lo,hi],
        'flat_face_range':[int(offsets[bi]+lo),int(offsets[bi]+hi)],'faces_processed':hi-lo,
        'aabb_candidates':aabb,'polygons':len(tids),'polygon_vertices':len(bary),
        'insufficient_capacity_rows':len(bad_rows),'zero_capacity_rows':len(zero_rows),
        'first_witnesses':witnesses}
   state['parts'].append(rec);state['next_bank']=bi if hi<len(f) else bi+1
   state['next_local_face']=hi if hi<len(f) else 0
   state['complete']=state['next_bank']==len(inv)
   state['faces_processed']=sum(p['faces_processed'] for p in state['parts'])
   state['polygons']=sum(p['polygons'] for p in state['parts'])
   state['insufficient_capacity_rows']=sum(p['insufficient_capacity_rows'] for p in state['parts'])
   state['zero_capacity_rows']=sum(p['zero_capacity_rows'] for p in state['parts'])
   elapsed=time.monotonic()-begun
   state['last_invocation_elapsed_s']=elapsed
   atomic(cp,state)
   print(json.dumps({k:state[k] for k in ('next_bank','next_local_face','faces_processed','polygons','insufficient_capacity_rows','zero_capacity_rows','complete')},sort_keys=True),flush=True)
   if elapsed>=budget:
    state['total_elapsed_s']+=elapsed;atomic(cp,state);return
 state['total_elapsed_s']+=time.monotonic()-begun;atomic(cp,state)
if __name__=='__main__':main()
