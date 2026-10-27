from pathlib import Path
from fractions import Fraction
import hashlib,importlib.util,json,sys
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/candidate.npz'
A=C.parent/'audit-159/report.json'
OUT=Path(__file__).resolve().parent
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
AUDIT_TOOL=ROOT/'publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
C_PIN='a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db'
A_PIN='2e870a160a1514e7a5621ceb504845303e984871532f60278bac12b6bbf817a8'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def exact_tri(t):return tuple(tuple(Fraction.from_float(float(x)) for x in p) for p in np.asarray(t).reshape(3,3))
def barycentric(p,tri,exact):
 n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]));axes=exact._axes(n)
 a,b,c,q=[tuple(v[k] for k in axes) for v in (tri[0],tri[1],tri[2],p)]
 den=exact._orient2(a,b,c)
 if den==0:raise ValueError('degenerate triangle during exact endpoint map')
 return (exact._orient2(b,c,q)/den,exact._orient2(c,a,q)/den,exact._orient2(a,b,q)/den)
def feature(w):
 zero=[i for i,x in enumerate(w) if x==0]
 if len(zero)>=2:return 'vertex'
 if len(zero)==1:return 'edge'
 return 'face'
def main():
 if sha(C)!=C_PIN or sha(A)!=A_PIN:raise ValueError('candidate or exact audit pin mismatch')
 with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
 P=np.asarray(Z['points_world_m'],dtype=np.float32).astype(np.float64);F=np.asarray(Z['source_O_faces'],dtype=np.int64)
 d=json.loads(A.read_text());bank=next(b for b in d['banks'] if (int(b['body']),int(b['stable_bone_id']))==(65,99))
 pairs=sorted({(int(sf),int(bid)) for sf,bid in bank['result']['triangle_pairs']})
 if len(pairs)!=148:raise ValueError(f'expected 148 exact child pairs, got {len(pairs)}')
 parent=np.asarray(Z['refinement_face_parent_face_row'],np.int64)
 mapped=sorted({(int(parent[sf]),bid) for sf,bid in pairs})
 if len(mapped)!=84:raise ValueError(f'expected 84 mapped identities, got {len(mapped)}')
 audit=mod('endpoint_audit159',AUDIT_TOOL);exact=audit.exact_module();helper=mod('endpoint_bankhelper',HELPER)
 T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
 mask=(ids[:,0]==65)&(ids[:,1]==99);bone=np.asarray(T[mask],np.float64);bids=np.asarray(ids[mask,3],np.int64)
 bmap={int(b):i for i,b in enumerate(bids)}
 if len(bone)!=342:raise ValueError('typed bank inventory is not 342 triangles')
 local=(P-pos[65])@M[65].T-off[65]
 receipts=[];total=0
 for sf,bid in pairs:
  bi=bmap.get(bid)
  if bi is None:raise ValueError(f'unknown bank triangle {bid}')
  st=exact_tri(local[F[sf]]);bt=exact_tri(bone[bi]);hits=sorted(set(exact.triangle_intersection_points(st,bt)))
  if not hits:raise ValueError(f'no exact endpoint reconstruction for {(sf,bid)}')
  ep=[];features=[]
  for q in hits:
   sw=barycentric(q,st,exact);bw=barycentric(q,bt,exact)
   if sum(sw)!=1 or min(sw)<0 or max(sw)>1 or sum(bw)!=1 or min(bw)<0 or max(bw)>1:raise ValueError(f'barycentric outside at {(sf,bid)}')
   if tuple(sum(sw[i]*st[i][k] for i in range(3)) for k in range(3))!=q:raise ValueError('skin rational reconstruction mismatch')
   if tuple(sum(bw[i]*bt[i][k] for i in range(3)) for k in range(3))!=q:raise ValueError('bone rational reconstruction mismatch')
   ep.append([str(v) for v in q]);features.append(feature(sw));total+=1
  receipts.append({'skin_face':sf,'bone_triangle':bid,'exact_local_endpoints_m':ep,'skin_features':features})
 if len(receipts)!=148:raise ValueError('exact endpoint inventory mismatch')
 pairdoc={'schema':'numi.human.b65_99.current_exact_child_pair_manifest.v1','candidate_path':str(C),'candidate_sha256':sha(C),'audit159_path':str(A),'audit159_sha256':sha(A),'raw_child_pair_count':len(pairs),'mapped_immediate_parent_pair_count':len(mapped),'mapped_identity_semantics':'(refinement_face_parent_face_row,stable_bone_triangle_id)','current_child_pairs':[list(q) for q in pairs],'current_mapped_parent_pairs':[list(q) for q in mapped],'geometry_or_native_admission':False}
 pairpath=OUT/'current-pair-manifest.json';pairpath.write_text(json.dumps(pairdoc,indent=2,sort_keys=True)+'\n')
 epdoc={'schema':'numi.human.b65_99.current-exact-endpoints.v1','candidate_path':str(C),'candidate_sha256':sha(C),'audit159_path':str(A),'audit159_sha256':sha(A),'pair_manifest_path':str(pairpath),'pair_manifest_sha256':sha(pairpath),'raw_child_pair_count':len(pairs),'mapped_immediate_parent_pair_count':len(mapped),'endpoint_count':total,'exact_intersections_reconstructed':True,'coordinate_semantics':'Exact rational narrowphase on the Float64 bank-local values produced by the pinned current Float32 world candidate and exact existing registration; barycentric reconstruction was verified on skin and bone for every exact endpoint. No endpoint is emitted.','exact_pair_endpoint_feature_witnesses':receipts,'geometry_or_native_admission':False}
 epath=OUT/'current-exact-endpoints.json';epath.write_text(json.dumps(epdoc,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'pair_manifest':str(pairpath),'pair_manifest_sha256':sha(pairpath),'endpoint_receipt':str(epath),'endpoint_receipt_sha256':sha(epath),'pair_count':len(pairs),'mapped_count':len(mapped),'endpoint_count':total},indent=2))
if __name__=='__main__':main()
