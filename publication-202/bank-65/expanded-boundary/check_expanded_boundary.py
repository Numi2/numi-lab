#!/usr/bin/env python3
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,importlib.util,json,os,subprocess,sys,time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
B=S/'per-bank-workers-20261027/b65-99'
BASE=B/'root-layer-guard-002/backtrack-0p25/candidate.npz'
AUD159=BASE.parent/'audit-159/report.json';AUD130=BASE.parent/'audit-130/summary.json'
SELFP=BASE.parent/'audit-130/surface-self-pairs.npy'
TOPO=B/'attempt-002/fan-hole-ancestry-check.json'
WIT=B/'attempt-004-bone-aware-connected-patch-prep-001/witness-preflight.json'
OLD_BASE=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
EXPORT=E/'E/generic-static-query-042/prepare_generic_query_points.py'
QUERYBIN=Path('/tmp/generic-query-overlay-build-042-work/matter/numi-matter-human-skin-fat-static-query')
OUT=Path(os.environ.get('BOUNDARY_RUN_DIR',str(Path(__file__).resolve().parent)))
ROWS_OVERRIDE=os.environ.get('BOUNDARY_ROWS_JSON')
CLEARANCE_M=5e-5
PINS={'candidate':'a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db','audit159':'2e870a160a1514e7a5621ceb504845303e984871532f60278bac12b6bbf817a8','audit130':'c1aa0b43b19cfa860ee7c92400ecd396d16f5442c0918bdb2ef4c434bfff6a0d','self_pairs':'55737cf1229ed3c3f82eb23b50874fb56749277d1c3a190bccfa1f3a991a57de','topology':'2cf84019b3c58e9300b063e61d9b3001c3111e8dfa84985ca773b8bc020c0281','witness_preflight':'579ab33b2a1922af8075d7825b93a2a132e0f84aaf072b85c56a2196f9083855','bank_helper':'ea833990d8b1ffb309d4960e62fb8c67e5cf0ecdbcdc0d8e9ad40c20eaf4c6db','query_exporter':'0e09fecb9cf0d2c5e9049356b822d138819e1560a2a2049bcbd4d8602336a1ee','query_binary':'5dc7b5ce2a96f58bd03ccc85e38645d9da2ec8be105a532c461a3e0cc6d25460'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmod(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def patch_boundary(F,rows):
 use=Counter();adj=defaultdict(set)
 for r in rows:
  f=F[int(r)]
  for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
   e=tuple(sorted((int(a),int(b))));use[e]+=1;adj[int(a)].add(int(b));adj[int(b)].add(int(a))
 if any(v not in (1,2) for v in use.values()):raise ValueError('patch edge incidence not 1/2')
 edges=sorted(e for e,n in use.items() if n==1);deg=Counter(x for e in edges for x in e)
 if not edges or any(n!=2 for n in deg.values()):raise ValueError('boundary is not degree-2')
 # Check that the induced patch is connected and the simple boundary has Euler characteristic one.
 pnodes=set(map(int,np.asarray(F[rows],np.int64).reshape(-1))) if False else set()
 pnodes=set()
 padj=defaultdict(set)
 for r in rows:
  f=F[int(r)];pnodes.update(map(int,f))
  for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):padj[int(a)].add(int(b));padj[int(b)].add(int(a))
 seen=set();components=0
 for v in pnodes:
  if v in seen:continue
  components+=1;stack=[v];seen.add(v)
  while stack:
   x=stack.pop()
   for y in padj[x]:
    if y not in seen:seen.add(y);stack.append(y)
 chi=len(pnodes)-len(use)+len(rows)
 if components!=1 or chi!=1:raise ValueError(f'patch not disk: components={components}, chi={chi}')
 comps=0;seen=set();bad=[]
 for v in deg:
  if v in seen:continue
  comps+=1;stack=[v];seen.add(v)
  while stack:
   u=stack.pop()
   for e in edges:
    if u in e:
     w=e[0] if e[1]==u else e[1]
     if w not in seen:seen.add(w);stack.append(w)
 chi=len(adj)-len(use)+len(rows)
 if comps!=1 or chi!=1:raise ValueError(f'boundary is not one disk: loops/components={comps}, chi={chi}')
 return edges,sorted(deg)
def pseg_d2(p,a,b):
 d=b-a;den=float(d@d)
 if den==0:return float((p-a)@(p-a))
 t=float(np.clip((p-a)@d/den,0.,1.));q=a+t*d;v=p-q;return float(v@v)
def ptri_d(p,t):
 a,b,c=t;ab=b-a;ac=c-a;n=np.cross(ab,ac);nn=float(n@n)
 if nn==0:return min(pseg_d2(p,a,b),pseg_d2(p,b,c),pseg_d2(p,c,a))
 signed=float((p-a)@n)/np.sqrt(nn);proj=p-signed*n/np.sqrt(nn)
 v0=ab;v1=ac;v2=proj-a;d00=float(v0@v0);d01=float(v0@v1);d11=float(v1@v1);d20=float(v2@v0);d21=float(v2@v1);den=d00*d11-d01*d01
 if den!=0:
  v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;u=1.-v-w
  if min(u,v,w)>=-1e-14:return signed*signed
 return min(pseg_d2(p,a,b),pseg_d2(p,b,c),pseg_d2(p,c,a))
def segseg_d2(p1,q1,p2,q2):
 u=q1-p1;v=q2-p2;w=p1-p2;a=float(u@u);b=float(u@v);c=float(v@v);d=float(u@w);e=float(v@w);D=a*c-b*b
 if a==0 and c==0:return float((p1-p2)@(p1-p2))
 if a==0:return pseg_d2(p1,p2,q2)
 if c==0:return pseg_d2(p2,p1,q1)
 sN=0.;sD=D;tN=0.;tD=D
 if D<1e-30:sN=0.;sD=1.;tN=e;tD=c
 else:
  sN=b*e-c*d;tN=a*e-b*d
  if sN<0:sN=0.;tN=e;tD=c
  elif sN>sD:sN=sD;tN=e+b;tD=c
 if tN<0:
  tN=0.
  if -d<0:sN=0.
  elif -d>a:sN=sD
  else:sN=-d;sD=a
 elif tN>tD:
  tN=tD
  if -d+b<0:sN=0.
  elif -d+b>a:sN=sD
  else:sN=-d+b;sD=a
 sc=0. if abs(sN)<1e-30 else sN/sD;tc=0. if abs(tN)<1e-30 else tN/tD
 r=w+sc*u-tc*v;return float(r@r)
def segtri_d2(a,b,t):
 d=min(ptri_d(a,t),ptri_d(b,t))
 for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):d=min(d,segseg_d2(a,b,u,v))
 return d
def main():
 if PINS['topology']=='' or PINS['witness_preflight']=='':raise RuntimeError('script pin constants need exact receipts')
 inputs={'candidate':sha(BASE),'audit159':sha(AUD159),'audit130':sha(AUD130),'self_pairs':sha(SELFP),'topology':sha(TOPO),'witness_preflight':sha(WIT),'bank_helper':sha(HELPER),'query_exporter':sha(EXPORT),'query_binary':sha(QUERYBIN)}
 if inputs!=PINS:raise ValueError(f'input hash mismatch: {inputs}')
 oldsha=sha(OLD_BASE)
 with np.load(BASE,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
 with np.load(OLD_BASE,allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
 if oldsha!='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af':raise ValueError(f'old patch topology source changed: {oldsha}')
 if set(old)!=set(Z) or not np.array_equal(old['source_O_faces'],Z['source_O_faces']):raise ValueError('current a69 topology cannot reuse the 248-face ancestry patch')
 TO=json.loads(TOPO.read_text());WI=json.loads(WIT.read_text())
 base_rows=sorted(set(map(int,TO['two_ring_patch_summary'][0]['face_rows']))|set(map(int,TO['fan_expansion'][0]['fan_face_rows']))|set(map(int,WI['measured_expansion']['added_face_rows'])))
 if len(base_rows)!=248 or hashlib.sha256(np.asarray(base_rows,np.int64).tobytes()).hexdigest()!='09d6f9205cfbc4c267607717a4358751d939882aa2e99759c73fb6d4b190b3fe':raise ValueError('248 patch row inventory mismatch')
 rows=sorted(set(map(int,json.loads(Path(ROWS_OVERRIDE).read_text())))) if ROWS_OVERRIDE else base_rows
 if not set(base_rows).issubset(rows) or any(r<0 or r>=42379 for r in rows):raise ValueError('expanded row set is not a superset of the pinned 248 patch')
 P32=np.asarray(Z['points_world_m'],np.float32);F=np.asarray(Z['source_O_faces'],np.int64);edges,bnodes=patch_boundary(F,rows)
 aud=json.loads(AUD159.read_text());bdoc=next(x for x in aud['banks'] if (int(x['body']),int(x['stable_bone_id']))==(65,99));cross_faces=sorted(set(int(x[0]) for x in bdoc['result']['triangle_pairs']));missing_cross=sorted(set(cross_faces)-set(rows))
 if missing_cross:raise ValueError(f'65/99 crossing faces omitted from patch: {missing_cross[:12]}')
 ind=np.asarray(Z['face_source_ancestry_indptr'],np.int64);anc=np.asarray(Z['face_source_ancestry_face_rows'],np.int64)
 missing=[int(r) for r in rows if ind[r+1]<=ind[r]]
 if missing:raise ValueError(f'patch faces lack source ancestry {missing[:8]}')
 # Shape baseline for current a69 patch, physical coordinates in metres.
 V=P32.astype(np.float64);angles=[];areas=[];aspects=[]
 for r in rows:
  t=V[F[r]];es=[np.linalg.norm(t[(i+1)%3]-t[i]) for i in range(3)];ar=np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))*.5;areas.append(float(ar));aspects.append(float(max(es)**2/(4*np.sqrt(3)*ar)))
  aa=[]
  for i in range(3):
   u=t[(i+1)%3]-t[i];v=t[(i+2)%3]-t[i];aa.append(float(np.degrees(np.arccos(np.clip((u@v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1)))))
  angles.append(min(aa))
 # Load exact static-bank inputs.
 helper=loadmod('boundary_bank_helper',HELPER)
 T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,exact=helper.read_bank_inputs(BANKDIR,E)
 if len(bankkeys)!=185:raise ValueError('bank key count not 185')
 # Boundary point classification: exact emitted world Float32 points against all 185 banks.
 qdir=OUT/'boundary-point-query-retry001';qdir.mkdir(exist_ok=False);qnpz=qdir/'points-world.npz';np.savez_compressed(qnpz,points_world_f32=P32[np.asarray(bnodes,np.int64)])
 qpoints=qdir/'points.csv';prep=qdir/'prepare.log';cp=subprocess.run([sys.executable,str(EXPORT),'--candidate-npz',str(qnpz),'--input-sha256',sha(qnpz),'--field','points_world_f32','--output',str(qpoints)],capture_output=True,text=True);prep.write_text(cp.stdout+cp.stderr)
 if cp.returncode:raise RuntimeError('query exporter failed: '+cp.stderr[-1200:])
 qpairs=qdir/'pairs.csv';qnodes=qdir/'nodes.csv';qlog=qdir/'query.log'
 cp=subprocess.run([str(QUERYBIN),str(BANKDIR/'exact-emitted.triangles.csv'),str(BANKDIR/'exact-emitted.bodies.csv'),str(qpoints),str(qpairs),str(qnodes),'185'],capture_output=True,text=True);qlog.write_text(cp.stdout+cp.stderr)
 if cp.returncode:raise RuntimeError('static query failed: '+cp.stderr[-1200:])
 point_rows=list(csv.DictReader(qpoints.open()));node_rows=list(csv.DictReader(qnodes.open()));pair_rows=list(csv.DictReader(qpairs.open()))
 if len(point_rows)!=len(bnodes) or len(node_rows)!=len(bnodes) or len(pair_rows)!=sum(int(r['parity_query_banks']) for r in node_rows):raise ValueError('boundary point query accounting incomplete')
 Xbits=P32.view(np.uint32)
 for i,(pr,nr) in enumerate(zip(point_rows,node_rows)):
  if int(pr['query_row'])!=i or int(nr['query_row'])!=i:raise ValueError('query order mismatch')
  if any(int(pr[k])!=int(Xbits[bnodes[i],j]) for j,k in enumerate(('x_bits_u32','y_bits_u32','z_bits_u32'))):raise ValueError('point query bits differ from candidate')
  if int(nr['total_bone_banks'])!=185 or int(nr['ambiguous_banks']) or int(nr['invalid_banks']):raise ValueError('boundary node query ambiguous/incomplete')
  if sum(int(nr[k]) for k in ('aabb_proven_outside_banks','parity_outside_banks','inside_banks','ambiguous_banks','invalid_banks'))!=185:raise ValueError('boundary node 185-bank accounting fail')
 inside=[]
 for r in pair_rows:
  if r['classification']=='inside':inside.append([int(bnodes[int(r['query_row'])]),int(r['bank_body']),int(r['bank_stable_id'])])
  elif r['classification']!='outside' or int(r['ambiguous']):raise ValueError('invalid boundary point-bank classification')
 # Exact boundary segment tests and conservative 0.05mm distance screen against all banks.
 world=P32.astype(np.float64);edge_reports=[];hits=[];uncertain=[];clearance_violations=[];min_global=float('inf');min_global_id=None;sum_aabb=0;exact_candidates=0
 tri_lows=T.min(axis=1);tri_highs=T.max(axis=1)
 for body,bone_id in bankkeys:
  body=int(body);bone_id=int(bone_id);m=(ids[:,0]==body)&(ids[:,1]==bone_id);bank=T[m];bids=ids[m,3].astype(np.int64)
  low=bank.min(axis=1);high=bank.max(axis=1)
  local=(world-pos[body])@M[body].T-off[body]
  for u,v in edges:
   a=local[u];b=local[v];seglo=np.minimum(a,b);seghi=np.maximum(a,b)
   near=np.flatnonzero(np.all(high>=seglo-CLEARANCE_M,axis=1)&np.all(low<=seghi+CLEARANCE_M,axis=1))
   if not len(near):continue
   # Exact crossing/boundary check on full bank using the shared owner helper.
   h=helper.exact_segment_triangle_hits(a,b,bank,bids,exact['classify_segment_triangle'])
   sum_aabb+=h['triangle_aabb_candidates'];exact_candidates+=h['exact_candidates']
   for x in h['hits']:
    item={'edge_nodes':[u,v],'bank':[body,bone_id],'bone_triangle_id':x['triangle_stable_id'],'classification':x['classification'],'reconstruction_error_m':x.get('hit_reconstruction_error_m')};hits.append(item)
   for x in h['uncertain']:
    uncertain.append({'edge_nodes':[u,v],'bank':[body,bone_id],'bone_triangle_id':x['triangle_stable_id'],'classification':x['classification']})
   best=float('inf');bestid=None
   for j in near:
    d2=segtri_d2(a,b,bank[int(j)])
    if d2<best:best=d2;bestid=int(bids[int(j)])
   dm=float(np.sqrt(max(0.,best)));sum_aabb+=0
   if dm<min_global:min_global=dm;min_global_id={'edge_nodes':[u,v],'bank':[body,bone_id],'bone_triangle_id':bestid}
   if dm<CLEARANCE_M:clearance_violations.append({'edge_nodes':[u,v],'bank':[body,bone_id],'bone_triangle_id':bestid,'distance_mm':dm*1000.})
   if dm<CLEARANCE_M or h['hits'] or h['uncertain']:edge_reports.append({'edge_nodes':[u,v],'bank':[body,bone_id],'distance_mm':dm,'minimum_distance_triangle_id':bestid,'aabb_candidates_in_clearance_box':len(near),'exact_hits':h['hits'],'exact_uncertain':h['uncertain']})
 # Must have a nearest candidate for every edge in at least one bank; otherwise min remains inf (meaning >= clearance by bounds).
 edge_has_near={tuple(r['edge_nodes']) for r in edge_reports}
 # The lower bound from AABB omission is exactly at least CLEARANCE_M. If an edge has no candidate for any bank, it is clear by coordinate separation.
 # Verify every patch face's parent ancestry continues to exist and typed anchors remain protected.
 anchors=np.asarray(Z['typed_coarse_anchor_output_rows_018'],np.int64);patch_nodes=np.unique(F[rows].ravel());patch_anchor=np.intersect1d(anchors,patch_nodes)
 report={'schema':'numi.human.b65-99.248-patch-boundary-remesh-preflight.v1','status':'BOUNDARY_SCREEN_COMPLETE','base_candidate_path':str(BASE),'base_candidate_sha256':sha(BASE),'old_patch_base_path':str(OLD_BASE),'old_patch_base_sha256':oldsha,'audit159_path':str(AUD159),'audit159_sha256':sha(AUD159),'audit130_path':str(AUD130),'audit130_sha256':sha(AUD130),'topology_receipt_sha256':sha(TOPO),'witness_expansion_receipt_sha256':sha(WIT),'source_patch_rows':rows,'source_patch_rows_sha256':hashlib.sha256(np.asarray(rows,np.int64).tobytes()).hexdigest(),'patch_face_count':len(rows),'patch_node_count':len(patch_nodes),'65_99_crossing_face_count':len(cross_faces),'65_99_crossing_faces_missing_from_patch':missing_cross,'rows_override_path':ROWS_OVERRIDE,'boundary_edges':[[int(u),int(v)] for u,v in edges],'boundary_edge_count':len(edges),'boundary_node_rows':bnodes,'boundary_node_count':len(bnodes),'patch_is_disk':True,'current_a69_face_topology_bitwise_same_as_old_patch_source':True,'only_base_archive_array_different_from_old_source':'points_world_m','patch_source_ancestry_missing_face_rows':missing,'patch_anchor_rows':patch_anchor.tolist(),'patch_anchor_count':len(patch_anchor),'current_patch_shape':{'min_angle_degrees':float(min(angles)),'p05_angle_degrees':float(np.percentile(angles,5)),'max_aspect_measure':float(max(aspects)),'min_area_m2':float(min(areas)),'max_area_m2':float(max(areas))},'boundary_point_screen':{'point_count':len(bnodes),'bank_count_per_node':[int(r['total_bone_banks']) for r in node_rows],'classification_rows':node_rows,'inside_node_bank_pairs':inside,'inside_count':len(inside),'ambiguous_or_invalid_classifications':0,'all_points_accounted_against_185_banks':True,'all_boundary_points_outside_all_banks':len(inside)==0},'boundary_edge_screen':{'clearance_threshold_mm':CLEARANCE_M*1000,'edge_count':len(edges),'unique_banks':len(bankkeys),'exact_hit_count':len(hits),'exact_hits':hits,'uncertain_count':len(uncertain),'uncertain':uncertain,'min_measured_edge_to_bone_triangle_distance_mm':None if min_global==float('inf') else min_global*1000,'min_distance_witness':min_global_id,'edge_clearance_violation_count':len(clearance_violations),'edge_clearance_violations':clearance_violations,'aabb_candidates_for_exact_segment_filter':sum_aabb,'exact_segment_candidates':exact_candidates,'all_boundary_edges_bone_hit_free':len(hits)==0 and len(uncertain)==0,'all_boundary_edges_clearance_ge_0p05mm':len(clearance_violations)==0},'boundary_gate_pass':len(inside)==0 and len(hits)==0 and len(uncertain)==0 and len(clearance_violations)==0,'remesh_construction_allowed_by_boundary_gate':len(inside)==0 and len(hits)==0 and len(uncertain)==0 and len(clearance_violations)==0,'limits':['point location screens the 62 patch-boundary nodes against all 185 typed bones','edge hit screening uses exact emitted-segment predicates after broad phase; reported positive distance is double-precision feature distance with 0.05mm expanded-AABB rejection','no candidate remesh or shape test has been run','no all-bank full-surface audit, self, skin-fat, physical or native admission']}
 out=OUT/'preflight.json';out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({k:report[k] for k in ('status','patch_face_count','patch_node_count','boundary_edge_count','patch_source_ancestry_missing_face_rows','patch_anchor_count','current_patch_shape','boundary_point_screen','boundary_edge_screen','boundary_gate_pass','remesh_construction_allowed_by_boundary_gate')},indent=2,allow_nan=False));print('report',out,'sha256',sha(out))
if __name__=='__main__':main()
