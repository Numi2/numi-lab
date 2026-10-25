from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
OUTROOT=ROOT/'connected-boundary-refinement-114'; TOOL=OUTROOT/'matter/tools'
Q=ROOT/'digit-source-diagnosis-110/batch-146-curved-source-midpoints-quality-114.json'
B=ROOT/'digit-source-diagnosis-110/batch-145-curved-source-midpoints-004.json'
ZP=ROOT/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
RP=ROOT/'connected-capacity-reference-115/candidate.npz'
MM=ROOT/'whole-reference-tetgen-084-bounded/facet-marker-map.csv'
OR=ROOT/'digit-source-diagnosis-110/source-path-orientation-audit-112.json'
OUT=OUTROOT/'run-002-source-normal-all-146'
PINS={'quality114':'aa683f25a629f867d20decaa5a8222535142901c2807ba2361748605979b741a','batch145':'e4c6c76d497e176e40ad802cb3159a4a143279cc131ae632fe515498aa38fafa','compact084':'5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db','reference115':'abff4290774cc96a17f6c68a21fe86fc3933a1878e0bd6512733e99589fb6dd9','marker_map':'8e7ac9430c037d29e74d4c2f7f9f838995f6541c0f5fdf29a7b7095d9a6d8ff7','orientation112':'e43908188f134182635ac36a9a1602abad1918df374a04f3f7cc9eb1812d7497'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def ash(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(a.dtype.str.encode());h.update(json.dumps(list(a.shape),separators=(',',':')).encode());h.update(a.tobytes());return h.hexdigest()
def unit(v):
 v=np.asarray(v,dtype=np.float64);n=float(np.linalg.norm(v))
 if not np.isfinite(n) or n<=1e-30: raise RuntimeError('zero/nonfinite normal')
 return v/n
def main():
 if OUT.exists(): raise RuntimeError(f'refusing existing output: {OUT}')
 actual={k:sha(p) for k,p in [('quality114',Q),('batch145',B),('compact084',ZP),('reference115',RP),('marker_map',MM),('orientation112',OR)]}
 if actual!=PINS: raise RuntimeError(f'input pin mismatch: {actual}')
 q=json.loads(Q.read_text());old=json.loads(B.read_text()); orient=json.loads(OR.read_text())
 if len(q['edges'])!=146 or len(old['edges'])!=145: raise RuntimeError('wrong edge inventory')
 if orient.get('counts',{}).get('orientable_components') not in (None,1) and orient.get('component_count') not in (None,1): raise RuntimeError('orientation audit is no longer single-component')
 oldmap={tuple(map(int,e['edge_output_rows'])):e for e in old['edges']}
 records=[]
 for i,e in enumerate(q['edges']):
  key=tuple(map(int,e['edge_output_rows']))
  r=dict(e)
  if i<145:
   o=oldmap.get(key)
   if o is None: raise RuntimeError(f'missing old explicit layer mapping for {key}')
   r['_old']=o
   r['edge_global_node_ids']=o['edge_global_node_ids']
   r['source_path_vertex_ids']=o['source_path_vertex_ids']
   r['path_selected_source_face_rows']=o['path_selected_source_face_rows']
   r['parent_source_O_face_rows']=o['parent_source_O_face_rows']
  else:r['_old']=None
  if not r.get('source_normal_proposal_eligible',False): raise RuntimeError(f'edge {i} not source-normal eligible')
  records.append(r)
 z=np.load(ZP,allow_pickle=False); r115=np.load(RP,allow_pickle=False)
 Pref=np.asarray(r115['proposed_reference_points_m']); Pcur=np.asarray(r115['points_world_f32'])
 if Pref.shape!=(7360,3) or Pref.dtype!=np.float64 or Pcur.shape!=(7360,3) or Pcur.dtype!=np.float32: raise RuntimeError('115 arrays changed')
 if not np.array_equal(r115['source_reference_points_m'],z['reference_points_m']) or not np.array_equal(r115['source_points_world_f32'],z['current037_points_m_f32']): raise RuntimeError('115 source prefix mismatch')
 faces={l:np.asarray(z[f'source_{l}_faces'],np.int64) for l in 'OID'}
 nodes={l:np.unique(faces[l]) for l in 'OID'}; offset={l:i*2411 for i,l in enumerate('OID')}
 for i,l in enumerate('OID'):
  if not np.array_equal(nodes[l],np.arange(i*2411,(i+1)*2411)): raise RuntimeError(f'{l} ordered node domain changed')
  if not np.array_equal(np.asarray(z['source_input_row'])[nodes[l]],np.arange(i*2411,(i+1)*2411)): raise RuntimeError(f'{l} source_input_row crosswalk mismatch')
 if not (np.array_equal(faces['I']-offset['I'],faces['O']) and np.array_equal(faces['D']-offset['D'],faces['O'])): raise RuntimeError('O/I/D face correspondence changed')
 # Map markers by exact source face row, preserve all 180 construction caps unchanged.
 marker_rows=list(csv.DictReader(MM.open(newline='')));code={'O':1,'I':2,'D':3};mp={};caps=[]
 for rec in marker_rows:
  s=rec['surface']; marker=int(rec['marker']); pr=int(rec['source_face_row']); fv=tuple(map(int,rec['vertex_ids_zero_based'].split()))
  if s in code:
   if not 0<=pr<len(faces[s]) or sorted(fv)!=sorted(map(int,faces[s][pr])) or marker!=code[s]*100000+pr+1: raise RuntimeError(f'marker mismatch {s}/{pr}')
   mp[(s,pr)]=marker
  elif s.startswith('SIDE-'): caps.append((marker,s,pr,fv))
  else: raise RuntimeError(f'unrecognized marker surface {s}')
 markers={l:np.asarray([mp[(l,j)] for j in range(len(faces[l]))],np.int64) for l in 'OID'}
 if len(caps)!=180: raise RuntimeError('cap count changed')
 caps.sort();cap_faces=np.asarray([x[3] for x in caps],np.int32);cap_markers=np.asarray([x[0] for x in caps],np.int32);cap_parent=np.asarray([x[2] for x in caps],np.int64)
 edges=[]; global_cols=[]; normals=[]; endpoint_depths={}; constructions={}; seen=set()
 for i,r in enumerate(records):
  u,v=map(int,r['edge_output_rows']); k=tuple(sorted((u,v)))
  if k in seen or u==v or min(k)<0 or max(k)>=2411: raise RuntimeError(f'invalid/duplicate edge {k}')
  seen.add(k);edges.append((u,v))
  reported_cols=tuple(map(int,r['edge_global_node_ids']))
  actual_cols=tuple(map(int,np.asarray(z['source_global_node_ids'])[[u,v]]))
  if i<145 and actual_cols!=reported_cols: raise RuntimeError(f'global-node join failed for source edge {k}')
  # The quality-only row in the 114 report labels [345,2333], while its explicit
  # endpoint/source rows are [344,2332]. Preserve that unresolved label discrepancy;
  # topology uses output rows and emitted IDs come from the pinned compact source table.
  global_cols.append(actual_cols)
  normals.append({})
  endpoint_depths[str(i)]={}
  constructions[str(i)]={}
  for frame in ('reference','current037'):
   prop=r[f'normal_scalar_source_proposal_{frame}']; n=unit(prop['outward_normal']); normals[i][frame]=n
   P=Pref if frame=='reference' else Pcur
   Orows=np.array([u,v]); Irows=Orows+offset['I']; Drows=Orows+offset['D']
   Oe=P[Orows].astype(np.float64); Ie=P[Irows].astype(np.float64); De=P[Drows].astype(np.float64)
   sd=-np.einsum('ij,j->i',Ie-Oe,n); fd=-np.einsum('ij,j->i',De-Ie,n)
   endpoint_depths[str(i)][frame]=np.stack([sd,fd],axis=1)
   # Preserve two signed endpoint projections; midpoint is a half-arclength source interpolation.
   skin=float(sd.mean()); fat=float(fd.mean()); outer=np.asarray(prop['O_m'],np.float64)
   O=outer; I=O-skin*n; D=O-(skin+fat)*n
   constructions[str(i)][frame]={'O':O,'I':I,'D':D,'skin_depth_m':skin,'fat_depth_m':fat,'normal':n}
 # Global points appended as row triples O/I/D; use source-normal proposals for O and signed projected layer scalar field for I/D.
 n0=len(Pref); mids={l:np.asarray([n0+3*i+j for i in range(146)],np.int64) for j,l in enumerate('OID')}
 points_by_frame={}; edge_map={l:{} for l in 'OID'}; prov={f:{l:{} for l in 'OID'} for f in ('reference','current037')}
 for frame,base in [('reference',Pref),('current037',Pcur)]:
  append=[]
  for i,r in enumerate(records):
   k=tuple(sorted(edges[i])); c=constructions[str(i)][frame]
   for j,l in enumerate('OID'):
    raw=c[l]; emitted=raw.astype(base.dtype); append.append(emitted)
    sd,fd=endpoint_depths[str(i)][frame][:,0],endpoint_depths[str(i)][frame][:,1]
    source_faces=tuple(map(int,r.get('path_selected_source_face_rows',r.get('source_face_rows',[]))))
    path_ids=tuple(map(int,r.get('source_path_vertex_ids',r.get('path_detail_rows',[]))))
    if not source_faces or len(path_ids)<2: raise RuntimeError(f'missing source path provenance on edge {i}')
    rec={'source_kind':'detailed_skin_source_normal_scalar','layer':l,'sample_point_world_m':emitted.astype(np.float64).tolist(),
      'outer_midpoint_world_m':c['O'].tolist(),'outward_normal':c['normal'].tolist(),'skin_depth_m':c['skin_depth_m'],'fat_depth_m':c['fat_depth_m'],
      'endpoint_depths_m':endpoint_depths[str(i)][frame].tolist(),'source_face_rows':list(source_faces),'source_path_vertex_ids':list(path_ids),
      'source_path_endpoint_vertex_ids':[int(path_ids[0]),int(path_ids[-1])],'reference_field_sha256':ash(Pref),
      'input_edge_output_rows':list(edges[i]),'source_global_column_ids':list(global_cols[i]),'source_column_ids':list(global_cols[i]),
      'construction_semantics':'fine-source-normal O sample; I/D placed by signed endpoint projections of existing 115 OI/ID offsets; midpoint interpolation; no clamping'}
    el=tuple(sorted((edges[i][0]+offset[l],edges[i][1]+offset[l])))
    edge_map[l][el]=int(mids[l][i]);prov[frame][l][el]=rec
  points_by_frame[frame]=np.concatenate([base.copy(),np.asarray(append,dtype=base.dtype).reshape(-1,3)],axis=0)
 # Parent-normal expectations: source component outward normal has coarse indexed winding parity -1.
 incidence={l:{} for l in 'OID'}
 for l in 'OID':
  for fi,f in enumerate(faces[l]):
   for a,b in ((int(f[0]),int(f[1])),(int(f[1]),int(f[2])),(int(f[2]),int(f[0]))): incidence[l].setdefault(tuple(sorted((a,b))),[]).append((fi,a,b))
 for i in range(146):
  for l in 'OID':
   uses=incidence[l].get(tuple(sorted((edges[i][0]+offset[l],edges[i][1]+offset[l]))),[])
   if len(uses)!=2 or uses[0][1:]!=uses[1][1:][::-1]: raise RuntimeError(f'{l} edge incidence/orientation failed for {edges[i]}')
 def parent_normals(frame,l):
  P=points_by_frame[frame]; F=faces[l]; out=[]
  for f in F:
   x=P[f].astype(np.float64); out.append(unit(np.cross(x[1]-x[0],x[2]-x[0])))
  out=np.asarray(out); used={}
  for i in range(146):
   k=tuple(sorted((edges[i][0]+offset[l],edges[i][1]+offset[l])))
   for fi,_,_ in incidence[l][k]: used.setdefault(fi,[]).append(-normals[i][frame])
  conflicts=[]
  for fi,vs in used.items():
   s=np.sum(vs,axis=0)
   if np.linalg.norm(s)<1e-10: conflicts.append(int(fi)); s=vs[0]
   out[fi]=unit(s)
  return out,conflicts
 sys.path.insert(0,str(TOOL)); from human_skin_fat_boundary_refinement import subdivide_oriented_surface_edges
 results={};normal_conflicts={}
 for frame in ('reference','current037'):
  for l in 'OID':
   en,conf=parent_normals(frame,l);normal_conflicts[f'{frame}_{l}']=conf
   results[(frame,l)]=subdivide_oriented_surface_edges(points_by_frame[frame],faces[l],edge_map[l],prov[frame][l],original_point_count=n0,
       parent_face_markers=markers[l],layer=l,expected_midpoint_rows=mids[l],expected_parent_normals=en,collect_orientation_failures=True)
 for l in 'OID':
  if not np.array_equal(results[('reference',l)].faces,results[('current037',l)].faces): raise RuntimeError(f'{l} topology differs across frames')
 all_faces=np.concatenate([results[('reference',l)].faces.astype(np.int32) for l in 'OID']+[cap_faces],axis=0)
 all_markers=np.concatenate([results[('reference',l)].child_parent_face_markers.astype(np.int32) for l in 'OID']+[cap_markers])
 all_parent=np.concatenate([results[('reference',l)].child_parent_face_rows.astype(np.int64) for l in 'OID']+[cap_parent])
 all_sid=np.concatenate([np.full(len(results[('reference',l)].faces),i,np.int8) for i,l in enumerate('OID')]+[np.full(len(cap_faces),3,np.int8)])
 if len(all_faces)!=len(all_markers) or len(all_faces)!=len(all_parent) or np.any(all_faces<0) or np.any(all_faces>=len(points_by_frame['reference'])): raise RuntimeError('invalid combined PLC')
 pathptr=[0];pathflat=[];faceptr=[0];faceflat=[]
 for r in records:
  path=tuple(map(int,r.get('source_path_vertex_ids',r.get('path_detail_rows',[]))));sf=tuple(map(int,r.get('path_selected_source_face_rows',r.get('source_face_rows',[]))))
  pathflat.extend(path);pathptr.append(len(pathflat));faceflat.extend(sf);faceptr.append(len(faceflat))
 OUT.mkdir(parents=True)
 cand=OUT/'surface-proposal.npz'
 np.savez_compressed(cand,reference_points_m=points_by_frame['reference'],current037_points_m_f32=points_by_frame['current037'],
   surface_faces_zero_based=all_faces,surface_face_markers=all_markers,surface_face_parent_rows=all_parent,surface_face_surface_ids=all_sid,
   source_O_faces=results[('reference','O')].faces.astype(np.int32),source_I_faces=results[('reference','I')].faces.astype(np.int32),source_D_faces=results[('reference','D')].faces.astype(np.int32),
   source_O_child_parent_rows=results[('reference','O')].child_parent_face_rows,source_I_child_parent_rows=results[('reference','I')].child_parent_face_rows,source_D_child_parent_rows=results[('reference','D')].child_parent_face_rows,
   source_O_child_parent_markers=results[('reference','O')].child_parent_face_markers,source_I_child_parent_markers=results[('reference','I')].child_parent_face_markers,source_D_child_parent_markers=results[('reference','D')].child_parent_face_markers,
   selected_outer_edges=np.asarray(edges,np.int64),selected_edge_global_column_ids=np.asarray(global_cols,np.int64),
   selected_edge_layer_endpoint_rows=np.asarray([[ [edges[i][0]+offset[l],edges[i][1]+offset[l]] for l in 'OID'] for i in range(146)],np.int64),
   selected_edge_midpoint_rows=np.asarray([[mids[l][i] for l in 'OID'] for i in range(146)],np.int64),
   selected_edge_reference_outward_normals=np.asarray([normals[i]['reference'] for i in range(146)]),selected_edge_current037_outward_normals=np.asarray([normals[i]['current037'] for i in range(146)]),
   selected_edge_reference_signed_endpoint_depths=np.asarray([endpoint_depths[str(i)]['reference'] for i in range(146)]),selected_edge_current037_signed_endpoint_depths=np.asarray([endpoint_depths[str(i)]['current037'] for i in range(146)]),
   source_path_indptr=np.asarray(pathptr,np.int64),source_path_detail_vertex_ids=np.asarray(pathflat,np.int64),source_face_indptr=np.asarray(faceptr,np.int64),selected_detail_source_face_rows=np.asarray(faceflat,np.int64),
   cap_faces_zero_based=cap_faces,cap_face_markers=cap_markers,cap_face_parent_rows=cap_parent)
 orientation={f'{frame}_{l}':results[(frame,l)].orientation_diagnostics for frame in ('reference','current037') for l in 'OID'}
 negative={frame:{'skin_endpoint':int(np.sum(np.asarray([endpoint_depths[str(i)][frame][:,0] for i in range(146)])<0)), 'fat_endpoint':int(np.sum(np.asarray([endpoint_depths[str(i)][frame][:,1] for i in range(146)])<0))} for frame in ('reference','current037')}
 report={'status':'raw_connected_source_normal_surface_proposal','interpretation':'146-edge surface-only conforming subdivision. O midpoints follow detailed NHSKIN source paths; I/D midpoints use signed projections of the existing 115 OI/ID fields onto source normals. No clamping. This is a reconstruction proposal, not measured thickness or an admitted geometry.',
   'input_sha256':actual,'builder_sha256':sha(Path(__file__)),'refiner_sha256':sha(TOOL/'human_skin_fat_boundary_refinement.py'),'array_sha256':{'115_reference':ash(Pref),'115_current037':ash(Pcur)},
   'inventory':{'crossing_source_edges':145,'quality_only_edge':1,'used_edges':146,'held_edges':0,'source_outward_to_coarse_face_winding_parity':-1,'O_I_D_face_topology_exact':True,'endpoint_depths_signed_recomputed_from_115':True,'negative_endpoint_projection_counts_115':negative},
   'topology':{'input_nodes':int(n0),'appended_nodes':438,'original_reference_prefix_exact':bool(np.array_equal(points_by_frame['reference'][:n0],Pref)),'original_current037_prefix_exact':bool(np.array_equal(points_by_frame['current037'][:n0],Pcur)),
     'input_faces_per_layer':{l:len(faces[l]) for l in 'OID'},'output_faces_per_layer':{l:len(results[('reference',l)].faces) for l in 'OID'},'unchanged_caps':len(cap_faces),'combined_faces':len(all_faces),
     'oriented_boundary_chain_preserved':all(results[(f,l)].orientation_diagnostics['oriented_boundary_chain_preserved'] for f in ('reference','current037') for l in 'OID'),'tetrahedra_included':False},
   'orientation_mode':{'expected_normal':'fine NHSKIN outward, corrected by audited fine-to-coarse winding parity -1','default_parent_normal_gate_changed':False,'orientation_failures_retained':True,'normal_mean_cancellation_faces':normal_conflicts,'diagnostics_by_frame_layer':orientation,'all_child_orientation_pass':all(x['failed_child_count']==0 for x in orientation.values())},
   'limitations':['No TetGen or tetrahedral rebuild.','No capacity, full-bank containment, mapped self-intersection, positive-volume, mass-transfer, native, or physical acceptance gate.','The projected OI/ID scalar layer samples are explicitly inferred from existing 115 offsets and source normals; signed negative endpoint projections are retained without clamping.','Source-normal local triangle checks do not establish global embeddedness.'],
   'candidate_sha256':sha(cand),'candidate_size_bytes':cand.stat().st_size}
 rp=OUT/'proposal-report.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 mf={'inputs':actual,'builder_sha256':sha(Path(__file__)),'refiner_sha256':sha(TOOL/'human_skin_fat_boundary_refinement.py'),'candidate_sha256':sha(cand),'report_sha256':sha(rp),'status':'raw proposal; downstream gates not run'}
 m=OUT/'manifest.json';m.write_text(json.dumps(mf,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'out':str(OUT),'candidate':str(cand),'candidate_sha256':sha(cand),'report':str(rp),'report_sha256':sha(rp),'manifest':str(m),'manifest_sha256':sha(m),'nodes':len(points_by_frame['reference']),'faces':len(all_faces),'orientation_failures':{k:v['failed_child_count'] for k,v in orientation.items()},'minimum_cosines':{k:v['minimum_signed_cosine'] for k,v in orientation.items()},'boundary_chain':report['topology']['oriented_boundary_chain_preserved']},indent=2))
if __name__=='__main__':main()
