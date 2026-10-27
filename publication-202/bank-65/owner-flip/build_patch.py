from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
B=S/'per-bank-workers-20261027/b65-99'; BASE=B/'root-layer-guard-002/backtrack-0p25/candidate.npz'
PATCH=B/'attempt-021-boundary-expansion-001/run-001/iteration-002/rows.json'
BOUND=B/'attempt-021-boundary-expansion-001/run-001/iteration-002/preflight.json'
A159=B/'root-layer-guard-002/backtrack-0p25/audit-159/report.json'
OWNER=Path('/Users/numi/Work/numi-human-connected-surface-069/D/local-two-parent-retriangulation-402/two_parent_patch_flips_402.py')
OUT=Path(__file__).resolve().parent/'run-001'; PY='/Users/numi/numi-human-anatomy-finish-1378/.venv/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def boundary(faces,rows):
 em={}
 for fi in map(int,rows):
  t=faces[fi]
  for a,b in zip(t,np.roll(t,-1)):
   e=(min(int(a),int(b)),max(int(a),int(b)));em.setdefault(e,0);em[e]+=1
 return sorted(e for e,n in em.items() if n==1)
def metric(x,f,owner):
 area=[];ang=[];asp=[]
 for t in f:
  _,a=owner.normal_area(x,t);p=np.asarray(x[t],np.float64);l=[np.linalg.norm(p[i]-p[(i+1)%3]) for i in range(3)]
  area.append(a);ang.append(owner.min_angle(x,t));asp.append(max(l)**2/(2*a) if a>0 else float('inf'))
 return {'min_area_m2':float(min(area)),'min_angle_deg':float(min(ang)),'p05_angle_deg':float(np.percentile(ang,5)),'max_aspect_longest_edge_sq_over_2area':float(max(asp))}
def main():
 OUT.mkdir(exist_ok=False);pins={'base':sha(BASE),'patch_rows':sha(PATCH),'boundary':sha(BOUND),'owner':sha(OWNER),'audit159':sha(A159)}
 if pins['base']!='a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db':raise ValueError('base hash mismatch')
 owner=mod('owner402_b65_patch',OWNER);c=load(BASE);F=c['source_O_faces'];n=len(F)
 rows=np.asarray(json.loads(PATCH.read_text()),np.int64);br=json.loads(BOUND.read_text())
 b0=sorted(tuple(map(int,e)) for e in br['boundary_edges'])
 if len(rows)!=261 or boundary(F,rows)!=b0:raise ValueError('patch or boundary pin mismatch')
 ar=json.loads(A159.read_text());bank=next(b for b in ar['banks'] if (int(b['body']),int(b['stable_bone_id']))==(65,99))
 raw={(int(a),int(t)) for a,t in bank['result']['triangle_pairs']}
 detail=np.asarray(c['parent_fine_detail_face_row']);g2l={int(g):i for i,g in enumerate(rows)}
 seedg=sorted({f for f,t in raw if f in g2l and int(detail[f]) in (5530,6103)})
 seeds=sorted(g2l[f] for f in seedg)
 sub={};face_keys=('source_O_faces','area_reference_037_source_O_faces','face_kind','parent_fine_detail_face_row','parent_fine_subface_row','parent_coarse_face_row','parent_source_face_marker','refinement_face_parent_face_row','refinement_face_parent_barycentric')
 for k in face_keys:sub[k]=np.array(c[k][rows],copy=True)
 for k in ('points_world_m','area_reference_037_points_world_m_f64'):sub[k]=np.array(c[k],copy=True)
 csr=(('face_source_ancestry_indptr','face_source_ancestry_face_rows'),('face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows'),('refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows'))
 for pk,vk in csr:
  old=owner.csr_rows(c[pk],c[vk],n);p,v=owner.csr_build([old[int(i)] for i in rows],np.asarray(c[vk]).dtype);sub[pk]=p;sub[vk]=v
 subout,rep=owner.propose_patch_flips(sub,[5530,6103],seeds,max_flips=8,max_rounds=1,min_angle_gain_deg=1e-8,min_normal_dot=0.0)
 ops=rep['operations'];changed=rows[np.asarray(rep['changed_face_rows'],np.int64)]
 rec={'schema':'numi.human.b65-99.boundary-owner-flip-construction.v1','pins':pins,'base_path':str(BASE),'expanded_patch_rows_path':str(PATCH),'expanded_boundary_path':str(BOUND),'patch_face_count':len(rows),'patch_boundary_edge_count':len(b0),'patch_detail_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(detail[rows],return_counts=True))},'seed_global_rows':seedg,'seed_count':len(seeds),'baseline_65_99_raw_pairs':len(raw),'owner_path':str(OWNER),'owner_report':rep,'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0}
 if not ops:
  rec.update(status='STOP_NO_SHAPE_SAFE_LOCAL_FLIP',candidate_constructed=False)
  (OUT/'receipt.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
  print(json.dumps({'status':rec['status'],'seed_count':len(seeds),'rejections':rep['rejection_reason_counts']}));return
 out={k:np.array(v,copy=True) for k,v in c.items()}
 for k in face_keys:out[k][rows]=subout[k]
 for pk,vk in csr:
  old=owner.csr_rows(c[pk],c[vk],n);new=owner.csr_rows(subout[pk],subout[vk],len(rows))
  for j,g in enumerate(rows):old[int(g)]=new[j]
  out[pk],out[vk]=owner.csr_build(old,np.asarray(c[vk]).dtype)
 for k in ('refinement_face_parent_ancestry_semantics','refinement_face_parent_barycentric_semantics'):
  if k in subout:out[k]=np.array(subout[k],copy=True)
 obs=np.flatnonzero(np.any(out['source_O_faces']!=F,axis=1))
 if not np.array_equal(obs,changed):raise ValueError('changed rows do not match owner report')
 if not np.array_equal(out['points_world_m'],c['points_world_m']):raise ValueError('point coords changed')
 if boundary(out['source_O_faces'],rows)!=b0:raise ValueError('expanded boundary changed')
 if not np.array_equal(out['area_reference_037_source_O_faces'],out['source_O_faces']):raise ValueError('037 connectivity mismatch')
 par0=owner.csr_rows(c[csr[2][0]],c[csr[2][1]],n);par1=owner.csr_rows(out[csr[2][0]],out[csr[2][1]],n)
 src0=owner.csr_rows(c[csr[0][0]],c[csr[0][1]],n);src1=owner.csr_rows(out[csr[0][0]],out[csr[0][1]],n)
 sf0=owner.csr_rows(c[csr[1][0]],c[csr[1][1]],n);sf1=owner.csr_rows(out[csr[1][0]],out[csr[1][1]],n)
 ancestry=True
 for op in ops:
  i,j=(int(rows[x]) for x in op['face_rows'])
  for r in (i,j):
   ancestry &= par1[r]==sorted(set(par0[i])|set(par0[j]))
   ancestry &= src1[r]==sorted(set(src0[i])|set(src0[j]))
   ancestry &= sf1[r]==sorted(set(sf0[i])|set(sf0[j]))
 if not ancestry:raise ValueError('CSR ancestry union mismatch')
 q0=metric(c['points_world_m'],F[rows],owner);q1=metric(out['points_world_m'],out['source_O_faces'][rows],owner)
 r0=metric(c['area_reference_037_points_world_m_f64'],F[rows],owner);r1=metric(out['area_reference_037_points_world_m_f64'],out['source_O_faces'][rows],owner)
 # Require positive areas and no patch min-area, min-angle or max-aspect regression.
 shape=q1['min_area_m2']>=q0['min_area_m2']*(1-1e-10) and q1['min_angle_deg']+1e-12>=q0['min_angle_deg'] and q1['max_aspect_longest_edge_sq_over_2area']<=q0['max_aspect_longest_edge_sq_over_2area']+1e-10
 shape &= r1['min_area_m2']>=r0['min_area_m2']*(1-1e-10) and r1['min_angle_deg']+1e-12>=r0['min_angle_deg'] and r1['max_aspect_longest_edge_sq_over_2area']<=r0['max_aspect_longest_edge_sq_over_2area']+1e-10
 shape &= all(owner.normal_area(out['points_world_m'],out['source_O_faces'][i])[1]>0 for i in changed)
 shape &= all(owner.normal_area(out['area_reference_037_points_world_m_f64'],out['source_O_faces'][i])[1]>0 for i in changed)
 rec.update({'changed_face_rows':changed.tolist(),'changed_face_count':len(changed),'flip_count':len(ops),'operations':ops,'coordinates_bitwise_unchanged':True,'boundary_set_unchanged':True,'ancestry_union_verified':bool(ancestry),'shape_before':q0,'shape_after':q1,'matched037_shape_before':r0,'matched037_shape_after':r1,'shape_gate_pass':bool(shape),'candidate_constructed':bool(shape)})
 if not shape:
  rec['status']='STOP_SHAPE_GATE_FAIL';np.savez_compressed(OUT/'candidate-rejected-shape.npz',**out);rec['rejected_candidate_sha256']=sha(OUT/'candidate-rejected-shape.npz')
  (OUT/'receipt.json').write_text(json.dumps(rec,indent=2,sort_keys=True,allow_nan=False)+'\n')
  print(json.dumps({'status':rec['status'],'changed_rows':changed.tolist(),'shape_before':q0,'shape_after':q1},indent=2));return
 cand=OUT/'candidate.npz';np.savez_compressed(cand,**out);rec.update(status='CANDIDATE_READY_FOR_EXACT_SCREEN',candidate_path=str(cand),candidate_sha256=sha(cand))
 (OUT/'receipt.json').write_text(json.dumps(rec,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':rec['status'],'candidate_sha256':rec['candidate_sha256'],'changed_face_rows':changed.tolist(),'flip_count':len(ops),'shape_before':q0,'shape_after':q1},indent=2))
if __name__=='__main__':main()
