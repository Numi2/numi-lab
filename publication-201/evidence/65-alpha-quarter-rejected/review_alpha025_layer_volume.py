from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
D=Path(__file__).resolve().parent
B=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25')
C=D/'fit-run-008-alpha-0p25-corrected/candidate-private-alpha-0p25.npz'
R=D/'fit-run-008-alpha-0p25-corrected/layer-identity-diff.json'
L=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/physical-layer-diagnostic-001/guarded-65-quarter/skin-fat-layers-541.npz')
OUT=D/'scoped-exact-run-001/layer-volume-review.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 pins={'base':sha(B/'candidate.npz'),'candidate':sha(C),'layer_report':sha(R),'layer':sha(L)}
 if pins!={'base':'a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db','candidate':'e51ff27acd6a29eefcb30783085c3f8af3b36b12ffd55d43ee714f1b872b380a','layer_report':'7cee89a174863bd79a6d4f625cf15a1211be1ff6cc0f7bb661406a9295525b86','layer':'78916a0c05eb130e5efb02174342295c2d8d4444eae963c7b36fc43a2a1b17fc'}:raise ValueError(pins)
 tree=ast.parse(Path(__file__).with_name('retain_alpha_0p25_layer_diff_corrected.py').read_text());fns=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('tetvol','edge_longest','cell_metrics')];ns={'np':np};exec(compile(ast.Module(body=fns,type_ignores=[]),'<cell_metrics>','exec'),ns);cell_metrics=ns['cell_metrics']
 rep=json.loads(R.read_text());rows=np.asarray(rep['changed_face_rows'],np.int64)
 with np.load(B/'candidate.npz',allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
 with np.load(C,allow_pickle=False) as z:cand={k:z[k].copy() for k in z.files}
 with np.load(L,allow_pickle=False) as z:
  faces=np.asarray(z['outer_faces'],np.int64);outer0=np.asarray(z['outer_points_f64'],np.float64);inner0=np.asarray(z['inner_points_f64'],np.float64);deep0=np.asarray(z['deep_points_f64'],np.float64)
  outer032=np.asarray(z['outer_points_f32'],np.float32);inner032=np.asarray(z['inner_points_f32'],np.float32);deep032=np.asarray(z['deep_points_f32'],np.float32)
  sf=np.asarray(z['skin_thickness_transferred_m'],np.float64);ff=np.asarray(z['fat_thickness_transferred_m'],np.float64)
  P=base['points_world_m'];Q=cand['points_world_m'];delta=Q.astype(np.float64)-P.astype(np.float64)
  outer1=Q.astype(np.float64);inner1=inner0+delta;deep1=deep0+delta
  outer132=Q.astype(np.float32);inner132=inner1.astype(np.float32);deep132=deep1.astype(np.float32)
  inputs=[('skin',outer0,inner0,outer032,inner032,outer1,inner1,outer132,inner132,sf),('fat',inner0,deep0,inner032,deep032,inner1,deep1,inner132,deep132,ff)]
 result=[]
 for name,o0,i0,o032,i032,o1,i1,o132,i132,field in inputs:
  before=cell_metrics(o0,i0,o032,i032,faces,rows,field);after=cell_metrics(o1,i1,o132,i132,faces,rows,field)
  bm={(x['face_row'],x['tet_index']):x for x in before['records']};am={(x['face_row'],x['tet_index']):x for x in after['records']}
  bad=[]
  for (fr,ti),x in bm.items():
   abc=np.sort(faces[fr]);th=max(1e-8,float(np.max(field[abc])))**3*1e-12*1e9
   authored_bad=(not np.isfinite(x['authored_oriented_f64_volume_mm3']) or x['authored_oriented_f64_volume_mm3']<=th)
   owner_nonpos=x['owner_f32_oriented_volume_mm3']<=0
   owner_cond=x['owner_f32_normalized_det']<1e-5
   if authored_bad or owner_nonpos or owner_cond:
    y=am[(fr,ti)]
    bad.append({'face_row':int(fr),'tet_index':int(ti),'baseline_invalid_categories':[n for n,v in [('authored_f64_orientation_or_floor',authored_bad),('owner_f32_nonpositive',owner_nonpos),('owner_f32_conditioned_lt_1e-5',owner_cond)] if v], 'baseline_authored_oriented_f64_mm3':x['authored_oriented_f64_volume_mm3'],'candidate_authored_oriented_f64_mm3':y['authored_oriented_f64_volume_mm3'],'delta_authored_oriented_f64_mm3':y['authored_oriented_f64_volume_mm3']-x['authored_oriented_f64_volume_mm3'],'baseline_owner_f32_mm3':x['owner_f32_oriented_volume_mm3'],'candidate_owner_f32_mm3':y['owner_f32_oriented_volume_mm3'],'delta_owner_f32_mm3':y['owner_f32_oriented_volume_mm3']-x['owner_f32_oriented_volume_mm3'],'baseline_normalized_det':x['owner_f32_normalized_det'],'candidate_normalized_det':y['owner_f32_normalized_det']})
  allcells=[]
  for key,x in bm.items():
   y=am[key];allcells.append({'face_row':key[0],'tet_index':key[1],'baseline_invalid':any(v in ('authored_f64_orientation_or_floor','owner_f32_nonpositive','owner_f32_conditioned_lt_1e-5') for v in next((q['baseline_invalid_categories'] for q in bad if (q['face_row'],q['tet_index'])==key),[])),'delta_owner_f32_mm3':y['owner_f32_oriented_volume_mm3']-x['owner_f32_oriented_volume_mm3']})
  result.append({'layer':name,'affected_faces':len(rows),'affected_tet_cells':len(bm),'baseline_invalid_cell_count':len(bad),'baseline_invalid_cells':bad,'worst_delta_among_baseline_invalid_cells':min(bad,key=lambda x:x['delta_owner_f32_mm3']) if bad else None,'worst_delta_over_all_affected_cells':min(allcells,key=lambda x:x['delta_owner_f32_mm3']) if allcells else None,'largest_delta_over_all_affected_cells':max(allcells,key=lambda x:x['delta_owner_f32_mm3']) if allcells else None})
 out={'schema':'numi.human.b65-99.alpha025.layer-volume-review.v1','status':'PRIVATE_DIAGNOSTIC_ONLY','pins':pins,'changed_face_count':len(rows),'layers':result,'limitations':['local cells incident to changed points only','baseline-invalid means an affected tetra failing the existing authored-f64 orientation/floor or owner-f32 nonpositive/conditioning criteria','no native physical admission']}
 OUT.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'output':str(OUT),'layers':[{'layer':x['layer'],'baseline_invalid_cell_count':x['baseline_invalid_cell_count'],'worst_delta_among_baseline_invalid_cells':x['worst_delta_among_baseline_invalid_cells'],'worst_delta_over_all_affected_cells':x['worst_delta_over_all_affected_cells']} for x in result]},indent=2))
if __name__=='__main__':main()
