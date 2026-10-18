from pathlib import Path
import sys,json,hashlib,numpy as np,time
sys.path.insert(0,'/Users/n/numi-human-closed-outer-skin-20261017/src')
from numilab_human.passive_attachment_composition import _read_nhtiss4,_nhtiss_row_arrays
from numilab_human.surface_precision_retriangulation import collapse_closed_component_edge_preserving_allocations
from numilab_human.passive_reference_receipt_composition import compose_reference_receipt
o=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001");i=json.loads((o/'invocation-003.json').read_text());src=Path(i['argv'][i['argv'].index('--soft-tissue-payload')+1]);base=Path(i['argv'][i['argv'].index('--resting-anatomy-receipt')+1]);out=o/'muscle158-precision-002';out.mkdir()
d=_read_nhtiss4(src);row=next(r for r in d['records'] if int(r[6])==158);a=_nhtiss_row_arrays(d,row);f=a['faces'];orig=np.arange(len(f));edges=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0);length=np.linalg.norm(a['positions'][edges[:,0]].astype(float)-a['positions'][edges[:,1]],axis=1);small=edges[length<2.5e-7];ops=[]
for x,y in small:
 if not np.isin([x,y],f).all():continue
 try:nf,proof=collapse_closed_component_edge_preserving_allocations(a['positions'].copy(),f,identity=(51005,158),keep_vertex=int(x),remove_vertex=int(y),max_endpoint_displacement_m=2.5e-7,max_abs_volume_delta_m3=2e-12)
 except ValueError as error:
  try:nf,proof=collapse_closed_component_edge_preserving_allocations(a['positions'].copy(),f,identity=(51005,158),keep_vertex=int(y),remove_vertex=int(x),max_endpoint_displacement_m=2.5e-7,max_abs_volume_delta_m3=2e-12)
  except ValueError as error:print('retained',int(x),int(y),str(error),flush=True);continue
 proof['source_payload_face_origins']=orig[proof['kept_parent_face_origins']].tolist();orig=orig[proof['kept_parent_face_origins']];f=nf;ops.append(proof)
patch=out/'row158.npz';report=out/'row158-report.json';np.savez(patch,vertices6=np.column_stack((a['positions'],a['normals'])),binding_indices=a['binding_indices'],weights=a['weights'],faces=f)
report.write_text(json.dumps(dict(stable_id=158,source_payload_path=str(src),source_sha256=d['sha256'],operations=ops,all_vertex_fields_and_body_bindings_preserved=True,anatomical_admission=False),indent=2))
result=compose_reference_receipt(base,out/'composed-receipt',[(158,patch,report)],extensions={});(out/'result.json').write_text(json.dumps(result,indent=2))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for flag,path in [('--soft-tissue-payload',result['payload_path']),('--resting-anatomy-receipt',str(out/'composed-receipt/resting-anatomy-receipt.json'))]:
 j=i['argv'].index(flag)+1;i['argv'][j]=path;i['asset_sha256'][path]=sha(path)
for p in (out/'composed-receipt').rglob('*'):
 if p.is_file():i['asset_sha256'][str(p)]=sha(p)
(o/'invocation-005.json').write_text(json.dumps(i,indent=2));print(json.dumps(dict(risk_edges=len(small),collapsed=len(ops),retired_faces=len(a['faces'])-len(f))),flush=True)
