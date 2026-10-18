from pathlib import Path
import sys,json,hashlib,numpy as np
sys.path.insert(0,'/Users/n/numi-human-closed-outer-skin-20261017/src')
from numilab_human import composed_bone_body_frame as cb
from numilab_human.model import numi_human_tendon_attachment_envelope_payload
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.surface_precision_retriangulation import collapse_closed_component_edge_preserving_allocations
from numilab_human.resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
from numilab_human.resting_common_field_passive_rebind import rebind_passive_rows
o=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001");out=o/'paired-bone49-precision';out.mkdir();i=json.loads((o/'invocation-006.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
bp=Path(i['argv'][3]);br=bp.read_bytes();bm=json.loads(bp.with_suffix('.manifest.json').read_text());oldhash=sha(br)
base=Path(i['argv'][i['argv'].index('--resting-anatomy-receipt')+1]);rec=json.loads(base.read_text());nr=Path(rec['payload']['path']).read_bytes();_,_,rows=_parse_payload(nr);row=dict(rows[313]);coords=row['vertices6'][:,:3]
h,records,vo,io=cb._decode_bones(br);r=next(r for r in records if r[5]==49);v=np.frombuffer(br,'<f4',h[3]*6,vo).reshape(-1,6)[r[1]:r[1]+r[2],:3].copy();f=np.frombuffer(br,'<u4',r[4],io+r[3]*4).reshape(-1,3)-r[1]
edges=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0);length=np.linalg.norm(v[edges[:,0]].astype(float)-v[edges[:,1]],axis=1);risk=edges[length<5e-7];ops=[]
for x,y in risk:
 success=False
 for keep,remove in [(int(x),int(y)),(int(y),int(x))]:
  params=dict(stable_id=49,keep_vertex=keep,remove_vertex=remove,flip_edges=[],max_endpoint_displacement_m=5e-7,minimum_altitude_m=1e-12,max_abs_volume_delta_m3=2e-12)
  try:
   nb,proof=cb.repair_bone_source_precision_edges(br,**params)
   matched=[np.flatnonzero(np.all(coords==v[k],axis=1)).astype(int).tolist() for k in [keep,remove]]
   newfaces=row['faces'];npf=None
   if all(len(z)==1 for z in matched):
    newfaces,npf=collapse_closed_component_edge_preserving_allocations(coords.copy(),row['faces'],identity=(51010,313),keep_vertex=matched[0][0],remove_vertex=matched[1][0],max_endpoint_displacement_m=5e-7,max_abs_volume_delta_m3=2e-12)
  except (ValueError,Exception) as error:
   print('direction rejected',keep,remove,str(error)[:150],flush=True);continue
  bd=out/('bone-step-'+str(len(ops)+1));bd.mkdir();candidate=bd/bp.name;manifest=dict(bm);manifest['payload']=dict(bm['payload']);manifest['payload'].update(sha256=sha(nb),bytes=len(nb),index_count=cb._BONE_HEADER.unpack_from(nb)[4])
  manifest['source_precision_retriangulation']=dict(input_artifact=str(bp.parent),input_manifest_sha256=sha(bp.with_suffix('.manifest.json').read_bytes()),input_payload_sha256=sha(br),parameters=params,proof=proof)
  candidate.write_bytes(nb);candidate.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
  cb.validate_composed_rib_reframe(bd,manifest,nb)
  row['faces']=newfaces;ops.append(dict(parameters=params,bone=proof,paired_nha=npf));br=nb;bp=candidate;bm=manifest;success=True;print('collapsed',keep,remove,flush=True);break
 if not success:print('retained risk edge',int(x),int(y),flush=True)
newnha,encoding=encode_payload_with_row_replacements(nr,{313:row});nha=out/'resting-thorax.nhanatomy';nha.write_bytes(newnha)
rec,rebind=rebind_passive_rows(rec,nr,newnha,changed_ids=(313,),output_path=str(nha))
maps={sid:rec['functional_bindings'][f'intercostal_313_rib{sid}_motion_attachment']['surface']['interface_local_vertex_ids'] for sid in [48,49,50]}
view=dict(payload=rec['payload'],functional_bindings=rec['functional_bindings'],provenance=dict(bones_payload_sha256=oldhash));cb.bind_intercostal_runtime_attachment_metadata(view,row,maps,nha_stable_id=313,nha_payload_sha256=sha(newnha),bones_payload_sha256=sha(br));cb.validate_intercostal_runtime_attachment_metadata(view,row,maps,nha_stable_id=313,nha_payload_sha256=sha(newnha),bones_payload_sha256=sha(br))
rec['provenance']['bones_payload']=str(bp);rec['provenance']['bones_payload_sha256']=sha(br)
cb._preserve_receipt_common_field_sidecars(rec,base.parent,out)
rec['provenance']['rope_swing_precision']=dict(operations=ops,source_bones_sha256=oldhash,source_nha_sha256=sha(nr),nha_encoding=encoding,owner_rebind=rebind,physical_state_or_force_owner_changed=False)
receipt=out/'resting-anatomy-receipt.json';receipt.write_text(json.dumps(rec,separators=(',',':')))
td=out/'compiled-tendon';td.mkdir();tm=numi_human_tendon_attachment_envelope_payload(o.parent/'current-manifest-recovery-004/recovered-artifact',bp.parent,td,.012,.012,4.,True,.025);tp=td/tm['payload']['file']
old=Path(i['argv'][i['argv'].index('--tendon-payload')+1]).read_bytes();new=tp.read_bytes();assert old[:112]==new[:112] and old[144:]==new[144:]
i['argv'][3]=str(bp)
for flag,p in [('--torso-anatomy-payload',nha),('--resting-anatomy-receipt',receipt),('--tendon-payload',tp)]:i['argv'][i['argv'].index(flag)+1]=str(p)
for p in out.rglob('*'):
 if p.is_file():i['asset_sha256'][str(p)]=sha(p.read_bytes())
(o/'invocation-007.json').write_text(json.dumps(i,indent=2));(out/'repair-report.json').write_text(json.dumps(dict(risk_edges=len(risk),collapsed=len(ops),operations=ops,physical_mass_and_tendon_endpoint_bytes_unchanged=True),indent=2));print('ready',len(ops),flush=True)
