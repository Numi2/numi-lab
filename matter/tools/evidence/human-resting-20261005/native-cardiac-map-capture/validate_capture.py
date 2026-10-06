from pathlib import Path
import json,csv,hashlib
E=Path('/Users/n/numi-human-resting-evidence-20261005')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
O=E/'cardiac-map-capture-validation-168';O.mkdir(exist_ok=False)
A=E/'cardiac-map-capture-native-165';B=E/'cardiac-annular-map-native-161'
reports=[]
for name,expected_map,expected_params,steps in [
('cardiac-map-capture-native-165','17dc35a3241e09589b4178560c50cf535bcd70dfd19602e801c922b1edb45f10','3c8737fd09de14e1dce03150331498d1bfbfa56b0ea58607a3db5c71221bf027',(0,63)),
('cardiac-map-capture-native-167','8700de474c21afd695beed6b1a44c12b7292e7df9563924c699803e0c8f64a45','de3798aa0e3e954d41cbd014bd0747631916aea84fb5710d31b7c46a5a630b8c',(0,))]:
 root=E/name;inv=json.loads((root/'invocation.json').read_text());r=json.loads((root/'ventricular-wall-map-identity.json').read_text())
 m=root/r['map_file'];p=root/r['parameters_file'];identitysha=sha(root/'ventricular-wall-map-identity.json')
 assert inv['return_code']==0 and inv['assets_before']==inv['assets_after'] and inv['runtime_before']==inv['runtime_after']
 assert sha(m)==r['map_sha256']==expected_map and sha(p)==r['parameters_sha256']==expected_params
 assert m.stat().st_size==r['map_bytes']==48*r['map_vertex_count'] and p.stat().st_size==r['parameters_bytes']==128
 assert hashlib.sha256(m.read_bytes()+p.read_bytes()).hexdigest()==r['bundle_sha256']
 assert r['source_anatomy_payload_sha256']==inv['assets_before'][inv['argv'][inv['argv'].index('--torso-anatomy-payload')+1]]
 for step in steps:
  x=json.loads((root/f'accepted-geometry/step-{step}.receipt.json').read_text())
  assert x['ventricular_wall_map_sha256']==r['map_sha256'] and x['ventricular_wall_parameters_sha256']==r['parameters_sha256']
  assert x['ventricular_wall_coefficient_bundle_sha256']==r['bundle_sha256'] and x['ventricular_wall_map_identity_receipt_sha256']==identitysha
 reports.append({'run':name,'invocation_sha256':sha(root/'invocation.json'),'identity_sha256':identitysha,'map_sha256':sha(m),'parameters_sha256':sha(p),'source_anatomy_payload_sha256':r['source_anatomy_payload_sha256'],'captured_steps':list(steps),'buffer_bytes_and_receipt_bindings_pass':True})
keys=['accepted_body_state_sha256','accepted_respiration_state_sha256','accepted_root_fingerprint_hex','captured_vertex_buffer_sha256','accepted_registered_body_poses','accepted_step','accepted_time_s','index_count','vertex_count','pack_content_hash','pack_file_sha256']
compare=[]
for step in (0,63):
 ar=json.loads((A/f'accepted-geometry/step-{step}.receipt.json').read_text());br=json.loads((B/f'accepted-geometry/step-{step}.receipt.json').read_text())
 checks={k:ar[k]==br[k] for k in keys}
 compare.append({'step':step,'checks':checks})
physa=list(csv.DictReader((A/'resting-coupled.csv').open()));physb=list(csv.DictReader((B/'resting-coupled.csv').open()))
sa=list(csv.DictReader((A/'resting-surface-audit.csv').open()));sb=list(csv.DictReader((B/'resting-surface-audit.csv').open()))
out={'scope':'Load-time input capture regression, using actual native Metal execution. Refined 64-step prefix compared with published 031 replay; unrefined one-step run checks only immutable buffer identity and remains anatomically unqualified. No new five-minute or physiology qualification.','driver_sha256':sha(__file__),'captures':reports,'refined_native_prefix_comparison':compare,'physiology_rows':len(physa),'physiology_prefix_equal':physa==physb[:len(physa)],'surface_rows':len(sa),'surface_prefix_equal':sa==sb[:len(sa)],'comparison_invocation_sha256':sha(B/'invocation.json')}
out['pass']=all(all(x['checks'].values()) for x in compare) and out['physiology_prefix_equal'] and out['surface_prefix_equal']
(O/'report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
