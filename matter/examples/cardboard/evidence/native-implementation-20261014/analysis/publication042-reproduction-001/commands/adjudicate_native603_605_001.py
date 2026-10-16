from pathlib import Path
import json,hashlib,datetime,struct
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=q/'analysis/native603-integrated254-history-adjudication-001';d.mkdir(exist_ok=False);new=q/'solver/composed254-seven-phase-regression-603';old=q/'solver/primary-cache-action-diagnostic-regression-599'
r=json.loads((new/'receipt.json').read_text());assert r['exit_code']==0 and r['source_inputs_unchanged'] and not r['timed_out']
names=[p.name for p in (old/'output').iterdir() if p.is_file() and (p.name.startswith(('accepted_step_','material_state_')) or p.name in ['compiled.nmatterpack','mesh.json','manifest.json','observations.csv','initial.obj','result.json'])]
assert len(names)==32
rows=[];inputs=[Path(__file__),new/'receipt.json',old/'receipt.json',q/'analysis/source254-build-002/report.json']
for n in sorted(names):
 p=new/'output'/n;o=old/'output'/n;assert sha(p)==sha(o),n;rows.append({'path':n,'sha256':sha(p),'bytes':p.stat().st_size});inputs.extend([p,o])
report={'schema':'numi.cardboard.native603-integrated254-history-adjudication.v1','verdict':'PASS_COMPLETE_TINY_SEVEN_PHASE_HISTORY_BYTE_PARITY','case':str(new.relative_to(q)),'comparison_case':str(old.relative_to(q)),'production_files':rows,'wall_seconds':r['wall_seconds'],'full_target_coupon_or_convergence_qualified':False,'shipping_only_observable_changes_exercised':False,'scope':'Integrated254 completes the unchanged .01 degree seven-control coupon regression. All32 production files are byte identical; shipping volume-damage/free-force fields require their separate shipping driver execution.','all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(q)):sha(p) for p in inputs},'report_sha256':sha(d/'report.json')},indent=2)+'\n');print('603',sha(d/'report.json'))
d=q/'analysis/native604-605-inline-mortar-root-adjudication-001';d.mkdir(exist_ok=False);base=q/'native/activation-full-residual-control-on-591';off=q/'native/mortar-inline-observer-off-604';on=q/'native/mortar-inline-observer-on-605';inputs=[Path(__file__),q/'analysis/native604-605-mortar-inline-trace-command-001/report.json']
same=[]
for c in [off,on]:
 r=json.loads((c/'receipt.json').read_text());assert r['exit_code']==1 and r['source_inputs_unchanged'] and not r['timed_out'];inputs.append(c/'receipt.json')
for p in sorted((base/'output/dense').glob('transaction-*')):
 a=off/'output/dense'/p.name;b=on/'output/dense'/p.name;assert sha(p)==sha(a)==sha(b),p.name;same.append({'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size});inputs.extend([p,a,b])
assert len(same)==12
jp=on/'diagnostics/capture-coupled-trial-capture/capture.coupled.json';bp=jp.with_suffix('.bin');j=json.loads(jp.read_text());raw=bp.read_bytes();assert len(raw)==j['payload']['byte_count'] and sha(bp)==j['payload']['sha256']
for a in j['arrays']:
 x=raw[a['offset']:a['offset']+a['byte_count']];assert len(x)==a['byte_count'] and hashlib.sha256(x).hexdigest()==a['sha256'],a['name']
matches=[a for a in j['arrays'] if a['name']=='actual_mortar_complement_first_uncertified_query'];assert len(matches)==1;a=matches[0];assert a['byte_count']==384 and a['phase']=='private_contact_rebuild_before_free_body_terms';x=raw[a['offset']:a['offset']+384];u=struct.unpack('<16I',x[:64]);f=struct.unpack('<80f',x[64:]);v=lambda xs:[list(xs[i:i+4]) for i in range(0,len(xs),4)]
decoded={'attempt':list(u[:4]),'candidate_identity':list(u[4:8]),'cells':list(u[8:12]),'feature':list(u[12:16]),'start':v(f[:24]),'finish':v(f[24:48]),'left_bary_map':v(f[48:60]),'right_bary_map':v(f[60:72]),'result_hit_certified_toi_separation':list(f[72:76]),'parameters_thickness_leftDt_rightDt_iteration':list(f[76:80])}
assert decoded['attempt']==[1,0,78,4] and decoded['candidate_identity']==[16,99,78,1] and decoded['feature'][:2]==[2,1] and decoded['result_hit_certified_toi_separation'][:2]==[0.,0.]
oldjp=q/'native/freebody-newton7-full-private-trial-601/diagnostics/capture-coupled-trial-capture/capture.coupled.json';oj=json.loads(oldjp.read_text());om={(a['phase'],a['name']):a for a in oj['arrays']};common=[];different=[]
for a in j['arrays']:
 key=(a['phase'],a['name'])
 if key in om:
  row={'phase':key[0],'name':key[1],'sha256':a['sha256']}
  (common if a['sha256']==om[key]['sha256'] else different).append(row)
report={'schema':'numi.cardboard.native604-605-inline-mortar-root-adjudication.v1','verdict':'PASS_OBSERVER_PRODUCTION_PARITY_FIRST_UNCERTIFIED_RIGHT_VERTEX_TRIANGLE_QUERY_LOCATED','transaction_files_byte_exact_to591':same,'capture_arrays_hash_verified':len(j['arrays']),'capture_bytes':len(raw),'decoded_inline_query':decoded,'inline_record_sha256':a['sha256'] if False else matches[0]['sha256'],'common601_arrays_identical':common,'common601_arrays_different':different,'scope':'The actual production traversal first returns uncertified for right subtriangle vertex2 against left subtriangle at complement cells3/1. No feasibility decision was changed; all production histories and rollback remain identical. A separation near thickness with certified=false is inconclusive, not a no-contact certificate.','contact_or_assembly_qualified':False,'all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');inputs.extend([jp,bp,oldjp,q/'snapshots/source-mortar-first-uncertified-audit-257-006/frozen-source.json',q/'build-source257-mortar-audit-007/source-build.json']);(d/'manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(q)):sha(p) for p in inputs},'report_sha256':sha(d/'report.json')},indent=2)+'\n');print(json.dumps({'604_605_report_sha256':sha(d/'report.json'),'production_files_exact':len(same),'verified_capture_arrays':len(j['arrays']),'601_common_arrays_exact':len(common),'601_common_arrays_different':len(different),'query':decoded['attempt'],'result':decoded['result_hit_certified_toi_separation']}))
