import json,pathlib
p=pathlib.Path(__file__).parent
candidate=json.loads((p/'on.json').read_text())['runs'];off=json.loads((p/'off.json').read_text())['runs'];geometry=json.loads((p/'geometry.json').read_text())
clean=lambda x:{k:v for k,v in x.items() if k!='elapsed_with_initialization'}
assert len(candidate)==len(off)==12
trace_equal=all(clean(x)==clean(y) for x,y in zip(candidate,off));repeat=all(x['repeat_identical'] and x['rejected_state_restored'] for x in candidate)
events={};key=None
for line in (p/'on.stderr').read_text().splitlines():
 if line.startswith('ccd_case='):
  c=json.loads(line.split('=',1)[1]);key=(c['scale'],c['fixture']);events[key]=[]
 elif line.startswith('ccd_event_trace='):events[key].append(json.loads(line.split('=',1)[1]))
rows=[]
for r in candidate:
 e=events[(r['scale'],r['fixture'])];assert len(e)==8 and e[:4]==e[4:];residuals=[]
 for frame in e[:4]:
  if frame['status']!=0 or not frame['contacts']:continue
  rr=[]
  for c in frame['contacts']:
   assert (c['body_a'],c['body_b'])==(0,1)
   j=c['jacobians'];v=frame['v'];n=c['normal'];jv=sum(n[a]*sum((j[18+a*6+k]-j[a*6+k])*v[k] for k in range(6)) for a in range(3));eq=jv-c['evaluated_target']+c['regularization']*c['J'];rr.append(abs(eq) if c['J']>1e-6 else max(-eq,0.))
  calc=max(rr);assert abs(calc-frame['residuals'][1])<2e-5
  residuals.append({'pass':frame['pass'],'reported_m_s':frame['residuals'][1],'reconstructed_m_s':calc})
 rows.append({'scale':r['scale'],'fixture':r['fixture'],'accepted':r['accepted'],'status':r['status'],'impulse_Ns':r['normal_impulse_Ns'],'unresponded_swept_hit':r['accepted_unresponded_swept_hit'],'residuals':residuals})
misses=all(r['accepted'] and r['impulse_Ns']==0 for r in rows if r['fixture'] in (13,15));impacts=all(r['accepted'] and r['impulse_Ns']>0 for r in rows if r['fixture'] in (12,14));normal=all(e['reconstructed_m_s']<=.002 for r in rows for e in r['residuals']);geom=geometry['all_accepted_segments_qualified'] and len(geometry['accepted_runs'])==sum(r['accepted'] for r in candidate)
out={'schema':'numi.ccd.monotone-hybrid.v1','trace_on_off_results_equal':trace_equal,'repeat_and_exposed_rollback_pass':repeat,'accepted_segments_geometry_pass':geom,'near_miss_gate_pass':misses,'thin_impact_completion_gate_pass':impacts,'normal_residual_gate_pass':normal,'mixed_rod_rigid_response_qualified':False,'candidate_promotable':False,'rows':rows,'limitations':['This is an unpromoted certificate and four-contact research solver.','Only a single floating box over a fixed plane is geometrically observed.','Complete owner restoration, force accuracy and general mixed contact are not qualified.']}
(p/'full-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
