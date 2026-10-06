from pathlib import Path
import json, hashlib, csv
r=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={'schema':'numi.cardboard.newton-trace-comparison.v1','physical_validation':False,'scope':'Descriptive failed-iterate diagnostic. No accepted mesh endpoint, true b-Ax check or causal attribution.','runs':{}}
receipts=[]
for budget,old in [(64,'persistent-nx16-001'),(256,'nx16-k256-first-001')]:
 name=f'nx16-k{budget}-newton-trace-002';d=json.loads((r/(name+'-newton-analysis.json')).read_text());receipt=json.loads((r/'receipts'/name/'receipt.json').read_text());receipts.append(receipt)
 assert d['integrity']=='ok' and len(d['iterations'])==28 and receipt['inputs_unchanged']
 rows=list(csv.DictReader((r/name/'observations.csv').open()));row=next(x for x in rows if x['environment']=='0')
 assert int(row['status_code'])==d['terminal']['status_code_at_certificate']==10
 same={f:sha(r/name/f)==sha(r/old/f) for f in ['observations.csv','initial.obj','mesh.json']}
 if not all(same.values()):raise RuntimeError(('diagnostic differs from prior trajectory',name,same))
 growing=[x['iteration'] for x in d['iterations'] if x['residual_grew'] and x['near_full_alpha']]
 report['runs'][name]={'trace_integrity':'ok','accepted_steps':0,'raw_terminal_status':int(row['status_code']),'terminal_residual':float(row['certificate_residual']),'iterations':len(d['iterations']),'near_full_steps_with_more_than_10pct_growth':growing,'screen_observations_with_accepted_inner_cycle':d['screen']['observations'],'prediction_supported_in_this_arm':bool(d['screen']['observations']),'final_cycle_accepted_count':sum(x['final_cycle_accepted'] is True for x in d['iterations']),'prior_untraced_name':old,'byte_identical_to_prior':same,'inputs_unchanged':receipt['inputs_unchanged'],'bindings':{f:sha(r/name/f) for f in ['observations.csv','initial.obj','mesh.json']},'analysis_sha256':sha(r/(name+'-newton-analysis.json'))}
assert receipts[0]['bindings']==receipts[1]['bindings']
report['same_native_source_binary_material_bindings']=True
report['prediction_supported_in_at_least_one_arm']=any(x['prediction_supported_in_this_arm'] for x in report['runs'].values())
report['next_prediction']='A native direct b-Ax diagnostic will determine whether estimated inner residuals predict actual linearized residuals at the first nonlinear-growth iteration. Do not alter the acceptance gate or attribute the failure to globalization alone before that check.'
with (r/'newton-traces-comparison.json').open('x') as out:out.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
