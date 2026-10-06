"""Compare only the two completed preregistered tight barrier arms.

The three-arm completion prediction is separately contradicted. The failed
factor-1 arm must not contribute a truncated full-cycle estimate.
"""
from pathlib import Path
import csv,hashlib,json,math
root=Path(__file__).resolve().parent
prior=json.loads((root/'tight-barrier-comparison.json').read_text())
names=['barrier50-scale2-tight-001','barrier50-scale4-tight-001']
curves=[];manifests=[];bindings=[]
for n in names:
 assert prior['runs'][n]['verdict']=='instrument_checks_passed'
 rec=json.loads((root/'receipts'/n/'receipt.json').read_text());assert rec['inputs_unchanged'];bindings.append(rec['bindings'])
 m=json.loads((root/n/'manifest.json').read_text());m.pop('compiled_world_fingerprint');m['tooling'].pop('prescribed_tool_barrier_stiffness_scale');manifests.append(m)
 obs=[r for r in csv.DictReader((root/n/'observations.csv').open()) if r['environment']=='0'];tools={int(r['step']):r for r in csv.DictReader((root/n/'tool-observations.csv').open()) if r['environment']=='0'}
 assert len(obs)==96 and all(r['step_accepted']=='1' and r['status_code']=='0' for r in obs)
 curves.append([(int(r['step']),r['phase'],float(r['time_s']),float(tools[int(r['step'])]['punch_force_z_N'])) for r in obs])
assert manifests[0]==manifests[1] and bindings[0]==bindings[1]
assert [x[:3] for x in curves[0]]==[x[:3] for x in curves[1]]
rms=lambda v: math.sqrt(sum(x*x for x in v)/len(v))
report={'schema':'numi.cardboard.completed-tight-barrier-pair.v1','physical_validation':False,'runs':names,'pair_status':'complete_same_native_inputs','three_arm_prediction':'contradicted_by_factor1_first_withdrawal_rejection','excluded_failed_arm':'barrier50-scale1-tight-001','curve_metrics':{},'input_report_sha256':hashlib.sha256((root/'tight-barrier-comparison.json').read_bytes()).hexdigest(),'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
p=[prior['runs'][n]['peak_accepted_punch_force_N'] for n in names];report['peak_relative_spread']=(max(p)-min(p))/max(p)
for phase in ['loading','hold','unloading','relaxation']:
 a=[x[3] for x in curves[0] if x[1]==phase];b=[x[3] for x in curves[1] if x[1]==phase]
 d=rms([x-y for x,y in zip(a,b)]);norm=max(rms(a),rms(b));report['curve_metrics'][phase]={'rms_difference_N':d,'relative_rms_difference':d/norm if norm else None,'samples':len(a)}
f=root/'completed-tight-barrier-pair-comparison.json';assert not f.exists();f.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
