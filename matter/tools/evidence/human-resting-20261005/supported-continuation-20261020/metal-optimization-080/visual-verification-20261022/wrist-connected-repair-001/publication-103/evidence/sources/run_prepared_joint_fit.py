from pathlib import Path
import importlib.util,json,subprocess,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';A=D/'automatic-connected-continuation-169'
spec=importlib.util.spec_from_file_location('adapt',A/'run_existing_continuation_002.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=Path(sys.argv[1])
m.run([m.PY,out/'run_connected_barycentric_fit.py'],out/'fit.log')
fp=out/'attempt-001/fit-report.json';fit=m.read(fp)
if not fit['solver']['accepted_steps']:
 m.write(out/'result.json',dict(status='NO_ACCEPTED_CONSTRUCTION_STEP',accepted_physical_steps=0,native_admission=False))
 print(json.dumps(dict(status='NO_ACCEPTED_CONSTRUCTION_STEP',path=str(out))))
else:
 candidate=out/'attempt-001/candidate.npz';point=m.last_point(fit,candidate)
 accepted=[t for s in fit['solver']['stages'] for t in s['trials'] if t['accepted']]
 merit=accepted[-1]['progress_merit'];surface=Path(merit['surface_report'])
 after=m.owner().summarize_surface_crossing_report(m.read(surface))
 result=dict(candidate=str(candidate),candidate_sha256=m.sha(candidate),point_summary=str(point),surface_report=str(surface),parent_fit=str(fp),after_surface_pairs=after['total_crossing_pairs'],inside_pairs_after=m.read(point)['inside_node_bank_pairs'],native_admission=False,accepted_physical_steps=0)
 m.write(out/'result.json',result);print(json.dumps(result),flush=True)
