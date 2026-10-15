from pathlib import Path
import datetime,hashlib,json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');P='/Users/home/.pyenv/versions/3.10.12/bin/python3'
S=B/'snapshots/source-explicit-material-budget-109';R=Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-109');O=B/'analysis/source109-material-budget-cli-contract-001';O.mkdir(exist_ok=False)
base=json.loads((B/'native-374-command.json').read_text());native=base[base.index('--')+1:]
native=[v.replace('source-true-residual-certification-105','source-explicit-material-budget-109').replace('provenance-105','provenance-109') for v in native]
rows=[]
for label,requested in [('default',None),('explicit16',16),('explicit32',32)]:
 case=O/label;argv=native.copy();i=argv.index('--material-iterations');del argv[i:i+2]
 argv[argv.index('--output')+1]=str(case/'output');argv+=['--compile-only']
 if requested is not None:argv+=['--material-iterations',str(requested)]
 command=[P,str(S/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(S),'--runtime-dir',str(R),'--build-provenance',str(R/'source-build.json'),'--case',str(case),'--timeout','120','--question','Does the CLI compile the effective accurate-bilinear15 budget it reports?','--prediction','CPU compile-only control: no native runtime initialized or physical steps. Default16 retained, explicit16 and explicit32 match compiled descriptor and reported effective budgets.','--',*argv]
 (O/(label+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
 with (O/(label+'-runner.log')).open('x') as out:rc=subprocess.call(command,stdout=out,stderr=subprocess.STDOUT)
 receipt=json.loads((case/'receipt.json').read_text());assert rc==0 and receipt['exit_code']==0 and not receipt['timed_out'] and receipt['termination_errors']==[] and receipt['source_inputs_unchanged']
 m=json.loads((case/'output/manifest.json').read_text());expected=16 if requested is None else requested
 solver=m['solver'];assert solver['local_material_newton_iterations']==expected
 assert solver['local_material_newton_iterations_explicit']==(requested is not None)
 assert solver['compiled_local_material_newton_iterations']==[expected]*3
 result=json.loads((case/'output/result.json').read_text());assert result['status']=='compile_only',result
 rows.append(dict(case=label,requested=requested,effective=solver['compiled_local_material_newton_iterations'],source_inputs_unchanged=True,physical_steps=0,native_runtime_initialized=False))
report=dict(schema='numi.cardboard.compile-only-budget-cli-contract.v1',cases=rows,all_passed=True,source_build_sha256=hashlib.sha256((R/'source-build.json').read_bytes()).hexdigest(),finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'audit.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(O.rglob('*')) if p.is_file() and 'inputs' not in p.relative_to(O).parts}),indent=2)+'\n');print(json.dumps(report,indent=2))
