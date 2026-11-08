from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-flight-source-diagnostics-20261108';B=R/'build/crow-flight-source-diagnostics-01'
base=json.load(open(A/'runs.json'))[1]
cmd=base['argv'][:]
cmd[cmd.index('--metallib')+1]=str(B/'filtered-action.metallib')
cmd[cmd.index('--steps')+1]='800'
cmd[cmd.index('--state-trace')+1]=str(B/'contact-capture-lane-2.tsv')
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(B/'diagnostic-lib')}
t=time.monotonic();p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=300)
(A/'contact-capture.stdout').write_text(p.stdout);(A/'contact-capture.stderr').write_text(p.stderr)
print('returncode',p.returncode,'elapsed',time.monotonic()-t,flush=True)
if p.returncode: print(p.stderr[-3000:]);raise SystemExit(p.returncode)
raw=[json.loads(line.split(' ',1)[1]) for line in p.stderr.splitlines() if line.startswith('CROW_CONTACT_TRACE ')]
trace1=(B/'held-out-a-lane-2.tsv').read_text().splitlines()
trace2=(B/'contact-capture-lane-2.tsv').read_text().splitlines()
match=trace1[1:801]==trace2[1:801]
report={'argv':cmd,'diagnostic_lib_sha256':hashlib.sha256((B/'diagnostic-lib/libmetalrobo.dylib').read_bytes()).hexdigest(),'filtered_shader_sha256':hashlib.sha256((B/'filtered-action.metallib').read_bytes()).hexdigest(),'physical_trace_matches_uninstrumented':match,'fatal_contact_records':raw,'warning':'Contact candidates are from the final published physics substep. Termination aggregates contacts across the control step, so a missing pair does not prove no earlier-substep contact.'}
(A/'contact-capture.json').write_text(json.dumps(report,indent=2)+'\n')
print('same_trace',match,'contact_records',len(raw))
for item in raw[:8]:print(item['step'],item['lane'],[(x['a_name'],x['b_name'],x['normal_impulse']) for x in item['contacts']])
