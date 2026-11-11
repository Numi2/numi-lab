from pathlib import Path
import gzip,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-two-segment-frozen-flight-failure-observer-20261111';B=R/'build/crow-v15-two-segment-frozen-flight-failure-observer-20261111';base=json.loads((R/'artifacts/crow-v15-two-segment-frozen-flight-batch-isolation-20261111/80-command.json').read_text());runs={}
for n in (2,3):
 cmd=list(base);cmd[cmd.index('--steps')+1]=str(n);cmd[cmd.index('--state-trace')+1]=str(B/(str(n)+'.tsv'));p=subprocess.run(cmd,cwd=R,env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','CROW_EARLY_FLIGHT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=180);(A/(str(n)+'.stdout')).write_text(p.stdout)
 with (A/(str(n)+'.stderr.gz')).open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 samples=[json.loads(l.split(' ',1)[1]) for l in p.stderr.splitlines() if l.startswith('CROW_FLIGHT_SAMPLE ')];runs[n]={'exit':p.returncode,'samples':{(x['step'],x['lane']):x for x in samples},'error':p.stderr.splitlines()[-1:]};print('steps',n,'exit',p.returncode,'samples',len(samples),flush=True)
a,b=runs[2]['samples'],runs[3]['samples'];diff=[list(k) for k in a if b.get(k)!=a[k]];expected={(t,l) for t in (1,2,3) for l in range(80)};lines=(R/'build/crow-v15-two-segment-frozen-flight-batch-isolation-20261111/80.tsv').read_text().splitlines()[1:];tsv={int(v[0]):list(map(float,v[1:])) for line in lines if (v:=line.split())};delta=[]
for t in (1,2):
 sample=a.get(((1 if t==1 else 3),0));r=tsv.get(t)
 if sample is None or r is None:delta.append({'step':t,'missing':True});continue
 vals=sample['q'];delta.append({'accepted_step':t,'observer_event':1 if t==1 else 3,'q_max_abs_difference':max(abs(x-y) for x,y in zip(vals,r[:len(vals)])),'q_values':len(vals)})
report={'two_step_exit':runs[2]['exit'],'three_step_exit':runs[3]['exit'],'two_step_missing_samples':len(expected-set(a)),'three_step_missing_samples':len(expected-set(b)),'accepted_sample_differences':diff,'env_zero_reference':delta,'failure':runs[3]['error'],'scope':'host observer events 1 and 2 duplicate first accepted state; event 3 is second accepted state; TSV contains q and scene only, not velocity; failure cause unqualified'};(A/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
