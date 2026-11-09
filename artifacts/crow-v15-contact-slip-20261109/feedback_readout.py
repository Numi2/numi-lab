from pathlib import Path
import difflib,gzip,hashlib,json,os,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-contact-slip-20261109';B=R/'build/crow-v15-contact-slip-20261109';C=B/'readout';C.mkdir(exist_ok=True);base=(B/'c_api.cpp').read_text()
needle='''                            array("policy_action",published.policyActions,lane*handle->taskProgram.layout().actionCount,handle->taskProgram.layout().actionCount);
''';assert base.count(needle)==1
injection='''                            std::fprintf(stderr,",\\"diagnostic_foot_velocity\\":[%.9g,%.9g]",published.transitions[lane].outcomeChannels1.z,published.transitions[lane].outcomeChannels1.w);
''';s=base.replace(needle,needle+injection);src=C/'c_api.cpp';src.write_text(s);original=(R/'src/c_api.cpp').read_text();(A/'feedback-readout-observer.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/src/c_api.cpp',tofile='b/feedback-readout-observer.cpp')))
lines=subprocess.check_output(['ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);obj=C/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=C/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);subprocess.run(args,cwd=R/'build',check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cmd=json.loads((A/'sensor-diagnostic-bindings.json').read_text())['argv']
for k,v in [('--state-trace',str(C/'trace.tsv')),('--crow-replay-pack',str(C/'replay.json'))]:cmd[cmd.index(k)+1]=v
(A/'feedback-readout-bindings.json').write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'feedback-readout-plan.md'),'observer_sha256':h(lib),'patch_sha256':h(A/'feedback-readout-observer.patch')},indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(C),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/'feedback-readout.stdout').write_text(p.stdout)
with (A/'feedback-readout.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError(p.stderr[-3000:])
frames=json.loads((C/'replay.json').read_bytes())['payload']['frames'];prior=json.load(gzip.open(A/'sensor-diagnostic.replay.json.gz','rt'))['payload']['frames'];same=len(frames)==len(prior) and all(all(a[k]==b[k] for k in ('q','v','body_states')) for a,b in zip(frames,prior));assert same,'readout changed physical states'
vals=[]
for l in p.stderr.splitlines():
 if not l.startswith('CROW_SUPPORT_SAMPLE '):continue
 x=json.loads(l.split(' ',1)[1]);
 if x['lane']==1 and 101<=x['step']<=250:vals.append(x['diagnostic_foot_velocity'])
report={'state_parity_q_v_body_states':same,'samples':len(vals),'minimum':[min(v[i] for v in vals) for i in (0,1)],'maximum':[max(v[i] for v in vals) for i in (0,1)],'nonzero_samples':[sum(abs(v[i])>1e-8 for v in vals) for i in (0,1)],'scope':'on-device computed proxy at accepted endpoints; not exact continuous contact velocity'};(A/'feedback-readout-summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
