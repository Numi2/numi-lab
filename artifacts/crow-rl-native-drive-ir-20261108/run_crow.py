from pathlib import Path
import csv,hashlib,json,os,subprocess,sys,time
R=Path.cwd();A=R/'artifacts/crow-rl-native-drive-ir-20261108';B=R/'build/crow-native-drive-ir-build-01';T=R/'.numi/retained-runtime/native-drive-ir-20261108'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
if '--prepare' in sys.argv:
 paths=[B/'crow-fixture',A/'crow-fixture.mm',A/'crow-plan.json',A/'run_crow.py']+[p for p in T.rglob('*') if p.is_file()]
 plan=dict(source_and_runtime_sha256={str(p.relative_to(R)):sha(p) for p in paths},commands=[dict(mode=i,args=[str(B/'crow-fixture'),str(B/('crow-mode'+str(i))),'0','1','32',str(i)]) for i in range(3)],timeout_seconds_each=120,baseline_manifest='artifacts/crow-rl-bounded-drive-integration-20261108/payload/manifest.json',baseline_manifest_sha256='5a64d7d4f0443a836723d3c41c8f48a8c32a96b55bc57107f0128c8134c1346f')
 write(A/'crow-execution-plan.json',plan);print(json.dumps({'commands':len(plan['commands']),'native_transition_ceiling':1920}));raise SystemExit()
p=json.loads((A/'crow-execution-plan.json').read_text())
for f,h in p['source_and_runtime_sha256'].items():assert sha(R/f)==h,f
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(T/'metalrobo/MetalRobo.metallib'))
for k in ['METALROBO_BOUNDED_FAILURE_CAPTURE','METALROBO_CCD_FAILURE_CAPTURE','METALROBO_METALLIB']:env.pop(k,None)
commands=[];outcomes=[]
for c in p['commands']:
 name='crow-mode'+str(c['mode']);start=time.monotonic()
 with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:
  try:code=subprocess.run(c['args'],env=env,stdout=out,stderr=err,timeout=p['timeout_seconds_each']).returncode
  except subprocess.TimeoutExpired:code=None
 loads=[s for s in (B/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 rec=dict(c,exit_code=code,seconds=time.monotonic()-start,loader_lines=loads);commands.append(rec);write(A/'crow-execution.json',commands);print(json.dumps(rec),flush=True)
 assert code in (0,1) and loads and all(s.endswith(str(T/'libmetalrobo.dylib')) for s in loads),rec
 D=B/name;rows=list(csv.DictReader((D/'trace.tsv').open(),delimiter='\t')) if (D/'trace.tsv').exists() else []
 terminal=json.loads((D/'terminal.json').read_text()) if (D/'terminal.json').exists() else None
 failure=json.loads((D/'failure.json').read_text()) if (D/'failure.json').exists() else None
 summary=json.loads((D/'summary.json').read_text()) if (D/'summary.json').exists() else None
 outcome=dict(mode=c['mode'],exit_code=code,trace_rows=len(rows),last_recorded_tick=max((int(r['tick']) for r in rows),default=None),maximum_recorded_penetration_m=max([max(-float(r['upper_gap_m']),-float(r['lower_gap_m']),0) for r in rows],default=None),terminal=terminal,native_failure=failure,summary=summary,diagnostic=(B/(name+'.stderr')).read_text().splitlines()[-4:])
 outcomes.append(outcome);write(A/'crow-outcomes.json',outcomes)
 for physical in ['initial-q.f32','initial-v.f32','initial-scene.bin','initial-geometry.json']:
  assert (D/physical).read_bytes()==(B/'crow-mode0'/physical).read_bytes(),(name,physical)
 if c['mode']==0:
  entries=tree()
  with Objects() as o:
   raw=o.read(entries[p['baseline_manifest']][1]);assert hashlib.sha256(raw).hexdigest()==p['baseline_manifest_sha256'];m=json.loads(raw)
   f=next(f for f in m['files'] if f['path']=='build/crow-bounded-drive-integration-build-01/terminal-crow32/trace.tsv');expected=b''.join(content(f,entries,o));actual=(D/'trace.tsv').read_bytes()
   baseline=dict(old_trace_rows=len(expected.splitlines())-1,old_prefix_byte_exact=actual.startswith(expected),new_trace_rows=len(rows),new_terminal_frame_recorded=True);write(A/'crow-baseline-regression.json',baseline)
   assert baseline['old_prefix_byte_exact'],baseline
  assert code==1 and terminal and terminal['begin_tick']==91 and terminal['failed']==0, outcome
candidate=outcomes[2];passed=candidate['exit_code']==0 and candidate['trace_rows']==640 and candidate['maximum_recorded_penetration_m']<=.0008 and not candidate['terminal'] and not candidate['native_failure']
write(A/'crow-analysis.json',dict(verdict='supported_engineering_gate' if passed else 'contradicted',candidate=candidate,initial_physical_files_equal=True,baseline_prefix_exact=True,limits=json.loads((A/'crow-plan.json').read_text())['interpretation'],policy_or_profile_promoted=False))
print(json.dumps({'candidate_verdict':'supported_engineering_gate' if passed else 'contradicted','outcomes':outcomes}),flush=True)
