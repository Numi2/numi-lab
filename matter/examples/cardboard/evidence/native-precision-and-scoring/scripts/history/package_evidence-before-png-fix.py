from pathlib import Path
import csv,gzip,hashlib,json
src=Path(__file__).parent
dst=Path('/Users/home/numi-cardboard-20261006/matter/examples/cardboard/evidence/native-precision-and-scoring')
dst.mkdir(parents=True,exist_ok=True)
def copy(p,q,compress=False):
 q.parent.mkdir(parents=True,exist_ok=True)
 data=p.read_bytes()
 if compress:data=gzip.compress(data,mtime=0)
 if q.exists() and q.read_bytes()!=data:raise RuntimeError('changed retained evidence: '+str(q))
 q.write_bytes(data)
count=0
for receipt in sorted((src/'receipts').iterdir()):
 if not (receipt/'receipt.json').exists():continue
 name=receipt.name;run=src/name;out=dst/'runs'/name;count+=1
 for name in ['manifest.json','result.json','observations.csv','tool-observations.csv','failure.txt','box-layout.json','contact-trace-analysis.json']:
  if (run/name).exists():copy(run/name,out/name)
 for p in [run/'compiled.nmatterpack',run/'mesh.json',run/'initial.obj',*sorted(run.glob('material_state_*.json'))]:
  if p.exists():copy(p,out/(p.name+'.gz'),True)
 rows=list(csv.DictReader((run/'observations.csv').open())) if (run/'observations.csv').exists() else []
 arm=[x for x in rows if x.get('environment')=='0'];steps=set()
 for i,row in enumerate(arm):
  if i+1==len(arm) or arm[i+1].get('phase')!=row.get('phase') or row.get('step_accepted')=='0':steps.add(int(row['step'])+1)
 for step in steps:
  for p in sorted(run.glob(f'accepted_step_{step:06d}_*.obj')):copy(p,out/(p.name+'.gz'),True)
 for filename in ['invocation.json','receipt.json','stdout.log','stderr.log']:
  p=receipt/filename
  if p.exists():
   compress=p.stat().st_size>65536
   copy(p,out/'receipt'/(filename+('.gz' if compress else '')),compress)
 if (receipt/'source.patch').exists():copy(receipt/'source.patch',out/'receipt/source.patch.gz',True)
 snapshots={}
 for p in receipt.iterdir():
  if p.suffix not in {'.mm','.hpp','.nmatter','.metalinc','.cpp','.h'}:continue
  sha=hashlib.sha256(p.read_bytes()).hexdigest();target=dst/'source-snapshots'/(sha+'-'+p.name+'.gz')
  copy(p,target,True);snapshots[p.name]={'sha256':sha,'archive':str(target.relative_to(dst))}
 (out/'receipt/source-archives.json').write_text(json.dumps(snapshots,indent=2)+'\n')
for p in sorted(src.glob('*plan.json')):copy(p,dst/p.name)
for pattern in ['*analysis.json','*comparison.json','*bindings.json','*.png','*.log']:
 for p in sorted(src.glob(pattern)):
  if p.name.endswith('-driver.log') and not (src/'receipts'/p.name.removesuffix('-driver.log')/'receipt.json').exists():continue
  compress=p.stat().st_size>65536
  copy(p,dst/'instrument-checks'/(p.name+('.gz' if compress else '')),compress)
print('Packaged',count,'completed or failed native invocations; raw source',src)

# Independent checks retain source, results and hashes; native binaries stay in raw storage.
for bundle in ['delamination-native-coupon-002','preconditioner-oracle','preconditioner-oracle-002','final-native-regressions-001','final-native-regressions-002','final-native-regressions-003','fem-reference-persistent-u-001','box-feature-check-001','globalization-telemetry-plan-001']:
 base=src/bundle
 if not base.exists():continue
 for p in sorted(base.rglob('*')):
  if not p.is_file() or p.name in {'SHA256SUMS'}:continue
  if p.suffix not in {'.json','.csv','.md','.txt','.log','.cpp','.mm','.hpp','.h','.metal','.nmatter','.svg'}:continue
  compress=p.stat().st_size>65536
  rel=p.relative_to(base)
  copy(p,dst/'independent-checks'/bundle/rel.parent/(rel.name+('.gz' if compress else '')),compress)
for name in ['run_native.py','run_native_resolution.py','run_native_mesh.py','run_native_trace.py','analyze_contact_trace.py','plot_blank.py','plot_tooling.py','analyze_barrier_study.py','plot_barrier_study.py','plot_matched_time.py','analyze_tight_barriers.py','plot_tight_barriers.py','compare_completed_tight_barriers.py','compare_final_identity.py','compare_newton_traces.py','plot_newton_traces.py','run_final_diagnostics.py','run_final_diagnostics_002.py','validate_box_layout.py','package_evidence.py']:
 p=src/name
 if p.exists():copy(p,dst/'scripts'/name)
