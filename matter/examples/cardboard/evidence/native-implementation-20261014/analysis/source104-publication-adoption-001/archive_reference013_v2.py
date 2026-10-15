from pathlib import Path
import gzip,hashlib,json,shutil

B=Path('/Users/home/cardboard-implementation-20261014')
REPO=Path('/Users/home/numi-cardboard-20261006')
D=REPO/'matter/examples/cardboard/evidence/native-implementation-20261014'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
index=json.loads((D/'summary.json').read_text())
records={r['case']:r for r in index['records']}
cases=[
 'analysis/source104-diagnostic-integration-001',
 'analysis/retained-source-snapshot104-001',
 'analysis/source104-native-parity-362-audit-001',
 'analysis/source104-native-parity-362-audit-002',
 'analysis/source101-capture360-352-parity-001',
 'analysis/source101-capture360-fp64-linear-solve-001',
 'analysis/source101-control2-newton23-audit-001',
 'analysis/source101-control2-newton23-audit-002',
 'analysis/source101-control2-newton23-audit-003',
 'analysis/source101-control2-newton23-audit-004',
 'analysis/source101-control2-newton23-audit-005',
 *[f'analysis/source101-regional361-actual-newton23-{n:03d}' for n in range(1,7)],
 'native/health-and-vt-marker-firstfatal-362',
 'solver/reset-augmentation-first-failure-capture-360',
 'solver/reset-augmentation-first-failure-rap-only-361',
 'cpu-checks-104',
]
for case in cases:
 if case in records:continue
 root=B/case
 assert root.is_dir(),case
 assert case not in records,case
 rows=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(root)
  if any(x in rel.parts for x in ['inputs','__pycache__','source','build','build-v2','build-v2-fresh']):continue
  if any(x.startswith('CMakeFiles') for x in rel.parts):continue
  if p.name=='.DS_Store':continue
  digest=sha(p);target=D/case/rel;target.parent.mkdir(parents=True,exist_ok=True)
  row=dict(name=str(rel),bytes=p.stat().st_size,sha256=digest,external_path=str(p))
  if p.stat().st_size<=256*1024 or p.suffix=='.gz':
   assert not target.exists() or sha(target)==digest;shutil.copy2(p,target)
   row['repository_copy']=str(target.relative_to(D))
  else:
   target=target.with_name(target.name+'.gz')
   data=gzip.compress(p.read_bytes(),compresslevel=6,mtime=0)
   assert not target.exists() or target.read_bytes()==data
   target.write_bytes(data)
   assert hashlib.sha256(gzip.decompress(data)).hexdigest()==digest
   row.update(repository_compressed_copy=str(target.relative_to(D)),compressed_bytes=len(data),compressed_sha256=sha(target),restore='gzip decompression restores original bytes; verify original bytes/sha256')
  rows.append(row)
 records[case]=dict(case=case,external_root=str(root),physical_box_qualification=False,files=rows)
 for name in ['receipt','prediction']:
  if (root/(name+'.json')).exists():records[case][name]=json.loads((root/(name+'.json')).read_text())
root=REPO/'build-cardboard-provenance-104';case='build/source-build-104'
assert case not in records
rows=[]
for p in [root/'source-build.json',*sorted(root.glob('provenance-command-*.log'))]:
 target=D/case/p.name;target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists() or sha(target)==digest;shutil.copy2(p,target)
 rows.append(dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p),external_path=str(p),repository_copy=str(target.relative_to(D))))
records[case]=dict(case=case,external_root=str(root),physical_box_qualification=False,files=rows)
index['records']=list(records.values())
index['milestone']='Source104 diagnostic-health parity; actual source101 failed-Newton operator/conditioning evidence; physical stages incomplete'
index['qualification']['solver_correctness']='Source104 diagnostic-only instrumentation passes68targetbuild/73CPUtests and failed-prefix362/359 physical-array/Armijo parity. Actual source101 Newton23 capture360 has false Arnoldi completion: fresh residual2.1578e-4Ns vs target1.1993e-5Ns. Regional361 AP/factors agree with cached FP64 reconstruction below5e-7; independent center material accuracy remains pending. FD perturbations fail so derivative comparison is inconclusive. Full coupled solver remains unqualified.'
index['qualification']['numerical_convergence']='No complete coupon/mesh/time/slower-loading series.360/361 remain two accepted controls of27 then rejection, preserving37ordinary files and64same-root Armijo rows exactly.'
(D/'summary.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
receipt=dict(schema='numi.cardboard.reference013-publication-binding.v1',
 cases=cases+['build/source-build-104'],summary_sha256=sha(D/'summary.json'),
 copied_files=sum(len(records[c]['files']) for c in cases)+len(rows),
 source104_build_sha256=sha(root/'source-build.json'),physical_validation=False)
(D/'publication-binding-reference013.json').write_text(json.dumps(receipt,indent=2)+'\n')
shutil.copy2(__file__,B/'analysis/source104-diagnostic-integration-001/archive_reference013.py')
print(json.dumps(receipt,indent=2))
