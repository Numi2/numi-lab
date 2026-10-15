from pathlib import Path
import gzip,hashlib,json,shutil
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');D=R/'matter/examples/cardboard/evidence/native-implementation-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
index=json.loads((D/'summary.json').read_text());records={r['case']:r for r in index['records']}
names='operator-fd-e2-adapter-001 operator-fd-e2-independent-adjudication-419-001 milestone-draft-reference020-419-001 selector5-6-runtime-outcome-overlay-001 selector5-6-runtime-outcome-overlay-002 source131-hybrid-outcome-integration-001 source132-cache-reuse-integration-001 retained-source-snapshots131-132-001 tangent-cache-projection-reuse-parity-125-002-capture419 cache-reuse-native427428-inputs-001 generated-dependency-table-retirement-005'.split()
cases=['analysis/'+n for n in names]
for p in sorted((B/'native').iterdir()):
 if p.is_dir() and p.name.rsplit('-',1)[-1].isdigit() and 425<=int(p.name.rsplit('-',1)[-1])<=428:cases.append(str(p.relative_to(B)))
O=B/'analysis/publication021-preflight-001';O.mkdir(exist_ok=False)
for n in range(425,429):
 for name in [f'native-{n}-command.json',f'run_native{n}.py',f'native-{n}-runner.log']:
  p=B/name
  if p.is_file():shutil.copy2(p,O/name)
for n in range(131,133):
 for name in [f'build-command-{n}.json',f'assemble_source_{n}.py',f'build{n}-runner.log']:
  p=B/name
  if p.is_file():shutil.copy2(p,O/name)
 p=R/f'build-cardboard-provenance-{n}/source-build.json'
 assert p.is_file();shutil.copy2(p,O/f'source-build-{n}.json')
(O/'archive.py').write_bytes(Path(__file__).read_bytes());cases+=['analysis/publication021-preflight-001'];paths=[]
for case in cases:
 assert case not in records,case
 root=B/case;assert root.is_dir(),case;rows=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(root)
  if any(x in rel.parts for x in ['inputs','__pycache__','source','build','base','candidate']):continue
  if any(x.startswith('CMakeFiles') for x in rel.parts):continue
  target=D/case/rel;target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists(),str(target);raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();row=dict(name=str(rel),bytes=len(raw),sha256=h,external_path=str(p))
  if len(raw)<=262144 or p.suffix=='.gz':target.write_bytes(raw);assert sha(target)==h;row['repository_copy']=str(target.relative_to(D))
  else:
   target=target.with_name(target.name+'.gz');packed=gzip.compress(raw,compresslevel=6,mtime=0);target.write_bytes(packed);assert hashlib.sha256(gzip.decompress(packed)).hexdigest()==h;row.update(repository_compressed_copy=str(target.relative_to(D)),compressed_bytes=len(packed),compressed_sha256=sha(target),restore='gzip decompression restores original bytes; verify original hash')
  rows.append(row);paths.append(str(target.relative_to(R)))
 record=dict(case=case,external_root=str(root),physical_box_qualification=False,files=rows)
 for name in ['receipt','prediction']:
  if (root/(name+'.json')).exists():record[name]=json.loads((root/(name+'.json')).read_text())
 records[case]=record
index['records']=list(records.values())
index['milestone']='Exact source125 Newton48 center material and assembled action agree with independentFP64; production-residual finite differences remain unresolved. The complete coupon remains failed after one accepted step. CCDbudget replay detects contact; small hybrid runtimes accept only with residual descent, then checker correction-field assertion rejects. Cache-reuse checker failures are retained; no production optimization adopted.'
q=index['qualification'];assert len(q)==6
q['solver_correctness']='Source123 repairs two exact retained local material failures. Diagnostic source125 preserves production histories at exact failure419. Independent center material derivatives and assembled action pass the selected comparisons (global action error2.822746e-5 relative); FP64 local directional FD at2^-22 passes1e-6. Native424 endpoints reproduce captured words but no epsilon resolves nativeFD. Full coupon and nonlinear-contact correctness remain open. Native426 both selectors accept one microstep with matching certificates under existing residual descent, but diagnostic checker subsequently rejects its correction-field assertion; no runtime parity qualification claimed.'
q['numerical_convergence']='No complete spatial, timestep or slower-loading series. Compile-only combined32x8x4 topology421 executes zero physics. No shipping allocation or scalability qualification.'
prior=D/'summary-reference020.json';assert not prior.exists();prior.write_bytes((D/'summary.json').read_bytes());paths.append(str(prior.relative_to(R)))
index.setdefault('historical_summary_snapshots',{})['reference020']=dict(path=prior.name,sha256=sha(prior))
(D/'summary.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n');binding=dict(schema='numi.cardboard.reference021-publication-binding.v1',cases=cases,summary_sha256=sha(D/'summary.json'),copied_files=sum(len(records[c]['files']) for c in cases),all_archived_bytes_verified=True,physical_validation=False);p=D/'publication-binding-reference021.json';assert not p.exists();p.write_text(json.dumps(binding,indent=2)+'\n');paths.extend([str((D/'summary.json').relative_to(R)),str(p.relative_to(R))]);(B/'publication-reference021-pathspec.nul').write_bytes(b''.join(p.encode()+b'\0' for p in paths));print('archive021',len(cases),'cases',binding['copied_files'],'files')
