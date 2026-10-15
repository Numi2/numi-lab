from pathlib import Path
import gzip,hashlib,json,shutil
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');D=R/'matter/examples/cardboard/evidence/native-implementation-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
index=json.loads((D/'summary.json').read_text());records={r['case']:r for r in index['records']}
names='''two-stage-coupon409-adjudication-001 paper-coupon-cost-audit-001 partial-graph-hybrid-selector6-overlay-001 partial-graph-hybrid-selector6-native-check-overlay-002 source124-hybrid-integration-001 source392-status5-observer-overlay-122 private-vtccd-budget-replay-overlay-001 source125-diagnostic-integration-001 selector6-fixture-repair-overlay-001 source126-hybrid-fixture-integration-001 vtccd-absolute-capture-path-overlay-001 source127-vtccd-reader-integration-001 source128-hybrid-status-integration-001 source129-hybrid-status-includes-integration-001 source129-fixture-cpu-validation-001 native417-input-preflight-001 vtccd-replay-input-alias-bindings-001 retained-source-snapshots124-128-001 retained-source-snapshots128-129-001 retained-source-snapshot130-001 native415-status5-parity-001 status5-alpha-lane-overlay-001 native419-history-parity-001 native417-vtccd-budget-adjudication-001 native-operator-term-replay-419-001 native-operator-term-replay-result-422 source125-native-operator-term-audit-001 native-residual-fd-sweep-419-001 native-residual-fd-sweep-result-424 native424-residual-sweep-audit-001 tangent-cache-projection-reuse-parity-125-001 source130-cache-reuse-integration-001 cache-reuse-native423-inputs-001 source131-integration-preflight-failure-001 generated-closed-object-dependency-retirement-009 generated-closed-object-dependency-retirement-010 generated-closed-object-dependency-retirement-011 generated-closed-object-dependency-retirement-012 generated-closed-object-dependency-retirement-013 generated-closed-object-dependency-retirement-014 generated-closed-shader-air-retirement-001 generated-closed-shader-air-retirement-002 generated-dependency-table-retirement-004'''.split()
cases=['analysis/'+n for n in names]
for p in sorted((B/'native').iterdir()):
 if p.is_dir() and p.name.rsplit('-',1)[-1].isdigit() and 412<=int(p.name.rsplit('-',1)[-1])<=423:cases.append(str(p.relative_to(B)))
cases+=['solver/two-stage-coupon-first-descent-capture-415','solver/two-stage-first-failure-rich-fd-419','cpu/hybrid-combined-refinement-preflight-421','cpu-checks-124','cpu-checks-126']
O=B/'analysis/publication020-preflight-001';O.mkdir(exist_ok=False)
for n in range(412,425):
 for name in [f'native-{n}-command.json',f'run_native{n}.py',f'native-{n}-runner.log']:
  p=B/name
  if p.is_file():shutil.copy2(p,O/name)
for n in range(124,131):
 for name in [f'build-command-{n}.json',f'assemble_source_{n}.py',f'build{n}-runner.log']:
  p=B/name
  if p.is_file():shutil.copy2(p,O/name)
 p=R/f'build-cardboard-provenance-{n}/source-build.json'
 assert p.is_file();shutil.copy2(p,O/f'source-build-{n}.json')
(O/'archive.py').write_bytes(Path(__file__).read_bytes());cases+=['analysis/publication020-preflight-001'];paths=[]
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
index['records']=list(records.values());index['milestone']='Exact source125 control1/Newton48 paired failure captured with accepted history, material state, frames, constraints, masses, correction and signed endpoints. Diagnostic production histories match409/415. Exact source125 native operator and residual replay preserve captured words; smaller native perturbations do not resolve a derivative. Full cycle remains failed after one accepted step. Higher CCD replay budget changes an uncertified query to a hit; complete contact and experimental hybrid runtime parity remain unqualified.'
q=index['qualification'];assert len(q)==6
q['solver_correctness']='Source123 repairs both retained local material failures. Source125 failure419 exactly preserves415 output files and409 Armijo/linear histories. Native422 fresh action matches captured419 bits; independent cached-coefficient frame/mass/gather audit passes the selected assembled1% comparison, but does not independently validate constitutive derivatives or noise floors. Native424 center and allfour retained endpoints match capture words; smaller epsilons remain unresolved. Native417 production16CCD return bits match retained405406; larger budgets detect the same contact. No complete coupon or nonlinear-contact qualification.'
q['numerical_convergence']='No complete spatial, timestep or slower-loading series. Exact32x8x4 selector6 compile-only topology preflight421 passes20682nodes/89088tets, zero physics. Experimentalhybrid kernel checks pass but accepted runtime parity fails420. No shipping allocation or scalability qualification.'
(D/'summary.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n');binding=dict(schema='numi.cardboard.reference020-publication-binding.v1',cases=cases,summary_sha256=sha(D/'summary.json'),copied_files=sum(len(records[c]['files']) for c in cases),all_archived_bytes_verified=True,physical_validation=False);p=D/'publication-binding-reference020.json';assert not p.exists();p.write_text(json.dumps(binding,indent=2)+'\n');paths.extend([str((D/'summary.json').relative_to(R)),str(p.relative_to(R))]);(B/'publication-reference020-pathspec.nul').write_bytes(b''.join(p.encode()+b'\0' for p in paths));print('archive020',len(cases),'cases',binding['copied_files'],'files')
