from pathlib import Path
import gzip, hashlib, json, shutil
B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
D=R/'matter/examples/cardboard/evidence/native-implementation-20261014'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
index=json.loads((D/'summary.json').read_text())
records={r['case']:r for r in index['records']}
cases=json.loads((B/'publication022-cases-prepared.json').read_text())['cases']
extra='''source135-cpu-contracts-001
source135-root-integration-preflight-001
captured419-linear-conditioning-001
source125-medium1691-independent-endpoint-004
native435-enabled-fallback-adjudication-001
captured419-regularized-directions-001
native-regularized-direction-replay-436
native-regularized-direction-adjudication-001
captured419-truncated-cgls-directions-001
cache-projection-reuse-runtime-overlay-136-001
source136-cache-runtime-integration-001
retained-source-snapshot136-001
cauchy-operator-tools-abi-overlay-137-001
repeated-cauchy-selector-overlay-137-001
repeated-cauchy-selector-review-137-001
source137-repeated-fallback-integration-001
retained-source-snapshot137-001'''.split()
cases += ['analysis/'+n for n in extra]
cases += ['analysis/source137-root-integration-preflight-001','cpu-checks-137',
    'solver/cache-reuse-full-history-parity-437','native/source137-direct-operator-abi-439']
O=B/'analysis/publication022-preflight-001';O.mkdir()
for n in [429,430,431,432,433,434,435,436,437]:
    for name in [f'native-{n}-command.json',f'run_native{n}.py',f'native-{n}-runner.log',f'prepare_native{n}.py',f'audit_native{n}.py']:
        p=B/name
        if p.is_file():shutil.copy2(p,O/name)
for n in [133,134,135,136,137]:
    for name in [f'build-command-{n}.json',f'assemble_source_{n}.py',f'assemble_source{n}.py',f'build{n}-runner.log',f'retain_snapshot{n}.py',f'retain_source_snapshot{n}_001.py']:
        p=B/name
        if p.is_file():shutil.copy2(p,O/name)
    p=R/f'build-cardboard-provenance-{n}/source-build.json'
    assert json.loads(p.read_text())['success'];shutil.copy2(p,O/f'source-build-{n}.json')
p=B/'analysis/emergency-closed-generated-object-retirement-001.json'
assert p.is_file();shutil.copy2(p,O/p.name)
(O/'archive.py').write_bytes(Path(__file__).read_bytes())
cases += ['analysis/publication022-preflight-001']
assert len(set(cases))==len(cases)
paths=[]
for case in cases:
    assert case not in records,case
    root=B/case;assert root.is_dir(),case
    if case.startswith(('solver/','native/')):
        receipt=json.loads((root/'receipt.json').read_text())
        assert receipt.get('execution_started',True) and receipt.get('source_inputs_unchanged',True),case
    rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(root)
        if any(x in rel.parts for x in ['inputs','__pycache__','source','build','base','candidate']):continue
        if any(x.startswith('CMakeFiles') for x in rel.parts):continue
        assert not p.is_symlink(),str(p)
        target=D/case/rel;target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists(),str(target)
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
        row=dict(name=str(rel),bytes=len(raw),sha256=h,external_path=str(p))
        if len(raw)<=262144 or p.suffix=='.gz':
            target.write_bytes(raw);assert sha(target)==h
            row['repository_copy']=str(target.relative_to(D))
        else:
            target=target.with_name(target.name+'.gz')
            packed=gzip.compress(raw,compresslevel=6,mtime=0);target.write_bytes(packed)
            assert hashlib.sha256(gzip.decompress(packed)).hexdigest()==h
            row.update(repository_compressed_copy=str(target.relative_to(D)),compressed_bytes=len(packed),compressed_sha256=sha(target),restore='gzip decompression restores original bytes; verify original hash')
        rows.append(row);paths.append(str(target.relative_to(R)))
    record=dict(case=case,external_root=str(root),physical_box_qualification=False,files=rows)
    for name in ['receipt','prediction']:
        if (root/(name+'.json')).is_file():record[name]=json.loads((root/(name+'.json')).read_text())
    records[case]=record
prior=D/'summary-reference021.json';assert not prior.exists()
prior.write_bytes((D/'summary.json').read_bytes());paths.append(str(prior.relative_to(R)))
index.setdefault('historical_summary_snapshots',{})['reference021']=dict(path=prior.name,sha256=sha(prior))
index['records']=list(records.values())
index['milestone']='The exact source135 native fallback repairs primary rejection1:48 with alpha1 and13.224% measured residual decrease, but Newton53 rejects and the full control rolls back. Independent complete endpoint secants agree with native within0.285% while disagreeing with the center tangent; no native derivative/noise-floor qualification. Captured A has condition number4.59e7. Integrated source137 builds69 selected targets, passes74 CPU tests and five native direct-operator regressions; its repeated fallback remains experimental. Cache reuse passes isolated exact-word checks431/432 but source136 requested full coupon437 rejects initialization. No complete coupon, convergence, shipping assembly or crush/drop result.'
q=index['qualification'];assert len(q)==6
q['solver_correctness']='Source123 local material repairs and source125 center derivative/assembly comparisons remain qualified only in their recorded scope. Actual native source135 Cauchy fallback435 agrees with independent kernel audit433, reduces the actual residual at1:48 by13.224%, then full coupon rejects at Newton53 and restores accepted geometry. Fallback-off434 reproduces415 numerical histories exactly. Complete independent endpoint recovery004 agrees with native finite secant within0.285%, but the finite secant remains109.49% from center A; native derivative remains unresolved. Captured matrix condition number4.59e7 identifies a dominant weak correction mode. Source13769-target build/74 CPU tests pass; repeated fallback and cache optimization have no complete native trajectory qualification.'
q['numerical_convergence']='No complete spatial, timestep or slower-loading series. Isolated tangent-cache reuse431/432 is bitexact for all cache/action words, but opt-in source136 full coupon437 rejects initialization before physics. No performance, shipping allocation or scalability qualification.'
(D/'summary.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
binding=dict(schema='numi.cardboard.reference022-publication-binding.v1',cases=cases,
    summary_sha256=sha(D/'summary.json'),copied_files=sum(len(records[c]['files']) for c in cases),
    all_archived_bytes_verified=True,physical_validation=False)
p=D/'publication-binding-reference022.json';assert not p.exists();p.write_text(json.dumps(binding,indent=2)+'\n')
paths.extend([str((D/'summary.json').relative_to(R)),str(p.relative_to(R))])
(B/'publication-reference022-pathspec.nul').write_bytes(b''.join(p.encode()+b'\0' for p in paths))
print('archive022',len(cases),'cases',binding['copied_files'],'files',flush=True)
