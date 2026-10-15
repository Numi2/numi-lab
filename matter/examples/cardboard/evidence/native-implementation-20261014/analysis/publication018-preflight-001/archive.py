from pathlib import Path
import gzip,hashlib,json,shutil
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');D=R/'matter/examples/cardboard/evidence/native-implementation-20261014'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
index=json.loads((D/'summary.json').read_text());records={r['case']:r for r in index['records']}
names='''source118-scaled-material-merit-integration-001 source118-restoration-cli-001 source119-integration-001 source120-integration-001 source121-integration-001 source121-integration-preflight-failed-001 source118-121-build-bindings-001 retained-source-snapshot118-001 retained-source-snapshots119-121-001 source119-121-restoration-cli-001 source121-main-adoption-001 dimensionless-material674-point-390-adjudication-001 dimensionless-material674-point-390-strict-reference-addendum-001 cardboard-source380-point-regression-verifier-001 cardboard-source380-point-regression-verifier-supersession-e47b7357-001 dimensionless-material674-root-verification-002 dimensionless-material674-root-verification-003 dimensionless-material674-root-verification-004 integrated-material674-native394-verification-001 coupon-392-status5-audit-001 source392-status5-capture-design-001 source392-tet858-point-overlay-001 native401-manifest-alias-bindings-001 armijo-all-trial-summary-overlay-001 armijo-all-trial-summary-overlay-002 armijo-all-trial-summary-overlay-003 armijo-all-trial-summary-analyzer-001 valid-summary-native396-verification-001 n9-linear-vs-noop-385-001 projection-trace-host-capacity-121-001 c1-native402-trace-capacity-verification-001 c1-native403-trace-capacity-verification-001 shipping-damage-face-ownership-memory-audit-115-002 retained-face-guard-fresh-compiler118-001 cmake-transitive-metal-deps-118-001 coupon-resolution-native-cap-audit-001 coupon-resolution-native-cap-audit-002-source119-current-checkout-addendum generated-118full-object-dependency-retirement-005 generated-closed-object-dependency-retirement-006 tet858-cpu-merit-experiments-001 material-two-stage-merit-prototype-001 material-two-stage-merit-prototype-002 exact-material-merit-prototype-003'''.split()
cases=['analysis/'+n for n in names]
for p in sorted((B/'native').iterdir()):
 if p.is_dir() and p.name.rsplit('-',1)[-1].isdigit() and 390<=int(p.name.rsplit('-',1)[-1])<=403:cases.append(str(p.relative_to(B)))
cases+=['solver/dimensionless-certified-coupon-392','cpu-checks-118-full','cpu-checks-119']
cases=[c for c in cases if c not in records]
O=B/'analysis/publication018-preflight-001';O.mkdir(exist_ok=False)
commands=[]
for n in range(390,404):
 for p in [B/f'native-{n}-command.json',B/f'run_native{n}.py',B/f'native-{n}-runner.log']:
  if p.is_file():shutil.copy2(p,O/p.name);commands.append(dict(path=str(p),sha256=sha(p)))
for n in ['118','118-full','119','120','121']:
 for p in [B/f'build-command-{n}.json',B/f'build{n}-runner.log']:
  if p.is_file():shutil.copy2(p,O/p.name);commands.append(dict(path=str(p),sha256=sha(p)))
(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
(O/'archive.py').write_bytes(Path(__file__).read_bytes());cases+=['analysis/publication018-preflight-001']
paths=[]
for case in cases:
 assert case not in records,case
 root=B/case;assert root.is_dir(),case;rows=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(root)
  if any(x in rel.parts for x in ['inputs','__pycache__','source','build','base','candidate']):continue
  if any(x.startswith('CMakeFiles') for x in rel.parts):continue
  target=D/case/rel;target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists(),str(target)
  row=dict(name=str(rel),bytes=p.stat().st_size,sha256=sha(p),external_path=str(p))
  if p.stat().st_size<=256*1024 or p.suffix=='.gz':
   shutil.copy2(p,target);assert sha(target)==row['sha256'];row['repository_copy']=str(target.relative_to(D))
  else:
   target=target.with_name(target.name+'.gz');packed=gzip.compress(p.read_bytes(),compresslevel=6,mtime=0);target.write_bytes(packed)
   assert hashlib.sha256(gzip.decompress(packed)).hexdigest()==row['sha256']
   row.update(repository_compressed_copy=str(target.relative_to(D)),compressed_bytes=len(packed),compressed_sha256=sha(target),restore='gzip decompression restores original bytes; verify original hash')
  rows.append(row);paths.append(str(target.relative_to(R)))
 record=dict(case=case,external_root=str(root),physical_box_qualification=False,files=rows)
 for name in ['receipt','prediction']:
  if (root/(name+'.json')).exists():record[name]=json.loads((root/(name+'.json')).read_text())
 records[case]=record
index['records']=list(records.values());index['milestone']='Source121 native material674 repair, exact retained tet858 local failure, complete rejected contact-trial evidence, diagnostic buffer repair and resolution admission audit. Native858 remains failed; CPU two-stage merit prototype requires native qualification.'
q=index['qualification']
q['solver_correctness']='Exact material674 scaled projection passes native390/394 with direct/tangent/trace-off bit parity and strict FP64 root comparison. Complete coupon392 rejects control0/tet858; native401 reproduces32-update local failure. Earlier14-update CPU result used wrong raw merit and is superseded. Exact-bit CPU prototype003 demonstrates raw-then-scaled candidate passes674/858, pending native trace and full coupon. Full-contact396 valid all-trial capture preserves32production histories;22CCD failures and2one-ULP residual increases remain rejected. No global tolerance relaxed. C1trace32buffer repair passes native402API and403API/GPUvalidation;400timeout retained.'
q['numerical_convergence']='No complete mesh/time/slower-loading coupon study. Requested single-axis budgeted layouts admitted, combined20682node/89088tet mesh exceeds selector5caps. Separate hybrid mode remains unqualified. Temporal planner tightens relative tolerance by factor; observed force residual bound alone does not establish equivalent absolute force accuracy. Prior nominal-direction FP64 failures and near-zero noise-floor limits remain.'
q['component_calibration']='Retained damage path now rejects complete shared-face ownership with rotated/sliver checks; this is not mesh-objective or calibrated damage. Robertsson absolute-energy conflict retained. Kraft/NSSC component and local starch/hot-melt fracture parameter identifiability remain open; public analogues do not close exact-recipe calibration.'
q['analogue_validation']='Published source grades and separate series retained; no new full-board analogue validation pass.'
q['physical_assembly']='Shipping explicit-flute topology still fails native admission; no fully scored, folded, bonded, released and settled shipping-box checkpoint.'
q['crush_impact_prediction']='No qualified native box compression curve, frozenFpeak, capacity-derived masses, gravity-crush or drop-impact result.'
(D/'summary.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
binding=dict(schema='numi.cardboard.reference018-publication-binding.v1',cases=cases,summary_sha256=sha(D/'summary.json'),copied_files=sum(len(records[c]['files']) for c in cases),physical_validation=False)
p=D/'publication-binding-reference018.json';assert not p.exists();p.write_text(json.dumps(binding,indent=2)+'\n');paths.extend([str((D/'summary.json').relative_to(R)),str(p.relative_to(R))])
(B/'publication-reference018-pathspec.nul').write_bytes(b''.join(p.encode()+b'\0' for p in paths));print(json.dumps(binding,indent=2))
