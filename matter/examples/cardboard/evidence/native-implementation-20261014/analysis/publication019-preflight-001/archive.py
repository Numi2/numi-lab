from pathlib import Path
import gzip,hashlib,json,shutil
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');D=R/'matter/examples/cardboard/evidence/native-implementation-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
index=json.loads((D/'summary.json').read_text());records={r['case']:r for r in index['records']}
names='''source122-integration-setup-failed-001 source122-integration-001 source123-two-stage-merit-integration-001 source122-123-build-bindings-001 source123-main-adoption-001 retained-source-snapshots122-123-001 private-trial-vtccd-call-observer-overlay-001 source392-tet858-trace-404-adjudication-001 trial21-22-exact-vtccd-adjudication-001 two-stage-native407-fp64-reference-001 two-stage-native410-sweep-summary-001 native-medium-full-matrix-410-001 controlled-material-sweep-410-001 native410-absolute-endpoint-error-001 source392-trace-input-provenance-overlay-001 generated-closed-object-dependency-retirement-007 generated-closed-object-dependency-retirement-008'''.split()
cases=['analysis/'+n for n in names]
for p in sorted((B/'native').iterdir()):
 if p.is_dir() and p.name.rsplit('-',1)[-1].isdigit() and 404<=int(p.name.rsplit('-',1)[-1])<=411:cases.append(str(p.relative_to(B)))
O=B/'analysis/publication019-preflight-001';O.mkdir(exist_ok=False)
for n in range(404,412):
 for p in [B/f'native-{n}-command.json',B/f'run_native{n}.py',B/f'native-{n}-runner.log']:
  if p.is_file() and n!=409:shutil.copy2(p,O/p.name)
for p in [B/'native-409-command.json',B/'run_native409.py',B/'build-command-122.json',B/'build-command-123.json',B/'build-command-123a.json',B/'build_retry123a.py']:
 assert p.is_file();shutil.copy2(p,O/p.name)
(O/'pending409.json').write_text(json.dumps(dict(case='solver/two-stage-certified-coupon-409',status='running_at_publication_preflight',source_build_sha256=sha(R/'build-cardboard-provenance-123a/source-build.json'),limits='Complete27controlhistory not yet available; onlycommand/runner retained here. Do not infer completion or qualification from currentacceptedprefix.',physical_box_validation=False),indent=2)+'\n');(O/'archive.py').write_bytes(Path(__file__).read_bytes());cases+=['analysis/publication019-preflight-001'];paths=[]
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
index['records']=list(records.values());index['milestone']='Source123 two-stage local material merit resolves exact native674/858 points and passes selected independent comparisons. All3072controllednative samples valid, smallerstepFDfailures retained. Actual privateVTCCDcall reveals separate iterationlimit at nearthresholdcontact. Full27controlcoupon409 remains running; no prefix qualifies convergence.'
q=index['qualification'];assert len(q)==6
q['solver_correctness']='Native407/411 tet85814updates and408 tet674fiveupdates pass unchanged localgates, traceoff parityzero. IndependentFP64 at858 stress5.20e-8/tangent3.44e-8relative. Native4103072samples valid, selectedmedium9columnanalyticgate passes. ExactprivateVTCCD405406 preserves129capturearrays/32Armijorows/transactions against398399; trial21 exhausts16iterationCCDcap,22nearthresholdhit. Firstfailuregeneration and fullcontact remainopen. Fullcoupon409 running; no completedhistory or fullsolver qualification.'
q['numerical_convergence']='No completed spatial/time/slowerloadingstudy. Native410selected3unique-state90digitFDaudit all27directions perstresspath pass1e-3 nonzerovectorgate at1e-4, worst1.03e-4; one direction failsat1e-5/1e-6, bothstresspaths retained. No universalnearzeronoisefloor or replacement for unresolvednominalFP64/element/assembledgates. Combinedrefinement still exceedsselector5admission; experimentalhybrid not yetnativequalified.'
(D/'summary.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n');binding=dict(schema='numi.cardboard.reference019-publication-binding.v1',cases=cases,summary_sha256=sha(D/'summary.json'),copied_files=sum(len(records[c]['files']) for c in cases),all_archived_bytes_verified=True,physical_validation=False);p=D/'publication-binding-reference019.json';assert not p.exists();p.write_text(json.dumps(binding,indent=2)+'\n');paths.extend([str((D/'summary.json').relative_to(R)),str(p.relative_to(R))]);(B/'publication-reference019-pathspec.nul').write_bytes(b''.join(p.encode()+b'\0' for p in paths));print('archive019',len(cases),'cases',binding['copied_files'],'files')
