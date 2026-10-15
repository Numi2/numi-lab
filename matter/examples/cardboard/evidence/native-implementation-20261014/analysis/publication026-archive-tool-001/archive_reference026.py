from pathlib import Path
import gzip,hashlib,json
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');D=R/'matter/examples/cardboard/evidence/native-implementation-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
EXCLUDE={'inputs','build','base','candidate','__pycache__','archives','restore-check'}
def allowed(rel):return not any(p in EXCLUDE or p.startswith(('CMakeFiles','.air')) for p in rel.parts)
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
oldbytes=(D/'summary.json').read_bytes();old=json.loads(oldbytes);binding=json.loads((D/'publication-binding-reference025.json').read_text());pathbytes=(B/'publication-reference025-pathspec.nul').read_bytes();assert binding['summary_sha256']==hashlib.sha256(oldbytes).hexdigest();assert len(old['records'])==1398
oldpaths={x.decode() for x in pathbytes.split(b'\0') if x};assert len(oldpaths)==4070
cases=json.loads((B/'publication026-cases-prepared.json').read_text())['cases'];assert len(cases)==26;assert not {x['case'] for x in old['records']}.intersection(cases);assert not (D/'summary-reference025.json').exists();assert not (D/'publication-binding-reference026.json').exists()
assert json.loads((B/'analysis/source149-right-scale-disabled462-parity-001/report.json').read_text())['parity_pass'];assert json.loads((B/'analysis/source149-root-integration-002/report.json').read_text())['source_files_verified']==787
(D/'summary-reference025.json').write_bytes(oldbytes);newrows=[];newpaths=set();rawcount=gzcount=0
for case in cases:
 root=B/case;rows=[]
 for src in sorted(root.rglob('*')):
  rel=src.relative_to(root)
  if not src.is_file() or not allowed(rel):continue
  assert not src.is_symlink(),src
  raw=src.read_bytes();digest=hashlib.sha256(raw).hexdigest();target=D/case/rel;assert not target.exists(),target;target.parent.mkdir(parents=True,exist_ok=True)
  row={'name':str(rel),'bytes':len(raw),'sha256':digest,'external_path':str(src)}
  if len(raw)<=262144 or src.suffix=='.gz':target.write_bytes(raw);assert sha(target)==digest;row['repository_copy']=str(target.relative_to(D));rawcount+=1
  else:
   packed=gzip.compress(raw,compresslevel=6,mtime=0);target=target.with_name(target.name+'.gz');assert not target.exists(),target;target.write_bytes(packed);assert hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest()==digest;row.update(repository_compressed_copy=str(target.relative_to(D)),compressed_bytes=len(packed),compressed_sha256=sha(target),restore='gzip decompression recovers original bytes; verify original sha256');gzcount+=1
  rows.append(row);newpaths.add('matter/examples/cardboard/evidence/native-implementation-20261014/'+str(target.relative_to(D)))
 assert rows,case
 rec={'case':case,'external_root':str(root),'physical_box_qualification':False,'files':rows}
 for key in ['receipt','prediction']:
  p=root/(key+'.json')
  if p.is_file():rec[key]=json.loads(p.read_text())
 newrows.append(rec)
verdicts={'solver_correctness': {'status': 'partial_component_evidence; complete_nonlinear_solve_unqualified', 'evidence': 'Native460 actual Metal scale/transpose/CGLS kernels pass synthetic checks. Native463 exact463->455->419 lineage and current local-block/action comparisons pass their scoped audits; all16 late privateupdatesaccepted but control1 still fails. Native462 complete disabledhistory parity passes. MetalFP32finite-difference noisefloor unresolved.'}, 'numerical_convergence': {'status': 'unqualified', 'evidence': 'No completecoupon spatial/time/slower-loadingstudy. Native463 rightblockscaling improves finalprivate residual9.4420302e-6Ns butfailsunchangedcertificate and rollsback; no completehistories.'}, 'component_calibration': {'status': 'partial_source_data_only', 'evidence': 'Publisheddata/grade/absoluteenergy/damage/interface implementations retained. Exactkraft/NSSCprogressivefailure and localstarchstrength/GIc/GIIc remainunidentified.'}, 'analogue_validation': {'status': 'unqualified_for_requested_full_sequence', 'evidence': 'Nearbygrade,fixture,conditioning andrate transfer limits remain; no new validationseriespassed.'}, 'physical_assembly_in_simulation': {'status': 'unqualified', 'evidence': 'No completed12-score fold/lap/endbond/release and fixture-free settledshippingbox. Explicitflutefullcontact/memoryadmission remainopen.'}, 'crush_impact_prediction': {'status': 'unexecuted', 'evidence': 'No qualified nominalcompressionfirstpeak, scenario masses, gravitycrush or100mmdrop; exactbareboximpactvalidationunavailable.'}}
summary=json.loads(oldbytes);summary.setdefault('historical_summary_snapshots',{})['reference025']={'path':'summary-reference025.json','sha256':hashlib.sha256(oldbytes).hexdigest()};summary['records'].extend(newrows);assert summary['records'][:1398]==old['records'];summary['verdicts']=verdicts
summary['milestone']='Source149 adds bounded opt-in native node-block right scaling with exact transpose and private scratch. Actual native460 synthetic-kernel checks pass; exact463/455/419 physical-state lineage and operator actions are audited. Native462 preserves the complete disabled attempted history. Native463 improves its final trial residual but still fails and rolls back control1. No complete coupon, numerical convergence, calibrated corrugated failure, released shipping assembly or crush/drop qualification.'
summary['qualification']['solver_correctness']='Scoped captured analytic comparisons, native synthetic conditioning checks and disabled-history controls pass. Complete nonlinear/contact solve remains unqualified; current projected history is unavailable in candidate capture and Metal FP32 derivative floor is inconclusive.'
summary['qualification']['numerical_convergence']='No complete coupon or mesh/time/slower-loading histories. Right scaling463 improves final trial to9.4420302e-6Ns butstillfailscontrol1; previous456/457alpha1counterexamples retained.'
write(D/'summary.json',summary)
for p in ['summary-reference025.json','publication-binding-reference026.json']:newpaths.add('matter/examples/cardboard/evidence/native-implementation-20261014/'+p)
assert not oldpaths.intersection(newpaths)
newbinding={'schema':'numi.cardboard.reference026-publication-binding.v1','prior_binding_sha256':sha(D/'publication-binding-reference025.json'),'prior_pathspec_sha256':hashlib.sha256(pathbytes).hexdigest(),'prior_pathspec_entries':len(oldpaths),'historical_prefix':{'path':'summary-reference025.json','records':1398,'sha256':hashlib.sha256(oldbytes).hexdigest()},'cases':cases,'summary_sha256':sha(D/'summary.json'),'previous_prefix_records':1398,'new_case_records':len(newrows),'copied_files':sum(len(x['files']) for x in newrows),'raw_files':rawcount,'gzip_files':gzcount,'all_archived_bytes_verified':True,'physical_validation':False,'six_axis_verdicts':verdicts,'excluded_directory_parts':sorted(EXCLUDE),'excluded_directory_prefixes':['CMakeFiles','.air'],'external_recovery_archives':'Closed rawinputs compaction019/020 archives remain external sole recovery archives; keep them. Restore metadata is published.'};write(D/'publication-binding-reference026.json',newbinding)
p=B/'publication-reference026-pathspec.nul';assert not p.exists();p.write_bytes(pathbytes+b''.join(x.encode()+b'\0' for x in sorted(newpaths)))
print(json.dumps({'cases':len(newrows),'files':newbinding['copied_files'],'raw':rawcount,'gzip':gzcount,'summary_records':len(summary['records']),'binding_sha256':sha(D/'publication-binding-reference026.json'),'pathspec_entries':len(oldpaths)+len(newpaths)},indent=2))
