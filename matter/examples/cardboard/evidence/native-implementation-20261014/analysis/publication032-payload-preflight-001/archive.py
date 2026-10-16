from pathlib import Path
import datetime,gzip,hashlib,json,os,shutil,subprocess
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');R=Q/'checkout-publication032'
P='matter/examples/cardboard/evidence/native-implementation-20261014/';D=R/P
O=Q/'analysis/publication032-payload-preflight-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
enc=lambda x:(json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def put(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b,p
 else:p.write_bytes(b)
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
assert not O.exists();O.mkdir()
assert git('rev-parse','HEAD').decode().strip()=='c622d2323261d6ae26e5b860b1ca4098c8b23951'
oldbytes=(D/'summary.json').read_bytes();assert hashlib.sha256(oldbytes).hexdigest()=='83fcfeb3f1604a2b1ca2fb5b4b82e35367f7bbcd5fb535d2f72002dafaf5e613'
old=json.loads(oldbytes);assert len(old['records'])==1607
S=Q/'snapshots/source-scaled-contact-geometry-188-buildfix002'
frozen=json.loads((S/'frozen-source.json').read_text());sources=frozen['source_sha256'];assert len(sources)==814
for n,h in sources.items():assert sha(R/n)==sha(S/n)==h,n
collection=Q/'publication032-local-evidence';local=json.loads((collection/'collection.json').read_text())
pairs=[(c,collection/c) for c in local['cases']]
extras=[
'analysis/source185-prebuild-verification-001',
'analysis/source188-contact-geometry-buildfix-002',
'analysis/source188-native-validation-001',
'analysis/source188-publication-source-binding-001',
'analysis/native506-scaled-contact-loaded-capture-command-001',
'analysis/native507-scaled-contact-center-capture-command-001',
'analysis/native508-scaled-contact-trial-capture-command-001',
'analysis/native509-contact-free-coupon-regression-001',
'analysis/native510-contact-free-coupon-regression-001',
'native/contact-scaled-nearest-point-503',
'native/contact-scaled-geometry-504',
'native/contact-scaled-geometry-fixture-505',
'native/contact-activation-capture-506',
'native/contact-activation-capture-507',
'native/contact-activation-capture-508',
'solver/warm-cgls-control20-diagnostic-replay-497',
'solver/warm-cgls-control20-diagnostic-replay-off-498',
'solver/warm-cgls-source188-control20-regression-509',
'solver/warm-cgls-source188-control20-regression-510',
'analysis/native497-nonlinear-correction-001',
'analysis/native497-lu-nonlinear-correction-001',
'analysis/native497-two-level-preconditioner-001',
'analysis/native497-two-level-preconditioner-002',
'analysis/native500-compensated-feature-001']
pairs += [(c,Q/c) for c in extras]
# Distinct host cases preserve post-transfer fields beyond the local collected form.
for rel in ['analysis/source184-build-validation-002','analysis/source185-build-validation-001','analysis/source188-contact-geometry-integration-001']:
 pairs.append((rel+'-macmini',Q/rel))
# Retain build metadata and exact executable/Metal bytes; avoid whole duplicate build trees.
H=Q/'analysis/source188-host-artifacts-002';assert not H.exists();H.mkdir()
for bn in ['build-source184-001','build-source184-002','build-source185-001','build-source185-002','build-source188-001','build-source188-002']:
 for name in ['CMakeCache.txt','compile_commands.json','provenance-command-0.log','provenance-command-1.log','source-build.json']:
  p=Q/bn/name;put(H/bn/name,p.read_bytes())
build=json.loads((Q/'build-source188-002/source-build.json').read_text());assert build['success'] and build['source_sha256']==sources
for n,h in build['artifacts_sha256'].items():
 p=Path(n);assert sha(p)==h
 put(H/'artifacts'/p.relative_to(Q/'build-source188-002'),p.read_bytes())
for snap in ['source-admission-encoder-182-buildfix003','source-shared-contact-budget-184','source-portable-reference-185','source-scaled-contact-geometry-188','source-scaled-contact-geometry-188-buildfix002']:
 p=Q/'snapshots'/snap
 if not p.exists():continue
 for n in ['frozen-source.json','configure-support-source.json']:
  f=p/n
  if f.is_file():put(H/'snapshots'/snap/n,f.read_bytes())
put(H/'collection.json',(collection/'collection.json').read_bytes())
put(H/'local-byte-verification.json',(collection/'byte-verification.json').read_bytes())
pairs.append(('analysis/source188-host-artifacts-002',H))
cases=[c for c,r in pairs];assert len(cases)==len(set(cases))
assert not (set(cases)&{r['case'] for r in old['records']})
excluded={'__pycache__','archives','base','build','restore-check','restore-test','inputs','CMakeFiles'}
records=[];payload_paths=[];external=[];raw=compressed=0;original_bytes=stored_bytes=0
for case,root in pairs:
 assert root.is_dir(),root
 files=[]
 for directory,dirs,names in os.walk(root):
  dirs[:]=sorted(n for n in dirs if n not in excluded and not n.startswith('.air'))
  for name in sorted(names):
   p=Path(directory)/name;assert p.is_file() and not p.is_symlink(),p
   data=p.read_bytes();h=hashlib.sha256(data).hexdigest();target=Path(case)/p.relative_to(root)
   row={'name':str(p.relative_to(root)),'bytes':len(data),'sha256':h,'external_path':str(p)}
   if len(data)<=262144 or p.suffix=='.gz':
    payload=data;row['repository_copy']=str(target);raw+=1
   else:
    payload=gzip.compress(data,compresslevel=6,mtime=0);assert gzip.decompress(payload)==data
    target=target.with_name(target.name+'.gz')
    row.update(repository_compressed_copy=str(target),compressed_bytes=len(payload),compressed_sha256=hashlib.sha256(payload).hexdigest(),restore='gzip decompression; verify original SHA-256')
    compressed+=1
   assert not (D/target).exists(),target
   put(D/target,payload);payload_paths.append(P+str(target));external.append((str(p),h))
   files.append(row);original_bytes+=len(data);stored_bytes+=len(payload)
 assert files,root
 rec={'case':case,'external_root':str(root),'physical_box_qualification':False,'files':files}
 for n in ['prediction','receipt']:
  p=root/(n+'.json')
  if p.is_file():rec[n]=json.loads(p.read_text())
 records.append(rec)
for n,h in external:assert sha(n)==h,n
verdicts=dict(old['verdicts'])
verdicts['solver_correctness']={'status':'partial_component_evidence; complete_nonlinear_solve_unqualified','evidence':'Source188 repairs scale-dependent nearest-feature and edge-mollifier algebra in the production contact helpers: 96 native analytical geometry cases pass. The shared full-contact scratch inventory now matches actual allocation, diagnostic replay preserves physical bytes, and the strict portable compensated reference is integrated. The loaded full-contact fixture still fails its unchanged 1e-7 residual gate; native506/507/508 retain captures and identical transaction snapshots. Global solver and complete tangent qualification remain open.'}
verdicts['numerical_convergence']={'status':'unqualified','evidence':'Native497 and498 reproduce35 physical files,4096 true-residual records and64 CGLS records exactly with observers enabled/disabled, while control20 still fails. Strict CPU replay of the actual total correction agrees at the center and shows a narrow nonlinear descent window shared by warm CGLS and exact LU. These CPU trials are not accepted native steps. No complete60-control coupon history, requested mesh study or timestep triplet passes. Native509/510 admission failures are preserved; checkpoint source/program guards remain strict.'}
verdicts['physical_assembly_in_simulation']={'status':'unqualified','evidence':'The exact shared full-contact scratch inventory counts528bytes per contact slot and exposes the actual bounded arena cost. The current explicit-flute shipping estimate is31115178266bytes, exceeding16GiB, before a measured whole-process qualification. No physically scored, folded, bonded, fixture-free300x200x200mm checkpoint exists.'}
summary=json.loads(oldbytes);summary.setdefault('historical_summary_snapshots',{})['reference031']={'path':'summary-reference031.json','sha256':hashlib.sha256(oldbytes).hexdigest()}
summary['records'].extend(records);assert summary['records'][:1607]==old['records']
summary['verdicts']=verdicts
for k in ['solver_correctness','numerical_convergence','physical_assembly_in_simulation']:summary['qualification'][k]=verdicts[k]['evidence']
summary['milestone']='Source188 production contact geometry and exact scratch accounting are implemented and host-tested; strict independent compensated replay and observer parity are retained. Loaded contact and the complete nonlinear coupon still fail. Calibration, mesh/time convergence, physical shipping-box assembly, crush and impact remain open. All selected failed builds, failed runs, admission failures and errata are preserved.'
data=enc(summary)
changed=set(git('diff','--name-only','-z').split(b'\0'))-{b''}
changed|=set(git('ls-files','--others','--exclude-standard','-z').split(b'\0'))-{b''}
# Evidence files were just installed; isolate source ownership through frozen map.
source_changes=sorted(x.decode() for x in changed if x.decode() in sources)
assert len(source_changes)>=18
new_paths=sorted(set(payload_paths+[P+'summary-reference031.json',P+'publication-binding-reference032.json',P+'summary.json']+source_changes))
binding={'schema':'numi.cardboard.reference032-publication-binding.v1','parent':'c622d2323261d6ae26e5b860b1ca4098c8b23951','prior_binding_sha256':sha(D/'publication-binding-reference031.json'),'historical_prefix':{'records':1607,'path':'summary-reference031.json','sha256':hashlib.sha256(oldbytes).hexdigest()},'new_case_records':len(records),'cases':cases,'copied_files':len(external),'raw_files':raw,'gzip_files':compressed,'original_payload_bytes':original_bytes,'stored_payload_bytes':stored_bytes,'summary_sha256':hashlib.sha256(data).hexdigest(),'six_axis_verdicts':verdicts,'source188_frozen_sha256':sha(S/'frozen-source.json'),'source188_host_build_sha256':sha(Q/'build-source188-002/source-build.json'),'selected_source_files':814,'changed_source_files':source_changes,'excluded_directory_parts':sorted(excluded),'exclusion_scope':'Closed cases only. Duplicate input snapshots and full build/restoration trees remain external. Ongoing source189 trial-contact repair, small-alpha consolidation and weak-mode CPU experiments are excluded. Exact generated dependency bytes removed locally have a remote verified restoration archive; research/source/accepted-state data are retained.','physical_validation':False}
put(D/'summary-reference031.json',oldbytes);put(D/'publication-binding-reference032.json',enc(binding))
assert (D/'summary.json').read_bytes()==oldbytes;(D/'summary.json').write_bytes(data)
put(Q/'publication032-staged-paths.nul',b''.join(p.encode()+b'\0' for p in new_paths))
report={'schema':'numi.cardboard.publication032-payload-preflight.v1','cases':len(records),'payloads':len(external),'raw':raw,'gzip':compressed,'original_payload_bytes':original_bytes,'stored_payload_bytes':stored_bytes,'summary_records':len(summary['records']),'old_records':1607,'summary_sha256':sha(D/'summary.json'),'binding_sha256':sha(D/'publication-binding-reference032.json'),'staged_pathlist_sha256':sha(Q/'publication032-staged-paths.nul'),'staged_paths':len(new_paths),'changed_source_files':source_changes,'active_results_excluded':True,'all_six_axes_open':True,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
put(O/'report.json',enc(report));put(O/'archive.py',Path(__file__).read_bytes())
print(json.dumps(report,indent=2),flush=True)
