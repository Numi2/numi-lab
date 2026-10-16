
from pathlib import Path
import datetime,gzip,hashlib,json,os,subprocess
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');R=Q/'checkout-publication032'
P='matter/examples/cardboard/evidence/native-implementation-20261014/';D=R/P;O=Q/'analysis/publication034-payload-preflight-002'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
enc=lambda x:(json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def git(*a):return subprocess.check_output(['git',*a],cwd=R)
def put(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b,p
 else:p.write_bytes(b)
assert not O.exists();O.mkdir()
parent='c610f03357c2e578a691defd63784b34d785a27b'
assert git('rev-parse','HEAD').decode().strip()==parent
oldbytes=(D/'summary.json').read_bytes();assert hashlib.sha256(oldbytes).hexdigest()=='bd20ddc8711092ab3e8086257b216c85419d852531d81882c591d09b9ea2d5d3'
old=json.loads(oldbytes);assert len(old['records'])==1814
prior={f.get('repository_copy',f.get('repository_compressed_copy')):f for r in old['records'] for f in r['files']}
selection=json.loads((Q/'publication034-case-selection.json').read_text());cases=selection['cases'];assert len(cases)==29 and len(cases)==len(set(cases))
assert not ({r['case'] for r in old['records']}&set(cases))
S=Q/'snapshots/source-contact-damped-memory-199-buildfix-002';sources=json.loads((S/'frozen-source.json').read_text())['source_sha256'];assert len(sources)==816
for n,h in sources.items():assert sha(R/n)==sha(S/n)==h,n
buildpath=Q/'analysis/source199-postcommit-fault-signature-002/source-build.json';build=json.loads(buildpath.read_text())
assert build['schema']=='numi.cardboard.source-build.v1' and build['source_frozen_file_sha256']==sha(S/'frozen-source.json')
assert build['cpu_tests']['failed']==0 and build['cpu_tests']['passed']==2
assert '100% tests passed' in Path(build['artifacts']['cpu_ctest_log']['path']).read_text()
assert 'Built target numi-matter-cardboard-full-contact-armijo-check' in Path(build['artifacts']['build_log']['path']).read_text()
for n,r in build['artifacts'].items():assert sha(r['path'])==r['sha256'] and Path(r['path']).stat().st_size==r['size_bytes'],n
records=[];paths=[];external=[];raw=compressed=original_bytes=stored_bytes=0;overlaps=[]
excluded={'__pycache__','archives','base','build','restore-check','restore-test','inputs','CMakeFiles'}
for case in cases:
 root=Q/case;assert root.is_dir();files=[]
 for directory,dirs,names in os.walk(root):
  dirs[:]=sorted(n for n in dirs if n not in excluded and not n.startswith('.air'))
  for n in sorted(names):
   p=Path(directory)/n;assert p.is_file() and not p.is_symlink()
   b=p.read_bytes();h=hashlib.sha256(b).hexdigest();target=Path(case)/p.relative_to(root)
   row={'name':str(p.relative_to(root)),'bytes':len(b),'sha256':h,'external_path':str(p)}
   if len(b)<=262144 or p.suffix=='.gz':
    payload=b;row['repository_copy']=str(target);israw=True
   else:
    payload=gzip.compress(b,compresslevel=6,mtime=0);assert gzip.decompress(payload)==b;target=target.with_name(target.name+'.gz')
    row.update(repository_compressed_copy=str(target),compressed_bytes=len(payload),compressed_sha256=hashlib.sha256(payload).hexdigest(),restore='gzip decompression; verify original SHA-256');israw=False
   if str(target) in prior:
    f=prior[str(target)];prior_data=(D/target).read_bytes()
    if 'repository_compressed_copy' in f:prior_data=gzip.decompress(prior_data)
    assert f['sha256']==h and f['bytes']==len(b) and prior_data==b
    overlaps.append({'case':case,'name':row['name'],'historical_payload':str(target),'sha256':h});continue
   put(D/target,payload);paths.append(P+str(target));external.append((str(p),h));files.append(row)
   raw+=int(israw);compressed+=int(not israw);original_bytes+=len(b);stored_bytes+=len(payload)
 assert files,root
 rec={'case':case,'external_root':str(root),'physical_box_qualification':False,'files':files}
 for n in ['prediction','receipt']:
  p=root/(n+'.json')
  if p.is_file():rec[n]=json.loads(p.read_text())
 records.append(rec)
for n,h in external:assert sha(n)==h,n
summary=json.loads(oldbytes);summary['records'].extend(records);assert summary['records'][:1814]==old['records']
summary.setdefault('historical_summary_snapshots',{})['reference033']={'path':'summary-reference033.json','sha256':hashlib.sha256(oldbytes).hexdigest()}
verdicts=dict(old['verdicts'])
verdicts['solver_correctness']={'status':'partial_component_evidence; complete_nonlinear_solve_unqualified','evidence':'Source199 strengthens the synthetic contact checker to require the exact prescribed-motion postCommit fault. Native528 passes and preserves26 raw files identical to526. Negative529 exits1 for the specific signature assertion because its scalar solve rejects earlier at Newton18/status10; all13 baseline outputs match527. Physical runtime/Metal bytes remain source198. Transaction rollback and fault specificity are bounded component evidence only.'}
verdicts['numerical_convergence']={'status':'unqualified','evidence':'Fresh native521 fails after17 accepted controls at zero-based17/Newton34: nonlinear residual1.63952734e-6 to1.63964694e-6, unchanged tolerance5e-7, smallest alpha1.19209290e-7. Its fresh linear residual is1.19334089e-7 with healthy status. All18 numeric-Newton3 damped records have zero candidates; no installed damping is evidenced at Newton34. Independent prefix audit finds34 accepted OBJ files and34 observation rows byte-identical to480. Exact failing correction and full accepted checkpoint were not captured. Active530 checkpoint experiment is excluded. No complete60-control coupon or mesh/time triplet is qualified.'}
summary['verdicts']=verdicts
for k in ['solver_correctness','numerical_convergence']:summary['qualification'][k]=verdicts[k]['evidence']
summary['milestone']='Source199 makes intended fault coverage strict and preserves its positive and negative native regressions. Terminal fresh native521 failure and its exact accepted-prefix parity are retained. Source198 physical operator remains unchanged. All six full-protocol qualification axes remain open; active530, helper-reinitialization and source201 work are excluded.'
data=enc(summary)
sourcechanges=sorted(n for n in sources if git('diff','HEAD','--',n) or not git('ls-files','--',n))
assert sourcechanges==['matter/tools/cardboard_full_contact_armijo_check.mm']
extras=['matter/examples/cardboard/solver-milestone-source199.md','matter/examples/cardboard/implementation-20261014.md']
newpaths=sorted(set(paths+[P+'summary-reference033.json',P+'publication-binding-reference034.json',P+'summary.json']+sourcechanges+extras))
binding={'schema':'numi.cardboard.reference034-publication-binding.v1','parent':parent,'prior_binding_sha256':sha(D/'publication-binding-reference033.json'),'historical_prefix':{'records':1814,'path':'summary-reference033.json','sha256':hashlib.sha256(oldbytes).hexdigest()},'new_case_records':len(records),'cases':cases,'case_selection_sha256':sha(Q/'publication034-case-selection.json'),'copied_files':len(external),'raw_files':raw,'gzip_files':compressed,'original_payload_bytes':original_bytes,'stored_payload_bytes':stored_bytes,'summary_sha256':hashlib.sha256(data).hexdigest(),'six_axis_verdicts':verdicts,'source199_frozen_sha256':sha(S/'frozen-source.json'),'source199_host_build_sha256':sha(buildpath),'selected_source_files':816,'changed_source_files':sourcechanges,'excluded_directory_parts':sorted(excluded),'verified_historical_overlaps':overlaps,'exclusion_scope':'Explicit closed-case list only; active530, reinitialization helper and source201 excluded. No research input or native output deleted. Failed builds and negative controls retained.','physical_validation':False}
put(D/'summary-reference033.json',oldbytes);put(D/'publication-binding-reference034.json',enc(binding))
assert (D/'summary.json').read_bytes()==oldbytes;(D/'summary.json').write_bytes(data)
put(Q/'publication034-staged-paths.nul',b''.join(n.encode()+b'\0' for n in newpaths))
report={'schema':'numi.cardboard.publication034-payload-preflight.v1','cases':len(records),'payloads':len(external),'raw':raw,'gzip':compressed,'original_payload_bytes':original_bytes,'stored_payload_bytes':stored_bytes,'summary_records':len(summary['records']),'old_records':1814,'summary_sha256':sha(D/'summary.json'),'binding_sha256':sha(D/'publication-binding-reference034.json'),'staged_pathlist_sha256':sha(Q/'publication034-staged-paths.nul'),'staged_paths':len(newpaths),'changed_source_files':sourcechanges,'active_results_excluded':True,'all_six_axes_open':True,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
put(O/'report.json',enc(report));put(O/'archive.py',Path(__file__).read_bytes());print(json.dumps(report,indent=2))
