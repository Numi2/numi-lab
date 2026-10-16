from pathlib import Path
import datetime,gzip,hashlib,json,os,subprocess
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');R=Q/'checkout-publication032'
P='matter/examples/cardboard/evidence/native-implementation-20261014/';D=R/P;O=Q/'analysis/publication033-payload-preflight-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
enc=lambda x:(json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def put(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b,p
 else:p.write_bytes(b)
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
assert not O.exists();O.mkdir()
assert git('rev-parse','HEAD').decode().strip()=='877cd6e69657c9a7b4e5032bcd475705bcc06c85'
oldbytes=(D/'summary.json').read_bytes();assert hashlib.sha256(oldbytes).hexdigest()=='5bb2a527a80528c8bbcd621f9fbb6bc6dad06be4a3e90888a70ccbbc98386ff7'
old=json.loads(oldbytes);assert len(old['records'])==1669;oldcases={x['case'] for x in old['records']}
S=Q/'snapshots/source-contact-damped-memory-198-001'
sources=json.loads((S/'frozen-source.json').read_text())['source_sha256'];assert len(sources)==816
for n,h in sources.items():assert sha(R/n)==sha(S/n)==h,n
# This list is fixed after CPU audits finish. The active source198 coupon and its mutable runner package are excluded.
selection=json.loads((Q/'publication033-case-selection.json').read_text())
pairs=[(n,Q/n) for n in selection['cases']]
assert len(pairs)==len({n for n,p in pairs}) and not (oldcases & {n for n,p in pairs})
excluded={'__pycache__','archives','base','build','restore-check','restore-test','inputs','CMakeFiles'}
records=[];paths=[];external=[];raw=compressed=original_bytes=stored_bytes=0
for case,root in pairs:
 assert root.is_dir(),root
 files=[]
 for directory,dirs,names in os.walk(root):
  dirs[:]=sorted(n for n in dirs if n not in excluded and not n.startswith('.air'))
  for name in sorted(names):
   p=Path(directory)/name
   assert p.is_file() and not p.is_symlink(),p
   b=p.read_bytes();h=hashlib.sha256(b).hexdigest();target=Path(case)/p.relative_to(root)
   row={'name':str(p.relative_to(root)),'bytes':len(b),'sha256':h,'external_path':str(p)}
   if len(b)<=262144 or p.suffix=='.gz':
    payload=b;row['repository_copy']=str(target);raw+=1
   else:
    payload=gzip.compress(b,compresslevel=6,mtime=0);assert gzip.decompress(payload)==b
    target=target.with_name(target.name+'.gz')
    row.update(repository_compressed_copy=str(target),compressed_bytes=len(payload),compressed_sha256=hashlib.sha256(payload).hexdigest(),restore='gzip decompression; verify original SHA-256');compressed+=1
   assert not (D/target).exists(),target
   put(D/target,payload);paths.append(P+str(target));external.append((str(p),h))
   files.append(row);original_bytes+=len(b);stored_bytes+=len(payload)
 assert files,root
 rec={'case':case,'external_root':str(root),'physical_box_qualification':False,'files':files}
 for n in ['prediction','receipt']:
  p=root/(n+'.json')
  if p.is_file():rec[n]=json.loads(p.read_text())
 records.append(rec)
for n,h in external:assert sha(n)==h,n
verdicts=dict(old['verdicts'])
verdicts['solver_correctness']={'status':'partial_component_evidence; complete_nonlinear_solve_unqualified','evidence':'Source198 makes private FEM and contact trial geometry share the exact tested update, preserves CCD and final certificates, and uses the complete residual merit for pure-FEM mixed contact. The regional combined-contact fixture526 accepts two controls and reaches the intended postCommit fault after passing the unchanged1e-7 residual gate; raw history rollback and shared/separate scratch parity are independently checked. Scalar525/527 accept two controls but fail residual descent before that injected fault. These fixtures do not establish global nonlinear or universal tangent correctness. The experimental full-from-zero damped CGLS direction passes its native production-kernel recurrence check in four positive-damping cases; the undamped controls expose denominator underflow and remain limitations.'}
verdicts['numerical_convergence']={'status':'unqualified','evidence':'The actual native520 contact capture shows short-budget FGMRES stagnation, with independently formed b-Jdelta agreeing with its saved workspace; no action discrepancy is demonstrated by that capture. Strict CPU native497 replay localizes the total correction and gives a positive-damping descent candidate, but it is not an accepted native trajectory. No complete60-control coupon, independent mesh series or timestep triplet passes. The new fresh native521 experiment is deliberately excluded while active. No tolerance or checkpoint-identity relaxation is used.'}
verdicts['physical_assembly_in_simulation']={'status':'unqualified','evidence':'The explicit-flute shipping topology remains528444nodes,1477728tetrahedra and260835cohesive faces. Source198 removes double counting of the single private arena and charges the optional sorted-incidence workspace, but the bounded node/tet/contact arenas still reject the shipping world and no measured whole-process16GiB peak is qualified. No physically scored, folded, bonded, fixture-free300x200x200mm checkpoint exists.'}
summary=json.loads(oldbytes)
summary.setdefault('historical_summary_snapshots',{})['reference032']={'path':'summary-reference032.json','sha256':hashlib.sha256(oldbytes).hexdigest()}
summary['records'].extend(records);assert summary['records'][:1669]==old['records']
summary['verdicts']=verdicts
for k in ['solver_correctness','numerical_convergence','physical_assembly_in_simulation']:summary['qualification'][k]=verdicts[k]['evidence']
summary['milestone']='Source198 repairs exact mixed-contact trial evaluation, retains strict global residual acceptance and transactional rollback, and adds an opt-in damped direction without changing the physical operator. Native regional fixture526 reaches the intended late fault; scalar525/527 mask it with earlier residual rejection. Source/build, failed wrappers, retracted review claims, helper reporting corrections and native failures are retained. Full coupon convergence, calibrated damage/debonding, shipping-box assembly and crush/drop predictions remain open.'
data=enc(summary)
sourcechanges=sorted(n for n in sources if git('diff','HEAD','--',n) or not git('ls-files','--',n))
assert len(sourcechanges)==12,sourcechanges
doc='matter/examples/cardboard/solver-milestone-source198.md'
assert (R/doc).is_file()
extras=[doc,'matter/examples/cardboard/implementation-20261014.md']
newpaths=sorted(set(paths+[P+'summary-reference032.json',P+'publication-binding-reference033.json',P+'summary.json']+sourcechanges+extras))
binding={'schema':'numi.cardboard.reference033-publication-binding.v1','parent':'877cd6e69657c9a7b4e5032bcd475705bcc06c85','prior_binding_sha256':sha(D/'publication-binding-reference032.json'),'historical_prefix':{'records':1669,'path':'summary-reference032.json','sha256':hashlib.sha256(oldbytes).hexdigest()},'new_case_records':len(records),'cases':selection['cases'],'case_selection_sha256':sha(Q/'publication033-case-selection.json'),'copied_files':len(external),'raw_files':raw,'gzip_files':compressed,'original_payload_bytes':original_bytes,'stored_payload_bytes':stored_bytes,'summary_sha256':hashlib.sha256(data).hexdigest(),'six_axis_verdicts':verdicts,'source198_frozen_sha256':sha(S/'frozen-source.json'),'source198_host_build_sha256':sha(Q/'build-source198-001/source-build.json'),'selected_source_files':816,'changed_source_files':sourcechanges,'excluded_directory_parts':sorted(excluded),'exclusion_scope':'Closed cases only. Active native521 and its mutable command002 package are excluded. Empty placeholders, duplicate source/input and full build/restoration trees remain external; exact selected source maps, build logs and executable/Metal bytes are preserved in the host artifact bundle. Case selection records every excluded candidate. No research or accepted-state output was deleted.','physical_validation':False}
put(D/'summary-reference032.json',oldbytes);put(D/'publication-binding-reference033.json',enc(binding))
assert (D/'summary.json').read_bytes()==oldbytes;(D/'summary.json').write_bytes(data)
put(Q/'publication033-staged-paths.nul',b''.join(n.encode()+b'\0' for n in newpaths))
report={'schema':'numi.cardboard.publication033-payload-preflight.v1','cases':len(records),'payloads':len(external),'raw':raw,'gzip':compressed,'original_payload_bytes':original_bytes,'stored_payload_bytes':stored_bytes,'summary_records':len(summary['records']),'old_records':1669,'summary_sha256':sha(D/'summary.json'),'binding_sha256':sha(D/'publication-binding-reference033.json'),'staged_pathlist_sha256':sha(Q/'publication033-staged-paths.nul'),'staged_paths':len(newpaths),'changed_source_files':sourcechanges,'active_results_excluded':True,'all_six_axes_open':True,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
put(O/'report.json',enc(report));put(O/'archive.py',Path(__file__).read_bytes())
print(json.dumps(report,indent=2),flush=True)
