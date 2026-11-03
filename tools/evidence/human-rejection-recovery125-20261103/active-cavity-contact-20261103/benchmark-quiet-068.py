from pathlib import Path
import subprocess,json,hashlib,re,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
src=Path('/tmp/run-performance-065.py').read_text()
src=src.replace("for f in ['CMakeLists.txt','matter/CMakeLists.txt']:","for f in ['CMakeLists.txt','matter/CMakeLists.txt','matter/tools/vascular_cavity_check.mm','matter/tools/fgmres_true_residual_native_check.mm']:")
src=src.replace("for a in [str(D/'performance-cleanup-064/build.log'),str(D/'performance-cleanup-064/native-check.stdout.log')]:","for a in [str(D/'completed-incidence-067/build-final.log'),str(D/'completed-incidence-067/solver-final.stdout.log'),str(D/'completed-incidence-067/gate-final.stdout.log'),str(D/'completed-incidence-067/cavity-final.stdout.log')]:")
src=src.replace("p['build_manifest']=pin(d/'build-manifest.json')","from quiet_launcher_068 import configure\nconfigure(d,p,pin)\np['build_manifest']=pin(d/'build-manifest.json')")
src=src.replace("if len(sys.argv)>4 and 'NM_FGMRES_INEXACT_NEWTON_MAX_ETA'", """if p['stages'][0]['environment'].get('NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE')=='1':
 p['predeclared_checks']['endpoint']='Bitwise full endpoint parity with published completed-linear-builders-063'
if len(sys.argv)>4 and 'NM_FGMRES_INEXACT_NEWTON_MAX_ETA'""")
Path('/tmp/run-performance-068.py').write_text(src)
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
jobs=[('final-combined-capture-068-retry1',1,1,1),
      ('quiet-control-069',0,0,0),('quiet-combined-070',1,1,0),
      ('quiet-combined-071',1,1,0),('quiet-control-072',0,0,0),
      ('quiet-control-073',0,0,0),('quiet-combined-074',1,1,0)]
summary=[]
for name,cavity,incidence,capture in jobs:
 overrides={'NM_VASCULAR_COOPERATIVE_CAVITY_OPERATOR':str(cavity),
            'NM_CONTACT_COMPLETED_INCIDENCE_REUSE':str(incidence),
            'NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE':str(capture)}
 print('START',name,flush=True)
 subprocess.run([sys.executable,'/tmp/run-performance-068.py',name,'1','run',json.dumps(overrides)],check=True)
 subprocess.run([sys.executable,'/tmp/collect-performance-065.py',name],check=True)
 p=D/name;r=json.loads((p/'compact-result.json').read_text())
 assert r['execution']['exit_code']==0
 assert len(r['profiles'])==4
 if capture:assert r['endpoint_sha256']=='66c476e14b867eb9a19039ec112750a3d517a18ce039b24bdb5181fec7ac24c7'
 trace=next((p/'install-motion').glob('resting-coupled.csv'))
 assert sha(trace)=='77edb932a1a06eacdba8efe1f97bc5492ff8185c37a2e31a5c8d09045eb41aee'
 work=[]
 for line in (p/'install-motion.stderr.log').read_text().splitlines():
  if line.startswith('matter_fgmres_active_work='):
   row=json.loads(line.split('=',1)[1]);work.append(row)
 assert len(work)==4 and all(row['used_columns_sum']==32 and row['active_newton_begins']==2 and row['status_code']==0 for row in work)
 # Verify raw accepted coordinate artifacts, independently of logging policy.
 wanted={'step-4.common-coordinates.bin':'fb9a69a2a2378811a4f18a4112ee2790738a9d5b8c64031a754c88bbcb67aa20',
         'step-4.cardiac-q.bin':'ef99cfd192ee2fe43a68cef2af40c85c2c215759f491c1b3fa09ed0f794f9201',
         'step-4.cardiac-wall-q.bin':'374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb'}
 artifacts={}
 for file,digest in wanted.items():
  hits=list(p.rglob(file));assert len(hits)==1,(file,hits)
  actual=sha(hits[0]);assert actual==digest,(file,actual);artifacts[file]=actual
 # Only completed, hash-equal generated outputs may share disk extents.
 code=Path('/tmp/collect-shared-049.py').read_text().split('donors={}',1)[1]
 scope={'D':D,'p':p,'Path':Path,'json':json,'hashlib':hashlib,'os':__import__('os')}
 exec('donors={}'+code.replace('.dedup-049','.dedup-'+name).replace('-20261103-049.json','-20261103-'+name+'.json'),scope)
 verification={'accepted_artifacts':artifacts,'physiology_sha256':sha(trace),'active_work':work,'comparison':'bitwise','full_endpoint':bool(capture)}
 (p/'accepted-parity.json').write_text(json.dumps(verification,indent=2)+'\n')
 gpu=[float(re.search(r' gpu_ms=([\d.eE+-]+)',line).group(1)) for line in r['profiles']]
 loop=float(re.search(r' wall_s=([\d.eE+-]+)',r['completion'][0]).group(1))
 summary.append({'name':name,'cavity':cavity,'incidence':incidence,'capture':capture,'mean_gpu_ms':sum(gpu)/4,'loop_wall_s':loop,'process_elapsed_s':r['execution']['elapsed_s']})
 (D/'paired-performance-068-074.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('PASS',name,'mean_gpu_ms',sum(gpu)/4,'loop_s',loop,flush=True)
print('ALL_MATCHED_RUNS_COMPLETE',flush=True)
