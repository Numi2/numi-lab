from pathlib import Path
import importlib.util,json,os,stat,subprocess
r=Path.cwd();base=r/'artifacts/crow-rl-deployment-mean-20261108';tmp=Path('/private/tmp/crow-deployment-mean-git-recovery-20261108')
spec=importlib.util.spec_from_file_location('archive',r/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
m=json.loads((base/'runtime-recovery.json').read_text());plan={'budget':'Two complete12-arm native evaluations,245760 transitions each,600seconds each; two native identity-oracle runs, no learning or retries.', 'prediction':'All behavior/GRU intervention files reproduce byte-exact from Git-restored runtime/policies/product inputs. Path-bound visual identities are recomputed by native owning functions at original/restored locations, every other JSON field exact. Canonical project candidate equals the restored mean pack.','scope':'Fixed two-cue fixture,3 allocation/action seeds, matching OS/device. No full lifelong or broader geometric qualification.'}
(base/'verification-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith('DYLD_')};env['DYLD_PRINT_LIBRARIES']='1'
def run(name,c,timeout):
 (base/(name+'.command.json')).write_text(json.dumps(c,indent=2)+'\n')
 p=subprocess.run(c,capture_output=True,text=True,env=env,timeout=timeout)
 (base/(name+'.stdout')).write_text(p.stdout);(base/(name+'.stderr')).write_text(p.stderr);(base/(name+'.receipt.json')).write_text(json.dumps({'returncode':p.returncode},indent=2)+'\n')
 if p.returncode:raise RuntimeError('Recovered native execution failed: '+name)
 loaded=[line for line in p.stderr.splitlines() if line.startswith('dyld[') and 'libmetalrobo.dylib' in line]
 if not loaded or any(not line.endswith(str(tmp/'lib/libmetalrobo.dylib')) for line in loaded):raise RuntimeError('Wrong recovery library: '+name)
 return {'stdout':p.stdout,'loaded_library':loaded}
try:
 compiler=['clang++','-std=c++20','-fobjc-arc','-I'+str(tmp/'source/include'),'-I'+str(tmp/'source/apps'),str(tmp/'source/identity_oracle.mm'),'-L'+str(tmp/'lib'),'-lmetalrobo','-framework','Foundation','-Wl,-rpath,@loader_path/../lib','-o',str(tmp/'bin/identity_oracle')]
 built=subprocess.run(compiler,capture_output=True,text=True,timeout=120)
 (base/'oracle-build.json').write_text(json.dumps({'command':compiler,'returncode':built.returncode,'stdout':built.stdout,'stderr':built.stderr},indent=2)+'\n')
 if built.returncode:raise RuntimeError('Native fingerprint oracle compilation failed')
 if a.file_hash(tmp/'project/environments/american-crow/action-holds-v1/policies/deployment97.policypack')!=a.file_hash(tmp/'input/mean.policypack'):raise RuntimeError('Canonical project deployment differs')
 entries=a.tree();manifest=json.loads((base/'manifest.json').read_text());results={}
 with a.Objects() as objects:
  for mode,record in m['evaluations'].items():
   dest=tmp/(mode+'-replay')
   execution=run(mode+'-recovery',[str(tmp/'bin/metalrobo_crow_memory_causal_check'),str(dest),str(tmp/record['policy']),str(tmp/'project/environments/american-crow/cued-foraging-v1/calibrated-ports.bin'),str(m['seed_base'])],600)
   oracle=run(mode+'-identity',[str(tmp/'bin/identity_oracle'),record['original_root'],str(dest),str(tmp/(mode+'-oracle-assets')),str(m['seed_base'])],120)
   identities=json.loads(oracle['stdout']);exact=0;relocations=[]
   for name,ref in record['files'].items():
    if name.endswith('/visual-scene.json'):
     f=next(f for f in manifest['files'] if f['sha256']==ref['sha256'] and f['bytes']==ref['bytes'])
     original=json.loads(b''.join(a.content(f,entries,objects)));restored=json.loads((dest/name).read_text());pair=identities[name.split('/')[0]]
     for value,location in ((original,'original'),(restored,'restored')):
      if value['fingerprint']!=pair[location]['manifest'] or value['render_scene']['fingerprint']!=pair[location]['render_scene']:raise RuntimeError('Native owner failed to explain relocated visual identity: '+mode+'/'+name)
     del original['fingerprint'];del original['render_scene']['fingerprint'];del restored['fingerprint'];del restored['render_scene']['fingerprint']
     if original!=restored:raise RuntimeError('Other visual manifest fields differ: '+mode+'/'+name)
     relocations.append({'path':name,'original_sha256':ref['sha256'],'restored_sha256':a.file_hash(dest/name),'native_owner_identities':pair,'all_other_JSON_fields_exact':True})
    else:
     if a.file_hash(dest/name)!=ref['sha256']:raise RuntimeError('Native behavior/intervention evidence differs: '+mode+'/'+name)
     exact+=1
   actual={str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()}
   if actual!=set(record['files']):raise RuntimeError('Native recovered file set differs')
   results[mode]={'execution':execution,'identity_oracle':oracle,'byte_exact_reference_files':exact,'relocated_manifests_verified':relocations,'native_summary':json.loads((dest/'summary.json').read_text())}
 files=[];dirs=[]
 for directory,subdirs,names in os.walk(tmp,followlinks=False):
  for name in subdirs:
   p=Path(directory)/name
   if p.is_symlink():raise RuntimeError('Unexpected recovery symlink')
   dirs.append(p)
  for name in names:
   p=Path(directory)/name;s=p.lstat()
   if not stat.S_ISREG(s.st_mode):raise RuntimeError('Unexpected recovery file type')
   files.append({'path':str(p),'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'sha256':a.file_hash(p)})
 handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(tmp)],capture_output=True,text=True,timeout=60)
 if handles.returncode not in (0,1) or handles.stdout or handles.stderr:raise RuntimeError('Open recovery handles')
 for rec in files:
  p=Path(rec['path']);s=p.lstat()
  if s.st_size!=rec['bytes'] or stat.S_IMODE(s.st_mode)!=rec['mode'] or s.st_mtime_ns!=rec['mtime_ns'] or a.file_hash(p)!=rec['sha256']:raise RuntimeError('Recovery file changed before cleanup')
 report={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'plan':plan,'results':results,'canonical_project_candidate_byte_exact':True,'restoration':json.loads((tmp/'runtime-restoration.json').read_text()),'temporary_cleanup':{'status':'validated','files':files,'open_handles':0,'df_before':subprocess.check_output(['df','-k','.'],text=True)}}
 (base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 for rec in files:Path(rec['path']).unlink()
 for p in sorted(dirs,key=lambda p:len(p.parts),reverse=True):p.rmdir()
 tmp.rmdir();report['temporary_cleanup'].update(status='removed',df_after=subprocess.check_output(['df','-k','.'],text=True));(base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':'passed','byte_exact_reference_files':{mode:v['byte_exact_reference_files'] for mode,v in results.items()},'native_verified_relocated_manifests':24,'temporary_files_removed':len(files),'temporary_bytes_removed':sum(rec['bytes'] for rec in files)}))
except Exception as error:
 (base/'verification-failure.json').write_text(json.dumps({'error':str(error),'type':type(error).__name__,'plan':plan},indent=2)+'\n');raise
