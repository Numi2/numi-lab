from pathlib import Path
import gzip,hashlib,json,os,subprocess,tempfile
R=Path.cwd();clone=Path('/tmp/crow-restore-check-20261110');branch='agent/american-crow-rl-20261107'
commit=subprocess.check_output(['git','-C',str(clone),'rev-parse','FETCH_HEAD^{commit}'],text=True).strip()
remote=subprocess.check_output(['git','-C',str(clone),'ls-remote','origin','refs/heads/'+branch],text=True).split()[0]
assert commit==remote,(commit,remote)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def streamsha(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def show(path):return subprocess.check_output(['git','-C',str(clone),'show',commit+':'+path])
pairs=[]
def add(artifact,sub,build,prefix,kinds):
 for kind in kinds:pairs.append((artifact,sub+'/'+kind+'.gz',build+'/'+prefix+'-'+kind))
add('crow-v15-qualified-route-imitation-eval-20261111','autonomous-20261126','crow-v15-qualified-route-imitation-eval-20261111','autonomous-20261126',['replay.json','trace.tsv'])
for mode in ('reference','fold'):add('crow-v15-flight-fold-self-contact-observer-20261111',mode,'crow-v15-flight-fold-self-contact-observer-20261111',mode,['replay.json','trace.tsv'])
add('crow-v15-flight-moderate-fold-control-20261111','moderate','crow-v15-flight-moderate-fold-control-20261111','moderate',['replay.json','trace.tsv'])
add('crow-v15-student-state-teacher-recovery-20261111','recovery-20261126','crow-v15-student-state-teacher-recovery-20261111','recovery-20261126',['replay.json','trace.tsv','rollout.bin'])
rows=[]
with tempfile.TemporaryDirectory(prefix='crow-independent-restore-') as td:
 for i,(artifact,relative,rawrel) in enumerate(pairs):
  remote_base='artifacts/'+artifact+'/'
  manifest=show(remote_base+'SHA256SUMS').decode();matches=[line.split('  ',1)[0] for line in manifest.splitlines() if line.split('  ',1)[1]==relative];assert len(matches)==1
  local=R/'build'/rawrel;assert local.is_file(),str(local)
  restored=Path(td)/str(i)
  with restored.open('wb') as f:subprocess.run(['git','-C',str(clone),'show',commit+':'+remote_base+relative],stdout=f,check=True)
  compressed=sha(restored);assert compressed==matches[0]
  with gzip.open(restored,'rb') as f:expanded=streamsha(f)
  rawsha=sha(local);assert rawsha==expanded
  check=subprocess.run(['lsof','-t','--',str(local)],capture_output=True,text=True);assert check.returncode==1 and not check.stdout.strip(),(str(local),check.stdout,check.stderr)
  rows.append({'raw_path':str(local),'bytes':local.stat().st_size,'raw_sha256':rawsha,'remote_object':remote_base+relative,'remote_compressed_sha256':compressed})
 # Recheck each path has no open handles immediately before retiring it.
 for row in rows:
  local=Path(row['raw_path']);check=subprocess.run(['lsof','-t','--',str(local)],capture_output=True,text=True);assert check.returncode==1 and not check.stdout.strip()
  assert sha(local)==row['raw_sha256'];local.unlink()
out={'verified_remote_commit':commit,'remote_branch':branch,'independent_clone':str(clone),'files_retired':rows,'bytes_retired':sum(r['bytes'] for r in rows),'method':'Independently fetched remote Git object; remote manifest hash; gzip expansion matches raw SHA-256; lsof clear; immediate rehash before removal. Tracked compressed evidence remains.'}
(Path('artifacts/crow-v15-evidence-retirement-round2-20261111')/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'files':len(rows),'bytes':out['bytes_retired'],'commit':commit},indent=2))
