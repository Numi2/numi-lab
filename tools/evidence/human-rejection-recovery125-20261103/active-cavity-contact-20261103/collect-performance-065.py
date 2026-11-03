from pathlib import Path
import json,hashlib,re,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
def digest(f):
 with f.open('rb') as r:return hashlib.file_digest(r,'sha256').hexdigest()
for name in sys.argv[1:]:
 p=D/name
 ex=json.loads((p/'install-motion.execution.json').read_text())
 h=hashlib.sha256();n=0;capture=False;profiles=[];finish=[]
 compact=p/'install-motion.stdout.compact.log'
 with (p/'install-motion.stdout.log').open('rb') as r,compact.open('wb') as out:
  for line in r:
   if line.startswith(b'resting_FEM_endpoint_capture=begin '):capture=True
   if capture:h.update(line);n+=len(line)
   if line.startswith(b'resting_FEM_endpoint_capture=end'):capture=False
   if b'resting_native_profile' in line:profiles.append(line.decode().strip())
   if b'resting_integrated_body=completed' in line:finish.append(line.decode().strip())
   if not line.startswith((b'resting_FEM_node,',b'resting_FEM_tet,',b'resting_FEM_endpoint_node',b'resting_FEM_endpoint_tet')):out.write(line)
 stages={};kernels={}
 with (p/'install-motion.stderr.log').open() as r:
  for line in r:
   m=re.search(r'matter_gpu_stage=(\S+) step=0 elapsed_ns=(\d+)',line)
   if m:stages.setdefault(m[1],[]).append(int(m[2])/1e6)
   m=re.search(r'matter_gpu_kernel=(\S+).*?elapsed_ns=(\d+)',line)
   if m:kernels.setdefault(m[1],[]).append(int(m[2])/1e6)
 result={'name':name,'execution':ex,'endpoint_bytes':n,'endpoint_sha256':h.hexdigest() if n else None,'endpoint_available':n>0,
  'profiles':profiles,'completion':finish,'stages_root0_ms':stages,'selected_kernel_ms':kernels}
 for f in p.rglob('*.csv'):
  if 'trace' in f.name or 'phys' in f.name or f.name=='resting-coupled.csv':result.setdefault('traces',[]).append({'path':str(f),'sha256':digest(f),'bytes':f.stat().st_size})
 (p/'compact-result.json').write_text(json.dumps(result,indent=2)+'\n')
 (p/'retained-full-stdout.json').write_text(json.dumps({'path':str(p/'install-motion.stdout.log'),'sha256':digest(p/'install-motion.stdout.log'),'bytes':(p/'install-motion.stdout.log').stat().st_size,'endpoint_bytes':n,'endpoint_sha256':h.hexdigest() if n else None,'endpoint_available':n>0,'compact_omits_only_endpoint_node_tet_rows':True},indent=2)+'\n')
 print(name,ex['exit_code'],'endpoint',n,h.hexdigest())
 print('\n'.join(profiles));print(finish)
 print({k:sum(v) for k,v in stages.items() if k in ['matter_linear','matter_final_certificate','matter_after_candidate_ready']})
