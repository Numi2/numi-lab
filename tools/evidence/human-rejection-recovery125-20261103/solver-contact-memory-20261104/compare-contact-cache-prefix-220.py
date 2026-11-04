from pathlib import Path
import hashlib,json,re,sys,csv,gzip
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def profile(p):
 f=p/'install-motion.stdout.log';op=open
 if not f.exists():f=f.with_suffix(f.suffix+'.gz');op=gzip.open
 rows=[];summary=[]
 with op(f,'rt') as src:
  for line in src:
   if line.startswith('resting_native_profile '):
    values=dict(re.findall(r'(\w+)=([^\s]+)',line))
    rows.append({k:float(values[k]) for k in ['step','command_ms','gpu_ms','host_copy_ms']})
   if line.startswith('resting_integrated_body='):summary.append(line.strip())
 return {'rows':rows,'completed_summary':summary,'sum_gpu_ms':sum(x['gpu_ms'] for x in rows),'sum_command_ms':sum(x['command_ms'] for x in rows),'roots1_to31_gpu_ms':sum(x['gpu_ms'] for x in rows if x['step']>1)}
a,b=D/sys.argv[1],D/sys.argv[2]
r={'baseline':str(a),'candidate':str(b),'baseline_profile':profile(a),'candidate_profile':profile(b)}
for key,pat,count in [('CSV','*.csv',11),('MRV','*.mrvpack',3)]:
 files=sorted((a/'install-motion').rglob(pat));assert len(files)==count
 r[key]=[]
 for x in files:
  rel=x.relative_to(a/'install-motion');y=b/'install-motion'/rel
  hx,hy=sha(x),sha(y);r[key].append({'path':str(rel),'equal':hx==hy,'source_sha256':hx,'candidate_sha256':hy})
r['all_11_traces_and_3_captures_equal']=all(x['equal'] for k in ['CSV','MRV'] for x in r[k])
assert len(r['candidate_profile']['rows'])==32 and r['candidate_profile']['rows'][-1]['step']==32
r['simulated_seconds']=32*.000015625
r['comparison_pair_count']=1
r['gpu_only_prefix_real_time_factor']=r['simulated_seconds']/(r['candidate_profile']['sum_gpu_ms']/1000)
r['single_pair_uninstrumented_gpu_throughput_ratio']=r['baseline_profile']['roots1_to31_gpu_ms']/r['candidate_profile']['roots1_to31_gpu_ms']
r['qualification']='One matched release-transient prefix,0.5ms simulated. Root0 instrumented. Roots1..31 have no selected-kernel timing; whole-command GPU timings remain. Not settled supported rest, repeated breathing/heartbeat motion, sustained rate,15wall-fps presentation or0.1x. Three matched measurement pairs remain unfulfilled.'
(D/('comparison-'+sys.argv[1]+'-'+sys.argv[2]+'.json')).write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ['CSV','MRV','baseline_profile','candidate_profile']},indent=2))
print('baseline GPU s',r['baseline_profile']['sum_gpu_ms']/1000,'candidate GPU s',r['candidate_profile']['sum_gpu_ms']/1000)
print('baseline completion',r['baseline_profile']['completed_summary'])
print('candidate completion',r['candidate_profile']['completed_summary'])
sys.exit(0 if r['all_11_traces_and_3_captures_equal'] else 2)
