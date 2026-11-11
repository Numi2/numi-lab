from pathlib import Path
import gzip,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-two-segment-frozen-flight-failure-timing-20261111';B=R/'build/crow-v15-two-segment-frozen-flight-failure-timing-20261111';L=R/'build/crow-v15-two-segment-frozen-flight-contact-map-20261111';base=json.loads((R/'artifacts/crow-v15-two-segment-frozen-flight-early-screen-20261111/command.json').read_text());rows=[];traces={}
def trial(n):
 cmd=list(base);cmd[cmd.index('--steps')+1]=str(n);trace=B/(str(n)+'.tsv');cmd[cmd.index('--state-trace')+1]=str(trace);p=subprocess.run(cmd,cwd=R,env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=180);(A/(str(n)+'.stdout')).write_text(p.stdout)
 with (A/(str(n)+'.stderr.gz')).open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 row={'steps':n,'return_code':p.returncode,'stderr_tail':p.stderr[-2000:]}
 if p.returncode==0:
  x=json.loads(p.stdout);row.update({'failed_environment_steps':x['failed_environment_steps'],'physical_failure_count':x['first_episode_navigation']['physical_failure_count']});lines=trace.read_text().splitlines();traces[n]=lines[1:]
  with (A/(str(n)+'.trace.tsv.gz')).open('wb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(trace.read_bytes())
 rows.append(row);print('horizon',n,'exit',p.returncode,p.stderr[-400:],flush=True);return p.returncode==0
low=2;high=None
for n in (4,8,16,32,64,128,200):
 if trial(n):low=n
 else:high=n;break
while high is not None and high-low>1:
 mid=(low+high)//2
 if trial(mid):low=mid
 else:high=mid
ordered=sorted(traces);comparisons=[]
for i,n in enumerate(ordered):
 for m in ordered[i+1:]:comparisons.append({'short':n,'long':m,'common_prefix_exact':traces[n]==traces[m][:len(traces[n])]})
(A/'analysis.json').write_text(json.dumps({'last_success_horizon':low,'first_failure_horizon':high,'cases':rows,'successful_prefix_comparisons':comparisons,'scope':'development failure timing; no flight qualification'},indent=2)+'\n')
