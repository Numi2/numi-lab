from pathlib import Path
import collections,gzip,hashlib,json,sys
R=Path('artifacts');A=R/'crow-v15-support-gap-audit-20261112';P=R/'crow-v15-full-foot-feedback-20261112';seed=int(sys.argv[1]);p=P/f'candidate-{seed}-rebound-phase/run.stderr.gz';data={};qs={};vs={}
with gzip.open(p,'rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0]=='CROW_TWO_TOE_SUBSTEP':
   t,k,e=map(int,z[1:4]);
   if 201<=t<=300:data[e,(t-201)*16+k]=list(map(float,z[6:11]))
  elif z[0] in ('CROW_Q_SUBSTEP','CROW_V_SUBSTEP'):
   t,k,e=map(int,z[1:4]);
   if 201<=t<=300:(qs if z[0]=='CROW_Q_SUBSTEP' else vs)[e,(t-201)*16+k]=list(map(float,z[4:]))
assert len(data)==len(qs)==len(vs)==12800
rows=[]
for e in range(4,8):
 gaps=[i for i in range(1600) if data[e,i][2]<=1e-7 and data[e,i][3]<=1e-7];groups=[]
 for i in gaps:
  if not groups or i>groups[-1][-1]+1:groups.append([i])
  else:groups[-1].append(i)
 for group in groups:
  i=group[0];last=group[-1]
  def sample(n):
   if not 0<=n<1600:return None
   c=data[e,n];q=qs[e,n];v=vs[e,n]
   return {'sample':n,'foot_normal_impulses_ns':c[2:4],'nonfoot_ground_impulse_ns':c[4],'root_height_m':q[2],'root_vertical_speed_m_s':v[2]}
  rows.append({'lane':e,'start_substep':i,'count':len(group),'duration_ms':len(group)*1.25,'before':sample(i-1),'first':sample(i),'last':sample(last),'after':sample(last+1)})
sensitivity={str(threshold):{str(e):sum(data[e,i][2]<=threshold and data[e,i][3]<=threshold for i in range(1600)) for e in range(4,8)} for threshold in (1e-7,1e-9,1e-12,1e-14)}
h=hashlib.sha256(p.read_bytes()).hexdigest();out={"threshold_sensitivity":sensitivity,'seed':seed,'input_sha256':h,'rows':rows,'scope':'Ground contact normal impulse summed for modeled foot and toe collision bodies, recorded on accepted 1.25 ms substeps. Neighboring impulses and root speed are descriptive; they cannot by themselves establish flight, balance, a fall, or support on unobserved surfaces. Threshold is 1e-7 Ns per foot and individual constraint sums are used by this logger.'}
(A/f'gap-{seed}.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(seed,r['lane'],'sample',r['start_substep'],'n',r['count'],'impulses',r['first']['foot_normal_impulses_ns'],'vZ',r['first']['root_vertical_speed_m_s'],'adjacent',None if r['before'] is None else r['before']['foot_normal_impulses_ns'],None if r['after'] is None else r['after']['foot_normal_impulses_ns'])
