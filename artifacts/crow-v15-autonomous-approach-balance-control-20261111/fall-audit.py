from pathlib import Path
import gzip,json,math
paths={'autonomous':'artifacts/crow-v15-autonomous-approach-balance-control-20261111/autonomous-20261126','teacher_reference':'artifacts/crow-v15-teacher-perch-balanced-approach-control-20261111/teacher-20261126'};out={}
for mode,path in paths.items():
 perch=json.loads((Path(path)/'perch-analysis.json').read_text())['rows'];wanted={0,5,6,7};samples={e:[] for e in wanted}
 with gzip.open(Path(path)/'run.stderr.gz','rt') as f:
  for line in f:
   if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
   x=json.loads(line.split(' ',1)[1]);e=x['lane'];t=x['step'];a=perch[e]['first_perch_foot_impulse_step'];end=perch[e]['first_episode_steps']
   if e in wanted and a and a<=t<=min(end,a+500) and ((t-a)%50==0 or t==end):
    q=x['q'];v=x['v'];qx,qy,qz,qw=q[3:7];pitch=math.asin(max(-1.,min(1.,2*(qw*qy-qz*qx))));samples[e].append({'step':t,'x_m':q[0],'z_m':q[2],'pitch_rad':pitch,'vx_m_s':v[0],'vz_m_s':v[2],'pitch_rate_rad_s':v[4]})
 out[mode]=samples
out['scope']='Endpoint samples from first episodes on the same development course and seed; the teacher reference is privileged. Falling is not established by pitch alone; first-episode termination is separately recorded.'
p=Path(paths['autonomous'])/'fall-audit.json';p.write_text(json.dumps(out,indent=2)+'\n');print('saved',p)
