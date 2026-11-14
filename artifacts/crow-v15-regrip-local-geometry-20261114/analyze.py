from pathlib import Path
import json,collections
A=Path('artifacts/crow-v15-regrip-local-geometry-20261114');d=collections.defaultdict(dict)
for line in (A/'output.tsv').read_text().splitlines():
 e,t,j,x,*gap=line.split();d[int(e),int(t),int(j)][float(x)]=list(map(float,gap))
ref=json.loads(Path('artifacts/crow-v15-contact-gated-regrip-20261114/geometry-analysis.json').read_text());samples={(v['lane'],a['zero_based_tick']):[a['groups'][n]['min_box_gap_m'] for n in ('foot','front','rear')] for v in ref.values() for a in v['samples']}
rows=[];errors=[]
for e in (0,1):
 for t in (250,270,290):
  base=d[e,t,14][0.];err=max(abs(a-b) for a,b in zip(base,samples[e,t]));errors.append(err);ders={str(j):[(a-b)/.04 for a,b in zip(d[e,t,j][.02],d[e,t,j][-.02])] for j in (14,15,16,23,24,25,26,27,28,29,30)};rows.append({'lane':e,'tick':t,'base_gap_m':base,'derivatives_m_per_rad':ders,'crosscheck_error_m':err})
limits={14:(-.85,.85),15:(-.1,1.55),16:(-.9,.7),**{j:(-.3,1.2) for j in range(23,31)}};viol=[]
for line in (A/'input.tsv').read_text().splitlines():
 x=line.split();e,t=map(int,x[:2]);q=list(map(float,x[2:43]));assert len(q)==41
 for j,(lo,hi) in limits.items():
  for dx in (-.02,.02):
   if not lo-1e-6<=q[j]+dx<=hi+1e-6:viol.append({'lane':e,'tick':t,'joint_q_index':j,'perturbation':dx,'position':q[j]+dx,'limit':[lo,hi]})
out={'rows':rows,'max_baseline_gap_error_m':max(errors),'baseline_crosscheck_pass':max(errors)<1e-6,'joint_bound_violations':viol,'scope':'fixed-state finite differences, not dynamically reached contact or physical grip'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
