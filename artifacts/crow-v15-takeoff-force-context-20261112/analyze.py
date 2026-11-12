from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-takeoff-force-context-20261112')
S=Path('artifacts/crow-v15-downstroke-pushoff-screen-20261112/autonomous-20261126/run.stderr.gz')
prior=json.load(open('artifacts/crow-v15-downstroke-pushoff-screen-20261112/analysis.json'))
ends={x['lane']:x['candidate_unloaded_rising_interval'] for x in prior['rows']}
data={}
with gzip.open(S,'rt') as f:
 for l in f:
  if not l.startswith('CROW_TWO_TOE_SUBSTEP '):continue
  z=l.split();t,k,e=map(int,z[1:4])
  if 348<=t<=ends[e]:
   assert (e,t,k) not in data and int(z[4])<=int(z[5]);data[e,t,k]=list(map(float,z[6:]))
mass=.45006;g=9.81;dt=.02/16;rows=[]
for e in range(8):
 seq=[data[e,t,k] for t in range(348,ends[e]+1) for k in range(16)]
 left=[v[0]/(dt*mass*g) for v in seq];right=[v[1]/(dt*mass*g) for v in seq];both=[x+y for x,y in zip(left,right)]
 rows.append({'lane':e,'substeps':len(seq),'peak_instantaneous_substep_equivalent_force_over_bodyweight_left':max(left),'peak_instantaneous_substep_equivalent_force_over_bodyweight_right':max(right),'peak_summed_feet_over_bodyweight':max(both),'integrated_upward_impulse_Ns':[sum(v[i] for v in seq) for i in (0,1)]})
out={'rows':rows,'window':'interval 348 through first two-unloaded-rising endpoint; each substep 0.00125 s','mass_kg':mass,'gravity_m_s2':g,'scope':'Unfiltered contact impulses divided by substep duration. Not directly comparable to 10 kHz, 40 Hz filtered experimental force data; no ground-reaction force waveform matching or American-crow biological validation.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows: print(x)
