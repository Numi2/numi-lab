from pathlib import Path
import json,csv,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path(__file__).resolve().parent;report=json.loads((r/'force-matched-time-comparison.json').read_text())
assert report['status']=='analyzed_descriptive'
fig,ax=plt.subplots(1,3,figsize=(14,4.5))
bind={}
for name,label,color in [('matched-baseline-001','25 µs / 1e-6','#247b9d'),('matched-halfdt-001','12.5 µs / 5e-7','#c55531')]:
 p=r/name/'observations.csv';rows=[x for x in csv.DictReader(p.open()) if x['environment']=='0'];assert all(x['step_accepted']=='1' for x in rows)
 ax[0].plot([1000*float(x['time_s']) for x in rows],[-float(x['reaction_moment_y_Nm']) for x in rows],label=label,color=color,lw=1.8)
 bind[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
phase=['loading','hold'];x=np.arange(2);metrics=report['endpoint_resolution_screen']['metrics_by_phase']
for j,(key,label,color) in enumerate([('reaction_moment_y_Nm','Moment','#247b9d'),('plastic_volume_rms_frobenius_ep','Plastic strain RMS','#c55531')]):
 ax[1].bar(x+(j-.5)*.3,[100*metrics[p][key]['relative_difference_with_floor'] for p in phase],width=.3,label=label,color=color)
curves=report['reaction_dynamics_and_energy_curves']
for j,(key,label,color) in enumerate([('reaction_x_N','Fx','#247b9d'),('reaction_y_N','Fy','#c55531'),('reaction_z_N','Fz','#7652a5')]):
 ax[2].bar(x+(j-1)*.23,[100*curves[p]['bent'][key]['relative_rms_difference'] for p in phase],width=.23,label=label,color=color)
ax[0].set(xlabel='Physical time (ms)',ylabel='Bending moment magnitude (N m)');ax[0].legend(fontsize=8)
ax[1].set(xticks=x,xticklabels=['Loading end','Hold end'],ylabel='Endpoint relative difference (%)');ax[1].legend(fontsize=8)
ax[2].set(xticks=x,xticklabels=['Loading','Hold'],ylabel='Force curve relative RMS difference (%)');ax[2].legend(fontsize=8)
for a in ax:a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
fig.suptitle('Same-build timestep sensitivity: matched tolerance / timestep ratio',fontsize=16)
fig.text(.5,.015,'48 and 96 accepted steps • measured force imbalance L2 < 0.04 N • loading/hold only, not mesh or physical fold qualification',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(r/'force-matched-time.png',dpi=150)
p=r/'force-matched-time-comparison.json';bind[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(r/'force-matched-plot-bindings.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':bind},indent=2)+'\n')
