from pathlib import Path
import csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
fig,axes=plt.subplots(1,3,figsize=(13.5,4.5))
bindings={}
for scale,color in [(1,'#c55531'),(2,'#247b9d'),(4,'#7652a5')]:
 name=f'barrier50-scale{scale}-001';run=root/name
 rows=[r for r in csv.DictReader((run/'tool-observations.csv').open()) if r['environment']=='0' and r['step_accepted']=='1']
 obs=[r for r in csv.DictReader((run/'observations.csv').open()) if r['environment']=='0' and r['status_code']=='0' and r['step_accepted']=='1']
 result=json.loads((run/'result.json').read_text());label=f'{scale}× barrier: {result["accepted_steps"]} accepted'+(' then rejected' if result['status']!='completed' else '')
 step=[int(r['step'])+1 for r in rows]
 axes[0].plot([float(r['commanded_end_travel_m'])*1e6 for r in rows],[float(r['punch_force_z_N']) for r in rows],label=label,color=color)
 axes[1].plot(step,[float(r['min_end_punch_node_gap_m'])*1e9 for r in rows],color=color)
 axes[2].plot([int(r['step'])+1 for r in obs],[float(r['free_force_imbalance_l2_N']) for r in obs],color=color)
 for filename in ['tool-observations.csv','observations.csv','result.json']:
  p=run/filename;bindings[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
axes[0].set(xlabel='Commanded punch travel (µm)',ylabel='Native punch reaction (N)');axes[0].legend(fontsize=8)
axes[1].set(xlabel='Accepted step',ylabel='Minimum sampled punch gap (nm)',yscale='log')
axes[2].set(xlabel='Accepted step',ylabel='Free-node force imbalance L2 (N)')
for ax in axes:ax.grid(alpha=.15)
fig.suptitle('50 µm native tool cycle: barrier conditioning sensitivity',fontsize=16)
fig.text(.5,.015,'Same collision floors and 1e-4 residual threshold • accepted states only • numerical experiment, not calibrated scoring',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(root/'barrier-sensitivity.png',dpi=150)
(root/'barrier-plot-bindings.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':bindings},indent=2)+'\n')
