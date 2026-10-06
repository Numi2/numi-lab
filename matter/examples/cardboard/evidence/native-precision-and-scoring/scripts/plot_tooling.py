from pathlib import Path
import csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent
fig,ax=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
bindings={}
for name,label,color in [('predictor20-001','20 µm: 96 accepted','#226c88'),('predictor50-001','50 µm: 15 accepted, then rejected','#bc5837')]:
 p=root/name/'tool-observations.csv';q=root/name/'observations.csv'
 for f in [p,q,root/name/'manifest.json',root/name/'result.json']:bindings[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
 obs={(r['step'],r['environment']):r for r in csv.DictReader(q.open())}
 rows=[r for r in csv.DictReader(p.open()) if r['environment']=='0' and obs[(r['step'],r['environment'])]['step_accepted']=='1']
 print(name,rows[0].keys())
 # Column names are the instrument's exact schema, not inferred labels.
 x=[1e6*float(r['commanded_end_travel_m']) for r in rows]
 ax[0].plot(x,[float(r['punch_force_z_N']) for r in rows],color=color,label=label)
 ax[1].plot([int(r['step'])+1 for r in rows],[1e6*float(r['min_end_punch_node_gap_m']) for r in rows],color=color)
 ax[2].plot([int(r['step'])+1 for r in rows],[float(obs[(r['step'],r['environment'])]['free_force_imbalance_l2_N']) for r in rows],color=color)
ax[0].set(xlabel='Commanded punch travel (µm)',ylabel='Native punch reaction (N)')
ax[1].set(xlabel='Accepted step',ylabel='Minimum sampled punch gap (µm)',yscale='log')
ax[2].set(xlabel='Accepted step',ylabel='Free-node force imbalance L2 (N)')
for a in ax:a.grid(alpha=.2)
ax[0].legend(fontsize=8)
fig.suptitle('Corrected native tool path: accepted states only',fontsize=16)
fig.text(.5,-.06,'Exploratory corrugated-board instrument • residual threshold 1e-4 • forces are not physically calibrated • 50 µm attempt did not complete',ha='center',fontsize=10)
fig.savefig(root/'tooling-comparison.png',dpi=170,bbox_inches='tight')
(root/'tooling-comparison.json').write_text(json.dumps({'schema':'numi.cardboard.tooling-plot.v1','bindings':bindings,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'physical_validation':False},indent=2)+'\n')
