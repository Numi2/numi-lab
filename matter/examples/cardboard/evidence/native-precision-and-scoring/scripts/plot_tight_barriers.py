"""Plot bound native observations; no interpolation or physical-data claim."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
report=json.loads((root/'tight-barrier-comparison.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
for factor,color in [(1,'#007c91'),(2,'#bb681f'),(4,'#8759b5')]:
 name=f'barrier50-scale{factor}-tight-001'
 rows=[r for r in csv.DictReader((root/name/'tool-observations.csv').open()) if r['environment']=='0']
 obs={int(r['step']):r for r in csv.DictReader((root/name/'observations.csv').open()) if r['environment']=='0'}
 rows=[r for r in rows if obs[int(r['step'])]['step_accepted']=='1' and obs[int(r['step'])]['status_code']=='0']
 t=[float(obs[int(r['step'])]['time_s'])*1000 for r in rows]
 axes[0].plot(t,[float(r['punch_force_z_N']) for r in rows],color=color,label=f'Factor {factor} — '+('48/96; withdrawal rejected' if factor==1 else '96/96 accepted'))
 axes[1].plot(t,[float(r['min_end_punch_node_gap_m'])*1e6 for r in rows],color=color)
axes[1].annotate('Factor 1 stops here',xy=(1.2,.25),xytext=(1.35,.35),arrowprops={'arrowstyle':'->','color':'#007c91'},fontsize=9,color='#007c91')
axes[0].set(ylabel='Punch reaction Z (N)',xlabel='Native physical time (ms)')
axes[1].set(ylabel='Minimum sampled node–punch gap (µm)',xlabel='Native physical time (ms)',yscale='log')
axes[0].legend(frameon=False)
for ax in axes:
 ax.grid(alpha=.2)
fig.suptitle('50 µm tool travel · residual tolerance 10⁻⁶\nNative accepted states; material damage and physical scoring remain unqualified',fontsize=12)
fig.savefig(root/'tight-barrier-comparison.png',dpi=170)
