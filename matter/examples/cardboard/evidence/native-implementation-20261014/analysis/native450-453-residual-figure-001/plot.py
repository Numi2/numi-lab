from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('/Users/home/cardboard-implementation-20261014');O=Path(__file__).resolve().parent
cases=[('450: Newton with one CGLS fallback',B/'solver/bounded-cgls64-exact-state-450','#a64b39'),('453: CGLS primary from iteration 48',B/'solver/cgls-primary-after48-control1-453','#186e82')]
fig,ax=plt.subplots(figsize=(8.2,5.0));counts={};bindings={}
for label,p,color in cases:
 rec=json.loads((p/'receipt.json').read_text());assert rec['source_inputs_unchanged'] and not rec['timed_out'] and rec['exit_code']==2
 result=json.loads((p/'output/result.json').read_text());assert result['status']=='failed' and result['accepted_steps']==1
 rows=[]
 for line in (p/'stderr.log').read_text().splitlines():
  if not line.startswith('fem_residual_armijo_iteration='):continue
  r=json.loads(line.split('=',1)[1]);e=r['environments'][0]
  if r['root']==1 and e['accepted'] and not e['certified_noop'] and e['trial_alpha']>0 and e['base_health']=='healthy_at_armijo_entry':
   rows.append((r['newton_iteration'],e['trial_l2']))
 selected={}
 for n,v in rows:selected[n]=v
 x=sorted(selected);y=[selected[n] for n in x];ax.plot(x,y,label=label,color=color,lw=1.8,marker='o',markersize=2.4)
 ax.annotate(f'{y[-1]:.2e}',(x[-1],y[-1]),xytext=(5,6),textcoords='offset points',color=color,fontsize=8)
 counts[label]={'accepted_trial_iterations':x,'last_trial_residual_Ns':y[-1],'observed_final_iteration':x[-1]}
 for q in [p/'receipt.json',p/'stderr.log',p/'output/result.json']:bindings[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
ax.axvline(48,color='#777777',lw=1,ls=':',label='Policy starts at 48')
ax.axhline(1e-6,color='#555555',lw=1,ls='--',label='Unchanged nonlinear threshold')
ax.set_yscale('log');ax.set_xlim(-2,70);ax.set_ylim(2e-7,2e-3);ax.set_xlabel('Newton iteration within physical control 1');ax.set_ylabel('Accepted private-trial residual (N s)');ax.set_title('Improved residual descent; physical control still rejected',fontsize=12)
ax.grid(which='both',alpha=.15);ax.legend(loc='lower left',fontsize=8,frameon=False);fig.tight_layout(rect=(0,.055,1,1));fig.text(.07,.015,'Both full attempts roll back control1. Native450 has no retained iteration63 Armijo record.',fontsize=8,color='#444444')
fig.savefig(O/'residuals.png',dpi=180);fig.savefig(O/'residuals.pdf');plt.close(fig)
(O/'manifest.json').write_text(json.dumps({'schema':'numi.cardboard.native_residual_figure.v1','scope':'source-bound private nonlinear trial histories; no physical-control acceptance','inputs':bindings,'selected_observations':counts,'files':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['plot.py','residuals.png','residuals.pdf']}},indent=2)+'\n');print('figure generated')
