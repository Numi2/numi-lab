from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).parent
fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True,layout='constrained')
for ax,budget in zip(axes,[64,256]):
 name=f'nx16-k{budget}-newton-trace-002';d=json.loads((r/(name+'-newton-analysis.json')).read_text());assert d['integrity']=='ok'
 it=d['iterations'];x=[z['iteration'] for z in it]+[it[-1]['iteration']+1];y=[z['pre_step_residual_norm'] for z in it]+[d['terminal']['certificate_residual_absolute_equivalent']]
 ax.semilogy(x,y,color='#245fa6',marker='.',label='Reassembled nonlinear residual')
 picked=[z for z in it if z['final_cycle_accepted']]
 ax.scatter([z['iteration'] for z in picked],[z['pre_step_residual_norm'] for z in picked],facecolors='none',edgecolors='#188351',s=70,label='Inner cycle accepted')
 ax.axhline(1e-6,color='#666666',linestyle='--',label='Nonlinear threshold (normalizer=1)')
 ax.set_title(f'{budget} Krylov columns: first increment rejected')
 ax.set_xlabel('Newton iteration / terminal certificate at 28');ax.grid(alpha=.2)
 ax.text(.03,.03,f"Growth >10%, full step: {sum(z['residual_grew'] and z['near_full_alpha'] for z in it)}\nScreen with accepted inner cycle: {len(d['screen']['observations'])}",transform=ax.transAxes,fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
axes[0].set_ylabel('Nonlinear impulse residual norm (N s)');axes[0].legend(fontsize=8,loc='upper right')
fig.suptitle('Refined-length mesh: every failed candidate retained\nSame native build; traces reproduce earlier untraced outputs exactly',fontsize=12)
fig.savefig(r/'newton-residual-comparison.png',dpi=170)
