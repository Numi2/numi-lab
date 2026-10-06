import json,pathlib,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
r=pathlib.Path('/Users/home/cardboard-closure-evidence-20261006/box-compile-001')
m=json.loads((r/'mesh.json').read_text()); layout=json.loads((r/'box-layout.json').read_text())
p=np.array(m['nodes_m'])*1000
faces={}
for t,material in zip(m['tetrahedra'],m['material_indices']):
 for f in [(t[0],t[1],t[2]),(t[0],t[3],t[1]),(t[0],t[2],t[3]),(t[1],t[3],t[2])]:
  key=tuple(sorted(f))
  if key in faces: faces[key]=None
  else: faces[key]=(f,material)
surface=[x for x in faces.values() if x]
fig=plt.figure(figsize=(13,7),facecolor='#faf8f3')
ax=fig.add_axes([0.03,0.10,0.70,0.79],projection='3d',computed_zorder=False,facecolor='#faf8f3')
colors=['#b08755','#caa775','#eedbb2']
collection=Poly3DCollection([p[list(f)] for f,mat in surface],facecolors=[colors[mat] for f,mat in surface],edgecolors='none',linewidths=0,zorder=1)
ax.add_collection3d(collection)
for line in layout['score_lines']:
 q=np.array([line['first'],line['second']])*1000
 q[:,2]=p[:,2].max()+.08
 ax.plot(q[:,0],q[:,1],q[:,2],color='#9c3e30',ls='--',lw=1,zorder=10)
ax.set_xlim(0,p[:,0].max());ax.set_ylim(0,p[:,1].max());ax.set_zlim(0,p[:,2].max())
ax.set_box_aspect((np.ptp(p[:,0]),np.ptp(p[:,1]),np.ptp(p[:,2])))
ax.view_init(elev=42,azim=-62);ax.set_proj_type('ortho');ax.set_axis_off()
fig.text(.05,.94,'Corrugated box blank',fontsize=23,fontweight='bold',color='#30291f')
fig.text(.05,.892,'FEFCO 0201 layout • source-derived layered geometry',fontsize=12,color='#5e564c')
fig.text(.745,.73,f"{len(p):,} nodes\n{len(m['tetrahedra']):,} tetrahedra\n\nLiners + flute + glue\n6 flap slots\nJoint tab",fontsize=12,linespacing=1.6,color='#443b2e')
fig.text(.745,.31,'Dashed lines are\nscore locations only.\n\nNot yet scored, folded\nor assembled.',fontsize=11,linespacing=1.5,color='#9c3e30')
fig.text(.05,.06,'Authored solver input. Actual mesh coordinates; no imposed folded shape. Dimensions in SI metres in the accompanying mesh.',fontsize=9,color='#6e655a')
fig.savefig(r.parent/'box-blank.png',dpi=180,facecolor=fig.get_facecolor())
