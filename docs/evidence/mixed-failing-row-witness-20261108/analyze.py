import copy,json,math
from pathlib import Path

def add(a,b):return [x+y for x,y in zip(a,b)]
def sub(a,b):return [x-y for x,y in zip(a,b)]
def mul(a,s):return [x*s for x in a]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def normalized(a):
 l=math.sqrt(dot(a,a));assert l>1e-10;return mul(a,1/l),l
arms={name:json.loads(Path(name+'.json').read_text()) for name in ('baseline','witness','velocity')}
def physical(x):
 x=copy.deepcopy(x)
 for step in x['steps']:
  for r in step['rows']:
   del r['scratch'];del r['row_trace']
 return x
base=arms['baseline'];assert all(physical(base)==physical(d) for d in arms.values())
assert all(d['capture_noninvasive'] and d['repeat'] and d['exact_trace'] for d in arms.values())
assert (base['success'],base['failed'],base['first_code'])==(7,13,11)
rows=[]
for i,step in enumerate(base['steps']):
 a=arms['witness']['steps'][i];b=arms['velocity']['steps'][i]
 assert len(step['rows'])==len(a['rows'])==len(b['rows'])
 for k,r in enumerate(step['rows']):
  if not r['flags']&8:continue
  w=a['rows'][k]['scratch'];v=b['rows'][k]['scratch']; ia,ib=int(w[0]),int(w[1]);t=w[2];radius=w[3];edge=w[4:7];twist=w[7];radial=w[8:11];target=w[11];va=v[0:3];vb=v[4:7];tool=v[8:11];reg=v[7];n=r['normal']
  assert w[0]==ia and w[1]==ib and 0<=ia<len(base['rod_masses']) and 0<=ib<len(base['rod_masses']) and 0<=t<=1 and radius>0
  axis,l=normalized(edge);dv=sub(vb,va);tangent_rate=mul(sub(dv,mul(axis,dot(axis,dv))),1/l);omega=add(cross(axis,tangent_rate),mul(axis,twist));rod=add(add(mul(va,1-t),mul(vb,t)),cross(omega,mul(radial,radius)));vn=dot(n,sub(tool,rod));eq=vn-target+reg*r['impulse'][0]
  assert all(math.isfinite(z) for z in [vn,eq,*w,*v]);rows.append({'step':i,'row':k,'nodes':[ia,ib],'weights':[1-t,t],'radius':radius,'normal':n,'relative_normal_fp64':vn,'relative_normal_gpu':v[11],'normal_equation_fp64':eq,'normal_equation_gpu':v[3], 'relative_error':abs(vn-v[11]),'equation_error':abs(eq-v[3]),'impulse':r['impulse'][0]})
assert rows
maximum_rel=max(x['relative_error'] for x in rows);maximum_eq=max(x['equation_error'] for x in rows)
assert maximum_rel<=2e-5 and maximum_eq<=2e-5
failed=next(i for i,s in enumerate(base['steps']) if s['code']!=0)
f=[r for r in rows if r['step']==failed];assert len(f)==2
# Nodal point interpolation with diagonal lumped masses only, excludes twist,
# frame rotation, rigid response and the full implicit rod inverse.
def proxy(a,b):
 total=0
 for ia,wa in zip(a['nodes'],a['weights']):
  for ib,wb in zip(b['nodes'],b['weights']):
   if ia==ib:total+=wa*wb/base['rod_masses'][ia]*dot(a['normal'],b['normal'])
 return total
aa=proxy(f[0],f[0]);ab=proxy(f[0],f[1]);bb=proxy(f[1],f[1]);cor=ab/math.sqrt(aa*bb)
out={'baseline_equivalent_exposed_fields':True,'replay_and_capture_checks':True,'first_rejected_step_zero_based':failed,'rod_row_observations_including_repeated_rejections':len(rows),'max_relative_velocity_difference_m_per_s':maximum_rel,'max_equation_difference_m_per_s':maximum_eq,'first_failure_rows':f,'nodal_inverse_mass_only_proxy':{'matrix':[[aa,ab],[ab,bb]],'normalized_off_diagonal':cor,'omits':['rotating rod frame and radius','twist','implicit elastic rod factor','rigid-body response','friction coupling']},'rows':rows}
Path('analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
