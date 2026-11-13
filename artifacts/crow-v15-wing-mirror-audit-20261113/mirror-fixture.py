from pathlib import Path
import json,math
A=Path('artifacts/crow-v15-wing-mirror-audit-20261113');D=A/'one-step-20261126';joints={}
for line in (D/'run.stderr').read_text().splitlines():
 z=line.split()
 if z and z[0]=='CROW_WING_FRAME':
  assert len(z)==23,(len(z),line)
  i,parent,child,qi,vi=map(int,z[1:6]);v=list(map(float,z[6:]));item=dict(parent=parent,child=child,q_index=qi,v_index=vi,axis=v[:3],parent_anchor=v[3:6],child_anchor=v[6:9],parent_rotation=v[9:13],child_rotation=v[13:17]);assert i not in joints or joints[i]==item;joints[i]=item
assert set(joints)==set(range(6)),joints
for j in joints.values():
 assert all(abs(x-y)<1e-7 for x,y in zip(j['parent_rotation'],[0,0,0,1]))
 assert all(abs(x-y)<1e-7 for x,y in zip(j['child_rotation'],[0,0,0,1]))
S=[1,-1,1]
for l,r in ((0,1),(2,3),(4,5)):
 for field in ('parent_anchor','child_anchor'):
  assert max(abs(joints[r][field][i]-S[i]*joints[l][field][i]) for i in range(3))<1e-7,(l,r,field)
def mul(a,b):return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def rot(axis,t):
 n=math.sqrt(sum(x*x for x in axis));assert n>0;x,y,z=[v/n for v in axis];c=math.cos(t);s=math.sin(t);v=1-c
 return [[c+x*x*v,x*y*v-z*s,x*z*v+y*s],[y*x*v+z*s,c+y*y*v,y*z*v-x*s],[z*x*v-y*s,z*y*v+x*s,c+z*z*v]]
def response(values,fix):
 L=mul(mul(rot(joints[0]['axis'],values[0]),rot(joints[2]['axis'],values[1])),rot(joints[4]['axis'],values[2]));r=joints[5]['axis'];r=[-v for v in r] if fix else r
 R=mul(mul(rot(joints[1]['axis'],values[0]),rot(joints[3]['axis'],values[1])),rot(r,values[2]));expected=[[S[i]*L[i][j]*S[j] for j in range(3)] for i in range(3)];return dict(max_matrix_error=max(abs(R[i][j]-expected[i][j]) for i in range(3) for j in range(3)),forward_chord_error=[R[i][0]-expected[i][0] for i in range(3)])
cases=[]
for values in ((0.,0.,.2),(.08,.35,.2),(-.05,-.3,-.18)):
 cases.append(dict(joint_coordinates=values,actual=response(values,False),hypothetical_right_axis_reversed=response(values,True)))
out=dict(joints=joints,cases=cases,actual_symmetric=all(c['actual']['max_matrix_error']<1e-6 for c in cases),hypothetical_symmetric=all(c['hypothetical_right_axis_reversed']['max_matrix_error']<1e-6 for c in cases),scope='Analytic joint-frame rotation symmetry at equal coordinates; not a dynamic or aerodynamic validation.')
(A/'mirror-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
