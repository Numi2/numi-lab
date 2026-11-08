import json, math, pathlib
p=pathlib.Path(__file__).parent
cases=json.loads((p/"output.json").read_text())["cases"]
def sub(a,b): return [x-y for x,y in zip(a,b)]
def add(a,b): return [x+y for x,y in zip(a,b)]
def mul(a,k): return [x*k for x in a]
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def norm(a): return math.sqrt(sum(x*x for x in a))
origin=[0.,1.,0.]; result=[]
for c in cases:
 assert c["accepted"] and c["status"]==0 and c["replay"] and c["contacts"]==1
 row=c["rows"][0]; assert (row["a"],row["b"])==(1,2)
 a0=sub(c["initial_q"][:3],origin); b0=sub(c["scene_start_position"][:3],origin)
 a1=sub(c["final_q"][:3],origin); b1=sub(c["scene_end_position"][:3],origin)
 va0=c["initial_v"][:3]; vb0=c["scene_start_velocity"][:3]; va1=c["final_v"][:3]; vb1=c["scene_end_velocity"][:3]
 wa0=c["initial_v"][3:6]; wa1=c["final_v"][3:6]; wb1=c["scene_end_angular"][:3]
 dp=add(sub(va1,va0),sub(vb1,vb0))
 spin=mul(add(sub(wa1,wa0),wb1),.001)
 initial=add(cross(a0,va0),cross(b0,vb0)); final0=add(cross(a0,va1),cross(b0,vb1)); final1=add(cross(a1,va1),cross(b1,vb1))
 delta=add(sub(final0,initial),spin); final_delta=add(sub(final1,initial),spin)
 n=row["normal"][:3]; t=row["tangent"][:3]; u=cross(n,t); ij=row["impulse"]
 impulse=add(add(mul(n,ij[0]),mul(t,ij[1])),mul(u,ij[2]))
 witness_a=add(a0,row["anchor_a"][:3]); witness_b=add(b0,row["anchor_b"][:3]); midpoint=sub(row["point"][:3],origin)
 assert abs(norm(sub(witness_b,witness_a))-c["gap"])<2e-7
 predicted=cross(sub(midpoint,witness_a),impulse)
 err=norm(sub(delta,predicted)); poserr=norm(sub(delta,final_delta))
 print(c["gap"], norm(dp), delta, predicted, err, poserr, file=__import__("sys").stderr)
 assert norm(dp)<1e-7 and err<1e-7 and poserr<1e-7
 if c["gap"]==0 or c["friction"]==0: assert norm(delta)<1e-7
 else: assert norm(delta)>1e-5
 result.append(dict(gap_m=c["gap"],friction=c["friction"],contact_impulse_Ns=impulse,delta_angular_momentum_Nms=delta,predicted_asymmetric_angular_impulse_Nms=predicted,angular_accounting_error_Nms=err,final_position_consistency_error_Nms=poserr,linear_momentum_error_Ns=norm(dp)))
print(json.dumps({"cases":result},indent=2))
