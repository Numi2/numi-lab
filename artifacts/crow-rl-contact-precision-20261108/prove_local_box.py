from pathlib import Path
import json,struct
from fractions import Fraction as F
R=Path.cwd();A=R/'artifacts/crow-rl-contact-precision-20261108'
source=(A/'verify_certificate.py').read_text().split('report=dict(schema=')[0];s={};exec(compile(source,'independent-exact-input-definitions','exec'),s)
nv=s['nv'];rows=s['rows'];v=s['v'];y=s['y'];jac=s['jac'];mass=s['mass'];blocks=s['blocks'];w=s['w'];active=s['active'];ulp=s['ulp'];indices=s['indices'];certificate=s['certificate'];raw=s['raw'];contact=s['contact'];raw_lower=s['raw_lower']
duals={index:integer for index,integer in zip(indices,certificate['integer_dual'])};all_columns=True
for column in range(nv):
    dual_h=sum((y[i]*(mass[i][column]+contact[i][column]) for i in range(nv)),F(0))
    expected=F(duals.get(column,0))/ulp[column] if duals.get(column,0) else F(0)
    all_columns &= dual_h==expected
assert all_columns
attempts=[];selected=None
for shrink in range(31):
    radius=F(1,10000*2**shrink);valid=True;counts={}
    for layout,values in blocks:
        at,dim,kind=layout[:3];r=values[8:8+dim];spread=[radius*sum(map(abs,jac[at+i]),F(0)) for i in range(dim)];lo=[w[at+i]-spread[i] for i in range(dim)];hi=[w[at+i]+spread[i] for i in range(dim)]
        if kind==0:
            lower,upper=values[16:18];base=-w[at]/r[0];minimum=-hi[0]/r[0];maximum=-lo[0]/r[0]
            if lower==upper:region='scalar_fixed';ok=True
            elif base<=lower:region='scalar_lower';ok=maximum<=lower
            elif base>=upper:region='scalar_upper';ok=minimum>=upper
            else:region='scalar_free';ok=lower<minimum and maximum<upper
        else:
            scale=values[:dim];p=[-w[at+i]/r[i]/scale[i] for i in range(dim)]
            if p[0]>=0 and p[0]**2>=sum((x*x for x in p[1:]),F(0)):
                region='cone_interior';normal_min=-hi[0]/r[0]/scale[0];tangent_max=[max(abs(lo[i]/r[i]/scale[i]),abs(hi[i]/r[i]/scale[i])) for i in range(1,dim)];ok=normal_min>=0 and normal_min**2>=sum((x*x for x in tangent_max),F(0))
            else:
                region='cone_vertex';normal_min=lo[0]*scale[0];tangent_max=[max(abs(lo[i]*scale[i]),abs(hi[i]*scale[i])) for i in range(1,dim)];ok=normal_min>=0 and normal_min**2>=sum((x*x for x in tangent_max),F(0))
        valid &= ok;counts[region]=counts.get(region,0)+1
    grid=[]
    for index,integer in duals.items():
        if not integer:continue
        bits=struct.unpack('<I',struct.pack('<f',float(v[index])))[0];exponent=((bits>>23)&255)-127;minimum_magnitude=F(2)**exponent;ok=abs(v[index])-radius>=minimum_magnitude;valid &=ok;grid.append(dict(dof=index,entire_box_on_required_spacing_lattice=ok))
    attempts.append(dict(radius=float(radius),all_branch_and_grid_hypotheses_proven=valid))
    if valid:selected=(radius,counts,grid);break
assert selected is not None
radius,counts,grid=selected
denominator=1+max(abs(s['action'][i])+radius*sum(map(abs,mass[i]),F(0)) for i in range(nv))+max(abs(s['constraint'][i])+radius*sum(map(abs,contact[i]),F(0)) for i in range(nv));scaled=raw_lower/denominator;assert scaled>F(1,10**10)
failed=struct.unpack_from('<'+str(nv)+'f',raw,s['offset']+4*(nv+rows));failed_distance=max(abs(F(x)-base) for x,base in zip(failed,v));zero_contact_dual=all(sum((y[i]*jac[row][i] for i in range(nv)),F(0))==0 for row in range(rows))
report=dict(schema='crow-contact-exact-local-box-impossibility-v1',capture_sha256=s['hashlib'].sha256(raw).hexdigest(),all_generalized_velocity_dual_identities_exact=True,excluded_coordinates_need_not_be_fixed=True,branch_interval_proofs=counts,required_fp32_spacing_proofs=grid,exact_box_radius=str(radius),box_radius=float(radius),exact_scaled_optimality_lower_bound=str(scaled),scaled_optimality_lower_bound=float(scaled),gate=1e-10,every_actual_fp32_velocity_in_box_fails_gate=True,exact_native_failed_distance=str(failed_distance),native_failed_distance=float(failed_distance),native_failed_velocity_inside_proven_box=failed_distance<=radius,contact_dual_rows_all_exactly_zero=zero_contact_dual,attempts=attempts,scope='Exact mathematical proof for every actual finite FP32 velocity in this explicitly bounded box around the rounded reference, using the compiled FP32 input model. No global impossibility claim outside the box; no physical qualification or tolerance change.',native_settings_changed=False,policy_or_profile_promoted=False,goal_complete=False)
(A/'local-box-proof.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True)
