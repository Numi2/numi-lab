from pathlib import Path
from fractions import Fraction as F
import hashlib,json,struct
R=Path.cwd();A=R/'artifacts/crow-rl-contact-precision-20261108';B=R/'build/crow-contact-precision-build-01'
raw=(B/'substep7-capture.bin').read_bytes();assert hashlib.sha256(raw).hexdigest()=='08bcd99d486addeed6f39b9a265ef35e3208ea8cc884c12d5549830be0108d0f'
header=struct.unpack_from('<8I',raw);d=struct.unpack_from('<16I8f',raw,32);nv,rows,count=d[2:5];assert header[2:6]==(0,0,7,len(raw))
blocks=[];offset=256
for i in range(count):
    layout=struct.unpack_from('<8I',raw,offset);values=struct.unpack_from('<20f',raw,offset+32);blocks.append((layout,[F(x) for x in values]));offset+=128
def read(n):
    global offset
    data=[F(x) for x in struct.unpack_from('<'+str(n)+'f',raw,offset)];offset+=n*4;return data
flat=read(nv*nv);mass=[flat[i*nv:(i+1)*nv] for i in range(nv)];flat=read(rows*nv);jac=[flat[i*nv:(i+1)*nv] for i in range(rows)];bias=read(rows);free=read(nv)
reference=struct.unpack('<'+str(nv)+'d',(B/'reference-v.f64').read_bytes());v=[F(struct.unpack('<f',struct.pack('<f',x))[0]) for x in reference]
# Derive upward FP32 spacings directly from the bit patterns, independently
# of the producer's NumPy/LLL/LDL implementation.
ulp=[]
for x in v:
    bits=struct.unpack('<I',struct.pack('<f',float(x)))[0];next_bits=bits-1 if x<0 else bits+1;next_value=F(struct.unpack('<f',struct.pack('<I',next_bits))[0]);ulp.append(next_value-x)
w=[bias[r]+sum((jac[r][j]*v[j] for j in range(nv)),F(0)) for r in range(rows)];impulse=[F(0)]*rows;active=[];regions={}
for layout,values in blocks:
    at,dim,kind=layout[:3];regularization=values[8:8+dim]
    if kind==0:
        lo,hi=values[16:18];p=-w[at]/regularization[0];impulse[at]=min(max(p,lo),hi);region='scalar_free' if lo<p<hi else 'scalar_bound'
        if region=='scalar_free':active.append((at,regularization[0]))
    else:
        assert kind==1 and values[16]==values[17]==0;scale=values[:dim];p=[-w[at+i]/regularization[i] for i in range(dim)];z=[p[i]/scale[i] for i in range(dim)]
        if z[0]>=0 and z[0]*z[0]>=sum((x*x for x in z[1:]),F(0)):
            region='interior'
            for i in range(dim):impulse[at+i]=p[i];active.append((at+i,regularization[i]))
        else:
            dual=[w[at+i]*scale[i] for i in range(dim)];assert dual[0]>=0 and dual[0]*dual[0]>=sum((x*x for x in dual[1:]),F(0));region='vertex'
    regions[region]=regions.get(region,0)+1
action=[sum((mass[i][j]*(v[j]-free[j]) for j in range(nv)),F(0)) for i in range(nv)];constraint=[sum((jac[r][i]*impulse[r] for r in range(rows)),F(0)) for i in range(nv)];gradient=[a-c for a,c in zip(action,constraint)]
first=json.loads((A/'lattice-certificate.json').read_text());certificate=json.loads((A/'mixed-dual-certificate.json').read_text());indices=first['indices'];y=[F(x) for x in certificate['exact_dual']];dual_row_dot={r:sum((y[i]*jac[r][i] for i in range(nv)),F(0)) for r,_ in active}
for column,integer in zip(indices,certificate['integer_dual']):
    dual_h=sum((y[i]*mass[i][column] for i in range(nv)),F(0))+sum((dual_row_dot[row]*jac[row][column]/r for row,r in active),F(0));assert dual_h*ulp[column]==integer
phase=sum((y[i]*gradient[i] for i in range(nv)),F(0));assert phase==F(certificate['exact_phase']);fraction=phase%1;raw_lower=min(fraction,1-fraction)/sum(map(abs,y),F(0));assert raw_lower==F(certificate['exact_raw_lower_bound'])
radius=F(1,10000);contact=[[F(0)]*nv for _ in range(nv)]
for row,r in active:
    nonzero=[i for i in range(nv) if jac[row][i]]
    for i in nonzero:
        for j in nonzero:contact[i][j]+=jac[row][i]*jac[row][j]/r
denominator=1+max(abs(action[i])+radius*sum((abs(mass[i][j]) for j in indices),F(0)) for i in range(nv))+max(abs(constraint[i])+radius*sum((abs(contact[i][j]) for j in indices),F(0)) for i in range(nv));scaled=raw_lower/denominator
assert scaled==F(certificate['exact_scaled_lower_bound']) and scaled>F(1,10**10)
report=dict(schema='crow-contact-independent-exact-certificate-verification-v1',capture_sha256=hashlib.sha256(raw).hexdigest(),parser='struct and exact fractions; no imports from producer or solver-analysis code',dual_identities_exact=len(indices),gradient_phase_exact=True,raw_lower_bound_exact=True,denominator_bound_exact=True,scaled_lower_bound_exact=True,scaled_optimality_lower_bound=float(scaled),unchanged_gate=1e-10,strictly_exceeds_gate=True,projection_regions=regions,hypotheses='Constant stated projection branches, base-spacing FP32 lattice, fixed excluded coordinates, and radius 1e-4 for the denominator bound; hypotheses are not established for every nearby FP32 state. No global impossibility or physical-qualification claim.',native_settings_changed=False,policy_or_profile_promoted=False,goal_complete=False)
(A/'certificate-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True)
