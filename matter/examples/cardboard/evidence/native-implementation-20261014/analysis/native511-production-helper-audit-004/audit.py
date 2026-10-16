#!/usr/bin/env python3
import json,hashlib,struct,math
from pathlib import Path
from decimal import Decimal as D, localcontext
B=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
P=B/"analysis/source189-shared-contact-trial-002"
S=B/"snapshots/source-shared-contact-trial-189-buildfix003"
I=B/"analysis/contact-armijo-energy-replay-499-001/case-496-inputs-v4"
N=B/"native/shared-contact-production-helper-511"
OUT=B/"analysis/native511-production-helper-audit-002"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(path):
 b=Path(path).read_bytes()
 return struct.unpack("<"+"f"*(len(b)//4),b)
def vsub(a,b): return tuple(x-y for x,y in zip(a,b))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def closest(p,a,b,c):
 ab=vsub(b,a); ac=vsub(c,a); ap=vsub(p,a)
 d1=dot(ab,ap); d2=dot(ac,ap)
 if d1<=0 and d2<=0:return a,"vertex_a",(D(1),D(0),D(0))
 bp=vsub(p,b); d3=dot(ab,bp); d4=dot(ac,bp)
 if d3>=0 and d4<=d3:return b,"vertex_b",(D(0),D(1),D(0))
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:
  t=d1/(d1-d3); return tuple(a[k]+t*ab[k] for k in range(3)),"edge_ab",(1-t,t,D(0))
 cp=vsub(p,c); d5=dot(ab,cp); d6=dot(ac,cp)
 if d6>=0 and d5<=d6:return c,"vertex_c",(D(0),D(0),D(1))
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:
  t=d2/(d2-d6); return tuple(a[k]+t*ac[k] for k in range(3)),"edge_ac",(1-t,D(0),t)
 va=d3*d6-d5*d4
 if va<=0 and d4-d3>=0 and d5-d6>=0:
  t=(d4-d3)/((d4-d3)+(d5-d6)); bc=vsub(c,b)
  return tuple(b[k]+t*bc[k] for k in range(3)),"edge_bc",(D(0),1-t,t)
 den=va+vb+vc
 if den==0:
  best=None
  for x,y,tag,ba,bb in [(a,b,"ab",(1,0,0),(0,1,0)),(a,c,"ac",(1,0,0),(0,0,1)),(b,c,"bc",(0,1,0),(0,0,1))]:
   e=vsub(y,x); ee=dot(e,e); t=max(D(0),min(D(1),dot(vsub(p,x),e)/ee)) if ee else D(0)
   q=tuple(x[k]+t*e[k] for k in range(3)); d=dot(vsub(p,q),vsub(p,q))
   bary=tuple(D(ba[k])*(1-t)+D(bb[k])*t for k in range(3))
   if best is None or d<best[0]:best=(d,q,tag,bary)
  return best[1],"degenerate_edge_"+best[2],best[3]
 v=vb/den; w=vc/den; u=D(1)-v-w
 return tuple(u*a[k]+v*b[k]+w*c[k] for k in range(3)),"face",(u,v,w)
def ref(pos,thick,stiff):
 with localcontext() as c:
  c.prec=90
  e1=vsub(pos[2],pos[1]); e2=vsub(pos[3],pos[1])
  s1=max(abs(x) for x in e1); s2=max(abs(x) for x in e2)
  if s1==0 or s2==0: moll=D(0)
  else:
   x=tuple(q/s1 for q in e1); y=tuple(q/s2 for q in e2)
   xn=dot(x,x).sqrt(); yn=dot(y,y).sqrt()
   u=tuple(q/xn for q in x); v=tuple(q/yn for q in y); z=cross(u,v)
   sin2=dot(z,z); r=max(D(0),min(D(1),sin2/D("0.001")))
   moll=r*(2-r)
  q,branch,bary=closest(pos[0],pos[1],pos[2],pos[3])
  delta=vsub(pos[0],q); sep=dot(delta,delta).sqrt()
  if thick<=0 or stiff<=0 or sep>=thick: base=D(0)
  else:
   floor=D("0.0001")*thick; d=max(sep,floor); ds=d*d; ts=thick*thick
   off=ds-ts; log=(ds/ts).ln(); at=max(-stiff*off*off*log,D(0))
   if sep>=floor:base=at
   else:
    first=-stiff*(D(2)*off*log+off*off/ds)
    base=max(at+D(2)*d*first*(sep-floor),D(0))
  return {"separation_m":str(sep),"energy_j":str(moll*base),"mollifier":str(moll),
   "branch":branch,"barycentric":[str(x) for x in bary],"barrier_active":bool(sep<thick and thick>0 and stiff>0)}
def bits(f):return struct.unpack("<I",struct.pack("<f",f))[0]
def makepos(raw,base,block):
 return [tuple(D.from_float(raw[base+block*4+k*4+q]) for q in range(3)) for k in range(4)]
def main():
 assert not OUT.exists()
 OUT.mkdir(parents=True)
 ib=(I/"contacts.bin").read_bytes(); inp=json.loads((I/"manifest.json").read_text())
 aa=f32(I/"alphas.bin"); ths=f32(I/"collision-thickness.bin")
 contacts=[]
 for c in range(3):
  o=c*432; nodes=struct.unpack_from("<4I",ib,o); ident=struct.unpack_from("<4I",ib,o+16)
  barrier=struct.unpack_from("<4f",ib,o+144); moll=struct.unpack_from("<4f",ib,o+272)
  assert ident[2]==0
  contacts.append({"nodes":list(nodes),"kind":ident[2],"thickness":D.from_float(barrier[2]),
   "stiffness":D.from_float(moll[3]),"input_thickness":ths[c],
   "slot":inp["records"][c]["contact_slot"],"active_slot":inp["records"][c]["active_slot"]})
 ob=(N/"output.bin").read_bytes(); raw=struct.unpack("<"+str(len(ob)//4)+"f",ob)
 assert len(raw)==24*21*4 and all(math.isfinite(x) for x in raw)
 records=[]
 for i in range(24):
  c=i//8; j=i%8; base=i*21*4; co=contacts[c]
  get=lambda k:raw[base+4*k:base+4*k+4]
  positions={name:makepos(raw,base,block) for name,block in
    (("current",0),("contact_trial",4),("applied_trial",8))}
  refs={k:ref(v,co["thickness"],co["stiffness"]) for k,v in positions.items()}
  n12=get(12); n13=get(13); ccd=get(14); ccdf=get(15); appliedccd=get(17); appliedf=get(18); arm=get(20)
  native_e={"current":n12[0],"contact_trial":n12[2],"applied_trial":n13[0]}
  native_d={"current":n12[1],"contact_trial":n12[3],"applied_trial":n13[1]}
  high_e={k:float(D(v["energy_j"])) for k,v in refs.items()}
  high_d={k:float(D(v["separation_m"])) for k,v in refs.items()}
  records.append({
   "row":i,"contact_index":c,"trial_index":j,"active_slot":co["active_slot"],"contact_slot":co["slot"],
   "alpha":arm[2],"nodes":co["nodes"],"barrier":{"thickness_m":str(co["thickness"]),
    "stiffness":str(co["stiffness"]),"input_thickness_m":co["input_thickness"]},
   "production":{"energy_j":native_e,"separation_m":native_d,
    "trial_valid_bits":n13[3],"contact_ccd":{"hit":ccd[0],"certified":ccd[1],"toi":ccd[2],
     "separation_at_return_m":ccd[3],"tested_fraction":ccdf[0],"retries":ccdf[1]},
    "applied_ccd":{"hit":appliedccd[0],"certified":appliedccd[1],"toi":appliedccd[2],
     "separation_at_return_m":appliedccd[3],"tested_fraction":appliedf[0],"retries":appliedf[1]},
    "directional_derivative_j_per_alpha":arm[0],"armijo_rhs_j":arm[1],"derivative_inputs_valid":arm[3]},
   "independent_decimal90":{"geometry_energy":refs,
    "energy_change_j":{"contact_trial":str(D(refs["contact_trial"]["energy_j"])-D(refs["current"]["energy_j"])),
     "applied_trial":str(D(refs["applied_trial"]["energy_j"])-D(refs["current"]["energy_j"]))},
    "clearance_m":{k:str(D(v["separation_m"])-co["thickness"]) for k,v in refs.items()},
    "native_energy_ulp":{"current":abs(bits(native_e["current"])-bits(float(D(refs["current"]["energy_j"])))),
     "contact_trial":abs(bits(native_e["contact_trial"])-bits(float(D(refs["contact_trial"]["energy_j"])))),
     "applied_trial":abs(bits(native_e["applied_trial"])-bits(float(D(refs["applied_trial"]["energy_j"]))))}}
  })
 all_contact=all(r["production"]["contact_ccd"]["hit"]==0 and r["production"]["contact_ccd"]["certified"]==1 and r["production"]["contact_ccd"]["tested_fraction"]==1 for r in records)
 all_applied=all(r["production"]["applied_ccd"]["hit"]==0 and r["production"]["applied_ccd"]["certified"]==1 and r["production"]["applied_ccd"]["tested_fraction"]==1 for r in records)
 all_clear=all(D(r["independent_decimal90"]["clearance_m"][k])>0 for r in records for k in ("current","contact_trial","applied_trial"))
 signs=[D(r["independent_decimal90"]["energy_change_j"]["applied_trial"]) for r in records]
 summary={"samples":len(records),"all_contact_paths_native_ccd_certified_full_clear":all_contact,
 "all_applied_paths_native_ccd_certified_full_clear":all_applied,"all_endpoints_decimal_clear_of_thickness":all_clear,
 "applied_energy_delta_counts":{"negative":sum(x<0 for x in signs),"zero":sum(x==0 for x in signs),"positive":sum(x>0 for x in signs)},
 "max_abs_native_vs_decimal_applied_energy_j":max(abs(r["production"]["energy_j"]["applied_trial"]-float(D(r["independent_decimal90"]["geometry_energy"]["applied_trial"]["energy_j"]))) for r in records),
 "max_abs_native_vs_decimal_applied_separation_m":max(abs(r["production"]["separation_m"]["applied_trial"]-float(D(r["independent_decimal90"]["geometry_energy"]["applied_trial"]["separation_m"]))) for r in records)}
 source_files={"runtime.mm":S/"matter/src/runtime.mm","fem.metalinc":S/"matter/src/metal/fem.metalinc",
  "contact.metalinc":S/"matter/src/metal/contact.metalinc","common.metalinc":S/"matter/src/metal/common.metalinc"}
 report={"schema":"numi.cardboard.native511-production-helper-audit.v1",
 "scope":"Independent Decimal 90-digit point-triangle closest distance, dimensionless feature mollifier and IPC energy evaluated on the exact emitted FP32 endpoint words. Production full-sweep CCD booleans are compared with independent high-precision endpoint clearance. Endpoint clearance does not replace a CCD proof.",
 "bindings":{"source_snapshot":str(S),"source_hashes":{k:sha(v) for k,v in source_files.items()},
  "input_manifest_sha256":sha(I/"manifest.json"),"native_output_sha256":sha(N/"output.bin"),
  "native_receipt_sha256":sha(N/"receipt.json"),"metallib_sha256":sha(P/"contact_barrier_trial_replay_source189.metallib"),
  "host_sha256":sha(P/"contact_barrier_trial_replay_source189_host"),"physical_transactions":0},
 "method":{"precision_decimal_digits":90,"fp32_endpoints":"exact emitted float32 words converted exactly to Python binary64 then Decimal.from_float",
  "closest_point":"Ericson triangle Voronoi closest-point regions evaluated using Decimal operations",
  "barrier":"source189 common.metalinc ipcBarrierEnergy, with feature mollifier from source189 contact.metalinc; evaluated in Decimal",
  "ccd":"source production certifier replayed for both contact-limiter endpoint and actual femTrialNodeAtAlpha endpoint; required hit=0, certified=1, testedFraction=1. Decimal endpoint clearance is an independent consistency check."},
 "summary":summary,"records":records}
 (OUT/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
 print(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False))
if __name__=="__main__":main()
