#!/usr/bin/env python3
import hashlib, importlib.util, json, math, struct
from pathlib import Path

ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020")
E=ROOT/"joint-functional-owner-audit-20261023/knee-projection-envelope-001"
OUT=ROOT/"joint-functional-owner-audit-20261023/knee-source-binding-002"
BASE=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/selector6-scalar-inverse-fix-001/results/root0-activation0-fixed-scalar-8steps-125us/invocation.json")
MODULE=Path("/Users/n/numi-human-anatomy-finish-1378/src/numilab_human/projected_joint_envelope.py")
MODSHA="fb316b6eab590ff0c5156bfba71c7f64e864b3edcc9f2bb1cd9cb595924f1d4c"

def sha(b): return hashlib.sha256(b).hexdigest()
def f32(v): return struct.unpack("<f",struct.pack("<f",v))[0]
def evalp(a,x):
 y=0.0
 for c in reversed(a): y=y*x+c
 return y
def roots_in(a,lo,hi):
 d=[a[1],2*a[2],3*a[3],4*a[4]]
 # Split the cubic derivative at its quadratic extrema so each segment is monotone.
 dd=[d[1],2*d[2],3*d[3]]
 cuts=[lo,hi]
 if abs(dd[2])>0:
  disc=dd[1]**2-4*dd[2]*dd[0]
  if disc>=0:
   z=math.sqrt(disc)
   cuts += [x for x in ((-dd[1]-z)/(2*dd[2]),(-dd[1]+z)/(2*dd[2])) if lo<x<hi]
 elif abs(dd[1])>0:
  x=-dd[0]/dd[1]
  if lo<x<hi: cuts.append(x)
 cuts=sorted(set(cuts)); brackets=[]
 for x in cuts:
  if abs(evalp(d,x))<=2e-15: brackets.append((x,x))
 for l0,r0 in zip(cuts,cuts[1:]):
  fl,fr=evalp(d,l0),evalp(d,r0)
  if fl==0 or fr==0 or fl*fr>0: continue
  l,r=l0,r0
  for _ in range(120):
   m=(l+r)/2; fm=evalp(d,m)
   if fm==0: l=r=m; break
   if fl*fm<=0: r,fr=m,fm
   else: l,fl=m,fm
  brackets.append((l,r))
 out=[]
 for br in sorted(brackets):
  if not out or abs(sum(br)/2-sum(out[-1])/2)>1e-11: out.append(br)
 return out

def rigid_layout(b):
 h=struct.unpack_from("<8s10I32s",b); magic,abi,engine,sources,bodies,joints,nq,nv,r0,virt,r1,digest=h
 assert (magic,abi,engine,r0,r1)==(b"NHRIGID2",1,5,0,0) and nq==nv+1
 d0=224+bodies*160+joints*144
 assert len(b)==d0+64*nv+4*(nq+nv)+32*sources
 rows={}
 for v in range(nv):
  r=struct.unpack_from("<8I8f",b,d0+64*v)
  if v>=6: assert r[3]==v and r[2]==v+1; rows[r[2]]=(v,r)
 default=struct.unpack_from(f"<{nq}f",b,d0+64*nv)
 return d0,nq,nv,digest,rows,default

proj=json.loads((E/"projected-joint-envelope.json").read_text())
pub=json.loads((E/"source-publication.json").read_text())
rp=Path(proj["input_paths"]["rigid"]); ep=Path(proj["input_paths"]["equalities"]); cp=Path(proj["output_path"])
rb,eb,cb=rp.read_bytes(),ep.read_bytes(),cp.read_bytes()
assert sha(rb)==proj["input_rigid_sha256"] and sha(eb)==proj["unchanged_equality_sha256"] and sha(cb)==proj["output_rigid_sha256"]
assert sha(MODULE.read_bytes())==MODSHA and pub["files"]["src/numilab_human/projected_joint_envelope.py"]==MODSHA
spec=importlib.util.spec_from_file_location("projected_joint_envelope_oracle",MODULE)
oracle=importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
regen,regen_receipt=oracle.prepare_projected_joint_envelope(rb,eb,[106,120])
assert regen==cb and regen_receipt["output_rigid_sha256"]==sha(cb)
d0,nq,nv,digest,src,defs=rigid_layout(rb)
od0,onq,onv,odigest,out,odefs=rigid_layout(cb)
assert (d0,nq,nv,digest,defs)==(od0,onq,onv,odigest,odefs)
eh=struct.unpack_from("<8s10I32s",eb); assert eh[0]==b"NHEQ1\0\0\0" and eh[-1]==digest
rows=[struct.unpack_from("<4I20f",eb,80+96*i) for i in range(eh[4])]
poly_expected=[0.010506000369787216,0.0247614998370409,-1.3164700269699097,0.7163370251655579,-0.1383019983768463]
extrema=[]
for ri,dq,mq,mv in ((43,114,106,105),(50,128,120,119)):
 row=rows[ri]; assert row[:4]==(dq,dq-1,mq,mv)
 assert row[4:6]==(0.0,0.0) and row[6:11]==tuple(poly_expected)
 dep_ref,master_ref=row[4:6]; poly=row[6:11]
 _,sd=src[dq]; _,od=out[dq]; _,sm=src[mq]; _,om=out[mq]
 assert defs[mq]==0.0 and defs[dq]==0.0
 assert dep_ref + evalp(poly, defs[mq]-master_ref)==sd[9]==poly_expected[0]
 assert sm[8:10]==om[8:10]==(0.0,f32(2.094399929046631))
 domain=sm[8:10]; stationary=roots_in(poly_expected,*domain)
 points=[domain[0],domain[1]]+[sum(z)/2 for z in stationary]
 vals=[evalp(poly_expected,x) for x in points]
 lo,hi=min(vals),max(vals)
 entry=next(x for x in proj["changed_dofs"] if x["dependent_q_index"]==dq)
 assert od[8]<=lo and od[9]>=hi
 assert od[8]<=entry["exact_polynomial_hull"][0]-entry["float32_roundoff_allowance"]
 assert od[9]>=entry["exact_polynomial_hull"][1]+entry["float32_roundoff_allowance"]
 extrema.append({"equality_row":ri,"dependent_q":dq,"master_q":mq,"domain_rad":list(domain),
  "derivative_root_brackets_rad":[list(x) for x in stationary],"candidate_points_rad":points,
  "polynomial_values_rad":vals,"analytic_min_rad":lo,"analytic_max_rad":hi,
  "original_hard_limits_rad":[sd[8],sd[9]],"candidate_hard_limits_rad":[od[8],od[9]],
  "candidate_margin_rad":[lo-od[8],od[9]-hi]})
allowed=set(); changedq=set()
for q,(v,sr) in src.items():
 if sr[8:10]!=out[q][1][8:10]:
  changedq.add(q); off=d0+64*v+32; allowed.update(range(off,off+8))
diffs={i for i,(x,y) in enumerate(zip(rb,cb)) if x!=y}
assert changedq=={x["dependent_q_index"] for x in proj["changed_dofs"]} and diffs and diffs<=allowed
for q in (106,120): assert src[q][1][8:10]==out[q][1][8:10]
base=json.loads(BASE.read_text()); argv=list(base["argv"]); old=argv[1]
assert sha(Path(old).read_bytes())==proj["input_rigid_sha256"]
assert base["activation"]==0 and float(argv[argv.index("--muscle-activation")+1])==0
assert float(argv[argv.index("--muscle-step-seconds")+1])==.000125 and int(argv[argv.index("--muscle-step-count")+1])==8
posed=[int(argv[i+1]) for i,t in enumerate(argv[:-2]) if t=="--pose-q"]
assert not ({106,120}&set(posed))
argv[1]=str(cp); ins=dict(base["inputs_verified"]); assert ins.pop(old)==proj["input_rigid_sha256"]; ins[str(cp)]=proj["output_rigid_sha256"]
bound=dict(base); bound["argv"]=argv; bound["inputs_verified"]=ins
bound["candidate_knee_envelope_binding"]={"rigid_input_sha256":proj["input_rigid_sha256"],
 "rigid_candidate_sha256":proj["output_rigid_sha256"],"equality_sha256":proj["unchanged_equality_sha256"],
 "source_module_sha256":MODSHA,"only_asset_changed":"NHRIGID dependent hard bounds",
 "knee_pose_override":"none; q106=q120 natural defaults 0","status":"prepared_only"}
bp=OUT/"projected-knee-zero-activation-base-invocation.json"
assert not bp.exists(); bp.write_text(json.dumps(bound,indent=2,sort_keys=True)+"\n")
# Verify the retained native baseline's equality-projected natural configuration.
state_path=Path(base["output_directory"])/"stdout.log"
initial=None
for line in state_path.read_text(errors="replace").splitlines():
 if line.startswith("stand_terminal_state="):
  state=json.loads(line.split("=",1)[1]); initial=state["initial_q"]; break
assert initial is not None
assert initial[106]==0.0 and initial[120]==0.0
assert initial[114]==poly_expected[0] and initial[128]==poly_expected[0]
receipt={"schema":"numi.human.knee-source-envelope-binding-audit.v1","status":"source_oracle_full_range_extrema_pass_prepared_only",
 "source_publication_commit":pub["commit"],"source_module_sha256":MODSHA,
 "original_rigid_sha256":sha(rb),"candidate_rigid_sha256":sha(cb),"equality_sha256":sha(eb),
 "master_q_indices":[106,120],"actual_natural_initial_q":{"106":initial[106],"114":initial[114],"120":initial[120],"128":initial[128]},"primary_master_ranges_unchanged":True,"default_q_v_bytes_unchanged":True,
 "source_polynomial_unchanged":True,"all_other_rigid_bytes_unchanged":True,
 "analytic_extrema":extrema,"candidate_regenerated_by_source_oracle_byte_exactly":True,
 "candidate_zero_activation_invocation":str(bp),
 "limits":["No native/GPU run in this audit.","Envelope consistency is not physiological ROM or contact qualification."]}
(OUT/"source-envelope-binding-audit.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(receipt,indent=2))
