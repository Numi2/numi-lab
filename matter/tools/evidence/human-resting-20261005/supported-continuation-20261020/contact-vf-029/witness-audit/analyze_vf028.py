#!/usr/bin/env python3
"""Independent float64 VF witness audit for captured native 028."""
import hashlib, json, math, struct
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TRACE=ROOT/"contact-budget64-028/native-integrated/analysis/selected-traces.json"
def f32(x): return struct.unpack("<f",struct.pack("<f",float(x)))[0]
def vec(v): return tuple(f32(x) for x in v)
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def mul(s,a): return tuple(s*x for x in a)
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def seg(p,a,b,name):
 d=sub(b,a); dd=dot(d,d); t=max(0.,min(1.,dot(sub(p,a),d)/dd)) if dd else 0.; q=add(a,mul(t,d)); r=sub(p,q)
 return {"feature":name,"point":q,"bary":None,"d2":dot(r,r),"t":t}
def closest(p,a,b,c):
 ab=sub(b,a); ac=sub(c,a); ap=sub(p,a); d00=dot(ab,ab); d01=dot(ab,ac); d11=dot(ac,ac); d20=dot(ap,ab); d21=dot(ap,ac); den=d00*d11-d01*d01; cand=[]
 if den>0:
  u=(d11*d20-d01*d21)/den; v=(d00*d21-d01*d20)/den
  if u>=0 and v>=0 and u+v<=1:
   q=add(a,add(mul(u,ab),mul(v,ac))); r=sub(p,q); cand.append({"feature":"face","point":q,"bary":(1-u-v,u,v),"d2":dot(r,r)})
 for q in (seg(p,a,b,"edge_ab"),seg(p,b,c,"edge_bc"),seg(p,c,a,"edge_ca")):
  t=q.pop("t")
  if q["feature"]=="edge_ab": q["bary"]=(1-t,t,0.)
  elif q["feature"]=="edge_bc": q["bary"]=(0.,1-t,t)
  else: q["bary"]=(t,0.,1-t)
  cand.append(q)
 w=min(cand,key=lambda z:z["d2"]); d=math.sqrt(w["d2"]); delta=sub(p,w["point"]); n=mul(1/d,delta) if d else (math.nan,)*3
 return w,d,n
def ipc(s,t,k):
 if s>=t:return 0.
 f=1e-4*t; d=max(s,f); d2=d*d;t2=t*t;o=d2-t2;e=max(-k*o*o*math.log(d2/t2),0.)
 if s>=f:return e
 first=-k*(2*o*math.log(d2/t2)+o*o/d2); return max(e+2*d*first*(s-f),0.)
def main():
 x=json.loads(TRACE.read_text()); r=next(r for r in x["matching_contact_records"] if r["iteration"]==22 and r["contact"]["contact_slot"]==2941); c=r["contact"]
 base=[vec(v) for v in c["feature_start"]]; oldn=vec(c["normal"]); wt=vec(c["weights"][:4]); thick=f32(c["barrier_parameters"][2]); stiff=f32(c["barrier_parameters"][3])
 def witness(points,label,alpha=None):
  p,a,b,cc=[vec(v) for v in points]; w,d,n=closest(p,a,b,cc); rel=sub(p,w["point"]); gap=d-thick; fixed=dot(oldn,rel)-thick
  return {"trial":label,"alpha":alpha,"feature":w["feature"],"bary_f64":list(w["bary"]),"separation_f64_m":d,"gap_f64_m":gap,"normal_f64":list(n),"old_normal_dot_true_normal":dot(oldn,n),"old_normal_signed_gap_m":fixed,"old_normal_gap_error_m":fixed-gap,"energy_f64_J":ipc(d,thick,stiff),"delta_f64_m":list(rel)}
 b=witness(base,"base"); trials=[]
 for t in c["per_alpha_trace"]["trials"]:
  a=f32(t["ccd"][0]); z=witness(t["trial_feature_endpoints"],t["index"],a); z["delta_distance_from_base_m"]=z["separation_f64_m"]-b["separation_f64_m"]; z["delta_energy_from_base_J"]=z["energy_f64_J"]-b["energy_f64_J"]; z["old_normal_dot_trial_point_motion_m"]=dot(oldn,sub(vec(t["trial_feature_endpoints"][0]),base[0])); z["traced_trial_energy_J"]=t["energy"][0]; z["traced_trial_separation_m"]=t["energy"][1]; z["traced_directional_gap_m"]=t["weighted_direction_and_normal_dot"][3]; z["weighted_direction_f32"]=list(vec(t["weighted_direction_and_normal_dot"][:3])); z["fp64_normal_directional_gap_m"]=dot(vec(b["normal_f64"]),z["weighted_direction_f32"]); trials.append(z)
 weighted=tuple(sum(wt[i]*base[i][k] for i in range(4)) for k in range(3)); tracedgap=f32(c["barrier"][0])
 out={"schema":"numi.human.contact_vf_fp64_witness_audit.v1","source_trace_path":str(TRACE),"source_trace_sha256":hashlib.sha256(TRACE.read_bytes()).hexdigest(),"stderr_sha256":x["stderr_sha256"],"capture":{"root":r["root"],"microtick":r["microtick"],"newton_iteration":r["iteration"],"contact_slot":c["contact_slot"],"identity":c["identity"],"kind":"vertex_triangle","input":"recorded decimal start/trial endpoint coordinates rounded to binary32, then evaluated in float64","algorithm":"valid face projection plus all three clamped edge projections; minimum squared distance"},"old_generated_normal_f32":list(oldn),"thickness_m":thick,"stiffness_N_per_m":stiff,"traced_solver_plane_gap_m":tracedgap,"base_weighted_position_m":list(weighted),"base":{**b,"traced_separation_m":c["barrier"][3],"traced_energy_J":c["barrier"][2]},"trials":trials,"limitations":["CPU-only independent geometry calculation; no GPU rerun.","Barrier energy is recomputed from serialized thickness/stiffness and source formula; feature mollifier is not separately serialized.","This diagnoses the captured old witness and does not qualify the mapped-witness repair."]}
 (HERE/"analysis.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 md=["# Captured VF witness audit: native contact-budget64-028","", "The script uses contact feature_start as the base configuration, uses per-alpha trial_feature_endpoints as trial configurations, rounds all recorded trace coordinates to IEEE-754 binary32, then independently computes the closest triangle point in float64 by checking an admissible face projection and all three clamped edges. No GPU/build was run.","",f"- Trace SHA-256: `{out['source_trace_sha256']}`",f"- stderr SHA-256: `{out['stderr_sha256']}`",f"- Capture root/tick/iteration/slot: {r['root']}/{r['microtick']}/{r['iteration']}/{c['contact_slot']}; identity `{c['identity']}`.",f"- Base feature/bary: {b['feature']} / `{b['bary_f64']}`.",f"- Base separation: FP64 `{b['separation_f64_m']:.12e} m`; trace `{c['barrier'][3]:.12e} m`.",f"- Old emitted normal cosine against FP64 closest-feature normal: `{b['old_normal_dot_true_normal']:.12g}` (angle `{math.degrees(math.acos(max(-1.,min(1.,b['old_normal_dot_true_normal'])))):.6g} deg`).",f"- Base exact signed distance-to-thickness gap: `{b['gap_f64_m']:.12e} m`; old-normal projection on the independent closest point: `{b['old_normal_signed_gap_m']:.12e} m` (error `{b['old_normal_gap_error_m']*1e9:.6g} nm`). Serialized solver plane-gap field is `{tracedgap:.12e} m`; it is retained separately because its reference offset and float accumulation are not the Euclidean gap.","", "| trial | alpha | feature | FP64 Δdistance (nm) | FP64 Δenergy (J) | old-normal linearized Δgap (nm) | FP64-normal linearized Δgap (nm) | old normal·true normal | fixed-normal gap error (nm) | trace separation (µm) | trace energy (J) |", "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
 for z in trials: md.append(f"| {z['trial']} | {z['alpha']:.9g} | {z['feature']} | {z['delta_distance_from_base_m']*1e9:.6g} | {z['delta_energy_from_base_J']:.6g} | {z['traced_directional_gap_m']*1e9:.6g} | {z['fp64_normal_directional_gap_m']*1e9:.6g} | {z['old_normal_dot_true_normal']:.9g} | {z['old_normal_gap_error_m']*1e9:.6g} | {z['traced_trial_separation_m']*1e6:.9g} | {z['traced_trial_energy_J']:.6g} |")
 md += ["",f"The base IPC energy recomputed in float64 with thickness `{thick:.9g} m` and stiffness `{stiff:.9g}` is `{b['energy_f64_J']:.12e} J` (trace `{c['barrier'][2]:.12e} J`); this is diagnostic only because feature mollifier is not serialized separately.","", "Interpretation is limited to the captured VF states. The FP64-normal linearized gap uses the recorded alpha-scaled relative displacement vector dotted with the independently computed base closest-feature normal. If exact FP64 Δdistance follows that prediction but disagrees with the recorded old-normal directional gap, that demonstrates a force-normal/closest-distance derivative mismatch for that trial. Do not treat this offline check as proof that the repaired mapped path or full native trial passes.",""]
 (HERE/"analysis.md").write_text("\n".join(md)); print(json.dumps({"analysis":str(HERE),"trace_sha256":out["source_trace_sha256"],"base":out["base"],"trial_count":len(trials)},indent=2))
if __name__=="__main__": main()
