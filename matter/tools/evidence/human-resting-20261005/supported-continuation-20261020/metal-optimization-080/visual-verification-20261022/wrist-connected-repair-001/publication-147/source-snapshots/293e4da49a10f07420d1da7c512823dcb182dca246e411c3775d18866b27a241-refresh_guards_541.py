#!/usr/bin/env python3
"""Rebuild candidate-bound 026 self seeds and 313 fixed-bone guards for topology 541."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
D=ROOT/"D"
OWNER026=ROOT/"publication-101/evidence/204/source/fit/self_separation_026.py"
OWNER313=ROOT/"publication-113/sources/connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py"
OWNER314=ROOT/"publication-113/sources/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py"
OWNER401=D/"retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py"
PIN={"026":"c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3",
"313":"dc9d4f953ecede00567c59cbbb6b6b63e0a3556b97972689de499cbce29d4c2e",
"314":"3422f30a61db0724d234c487d70fc3d74237a802261c2f9a72af9e15843a1c78",
"401":"2f1eda3f0d46d0f5497a1a04075b5dc2c3b37afe906d0389ff636fca8abc5869"}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(x): return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
def loadmod(n,p):
 s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def loadnpz(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]).copy() for k in z.files}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def pair_set(report):
 out=set()
 for b in report["banks"]:
  for f,t in ((b.get("result")or{}).get("triangle_pairs")or[]):
   k=(int(b["body"]),int(b["stable_bone_id"]),int(f),int(t))
   if k in out:raise ValueError(f"duplicate exact 159 identity {k}")
   out.add(k)
 return out
def face_index(faces):
 out={}
 for i,f in enumerate(np.asarray(faces)):out.setdefault(tuple(sorted(map(int,f))),[]).append(i)
 return out
def map_pairs(source_faces,dest_faces,pairs):
 ix=face_index(dest_faces); out=[]
 a=np.asarray(source_faces);p=np.asarray(pairs)
 if p.ndim!=2 or p.shape[1]!=2 or p.dtype.kind not in "iu":raise ValueError("self pairs must be integer (K,2)")
 for i,j in p:
  if min(int(i),int(j))<0 or max(int(i),int(j))>=len(a) or i==j:raise ValueError("invalid source face-pair")
  rows=[]
  for old in (int(i),int(j)):
   hits=ix.get(tuple(sorted(map(int,a[old]))),[])
   if len(hits)!=1:raise ValueError(f"face {old} has {len(hits)} unchanged-geometry matches")
   rows.append(hits[0])
  if rows[0]==rows[1]:raise ValueError("two seed faces map to one face")
  out.append(sorted(rows))
 q=np.asarray(sorted(set(map(tuple,out))),np.int64).reshape(-1,2)
 if len(q)!=len(p):raise ValueError("face-pair remap collapsed distinct identities")
 return q
def pack(rows):
 ptr=np.zeros(len(rows)+1,np.int64); nodes=[];coeff=[];lower=[]
 for i,r in enumerate(rows):
  n=np.asarray(r["nodes"],np.int64).reshape(-1);c=np.asarray(r["coefficients"],np.float64)
  if c.shape!=(len(n),3):raise ValueError("malformed 026 row")
  nodes.extend(n.tolist());coeff.extend(c.tolist());lower.append(float(r["lower"]));ptr[i+1]=len(nodes)
 return {"indptr":ptr,"nodes":np.asarray(nodes,np.int64),"coefficients":np.asarray(coeff,np.float64).reshape(-1,3),"lower":np.asarray(lower,np.float64)}
def validate_ancestry(candidate, parent_candidate, surface):
 F=np.asarray(candidate["source_O_faces"],np.int64); n=len(F)
 parent_faces=np.asarray(parent_candidate["source_O_faces"],np.int64)
 ptr=np.asarray(candidate["refinement_face_parent_ancestry_indptr"],np.int64)
 vals=np.asarray(candidate["refinement_face_parent_ancestry_face_rows"],np.int64)
 dptr=np.asarray(candidate["face_fine_detail_ancestry_indptr"],np.int64)
 dvals=np.asarray(candidate["face_fine_detail_ancestry_rows"],np.int64)
 kind=np.asarray(candidate["face_kind"]); fine=np.asarray(candidate["parent_fine_detail_face_row"],np.int64)
 if ptr.shape!=(n+1,) or ptr[0]!=0 or ptr[-1]!=len(vals) or np.any(ptr[1:]<ptr[:-1]): raise ValueError("refinement-face ancestry CSR malformed")
 if np.any(vals<0) or np.any(vals>=len(parent_faces)): raise ValueError("multi-parent CSR index outside pinned parent face domain")
 if dptr.shape!=(n+1,) or dptr[0]!=0 or dptr[-1]!=len(dvals) or np.any(dptr[1:]<dptr[:-1]): raise ValueError("fine-detail ancestry CSR malformed")
 crossed=sorted({int(f) for b in surface["banks"] for f,t in ((b.get("result")or{}).get("triangle_pairs")or[])})
 lens=[]; multi=0
 for f in crossed:
  lo,hi=int(dptr[f]),int(dptr[f+1])
  if int(kind[f])!=2 or hi-lo!=1 or int(dvals[lo])!=int(fine[f]): raise ValueError(f"crossing face {f} lacks one matching registered fine-detail ancestry")
  lens.append(hi-lo); multi+=int(ptr[f+1]-ptr[f]>1)
 return {"parent_candidate_path":str(parent_candidate.get("candidate_path","provided")),"parent_face_count":len(parent_faces),
  "current_face_count":n,"refinement_parent_csr_rows":len(vals),"multi_parent_current_face_count":int(np.count_nonzero(np.diff(ptr)>1)),
  "crossing_face_count":len(crossed),"crossing_faces_with_multi_parent_csr":multi,
  "crossing_face_kind_counts":{"2":len(crossed)},"crossing_fine_detail_ancestry_unique_and_matching":True,
  "parent_indices_in_range":True}

def validate130(path,pair_path,cand_sha,npoint,nface):
 r=json.loads(Path(path).read_text());p=np.load(pair_path,allow_pickle=False)
 if r.get("candidate_sha256")!=cand_sha or r.get("complete_surface_audit") is not True or int(r.get("exact_PLC_surface_pairs",-1))!=len(p) or int(r.get("points",-1))!=npoint or int(r.get("faces",-1))!=nface or r.get("geometry_admitted") is not False:
  raise ValueError("130 audit/pair payload is stale or incomplete")
 if p.ndim!=2 or p.shape[1]!=2 or p.dtype.kind not in "iu":raise ValueError("130 pair payload malformed")
 return r,p
def main():
 ap=argparse.ArgumentParser()
 for k in ("candidate","audit130","audit159","query-dir","ancestry-parent-candidate","seed-source-candidate","seed-source-audit130","seed-source-pairs","prior-guard-candidate","prior-guard-report","prior-guard-npz","output-dir"):
  ap.add_argument("--"+k,required=True,type=Path)
 for k in ("expected-candidate-sha256","expected-audit130-sha256","expected-audit159-sha256"):
  ap.add_argument("--"+k,required=True)
 a=ap.parse_args()
 c=a.candidate.resolve(); a130=a.audit130.resolve();a159=a.audit159.resolve(); qd=a.query_dir.resolve();out=a.output_dir.resolve()
 if out.exists():raise FileExistsError(out)
 files={"026":OWNER026,"313":OWNER313,"314":OWNER314,"401":OWNER401}
 actual={k:sha(v) for k,v in files.items()}
 if actual!=PIN:raise ValueError(f"owner pin mismatch: {actual}")
 csha=sha(c)
 if csha!=a.expected_candidate_sha256 or sha(a130)!=a.expected_audit130_sha256 or sha(a159)!=a.expected_audit159_sha256:raise ValueError("candidate or exact-audit hash mismatch")
 z=loadnpz(c);F=z["source_O_faces"];P=z["points_world_m"]
 ancestry_parent=loadnpz(a.ancestry_parent_candidate.resolve())
 current_a130,current_pairs=validate130(a130,a130.parent/"surface-self-pairs.npy",csha,len(P),len(F))
 current_surface=json.loads(a159.read_text())
 if (current_surface.get("candidate_sha256")!=csha or current_surface.get("complete") is not True
    or current_surface.get("status")!="COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT"
    or int(current_surface.get("bank_count",-1))!=185 or int(current_surface.get("expected_bank_count",-1))!=185
    or int(current_surface.get("surface_point_count",-1))!=len(P) or int(current_surface.get("surface_face_count",-1))!=len(F)):
  raise ValueError("159 report is stale or incomplete for current candidate")
 active=pair_set(current_surface)
 ancestry_receipt=validate_ancestry(z,ancestry_parent,current_surface)
 # Validate current query receipt is from the exact point field.
 receipt=json.loads((qd/"query-receipt.json").read_text())
 qnpz=loadnpz(qd/"points-world.npz")
 if not np.array_equal(qnpz["points_world_f32"],P) or receipt.get("identity_domain")!="mechanics_node_row" or int(receipt.get("physical_steps",-1))!=0:
  raise ValueError("130 query receipt/point bytes do not match current candidate")
 for n in ("points","pairs","nodes"):
  file=qd/(n+".csv")
  if not file.is_file() or sha(file)!=receipt.get(n+"_sha256"):raise ValueError(f"query CSV hash mismatch {n}")
 out.mkdir(parents=True);sdir=out/"self-guards";sdir.mkdir()
 sat=loadmod("self_sep_026_541",OWNER026)
 # Current exact-self pairs are regenerated from this topology's matched 130.
 np.save(sdir/"current-self-pairs.npy",current_pairs)
 current_rows,current_records=sat.build_pair_rows(P.astype(np.float64)*1000.0,F,current_pairs,margin_fraction=.25)
 current_pack=pack(current_rows);current_rhs=max((float(r["lower"]) for r in current_rows),default=0.0)
 if current_rhs>2e-8:raise ValueError("current self rows fail zero-step certificate")
 np.savez_compressed(sdir/"current-self-rows.npz",**current_pack)
 # Import 9 saved-trial pairs only through exact unchanged triangle identities.
 seedc=a.seed_source_candidate.resolve();seed_audit=a.seed_source_audit130.resolve();seedpairs=a.seed_source_pairs.resolve()
 seedz=loadnpz(seedc);seedsha=sha(seedc)
 seed_summary,source_pairs=validate130(seed_audit,seedpairs,seedsha,len(seedz["points_world_m"]),len(seedz["source_O_faces"]))
 mapped=map_pairs(seedz["source_O_faces"],F,source_pairs)
 mapped_rows,mapped_records=sat.build_pair_rows(P.astype(np.float64)*1000.0,F,mapped,margin_fraction=.25)
 mapped_pack=pack(mapped_rows);max_rhs=max((float(r["lower"]) for r in mapped_rows),default=0.0)
 if max_rhs>2e-8:raise ValueError(f"mapped saved-trial 026 rows fail zero-step certificate: {max_rhs}")
 seed_max_rhs=max_rhs
 mapped_path=sdir/"mapped-trial-self-pairs.npy";np.save(mapped_path,mapped)
 np.savez_compressed(sdir/"mapped-trial-self-rows.npz",**mapped_pack)
 seed_report={"schema":"numi.human.continuation-self-seeds.v1","status":"PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY",
  "candidate_path":str(c),"candidate_sha256":csha,"source_trial_candidate_path":str(seedc),"source_trial_candidate_sha256":seedsha,
  "source_trial_audit130_path":str(seed_audit),"source_trial_audit130_sha256":sha(seed_audit),
  "source_trial_pairs_path":str(seedpairs),"source_trial_pairs_sha256":sha(seedpairs),
  "mapping":"unique exact unordered node-triangle identity only; never row number alone",
  "source_pair_count":len(source_pairs),"mapped_unique_pair_count":len(mapped),"merged_unique_pair_count":len(mapped),
  "zero_step_max_rhs_mm":max_rhs,"zero_step_feasible":True,"output_pairs_path":str(mapped_path),
  "output_pairs_sha256":sha(mapped_path),"pair_owner_path":str(OWNER026),"pair_owner_sha256":actual["026"],
  "construction_only":True,"native_admission":False}
 write(sdir/"mapped-trial-self-seeds-report.json",seed_report)
 # Map old 314 guards by unchanged face geometry, then remove exact active current crossings.
 prev=a.prior_guard_candidate.resolve();prp=a.prior_guard_report.resolve();pnp=a.prior_guard_npz.resolve()
 prev_sha=sha(prev);prevz=loadnpz(prev);prevreport=json.loads(prp.read_text());oldarr=loadnpz(pnp)
 if prevreport.get("base_candidate_sha256")!=prev_sha or prevreport.get("output_npz_sha256")!=sha(pnp):raise ValueError("prior 314 guard receipt mismatch")
 val=loadmod("fixed_boundary_validator_401_541",OWNER401)
 oldrecords,_=val.validate_fixed_boundary_payload(oldarr,prevreport,candidate_sha256=prev_sha,
  candidate_points=prevz["points_world_m"],candidate_faces=prevz["source_O_faces"],npz_path=pnp)
 ix=face_index(F);mappedset=set();face_matched=0;face_missing=0;face_ambiguous=0;excluded=0
 for rec in oldrecords:
  old=int(rec["skin_face_row"]);hits=ix.get(tuple(sorted(map(int,prevz["source_O_faces"][old]))),[])
  if len(hits)!=1:
   if hits:face_ambiguous+=1
   else:face_missing+=1
   continue
  face_matched+=1;key=(int(rec["bank_body"]),int(rec["stable_bone_id"]),int(hits[0]),int(rec["bank_triangle_stable_id"]))
  if key in active:excluded+=1
  else:mappedset.add(key)
 exporter=loadmod("fixed_boundary_exporter_314_541",OWNER314)
 bank_reader=loadmod("bank_reader_084_541",exporter.BANK_HELPER)
 bank_reader_sha=sha(exporter.BANK_HELPER)
 if prevreport.get("bank_reader_sha256")!=bank_reader_sha:raise ValueError("pinned bank reader differs from prior guard report")
 tris,ids,_g,_b,matrices,offsets,positions,bankkeys,bankpins,_=bank_reader.read_bank_inputs(exporter.BANK_DIR,exporter.BANK_EVIDENCE)
 if len(bankkeys)!=185 or len(set(tuple(map(int,x)) for x in bankkeys))!=185:raise ValueError("314 reader did not return 185 typed banks")
 trans={int(body):(matrices[int(body)],offsets[int(body)],positions[int(body)]) for body in matrices}
 arrays=exporter.make_barrier_arrays(sorted(mappedset),exporter.typed_triangle_lookup(tris,ids),trans)
 gdir=out/"fixed-guards";gdir.mkdir()
 gn=gdir/"fixed-skin-bone-guards.npz";np.savez_compressed(gn,**arrays)
 typed=list(zip(arrays["skin_face_rows"].tolist(),arrays["bank_body"].tolist(),arrays["stable_bone_id"].tolist(),arrays["bank_triangle_stable_id"].tolist()))
 tr=[{"skin_face_row":int(f),"bank_body":int(b),"stable_bone_id":int(bo),"bank_triangle_stable_id":int(t)} for f,b,bo,t in sorted(typed,key=lambda x:(x[1],x[2],x[0],x[3]))]
 greport={"schema":"numi.fixed_skin_bone_barriers.v1","status":"COMPLETE_TYPED_INTRODUCED_SKIN_BONE_BARRIER_EXPORT",
  "filtered_builder_path":str(Path(__file__).resolve()),"filtered_builder_sha256":sha(Path(__file__)),
  "base_candidate_path":str(c),"base_candidate_sha256":csha,"candidate_points_world_m_array_sha256":arrsha(P),
  "skin_faces_array_sha256":arrsha(F),"point_count":len(P),"face_count":len(F),"bank_count":185,"expected_bank_count":185,
  "fixed_triangles_units":"mm","fixed_triangles_frame":"world","geometry_or_native_admission":False,
  "bank_reader_path":str(exporter.BANK_HELPER),"bank_reader_sha256":bank_reader_sha,
  "bank_source_pins":bankpins,
  "output_npz":gn.name,"output_npz_sha256":sha(gn),"introduced_pair_count":len(tr),"typed_pairs":tr,
  "inputs":[{"role":"base","candidate_path":str(c),"candidate_sha256":csha,"audit159_path":str(a159),"audit159_sha256":sha(a159)}],
  "mapping_source":{"prior_candidate_path":str(prev),"prior_candidate_sha256":prev_sha,"prior_guard_report_sha256":sha(prp),
   "prior_guard_npz_sha256":sha(pnp),"exact_face_matches":face_matched,"unmapped_faces":face_missing,
   "ambiguous_face_matches":face_ambiguous,"active_current_crossings_excluded":excluded},
  "limitations":["Only exact unchanged node-triangle identities were mapped; unmapped faces were excluded.",
   "Current exact 159 crossings were excluded from fixed guards.","Construction only; exact endpoint audit is still required."]}
 gr=gdir/"report.json";write(gr,greport)
 records,_=val.validate_fixed_boundary_payload(arrays,greport,candidate_sha256=csha,candidate_points=P,candidate_faces=F,npz_path=gn)
 rows_owner=loadmod("fixed_boundary_rows_313_541",OWNER313)
 hard,typed_rows=val.build_fixed_boundary_barriers(P.astype(np.float64)*1000.0,F.astype(np.int64),records,sat,rows_owner,margin_fraction=.25)
 max_rhs=max((float(r["lower"]) for r in hard),default=0.0)
 if max_rhs>2e-8:raise ValueError(f"rebuilt 313 guards fail zero-step certificate: {max_rhs}")
 gpack=pack(hard);rp=gdir/"fixed-guard-313-rows.npz";np.savez_compressed(rp,**gpack)
 validation={"status":"PASS_401_FIXED_BOUNDARY_PAYLOAD_AND_313_ROW_VALIDATION","candidate_sha256":csha,
  "npz_sha256":sha(gn),"report_sha256":sha(gr),"filtered_guard_pairs":len(typed_rows),
  "313_rows":len(hard),"all_rows_zero_feasible":True,"maximum_zero_step_rhs_mm":max_rhs,
  "exact_face_matches":face_matched,"unmapped_faces":face_missing,"ambiguous_face_matches":face_ambiguous,
  "active_current_crossings_excluded":excluded}
 vp=gdir/"validation.json";write(vp,validation)
 result={"schema":"numi.human.focused-clearance-541.current-constraints.v1","status":"PASS_CONSTRUCTION_ROWS_ONLY",
  "candidate_path":str(c),"candidate_sha256":csha,"audit130_path":str(a130),"audit130_sha256":sha(a130),
  "audit159_path":str(a159),"audit159_sha256":sha(a159),"current_exact_self_pairs":len(current_pairs),
  "current_exact_bone_crossing_pairs":len(active),"current_face_ancestry":ancestry_receipt,
  "mapped_trial_seed_pairs":len(mapped),"mapped_trial_seed_rows":len(mapped_rows),"mapped_trial_seed_max_rhs_mm":seed_max_rhs,
  "fixed_guard_source_pairs":len(oldrecords),"mapped_guard_faces":face_matched,"unmapped_guard_faces":face_missing,
  "ambiguous_guard_faces":face_ambiguous,"active_current_crossings_excluded":excluded,
  "fixed_guard_pair_count":len(typed_rows),"fixed_313_row_count":len(hard),"fixed_313_max_rhs_mm":max_rhs,
  "seed_pairs_path":str(mapped_path),"seed_pairs_sha256":sha(mapped_path),
  "seed_report_path":str(sdir/"mapped-trial-self-seeds-report.json"),"seed_report_sha256":sha(sdir/"mapped-trial-self-seeds-report.json"),
  "fixed_guard_npz_path":str(gn),"fixed_guard_npz_sha256":sha(gn),"fixed_guard_report_path":str(gr),
  "fixed_guard_report_sha256":sha(gr),"fixed_313_rows_path":str(rp),"fixed_313_rows_sha256":sha(rp),
  "fixed_guard_validation_path":str(vp),"fixed_guard_validation_sha256":sha(vp),
  "source_pins":actual,"accepted_physical_steps":0,"native_admission":False}
 write(out/"refresh-report.json",result);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
