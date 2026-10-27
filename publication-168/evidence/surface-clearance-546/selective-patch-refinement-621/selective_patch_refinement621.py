#!/usr/bin/env python3
"""Selective conforming midpoint refinement of one exact crossed-face patch."""
import argparse, hashlib, importlib.util, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
D=ROOT/"D/surface-clearance-546"
OWNER=D/"edge-only-refinement-585/edge_only_refinement585.py"
OWNER_SHA="07d466ba25c7d565bb8be61c9ff87a7725699f13e4080587a48cbbeec24056a3"
ADAPTER=D/"connected-refinement-adapter546/connected_refinement_adapter546.py"
ADAPTER_SHA="06081873eaf0d5452fbf4beee3d580a0eba97c3a546a9f7e7a06a0c9f548cbc6"
P037=ROOT/"full-connected-clearance-150/run-037-local-refinement-006-prep/refine_conforming_006.py"
P037_SHA="d3a6e0c31e34f2165d853ffaf411ad67faaa1d307a3af1a6c79ba41a129f4f31"
P084=Path("/Users/n/Work/numi-human-connected-surface-069/publication-084/canonical/matter/tools/human_skin_fat_boundary_refinement.py")
P084_SHA="eab8deda28164964051a6fa6ff346b8d94bc8bede04e896411dfdd02d46594f6"
NBANK=185

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def rawsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8).tobytes()).hexdigest()

def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()

def edge(a,b):
    a,b=int(a),int(b)
    return (a,b) if a<b else (b,a)

def select_patch(faces, report, max_edges, point_count=None):
    F=np.asarray(faces)
    if F.ndim!=2 or F.shape[1]!=3 or F.dtype.kind not in "iu" or not len(F): raise ValueError("bad face array")
    if F.min(initial=0)<0 or (point_count is not None and F.max(initial=-1)>=point_count): raise ValueError("face node index out of range")
    if isinstance(max_edges,bool) or int(max_edges)<=0: raise ValueError("max_edges must be positive")
    if report.get("complete") is not True or report.get("geometry_or_native_admission") is not False: raise ValueError("report is not complete diagnostic evidence")
    if not isinstance(report.get("banks"),list) or not report["banks"]: raise ValueError("missing bank rows")
    fw=Counter(); bw=defaultdict(Counter); total=0
    for bank in report["banks"]:
        result=bank.get("result")
        if result is None:
            if int(bank.get("skin_aabb_active",-1))==0 and int(bank.get("bank_aabb_active",-1))==0: continue
            raise ValueError("bank skipped despite active AABB candidates")
        if not isinstance(result,dict): raise ValueError("incomplete bank")
        pairs=result.get("triangle_pairs")
        if not isinstance(pairs,list) or int(result.get("count",-1))!=len(pairs): raise ValueError("bank pair count mismatch")
        bk=(int(bank["body"]),int(bank["stable_bone_id"])); seen=set()
        for pair in pairs:
            if not isinstance(pair,(list,tuple)) or len(pair)!=2: raise ValueError("malformed pair")
            f,t=map(int,pair)
            if f<0 or f>=len(F) or t<0: raise ValueError("crossed face or bank triangle out of range")
            if (f,t) in seen: raise ValueError("duplicate typed pair witness")
            seen.add((f,t)); fw[f]+=1; bw[bk][f]+=1; total+=1
    if not fw: raise ValueError("no crossing witnesses")
    ef=defaultdict(list)
    for r in sorted(fw):
        a,b,c=map(int,F[r])
        for u,v in ((a,b),(b,c),(c,a)): ef[edge(u,v)].append(r)
    adj={r:set() for r in fw}
    for rows in ef.values():
        for r in rows: adj[r].update(s for s in rows if s!=r)
    comps=[]; seen=set()
    for start in sorted(fw):
        if start in seen: continue
        seen.add(start); stack=[start]; comp=[]
        while stack:
            r=stack.pop(); comp.append(r)
            for q in sorted(adj[r],reverse=True):
                if q not in seen: seen.add(q); stack.append(q)
        comps.append(tuple(sorted(comp)))
    chosen=sorted(comps,key=lambda c:(-sum(fw[r] for r in c),-len(c),c))[0]
    edges=sorted({edge(u,v) for r in chosen for u,v in ((F[r,0],F[r,1]),(F[r,1],F[r,2]),(F[r,2],F[r,0]))})
    if len(edges)>int(max_edges): raise ValueError("selected patch requires %d edges; cap is %d"%(len(edges),max_edges))
    bc=[]
    for (body,bone),weights in sorted(bw.items()):
        n=sum(weights.get(r,0) for r in chosen)
        if n: bc.append({"body":body,"stable_bone_id":bone,"triangle_pair_count":int(n),"crossed_face_count":int(sum(r in weights for r in chosen))})
    return {"selected_face_rows":list(chosen),"selected_face_count":len(chosen),
        "selected_pair_count":int(sum(fw[r] for r in chosen)),"selected_edges":[list(e) for e in edges],
        "selected_edge_count":len(edges),"patch_bank_pair_counts":bc,"crossing_pair_count":total,
        "crossed_face_count":len(fw),"patch_count":len(comps),
        "patch_score_order":"descending exact pair count, descending face count, ascending face-row tuple",
        "connectivity":"shared undirected mesh edge","selection_is_construction_priority_only":True,
        "unselected_crossed_faces_remain_unresolved":len(chosen)<len(fw),"max_edges":int(max_edges)}

def validate_baseline(state,check,point,surface,selfpairs,csha,candidate):
    if Path(state.get("candidate","")).resolve()!=candidate.resolve() or state.get("candidate_sha256")!=csha: raise ValueError("state candidate pin mismatch")
    if state.get("continuation_mode")!="strict_surface" or state.get("accepted_physical_steps")!=0 or state.get("native_admission") is not False or state.get("temporary_unselected_construction") is True: raise ValueError("not a strict construction-only baseline")
    if check.get("status")!="COMPLETE_EXACT_CONSTRUCTION_CHECKS" or Path(check.get("candidate_path","")).resolve()!=candidate.resolve() or check.get("candidate_sha256")!=csha: raise ValueError("full check mismatch")
    if check.get("arithmetic_pass") is not True or int(check.get("complete_banks",-1))!=NBANK or int(check.get("contained_pairs",-1))!=0 or int(check.get("self_pairs",-1))!=0: raise ValueError("strict construction gates fail")
    if point.get("candidate_sha256")!=csha or point.get("complete_surface_audit") is not True or int(point.get("inside_nodes",-1))!=0 or int(point.get("inside_node_bank_pairs",-1))!=0 or int(point.get("exact_PLC_surface_pairs",-1))!=0: raise ValueError("130 zero-contained/self proof failed")
    if selfpairs.ndim!=2 or selfpairs.shape!=(0,2) or selfpairs.dtype.kind not in "iu": raise ValueError("130 self-pair array not empty Nx2")
    if surface.get("status")!="COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT" or surface.get("candidate_sha256")!=csha or surface.get("complete") is not True or surface.get("geometry_or_native_admission") is not False: raise ValueError("159 mismatch or incomplete")
    if int(surface.get("bank_count",-1))!=NBANK or int(surface.get("expected_bank_count",-1))!=NBANK or len(surface.get("banks",[]))!=NBANK: raise ValueError("not all 185 banks")
    pairs=0
    for b in surface["banks"]:
        r=b.get("result")
        if r is None:
            if int(b.get("skin_aabb_active",-1))==0 and int(b.get("bank_aabb_active",-1))==0: continue
            raise ValueError("159 skipped a bank with active AABB candidates")
        if not isinstance(r,dict) or not isinstance(r.get("triangle_pairs"),list) or int(r.get("count",-1))!=len(r["triangle_pairs"]): raise ValueError("incomplete 159 bank rows")
        pairs+=len(r["triangle_pairs"])
    if pairs!=int(check.get("surface_pairs",-1)): raise ValueError("159/full-check pair count mismatch")
    lin=check.get("current_anchor_lineage")
    if not isinstance(lin,dict) or lin.get("pass") is not True or int(lin.get("anchors",-1))!=237 or lin.get("source_domain")!="R0 kind0 node_source_row" or lin.get("legacy_coarse_boundary_rows_used_as_current_anchors") is not False: raise ValueError("237 current anchor lineage not proven")
    anchors=np.asarray(lin.get("current_output_rows"),dtype=np.int64)
    if anchors.shape!=(237,) or len(np.unique(anchors))!=237: raise ValueError("invalid current anchor rows")
    return anchors

def verify_pins():
    items={"585":(OWNER,"07d466ba25c7d565bb8be61c9ff87a7725699f13e4080587a48cbbeec24056a3"),
           "546":(ADAPTER,ADAPTER_SHA),"037":(P037,P037_SHA),"084":(P084,P084_SHA)}
    out={}
    for n,(p,want) in items.items():
        got=sha(p)
        if got!=want: raise ValueError("pinned source changed: "+n+" "+got)
        out[n]={"path":str(p),"sha256":got}
    return out

def prepare(state_path,check_path,outdir,max_edges=256):
    state_path,check_path=Path(state_path).resolve(),Path(check_path).resolve(); outdir=Path(outdir)
    if outdir.exists(): raise FileExistsError("refuse overwrite "+str(outdir))
    state=json.loads(state_path.read_text()); cand=Path(state["candidate"]).resolve(); csha=sha(cand)
    if state.get("candidate_sha256")!=csha: raise ValueError("candidate SHA differs from state")
    sp,pp=Path(state["surface_report"]).resolve(),Path(state["point_summary"]).resolve()
    for p,k in ((sp,"surface_report_sha256"),(pp,"point_summary_sha256")):
        if not p.is_file() or sha(p)!=state.get(k): raise ValueError("state report pin mismatch "+str(p))
    point,surf,check=json.loads(pp.read_text()),json.loads(sp.read_text()),json.loads(check_path.read_text())
    selfpath=pp.parent/"surface-self-pairs.npy"; selfpairs=np.load(selfpath,allow_pickle=False)
    anchors=validate_baseline(state,check,point,surf,selfpairs,csha,cand); pins=verify_pins()
    with np.load(cand,allow_pickle=False) as z: src={k:z[k].copy() for k in z.files}
    X,F=np.asarray(src.get("points_world_m")),np.asarray(src.get("source_O_faces"))
    if X.ndim!=2 or X.shape[1]!=3 or X.dtype!=np.float32 or F.ndim!=2 or F.shape[1]!=3 or F.dtype.kind not in "iu": raise ValueError("candidate point/face schema mismatch")
    if F.min(initial=0)<0 or F.max(initial=-1)>=len(X) or anchors.min(initial=0)<0 or anchors.max(initial=-1)>=len(X): raise ValueError("node index outside current domain")
    ref=np.asarray(src.get("area_reference_037_input_sha256"))
    if ref.shape!=(1,) or ref.dtype.kind not in "US" or not re.fullmatch(r"[0-9a-f]{64}",str(ref[0])): raise ValueError("missing or invalid 037 reference pin")
    sel=select_patch(F,surf,max_edges,point_count=len(X)); edges=np.asarray(sel["selected_edges"],dtype=np.int64)
    spec=importlib.util.spec_from_file_location("edge_refinement585_for621",OWNER)
    if spec is None or spec.loader is None: raise ImportError("cannot load pinned 585 owner")
    owner=importlib.util.module_from_spec(spec); sys.modules[spec.name]=owner; spec.loader.exec_module(owner)
    rp,rf=src.get("area_reference_037_points_world_m_f64"),src.get("area_reference_037_source_O_faces")
    if rp is None or rf is None: raise ValueError("missing matched 037 arrays")
    arrays,built=owner.build_refined_arrays585(src,surf,rp,rf,candidate_sha256=csha,
        crossing_report_sha256=sha(sp),area_reference_sha256=str(ref[0]),selected_edges_override=edges,selection_policy=sel)
    Xo=np.asarray(arrays["points_world_m"]); suffix=np.arange(len(X),len(Xo),dtype=np.int64)
    if not same(Xo[:len(X)],X) or not same(Xo[anchors],X[anchors]): raise AssertionError("existing points/anchors changed")
    actual=np.sort(np.asarray(arrays["refinement_node_parent_edge_rows"])[suffix].astype(np.int64),axis=1)
    if len(suffix)!=len(edges) or not np.array_equal(actual,edges): raise AssertionError("new midpoint ancestry differs from selected edges")
    kinds=np.asarray(arrays["refinement_node_kind"])[suffix]
    if not np.array_equal(kinds,np.ones(len(suffix),dtype=kinds.dtype)): raise AssertionError("non-midpoint node in output suffix")
    outdir.mkdir(parents=True,exist_ok=False); cp=outdir/"candidate.npz"; rpout=outdir/"report.json"
    with cp.open("xb") as f: np.savez_compressed(f,**arrays)
    result=dict(built)
    midpoint_rows=np.arange(len(X),len(Xo),dtype="<i8")
    parent_edges=np.asarray(arrays["refinement_node_parent_edge_rows"])[suffix]
    result.update({
        "parent_prefix_node_count":int(len(X)),"input_points":int(len(X)),
        "parent_prefix_points_bytes_sha256":rawsha(X),
        "edge_midpoint_rows":midpoint_rows.tolist(),"edge_midpoint_rows_count":int(len(midpoint_rows)),
        "edge_midpoint_rows_sha256":rawsha(midpoint_rows),
        "edge_midpoint_rows_sha256_encoding":"sha256 of C-contiguous little-endian int64 row IDs as raw bytes",
        "edge_midpoint_parent_edges_sha256":rawsha(parent_edges),
        "added_nodes_are_only_edge_midpoints":True,"added_face_centroids":0,
        "input_state_path":str(state_path),"input_state_sha256":sha(state_path),
        "input_candidate_path":str(cand),"input_candidate_sha256":csha,
        "input_construction_report_path":str(check_path),"input_construction_report_sha256":sha(check_path),
        "input_point_summary_path":str(pp),"input_point_summary_sha256":sha(pp),
        "input_surface_report_path":str(sp),"input_surface_report_sha256":sha(sp),
        "input_self_pairs_path":str(selfpath),"input_self_pairs_sha256":sha(selfpath),
        "current_anchor_rows":anchors.tolist(),"current_anchor_count":len(anchors),
        "original_point_prefix_bytes_sha256":rawsha(X),"original_point_prefix_preserved_bitwise":True,
        "anchor_positions_preserved_bitwise":True,"selected_edge_subset_policy":sel,
        "output_candidate_path":str(cp),"output_candidate_sha256":sha(cp),"output_report_path":str(rpout),
        "source_pins":pins,"geometry_moved":False,"geometry_admitted":False,
        "containment_or_clearance_claim":False,"native_admission":False,"accepted_physical_steps":0,
        "requires_fresh_130_159_and_545_checks":True})
    rpout.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return result

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--state",type=Path,required=True)
    ap.add_argument("--construction-report",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--max-edges",type=int,default=256); a=ap.parse_args(argv)
    r=prepare(a.state,a.construction_report,a.out,a.max_edges)
    print(json.dumps({"status":r.get("status"),"input_candidate_sha256":r["input_candidate_sha256"],
      "selected_face_count":r["selected_edge_subset_policy"]["selected_face_count"],
      "selected_pair_count":r["selected_edge_subset_policy"]["selected_pair_count"],
      "selected_edge_count":r["selected_edge_subset_policy"]["selected_edge_count"],
      "output_candidate_sha256":r["output_candidate_sha256"],"geometry_admitted":False,
      "accepted_physical_steps":0},sort_keys=True),flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())
