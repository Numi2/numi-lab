"""Validated consumer adapter for component_sat_selector638 -> frozen 637."""
from __future__ import annotations
import hashlib, importlib.util, json, os, sys
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
HERE=Path(__file__).resolve().parent
SAT_PATH=D/"D/surface-clearance-546/pair-sat-routing-551/pair_sat_routing_551.py"
SAT_SHA="27431e9ed3e92886394b59f1156a743e508fba8c81cf8b1ff94a7d80e3ef201f"
BOX_PATH=D/"D/surface-clearance-546/directional-source-bounds-636/directional_bounds636_v2.py"
BOX_SHA="fb3e9cda895b086cf2f5f15ca28f35b43e84c912c73db443e3c7d04cacbd10a8"
RUN603_PATH=D/"D/surface-clearance-546/component-scp-area-backtrack-603/frozen-v1/run_joint_full_coordinate_fit603.py"
RUN603_SHA="38c3dfa9ddd961250111c87f56d6184075bde43be07c46e09d239f057b06d692"
RUN637_PATH=D/"D/surface-clearance-546/box-aware-sat-fit-637/run_box_aware_sat637.py"
RUN637_SHA="3a142578939683981274ebef061d797b471594ce6694cd888b0e450bbaf6bff2"
HELPER637_PATH=D/"D/surface-clearance-546/box-aware-sat-fit-637/box_aware_sat637.py"
HELPER637_SHA="a226a8f7e5b4afaeeb9d484883f6a213b84220bb639a15388f3c37dd944069bd"
OWNER408_PATH=D/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
OWNER408_SHA="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
RUNNER458_PATH=D/"D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py"
RUNNER458_SHA="08cd0e2681bb5ae16a85a0384514bb30739f8d62a06ea2ab1bdfcd896be4ef93"
SELECTOR_HELPER_PATH=HERE/"connected_patch_sat_selector638.py"
SELECTOR_HELPER_SHA="a188e5a55d2aebb0ce27848b0d3fb03d7aa2f8715ee6a8da7bbc57d781485a9a"
SELECTOR_RUNNER_PATH=HERE/"run_connected_patch_sat_selector638.py"
SELECTOR_RUNNER_SHA="bbaf1ca9fef90f370c5a877dd1f0249779f0091a4dad8013a581820d110ed92f"
TOL=3e-7
INSTALLS=[]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def array_sha(a,dtype):
    return hashlib.sha256(np.ascontiguousarray(a,dtype=dtype).tobytes()).hexdigest()

def load_module(name,path,digest):
    if sha(path)!=digest: raise RuntimeError(f"pinned source changed: {path}")
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def arg_value(argv,flag):
    if argv.count(flag)!=1: raise ValueError(f"expected exactly one {flag}")
    i=argv.index(flag)
    if i+1>=len(argv): raise ValueError(f"missing value for {flag}")
    return Path(argv[i+1]).resolve()


def captured_fit_argv(environ=None):
    """Recover the original fit args saved by the outer launcher before 490 parsing."""
    env=os.environ if environ is None else environ
    raw=env.get("NUMI_AXIS_MAP638_ORIGINAL_FIT_ARGV_JSON")
    if not raw:raise ValueError("outer launcher did not preserve original fit argv")
    argv=json.loads(raw)
    if not isinstance(argv,list) or any(not isinstance(x,str) for x in argv):
        raise ValueError("saved original fit argv has invalid schema")
    if "--selector-npz" in argv or "--selector-report" in argv:
        raise ValueError("adapter-only flags must not enter original 603 argv")
    return argv

def validate_selector_arrays(arrays, report, expected_identity, expected_nodes, expected_inputs,
                             scope_sha256, lower_sha256, upper_sha256):
    """Validate selection against exact prepared pair order and runtime box pins."""
    if report.get("schema")!="numi.connected_patch_sat_selector638.result.v1":
        raise ValueError("unsupported selector result schema")
    pre=report.get("preflight")
    if not isinstance(pre,dict) or pre.get("schema")!="numi.connected_patch_sat_selector638.preflight.v1":
        raise ValueError("missing selector preflight provenance")
    inp=pre.get("inputs",{})
    for role,expected in expected_inputs.items():
        got=inp.get(role)
        if not isinstance(got,dict) or got.get("sha256")!=expected["sha256"] or Path(got.get("path","")).resolve()!=Path(expected["path"]).resolve():
            raise ValueError(f"selector input mismatch for {role}")
    if inp.get("basis_scope_rows_sha256")!=scope_sha256:
        raise ValueError("selector 408 ordered scope hash differs")
    if inp.get("directional636_lower_sha256")!=lower_sha256:
        raise ValueError("selector 636 lower-bound hash differs")
    if inp.get("directional636_upper_sha256")!=upper_sha256:
        raise ValueError("selector 636 upper-bound hash differs")
    ident=np.asarray(arrays["pair_identity"],dtype=np.int64)
    nodes=np.asarray(arrays["pair_nodes"],dtype=np.int64)
    selected=np.asarray(arrays["selected"],dtype=bool)
    n=len(np.asarray(expected_identity))
    if ident.shape!=(n,4) or nodes.shape!=(n,3) or selected.shape!=(n,):
        raise ValueError("selector arrays have wrong pair inventory shape")
    if not np.array_equal(ident,np.asarray(expected_identity,dtype=np.int64)):
        raise ValueError("selector exact typed-pair identity/order differs")
    if not np.array_equal(nodes,np.asarray(expected_nodes,dtype=np.int64)):
        raise ValueError("selector skin node identity/order differs")
    if len(np.unique(ident,axis=0))!=n:
        raise ValueError("duplicate typed pair identity")
    if np.any(nodes[:,0]==nodes[:,1]) or np.any(nodes[:,0]==nodes[:,2]) or np.any(nodes[:,1]==nodes[:,2]):
        raise ValueError("a skin triangle row must contain three distinct node IDs")
    if int(report.get("selected_pair_count",-1))!=int(selected.sum()):
        raise ValueError("selected count differs from selector report")
    if int(pre.get("pair_count",-1))!=n:
        raise ValueError("preflight pair count differs")
    normals=np.asarray(arrays["selected_normal_world_unit"],dtype=np.float64)
    source=np.asarray(arrays["selected_axis_source"]).astype(str)
    axis=np.asarray(arrays["selected_axis_index"],dtype=np.int64)
    sign=np.asarray(arrays["selected_axis_sign"],dtype=np.int64)
    source_order=np.asarray(arrays["selected_axis_source_order"],dtype=np.int64)
    sign_order=np.asarray(arrays["selected_axis_sign_order"],dtype=np.int64)
    extra=np.asarray(arrays["selected_axis_extra_sign"],dtype=np.int64)
    if normals.shape!=(n,3) or any(a.shape!=(n,) for a in (source,axis,sign,source_order,sign_order,extra)):
        raise ValueError("selected axis metadata shape mismatch")
    if not np.isfinite(normals[selected]).all() or np.any(np.abs(np.linalg.norm(normals[selected],axis=1)-1.0)>2e-8):
        raise ValueError("selected normals are non-finite or not unit")
    if np.any(axis[selected]<0) or np.any(np.isin(source[selected],[""])):
        raise ValueError("selected axis identity is incomplete")
    drows=np.asarray(arrays["displacement_node_rows"],dtype=np.int64)
    dvals=np.asarray(arrays["displacement_mm"],dtype=np.float64)
    if drows.ndim!=1 or dvals.shape!=(len(drows),3) or len(np.unique(drows))!=len(drows) or not np.isfinite(dvals).all():
        raise ValueError("selector displacement witness malformed")
    if int(report.get("selected_pair_count",0)) and not len(drows):
        raise ValueError("selected axes lack displacement witnesses")
    displacement_rows=set(map(int,drows.tolist()))
    for i in np.flatnonzero(selected):
        if any(int(node) not in displacement_rows for node in nodes[i]):
            raise ValueError("selected pair lacks a displacement for one of its shared nodes")
    return {"identity":ident,"nodes":nodes,"selected":selected,"normals":normals,
            "axis_source":source,"axis_index":axis,"axis_sign":sign,
            "axis_source_order":source_order,"axis_sign_order":sign_order,"axis_extra_sign":extra,
            "displacement_by_node":{int(k):dvals[i].copy() for i,k in enumerate(drows)}}

def validate_selector_source_pins(source_map):
    expected={str(SAT_PATH):SAT_SHA,str(BOX_PATH):BOX_SHA,str(OWNER408_PATH):OWNER408_SHA,
        str(RUNNER458_PATH):RUNNER458_SHA,str(SELECTOR_HELPER_PATH):SELECTOR_HELPER_SHA,
        str(SELECTOR_RUNNER_PATH):SELECTOR_RUNNER_SHA}
    if not isinstance(source_map,dict):
        raise ValueError("selector report lacks source pins")
    for path,digest in expected.items():
        if source_map.get(path)!=digest:
            raise ValueError(f"selector source pin mismatch: {path}")
    for path,digest in expected.items():
        if sha(path)!=digest:
            raise RuntimeError(f"local selector source changed: {path}")
    return expected

def match_axis_record(axes, meta, normal):
    """Return the unique 551 axis matching all stored identity fields and normal."""
    wanted=(str(meta["source"]),int(meta["index"]),int(meta["sign"]),
            int(meta["source_order"]),int(meta["sign_order"]),int(meta["extra_sign"]))
    found=[]
    for a in axes:
        got=(str(a["axis_source"]),int(a["axis_index"]),int(a["axis_sign"]),
             int(a["axis_source_order"]),int(a["axis_sign_order"]),int(a["extra_sign"]))
        if got==wanted and np.allclose(np.asarray(a["normal_world_unit"],dtype=np.float64),
                                      np.asarray(normal,dtype=np.float64),rtol=0,atol=2e-12):
            found.append(a)
    if len(found)!=1:
        raise ValueError("selected axis is not unique in the pinned 551 inventory")
    return found[0]

def route_pair_record(selected, fallback_factory, selected_factory):
    """Use frozen 637's exact fallback for unselected rows."""
    if not selected:
        rec=dict(fallback_factory())
        rec.update({"axis_strategy":"finite_pair_sat_box_aware_soft_target_637_fallback",
                    "selector_pair_selected":False,"selector_displacement_applied_to_fit":False})
        return rec
    rec=dict(selected_factory())
    rec["selector_pair_selected"]=True
    return rec

def _tri_key(nodes,skin,bone):
    return (tuple(map(int,np.asarray(nodes,dtype=np.int64))),
            np.ascontiguousarray(skin,dtype="<f8").tobytes(),
            np.ascontiguousarray(bone,dtype="<f8").tobytes())


def make_pair_queue_template(pair_identity,pair_nodes,points,bone_triangles):
    template=defaultdict(list)
    for i,(ident,nodes) in enumerate(zip(np.asarray(pair_identity),np.asarray(pair_nodes))):
        face_row,body,bone,tri=map(int,ident)
        skin=np.asarray(points,dtype=np.float64)[nodes]
        bone_tri=np.asarray(bone_triangles[(body,bone,tri)],dtype=np.float64)
        template[_tri_key(nodes,skin,bone_tri)].append(i)
    return {key:tuple(indices) for key,indices in template.items()}

def fresh_pair_queues(template):
    return defaultdict(deque,{key:deque(indices) for key,indices in template.items()})

def _record_from_selected(skin,bone,nodes,support,lower,upper,normal,axis_meta,delta_by_node,sat,axis_count):
    skin,bone=sat._validated_triangles(skin,bone)
    n=np.asarray(normal,dtype=np.float64)
    if n.shape!=(3,) or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1.0)>2e-8:
        raise ValueError("selected SAT normal invalid")
    nodes=np.asarray(nodes,dtype=np.int64);scope=np.asarray(support,dtype=np.int64)
    lo=np.asarray(lower,dtype=np.float64).reshape(-1,3);hi=np.asarray(upper,dtype=np.float64).reshape(-1,3)
    if lo.shape!=hi.shape or lo.shape!=(len(scope),3):raise ValueError("runtime 636 bounds shape mismatch")
    pos={int(node):i for i,node in enumerate(scope)}
    rhs=float(np.max(bone@n))+float(sat._float32_world_m_emission_margin_mm(skin,bone,n))-skin@n
    maxmove=[];unmet=[];witness_slack=[]
    for j,node in enumerate(nodes):
        if int(node) not in pos:raise ValueError("selected pair node missing from runtime bounds")
        r=pos[int(node)]
        maxmove.append(float(np.sum(np.where(n>=0,n*hi[r],n*lo[r]))))
        unmet.append(float(max(0.,rhs[j]-maxmove[-1])))
        if int(node) not in delta_by_node:raise ValueError(f"selected pair node {node} missing displacement witness")
        d=np.asarray(delta_by_node[int(node)],dtype=np.float64)
        if np.any(d<lo[r]-TOL) or np.any(d>hi[r]+TOL):
            raise ValueError(f"selector witness displacement outside 636 bounds at node {node}")
        witness_slack.append(float(np.dot(n,d)-rhs[j]))
    if min(witness_slack,default=0.) < -TOL:
        raise ValueError("selected finite-axis rows are not satisfied by shared displacement witness")
    q=float(np.max(bone@n));margin=float(sat._float32_world_m_emission_margin_mm(skin,bone,n))
    return {"normal_world_unit":n.copy(),"plane_point_mm":bone[int(np.argmax(bone@n))].copy(),
        "float32_ulp_margin_mm":margin,"corner_signed_distance_mm":skin@n-q,
        "corner_lower_mm":rhs.copy(),"positive_corner_deficits_mm":np.maximum(rhs,0.),
        "axis_strategy":"joint_component_sat_box_witness_638","axis_source":axis_meta["source"],
        "axis_index":int(axis_meta["index"]),"axis_sign":int(axis_meta["sign"]),
        "axis_sign_order":int(axis_meta["sign_order"]),"axis_extra_sign":int(axis_meta["extra_sign"]),
        "axis_source_order":int(axis_meta["source_order"]),"candidate_axis_count":int(axis_count),
        "box_corner_max_progress_mm":np.asarray(maxmove),"box_corner_unmet_mm":np.asarray(unmet),
        "box_feasible":bool(max(unmet,default=0.)<=TOL),"box_max_unmet_mm":float(max(unmet,default=0.)),
        "selector_witness_min_row_slack_mm":float(min(witness_slack,default=0.)),
        "selector_box_witness_verified":True,"selector_displacement_applied_to_fit":False,
        "whole_bank_clearance_certificate":False,"shared_node_axis_compatibility_certified":True,
        "native_admission":False,"accepted_physical_steps":0}

def install_axis_map_from_cli(owner,objective,crossing):
    """Install verified selected axes; leave 637 fallback active for unselected pairs."""
    npz_path=Path(os.environ["NUMI_AXIS_MAP638_NPZ"]).resolve()
    report_path=Path(os.environ["NUMI_AXIS_MAP638_REPORT"]).resolve()
    report=json.loads(report_path.read_text())
    pre=report.get("preflight",{})
    inp=pre.get("inputs",{})
    argv=captured_fit_argv()
    base=arg_value(argv,"--base-candidate");source=arg_value(argv,"--source-base-npz")
    audit=arg_value(argv,"--audit159-490");point=arg_value(argv,"--point-target-npz");face=arg_value(argv,"--face-target-npz")
    expected_inputs={
        "candidate":{"path":str(base),"sha256":sha(base)},
        "source_base":{"path":str(source),"sha256":sha(source)},
        "complete_159_report":{"path":str(audit),"sha256":sha(audit)},
        "point_target_npz":{"path":str(point),"sha256":sha(point)},
        "face_target_npz":{"path":str(face),"sha256":sha(face)},
    }
    for k,v in expected_inputs.items():
        got=inp.get(k,{})
        if Path(got.get("path","")).resolve()!=Path(v["path"]).resolve() or got.get("sha256")!=v["sha256"]:
            raise ValueError(f"selector-to-fit input pin mismatch: {k}")
    additional={
        "point_target_report":"--point-target-report",
        "face_target_report":"--face-target-report",
        "boundary_npz":"--fixed-boundary-npz",
        "boundary_report":"--fixed-boundary-report",
        "self_pair_seeds":"--self-pairs",
    }
    for role,flag in additional.items():
        path=arg_value(argv,flag)
        got=inp.get(role,{})
        if Path(got.get("path","")).resolve()!=path or got.get("sha256")!=sha(path):
            raise ValueError(f"selector-to-fit prepared-role mismatch: {role}")
    if sha(npz_path)!=report.get("selection_npz_sha256"):
        raise ValueError("selector NPZ hash differs from report")
    declared=Path(report.get("selection_npz",""))
    if not declared.is_absolute(): declared=(report_path.parent/declared).resolve()
    if declared!=npz_path:raise ValueError("selector report points at another selection NPZ")
    sat=load_module("sat551_for_axis_map638",SAT_PATH,SAT_SHA)
    box=load_module("bounds636_for_axis_map638",BOX_PATH,BOX_SHA)
    import box_aware_sat637 as helper637
    if sha(RUN603_PATH)!=RUN603_SHA or sha(RUN637_PATH)!=RUN637_SHA or sha(HELPER637_PATH)!=HELPER637_SHA:
        raise RuntimeError("frozen 603/637 source pin mismatch")
    if not helper637.INSTALLS:raise RuntimeError("637 install must precede 638 axis-map install")
    st=helper637.INSTALLS[-1]
    if st.get("base_candidate_sha256")!=sha(base) or st.get("source_base_sha256")!=sha(source):
        raise ValueError("637 runtime base/source differs from selector inputs")
    if int(st.get("fixed_prefix_nodes",0))!=0:
        raise ValueError("selector map was prepared for zero fixed-prefix nodes")
    pair_identity=np.asarray([[int(r["face_row"]),int(r["body"]),int(r["bone_id"]),int(r["triangle_id"])] for r in crossing["pairs"]],dtype=np.int64)
    pair_nodes=np.asarray(crossing["faces"],dtype=np.int64)[pair_identity[:,0]]
    with np.load(npz_path,allow_pickle=False) as z:
        arrays={k:np.asarray(z[k]).copy() for k in z.files}
    # 408 support and 636 bounds are checked on the first actual 637 chooser call,
    # then all pair identities and the shared displacement witness are checked now.
    preinputs=pre.get("inputs",{})
    validate_selector_source_pins(preinputs.get("sources"))
    loaded=validate_selector_arrays(arrays,report,pair_identity,pair_nodes,expected_inputs,
        preinputs.get("basis_scope_rows_sha256",""),
        preinputs.get("directional636_lower_sha256",""),
        preinputs.get("directional636_upper_sha256",""))
    points=np.asarray(crossing["points_mm"],dtype=np.float64)
    bones=crossing["bone_triangles_mm"]
    axis_records={}
    for i in np.flatnonzero(loaded["selected"]):
        ident=tuple(map(int,loaded["identity"][i]));nodes=loaded["nodes"][i]
        skin=points[nodes]
        face_row,body,bone,tri=ident
        bone_tri=np.asarray(bones[(body,bone,tri)],dtype=np.float64)
        normal=loaded["normals"][i]
        possible=sat._signed_sat_axes(skin,bone_tri)
        match=[]
        expected_meta=(str(loaded["axis_source"][i]),int(loaded["axis_index"][i]),
            int(loaded["axis_sign"][i]),int(loaded["axis_source_order"][i]),
            int(loaded["axis_sign_order"][i]),int(loaded["axis_extra_sign"][i]))
        for a in possible:
            meta=(str(a["axis_source"]),int(a["axis_index"]),int(a["axis_sign"]),
                  int(a["axis_source_order"]),int(a["axis_sign_order"]),int(a["extra_sign"]))
            if meta==expected_meta and np.allclose(np.asarray(a["normal_world_unit"],dtype=np.float64),normal,rtol=0,atol=2e-12):
                match.append(a)
        match_axis_record(possible,{"source":expected_meta[0],"index":expected_meta[1],
            "sign":expected_meta[2],"source_order":expected_meta[3],
            "sign_order":expected_meta[4],"extra_sign":expected_meta[5]},normal)
        axis_records[i]=expected_meta
    # Build an exact triangle-key queue. Frozen 637 iterates its input pair list;
    # each callback is matched by node order and emitted skin/bone coordinates.
    queue_template=make_pair_queue_template(pair_identity,pair_nodes,points,bones)
    queues=fresh_pair_queues(queue_template)
    original=helper637.choose_pair_sat_plane_with_bounds
    original_build=objective.build_fixed_bone_outward_plane_rows
    state={"selector_report_path":str(report_path),"selector_report_sha256":sha(report_path),
        "selector_npz_path":str(npz_path),"selector_npz_sha256":sha(npz_path),
        "selected_pair_count":int(loaded["selected"].sum()),"pair_count":len(pair_identity),
        "selected_identities":loaded["identity"][loaded["selected"]].tolist(),
        "selected_axis_records":[{"identity":loaded["identity"][i].tolist(),
            "normal_world_unit":loaded["normals"][i].tolist(),
            "axis_source":str(loaded["axis_source"][i]),"axis_index":int(loaded["axis_index"][i]),
            "axis_sign":int(loaded["axis_sign"][i]),"axis_source_order":int(loaded["axis_source_order"][i]),
            "axis_sign_order":int(loaded["axis_sign_order"][i]),"extra_sign":int(loaded["axis_extra_sign"][i])}
            for i in np.flatnonzero(loaded["selected"])],
        "fallback_identities":loaded["identity"][~loaded["selected"]].tolist(),
        "box_only_selector_claim":pre.get("claim_limit"),"native_admission":False,"accepted_physical_steps":0,
        "runtime_bounds_verified":False,"callbacks_consumed":0}
    def wrapped(skin,bone,nodes,support,lower,upper,tol=2e-9):
        if not state["runtime_bounds_verified"]:
            scope=np.asarray(support,dtype=np.int64)
            lo=np.asarray(lower,dtype=np.float64).reshape(-1)
            hi=np.asarray(upper,dtype=np.float64).reshape(-1)
            if array_sha(scope,"<i8")!=preinputs.get("basis_scope_rows_sha256"):
                raise ValueError("637 runtime ordered support differs from selector 408 support")
            if array_sha(lo,"<f8")!=preinputs.get("directional636_lower_sha256"):
                raise ValueError("637 runtime lower bounds differ from selector 636 box")
            if array_sha(hi,"<f8")!=preinputs.get("directional636_upper_sha256"):
                raise ValueError("637 runtime upper bounds differ from selector 636 box")
            state["runtime_bounds_verified"]=True
            state["runtime_support_rows_sha256"]=array_sha(scope,"<i8")
            state["runtime_lower_sha256"]=array_sha(lo,"<f8")
            state["runtime_upper_sha256"]=array_sha(hi,"<f8")
        key=_tri_key(nodes,skin,bone)
        q=queues.get(key)
        if not q:raise ValueError("637 chooser callback does not match any unused 159 typed pair")
        i=q.popleft();state["callbacks_consumed"]+=1
        meta=axis_records.get(i)
        return route_pair_record(bool(loaded["selected"][i]),
            lambda: original(skin,bone,nodes,support,lower,upper,tol),
            lambda: dict(_record_from_selected(skin,bone,nodes,support,lower,upper,loaded["normals"][i],
                {"source":meta[0],"index":meta[1],"sign":meta[2],"source_order":meta[3],
                 "sign_order":meta[4],"extra_sign":meta[5]},
                loaded["displacement_by_node"],sat,len(sat._signed_sat_axes(skin,bone))),
                selector_identity=loaded["identity"][i].tolist()))
    def checked_build(points_mm,faces,support,anchors,pairs,bone_triangles_mm,total_weight=1.0):
        nonlocal queues
        queues=fresh_pair_queues(queue_template)
        state["callbacks_consumed"]=0
        actual_identity=np.asarray([[int(r["face_row"]),int(r["body"]),int(r["bone_id"]),int(r["triangle_id"])] for r in pairs],dtype=np.int64)
        if not np.array_equal(actual_identity,pair_identity):
            raise ValueError("637 row builder typed-pair order differs from selector inventory")
        result=original_build(points_mm,faces,support,anchors,pairs,bone_triangles_mm,total_weight=total_weight)
        if state["callbacks_consumed"]!=len(pair_identity):
            raise ValueError(f"637 used {state['callbacks_consumed']} of {len(pair_identity)} typed pairs")
        if any(queues.values()):raise ValueError("unconsumed crossing-pair records remain")
        for group in ("metadata","guard_metadata"):
            for rec in result.get(group,[]):
                i=int(rec["pair_index"])
                if i<0 or i>=len(loaded["selected"]):
                    raise ValueError("637 metadata pair index outside selector inventory")
                chosen=bool(loaded["selected"][i])
                rec["plane_selection"]=("joint_component_sat_box_witness_638" if chosen else
                    "finite_pair_sat_box_aware_soft_target_637_fallback")
                rec["selector_pair_selected"]=chosen
                if chosen:
                    rec["selector_identity"]=loaded["identity"][i].tolist()
                    rec["selector_axis_source"]=str(loaded["axis_source"][i])
                    rec["selector_axis_index"]=int(loaded["axis_index"][i])
                    rec["selector_axis_sign"]=int(loaded["axis_sign"][i])
        return result
    helper637.choose_pair_sat_plane_with_bounds=wrapped
    objective.build_fixed_bone_outward_plane_rows=checked_build
    state["strategy"]="selector-driven selected axes; frozen 637 fallback for unselected pairs"
    state["objective_contract"]="Same 637 soft positive corner rows and nonpositive guards; only per-pair target axis selection changes for selector-selected pairs. The selector displacement is a feasibility witness and is not applied to the fit."
    state["selected_axes_verified_551"]=len(axis_records)
    state["selected_shared_displacement_witness_verified"]=True
    state["whole_bank_clearance_certificate"]=False
    state["physical_acceptance"]="not evaluated; original 603/545/130/159 gates remain required"
    INSTALLS.append(state)
    return state

def write_result(output_dir):
    if not INSTALLS:return None
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    state=INSTALLS[-1]
    p=out/"axis-map-638.json"
    p.write_text(json.dumps(state,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return p

