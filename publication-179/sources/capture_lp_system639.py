"""Opt-in, read-only capture of the first 603/408 LP matrices passed to 485.

The capture observes the already-built system and calls the original solver with
identical objects. It does not change constraints, objective, targets, bounds,
or acceptance. Captures are construction diagnostics, never admission evidence.
"""
from __future__ import annotations
import gzip, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse

SCHEMA = "numi.human.hard_lp_system_capture639.v1"
COMPONENT_MODULE = "component_lp_decomposition_485_for_run"

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def _jsonable(x):
    if isinstance(x, dict): return {str(k):_jsonable(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if sparse.issparse(x): return {"shape":list(x.shape),"nnz":int(x.nnz),"format":x.getformat()}
    if isinstance(x, Path): return str(x)
    if isinstance(x, (str,int,float,bool)) or x is None: return x
    return repr(x)

def _csr_copy(x, name):
    m=sparse.csr_matrix(x, dtype=np.float64, copy=True)
    if not np.isfinite(m.data).all(): raise ValueError(f"{name} has nonfinite data")
    return m

def _matrix_digest(m):
    m=_csr_copy(m,"matrix")
    return {"format":"csr","shape":[int(v) for v in m.shape],"nnz":int(m.nnz),
            "data_dtype":str(m.data.dtype),"indices_dtype":str(m.indices.dtype),
            "indptr_dtype":str(m.indptr.dtype),
            "data_sha256":sha_bytes(np.ascontiguousarray(m.data).tobytes()),
            "indices_sha256":sha_bytes(np.ascontiguousarray(m.indices).tobytes()),
            "indptr_sha256":sha_bytes(np.ascontiguousarray(m.indptr).tobytes())}

def _write_json(path, obj):
    Path(path).write_text(json.dumps(_jsonable(obj),indent=2,sort_keys=True,allow_nan=False)+"\n")

def _row_summary(rows):
    rows=list(rows)
    kinds={}
    for r in rows:
        k=str(r.get("kind",r.get("row_type","unknown"))) if isinstance(r,dict) else "unknown"
        kinds[k]=kinds.get(k,0)+1
    return {"count":len(rows),"kind_counts":dict(sorted(kinds.items())),
            "records":_jsonable(rows)}

class Capture:
    def __init__(self, output_dir, *, source_pins, argv, base_candidate, selector_npz=None,
                 selector_report=None, max_calls=1):
        self.output_dir=Path(output_dir).resolve()
        self.source_pins=dict(source_pins)
        self.argv=list(argv)
        self.base_candidate=Path(base_candidate).resolve()
        self.selector_npz=None if selector_npz is None else Path(selector_npz).resolve()
        self.selector_report=None if selector_report is None else Path(selector_report).resolve()
        self.max_calls=int(max_calls)
        if self.max_calls != 1: raise ValueError("this owner is deliberately bounded to one matrix capture")
        if self.output_dir.exists(): raise FileExistsError(self.output_dir)
        if not self.output_dir.parent.is_dir(): raise FileNotFoundError(self.output_dir.parent)
        if not self.base_candidate.is_file(): raise FileNotFoundError(self.base_candidate)
        self.events={}
        self.calls=[]
        self._solver_original=None
        self._installed=False

    def _event(self, role, payload):
        if role in self.events: raise RuntimeError(f"duplicate capture event: {role}")
        self.events[role]=payload

    def wrap_owner(self, owner):
        if self._installed: raise RuntimeError("capture hook installed more than once")
        for name in ("build_immediate_parent_basis","build_clearance_rows","_owner_matrix"):
            if not callable(getattr(owner,name,None)): raise RuntimeError(f"408 owner missing hook {name}")
        self._installed=True

        original_basis=owner.build_immediate_parent_basis
        def basis(*args,**kwargs):
            result=original_basis(*args,**kwargs)
            if "basis" not in self.events:
                if not isinstance(result,tuple) or len(result)<3:
                    raise ValueError("408/603 basis return contract changed")
                B,parent,report=result[:3]
                self.events["basis"]={"matrix":_csr_copy(B,"basis"),"parent":np.asarray(parent).copy(),
                    "report":_jsonable(report),"support":np.asarray(args[1] if len(args)>1 else kwargs["support"],dtype=np.int64).copy(),
                    "anchors":np.asarray(args[2] if len(args)>2 else kwargs["anchors"],dtype=np.int64).copy()}
            return result
        owner.build_immediate_parent_basis=basis

        original_clearance=owner.build_clearance_rows
        def clearance(*args,**kwargs):
            result=original_clearance(*args,**kwargs)
            if "clearance_metadata" not in self.events:
                if not isinstance(result,tuple) or len(result)<4:
                    raise ValueError("603 clearance-row return contract changed")
                C,h,w,meta=result[:4]
                self.events["clearance_metadata"]={"C":_csr_copy(C,"C_metadata"),"h":np.asarray(h,dtype=np.float64).copy(),
                    "weights":np.asarray(w,dtype=np.float64).copy(),"metadata":_jsonable(meta)}
            return result
        owner.build_clearance_rows=clearance

        original_owner_matrix=owner._owner_matrix
        def owner_matrix(pair_rows,area_rows,area_rhs,support):
            result=original_owner_matrix(pair_rows,area_rows,area_rhs,support)
            if "owner_matrix" not in self.events:
                A,b=result
                self.events["owner_matrix"]={"A":_csr_copy(A,"owner_A"),"b":np.asarray(b,dtype=np.float64).copy(),
                    "area_rows":_csr_copy(area_rows,"area_rows"),"area_rhs":np.asarray(area_rhs,dtype=np.float64).copy(),
                    "support":np.asarray(support,dtype=np.int64).copy(),"pair_rows":_row_summary(pair_rows)}
            return result
        owner._owner_matrix=owner_matrix
        return {"wrapped":["build_immediate_parent_basis","build_clearance_rows","_owner_matrix"],
                "source_sha256":dict(self.source_pins)}

    def wrap_component_solver(self, module):
        if getattr(module,"__name__",None)!=COMPONENT_MODULE:
            raise ValueError("unexpected 485 module identity")
        original=getattr(module,"solve_by_components",None)
        if not callable(original): raise RuntimeError("485 component solver missing")
        if getattr(module,"_capture639_wrapped",False): raise RuntimeError("485 solver already wrapped")
        self._solver_original=original
        def wrapped(*args,**kwargs):
            call_no=len(self.calls)+1
            if call_no == 1:
                self._capture_call(args,kwargs)
            else:
                self.calls.append({"call_index":call_no,"captured":False,"reason":"bounded first-call capture",
                                   "argument_count":len(args),"keyword_names":sorted(kwargs)})
                self._update_call_log()
            try:
                result=original(*args,**kwargs)
            except BaseException as exc:
                if call_no==1: self._finish_call({"status":"raised","error_type":type(exc).__name__,"error":str(exc)})
                raise
            if call_no==1:
                self._finish_call({"status":"returned","success":bool(result.get("success",False)) if isinstance(result,dict) else None,
                    "solver_status":_jsonable(result.get("status")) if isinstance(result,dict) else None,
                    "solver_message":_jsonable(result.get("message")) if isinstance(result,dict) else None})
            return result
        module.solve_by_components=wrapped
        module._capture639_wrapped=True
        return {"module":COMPONENT_MODULE,"solver_hook":"solve_by_components","bounded_calls":1}

    def _capture_call(self,args,kwargs):
        if len(args)<10: raise ValueError("485 call does not match pinned 10-argument prefix")
        # solve_by_components(original_solver,A,b,target,lower,upper,C,h,weights,M,time_limit,...)
        vals=args[1:10]
        names=("A","b","target_output","lower_output","upper_output","C","h","weights","output_map")
        d=dict(zip(names,vals))
        time_limit=args[10] if len(args)>10 else kwargs.get("time_limit_s")
        if time_limit is None: raise ValueError("485 time_limit_s missing")
        A=_csr_copy(d["A"],"A"); C=_csr_copy(d["C"],"C"); M=_csr_copy(d["output_map"],"output_map")
        vectors={k:np.asarray(d[k],dtype=np.float64).reshape(-1).copy() for k in
                 ("b","target_output","lower_output","upper_output","h","weights")}
        nout,ncontrol=M.shape
        if A.shape!=(len(vectors["b"]),nout) or C.shape!=(len(vectors["h"]),nout):
            raise ValueError("captured A/C/output dimensions do not align")
        if vectors["target_output"].shape!=(nout,) or vectors["lower_output"].shape!=(nout,) or vectors["upper_output"].shape!=(nout,) or vectors["weights"].shape!=(len(vectors["h"]),):
            raise ValueError("captured output vectors do not align")
        if not all(np.isfinite(v).all() for v in vectors.values()): raise ValueError("captured LP vector is nonfinite")
        if np.any(vectors["lower_output"]>vectors["upper_output"]): raise ValueError("captured bound ordering invalid")
        self.output_dir.mkdir(parents=False,exist_ok=False)
        mats={"hard_A":A,"clearance_C":C,"output_map_M":M}
        meta={"schema":SCHEMA,"capture_index":1,"capture_scope":"exact first solver-call inputs before 485 component decomposition",
              "units":"408 output coordinates and target/bounds in millimetres; LP row units as emitted by frozen owners",
              "matrices":{},"vectors":{},"source_pins":self.source_pins,
              "inputs":{"base_candidate_path":str(self.base_candidate),"base_candidate_sha256":sha_file(self.base_candidate),
                        "selector_npz_path":str(self.selector_npz) if self.selector_npz else None,
                        "selector_npz_sha256":sha_file(self.selector_npz) if self.selector_npz else None,
                        "selector_report_path":str(self.selector_report) if self.selector_report else None,
                        "selector_report_sha256":sha_file(self.selector_report) if self.selector_report else None},
              "solver_call":{"time_limit_s":float(time_limit),"argument_count":len(args),"keyword_names":sorted(kwargs),
                             "optional_args":_jsonable(args[11:]),"optional_kwargs":_jsonable(kwargs)},
              "dimensions":{"hard_rows":A.shape[0],"output_dimension":nout,"control_dimension":ncontrol,
                            "clearance_rows":C.shape[0],"basis_csr":M.shape},
              "metadata_coverage":{"basis":False,"clearance_rows":False,"owner_matrix":False},
              "construction_only":True,"native_admission":False,"accepted_physical_steps":0}
        for name,m in mats.items():
            path=self.output_dir/(name+".npz"); sparse.save_npz(path,m,compressed=True)
            meta["matrices"][name]={**_matrix_digest(m),"path":str(path),"file_sha256":sha_file(path)}
        for name,v in vectors.items():
            meta["vectors"][name]={"shape":list(v.shape),"dtype":str(v.dtype),
                "sha256":sha_bytes(np.ascontiguousarray(v).tobytes())}
        npz=self.output_dir/"vectors.npz"
        np.savez_compressed(npz,**vectors)
        meta["vectors_file"]={"path":str(npz),"sha256":sha_file(npz)}
        if "basis" in self.events:
            e=self.events["basis"]; path=self.output_dir/"captured_basis.npz"; sparse.save_npz(path,e["matrix"],compressed=True)
            np.savez_compressed(self.output_dir/"basis_domain.npz",parent=e["parent"],support=e["support"],anchors=e["anchors"],
                                output_scope_rows=np.asarray(e["report"].get("output_scope_rows",[]),dtype=np.int64),
                                control_parent_rows=np.asarray(e["report"].get("control_parent_rows",[]),dtype=np.int64))
            meta["metadata_coverage"]["basis"]=True
            meta["basis"]={"matrix":_matrix_digest(e["matrix"]),"matrix_file_sha256":sha_file(path),
                           "report":e["report"],"parent_shape":list(e["parent"].shape),
                           "support_count":len(e["support"]),"anchor_count":len(e["anchors"])}
        if "clearance_metadata" in self.events:
            e=self.events["clearance_metadata"]
            if _matrix_digest(e["C"])!=_matrix_digest(C) or not np.array_equal(e["h"],vectors["h"]) or not np.array_equal(e["weights"],vectors["weights"]):
                raise ValueError("captured 408 row metadata payload differs from solver-call C/h/weights")
            _write_json(self.output_dir/"clearance-row-metadata.json",e["metadata"])
            meta["metadata_coverage"]["clearance_rows"]=True
            meta["clearance_row_metadata"]={"path":str(self.output_dir/"clearance-row-metadata.json"),
                "sha256":sha_file(self.output_dir/"clearance-row-metadata.json"),"count":len(e["metadata"])}
        if "owner_matrix" in self.events:
            e=self.events["owner_matrix"]
            abase=e["A"]; bbase=e["b"]
            prefix_ok=(A.shape[0]>=abase.shape[0] and _matrix_digest(A[:abase.shape[0]])==_matrix_digest(abase)
                       and np.array_equal(vectors["b"][:len(bbase)],bbase))
            sparse.save_npz(self.output_dir/"owner_matrix_A_before_edge.npz",abase,compressed=True)
            sparse.save_npz(self.output_dir/"area_linearization_rows.npz",e["area_rows"],compressed=True)
            np.savez_compressed(self.output_dir/"owner_matrix_vectors.npz",b_before_edge=bbase,area_rhs=e["area_rhs"],
                                support=e["support"])
            _write_json(self.output_dir/"hard-row-metadata.json",e["pair_rows"])
            meta["metadata_coverage"]["owner_matrix"]=True
            meta["owner_matrix"]={"A_before_edge":_matrix_digest(abase),
                "A_before_edge_file_sha256":sha_file(self.output_dir/"owner_matrix_A_before_edge.npz"),
                "area_rows":_matrix_digest(e["area_rows"]),
                "area_rows_file_sha256":sha_file(self.output_dir/"area_linearization_rows.npz"),
                "metadata_path":str(self.output_dir/"hard-row-metadata.json"),
                "metadata_sha256":sha_file(self.output_dir/"hard-row-metadata.json"),
                "pair_row_count":e["pair_rows"]["count"],"area_row_count":e["area_rows"].shape[0],
                "final_A_has_owner_matrix_prefix":bool(prefix_ok),
                "trailing_hard_row_count":int(A.shape[0]-abase.shape[0])}
        self.calls.append({"call_index":1,"captured":True,"status":"captured-before-solver"})
        self._write(meta)

    def _write(self,meta):
        _write_json(self.output_dir/"capture.json",meta)
        self._update_call_log()

    def _update_call_log(self):
        if not self.output_dir.exists():return
        _write_json(self.output_dir/"calls.json",{"schema":SCHEMA,"calls":self.calls,
            "native_admission":False,"accepted_physical_steps":0})

    def _finish_call(self,outcome):
        if not self.output_dir.exists():return
        path=self.output_dir/"solver-outcome.json"
        _write_json(path,{"schema":SCHEMA,"outcome":outcome,"native_admission":False,"accepted_physical_steps":0})

    def finalize(self, run_status, error=None):
        receipt={"schema":SCHEMA,"run_status":run_status,
            "error":None if error is None else {"type":type(error).__name__,"message":str(error)},
            "calls":self.calls,"capture_complete":(self.output_dir/"capture.json").is_file(),
            "native_admission":False,"accepted_physical_steps":0}
        if self.output_dir.exists():
            _write_json(self.output_dir/"run.json",receipt)
        else:
            _write_json(self.output_dir.with_name(self.output_dir.name+"-run.json"),receipt)

def install_capture_after_axis_adapter(original_install, capture: Capture, module_lookup=None):
    """Return an adapter install hook; original adapter runs first and is unchanged."""
    def wrapped(owner,objective,crossing):
        result=original_install(owner,objective,crossing)
        module=sys.modules.get(COMPONENT_MODULE) if module_lookup is None else module_lookup()
        if module is None: raise RuntimeError("pinned 485 component module is not loaded at 638 hook")
        capture.wrap_owner(owner)
        capture.wrap_component_solver(module)
        return result
    return wrapped
