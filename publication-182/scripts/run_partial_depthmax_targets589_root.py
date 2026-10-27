#!/usr/bin/env python3
"""Depth-maximizing 357 target adapter, with current/source sign and source-cosine gates.

Pinned 545 still certifies each first empty interval over all 185 banks. This narrow v2
allows only depth-maximizing with cmin=.05, after the same O-to-I current/source normal check.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
BUILDER=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/partial-midpoint-continuation-root-001/build_partial_depth_targets545_root.py')
BUILDER_SHA='74e6c3c90f3db016f9b03784754fce606bb2f38a0cab0354855f7eda4a4da057'
EXT=ROOT/'D/source-outward-proposal-500/build_outward_targets_multi_source_546.py'
EXT_SHA='dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68'
OWNER500=ROOT/'D/source-outward-proposal-500/build_outward_nearest_union_targets_500.py'
OWNER500_SHA='454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf'
NORMAL_AUDIT_SCRIPT=ROOT/'D/surface-clearance-546/source-normal-audit-588/audit_midpoint_normals588.py'
NORMAL_AUDIT_SHA='a6f7b8c1cd333e117f9a8f6f31bb9cc40f8bf5569c6f01ea7cc2074a99c3b703'
BASE589=ROOT/'D/surface-clearance-546/source-outward-targets-589/run_source_outward_targets589.py'
BASE589_SHA='11f8a4a4b906294e9832173b63601f438c9c4bf1f57e488e8fd95f398f0038ec'
MINIMUM_COSINE=.05

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def loadmod(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(f'cannot load {path}')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def require_depth_maximizing(argv):
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--direction-mode',choices=('source-outward','depth-compatible','depth-maximizing','direct-depth'),default='source-outward')
    ap.add_argument('--minimum-depth-cosine',type=float,default=.05)
    ap.add_argument('--point-depth-rows-npz',type=Path)
    ap.add_argument('--point-depth-rows-report',type=Path)
    ns,_=ap.parse_known_args(list(argv))
    if ns.direction_mode!='depth-maximizing': raise ValueError('589v2 adapter requires --direction-mode depth-maximizing')
    if not np.isfinite(ns.minimum_depth_cosine) or abs(ns.minimum_depth_cosine-MINIMUM_COSINE)>1e-12:
        raise ValueError('589v2 adapter pins minimum-depth-cosine to 0.05')
    if ns.point_depth_rows_npz is None or ns.point_depth_rows_report is None:
        raise ValueError('depth-maximizing mode requires both pinned 357 point-depth row inputs')
    return ns.direction_mode

def current_source_alignment(points,faces,target_rows,source_normals,coarse_outward_sign):
    P=np.asarray(points); F=np.asarray(faces); rows=np.asarray(target_rows)
    sign=int(coarse_outward_sign)
    if P.dtype!=np.float32 or P.ndim!=2 or P.shape[1]!=3 or not np.isfinite(P).all(): raise ValueError('current points must be finite Float32 Nx3')
    if F.ndim!=2 or F.shape[1]!=3 or F.dtype.kind not in 'iu' or np.any(F<0) or np.any(F>=len(P)): raise ValueError('current faces invalid')
    if rows.ndim!=1 or rows.dtype.kind not in 'iu' or len(np.unique(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=len(P)): raise ValueError('target rows invalid')
    if sign not in (-1,1): raise ValueError('O-to-I current outward winding sign unresolved')
    incidence={int(r):[] for r in rows}
    for fi,tri in enumerate(F):
        for node in set(map(int,tri)):
            if node in incidence: incidence[node].append(fi)
    diagnostics=[]; disagreements=[]
    for row in map(int,rows):
        face_rows=incidence[row]
        if not face_rows: raise ValueError(f'target row {row} has no current incident faces')
        tri=P[F[face_rows]].astype(np.float64)
        crosses=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        lengths=np.linalg.norm(crosses,axis=1); area_sum=float(lengths.sum())
        resultant=crosses.sum(axis=0); rnorm=float(np.linalg.norm(resultant))
        ratio=rnorm/area_sum if area_sum>0 else 0.0
        src=np.asarray(source_normals[row],dtype=np.float64)
        if src.shape!=(3,) or not np.isfinite(src).all(): raise ValueError(f'target row {row} source normal invalid')
        sn=float(np.linalg.norm(src))
        if sn<=0: raise ValueError(f'target row {row} source normal degenerate')
        src=src/sn
        if not np.isfinite(ratio) or area_sum<=0 or rnorm<=64*np.finfo(np.float64).eps*area_sum:
            diag={'query_row':row,'status':'CURRENT_NORMAL_CANCELLED_OR_UNRESOLVED','incident_faces':face_rows,
                  'current_area_sum_m2':area_sum,'resultant_over_area_sum':ratio,'coarse_outward_sign':sign}
        else:
            current_out=(sign/rnorm)*resultant
            dot=float(current_out@src)
            diag={'query_row':row,'status':'PASS' if dot>=0.0 else 'CURRENT_SOURCE_OUTWARD_DISAGREEMENT',
                  'incident_faces':face_rows,'current_area_sum_m2':area_sum,'resultant_over_area_sum':ratio,
                  'current_outward_normal_world_unit':current_out.tolist(),'registered_source_outward_normal_world_unit':src.tolist(),
                  'current_source_outward_dot':dot,'coarse_outward_sign':sign}
        diagnostics.append(diag)
        if diag['status']!='PASS': disagreements.append(diag)
    return {'status':'PASS' if not disagreements else 'CURRENT_SOURCE_OUTWARD_DISAGREEMENT',
            'coarse_outward_sign':sign,'target_row_count':len(diagnostics),
            'agreement_count':len(diagnostics)-len(disagreements),'disagreement_count':len(disagreements),
            'rows':diagnostics,'witnesses':disagreements}

def validate_545_depth_maximizing_report(report,target_archive,minimum_cosine=MINIMUM_COSINE):
    if report.get('direction_mode')!='depth-maximizing' or report.get('physical_steps')!=0:
        raise ValueError('545 report is not depth-maximizing proposal-only')
    if abs(float(report.get('minimum_depth_cosine',float('nan')))-minimum_cosine)>1e-12:
        raise ValueError('545 depth/source cosine floor differs from pinned 0.05')
    q=np.asarray(target_archive.get('query_rows')); src=np.asarray(target_archive.get('source_surface_normal_world_unit'))
    prop=np.asarray(target_archive.get('proposal_direction_world_unit')); depth=np.asarray(target_archive.get('depth_covector_world_unit'))
    mask=np.asarray(target_archive.get('pass_mask')); outside=np.asarray(target_archive.get('target_exactly_outside_all185'))
    n=len(q)
    if (q.shape!=(n,) or src.shape!=(n,3) or prop.shape!=(n,3) or depth.shape!=(n,3)
        or mask.shape!=(n,) or outside.shape!=(n,) or not np.isfinite(src).all() or not np.isfinite(prop).all()
        or not np.isfinite(depth).all() or not np.all(mask) or not np.all(outside)):
        raise ValueError('589v2 arrays lack all passing directions/depth rows/all185 endpoints')
    records=report.get('targets')
    if not isinstance(records,list) or len(records)!=n or report.get('all_bank_ray_results')!=n or report.get('passing_targets')!=n:
        raise ValueError('545 target record counts do not cover every inside row')
    for i,item in enumerate(records):
        if item.get('query_row')!=int(q[i]) or item.get('status')!='PASS_FIRST_UNION_EXIT':
            raise ValueError('545 target row identity/status mismatch')
        if item.get('target_exactly_outside_all185') is not True or item.get('exact_endpoint_bank_count')!=185 or item.get('end_outside_all185_by_event_parity') is not True:
            raise ValueError(f"target row {q[i]} lacks complete all185 endpoint certificate")
        sel=item.get('selection',{})
        if sel.get('target_exactly_outside_all_banks') is not True or sel.get('classified_bank_count')!=185:
            raise ValueError(f"target row {q[i]} lacks complete all185 selector receipt")
        compat=item.get('direction_compatibility',{})
        sc=float(compat.get('selected_source_hemisphere_cosine',float('nan')))
        dc=float(compat.get('selected_depth_cosine',float('nan')))
        s=np.asarray(src[i],np.float64);d=np.asarray(prop[i],np.float64);nvec=np.asarray(depth[i],np.float64)
        if any(not np.isfinite(v).all() for v in (s,d,nvec)) or min(np.linalg.norm(s),np.linalg.norm(d),np.linalg.norm(nvec))<=0:
            raise ValueError(f"target row {q[i]} has degenerate source/depth/proposal vector")
        s=s/np.linalg.norm(s);d=d/np.linalg.norm(d);nvec=nvec/np.linalg.norm(nvec)
        if sc<minimum_cosine-1e-10 or dc<minimum_cosine-1e-10 or float(s@d)<minimum_cosine-1e-10 or float(nvec@d)<minimum_cosine-1e-10:
            raise ValueError(f"target row {q[i]} violates pinned source/depth cosine floors")
        if not np.array_equal(np.asarray(item.get('proposal_direction_world_unit'),np.float64),np.asarray(prop[i],np.float64)):
            raise ValueError(f"target row {q[i]} report/NPZ proposal direction mismatch")
    return {'target_count':n,'passing_targets':n,'all_passing_endpoints_exactly_outside_185':True,
            'minimum_source_cosine':minimum_cosine,'minimum_depth_cosine':minimum_cosine,'physical_steps':0}

def install_source303_extension(owner,extension,face_provenance=None):
    if face_provenance is None: face_provenance={}
    original_multi=extension.multi_parent_source_outward_normal
    def capture(face_row,candidate,detail,orientation):
        normal,prov=original_multi(face_row,candidate,detail,orientation)
        face_provenance[int(face_row)]=prov
        return normal,prov
    extension.multi_parent_source_outward_normal=capture
    return extension.install_extension(owner)

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    require_depth_maximizing(argv)
    for path,expected,label in ((BUILDER,BUILDER_SHA,'545 builder'),(EXT,EXT_SHA,'546 multiparent helper'),(OWNER500,OWNER500_SHA,'500 source normal owner'),(NORMAL_AUDIT_SCRIPT,NORMAL_AUDIT_SHA,'588 orientation audit'),(BASE589,BASE589_SHA,'589 source-outward baseline adapter')):
        if sha(path)!=expected: raise ValueError(f'pinned {label} changed')
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    ns,_=parser.parse_known_args(argv);out=ns.output_dir.resolve();candidate_path=ns.candidate.resolve()
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    audit_path=out.parent/(out.name+'.current-source-normal-audit.json')
    if audit_path.exists(): raise FileExistsError(audit_path)
    extension=loadmod(EXT,'multi_parent_source_normals_546_for_589')
    if sha(extension.OWNER500)!=OWNER500_SHA or extension.OWNER500_SHA!=OWNER500_SHA: raise ValueError('546 extension 500 owner pin mismatch')
    builder=loadmod(BUILDER,'source_outward_target_builder545_for_589')
    face_provenance={}; original_loadmod=builder.loadmod
    def load_with_multisource(path,name):
        module=original_loadmod(path,name)
        if Path(path).resolve()==Path(builder.SOURCE303).resolve(): install_source303_extension(module,extension,face_provenance)
        return module
    builder.loadmod=load_with_multisource
    original_vertex=builder.vertex_source_outward_normals
    audit_holder={}
    def checked_vertex_normals(candidate,source303,detail,base,orientation,target_rows):
        normals,provenance=original_vertex(candidate,source303,detail,base,orientation,target_rows)
        result=current_source_alignment(candidate['points_world_m'],candidate['source_O_faces'],target_rows,normals,orientation.get('coarse_outward_sign',0))
        orient=json.loads(Path(source303.ORIENTATION).read_text())
        fine=orient['component_outward_sign_from_layer_order']['0']; parity=orient['component_fine_to_coarse_winding_anchors']['0']
        audit={'schema':'numi.surface-clearance-546.source-current-normal-agreement.589.v1',
          'classification':'PARTIAL target-row orientation preflight for unadmitted target proposals; no candidate relocation or physical step',
          'candidate_path':str(candidate_path),'candidate_sha256':sha(candidate_path),
          'source_orientation_report_path':str(Path(source303.ORIENTATION).resolve()),
          'source_orientation_report_sha256':sha(source303.ORIENTATION),
          'fine_outward_sign':int(fine['chosen_outward_sign']),'fine_outward_anchor_count':int(fine['anchor_count']),
          'fine_outward_agreement_fraction':float(fine['agreement_fraction']),
          'fine_to_coarse_winding_parity':int(parity['chosen_outward_parity']),'parity_anchor_count':int(parity['anchor_count']),
          'parity_agreement_fraction':float(parity['agreement_fraction']),
          'orientation_owner_derived_coarse_outward_sign':int(orientation['coarse_outward_sign']),**result}
        audit_holder.update(audit)
        for record in result['rows']:
            provenance.setdefault(int(record['query_row']),{})['current_source_outward_agreement']=record
        if result['status']!='PASS':
            audit_path.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
            raise ValueError('current/source outward-normal disagreement retained at '+','.join(str(x['query_row']) for x in result['witnesses']))
        return normals,provenance
    builder.vertex_source_outward_normals=checked_vertex_normals
    try:
        builder.main(argv)
    except Exception:
        # If the agreement gate itself wrote a witness, preserve it exactly and propagate failure.
        raise
    report_path=out/'nearest-union-exits.json'; npz_path=out/'nearest-union-exits.npz'
    if not report_path.is_file() or not npz_path.is_file(): raise RuntimeError('545 did not emit complete target report and NPZ')
    report=json.loads(report_path.read_text()); target_archive=np.load(npz_path,allow_pickle=False); target_validation=validate_545_depth_maximizing_report(report,target_archive)
    if audit_holder:
        audit_path.write_text(json.dumps(audit_holder,indent=2,allow_nan=False)+'\n')
    serial=json.dumps(face_provenance,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    report['source_current_outward_normal_guard']={'status':'PASS_CURRENT_SOURCE_OUTWARD_AGREEMENT',
      'audit_path':str(audit_path),'audit_sha256':sha(audit_path),'normal_audit_588_path':str(NORMAL_AUDIT_SCRIPT),
      'normal_audit_588_sha256':sha(NORMAL_AUDIT_SCRIPT),'agreement_gate':'area-weighted current emitted-F32 face normals multiplied by pinned O-to-I coarse outward sign must have nonnegative dot with the registered source-outward vertex normal',
      'nonagreement_policy':'fail closed and retain per-row witness; no nearest/depth fallback',
      'current_source_row_count':audit_holder.get('target_row_count',0),'current_source_agreement_count':audit_holder.get('agreement_count',0),
      'current_source_disagreement_count':audit_holder.get('disagreement_count',0),
      'coarse_outward_sign':audit_holder.get('coarse_outward_sign'),
      'orientation_receipt_sha256':audit_holder.get('source_orientation_report_sha256')}
    report['multi_parent_source_normal_extension']={'adapter_path':str(Path(__file__).resolve()),'adapter_sha256':sha(__file__),
      'pinned_545_builder_path':str(BUILDER),'pinned_545_builder_sha256':BUILDER_SHA,
      'pinned_546_extension_path':str(EXT),'pinned_546_extension_sha256':EXT_SHA,
      'pinned_500_owner_path':str(OWNER500),'pinned_500_owner_sha256':OWNER500_SHA,
      'single_parent_303_function_preserved':True,'multi_parent_face_rows':sorted(face_provenance),
      'multi_parent_face_count':len(face_provenance),'multi_parent_provenance_sha256':hashlib.sha256(serial).hexdigest(),
      'exact_185_bank_ray_and_endpoint_owner_unchanged':True,'direction_mode':'depth-maximizing',
      'source_and_depth_minimum_cosine':MINIMUM_COSINE,'native_admission':False,'accepted_physical_steps':0}
    report['source_outward_target_validation_589']=target_validation
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(report_path),'report_sha256':sha(report_path),'target_npz':str(npz_path),'target_npz_sha256':sha(npz_path),
      'current_source_audit':str(audit_path),'current_source_audit_sha256':sha(audit_path),'status':report['status'],**target_validation},indent=2))
if __name__=='__main__': main()
