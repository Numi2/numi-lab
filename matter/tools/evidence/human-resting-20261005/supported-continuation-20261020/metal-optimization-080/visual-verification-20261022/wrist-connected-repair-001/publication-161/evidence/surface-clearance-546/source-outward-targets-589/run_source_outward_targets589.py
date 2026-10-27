#!/usr/bin/env python3
"""Source-outward target adapter for 585-compatible candidates, with current/source sign gate.

The pinned 545 owner still chooses and certifies each first empty interval over all 185 banks.
The only added gate checks that its registered source-outward node normal agrees with the
current emitted surface normal after applying the O-to-I-anchored coarse-winding sign.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
BUILDER=ROOT/'D/contained-clearance-545/build_direct_depth_targets545.py'
BUILDER_SHA='2864d9547d2715c61c9c27b17bcfd11fac250197c47f4fd4870f8f6898cecb60'
EXT=ROOT/'D/source-outward-proposal-500/build_outward_targets_multi_source_546.py'
EXT_SHA='dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68'
OWNER500=ROOT/'D/source-outward-proposal-500/build_outward_nearest_union_targets_500.py'
OWNER500_SHA='454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf'
NORMAL_AUDIT_SCRIPT=ROOT/'D/surface-clearance-546/source-normal-audit-588/audit_midpoint_normals588.py'
NORMAL_AUDIT_SHA='a6f7b8c1cd333e117f9a8f6f31bb9cc40f8bf5569c6f01ea7cc2074a99c3b703'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def loadmod(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(f'cannot load {path}')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def require_source_outward(argv):
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--direction-mode',choices=('source-outward','depth-compatible','depth-maximizing','direct-depth'),default='source-outward')
    ns,_=ap.parse_known_args(list(argv))
    if ns.direction_mode!='source-outward': raise ValueError('589 adapter requires --direction-mode source-outward')
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

def validate_545_source_outward_report(report):
    if report.get('direction_mode')!='source-outward': raise ValueError('545 report is not source-outward mode')
    if report.get('physical_steps')!=0: raise ValueError('source-outward target owner reports physical steps')
    records=report.get('targets')
    if not isinstance(records,list) or len(records)!=int(report.get('all_bank_ray_results',-1)):
        raise ValueError('545 target record count differs from certified ray result count')
    passed=[]
    for item in records:
        if item.get('status')=='PASS_FIRST_UNION_EXIT':
            if item.get('target_exactly_outside_all185') is not True or item.get('exact_endpoint_bank_count')!=185:
                raise ValueError(f"target row {item.get('query_row')} lacks complete all185 endpoint certificate")
            passed.append(item)
    if len(passed)!=int(report.get('passing_targets',-1)):
        raise ValueError('545 passing target count differs from target records')
    return {'target_count':len(records),'passing_targets':len(passed),
            'all_passing_endpoints_exactly_outside_185':True if len(passed)==int(report.get('passing_targets',-1)) else False,
            'physical_steps':0}

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
    require_source_outward(argv)
    for path,expected,label in ((BUILDER,BUILDER_SHA,'545 builder'),(EXT,EXT_SHA,'546 multiparent helper'),(OWNER500,OWNER500_SHA,'500 source normal owner'),(NORMAL_AUDIT_SCRIPT,NORMAL_AUDIT_SHA,'588 orientation audit')):
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
          'classification':'orientation preflight for unadmitted target proposals; no candidate relocation or physical step',
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
    report=json.loads(report_path.read_text()); target_validation=validate_545_source_outward_report(report)
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
      'exact_185_bank_ray_and_endpoint_owner_unchanged':True,'direction_mode':'source-outward',
      'native_admission':False,'accepted_physical_steps':0}
    report['source_outward_target_validation_589']=target_validation
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(report_path),'report_sha256':sha(report_path),'target_npz':str(npz_path),'target_npz_sha256':sha(npz_path),
      'current_source_audit':str(audit_path),'current_source_audit_sha256':sha(audit_path),'status':report['status'],**target_validation},indent=2))
if __name__=='__main__': main()
