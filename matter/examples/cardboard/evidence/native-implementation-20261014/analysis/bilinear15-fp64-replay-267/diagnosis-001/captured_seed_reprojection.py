"""Independent FP64 reproject from exact captured local candidates.
Uses authored 15-state equations and the production residual L-infinity line-search
metric (common residual_scale factor omitted from Newton solve and reinserted in
reported norm). It never calls production compiler/VM/Metal.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
base=Path('/Users/home/cardboard-implementation-20261014')
root=base/'analysis/bilinear15-fp64-replay-267'
sys.path.insert(0,str(root)); import bilinear15_reference as ref

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def words(x):return np.asarray(x,dtype=np.float32).astype(float)
def pair(d,k):return words(d[k])+words(d.get(k+'_low',np.zeros_like(d[k])))
def norm(p,f,old,z):
    r=ref.residual(p,f,old,z)
    # Authored implicit rows all carry residual_scale=10; line search uses
    # their unweighted max absolute value. The factor 10 is retained for
    # comparison to the GPU local projection diagnostic.
    return 10.0*float(np.max(np.abs(r))),r

def project_from_candidate(p,f,old,zc,tol,maxiter=40):
    z=zc.copy();trace=[]
    for it in range(maxiter+1):
        rn,r=norm(p,f,old,z); sn=float(np.max(np.abs(z))); threshold=tol*(1+sn)
        trace.append({'iteration':it,'scaled_residual_inf':rn,'state_inf':sn,'threshold':threshold})
        if rn<=threshold:return z,{'status':'converged','iterations':it,'residual':rn,'state_norm':sn,'trace':trace}
        J=ref.jacobian(p,f,old,z)
        try:d=np.linalg.solve(J,-r)
        except np.linalg.LinAlgError:return z,{'status':'singular_jacobian','iterations':it,'residual':rn,'trace':trace}
        accepted=False; a=1.0; best=(rn,None,None)
        for bt in range(50):
            trial=z+a*d; tn,_=norm(p,f,old,trial)
            if np.isfinite(tn) and tn<best[0]:best=(tn,a,trial.copy())
            if np.isfinite(tn) and tn < rn:
                z=trial;accepted=True;break
            a*=0.5
        if not accepted:
            if best[1] is not None:
                z=best[2];accepted=True
            else:
                return z,{'status':'line_search_stalled','iterations':it,'residual':rn,'trace':trace,'direction_inf':float(np.max(np.abs(d)))}
    return z,{'status':'iteration_limit','iterations':maxiter,'residual':norm(p,f,old,z)[0],'trace':trace}

def rel_l2(x,y): return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-300))
def rel_max(x,y): return float(np.max(np.abs(x-y))/max(np.max(np.abs(y)),1e-300))

def run(capture_path,out_path):
    t0=time.monotonic(); capture=Path(capture_path); records=[json.loads(s) for s in capture.read_text().splitlines() if s.strip()]; h=records[0]
    assert h['schema']=='numi.cardboard.fem-newton-capture.v1' and records[-1]['record']=='integrity' and records[-1]['production_inputs_bitwise_unchanged']
    pars=words(h['parameters_by_environment']); tets=[r for r in records if r.get('record')=='tetrahedron' and r.get('material_index') in (0,1)]
    rows=[]
    for i,t in enumerate(tets):
        c=t['center'];p=pars[t['environment'],28*t['material_index']:28*(t['material_index']+1)];old=words(c['accepted_state']);f=pair(c,'F_local');zc=pair(c,'projected_state');direction=words(c['F_direction_local_per_s']);tn=words(c['tangent_local_Pa_per_s']);
        # Replay at two increasingly strict FP64 residual targets. The second
        # solve starts again from the captured state, not from prior FP64 state.
        z1,d1=project_from_candidate(p,f,old,zc,2e-12)
        z2,d2=project_from_candidate(p,f,old,zc,2e-14)
        tc=ref.tangent(p,f,old,zc,direction)
        t1=ref.tangent(p,f,old,z1,direction) if d1['status']=='converged' else np.full_like(tn,np.nan)
        t2=ref.tangent(p,f,old,z2,direction) if d2['status']=='converged' else np.full_like(tn,np.nan)
        capnorm,_=norm(p,f,old,zc); prodDiag=c.get('projection',c.get('projection_diagnostics'))
        q,_=ref.hill(p,zc[6:12]);y=float(1+p[17]*p[19]*p[21]*zc[13]/p[10]+p[18]*p[20]*p[22]*zc[14]/p[10]+p[18]*p[20]*p[22]*0/p[10]-q)
        dl=float(zc[12]-old[12]);
        row={'array_index':i,'tet_id':int(t['tet_id']),'environment':int(t['environment']),'material_index':int(t['material_index']),'native_projection_report':prodDiag,'fp64_projection_norm_at_candidate':capnorm,'candidate_norm_matches_native_report_ratio':float(capnorm/prodDiag[0]) if prodDiag and prodDiag[0] else None,'target12':d1,'target14':d2,'captured_state_to_refined_state_inf':float(np.max(np.abs(zc-z2))) if d2['status']=='converged' else None,'captured_yield_gap':y,'captured_delta_multiplier':dl,'captured_fb_residual':float(ref.residual(p,f,old,zc)[12]),'captured_tangent_vs_native_relative_l2':rel_l2(tc,tn),'refined_tangent_vs_native_relative_l2':rel_l2(t2,tn) if d2['status']=='converged' else None,'refined_tangent_vs_captured_candidate_relative_l2':rel_l2(t2,tc) if d2['status']=='converged' else None,'tol12_vs_tol14_tangent_relative_l2':rel_l2(t1,t2) if d1['status']=='converged' and d2['status']=='converged' else None,'native_tangent_absmax':float(np.max(np.abs(tn))),'refined_tangent_absmax':float(np.max(np.abs(t2))) if d2['status']=='converged' else None,'candidate':zc.tolist(),'refined_state':z2.tolist() if d2['status']=='converged' else None,'native_tangent':tn.tolist(),'fp64_tangent_at_candidate':tc.tolist(),'fp64_tangent_refined':t2.tolist() if d2['status']=='converged' else None}
        rows.append(row)
    valid=[x for x in rows if x['refined_tangent_vs_native_relative_l2'] is not None]
    diffs=np.asarray([np.asarray(x['native_tangent'])-np.asarray(x['fp64_tangent_refined']) for x in valid])
    refs=np.asarray([x['fp64_tangent_refined'] for x in valid]); natives=np.asarray([x['native_tangent'] for x in valid]);
    cmpcand=np.asarray([np.asarray(x['native_tangent'])-np.asarray(x['fp64_tangent_at_candidate']) for x in valid]);refcand=np.asarray([x['fp64_tangent_at_candidate'] for x in valid])
    rels=np.asarray([x['refined_tangent_vs_native_relative_l2'] for x in valid]); shifts=np.asarray([x['refined_tangent_vs_captured_candidate_relative_l2'] for x in valid])
    report={'schema':'numi.cardboard.bilinear15-captured-seed-fp64-reprojection.v1','capture':str(capture),'capture_sha256':sha(capture),'root':h['root'],'newton_iteration':h['newton_iteration'],'environment_count':h['environment_count'],'equations':'Independent authored 15-state residual/Jacobian/stress, FP64 safeguarded Newton; independent projection seeded by the captured local candidate. L-infinity Armijo uses same unweighted authored residual rows (common residual_scale=10); reported residual restores factor 10. Not a production call.','production_material_assets':{'snapshot':str(base/'snapshots/source-stateless-compensated-and-validation-admission-077'),'liner_sha256':sha(base/'snapshots/source-stateless-compensated-and-validation-admission-077/matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'),'medium_sha256':sha(base/'snapshots/source-stateless-compensated-and-validation-admission-077/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'),'common_metalinc_sha256':sha(base/'snapshots/source-stateless-compensated-and-validation-admission-077/matter/src/metal/common.metalinc')},'reference_sha256':sha(root/'bilinear15_reference.py'),'capture_tetrahedra':len(rows),'tol12_converged':sum(x['target12']['status']=='converged' for x in rows),'tol14_converged':sum(x['target14']['status']=='converged' for x in rows),'tol12_to_tol14_tangent_max_relative_l2':float(max((x['tol12_vs_tol14_tangent_relative_l2'] or 0) for x in valid)),'candidate_equation_norm_vs_shader_report':{'median_ratio':float(np.median([x['candidate_norm_matches_native_report_ratio'] for x in rows if x['candidate_norm_matches_native_report_ratio'] is not None])),'max_abs_ratio_error':float(max(abs(x['candidate_norm_matches_native_report_ratio']-1) for x in rows if x['candidate_norm_matches_native_report_ratio'] is not None)),'fraction_shader_norm_within_2e-8_stop':sum(x['native_projection_report'] and x['native_projection_report'][0]<=2e-8*(1+x['native_projection_report'][1]) for x in rows)/len(rows)},'tangent_metrics':{'captured_point_fp64_vs_native_global_l2_relative':float(np.linalg.norm(cmpcand)/np.linalg.norm(refcand)),'refined_root_fp64_vs_native_global_l2_relative':float(np.linalg.norm(diffs)/np.linalg.norm(refs)),'max_per_tet_l2_relative_captured_fp64_vs_native':float(max(x['captured_tangent_vs_native_relative_l2'] for x in valid)),'max_per_tet_l2_relative_refined_vs_native':float(max(rels)),'median_per_tet_l2_relative_refined_vs_native':float(np.median(rels)),'max_per_tet_l2_relative_refined_vs_captured_candidate':float(max(shifts)),'median_per_tet_l2_relative_refined_vs_captured_candidate':float(np.median(shifts)),'worst_rows_by_projection_sensitivity':[x for x in sorted(valid,key=lambda x:x['refined_tangent_vs_captured_candidate_relative_l2'],reverse=True)[:20]],'worst_rows_by_native_refined_disagreement':[x for x in sorted(valid,key=lambda x:x['refined_tangent_vs_native_relative_l2'],reverse=True)[:20]]},'limits':['This reprojects from the captured state; it diagnoses local equation accuracy, not independent solver basin selection.','No constitutive-law, full coupon, or physical validation claim.','The native capture records Newton63 linearization inputs, but the native FE global solve failed its control after this linearization; this does not diagnose the global convergence failure.'],'wall_seconds':time.monotonic()-t0}
    out=Path(out_path);out.mkdir(parents=True,exist_ok=False);(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');np.savez_compressed(out/'rows.npz',native=np.asarray([x['native_tangent'] for x in valid]),fp64_capture=np.asarray([x['fp64_tangent_at_candidate'] for x in valid]),fp64_refined=np.asarray([x['fp64_tangent_refined'] for x in valid]),states_capture=np.asarray([x['candidate'] for x in valid]),states_refined=np.asarray([x['refined_state'] for x in valid]));return report
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('capture');ap.add_argument('output');args=ap.parse_args();print(json.dumps(run(args.capture,args.output),indent=2,default=lambda o:int(o) if isinstance(o,np.integer) else float(o) if isinstance(o,np.floating) else o))
