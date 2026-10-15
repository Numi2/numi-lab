from pathlib import Path
import sys, json, hashlib
import numpy as np
base=Path('/Users/home/cardboard-implementation-20261014')
root=base/'analysis/bilinear15-fp64-replay-267'
sys.path.insert(0,str(root))
import bilinear15_reference as ref
npz=np.load(root/'early-output/arrays.npz')
cap=Path('/Users/home/cardboard-implementation-20261014/solver/sourcefit15-fg32-control1-capture-cycle-258/operator-capture')
records=[json.loads(x) for x in cap.read_text().splitlines() if x.strip()]
header=records[0]
tets=[x for x in records if x.get('record')=='tetrahedron' and x.get('material_index') in (0,1)]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prodnorm(p,f,old,z):
    rr=ref.residual(p,f,old,z)
    # The authored program multiplies every implicit residual by 10.
    n=10.0*np.max(np.abs(rr))
    sn=np.max(np.abs(z))
    return n,sn,rr

def project_match_prod(p,f,old,zinit,tol,maxiter=80):
    z=zinit.copy()
    counts=0
    for k in range(maxiter+1):
        rn,sn,r=prodnorm(p,f,old,z)
        if rn <= tol*(1.0+sn): return z,k,rn,sn
        J=ref.jacobian(p,f,old,z)
        d=np.linalg.solve(J,-r)
        accepted=False
        a=1.0
        for _ in range(30):
            trial=z+a*d
            trn,_,_=prodnorm(p,f,old,trial)
            if np.isfinite(trn) and trn<rn:
                z=trial;accepted=True;break
            a*=0.5
        if not accepted: raise RuntimeError(f'line search failed iter{k} norm{rn} tol{tol}')
    raise RuntimeError(f'iteration limit tol{tol}')

def normrel(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))

# Validate the independent equations against each authored constitutive row at
# the captured candidate using the captured FP64-representable words.
recon=[]
for i,t in enumerate(tets):
    p=npz['p'][i];f=npz['F'][i];old=npz['old'][i];z=npz['captured_candidate'][i]
    rn,sn,rr=prodnorm(p,f,old,z)
    prod=t['center']['projection']
    reported=float(prod[0])
    if reported>0 and rn>0:
        ratio=rn/reported
    else: ratio=None
    recon.append({'array_index':i,'tet_id':int(t['tet_id']),'environment':int(t['environment']),'material_index':int(t['material_index']),'reference_residual_norm_production_scaled':rn,'captured_shader_projection_norm':reported,'ratio':ratio,'state_norm':sn,'production_threshold':2e-8*(1+sn),'threshold_fraction_shader':reported/(2e-8*(1+sn)) if reported>0 else 0.0,'reference_unscaled_residual_max':float(np.max(np.abs(rr))),'reference_yield_residual':float(rr[12])})

# Rows with nonzero production/reference projected tangent and large changed-root error.
TN=npz['nativeT'];TR=npz['referenceProjectedT'];
absrow=np.max(np.abs(TN-TR),axis=(1,2)); refrow=np.max(np.abs(TR),axis=(1,2));
relrow=absrow/np.maximum(refrow,1e-300)
active=np.flatnonzero(refrow>1e-8)
order=active[np.argsort(relrow[active])[::-1]]
selected=[]
for i in order[:24]:
    t=tets[int(i)]; p=npz['p'][i]; f=npz['F'][i];old=npz['old'][i];zc=npz['captured_candidate'][i];h=npz['direction'][i]
    q,flow=ref.hill(p,zc[6:12]);dl=float(zc[12]-old[12]); y=float(1+p[17]*p[19]*p[21]*zc[13]/p[10]+p[18]*p[20]*p[22]*zc[14]/p[10]-q)
    sv=np.linalg.svd(ref.jacobian(p,f,old,zc),compute_uv=False)
    selected.append({'array_index':int(i),'tet_id':int(t['tet_id']),'environment':int(t['environment']),'material_index':int(t['material_index']),'relative_max_component_error':float(relrow[i]),'absolute_max_component_error':float(absrow[i]),'native_tangent_abs_max':float(np.max(np.abs(TN[i]))),'reference_projected_tangent_abs_max':float(refrow[i]),'captured_projection_norm_shader':float(t['center']['projection'][0]),'candidate_state_norm_shader':float(t['center']['projection'][1]),'captured_state_yield_y':y,'captured_delta_multiplier':dl,'captured_fb_epsilon':float(p[26]),'captured_fb_residual_unscaled':float(ref.residual(p,f,old,zc)[12]),'candidate_jacobian_condition_2':float(sv[0]/sv[-1]),'candidate_jacobian_sigma_min':float(sv[-1]),'captured_candidate':zc.tolist(),'reference_projected_state':npz['projected_candidate'][i].tolist(),'direction':h.tolist(),'native_tangent':TN[i].tolist(),'reference_tangent_at_capture':npz['referenceT'][i].tolist(),'reference_tangent_at_projected_root':TR[i].tolist()})

# Tighten FP64 stop test from the same float32-captured candidate on worst rows.
sweep=[]
for i in order[:8]:
    p=npz['p'][i];f=npz['F'][i];old=npz['old'][i];zcap=npz['captured_candidate'][i];h=npz['direction'][i];tn=TN[i]
    t=tets[int(i)]
    for tol in [2.0e-8,2.0e-9,2.0e-10,2.0e-11,2.0e-12,2.0e-13,2.0e-14]:
        z,k,rn,sn=project_match_prod(p,f,old,zcap,tol)
        tj=ref.tangent(p,f,old,z,h)
        q,flow=ref.hill(p,z[6:12]);dl=float(z[12]-old[12]);y=float(1+p[17]*p[19]*p[21]*z[13]/p[10]+p[18]*p[20]*p[22]*z[14]/p[10]-q)
        sweep.append({'array_index':int(i),'tet_id':int(t['tet_id']),'environment':int(t['environment']),'material_index':int(t['material_index']),'target_production_scaled_tolerance':tol,'iterations':k,'final_production_scaled_residual':rn,'state_norm':sn,'threshold':tol*(1+sn),'tangent_relative_l2_to_captured_native':normrel(tj,tn),'tangent_relative_l2_to_fp64_captured_candidate':normrel(tj,npz['referenceT'][i]),'tangent_relative_l2_to_tightest':None,'max_tangent_component_abs':float(np.max(np.abs(tj))),'max_tangent_component_error_vs_native':float(np.max(np.abs(tj-tn))),'q':float(q),'yield_gap_y':y,'delta_multiplier':dl,'fb_residual_unscaled':float(ref.residual(p,f,old,z)[12]),'projected_state':z.tolist()})
# annotate convergence-to-tightest per selected row
for i in order[:8]:
    rr=[x for x in sweep if x['array_index']==int(i)]
    final=rr[-1]['tangent_relative_l2_to_captured_native']
    for x in rr: x['tangent_relative_l2_to_tightest']=normrel(ref.tangent(npz['p'][i],npz['F'][i],npz['old'][i],np.array(x['projected_state']),npz['direction'][i]), ref.tangent(npz['p'][i],npz['F'][i],npz['old'][i],np.array(rr[-1]['projected_state']),npz['direction'][i]))

pred=base/'solver/sourcefit15-fg32-control1-capture-cycle-258/prediction.json'
material_dir=base/'snapshots/source-stateless-compensated-and-validation-admission-077/matter/materials'
assets=[material_dir/'hajali2009_liner_hill_bilinear_fit.nmatter',material_dir/'hajali2009_medium_hill_bilinear_fit.nmatter']
source_common=base/'snapshots/source-stateless-compensated-and-validation-admission-077/matter/src/metal/common.metalinc'
report={'schema':'numi.cardboard.bilinear15-local-accuracy-diagnosis.v1','capture':str(cap),'capture_sha256':sha(cap),'prediction_sha256':sha(pred),'source_snapshot':'/Users/home/cardboard-implementation-20261014/snapshots/source-stateless-compensated-and-validation-admission-077','authored_material_sha256':{p.name:sha(p) for p in assets},'production_projector_sha256':sha(source_common),'independent_reference_sha256':sha(root/'bilinear15_reference.py'),'capture_records':len(records),'selected_rows':selected,'projection_norm_reconciliation':{'count':len(recon),'max_ratio_ref_scaled_to_shader':max((x['ratio'] for x in recon if x['ratio'] is not None),default=None),'median_ratio':float(np.median([x['ratio'] for x in recon if x['ratio'] is not None])),'max_shader_threshold_fraction':max(x['threshold_fraction_shader'] for x in recon),'fraction_within_reported_production_threshold':sum(x['threshold_fraction_shader']<=1.0 for x in recon)/len(recon),'rows_near_production_stop_gt_0.5':sum(x['threshold_fraction_shader']>0.5 for x in recon),'rows_over_production_stop':sum(x['threshold_fraction_shader']>1.0 for x in recon),'worst_threshold_rows':sorted(recon,key=lambda x:x['threshold_fraction_shader'],reverse=True)[:20]},'tightening_sweep':sweep,'narrow_question':'Does the projected-root tangent converge away from the production candidate at tolerances tighter than the authored production acceptance threshold?','classification':'For N0 control1 only; this is equation/projection analysis, not a general material or coupon qualification.','wall_seconds':None}
out=Path(__file__).parent/'report.json';out.write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda o: int(o) if isinstance(o,np.integer) else float(o) if isinstance(o,np.floating) else o)+'\n')
print(json.dumps({k:report[k] for k in ['projection_norm_reconciliation','tightening_sweep']},indent=2,default=lambda o: int(o) if isinstance(o,np.integer) else float(o) if isinstance(o,np.floating) else o))
