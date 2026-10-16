"""Independent, batched FP64 15-state Hill/tensile-hardening equations.
No production compiler, bytecode, local projector, or derivative programs.
State order: ep(6), normalized second Piola(6), multiplier, MD/CD histories.
"""
import numpy as np

def stiffness(p):
    c=np.zeros(p.shape[:-1]+(6,6),dtype=p.dtype)
    c[...,0,0]=p[...,1];c[...,1,1]=p[...,2];c[...,2,2]=p[...,3]
    for i,j,k in [(0,1,4),(0,2,5),(1,2,6)]:c[...,i,j]=c[...,j,i]=p[...,k]
    c[...,3,3]=2*p[...,9];c[...,4,4]=2*p[...,8];c[...,5,5]=2*p[...,7]
    return c/p[...,10,None,None]

def strain(f):
    # Algebraically identical Green strain, avoiding a near-unit F.T@F-I subtraction.
    d = f - np.eye(3, dtype=f.dtype)
    e = (d + np.swapaxes(d, -1, -2) + np.einsum('...ki,...kj->...ij', d, d)) / 2
    return e[...,(0,1,2,1,0,0),(0,1,2,2,2,1)]

def hill(p,s):
    f,g,h,l,m,n=[p[...,i] for i in range(11,17)]
    a,b,c,d,e,v=np.moveaxis(s,-1,0)
    q=np.sqrt(f*(b-c)**2+g*(c-a)**2+h*(a-b)**2+2*l*d*d+2*m*e*e+2*n*v*v+p[...,25]**2)
    flow=np.stack([g*(a-c)+h*(a-b),f*(b-c)+h*(b-a),f*(c-b)+g*(c-a),l*d,m*e,n*v],axis=-1)/q[...,None]
    return q,flow

def residual(p,f,old,z):
    s=z[...,6:12];q,flow=hill(p,s);dl=z[...,12]-old[...,12]
    r=np.empty_like(z,dtype=np.result_type(p,f,old,z))
    r[...,:6]=z[...,:6]-old[...,:6]-dl[...,None]*flow
    r[...,6:12]=s-np.einsum('...ij,...j->...i',stiffness(p),strain(f)-z[...,:6])
    # The common nondimensionalization cancels algebraically. Evaluate the
    # physical positive-stress weights directly in FP64.
    pos=(s[...,:3]+np.sqrt(s[...,:3]**2+p[...,23,None]**2))/2
    w=pos**2/(np.sum(pos**2,axis=-1)+p[...,25]**2)[...,None]
    r[...,13:15]=z[...,13:15]-old[...,13:15]-dl[...,None]*w[...,:2]
    y=1+p[...,17]*p[...,19]*p[...,21]*z[...,13]/p[...,10]+p[...,18]*p[...,20]*p[...,22]*z[...,14]/p[...,10]-q
    root=np.sqrt(dl*dl+y*y+p[...,26]**2)
    # Rationalize FB near elastic or plastic complementarity to retain tiny
    # residuals. jacobian() differentiates the authored FB row directly.
    den=root+dl+y
    r[...,12]=np.where(np.real(den)>1e-14,(p[...,26]**2-2*dl*y)/den,root-dl-y)
    return r

def stress(p,f,z):
    s=p[...,10,None]*np.einsum('...ij,...j->...i',stiffness(p),strain(f)-z[...,:6])
    second=np.empty(f.shape,dtype=np.result_type(f,z))
    second[...,0,0]=s[...,0];second[...,1,1]=s[...,1];second[...,2,2]=s[...,2]
    second[...,1,2]=second[...,2,1]=s[...,3]
    second[...,0,2]=second[...,2,0]=s[...,4]
    second[...,0,1]=second[...,1,0]=s[...,5]
    return f@second

def jacobian(p,f,old,z):
    j=np.empty(z.shape+(15,),dtype=float)
    for col in range(15):
        q=z.astype(complex);q[...,col]+=1e-25j
        j[...,col]=np.imag(residual(p,f,old,q))/1e-25
    # Keep the rationalized Fischer-Burmeister residual value above, but do
    # not complex-step differentiate its quotient near the cancellation
    # threshold. Differentiate the authored phi=sqrt(dl^2+y^2+eps^2)-dl-y
    # directly; its row is smooth because fb_epsilon is positive.
    s=z[...,6:12]
    dl=z[...,12]-old[...,12]
    q,flow=hill(p,s)
    y=(1+p[...,17]*p[...,19]*p[...,21]*z[...,13]/p[...,10]
       +p[...,18]*p[...,20]*p[...,22]*z[...,14]/p[...,10]-q)
    eps=p[...,26]
    root=np.sqrt(dl*dl+y*y+eps*eps)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        coeff_dl=dl/root-1.0
        positive_dl=dl>0.0
        denominator_dl=root*(root+dl)
        stable_dl=np.divide(-(y*y+eps*eps),denominator_dl,
                            out=np.full_like(root,np.nan),where=denominator_dl>0.0)
        coeff_dl=np.where(positive_dl,stable_dl,coeff_dl)

        coeff_y=y/root-1.0
        positive_y=y>0.0
        denominator_y=root*(root+y)
        stable_y=np.divide(-(dl*dl+eps*eps),denominator_y,
                           out=np.full_like(root,np.nan),where=denominator_y>0.0)
        coeff_y=np.where(positive_y,stable_y,coeff_y)

    fb_row=np.zeros_like(z)
    fb_row[...,12]=coeff_dl
    # Hill's flow helper uses the authored shear half-gradient convention;
    # y=-q therefore has twice those shear derivatives.
    fb_row[...,6:9]=-coeff_y[...,None]*flow[...,0:3]
    fb_row[...,9:12]=-2.0*coeff_y[...,None]*flow[...,3:6]
    fb_row[...,13]=coeff_y*p[...,17]*p[...,19]*p[...,21]/p[...,10]
    fb_row[...,14]=coeff_y*p[...,18]*p[...,20]*p[...,22]/p[...,10]
    j[...,12,:]=fb_row
    return j

def project(p,f,old,maxiter=80):
    z=old.copy();z[...,6:12]=np.einsum('...ij,...j->...i',stiffness(p),strain(f)-old[...,:6])
    counts=np.zeros(z.shape[:-1],dtype=int);scale=np.array([1e4]*6+[1]*6+[1e4]*3)
    for iteration in range(maxiter):
        r=residual(p,f,old,z);norm=np.max(np.abs(r)*scale,axis=-1); active=norm>2e-10
        if not active.any():return z,counts,norm
        ids=np.flatnonzero(active);pa=p[ids];fa=f[ids];oa=old[ids];za=z[ids]
        j=jacobian(pa,fa,oa,za);d=np.linalg.solve(j,-r[ids,...,None])[...,0]
        alpha=np.ones(len(ids));pending=np.ones(len(ids),dtype=bool);trial=za.copy()
        for bt in range(30):
            ti=za+alpha[:,None]*d;tn=np.max(np.abs(residual(pa,fa,oa,ti))*scale,axis=-1)
            ok=pending&np.isfinite(tn)&(tn<norm[ids]);trial[ok]=ti[ok];pending[ok]=False
            if not pending.any():break
            alpha[pending]*=.5
        if pending.any():raise RuntimeError(f'FP64 return map stalled iteration{iteration} rows{ids[pending][:10]} norm{norm[ids[pending]][:10]}')
        z[ids]=trial;counts[ids]+=1
    raise RuntimeError(f'FP64 projection iteration ceiling; worst{norm.max()}')

def tangent(p,f,old,z,h):
    rf=np.imag(residual(p,f.astype(complex)+1e-25j*h,old,z))/1e-25
    dz=np.linalg.solve(jacobian(p,f,old,z),-rf[...,None])[...,0]
    return np.imag(stress(p,f.astype(complex)+1e-25j*h,z.astype(complex)+1e-25j*dz))/1e-25


def project_resilient(ref,p,F,old,maxiter=80):
 """Batch project, recursively isolate any failed rows while preserving them."""
 n=len(p); z=np.full((n,15),np.nan); it=np.full(n,-1,dtype=np.int32); rn=np.full(n,np.nan); ok=np.zeros(n,dtype=bool); reasons=[None]*n
 def run(ids):
  try:
   zz,ii,rr=ref.project(p[ids],F[ids],old[ids],maxiter=maxiter)
   if np.all(np.isfinite(zz)) and np.all(np.isfinite(rr)):
    z[ids]=zz; it[ids]=ii; rn[ids]=rr; ok[ids]=True; return
   if len(ids)==1: reasons[ids[0]]='nonfinite projection output'; return
  except Exception as e:
   if len(ids)==1: reasons[ids[0]]=f'{type(e).__name__}: {e}'; return
  mid=len(ids)//2
  run(ids[:mid]); run(ids[mid:])
 run(np.arange(n,dtype=np.int64))
 return z,it,rn,ok,reasons


def project_centers(ref, p, F, old, p26):
    """Project, then enforce source-373's residual and dynamic correction gates."""
    z, iterations, scaled_residual, ok, reasons = project_resilient(ref, p, F, old)
    refinement = []
    residual_scale = np.array([1e4] * 6 + [1.0] * 6 + [1e4] * 3, dtype=np.float64)
    dynamic_correction = np.full(len(p), np.nan)
    for refinement_iteration in range(5):
        ids = np.flatnonzero(ok)
        if ids.size == 0:
            break
        r = ref.residual(p[ids], F[ids], old[ids], z[ids])
        try:
            corr = np.linalg.solve(ref.jacobian(p[ids], F[ids], old[ids], z[ids]),
                                   -r[..., None])[..., 0]
        except Exception:
            # Preserve individual failures instead of losing a whole batch.
            valid_ids = []
            valid_corr = []
            valid_residual = []
            for original in ids:
                try:
                    rr = ref.residual(p[original:original + 1], F[original:original + 1],
                                      old[original:original + 1], z[original:original + 1])
                    jj = ref.jacobian(p[original:original + 1], F[original:original + 1],
                                      old[original:original + 1], z[original:original + 1])
                    cc = np.linalg.solve(jj, -rr[..., None])[0, :, 0]
                    valid_ids.append(original)
                    valid_corr.append(cc)
                    valid_residual.append(rr[0])
                except Exception as e:
                    ok[original] = False
                    reasons[original] = f'FP64 correction solve: {type(e).__name__}: {e}'
            ids = np.asarray(valid_ids, dtype=np.int64)
            corr = np.asarray(valid_corr, dtype=np.float64).reshape(-1, 15)
            r = np.asarray(valid_residual, dtype=np.float64).reshape(-1, 15)
            if ids.size == 0:
                break
        state_scale = np.maximum(np.maximum(np.abs(old[ids]), np.abs(z[ids])),
                                 p26[ids, None])
        corr_norm = np.max(np.abs(corr) / state_scale, axis=1)
        dynamic_correction[ids] = corr_norm
        selected_local = np.flatnonzero(corr_norm > 2e-8)
        refinement.append({
            'iteration': refinement_iteration,
            'rows_above_dynamic_correction_gate': int(selected_local.size),
            'maximum_dynamic_scaled_correction': float(corr_norm.max()) if corr_norm.size else None,
        })
        if selected_local.size == 0:
            break
        if refinement_iteration >= 4:
            for local in selected_local:
                original = ids[local]
                ok[original] = False
                reasons[original] = 'bounded FP64 dynamic-correction refinement exhausted'
            break
        selected = ids[selected_local]
        corr_selected = corr[selected_local]
        baseline = np.max(np.abs(r[selected_local]) * residual_scale, axis=1)
        pending = np.ones(len(selected), dtype=bool)
        alpha = np.ones(len(selected), dtype=np.float64)
        next_z = z[selected].copy()
        for backtrack in range(24):
            candidate = z[selected] + alpha[:, None] * corr_selected
            nr = np.max(np.abs(ref.residual(p[selected], F[selected], old[selected], candidate)) *
                        residual_scale, axis=1)
            accepted = pending & np.isfinite(nr) & (nr < baseline)
            next_z[accepted] = candidate[accepted]
            pending[accepted] = False
            if not np.any(pending):
                break
            alpha[pending] *= 0.5
        for local in np.flatnonzero(pending):
            original = selected[local]
            ok[original] = False
            reasons[original] = 'bounded FP64 residual-descent refinement stalled'
        z[selected[~pending]] = next_z[~pending]
        iterations[selected[~pending]] += 1
    # Re-evaluate both gates at the final returned states.
    ids = np.flatnonzero(ok)
    final_scaled_residual = np.full(len(p), np.nan)
    final_dynamic_correction = np.full(len(p), np.nan)
    if ids.size:
        rr = ref.residual(p[ids], F[ids], old[ids], z[ids])
        final_scaled_residual[ids] = np.max(np.abs(rr) * residual_scale, axis=1)
        jj = ref.jacobian(p[ids], F[ids], old[ids], z[ids])
        cc = np.linalg.solve(jj, -rr[..., None])[..., 0]
        ss = np.maximum(np.maximum(np.abs(old[ids]), np.abs(z[ids])), p26[ids, None])
        final_dynamic_correction[ids] = np.max(np.abs(cc) / ss, axis=1)
        bad = (final_scaled_residual[ids] > 2e-10) | (final_dynamic_correction[ids] > 2e-8)
        for local in np.flatnonzero(bad):
            original = ids[local]
            ok[original] = False
            reasons[original] = 'final FP64 residual/correction gate failed'
    return z, iterations, final_scaled_residual, final_dynamic_correction, ok, reasons, refinement


def paper_branch(ref, p, old, z):
    lam_increment = z[:, 12] - old[:, 12]
    q, _ = ref.hill(p, z[:, 6:12])
    yield_value = (1.0 + p[:, 17] * p[:, 19] * p[:, 21] * z[:, 13] / p[:, 10]
                   + p[:, 18] * p[:, 20] * p[:, 22] * z[:, 14] / p[:, 10] - q)
    return {
        'plastic_multiplier_increment': lam_increment,
        'plastic_active_lambda_positive': lam_increment > 0.0,
        'yield_function_residual_value': yield_value,
        'q_hill': q,
    }


def glue_stress(f, mu, lam):
    invt = np.swapaxes(np.linalg.inv(f), -1, -2)
    det = np.linalg.det(f)
    return mu[:, None, None] * (f - invt) + lam[:, None, None] * np.log(det)[..., None, None] * invt


def glue_tangent_action(f, h, mu, lam):
    inv = np.linalg.inv(f)
    invt = np.swapaxes(inv, -1, -2)
    logj = np.log(np.linalg.det(f))
    tr = np.einsum('nij,nji->n', inv, h)
    second = np.swapaxes(h, -1, -2)
    return (mu[:, None, None] * h + (mu - lam * logj)[:, None, None] * (invt @ second @ invt)
            + lam[:, None, None] * tr[:, None, None] * invt)

TIGHT_RESIDUAL = 2e-12
TIGHT_DYNAMIC_CORRECTION = 2e-11


def continue_projection_tight(ref, p, F, old, p26, base_result):
    """Continue the exact same FP64 return map to stricter diagnostic gates."""
    z, iterations, rn, cn, ok, reasons, base_passes = base_result
    z = np.asarray(z, dtype=np.float64).copy()
    iterations = np.asarray(iterations, dtype=np.int32).copy()
    rn = np.asarray(rn, dtype=np.float64).copy()
    cn = np.asarray(cn, dtype=np.float64).copy()
    ok = np.asarray(ok, dtype=bool).copy()
    reasons = list(reasons)
    residual_scale = np.array([1e4] * 6 + [1.0] * 6 + [1e4] * 3, dtype=np.float64)
    passes = []
    for extra in range(16):
        pending_ids = np.flatnonzero(ok & ((rn > TIGHT_RESIDUAL) | (cn > TIGHT_DYNAMIC_CORRECTION)))
        if pending_ids.size == 0:
            break
        r = ref.residual(p[pending_ids], F[pending_ids], old[pending_ids], z[pending_ids])
        norm = np.max(np.abs(r) * residual_scale, axis=1)
        try:
            j = ref.jacobian(p[pending_ids], F[pending_ids], old[pending_ids], z[pending_ids])
            corr = np.linalg.solve(j, -r[..., None])[..., 0]
        except Exception:
            corr = np.full((len(pending_ids), 15), np.nan)
            for local, row in enumerate(pending_ids):
                try:
                    rr = ref.residual(p[row:row+1], F[row:row+1], old[row:row+1], z[row:row+1])
                    jj = ref.jacobian(p[row:row+1], F[row:row+1], old[row:row+1], z[row:row+1])
                    corr[local] = np.linalg.solve(jj, -rr[..., None])[0, :, 0]
                except Exception as e:
                    ok[row] = False
                    reasons[row] = f'tight diagnostic correction solve: {type(e).__name__}: {e}'
        state_scale = np.maximum(np.maximum(np.abs(old[pending_ids]), np.abs(z[pending_ids])), p26[pending_ids, None])
        dyn = np.max(np.abs(corr) / state_scale, axis=1)
        need = np.isfinite(norm) & np.isfinite(dyn) & ((norm > TIGHT_RESIDUAL) | (dyn > TIGHT_DYNAMIC_CORRECTION))
        if not np.any(need):
            ok[pending_ids[~np.isfinite(norm) | ~np.isfinite(dyn)]] = False
            break
        chosen = pending_ids[need]
        corr_chosen = corr[need]
        base_norm = norm[need]
        alpha = np.ones(len(chosen), dtype=np.float64)
        pending = np.ones(len(chosen), dtype=bool)
        next_z = z[chosen].copy()
        for backtrack in range(32):
            candidate = z[chosen] + alpha[:, None] * corr_chosen
            trial_r = ref.residual(p[chosen], F[chosen], old[chosen], candidate)
            trial_norm = np.max(np.abs(trial_r) * residual_scale, axis=1)
            accept = pending & np.isfinite(trial_norm) & (trial_norm < base_norm)
            next_z[accept] = candidate[accept]
            pending[accept] = False
            if not np.any(pending):
                break
            alpha[pending] *= 0.5
        accepted_ids = chosen[~pending]
        z[accepted_ids] = next_z[~pending]
        iterations[accepted_ids] += 1
        for row in chosen[pending]:
            ok[row] = False
            reasons[row] = f'tight diagnostic residual descent stalled at residual {rn[row]:.17g}, correction {cn[row]:.17g}'
        passes.append({'extra_iteration': extra, 'rows_considered': int(len(pending_ids)),
                       'rows_updated': int(len(accepted_ids)), 'rows_stalled': int(pending.sum()),
                       'max_preupdate_scaled_residual': float(np.max(norm[need])) if np.any(need) else None,
                       'max_preupdate_dynamic_correction': float(np.max(dyn[need])) if np.any(need) else None})
        # Refresh gates on all still-healthy rows, including ones that met both limits already.
        ids = np.flatnonzero(ok)
        if ids.size:
            rr = ref.residual(p[ids], F[ids], old[ids], z[ids])
            rn[ids] = np.max(np.abs(rr) * residual_scale, axis=1)
            jj = ref.jacobian(p[ids], F[ids], old[ids], z[ids])
            cc = np.linalg.solve(jj, -rr[..., None])[..., 0]
            scale = np.maximum(np.maximum(np.abs(old[ids]), np.abs(z[ids])), p26[ids, None])
            cn[ids] = np.max(np.abs(cc) / scale, axis=1)
    # Final recalc and fail closed if either tighter diagnostic criterion remains exceeded.
    ids = np.flatnonzero(ok)
    if ids.size:
        rr = ref.residual(p[ids], F[ids], old[ids], z[ids])
        rn[ids] = np.max(np.abs(rr) * residual_scale, axis=1)
        jj = ref.jacobian(p[ids], F[ids], old[ids], z[ids])
        cc = np.linalg.solve(jj, -rr[..., None])[..., 0]
        scale = np.maximum(np.maximum(np.abs(old[ids]), np.abs(z[ids])), p26[ids, None])
        cn[ids] = np.max(np.abs(cc) / scale, axis=1)
        fail = (rn[ids] > TIGHT_RESIDUAL) | (cn[ids] > TIGHT_DYNAMIC_CORRECTION)
        for local in np.flatnonzero(fail):
            row = ids[local]
            ok[row] = False
            reasons[row] = 'tight diagnostic residual/correction gate failed'
    passes = {'base_projection_passes': base_passes,
              'tight_gate_residual': TIGHT_RESIDUAL,
              'tight_gate_dynamic_correction': TIGHT_DYNAMIC_CORRECTION,
              'tight_continuation': passes}
    return z, iterations, rn, cn, ok, reasons, passes
