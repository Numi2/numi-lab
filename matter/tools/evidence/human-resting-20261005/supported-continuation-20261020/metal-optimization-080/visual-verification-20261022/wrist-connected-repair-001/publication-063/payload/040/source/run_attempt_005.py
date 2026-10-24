#!/usr/bin/env python3
"""Bounded shared-column retry with Neo-Hookean energy/Piola guards.
Candidate construction only; emits F32 audit metrics and no admission claim.
"""
from pathlib import Path
import hashlib, json, time, sys
import numpy as np
from scipy.optimize import minimize
from column_fit_support import quality_and_column_jacobian, J_and_column_jacobian, plane_constraint_slack_and_jacobian, surface_edge_pairs
from neo_hookean_fit_support import prepare_reference_tetrahedra, evaluate_neo_hookean_batch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ATT=HERE/'attempt-005'
INP=ATT/'retry-input.npz'
AP=ROOT/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz'
PP=ROOT/'native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz'
REF=ROOT/'root-reference-relative-j-062/source-preflight/identity-reference-points.csv'
HELPER=HERE/'neo_hookean_fit_support.py'
BASE_SCRIPT=HERE/'column_fit_support.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_ref(path,n):
    import csv
    pts=np.empty((n,3),np.float64); seen=np.zeros(n,bool)
    with open(path,newline='') as f:
        for r in csv.DictReader(f):
            i=int(r['local_volume_node'])
            if i<0 or i>=n or seen[i]: raise ValueError('reference row IDs invalid')
            pts[i]=[float(r['x_m']),float(r['y_m']),float(r['z_m'])];seen[i]=True
    if not np.all(seen) or not np.all(np.isfinite(pts)):raise ValueError('reference incomplete/nonfinite')
    return pts

def quality(X,T):
    V=X[T]; u=V[:,1]-V[:,0];v=V[:,2]-V[:,0];w=V[:,3]-V[:,0]
    det=np.einsum('ij,ij->i',u,np.cross(v,w))
    e=np.stack([np.sum((V[:,i]-V[:,j])**2,axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))],axis=1)
    return det/np.maximum(e.max(axis=1),1e-30)**1.5,det

def jac_scatter(local_jac, ids, lookup, ncol):
    """Scatter [tet,slot,xyz] to dense [tet,3*ncol], ignoring fixed slots."""
    n=len(ids); out=np.zeros((n,3*ncol),np.float64); rr=np.arange(n)
    for s in range(ids.shape[1]):
        li=lookup[ids[:,s]]; ok=li>=0
        if np.any(ok):
            for a in range(3): np.add.at(out,(rr[ok],3*li[ok]+a),local_jac[ok,s,a])
    return out

def layer_values(X, Xbase):
    O0,I0,D0=Xbase[:2411],Xbase[2411:4822],Xbase[4822:]
    so=O0-I0; fa=I0-D0
    sn=so/np.maximum(np.linalg.norm(so,axis=1),1e-30)[:,None]
    fn=fa/np.maximum(np.linalg.norm(fa,axis=1),1e-30)[:,None]
    s=np.einsum('ij,ij->i',X[:2411]-X[2411:4822],sn)
    f=np.einsum('ij,ij->i',X[2411:4822]-X[4822:],fn)
    s0=np.linalg.norm(so,axis=1);f0=np.linalg.norm(fa,axis=1)
    return s,f,np.minimum(.2*s0,.0001),np.minimum(.2*f0,.0001)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--budget-s',type=float,default=60);p.add_argument('--maxiter',type=int,default=45);p.add_argument('--max-displacement-mm',type=float,default=15);p.add_argument('--output',default='fit-001');args=p.parse_args()
    out=ATT/args.output;out.mkdir(parents=True,exist_ok=True)
    z=np.load(INP,allow_pickle=False); APz=np.load(AP,allow_pickle=False); PPz=np.load(PP,allow_pickle=False)
    Xbase=np.asarray(z['points_baseline_f32'],np.float32).astype(np.float64)
    Xstart=np.asarray(z['points_start_pass_only_f32'],np.float32).astype(np.float64)
    T=np.asarray(z['tetrahedra'],np.int64); mat=np.asarray(z['material_ids'],np.int64); F=np.asarray(z['O_faces'],np.int64)
    Ref=read_ref(REF,len(Xbase));
    if Xbase.shape!=(7233,3) or Xstart.shape!=Xbase.shape or T.shape!=(28632,4) or mat.shape!=(len(T),):raise ValueError('unexpected pinned FEM dimensions')
    if not np.array_equal(T,PPz['tetrahedra']) or not np.array_equal(mat,PPz['material_ids']):raise ValueError('topology/material source mismatch')
    if not np.array_equal(Xbase.astype(np.float32),APz['points']):raise ValueError('attempt005/AP current point mismatch')
    if not np.array_equal(APz['global_nodes'],z['global_nodes']):raise ValueError('global node identity mismatch')
    source_ref=Ref[T]; ref_full=prepare_reference_tetrahedra(source_ref)
    qo,do=quality(Xstart,T)
    if np.any(do<=0) or np.any(~np.isfinite(qo)):raise ValueError('pass-only start has nonpositive/nonfinite baseline geometry')
    # A common translation is the only variable per O/I/D column. Reject a start
    # whose existing layers do not already share that displacement.
    dcols=[]
    for c in range(2411):
        ds=np.stack([Xstart[l*2411+c]-Xbase[l*2411+c] for l in range(3)])
        if np.max(np.linalg.norm(ds-ds[0],axis=1))>2.0e-7:raise ValueError(f'pass-only column {c} has non-common layer displacement')
        dcols.append(ds[0]*1000.)
    dcols=np.asarray(dcols)
    # Static layer floors are checked on the pass-only start; shared translations
    # preserve them up to F32 emission rounding.
    L0=layer_values(Xbase,Xbase);Ls=layer_values(Xstart,Xbase)
    if np.any(Ls[0]<Ls[2]-2e-8) or np.any(Ls[1]<Ls[3]-2e-8):raise ValueError('pass-only start violates a thickness floor no shared translation can repair')
    ro=np.asarray(z['region_column_offsets'],np.int64); rc=np.asarray(z['region_columns'],np.int64)
    rto=np.asarray(z['region_tet_offsets'],np.int64); rt=np.asarray(z['region_tet_rows'],np.int64)
    reo=np.asarray(z['region_endpoint_offsets'],np.int64); re=np.asarray(z['region_endpoint_rows'],np.int64)
    epcol=np.asarray(z['endpoint_column_rows'],np.int64); epnode=np.asarray(z['constraint_endpoint_node_rows'],np.int64)
    hit=np.asarray(z['constraint_hit_points_world_m'],np.float64); norm=np.asarray(z['constraint_normals_world'],np.float64); margin=np.asarray(z['constraint_margins_m'],np.float64)
    nreg=len(ro)-1
    if not(len(rto)==len(reo)==nreg+1):raise ValueError('region ragged arrays misaligned')
    current_mm=Xstart*1000.; reports=[]; start=time.monotonic(); budget_stop=False
    # Interior guard targets leave headroom for final F32 emission.
    qguard=1.25e-5; jlo=.355; jhi=2.495; wlim=4.99e6; plim=4.99e6
    order=sorted(range(nreg),key=lambda i:(ro[i+1]-ro[i],i))
    for ri in order:
        if time.monotonic()-start>=args.budget_s: budget_stop=True; break
        cols=rc[ro[ri]:ro[ri+1]].astype(np.int64); trows=rt[rto[ri]:rto[ri+1]].astype(np.int64); erows=re[reo[ri]:reo[ri+1]].astype(np.int64)
        if len(cols)==0 or len(trows)==0:continue
        # Region definition must cover every corner that is movable here and no
        # cell may couple two separately fitted regions.
        colset=np.zeros(2411,bool);colset[cols]=True
        if np.any(~colset[np.unique(T[trows]%2411)] & np.isin(np.unique(T[trows]%2411),np.concatenate([rc[ro[j]:ro[j+1]] for j in range(nreg) if j!=ri]))):
            raise ValueError(f'region {ri} overlaps another movable column in its tet support')
        lookup=np.full(2411,-1,np.int64);lookup[cols]=np.arange(len(cols))
        row_to_local=np.full(len(Xbase),-1,np.int64)
        for layer in range(3):row_to_local[layer*2411+cols]=np.arange(len(cols))
        erows=erows[(epcol[erows]>=0)&(epcol[erows]<2411)]
        if np.any(~np.isin(epcol[erows],cols)):raise ValueError(f'region {ri} endpoint rows cross its free-column set')
        en=epnode[erows]; ph=hit[erows]; pn=norm[erows]; pm=margin[erows]
        # Preserve the pass-only correction as the warm-start reference.
        start_delta=dcols[cols].copy()
        Xanchor=current_mm.copy()
        def assemble(x):
            X=Xanchor.copy(); dx=np.asarray(x,np.float64).reshape(len(cols),3)-start_delta
            for l in range(3):X[l*2411+cols]+=dx
            return X
        edges=surface_edge_pairs(F[np.asarray(z['region_face_rows'][np.asarray([0,0])],np.int64)]) if False else None
        face_offsets=np.asarray(z['region_face_offsets'],np.int64); face_rows=np.asarray(z['region_face_rows'],np.int64)[face_offsets[ri]:face_offsets[ri+1]]
        eall=surface_edge_pairs(F[face_rows]); elook=np.full(2411,-1,np.int64);elook[cols]=np.arange(len(cols))
        edge_local=elook[eall]; edge_local=edge_local[np.all(edge_local>=0,axis=1)]
        # Only regional cells enter the nonlinear material constraints.
        ref=prepare_reference_tetrahedra(Ref[T[trows]])
        mloc=mat[trows]; ccols=T[trows]%2411
        ncol=len(cols)
        def objective(x):
            d=np.asarray(x,np.float64).reshape(ncol,3)-start_delta
            v=float(np.sum(d*d))
            if len(edge_local):
                de=d[edge_local[:,0]]-d[edge_local[:,1]];v+=.08*float(np.sum(de*de))
            return v
        def objective_jac(x):
            d=np.asarray(x,np.float64).reshape(ncol,3)-start_delta;G=2*d.copy()
            if len(edge_local):
                de=d[edge_local[:,0]]-d[edge_local[:,1]];g=.16*de
                np.add.at(G,edge_local[:,0],g);np.add.at(G,edge_local[:,1],-g)
            return G.ravel()
        def constraints(x, want_jac=False):
            Xmm=assemble(x); X=Xmm/1000.
            q,Gq=quality_and_column_jacobian(Xmm,T,trows,cols)
            J,GJ=J_and_column_jacobian(X,Ref,T,trows,cols); GJ*=.001
            nh=evaluate_neo_hookean_batch(X[T[trows]],ref,mloc,ccols,max_abs_piola_pa=5.0e6)
            valid=np.asarray(nh['valid'],bool)
            W=np.asarray(nh['energy_density_pa'],np.float64)
            P=np.asarray(nh['piola_components_pa'],np.float64)
            # q and material J guards
            skin=mloc==1;fat=mloc==2
            vq=(q-qguard)*1e5
            vjl=(J[skin]-jlo)*100.; vju=(jhi-J[skin])*100.
            vfl=(J[fat]-jlo)*100.; vfu=(jhi-J[fat])*100.
            # Cell energy density and max-absolute Piola; one active stress
            # component per cell, with all nine rechecked after F32 emission.
            vw=(wlim-W)/1e6
            pidx=np.argmax(np.abs(P),axis=1); prow=np.arange(len(P)); pactive=P[prow,pidx]
            vp=(plim-np.abs(pactive))/1e6
            vw=np.where(valid,vw,-10.0);vp=np.where(valid,vp,-10.0)
            rt_slack=plane_constraint_slack_and_jacobian(Xmm,cols,ph*1000.,pn,en[:,None],pm*1000.)['slack']
            vals=np.r_[vq,vjl,vju,vfl,vfu,vw,vp,rt_slack]
            if not want_jac:return vals
            qj=Gq*1e5
            jjl=GJ[skin]*100.;jju=-GJ[skin]*100.;jfl=GJ[fat]*100.;jfu=-GJ[fat]*100.
            ids=np.asarray(nh['jacobian_column_ids'],np.int64)
            lookup_local=np.full(2411,-1,np.int64);lookup_local[cols]=np.arange(ncol)
            wloc=np.asarray(nh['energy_jacobian_j_per_m'],np.float64)/ref.volume_m3[:,None,None]
            wj=-jac_scatter(wloc*1e-3,ids,lookup_local,ncol)/1e6
            pjac=np.asarray(nh['piola_component_jacobian_pa_per_m'],np.float64)
            active_j=pjac[prow,pidx,:,:]*np.sign(pactive)[:,None,None]
            pj=-jac_scatter(active_j*1e-3,ids,lookup_local,ncol)/1e6
            rj=plane_constraint_slack_and_jacobian(Xmm,cols,ph*1000.,pn,en[:,None],pm*1000.)['jacobian']
            return np.vstack((qj,jjl,jju,jfl,jfu,wj,pj,rj))
        # Cheap real-component Jacobian and start feasibility preflight.
        x0=start_delta.reshape(-1)
        v0=constraints(x0,False);g0=constraints(x0,True)
        if g0.shape!=(len(v0),len(x0)) or not np.all(np.isfinite(g0)) or not np.all(np.isfinite(v0)):raise ValueError(f'region {ri} finite/Jacobian preflight failed')
        # Finite-difference a nonzero analytic direction from the actual local
        # objective/constraints to catch unit/order/scatter errors before solve.
        nz=np.flatnonzero(np.linalg.norm(g0,axis=0)>1e-10)
        if len(nz):
            k=int(nz[len(nz)//2]);h=1e-6;xp=x0.copy();xm=x0.copy();xp[k]+=h;xm[k]-=h
            fd=(constraints(xp,False)-constraints(xm,False))/(2*h)
            err=float(np.max(np.abs(fd-g0[:,k])));scale=max(1.,float(np.max(np.abs(fd))))
            if err>4e-3*scale:raise ValueError(f'region {ri} constraint Jacobian FD failed: {err:g}/{scale:g}')
        else:err=0.
        last={'x':x0.copy(),'iterations':0}; region_started=time.monotonic()
        def callback(x):
            last['x']=np.asarray(x,np.float64).copy();last['iterations']+=1
            if last['iterations']%2==0:
                np.savez_compressed(out/f'checkpoint-region-{ri:02d}.npz',points_mm=assemble(last['x']),region=np.asarray(ri),columns=cols,elapsed_s=np.asarray(time.monotonic()-start))
            if time.monotonic()-start>=args.budget_s:raise TimeoutError('bounded attempt-005 total budget')
        try:
            res=minimize(objective,x0,jac=objective_jac,method='SLSQP',bounds=[(x0[i]-args.max_displacement_mm,x0[i]+args.max_displacement_mm) for i in range(len(x0))],constraints=[{'type':'ineq','fun':lambda x:constraints(x,False),'jac':lambda x:constraints(x,True)}],callback=callback,options={'maxiter':args.maxiter,'ftol':1e-8,'disp':False})
            accepted=np.asarray(res.x,np.float64); success=bool(res.success); message=str(res.message); nit=int(res.nit)
        except TimeoutError:
            accepted=last['x'];success=False;message='bounded total budget';nit=last['iterations'];budget_stop=True
        Xacc=assemble(accepted); v=constraints(accepted,False); qloc,detloc=quality(Xacc,T[trows]); jloc=Jvalues(Xacc,Ref,T[trows]) if False else None
        # J helper on the regional tet rows to keep ordering and scale explicit.
        from column_fit_support import reference_J_and_gradient
        jloc,_=reference_J_and_gradient(Xacc[T[trows]],Ref[T[trows]])
        nhacc=evaluate_neo_hookean_batch((Xacc/1000.)[T[trows]],ref,mloc,ccols,max_abs_piola_pa=5.0e6)
        perclass={}
        for name,ar in [('q_slack',(qloc-qguard)),('J_min_skin',jloc[mloc==1]-jlo),('J_max_skin',jhi-jloc[mloc==1]),('J_min_fat',jloc[mloc==2]-jlo),('J_max_fat',jhi-jloc[mloc==2]),('W_slack_MPa',(wlim-nhacc['energy_density_pa'])/1e6),('P_slack_MPa',(plim-np.max(np.abs(nhacc['piola_components_pa']),axis=1))/1e6)]:
            perclass[name]={'minimum':float(np.min(ar)) if len(ar) else None,'violations':int(np.sum(np.asarray(ar)<-1e-8))}
        route_slack=plane_constraint_slack_and_jacobian(Xacc,cols,ph*1000.,pn,en[:,None],pm*1000.)['slack']
        perclass['route_slack_mm']={'minimum':float(route_slack.min()) if len(route_slack) else None,'violations':int(np.sum(route_slack < -2e-4))}
        reports.append({'region':int(ri),'columns':int(len(cols)),'tets':int(len(trows)),'endpoint_rows':int(len(en)),'solver_success':success,'solver_message':message,'iterations':nit,'elapsed_s':time.monotonic()-region_started,'max_correction_mm':float(np.linalg.norm((accepted-x0).reshape(-1,3),axis=1).max()),'jacobian_fd_max_abs_error':err,'class_gates':perclass})
        # Accepted iterate is starting point for remaining disjoint regions.
        current_mm=Xacc
        np.savez_compressed(out/f'checkpoint-region-{ri:02d}.npz',points_mm=current_mm,region=np.asarray(ri),columns=cols,elapsed_s=np.asarray(time.monotonic()-start))
        (out/'progress.json').write_text(json.dumps({'status':'running','regions':reports,'elapsed_s':time.monotonic()-start},indent=2,sort_keys=True)+'\n')
        print('REGION',json.dumps(reports[-1],sort_keys=True),flush=True)
        if budget_stop:break
    X64=current_mm/1000.; Xf=X64.astype(np.float32).astype(np.float64)
    qf,detf=quality(Xf,T)
    # Reference J and descriptor energy/stress for all physical FEM rows.
    from column_fit_support import reference_J_and_gradient
    Jf,_=reference_J_and_gradient(Xf[T],Ref[T])
    nhfull=evaluate_neo_hookean_batch(Xf[T],ref_full,mat,T%2411,max_abs_piola_pa=5.0e6)
    W=np.asarray(nhfull['energy_density_pa']); P=np.asarray(nhfull['piola_components_pa']); valid=np.asarray(nhfull['valid'])
    maxP=np.max(np.abs(P),axis=1)
    # Route and retained baseline-O constraints; endpoint node rows are exact AP row IDs.
    en=np.asarray(z['constraint_endpoint_node_rows'],np.int64); ph=np.asarray(z['constraint_hit_points_world_m']);pn=np.asarray(z['constraint_normals_world']);pm=np.asarray(z['constraint_margins_m'])
    gaps=np.einsum('ij,ij->i',pn,Xf[en]-ph)-pm
    L=layer_values(Xf,Xbase)
    # Material volumes and fixed-density mass are diagnostics only.
    vols={str(k):float(np.sum(detf[mat==k])/6.) for k in (1,2)}
    baseq,basedet=quality(Xbase,T); basevols={str(k):float(np.sum(basedet[mat==k])/6.) for k in (1,2)}
    final={'float32_nonpositive':int(np.sum(detf<=0)),'q_below_1e-5':int(np.sum(qf<1e-5)),'q_below_1.25e-5':int(np.sum(qf<1.25e-5)),'J_below_.35':int(np.sum(Jf<.35)),'J_above_2.5':int(np.sum(Jf>2.5)),'W_above_5MPa':int(np.sum(W>5e6)),'Piola_above_5MPa':int(np.sum(maxP>5e6)),'invalid_NH_cells':int(np.sum(~valid)),'min_q':float(qf.min()),'min_J':float(Jf.min()),'max_J':float(Jf.max()),'max_W_Jm3':float(W.max()),'max_abs_P_Pa':float(maxP.max()),'min_route_or_O12_gap_m':float(gaps.min()),'route_or_O12_violations':int(np.sum(gaps < -2e-7)),'layer_skin_floor_violations':int(np.sum(L[0]<L[2]-2e-8)),'layer_fat_floor_violations':int(np.sum(L[1]<L[3]-2e-8)),'min_skin_mm':float(L[0].min()*1000),'min_fat_mm':float(L[1].min()*1000),'skin_volume_m3':vols['1'],'fat_volume_m3':vols['2'],'skin_fixed_density_mass_kg':vols['1']*1000.,'fat_fixed_density_mass_kg':vols['2']*900.,'baseline_skin_volume_m3':basevols['1'],'baseline_fat_volume_m3':basevols['2'],'skin_volume_relative_change':vols['1']/basevols['1']-1,'fat_volume_relative_change':vols['2']/basevols['2']-1,'moved_nodes':int(np.sum(np.any(Xf!=Xbase,axis=1))),'max_displacement_from_baseline_m':float(np.linalg.norm(Xf-Xbase,axis=1).max())}
    basic=all(final[k]==0 for k in ('float32_nonpositive','q_below_1e-5','J_below_.35','J_above_2.5','W_above_5MPa','Piola_above_5MPa','invalid_NH_cells','route_or_O12_violations','layer_skin_floor_violations','layer_fat_floor_violations'))
    status='candidate_basic_geometry_material_prefilter_pass_not_admitted' if basic and not budget_stop else ('candidate_rejected_or_incomplete' if (not basic or budget_stop) else 'candidate_unadmitted')
    np.savez_compressed(out/'candidate.npz',points_world_f32=Xf.astype(np.float32),tetrahedra=T.astype(np.int32),material_ids=mat.astype(np.int8),global_nodes=z['global_nodes'],region_columns=rc,region_column_offsets=ro,route_endpoint_node_rows=en,route_or_O12_gap_m=gaps,reference_J_f32=Jf,q_f32=qf,energy_density_pa_f32=W,max_abs_piola_pa_f32=maxP)
    report={'schema':'numi.human.shared-column-attempt-005.neo-hookean-prefilter.v1','status':status,'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (INP,AP,PP,REF,HELPER,BASE_SCRIPT)},'helper_reference_sha256':ref_full.sha256,'budget_s':args.budget_s,'elapsed_s':time.monotonic()-start,'regions':reports,'final_global_F32':final,'limitations':['Offline analytic Neo-Hookean energy/first-Piola are conservative descriptor prefilters; native mixed First-Piola semantics must still be checked by the owning solver.','Construction-plane and O12 checks are necessary conditions only; fresh exact all-bank solid containment, self/retained surface, detailed-source transfer, ancestry/mass ownership, and native J/stress/energy gates remain mandatory.','Fixed-density mass values are geometric diagnostics, not a conserved material ledger.','No candidate admission or physical qualification is claimed.']}
    report['candidate_sha256']=sha(out/'candidate.npz');(out/'fit-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('FINAL',json.dumps({'status':status,'candidate':str(out/'candidate.npz'),'candidate_sha256':report['candidate_sha256'],'elapsed_s':report['elapsed_s'],'global':final},sort_keys=True),flush=True)
if __name__=='__main__':main()
