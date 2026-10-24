#!/usr/bin/env python3
"""One bounded sparse trust-constr Neo-Hookean retry for regions 0 and 1."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
ATT=ROOT/'attempt-005'
INP=ATT/'trust-constr-input-001.npz'
META=ATT/'trust-constr-input-001.json'
RUNNER=ROOT/'run_attempt_005.py'
PARTIAL=ATT/'fit-001/partial-regions-002-005.npz'
HELPER=ROOT/'neo_hookean_fit_support.py'
COLUMN_HELPER=ROOT/'column_fit_support.py'
AP=ROOT.parent/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz'
PP=ROOT.parent/'native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz'
REF=ROOT.parent/'root-reference-relative-j-062/source-preflight/identity-reference-points.csv'
EXPECTED_INPUT='462f125a7945515b7879d8ab7a7bf1a7183de2e098d2ab323ef70ceb46753e00'
EXPECTED_RUNNER='bb2235ac0356158560645951ce638a393a978b48881789ef8c06946fbb9fba0b'
EXPECTED_PARTIAL='e8590f311129b2156d8dd131744c8b94ebf7356d7cdbd6010bf94faa9e372102'
EXPECTED_HELPER='815c256524045d3dbe7cbb23ef8c0fe7d9a02beb327fb929c67bb4bccb9096f5'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_ref(path,n):
    import csv
    pts=np.empty((n,3),np.float64);seen=np.zeros(n,bool)
    with open(path,newline='') as f:
        for r in csv.DictReader(f):
            i=int(r['local_volume_node'])
            if i<0 or i>=n or seen[i]:raise ValueError('reference row IDs invalid')
            pts[i]=[float(r['x_m']),float(r['y_m']),float(r['z_m'])];seen[i]=True
    if not np.all(seen) or not np.all(np.isfinite(pts)):raise ValueError('reference incomplete/nonfinite')
    return pts

def quality(X,T):
    V=X[T];u=V[:,1]-V[:,0];v=V[:,2]-V[:,0];w=V[:,3]-V[:,0]
    det=np.einsum('ij,ij->i',u,np.cross(v,w))
    e=np.stack([np.sum((V[:,i]-V[:,j])**2,axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))],axis=1)
    return det/np.maximum(e.max(axis=1),1e-30)**1.5,det

def gradient_csr(vertex_grad,column_rows,lookup,ncol,scale=1.0):
    G=np.asarray(vertex_grad,np.float64);C=np.asarray(column_rows,np.int64)
    if G.shape!=(len(C),4,3) or C.shape!=(len(G),4):raise ValueError('vertex gradient/map shape mismatch')
    from scipy.sparse import coo_matrix,csr_matrix
    nr=len(G);rr=[];cc=[];dd=[]
    for k in range(4):
        li=lookup[C[:,k]%2411];ok=li>=0
        if not np.any(ok):continue
        rowids=np.flatnonzero(ok);base=3*li[ok]
        for a in range(3):
            val=G[ok,k,a]*scale;nz=val!=0
            if np.any(nz):rr.append(rowids[nz]);cc.append(base[nz]+a);dd.append(val[nz])
    if not dd:return csr_matrix((nr,3*ncol),dtype=np.float64)
    return coo_matrix((np.concatenate(dd),(np.concatenate(rr),np.concatenate(cc))),shape=(nr,3*ncol)).tocsr()

def compact_gradient_csr(local_grad,column_ids,lookup,ncol,scale=1.0):
    G=np.asarray(local_grad,np.float64);C=np.asarray(column_ids,np.int64)
    from scipy.sparse import coo_matrix,csr_matrix
    if G.shape[:2]!=C.shape or G.ndim!=3 or G.shape[2]!=3:raise ValueError('compact gradient/map mismatch')
    nr,ns,_=G.shape;rr=[];cc=[];dd=[]
    for k in range(ns):
        ids=C[:,k];ok=ids>=0
        if not np.any(ok):continue
        rows=np.flatnonzero(ok);li=lookup[ids[ok]];good=li>=0
        rows=rows[good];base=3*li[good]
        for ax in range(3):
            val=G[rows,k,ax]*scale;nz=val!=0
            if np.any(nz):rr.append(rows[nz]);cc.append(base[nz]+ax);dd.append(val[nz])
    if not dd:return csr_matrix((nr,3*ncol),dtype=np.float64)
    return coo_matrix((np.concatenate(dd),(np.concatenate(rr),np.concatenate(cc))),shape=(nr,3*ncol)).tocsr()

def compact_gradient_sum(local_grad,column_ids,lookup,ncol,scale=1.0):
    M=compact_gradient_csr(local_grad,column_ids,lookup,ncol,scale)
    return np.asarray(M.sum(axis=0)).reshape(-1)

def atomic_npz(path,**arrays):
    tmp=path.with_name(path.name+'.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)

def main():
    from scipy.optimize import minimize,NonlinearConstraint,Bounds,BFGS
    from scipy.sparse import csr_matrix,vstack
    from column_fit_support import normalized_tet_quality_and_gradient,reference_J_and_gradient,plane_constraint_slack_and_jacobian
    from neo_hookean_fit_support import prepare_reference_tetrahedra,evaluate_neo_hookean_batch
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--budget-s',type=float,default=48.0)
    p.add_argument('--maxiter',type=int,default=40)
    p.add_argument('--max-displacement-mm',type=float,default=15.0)
    p.add_argument('--run-name',default='trust-constr-001')
    args=p.parse_args()
    out=HERE/args.run_name
    if out.exists() and any(out.iterdir()):raise FileExistsError('refusing to overwrite nonempty run directory')
    out.mkdir(parents=True,exist_ok=True)
    pins={'trust_input':(INP,EXPECTED_INPUT),'runner_005':(RUNNER,EXPECTED_RUNNER),'partial_candidate':(PARTIAL,EXPECTED_PARTIAL),'neo_hookean_helper':(HELPER,EXPECTED_HELPER)}
    pin_report={k:{'path':str(path),'expected_sha256':want,'actual_sha256':sha(path),'pass':sha(path)==want} for k,(path,want) in pins.items()}
    if not all(v['pass'] for v in pin_report.values()):raise ValueError('pinned input/code mismatch: '+json.dumps(pin_report,sort_keys=True))
    meta=json.loads(META.read_text())
    if meta.get('sha256')!=EXPECTED_INPUT or meta.get('schema')!='attempt-005-trust-constr-retry-input.v1':raise ValueError('trust input JSON mismatch')
    z=np.load(INP,allow_pickle=False);part=np.load(PARTIAL,allow_pickle=False)
    Xbase=np.asarray(z['points_original_f32'],np.float32).astype(np.float64)
    Xstart=np.asarray(z['points_start_f32'],np.float32).astype(np.float64)
    T=np.asarray(z['tetrahedra'],np.int64);mat=np.asarray(z['material_ids'],np.int64)
    F=np.asarray(z['O_faces'],np.int64);Ref=np.asarray(z['reference_points_m'],np.float64);gn=np.asarray(z['global_nodes'],np.int64)
    if not np.array_equal(Xstart.astype(np.float32),part['points_world_f32']):raise ValueError('start points differ from pinned pass-only candidate')
    if not np.array_equal(T,part['tetrahedra']) or not np.array_equal(mat,part['material_ids']) or not np.array_equal(gn,part['global_nodes']):raise ValueError('topology/material/global nodes differ from pinned pass-only candidate')
    if Xbase.shape!=(7233,3) or Xstart.shape!=Xbase.shape or T.shape!=(28632,4) or Ref.shape!=Xbase.shape or F.ndim!=2 or F.shape[1]!=3:raise ValueError('unexpected pinned FEM shape')
    ap=np.load(AP,allow_pickle=False);pp=np.load(PP,allow_pickle=False)
    if not np.array_equal(T,pp['tetrahedra']) or not np.array_equal(mat,pp['material_ids']):raise ValueError('physical topology/material mismatch')
    if not np.array_equal(Xbase.astype(np.float32),ap['points']):raise ValueError('original coordinates mismatch physical baseline')
    if not np.array_equal(Ref,read_ref(REF,len(Xbase))):raise ValueError('reference points differ from pinned reference CSV')
    ref_full=prepare_reference_tetrahedra(Ref[T]);qstart,dstart=quality(Xstart,T)
    if np.any(dstart<=0) or np.any(~np.isfinite(qstart)):raise ValueError('start includes nonpositive/nonfinite cells')
    dcols=np.empty((2411,3),np.float64)
    for c in range(2411):
        ds=np.stack([Xstart[l*2411+c]-Xbase[l*2411+c] for l in range(3)])
        if np.max(np.linalg.norm(ds-ds[0],axis=1))>2e-7:raise ValueError(f'column {c} displacement is not shared across O/I/D')
        dcols[c]=ds[0]*1000.0
    ro=np.asarray(z['region_column_offsets'],np.int64);rc=np.asarray(z['region_columns'],np.int64)
    rto=np.asarray(z['region_tet_offsets'],np.int64);rt=np.asarray(z['region_tet_rows'],np.int64)
    reo=np.asarray(z['region_endpoint_offsets'],np.int64);re=np.asarray(z['region_endpoint_rows'],np.int64)
    region_ids=np.asarray(z['region_ids'],np.int64)
    if not np.array_equal(region_ids,[0,1]) or any(len(x)!=3 for x in (ro,rto,reo)):raise ValueError('expected exactly regions 0 and 1')
    epcol=np.asarray(z['endpoint_column_rows'],np.int64);epnode=np.asarray(z['endpoint_node_rows'],np.int64)
    hit=np.asarray(z['hit_points_world_m'],np.float64);norm=np.asarray(z['normals_world'],np.float64);margin=np.asarray(z['margins_m'],np.float64)
    start=time.monotonic();deadline=start+args.budget_s;current_mm=Xstart*1000.;reports=[];budget_stop=False
    qguard=1.25e-5;jlo=.355;jhi=2.495;wlim=4.99e6;plim=4.99e6
    for ix,ri0 in enumerate(region_ids):
        ri=int(ri0)
        if time.monotonic()>=deadline:budget_stop=True;break
        cols=rc[ro[ix]:ro[ix+1]].astype(np.int64);trows=rt[rto[ix]:rto[ix+1]].astype(np.int64);erows=re[reo[ix]:reo[ix+1]].astype(np.int64)
        if not len(cols) or not len(trows) or len(np.unique(cols))!=len(cols):raise ValueError(f'invalid region {ri} dimensions')
        lookup=np.full(2411,-1,np.int64);lookup[cols]=np.arange(len(cols));ncol=len(cols);nvar=3*ncol
        cm=T[trows]%2411
        present=np.unique(cm)
        region_vars=np.unique(rc)
        if np.any(np.isin(present,region_vars) & ~np.isin(present,cols)):
            raise ValueError(f'region {ri} tet support couples another movable region')
        if np.any(~np.isin(epcol[erows],cols)):raise ValueError(f'region {ri} route endpoints cross the movable-column set')
        en=epnode[erows];ph=hit[erows];pn=norm[erows];pm=margin[erows]
        x0=dcols[cols].reshape(-1).copy();anchor=current_mm.copy()
        def assemble(x):
            X=anchor.copy();dx=np.asarray(x,np.float64).reshape(ncol,3)-dcols[cols]
            for layer in range(3):X[layer*2411+cols]+=dx
            return X
        ref_local=prepare_reference_tetrahedra(Ref[T[trows]]);mloc=mat[trows]
        cache={}
        def calc(x):
            xv=np.asarray(x,np.float64)
            if 'x' in cache and np.array_equal(cache['x'],xv):return cache
            Xmm=assemble(xv);Xm=Xmm/1000.;V=Xm[T[trows]]
            q,Gq=normalized_tet_quality_and_gradient(Xmm[T[trows]])
            J,GJ=reference_J_and_gradient(V,Ref[T[trows]])
            nh=evaluate_neo_hookean_batch(V,ref_local,mloc,cm,max_abs_piola_pa=5e6)
            route=plane_constraint_slack_and_jacobian(Xmm,cols,ph*1000.,pn,en[:,None],pm*1000.)
            cache.clear();cache.update(x=xv.copy(),Xmm=Xmm,Xm=Xm,V=V,q=q,Gq=Gq,J=J,GJ=GJ,nh=nh,route=route)
            return cache
        skin=mloc==1;fat=mloc==2
        init=calc(x0);E0=float(np.sum(init['nh']['energy_joule']));Escale=max(E0,1e-9);anchor_weight=1e-5
        def objective(x):
            st=calc(x);d=np.asarray(x,np.float64)-x0
            return float(np.sum(st['nh']['energy_joule'])/Escale+anchor_weight*np.dot(d,d))
        def objective_jac(x):
            st=calc(x);nh=st['nh']
            local=np.asarray(nh['energy_jacobian_j_per_m'],np.float64)*(.001/Escale)
            return compact_gradient_sum(local,nh['jacobian_column_ids'],lookup,ncol)+2*anchor_weight*(np.asarray(x,np.float64)-x0)
        def values(x):
            st=calc(x);nh=st['nh'];J=st['J'];W=np.asarray(nh['energy_density_pa']);P=np.asarray(nh['piola_components_pa']);valid=np.asarray(nh['valid'],bool)
            vq=(st['q']-qguard)*1e5
            vjl=(J[skin]-jlo)*100.;vju=(jhi-J[skin])*100.
            vfl=(J[fat]-jlo)*100.;vfu=(jhi-J[fat])*100.
            vw=(wlim-W)/1e6
            pidx=np.argmax(np.abs(P),axis=1);pa=P[np.arange(len(P)),pidx]
            vp=(plim-np.abs(pa))/1e6
            return np.r_[vq,vjl,vju,vfl,vfu,np.where(valid,vw,-10.),np.where(valid,vp,-10.),st['route']['slack']]
        def jac(x):
            st=calc(x);nh=st['nh'];valid=np.asarray(nh['valid'],bool)
            qj=gradient_csr(st['Gq'],T[trows],lookup,ncol,1e5)
            jjl=gradient_csr(st['GJ'],T[trows],lookup,ncol,.1)[skin]
            jju=gradient_csr(st['GJ'],T[trows],lookup,ncol,-.1)[skin]
            jfl=gradient_csr(st['GJ'],T[trows],lookup,ncol,.1)[fat]
            jfu=gradient_csr(st['GJ'],T[trows],lookup,ncol,-.1)[fat]
            ids=np.asarray(nh['jacobian_column_ids'],np.int64)
            wloc=np.asarray(nh['energy_jacobian_j_per_m'],np.float64)/ref_local.volume_m3[:,None,None]
            wj=compact_gradient_csr(-wloc*.001e-6,ids,lookup,ncol)
            P=np.asarray(nh['piola_components_pa']);pjac=np.asarray(nh['piola_component_jacobian_pa_per_m'])
            idx=np.argmax(np.abs(P),axis=1);pr=np.arange(len(P));pa=P[pr,idx]
            active=pjac[pr,idx]*np.sign(pa)[:,None,None]
            pj=compact_gradient_csr(-active*.001e-6,ids,lookup,ncol)
            if np.any(~valid):
                bad=np.flatnonzero(~valid);wj=wj.tolil();pj=pj.tolil();wj[bad,:]=0.;pj[bad,:]=0.;wj=wj.tocsr();pj=pj.tocsr()
            return vstack((qj,jjl,jju,jfl,jfu,wj,pj,csr_matrix(st['route']['jacobian'])),format='csr')
        c0=values(x0);j0=jac(x0);g0=objective_jac(x0)
        if j0.shape!=(len(c0),nvar) or not np.all(np.isfinite(c0)) or not np.all(np.isfinite(j0.data)) or not np.all(np.isfinite(g0)):raise ValueError(f'region {ri} sparse derivative preflight failed')
        nz=np.flatnonzero(j0.getnnz(axis=0)>0)
        h=1e-6;k=int(nz[len(nz)//2]) if len(nz) else 0
        xp=x0.copy();xm=x0.copy();xp[k]+=h;xm[k]-=h
        fd=(values(xp)-values(xm))/(2*h);an=j0[:,k].toarray().ravel()
        constraint_fd=float(np.max(np.abs(fd-an))) if len(fd) else 0.
        ofd=(objective(xp)-objective(xm))/(2*h);objective_fd=abs(ofd-g0[k])
        if constraint_fd>5e-3*max(1.,float(np.max(np.abs(fd)))) or objective_fd>5e-3*max(1.,abs(ofd)):
            raise ValueError(f'region {ri} derivative FD failed: constraint={constraint_fd:g}, objective={objective_fd:g}')
        # checkpoint every accepted iterate and retain the best feasibility merit.
        runstate={'iterations':0,'last_x':x0.copy(),'best_violation':float('inf'),'best_feasible_objective':float('inf')}
        def callback(xk,state=None):
            xv=np.asarray(xk,np.float64).copy();runstate['last_x']=xv;runstate['iterations']+=1
            X=assemble(xv);violation=float(getattr(state,'constr_violation',np.inf)) if state is not None else np.inf
            atomic_npz(out/f'checkpoint-region-{ri:02d}.npz',points_world_f32=(X/1000).astype(np.float32),x_mm=xv,region=np.asarray(ri),elapsed_s=np.asarray(time.monotonic()-start),constr_violation=np.asarray(violation))
            if violation<runstate['best_violation']:
                runstate['best_violation']=violation
                atomic_npz(out/f'best-merit-region-{ri:02d}.npz',points_world_f32=(X/1000).astype(np.float32),x_mm=xv,region=np.asarray(ri),constr_violation=np.asarray(violation))
            if violation<=1e-8:
                obj=objective(xv)
                if obj<runstate['best_feasible_objective']:
                    runstate['best_feasible_objective']=obj
                    atomic_npz(out/f'best-feasible-region-{ri:02d}.npz',points_world_f32=(X/1000).astype(np.float32),x_mm=xv,objective=np.asarray(obj))
            return time.monotonic()>=deadline
        nl=NonlinearConstraint(values,0.,np.inf,jac=jac,hess=BFGS())
        region_start=time.monotonic();solve_error=None
        try:
            res=minimize(objective,x0,method='trust-constr',jac=objective_jac,hess=BFGS(),
                         bounds=Bounds(x0-args.max_displacement_mm,x0+args.max_displacement_mm),
                         constraints=[nl],callback=callback,
                         options={'maxiter':args.maxiter,'gtol':1e-6,'xtol':1e-8,'barrier_tol':1e-8,
                                  'sparse_jacobian':True,'factorization_method':'AugmentedSystem','verbose':0,'initial_tr_radius':1.0,'initial_constr_penalty':1.0})
            terminal_x=np.asarray(res.x,np.float64);success=bool(res.success);message=str(res.message);nit=int(res.niter);status=int(res.status)
        except Exception as e:
            res=None;terminal_x=runstate['last_x'].copy();success=False;message=f'{type(e).__name__}: {e}';nit=runstate['iterations'];status=-1;solve_error=message
        accepted=terminal_x.copy();accepted_checkpoint='solver_terminal'
        if np.isfinite(runstate['best_feasible_objective']):
            bf=np.load(out/f'best-feasible-region-{ri:02d}.npz',allow_pickle=False)
            accepted=np.asarray(bf['x_mm'],np.float64)
            accepted_checkpoint='best_feasible_callback_iterate'
        Xacc=assemble(accepted)
        atomic_npz(out/f'candidate-region-{ri:02d}-raw.npz',points_world_f32=(Xacc/1000).astype(np.float32),x_mm=accepted,terminal_x_mm=terminal_x,region=np.asarray(ri),accepted_checkpoint=np.asarray(accepted_checkpoint))
        # J is recomputed using metre coordinates here, never Xacc in millimetres.
        qloc,detloc=quality(Xacc/1000.,T[trows])
        Jloc,_=reference_J_and_gradient((Xacc/1000.)[T[trows]],Ref[T[trows]])
        nhacc=evaluate_neo_hookean_batch((Xacc/1000.)[T[trows]],ref_local,mloc,cm,max_abs_piola_pa=5e6)
        P=np.asarray(nhacc['piola_components_pa']);W=np.asarray(nhacc['energy_density_pa']);mx=np.max(np.abs(P),axis=1)
        route=plane_constraint_slack_and_jacobian(Xacc,cols,ph*1000.,pn,en[:,None],pm*1000.)['slack']
        perclass={}
        for name,a in [('q_slack',qloc-1.25e-5),('J_min_skin',Jloc[skin]-.355),('J_max_skin',2.495-Jloc[skin]),('J_min_fat',Jloc[fat]-.355),('J_max_fat',2.495-Jloc[fat]),('W_slack_MPa',(4.99e6-W)/1e6),('P_slack_MPa',(4.99e6-mx)/1e6)]:
            perclass[name]={'minimum':float(a.min()) if len(a) else None,'violations':int(np.sum(a < -1e-8))}
        perclass['route_slack_mm']={'minimum':float(route.min()) if len(route) else None,'violations':int(np.sum(route < -2e-4))}
        rep={'region':ri,'columns':int(len(cols)),'tets':int(len(trows)),'route_endpoint_rows':int(len(en)),'solver_success':success,'solver_status':status,'solver_message':message,'iterations':nit,'solver_error':solve_error,'accepted_checkpoint':accepted_checkpoint,'terminal_solver_objective':float(objective(terminal_x)),'region_elapsed_s':time.monotonic()-region_start,'total_elapsed_s':time.monotonic()-start,'objective_start_energy_j':E0,'energy_scale_j':Escale,'anchor_weight':anchor_weight,'objective_final':float(objective(accepted)),'constraint_min_scaled':float(values(accepted).min()),'best_constraint_violation':runstate['best_violation'],'best_feasible_objective':runstate['best_feasible_objective'] if np.isfinite(runstate['best_feasible_objective']) else None,'max_translation_change_mm':float(np.linalg.norm((accepted-x0).reshape(-1,3),axis=1).max()),'constraint_jacobian_fd_abs_error':constraint_fd,'objective_jacobian_fd_abs_error':objective_fd,'jacobian_shape':list(j0.shape),'jacobian_nnz':int(j0.nnz),'regional_J_recomputed_from_metre_coordinates':True,'class_gates':perclass}
        reports.append(rep);current_mm=Xacc
        atomic_npz(out/f'checkpoint-region-{ri:02d}.npz',points_world_f32=(current_mm/1000.).astype(np.float32),x_mm=accepted,region=np.asarray(ri),elapsed_s=np.asarray(time.monotonic()-start),constr_violation=np.asarray(float(getattr(locals().get('res',None),'constr_violation',np.inf))))
        (out/'progress.json').write_text(json.dumps({'status':'running','regions':reports,'elapsed_s':time.monotonic()-start},indent=2,sort_keys=True)+'\n')
        print('REGION',json.dumps(rep,sort_keys=True),flush=True)
        if time.monotonic()>=deadline:budget_stop=True;break
    X64=current_mm/1000.;Xf=X64.astype(np.float32).astype(np.float64)
    # Freeze emitted candidate before global diagnostics.
    atomic_npz(out/'candidate.npz',points_world_f32=Xf.astype(np.float32),tetrahedra=T.astype(np.int32),material_ids=mat.astype(np.int8),global_nodes=gn,region_ids=region_ids,region_columns=rc,region_column_offsets=ro)
    qf,detf=quality(Xf,T);Jf,_=reference_J_and_gradient(Xf[T],Ref[T])
    nh=evaluate_neo_hookean_batch(Xf[T],ref_full,mat,T%2411,max_abs_piola_pa=5e6)
    W=np.asarray(nh['energy_density_pa']);P=np.asarray(nh['piola_components_pa']);valid=np.asarray(nh['valid']);maxP=np.max(np.abs(P),axis=1)
    en=np.asarray(z['endpoint_node_rows'],np.int64);ph=np.asarray(z['hit_points_world_m'],np.float64);pn=np.asarray(z['normals_world'],np.float64);pm=np.asarray(z['margins_m'],np.float64)
    gaps=np.einsum('ij,ij->i',pn,Xf[en]-ph)-pm
    O0,I0,D0=Xbase[:2411],Xbase[2411:4822],Xbase[4822:];so=O0-I0;fa=I0-D0
    sn=so/np.maximum(np.linalg.norm(so,axis=1),1e-30)[:,None];fn=fa/np.maximum(np.linalg.norm(fa,axis=1),1e-30)[:,None]
    skin=np.einsum('ij,ij->i',Xf[:2411]-Xf[2411:4822],sn);fat=np.einsum('ij,ij->i',Xf[2411:4822]-Xf[4822:],fn)
    sfloor=np.minimum(.2*np.linalg.norm(so,axis=1),.0001);ffloor=np.minimum(.2*np.linalg.norm(fa,axis=1),.0001)
    vols={str(k):float(np.sum(detf[mat==k])/6.) for k in (1,2)}
    _,bd=quality(Xbase,T);basevols={str(k):float(np.sum(bd[mat==k])/6.) for k in (1,2)}
    final={'float32_nonpositive':int(np.sum(detf<=0)),'q_below_1e-5':int(np.sum(qf<1e-5)),'q_below_1.25e-5':int(np.sum(qf<1.25e-5)),'J_below_.35':int(np.sum(Jf<.35)),'J_above_2.5':int(np.sum(Jf>2.5)),'W_above_5MPa':int(np.sum(W>5e6)),'Piola_above_5MPa':int(np.sum(maxP>5e6)),'invalid_NH_cells':int(np.sum(~valid)),'min_q':float(qf.min()),'min_J':float(Jf.min()),'max_J':float(Jf.max()),'max_W_Jm3':float(W.max()),'max_abs_P_Pa':float(maxP.max()),'min_route_or_O12_gap_m':float(gaps.min()),'route_or_O12_violations':int(np.sum(gaps < -2e-7)),'skin_floor_violations':int(np.sum(skin<sfloor-2e-8)),'fat_floor_violations':int(np.sum(fat<ffloor-2e-8)),'min_skin_mm':float(skin.min()*1000),'min_fat_mm':float(fat.min()*1000),'skin_volume_m3':vols['1'],'fat_volume_m3':vols['2'],'skin_fixed_density_mass_kg':vols['1']*1000.,'fat_fixed_density_mass_kg':vols['2']*900.,'baseline_skin_volume_m3':basevols['1'],'baseline_fat_volume_m3':basevols['2'],'skin_volume_relative_change':vols['1']/basevols['1']-1,'fat_volume_relative_change':vols['2']/basevols['2']-1,'moved_nodes':int(np.sum(np.any(Xf!=Xbase,axis=1))),'max_displacement_from_baseline_m':float(np.linalg.norm(Xf-Xbase,axis=1).max())}
    passed=all(final[k]==0 for k in ('float32_nonpositive','q_below_1e-5','J_below_.35','J_above_2.5','W_above_5MPa','Piola_above_5MPa','invalid_NH_cells','route_or_O12_violations','skin_floor_violations','fat_floor_violations'))
    status='candidate_basic_geometry_material_prefilter_pass_not_admitted' if passed and not budget_stop else 'candidate_rejected_or_incomplete'
    report={'schema':'numi.human.shared-column-attempt-006.sparse-trust-constr.v1','status':status,'input_pins':pin_report,'trust_input_metadata_sha256':sha(META),'partial_candidate_sha256':sha(PARTIAL),'runner_005_sha256':sha(RUNNER),'neo_hookean_helper_sha256':sha(HELPER),'column_fit_helper_sha256':sha(COLUMN_HELPER),'reference_csv_sha256':sha(REF),'source_retry_input_sha256':meta['source_retry_input']['sha256'],'objective':'sum regional Neo-Hookean element joules / start-energy scale + 1e-5 squared translation anchor from pass-only partial; energy Jacobian is analytic','constraints':'same q/J/W/max-active-|P|/route constraints and scales as attempt-005','budget_s':args.budget_s,'maxiter_per_region':args.maxiter,'max_displacement_mm':args.max_displacement_mm,'elapsed_s':time.monotonic()-start,'budget_stop':budget_stop,'regions':reports,'global_emitted_F32':final,'candidate_sha256':sha(out/'candidate.npz'),'limitations':['Offline candidate fit only; no physical or native admission.','Regions 0 and 1 only were retried; other regions remain the pinned pass-only partial candidate.','All-bank containment, exact surface, topology, transfer/mass ownership, and native solver gates remain outstanding.']}
    (out/'fit-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('FINAL',json.dumps({'status':status,'candidate':str(out/'candidate.npz'),'candidate_sha256':report['candidate_sha256'],'elapsed_s':report['elapsed_s'],'global':final},sort_keys=True),flush=True)
if __name__=='__main__':main()
