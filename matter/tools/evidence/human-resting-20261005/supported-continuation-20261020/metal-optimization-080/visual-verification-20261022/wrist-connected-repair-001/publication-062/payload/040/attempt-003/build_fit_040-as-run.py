#!/usr/bin/env python3
"""Bounded shared O/I/D column translation fit using exact 041 route planes.

Candidate construction only. It preserves each column's double-precision layer
offsets, enforces material-specific reference-J and normalized tet-quality
constraints on every touched cell, and requires a later exact solid audit.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,time
import numpy as np
from scipy.optimize import minimize
from column_fit_support import (expand_seed_patch,expand_construction_plane_targets,
    plane_constraint_slack_and_jacobian,shared_column_components,surface_edge_pairs,
    quality_and_column_jacobian,J_and_column_jacobian)

ROOT=Path('/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
AP=ROOT/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz'
PP=ROOT/'native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz'
PREP=ROOT/'bone-clearance-fit-037-preparation-035'
SEED=PREP/'per-bone-constraints-037.npz'
OLDQ=PREP/'restart-state-037.npz'
REF=ROOT/'root-reference-relative-j-062/source-preflight/identity-reference-points.csv'
ROUTE=ROOT/'bone-clearance-fit-041-exact-material-routes/route-report.json'
ROUTE_NPZ=ROOT/'bone-clearance-fit-041-exact-material-routes/construction-plane-constraints.npz'
O12_NPZ=ROOT/'bone-clearance-fit-041-exact-material-routes/baseline-outer-inside-local-planes.npz'
HERE=ROOT/'bone-clearance-fit-040-shared-columns'

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_ref(path,n):
    pts=np.empty((n,3),np.float64);seen=np.zeros(n,bool)
    with open(path,newline='') as f:
        for r in csv.DictReader(f):
            i=int(r['local_volume_node'])
            if i<0 or i>=n or seen[i]:raise ValueError('reference row IDs duplicated/out of range')
            pts[i]=[float(r['x_m']),float(r['y_m']),float(r['z_m'])];seen[i]=True
    if not np.all(seen) or not np.all(np.isfinite(pts)):raise ValueError('reference rows incomplete/nonfinite')
    return pts

def quality(X,T):
    V=X[T];u=V[:,1]-V[:,0];v=V[:,2]-V[:,0];w=V[:,3]-V[:,0]
    det=np.einsum('ij,ij->i',u,np.cross(v,w))
    e=np.stack([np.sum((V[:,i]-V[:,j])**2,axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))],axis=1)
    return det/np.maximum(e.max(axis=1),1e-30)**1.5,det

def Jvalues(X,Ref,T):
    V=X[T];R=Ref[T]
    U=R[:,1]-R[:,0];V1=R[:,2]-R[:,0];W=R[:,3]-R[:,0]
    den=np.einsum('ij,ij->i',U,np.cross(V1,W))
    if np.any(~np.isfinite(den)) or np.any(np.abs(den)<1e-24):raise ValueError('bad reference tet determinant')
    u=V[:,1]-V[:,0];v=V[:,2]-V[:,0];w=V[:,3]-V[:,0]
    return np.einsum('ij,ij->i',u,np.cross(v,w))/den

def layer_measure(X,ring,base):
    O0,I0,D0=base[:2411],base[2411:4822],base[4822:]
    so=O0[ring]-I0[ring];fa=I0[ring]-D0[ring]
    sn=so/np.linalg.norm(so,axis=1)[:,None];fn=fa/np.linalg.norm(fa,axis=1)[:,None]
    O,I,D=X[:2411],X[2411:4822],X[4822:]
    s=np.einsum('ij,ij->i',O[ring]-I[ring],sn)
    f=np.einsum('ij,ij->i',I[ring]-D[ring],fn)
    s0=np.linalg.norm(so,axis=1);f0=np.linalg.norm(fa,axis=1)
    sf=np.minimum(.2*s0,.0001);ff=np.minimum(.2*f0,.0001)
    return s,f,sf,ff

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--budget-s',type=float,default=105.0);pa.add_argument('--max-displacement-mm',type=float,default=15.0);pa.add_argument('--preflight-only',action='store_true');pa.add_argument('--output',default='attempt-001');args=pa.parse_args()
    out=HERE/args.output;out.mkdir(parents=True,exist_ok=True)
    ap=np.load(AP,allow_pickle=False);pp=np.load(PP,allow_pickle=False);seed=np.load(SEED,allow_pickle=False);oldq=np.load(OLDQ,allow_pickle=False)
    X0=np.asarray(ap['points'],np.float32).astype(np.float64);T=np.asarray(ap['tetrahedra'],np.int64);mat=np.asarray(pp['material_ids'],np.int64)
    if X0.shape!=(7233,3) or T.shape!=(28632,4) or mat.shape!=(len(T),):raise ValueError('unexpected baseline FEM dimensions')
    if not np.array_equal(T,pp['tetrahedra']):raise ValueError('baseline topology mismatch against pinned 026 payload')
    if not np.array_equal(mat,pp['material_ids']) or set(np.unique(mat).tolist())!={1,2}:raise ValueError('unexpected physical material IDs')
    Ref=read_ref(REF,len(X0))
    route_data=json.loads(ROUTE.read_text())
    rz=np.load(ROUTE_NPZ,allow_pickle=False)
    required_route={"plane_row","plane_segment_code","plane_start_global_node","plane_end_global_node","plane_target_layer_mask","plane_point_world_m","plane_outside_normal_world","plane_margin_m","constraint_source_hit_row","constraint_endpoint_node_row","constraint_endpoint_global_node","constraint_endpoint_layer","constraint_hit_point_world_m","constraint_outside_normal_world","constraint_margin_m","constraint_policy"}
    if not required_route.issubset(rz.files):raise ValueError(f"frozen 041 route NPZ lacks {sorted(required_route-set(rz.files))}")
    cols0=np.asarray(rz['plane_row'],np.int64);plane_count=len(cols0)
    if not plane_count:raise ValueError('no certified construction planes')
    masks=np.asarray(rz['plane_target_layer_mask'],np.int64);hits=np.asarray(rz['plane_point_world_m'],np.float64)
    normals=np.asarray(rz['plane_outside_normal_world'],np.float64);margins=np.asarray(rz['plane_margin_m'],np.float64)
    if np.any(np.asarray(rz['plane_segment_code'],np.int64)==0) and not np.all(masks[np.asarray(rz['plane_segment_code'],np.int64)==0]==6):raise ValueError('O-I route mask must target I and D')
    if np.any(np.asarray(rz['plane_segment_code'],np.int64)==1) and not np.all(masks[np.asarray(rz['plane_segment_code'],np.int64)==1]==4):raise ValueError('I-D route mask must target D')
    # Independently verify the route adapter expanded precisely the policy endpoints.
    pe=np.asarray(rz['constraint_endpoint_node_row'],np.int64);peg=np.asarray(rz['constraint_endpoint_global_node'],np.int64)
    ps=np.asarray(rz['constraint_source_hit_row'],np.int64);pc=np.asarray(rz['constraint_plane_index'],np.int64)
    pi=np.asarray(rz['constraint_hit_point_world_m'],np.float64);pn=np.asarray(rz['constraint_outside_normal_world'],np.float64);pm=np.asarray(rz['constraint_margin_m'],np.float64)
    pol=np.asarray(rz['constraint_policy']).astype(str)
    if pe.ndim!=1 or peg.shape!=pe.shape or ps.shape!=pe.shape or pc.shape!=pe.shape or pi.shape!=(len(pe),3) or pn.shape!=(len(pe),3) or pm.shape!=pe.shape or pol.shape!=pe.shape:raise ValueError('expanded 041 route arrays are not row aligned')
    if np.any(pc<0) or np.any(pc>=plane_count) or np.any(ps!=cols0[pc]) or not np.array_equal(pe,cols0[pc]):raise ValueError('expanded route endpoint/source column identities mismatch')
    if not np.allclose(pi,hits[pc],rtol=0,atol=2e-15) or not np.allclose(pn,normals[pc],rtol=0,atol=2e-15) or not np.allclose(pm,margins[pc],rtol=0,atol=0):raise ValueError('expanded route plane values differ from frozen plane table')
    expected_endpoint_row=np.where(np.asarray(rz['constraint_endpoint_layer'],np.int64)==1,2411+cols0[pc],4822+cols0[pc])
    expected_endpoint_global=np.asarray(ap['global_nodes'],np.int64)[expected_endpoint_row]
    if not np.array_equal(peg,expected_endpoint_global):raise ValueError('expanded external global endpoint ID does not match source column/layer map')
    global_to_local={int(g):i for i,g in enumerate(np.asarray(ap['global_nodes'],np.int64))}
    try: route_endpoint_local=np.asarray([global_to_local[int(g)] for g in peg],np.int64)
    except KeyError as exc: raise ValueError(f'route global endpoint missing from AP node map: {exc}')
    if np.any((pol=='OI_entry_requires_I_and_D') & (np.asarray(rz['constraint_endpoint_layer'],np.int64)!=1) & (np.asarray(rz['constraint_endpoint_layer'],np.int64)!=2)):raise ValueError('O-I constraint has invalid endpoint layer')
    if np.any((pol=='ID_entry_requires_D') & (np.asarray(rz['constraint_endpoint_layer'],np.int64)!=2)):raise ValueError('I-D constraint has invalid endpoint layer')
    # Preserve the separately audited 12 O-layer nearest-exit constraints. These are
    # baseline local requirements, not first-entry evidence and not a full solid audit.
    o12=np.load(O12_NPZ,allow_pickle=False)
    o_node=np.asarray(o12['constraint_local_node'],np.int64)
    o_hit=np.asarray(o12['constraint_closest_world_m'],np.float64)
    o_norm=np.asarray(o12['constraint_outward_normal_world'],np.float64)
    o_margin=np.asarray(o12['constraint_margin_m'],np.float64)
    if o_node.shape!=(12,) or o_hit.shape!=(12,3) or o_norm.shape!=(12,3) or o_margin.shape!=(12,):raise ValueError('O12 baseline plane arrays have unexpected shapes')
    if np.any(o_node<0) or np.any(o_node>=2411) or len(np.unique(o_node))!=12:raise ValueError('O12 rows are not unique outer-layer node IDs')
    if not np.all(np.isfinite(o_hit)) or not np.all(np.isfinite(o_norm)) or not np.all(np.isfinite(o_margin)) or np.any(o_margin<0):raise ValueError('O12 baseline planes contain invalid values')
    o_norm_len=np.linalg.norm(o_norm,axis=1)
    if np.any(np.abs(o_norm_len-1)>2e-8):raise ValueError('O12 normals are not unit covectors')
    seed_o=np.flatnonzero(np.asarray(seed['constraint_local_node'],np.int64)<2411)
    if len(seed_o)!=12:raise ValueError(f'expected exactly 12 O rows in 037 seed, found {len(seed_o)}')
    for key,side in [('constraint_local_node','node'),('constraint_body_index','body'),('constraint_bone_stable_id','bone'),('constraint_nearest_triangle','triangle')]:
        if not np.array_equal(np.asarray(seed[key])[seed_o],np.asarray(o12[key])):raise ValueError(f'O12 identity mismatch in {key}')
    full_endpoints=np.r_[route_endpoint_local,o_node]
    full_endpoint_global=np.r_[peg,np.asarray(ap['global_nodes'],np.int64)[o_node]]
    full_hit=np.vstack((pi,o_hit))
    full_norm=np.vstack((pn,o_norm))
    full_margin=np.r_[pm,o_margin]
    full_source=np.r_[ps,-o_node-1]
    full_kind=np.r_[np.zeros(len(peg),np.int8),np.ones(len(o_node),np.int8)]
    full_policy=np.r_[pol,np.full(len(o_node),'baseline_O12_local_escape',dtype='<U32')]
    full_layer=np.r_[np.asarray(rz['constraint_endpoint_layer'],np.int8),np.zeros(len(o_node),np.int8)]
    old_seed=np.asarray(seed['constraint_local_node'],np.int64)%2411
    seed_columns=np.unique(np.r_[cols0,old_seed])
    F=np.asarray(pp['O_faces'],np.int64)
    patch=expand_seed_patch(seed_columns,F,max_rings=6)
    movable=patch['movable_columns'];boundary=patch['boundary_columns'];face_rows=patch['face_rows']
    if not len(movable):raise ValueError('measured patch has no interior shared-column DOFs')
    locked_seed=np.intersect1d(seed_columns,boundary)
    edges=surface_edge_pairs(F[face_rows])
    # Every 037 preparation cell plus every tetrahedron incident to a moved column.
    incident=np.flatnonzero(np.isin(T%2411,movable).any(axis=1))
    touched=np.unique(np.r_[incident,np.asarray(oldq['affected_tet_rows'],np.int64)])
    groups,gid=shared_column_components(T[touched],movable,coupling_edges=edges)
    # Route endpoints that are locked are tested now; a negative one is an explicit no-fit blocker.
    locked_bad=[];movable_set=set(map(int,movable.tolist()))
    for i,node in enumerate(full_endpoints):
        if int(node)%2411 not in movable_set:
            gap=float(np.dot(full_norm[i],X0[int(node)]-full_hit[i])-full_margin[i])
            if gap < -2e-7:locked_bad.append({'hit_row':int(full_source[i]),'node':int(node),'gap_m':gap})
    if locked_bad:
        report={'status':'blocked_fixed_patch_endpoint','locked_endpoint_violations':locked_bad,'route_planes':int(plane_count),'patch':{'seed_count':int(len(seed_columns)),'face_count':int(len(face_rows)),'column_count':int(len(patch['patch_columns'])),'boundary_count':int(len(boundary)),'movable_count':int(len(movable)),'ring_trace':patch['trace']}}
        (out/'preflight-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report));return
    qbase,_=quality(X0,T);Jbase=Jvalues(X0,Ref,T)
    q_touched=qbase[touched];J_touched=Jbase[touched]
    # All constraints are evaluated before any solve; no initializer is treated as a bound.
    if np.any(~np.isfinite(qbase)) or np.any(~np.isfinite(Jbase)):raise ValueError('nonfinite baseline geometry')
    inputs=[AP,PP,SEED,OLDQ,REF,ROUTE,ROUTE_NPZ,O12_NPZ,HERE/'column_fit_support.py']
    report={'schema':'numi.human.shared-column-bone-entry-fit.v1','status':'preflight_pass',
       'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
       'point_source':'AP current-world F32 rows (verified against 035 identity baseline); PP 026 supplies only matching topology/material and O-face connectivity',
       'route':{'report_sha256':sha(ROUTE),'status':route_data.get('status'),'construction_plane_count':int(plane_count),'plane_segment_counts':{str(k):int(np.sum(np.asarray(rz['plane_segment_code'],np.int64)==k)) for k in [0,1]},'plane_target_masks':{str(k):int(np.sum(masks==k)) for k in [4,6]}},
       'patch':{'seed_count':int(len(seed_columns)),'route_seed_count':int(len(np.unique(cols0))),'old_inside_seed_count':int(len(np.unique(old_seed))),'face_count':int(len(face_rows)),'column_count':int(len(patch['patch_columns'])),'boundary_count':int(len(boundary)),'movable_count':int(len(movable)),'locked_seed_count':int(len(locked_seed)),'ring_trace':patch['trace']},
       'constraints':{'touched_tets':int(len(touched)),'old_7060_tet_rows':int(len(oldq['affected_tet_rows'])),'components':{'column_counts':[int(len(g)) for g in groups]},'expanded_endpoint_rows':int(len(full_endpoints)),'route_endpoint_rows':int(len(peg)),'baseline_O12_rows':int(len(o_node)),'fixed_violating_endpoints':0,
           'baseline_touched_q_min':float(q_touched.min()),'baseline_touched_J_min':float(J_touched.min()),'baseline_touched_J_max':float(J_touched.max()),'J_material_counts':{str(k):int(np.sum(mat[touched]==k)) for k in [1,2]},'q_target':1.25e-5,'J_bounds':[.35,2.5],'plane_margin_m_min':float(pm.min()),'plane_margin_m_max':float(pm.max())},
       'budget_s':args.budget_s,'max_displacement_mm':args.max_displacement_mm,'limitations':['Construction fit only. Route planes are necessary constraints; exact solid containment and self/retained-surface audits remain mandatory.','No detailed-source mapping, physical mass transfer, physiological thickness, or native admission is claimed.']}
    (out/'preflight-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(out/'fit-input.npz',points_baseline_f32=X0.astype(np.float32),tetrahedra=T.astype(np.int32),material_ids=mat.astype(np.int8),reference_points_m=Ref,seed_columns=seed_columns,patch_face_rows=face_rows,patch_columns=patch['patch_columns'],boundary_columns=boundary,movable_columns=movable,touched_tet_rows=touched,constraint_source_rows=full_source,constraint_source_hit_row=full_source,constraint_kind=full_kind,constraint_endpoint_layer=full_layer,constraint_policy=full_policy,constraint_endpoint_column_rows=np.r_[pe,o_node],constraint_endpoint_node_rows=full_endpoints,constraint_endpoint_global_ids=full_endpoint_global,constraint_plane_point_world_m=full_hit,constraint_normal_world=full_norm,constraint_clearance_m=full_margin)
    if args.preflight_only:
        print('PREFLIGHT_PASS',json.dumps({'route_planes':int(plane_count),'seeds':len(seed_columns),'patch_faces':len(face_rows),'boundary_columns':len(boundary),'movable_columns':len(movable),'touched_tets':len(touched),'components':[len(g) for g in groups],'baseline_q_min':float(qbase.min()),'baseline_J_range':[float(Jbase.min()),float(Jbase.max())]}),flush=True);return
    # SLSQP is run componentwise because all cell, layer, plane, and surface-edge
    # terms are within a measured connected component. Fixed patch boundary stays exact.
    from column_fit_support import plane_constraint_slack_and_jacobian,quality_and_column_jacobian,J_and_column_jacobian
    start=time.monotonic();current_mm=X0*1000.;component_reports=[];budget_stop=False
    route_endpoint_col=full_endpoints%2411
    # Material-specific target guards absorb solver/emission roundoff; exact [.35,2.5] remains final gate.
    q_target=1.25e-5;jlo=.3501;jhi=2.4999
    for ci,gcols in sorted(enumerate(groups),key=lambda x:(len(x[1]),x[0])):
        if time.monotonic()-start>=args.budget_s:budget_stop=True;break
        gcols=np.asarray(gcols,np.int64)
        trows=touched[np.isin(T[touched]%2411,gcols).any(axis=1)]
        if not len(trows):continue
        lookup=np.full(2411,-1,np.int64);lookup[gcols]=np.arange(len(gcols))
        route_mask=np.isin(route_endpoint_col,gcols)
        # Every endpoint row in a hit shares the same column and therefore one component.
        ph=full_hit[route_mask];pnloc=full_norm[route_mask];pmloc=full_margin[route_mask];en=full_endpoints[route_mask]
        hitrows=full_source[route_mask]
        if len(en) and np.any(~np.isin(en%2411,gcols)):raise ValueError('one route hit spans components unexpectedly')
        cols_edges=edges[np.isin(edges,gcols).any(axis=1)]
        edge_local=[]
        for a,b in cols_edges:
            ia=int(lookup[a]);ib=int(lookup[b])
            if ia>=0 or ib>=0:edge_local.append((ia,ib))
        edge_local=np.asarray(edge_local,np.int64).reshape(-1,2)
        base_local=np.zeros((len(gcols),3),np.float64)
        def assemble(delta):
            X=current_mm.copy()
            d=np.asarray(delta,np.float64).reshape(-1,3)
            for layer in range(3):X[layer*2411+gcols]+=d
            return X
        def obj(x):
            d=np.asarray(x).reshape(-1,3);v=float(np.sum(d*d))
            if len(edge_local):
                da=np.zeros((len(edge_local),3));db=np.zeros_like(da);a,b=edge_local.T
                ma=a>=0;mb=b>=0
                da[ma]=d[a[ma]];db[mb]=d[b[mb]]
                v+=.08*float(np.sum((da-db)**2))
            return v
        def objjac(x):
            d=np.asarray(x).reshape(-1,3);G=2*d.copy()
            if len(edge_local):
                a,b=edge_local.T;da=np.zeros((len(a),3));db=np.zeros_like(da);ma=a>=0;mb=b>=0
                da[ma]=d[a[ma]];db[mb]=d[b[mb]];gg=.16*(da-db)
                np.add.at(G,a[ma],gg[ma]);np.add.at(G,b[mb],-gg[mb])
            return G.ravel()
        def cfun(x,jac=False):
            Xmm=assemble(x);q,Gq=quality_and_column_jacobian(Xmm,T,trows,gcols)
            sel=mat[trows]==1;fat=mat[trows]==2
            J,gj=J_and_column_jacobian(Xmm/1000.,Ref,T,trows,gcols);gj=gj*.001
            route=plane_constraint_slack_and_jacobian(Xmm,gcols,ph*1000.,pnloc,en[:,None],pmloc*1000.)
            vals=np.r_[(q-q_target)*1e5,(J[sel]-jlo)*100.,(jhi-J[sel])*100.,(J[fat]-jlo)*100.,(jhi-J[fat])*100.,route['slack']]
            if not jac:return vals
            JJ=[Gq*1e5,gj[sel]*100.,-gj[sel]*100.,gj[fat]*100.,-gj[fat]*100.,route['jacobian']]
            return np.vstack(JJ)
        # Confirm the warm start and its Jacobian are finite before the optimization call.
        x0=np.zeros(3*len(gcols),np.float64);v0=cfun(x0,False);J0=cfun(x0,True)
        if J0.shape!=(len(v0),len(x0)) or not np.all(np.isfinite(J0)) or not np.all(np.isfinite(v0)):raise ValueError(f'component {ci} constraint preflight failed')
        # A bounded callback saves the best iterate and stops on the global deadline.
        last={'x':x0.copy(),'iter':0}
        def cb(x):
            last['x']=np.asarray(x,np.float64).copy();last['iter']+=1
            if last['iter']%3==0:
                np.savez_compressed(out/'solver-checkpoint.npz',component=np.asarray(ci),columns=gcols,translation_mm=last['x'].reshape(-1,3),elapsed_s=np.asarray(time.monotonic()-start))
            if time.monotonic()-start>=args.budget_s:raise TimeoutError('global fit budget')
        result=None
        try:
            result=minimize(obj,x0,jac=objjac,method='SLSQP',bounds=[(-args.max_displacement_mm,args.max_displacement_mm)]*len(x0),constraints=[{'type':'ineq','fun':cfun,'jac':lambda x:cfun(x,True)}],callback=cb,options={'maxiter':60,'ftol':2e-7,'disp':False})
            accepted=np.asarray(result.x,np.float64)
        except TimeoutError:
            accepted=last['x'];budget_stop=True
        Xacc=assemble(accepted)
        vals=cfun(accepted,False)
        comp={'component':int(ci),'columns':int(len(gcols)),'dofs':int(len(x0)),'tets':int(len(trows)),'route_endpoint_rows':int(len(en)),'iterations':int(getattr(result,'nit',last['iter']) if result is not None else last['iter']),'solver_success':bool(result.success) if result is not None else False,'solver_message':str(result.message) if result is not None else 'budget stop','minimum_constraint_slack':float(vals.min()) if len(vals) else float('inf'),'max_translation_mm':float(np.linalg.norm(accepted.reshape(-1,3),axis=1).max()),'elapsed_s':time.monotonic()-start}
        component_reports.append(comp);current_mm=Xacc
        (out/'fit-progress.json').write_text(json.dumps({'status':'running','components':component_reports,'elapsed_s':time.monotonic()-start},indent=2,sort_keys=True)+'\n')
        if budget_stop:break
    Xcand=current_mm/1000.;Xf32=Xcand.astype(np.float32).astype(np.float64)
    qf,detf=quality(Xf32,T);Jf=Jvalues(Xf32,Ref,T)
    q64,det64=quality(Xcand,T);J64=Jvalues(Xcand,Ref,T)
    route_eval=plane_constraint_slack_and_jacobian(Xf32*1000.,np.asarray([],np.int64),full_hit*1000.,full_norm,full_endpoints[:,None],full_margin*1000.)
    gaps=route_eval['slack']/1000.
    base_layer=layer_measure(X0,patch['patch_columns'],X0)
    final_layer=layer_measure(Xf32,patch['patch_columns'],X0)
    vols={str(k):float(np.sum(detf[mat==k])/6.) for k in (1,2)}
    base_vols={str(k):float(np.sum(quality(X0,T)[1][mat==k])/6.) for k in (1,2)}
    ok=bool(np.all(detf>0)&np.all(qf>=1e-5)&np.all(Jf>=.35)&np.all(Jf<=2.5)&np.all(gaps>=-2e-7)&np.all(final_layer[0]>=final_layer[2]-2e-8)&np.all(final_layer[1]>=final_layer[3]-2e-8))
    moved=np.flatnonzero(np.linalg.norm(Xf32-X0,axis=1)>0)
    status='candidate_basic_geometry_pass_not_admitted' if ok and not budget_stop else ('bounded_fit_passed_basic_gates' if ok else 'candidate_rejected_basic_gate')
    np.savez_compressed(out/'candidate.npz',points_world_f32=Xf32.astype(np.float32),tetrahedra=T.astype(np.int32),material_ids=mat.astype(np.int8),global_nodes=ap['global_nodes'],movable_columns=movable,patch_face_rows=face_rows,boundary_columns=boundary,seed_columns=seed_columns,plane_source_rows=full_source,plane_endpoint_column_rows=pe,plane_endpoint_node_rows=route_endpoint_local,plane_endpoint_global_ids=peg,plane_point_world_m=pi,plane_normal_world=pn,plane_clearance_m=pm,plane_final_endpoint_gap_m=gaps,component_id=gid[movable],component_translations_mm=(current_mm-X0*1000.)[movable],touched_tet_rows=touched,q_float32_all=qf,J_float32_all=Jf)
    report={'schema':'numi.human.shared-column-bone-entry-fit.v1','status':status,'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'route_status':route_data.get('status'),'route_report_sha256':sha(ROUTE),'baseline_O12_sha256':sha(O12_NPZ),'preflight':report,'components':component_reports,'elapsed_s':time.monotonic()-start,'final':{'basic_gate_pass':ok,'float32_nonpositive_tets':int(np.sum(detf<=0)),'float32_q_below_1e-5':int(np.sum(qf<1e-5)),'float32_q_below_1.25e-5':int(np.sum(qf<1.25e-5)),'float32_J_below_.35':int(np.sum(Jf<.35)),'float32_J_above_2.5':int(np.sum(Jf>2.5)),'min_q_f32':float(qf.min()),'min_J_f32':float(Jf.min()),'max_J_f32':float(Jf.max()),'constraint_violation_count':int(np.sum(gaps < -2e-7)),'route_endpoint_violation_count':int(np.sum(gaps[:len(peg)] < -2e-7)),'baseline_O12_violation_count':int(np.sum(gaps[len(peg):] < -2e-7)),'minimum_constraint_gap_m':float(gaps.min()),'material_volume_m3':vols,'baseline_material_volume_m3':base_vols,'material_volume_relative_change':{k:(vols[k]/base_vols[k]-1) for k in vols},'moved_node_count':int(len(moved)),'max_node_displacement_m':float(np.linalg.norm(Xf32-X0,axis=1).max()),'layer_projection':{'min_skin_mm':float(final_layer[0].min()*1000),'min_fat_mm':float(final_layer[1].min()*1000),'min_skin_floor_mm':float(final_layer[2].min()*1000),'min_fat_floor_mm':float(final_layer[3].min()*1000),'skin_floor_violations':int(np.sum(final_layer[0]<final_layer[2]-2e-8)),'fat_floor_violations':int(np.sum(final_layer[1]<final_layer[3]-2e-8))}},'limitations':['Only certified 041 construction planes were used. Exact all-bank solid containment must be rerun on this candidate.','No exact self/retained surface audit, detailed-source skin transfer, conservative source-mass map, or native compile/run is included.','The volume changes are geometric diagnostics, not a regenerated density/mass ledger.','Construction layer floors preserve mechanical feasibility only; physiological thickness is not qualified.']}
    report['candidate_sha256']=sha(out/'candidate.npz');(out/'fit-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('FINAL',json.dumps({'status':status,'candidate':str(out/'candidate.npz'),'sha256':report['candidate_sha256'],'final':report['final'],'elapsed_s':report['elapsed_s']},indent=2),flush=True)

if __name__=='__main__':main()
