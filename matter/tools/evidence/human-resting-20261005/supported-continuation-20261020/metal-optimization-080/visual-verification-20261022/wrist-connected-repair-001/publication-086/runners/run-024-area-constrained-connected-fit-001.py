#!/usr/bin/env python3
"""Bounded sequential-convex connected surface fit with oriented area guards.

Construction experiment only. No biological, physical, or native-runtime claim.
Inputs are the frozen 018 start, 702 reference, and first-union exit proposals.
"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from area_scp_support_023 import project_ball_halfspace, InfeasibleProjection, ball_intersection_support, progressive_plane_lower, project_two_balls_halfspace

D=Path('/Users/n/Work/numi-human-connected-surface-069')
R=D/'full-connected-clearance-150'
START=R/'run-018-generated-node-exit-targets-002/anchor-corrected-refined-start.npz'
TARGET_JSON=R/'run-018-generated-node-exit-targets-002/generated-node-nearest-union-exits-v1.json'
TARGET_NPZ=R/'run-018-generated-node-exit-targets-002/generated-node-nearest-union-exits-v1.npz'
BASE=R/'run-009-certified-source-normal-union-rays/endpoint-candidate.npz'
OLD_TRIAL=R/'run-020-nearest-exit-connected-fit-002/candidate.npz'
PRED=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py')
OUT=R/'run-024-area-constrained-connected-fit-001'
PIN={'start':'34dc98d228ccc2ae9dc7ac51a65dcf45d894f607186531ac7d0e099f9da42d25','target_json':'792a5109e8b409c225589c0f87c92b54eec4693fcf9caf1d99779b4317b17891','target_npz':'1bb0111f39bfe183f07d271380f76c5087be0d40d6a040ce6ff988e5277452f1','base':'70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31','old_trial':'cad4e517de2c498aeb18ec7b81ddf6300715f5070355690b0c5fa7fa6bc2dc02','predicate':'f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'}
TARGET_MAX_MM=6.0
SOURCE_CAP_MM=7.0
STEP_MM=2.0
AREA_FLOOR_RATIO=0.1
MAX_SCP=4
MAX_ADMM=1200
MAX_SOLVE_S=35.0
RHO=3.0
LAMBDA=1.5
TARGET_WEIGHT=8.0
ETA=.02
AREA_SWEEPS=80
NODE_SWEEPS=120


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    deps=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/incremental-exact-audit-owner-253/src')
    sys.path.insert(0,str(deps)); sys.path.insert(0,str(Path(path).parents[1]))
    import numilab_human
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def project_ball(v,c,r):
    v=np.asarray(v,float); c=np.asarray(c,float); w=v-c
    norm=np.linalg.norm(w,axis=1)
    return c+w*np.minimum(1.0,float(r)/np.maximum(norm,1e-300))[:,None]

def project_node_sets(v, source_center, source_cap, plane_rows, plane_normals, plane_lower, anchors, trust_center, trust_radius):
    """Exact per-node active-set projection onto source/trust balls and exit plane."""
    v=np.asarray(v,float); source_center=np.asarray(source_center,float); trust_center=np.asarray(trust_center,float)
    out=project_ball(v,trust_center,trust_radius)
    plane_by_node={int(i):(np.asarray(n,float),float(h)) for i,n,h in zip(plane_rows,plane_normals,plane_lower)}
    anchor_set=set(map(int,anchors)); overlap_count=0; full_enumeration_count=0
    for row in range(len(v)):
        if row in anchor_set:
            out[row]=0.0
            continue
        pair=plane_by_node.get(row)
        source_contains_trust=np.linalg.norm(trust_center[row]-source_center[row])+trust_radius<=source_cap+1e-10
        if pair is not None:
            n,h=pair
            if source_contains_trust:
                out[row]=project_ball_halfspace(v[row:row+1],trust_center[row:row+1],trust_radius,n[None,:],np.array([h]),np.array([0],np.int64))[0]
            else:
                overlap_count+=1; full_enumeration_count+=1
                out[row]=project_two_balls_halfspace(v[row],source_center[row],source_cap,trust_center[row],trust_radius,n,h)
        elif np.linalg.norm(out[row]-source_center[row])>source_cap+1e-10:
            overlap_count+=1; full_enumeration_count+=1
            out[row]=project_two_balls_halfspace(v[row],source_center[row],source_cap,trust_center[row],trust_radius)
    for row,(n,h) in plane_by_node.items():
        if row in anchor_set and h>1e-10: raise InfeasibleProjection(row,h,source_cap)
    ball_violation=np.maximum(0.0,np.linalg.norm(out-source_center,axis=1)-source_cap)
    trust_violation=np.maximum(0.0,np.linalg.norm(out-trust_center,axis=1)-trust_radius)
    plane_violation=np.array([max(0.0,h-float(np.dot(n,out[row]))) for row,(n,h) in plane_by_node.items()])
    err=max(float(ball_violation.max(initial=0.0)),float(trust_violation.max(initial=0.0)),float(plane_violation.max(initial=0.0)))
    return out, {"method":"exact_active_set_candidates","intersection_residual_mm":err,"overlap_nodes":overlap_count,"full_active_set_enumerations":full_enumeration_count}

def color_face_rows(tri):
    """Greedy deterministic coloring so rows in each batch share no vertices."""
    groups=[]; used=[]
    for ri, nodes in enumerate(np.asarray(tri,np.int64)):
        node_set=set(map(int,nodes)); placed=False
        for ci, occupied in enumerate(used):
            if node_set.isdisjoint(occupied):
                groups[ci].append(ri); occupied.update(node_set); placed=True; break
        if not placed:
            groups.append([ri]); used.append(set(node_set))
    return [np.asarray(g,np.int64) for g in groups]

def area_rows_at(current_mm, ref_cross, face_ids, faces):
    tri=np.asarray(faces,np.int64)[face_ids]
    ref_area=np.linalg.norm(ref_cross[face_ids],axis=1)
    if np.any(ref_area<=1e-16): raise RuntimeError('degenerate reference face in area constraints')
    normals=ref_cross[face_ids]/ref_area[:,None]
    a,b,c=current_mm[tri[:,0]],current_mm[tri[:,1]],current_mm[tri[:,2]]
    e1=b-a; e2=c-a
    gb=np.cross(e2,normals); gc=np.cross(normals,e1); ga=-gb-gc
    glocal=np.stack((ga,gb,gc),axis=1)
    q=np.einsum('ij,ij->i',normals,np.cross(e1,e2))
    return tri,glocal,AREA_FLOOR_RATIO*ref_area-q,ref_area,normals,q

def project_area_rows(v,tri,g,rhs,groups,max_sweeps=AREA_SWEEPS,tol=1e-4):
    """Dykstra projection onto sparse per-face linearized oriented-area halfspaces."""
    d=np.asarray(v,float).copy(); corr=np.zeros(len(tri),float)
    g2=np.einsum('mij,mij->m',g,g)
    bad=np.flatnonzero(g2<=1e-24)
    if len(bad) and np.any(rhs[bad]>tol):
        return d, {'converged':False,'reason':'zero_gradient_infeasible_area_rows','bad_rows':bad[:20].tolist(),'max_violation_mm2':float(rhs[bad].max())}
    g2=np.maximum(g2,1e-24)
    maxv=float('inf')
    for sweep in range(max_sweeps):
        for ids in groups:
            nodes=tri[ids]
            y=d[nodes]+corr[ids,None,None]*g[ids]
            val=np.einsum('mij,mij->m',y,g[ids])
            step=np.maximum(0.0,(rhs[ids]-val)/g2[ids])
            z=y+step[:,None,None]*g[ids]
            d[nodes]=z
            corr[ids]=-step
        val=np.einsum('mij,mij->m',d[tri],g)
        maxv=float(np.maximum(0.0,rhs-val).max(initial=0.0))
        if maxv<=tol: break
    return d, {'converged':bool(maxv<=tol),'sweeps':sweep+1,'max_violation_mm2':maxv}

def area_ratios_mm(points_m, faces, ref_cross):
    p=np.asarray(points_m,np.float32).astype(np.float64)*1000.0
    tri=np.asarray(faces,np.int64)
    cc=np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
    ref_area=np.linalg.norm(ref_cross,axis=1)
    ratios=np.einsum('ij,ij->i',cc,ref_cross)/np.maximum(ref_area*ref_area,1e-300)
    return ratios,cc

def add_one_ring(active, faces, violation_rows):
    bad=np.asarray(violation_rows,np.int64)
    if len(bad)==0:return active
    verts=np.unique(np.asarray(faces,np.int64)[bad].ravel())
    touch=np.flatnonzero(np.isin(faces,verts).any(axis=1))
    return np.unique(np.r_[active,bad,touch]).astype(np.int64)

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    started=time.monotonic()
    for p,k in [(START,'start'),(TARGET_JSON,'target_json'),(TARGET_NPZ,'target_npz'),(BASE,'base'),(OLD_TRIAL,'old_trial'),(PRED,'predicate')]:
        if sha(p)!=PIN[k]:raise RuntimeError(f'input SHA mismatch {k}: {sha(p)}')
    z=np.load(START,allow_pickle=False); base=np.load(BASE,allow_pickle=False); tj=json.loads(TARGET_JSON.read_text()); tn=np.load(TARGET_NPZ,allow_pickle=False); old=np.load(OLD_TRIAL,allow_pickle=False)
    X=np.asarray(z['points_world_m'],np.float32).astype(np.float64); F=np.asarray(z['source_O_faces'],np.int32)
    if X.shape!=(10145,3) or F.shape!=(20245,3):raise RuntimeError('unexpected start geometry shape')
    if tj.get('inputs',{}).get('candidate_sha256')!=PIN['start'] or tj.get('status')!='PASS_FIRST_UNION_EXIT_TARGETS_ONLY' or int(tj.get('passing_targets',0))!=380:raise RuntimeError('target report mismatch')
    rows=np.asarray(tn['query_rows'],np.int64); target_xyz=np.asarray(tn['target_world_f32_m'],np.float32); pm=np.asarray(tn['pass_mask'],bool)
    if len(rows)!=380 or target_xyz.shape!=(380,3) or not np.all(pm):raise RuntimeError('incomplete target NPZ')
    target_map={int(r['query_row']):r for r in tj['targets']}
    if len(target_map)!=380:raise RuntimeError('target JSON row mismatch')
    nold=len(base['points_world_m']); ind=np.asarray(z['source_prolongation_indptr'],np.int64); cols=np.asarray(z['source_prolongation_node_rows'],np.int64); w=np.asarray(z['source_prolongation_weights'],np.float64)
    if len(ind)!=len(X)+1 or ind[0]!=0 or ind[-1]!=len(cols) or len(cols)!=len(w) or np.any(np.diff(ind)<0) or np.any(cols<0) or np.any(cols>=nold):raise RuntimeError('bad source prolongation CSR')
    sums=np.asarray([w[ind[i]:ind[i+1]].sum() for i in range(len(X))])
    if np.any(np.diff(ind)==0) or np.max(np.abs(sums-1.0))>2e-10:raise RuntimeError('source prolongation rows not affine')
    P=sparse.csr_matrix((w,cols,ind),shape=(len(X),nold))
    R0=np.asarray(P@np.asarray(base['points_before_joint_self_pair_repair_m'],np.float64),np.float64)
    anchors=np.asarray(z['typed_coarse_anchor_output_rows_018'],np.int64); anchor_ids=np.asarray(z['typed_coarse_anchor_source_ids_018'],np.int64)
    if len(anchors)!=237 or len(np.unique(anchors))!=237 or len(anchor_ids)!=237:raise RuntimeError('typed anchor set incomplete')
    if not np.array_equal(np.ascontiguousarray(X[anchors],dtype=np.float32).tobytes(),np.ascontiguousarray(R0[anchors],dtype=np.float32).tobytes()):raise RuntimeError('true anchors not at registered reference')
    Xmm=X*1000.0; center=(R0-X)*1000.0; start_disp=np.linalg.norm(center,axis=1)
    if float(start_disp.max())>SOURCE_CAP_MM+2e-5:raise RuntimeError('start exceeds declared source cap')
    # Read proposal rows and validate first-union witness bytes/frames.
    prop_rows=[]; prop_dirs=[]; prop_h=[]; prop_t=[]; unresolved=[]
    for j,rr0 in enumerate(rows):
        row=int(rr0); rr=target_map[row]
        if rr.get('status')!='PASS_FIRST_UNION_EXIT':raise RuntimeError(f'proposal {row} lacks first-union status')
        if not np.array_equal(target_xyz[j].view(np.uint32),np.asarray(rr['target_world_f32_m'],np.float32).view(np.uint32)):raise RuntimeError(f'proposal array/report mismatch at {row}')
        st=np.asarray(rr['start_world_f32_m'],np.float32)
        if not np.array_equal(st.view(np.uint32),X[row].astype(np.float32).view(np.uint32)):raise RuntimeError(f'proposal start mismatch at {row}')
        d=np.asarray(rr['proposal_direction_world_unit'],float); dn=float(np.linalg.norm(d))
        if dn<=0:raise RuntimeError(f'zero proposal direction at {row}')
        d=d/dn; dist=float(np.dot(d,target_xyz[j].astype(float)*1000.0-st.astype(float)*1000.0))
        if dist < -1e-7 or abs(dist-float(rr['ray_distance_m'])*1000.0)>0.02:raise RuntimeError(f'proposal distance mismatch at {row}')
        vec=(target_xyz[j].astype(float)-st.astype(float))*1000.0
        if dist>TARGET_MAX_MM+1e-7:
            unresolved.append({'query_row':row,'reason':'first_union_target_exceeds_6mm_construction_target_cap','distance_mm':dist})
        else:
            source_support=float(np.dot(d,center[row])+SOURCE_CAP_MM)
            if dist>source_support+1e-7:
                unresolved.append({'query_row':row,'reason':'target_halfspace_outside_source_ball','distance_mm':dist,'max_source_ball_projection_mm':source_support})
        prop_rows.append(row);prop_dirs.append(d);prop_h.append(dist);prop_t.append(vec)
    prop_rows=np.asarray(prop_rows,np.int64); prop_dirs=np.asarray(prop_dirs,float); prop_h=np.asarray(prop_h,float); prop_t=np.asarray(prop_t,float)
    hard_mask=(prop_h<=TARGET_MAX_MM+1e-7)&(prop_h<=np.einsum('ij,ij->i',prop_dirs,center[prop_rows])+SOURCE_CAP_MM+1e-7)&(~np.isin(prop_rows,anchors))
    hard_rows=prop_rows[hard_mask]; hard_dirs=prop_dirs[hard_mask]; hard_targets=prop_h[hard_mask]
    if len(hard_rows)==0:raise RuntimeError('no feasible ≤6mm hard exit proposals')
    # Laplacian/target objective.
    edges=np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])).astype(np.int64); edges.sort(axis=1); edges=np.unique(edges,axis=0)
    lengths=np.linalg.norm(X[edges[:,0]]-X[edges[:,1]],axis=1); positive=lengths[lengths>0]
    if len(positive)!=len(lengths):raise RuntimeError('zero-length O edge')
    med=float(np.median(positive)); ew=np.clip(med/positive,.05,20.0)
    er=np.r_[edges[:,0],edges[:,1],edges[:,0],edges[:,1]]; ec=np.r_[edges[:,0],edges[:,1],edges[:,1],edges[:,0]]; ev=np.r_[ew,ew,-ew,-ew]
    L=sparse.coo_matrix((ev,(er,ec)),shape=(len(X),len(X))).tocsr()
    tw=np.zeros(len(X)); tt=np.zeros((len(X),3))
    for i,row in enumerate(prop_rows):
        tw[row]=TARGET_WEIGHT; tt[row]=prop_t[i]
        nn=float(np.linalg.norm(tt[row]))
        if nn>TARGET_MAX_MM:tt[row]*=TARGET_MAX_MM/nn
    A=(L*LAMBDA)+sparse.diags(tw+ETA)
    ref_cross=np.cross(Xmm[F[:,1]]-Xmm[F[:,0]],Xmm[F[:,2]]-Xmm[F[:,0]])
    ref_area=np.linalg.norm(ref_cross,axis=1)
    if np.any(ref_area<=1e-10):raise RuntimeError('degenerate emitted start face')
    # Seed the active area set with all 37 measured prior reversals and their full face one-ring.
    oldY=np.asarray(old['points_world_m'],np.float32).astype(np.float64)
    old_cross=np.cross(oldY[F[:,1]]-oldY[F[:,0]],oldY[F[:,2]]-oldY[F[:,0]])
    start_cross=np.cross(X[F[:,1]]-X[F[:,0]],X[F[:,2]]-X[F[:,0]])
    flipped=np.flatnonzero(np.einsum('ij,ij->i',old_cross,start_cross)<=0)
    if len(flipped)!=37:raise RuntimeError(f'expected frozen run020 37 flipped faces, got {len(flipped)}')
    active=add_one_ring(np.empty(0,np.int64),F,flipped)
    # Baseline sparse factor is reused across SCP iterations.
    rho=RHO; Q=(A+sparse.eye(len(X),format='csr')*(2*rho)).tocsc(); factor_t=time.monotonic(); lu=splu(Q); factor_s=time.monotonic()-factor_t
    pred=module('numilab_human._area_scp_021',PRED)
    def exact_self(points):
        rec,_=pred.float32_surface_records(np.asarray(points,np.float32),F,source_seam_groups=())
        return pred._audit_self_pairs_touching_faces(rec,set(range(len(F))))
    start_self=exact_self(X)
    if not start_self.get('complete_surface_audit') or len(start_self.get('triangle_pairs',[])):
        raise RuntimeError('SCP start fails complete exact self audit')
    dcur=np.zeros_like(Xmm); stages=[]; accepted_any=False; stop_reason='max_scp_iterations'
    for scp in range(MAX_SCP):
        t0=time.monotonic(); cur_m=(Xmm+dcur)/1000.0
        # Progressive plane targets: at most 2 mm additional certified progress per SCP step.
        curproj=np.einsum('ij,ij->i',hard_dirs,dcur[hard_rows])
        maxproj=np.einsum('ij,ij->i',hard_dirs,center[hard_rows])+SOURCE_CAP_MM
        support=np.asarray([ball_intersection_support(center[row],SOURCE_CAP_MM,dcur[row],STEP_MM,n) for row,n in zip(hard_rows,hard_dirs)],dtype=np.float64)
        lower=progressive_plane_lower(curproj,hard_targets,support,0.95)
        # no positive hard constraint may be imposed on fixed anchors
        tri,glocal,base_rhs,sel_area,normals,qcur=area_rows_at((cur_m*1000.0),ref_cross,active,F)
        dloc=dcur[tri]
        arhs=base_rhs+np.einsum('mij,mij->m',glocal,dloc)
        groups=color_face_rows(tri)
        # The area constraints are tied to emitted F32 current geometry; all faces are checked afterward.
        rhs=tw[:,None]*tt
        x=dcur.copy(); zn=dcur.copy(); za=dcur.copy(); un=np.zeros_like(Xmm); ua=np.zeros_like(Xmm)
        log=[]; converged=False; node_projection={}; area_projection={}; solve_t=time.monotonic()
        for it in range(MAX_ADMM):
            x=lu.solve((rhs+rho*((zn-un)+(za-ua))).astype(np.float64))
            oldzn=zn; oldza=za
            zn,node_projection=project_node_sets(x+un,center,SOURCE_CAP_MM,hard_rows,hard_dirs,lower,anchors,dcur,STEP_MM)
            za,area_projection=project_area_rows(x+ua,tri,glocal,arhs,groups)
            un+=x-zn; ua+=x-za
            pr=max(float(np.max(np.linalg.norm(x-zn,axis=1))),float(np.max(np.linalg.norm(x-za,axis=1))))
            du=rho*max(float(np.max(np.linalg.norm(zn-oldzn,axis=1))),float(np.max(np.linalg.norm(za-oldza,axis=1))))
            if it%20==0 or it==MAX_ADMM-1:log.append({'iteration':it+1,'primal_inf_mm':pr,'dual_inf_mm':du,'area_projection':area_projection,'node_projection':node_projection})
            if pr<2e-4 and du<2e-4 and area_projection.get('converged',False) and node_projection.get('intersection_residual_mm',float('inf')) < 2e-4:
                converged=True;break
            if time.monotonic()-solve_t>MAX_SOLVE_S:break
        solve_s=time.monotonic()-solve_t
        if area_projection.get('reason') or node_projection.get('reason'):
            stop_reason='projection_infeasible'; stages.append({'scp_iteration':scp+1,'status':stop_reason,'area':area_projection,'node':node_projection});break
        if not converged:
            stop_reason='admm_not_converged'; stages.append({'scp_iteration':scp+1,'status':stop_reason,'admm':log[-1] if log else {},'solve_seconds':solve_s});break
        raw=0.5*(zn+za)
        # Evaluate every face on emitted Float32 coordinates; backtrack before exact-self calls.
        chosen=None; trial_records=[]; newly_bad=[]
        current_proj=np.einsum('ij,ij->i',hard_dirs,dcur[hard_rows])
        for alpha in [1.0,0.75,0.5,0.25]:
            td=dcur+alpha*(raw-dcur)
            yp=((Xmm+td)/1000.0).astype(np.float32).astype(np.float64)
            if len(anchors):yp[anchors]=X[anchors]
            ratios,_=area_ratios_mm(yp,F,ref_cross)
            flipped_now=np.flatnonzero(ratios<=0.0)
            active_low=np.flatnonzero(ratios[active] < AREA_FLOOR_RATIO-5e-6)
            capdist=np.linalg.norm(yp-R0,axis=1)*1000.0
            capok=bool(np.all(capdist<=SOURCE_CAP_MM+0.002))
            anchorok=bool(np.ascontiguousarray(yp[anchors],dtype=np.float32).tobytes()==np.ascontiguousarray(X[anchors],dtype=np.float32).tobytes())
            stepmax=float(np.linalg.norm(yp-((Xmm+dcur)/1000.0),axis=1).max()*1000.0)
            stepok=stepmax<=STEP_MM+0.003
            pnow=np.einsum('ij,ij->i',hard_dirs,(yp[hard_rows]*1000.0-Xmm[hard_rows]))
            progressok=bool(np.all(pnow>=current_proj-0.002))
            local_area_ok=(len(flipped_now)==0 and len(active_low)==0)
            self_report={}; self_pairs=np.empty((0,2),dtype=np.int64)
            if capok and anchorok and stepok and progressok and local_area_ok:
                self_report=exact_self(yp)
                self_pairs=np.asarray(self_report.get('triangle_pairs',[]),np.int64).reshape(-1,2)
            exactok=bool(self_report.get('complete_surface_audit') and len(self_pairs)==0)
            good=capok and anchorok and stepok and progressok and local_area_ok and exactok
            trial_records.append({'alpha':alpha,'max_cumulative_source_displacement_mm':float(capdist.max()),'source_cap_7mm_ok':capok,'anchors_bitwise_unchanged':anchorok,'max_increment_mm':stepmax,'step_2mm_ok':stepok,'exit_progress_nonregressing':progressok,'hard_exit_rows':len(hard_rows),'hard_exit_max_remaining_mm':float(np.maximum(0,hard_targets-pnow).max(initial=0)),'allface_flip_count':int(len(flipped_now)),'first_flipped_face_rows':flipped_now[:20].tolist(),'active_face_below_10pct_count':int(len(active_low)),'exact_self_complete':bool(self_report.get('complete_surface_audit',False)),'exact_self_pair_count':int(len(self_pairs)),'self_pairs_first8':self_pairs[:8].tolist(),'accepted':bool(good)})
            if len(flipped_now) or len(active_low):
                newly_bad.extend(np.asarray(active[np.asarray(active_low,np.int64)],np.int64).tolist())
                newly_bad.extend(flipped_now.tolist())
                newly_bad.extend(np.flatnonzero(ratios < AREA_FLOOR_RATIO).tolist())
            if good:
                chosen=(alpha,yp,self_report,self_pairs,ratios)
                break
        stage={'scp_iteration':scp+1,'status':'ACCEPTED' if chosen else 'NO_ACCEPTABLE_BACKTRACK','active_area_face_count':len(active),'active_face_color_count':len(groups),'area_linearized_max_violation_mm2':float(area_projection.get('max_violation_mm2',float('inf'))),'admm_iterations':log[-1]['iteration'] if log else 0,'admm_residual_log':log,'node_projection':node_projection,'area_projection':area_projection,'factor_seconds':factor_s,'solve_seconds':solve_s,'elapsed_seconds':time.monotonic()-t0,'exit_constraint_progress_max_mm':float((lower-curproj).max(initial=0)),'trials':trial_records}
        stages.append(stage)
        if chosen:
            alpha,yp,sr,sp,ratios=chosen
            dcur=yp*1000.0-Xmm; accepted_any=True
            if np.max(np.maximum(0,hard_targets-np.einsum('ij,ij->i',hard_dirs,dcur[hard_rows])))<=0.05:
                stop_reason='all_feasible_exit_targets_reached';break
            continue
        # Refine the linearized face set for the next SCP pass; do not mutate the retained iterate.
        add=np.unique(np.asarray(newly_bad,np.int64))
        if len(add):
            active=add_one_ring(active,F,add)
            stage['added_area_rows_for_next_scp']=int(len(active))
            continue
        stop_reason='no_safe_area_and_self_backtrack';break
    # Preserve the latest accepted point. If no update passed local gates, preserve the start as a failed no-op receipt.
    final_m=(Xmm+dcur)/1000.0
    final_f32=final_m.astype(np.float32)
    final_ratios,_=area_ratios_mm(final_f32,F,ref_cross)
    final_flip=np.flatnonzero(final_ratios<=0)
    OUT.mkdir(parents=True)
    payload={k:z[k] for k in z.files};payload['points_world_m']=final_f32
    CAND=OUT/'candidate.npz'
    with CAND.open('xb') as fp:np.savez_compressed(fp,**payload)
    if len(final_flip):np.save(OUT/'face-area-nonpositive-rows.npy',final_flip)
    report={'schema':'numi.human.area-constrained-connected-scp.v3','status':'CANDIDATE_WRITTEN_UNADMITTED' if accepted_any else 'NO_ACCEPTED_SCP_STEP_START_PRESERVED','classification':'construction_geometry_only_no_anatomical_physical_or_native_admission','inputs':{'start_sha256':sha(START),'base_702_sha256':sha(BASE),'old_rejected_fit_sha256':sha(OLD_TRIAL),'nearest_union_report_sha256':sha(TARGET_JSON),'nearest_union_npz_sha256':sha(TARGET_NPZ),'exact_predicate_sha256':sha(PRED),'support_helper_sha256':sha(Path(__file__).with_name('area_scp_support_023.py'))},'constraints':{'source_cap_mm':SOURCE_CAP_MM,'proposal_target_max_mm':TARGET_MAX_MM,'local_trust_step_mm':STEP_MM,'progressive_lens_support_fraction':0.95,'area_floor_ratio':AREA_FLOOR_RATIO,'area_constrained_seed_faces':flipped.astype(int).tolist(),'area_constrained_seed_face_count':int(len(flipped)),'active_area_faces_final':int(len(active)),'hard_exit_feasible_rows':int(len(hard_rows)),'unresolved_proposals':unresolved,'fixed_anchor_rows':len(anchors),'all_faces_checked_without_rounding_mask':True},'solver':{'method':'SCP of sparse graph-Laplacian soft-target objective; each node uses exact active-set candidate projection onto source ball, exit halfspace, fixed anchor, and 2mm trust ball; oriented-area rows use cyclic Dykstra; full emitted-F32 area and exact self gates before each accepted iterate','target_weight':TARGET_WEIGHT,'laplacian_weight':LAMBDA,'source_regularization':ETA,'admm_rho':RHO,'stop_reason':stop_reason,'accepted_steps':sum(s['status']=='ACCEPTED' for s in stages),'stages':stages},'final_geometry':{'candidate_path':str(CAND),'candidate_sha256':sha(CAND),'points':len(final_f32),'faces':len(F),'changed_nodes':int(np.count_nonzero(np.any(final_f32!=X.astype(np.float32),axis=1))),'max_source_displacement_mm':float(np.linalg.norm(final_f32.astype(float)-R0,axis=1).max()*1000),'source_cap_pass':bool(np.linalg.norm(final_f32.astype(float)-R0,axis=1).max()*1000<=SOURCE_CAP_MM+0.002),'anchors_bitwise_unchanged':bool(np.ascontiguousarray(final_f32[anchors]).tobytes()==np.ascontiguousarray(X.astype(np.float32)[anchors]).tobytes()),'minimum_allface_oriented_area_ratio':float(final_ratios.min()),'allface_nonpositive_count':int(len(final_flip)),'allface_first_nonpositive_rows':final_flip[:30].astype(int).tolist(),'self_audit_ran_on_each_accepted_iterate':True,'wall_seconds':time.monotonic()-started},'limitations':'No exact bone query or material/layer reconstruction included. Exit proposals are geometric first-union suggestions and do not establish anatomical outwardness. A candidate can remain unadmitted even when the listed local gates pass.'}
    (OUT/'fit-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'candidate':str(CAND),'sha256':sha(CAND),'report':str(OUT/'fit-report.json'),'report_sha256':sha(OUT/'fit-report.json'),'stop_reason':stop_reason,'accepted_steps':report['solver']['accepted_steps'],'allface_nonpositive':len(final_flip),'max_source_displacement_mm':report['final_geometry']['max_source_displacement_mm']},indent=2))

if __name__=='__main__': main()
