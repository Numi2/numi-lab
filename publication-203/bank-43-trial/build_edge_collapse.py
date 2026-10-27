#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import hashlib, json
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027'
D=W/'b43-16/attempt-028-edge-collapse-001'
BASE=W/'combined-d071-b109-b61-001/candidate.npz'
TOPO=W/'b43-16/attempt-027-cdt-owner-current-806-001/bulk-patch-topology.json'
BOUND=W/'b43-16/attempt-027-cdt-owner-current-806-001/exact-source-boundary-checks.json'
AUD130=W/'combined-d071-b109-b61-001/audit-130/summary.json'
PIN='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cross2(a,b): return float(a[0]*b[1]-a[1]*b[0])
def orient2(t): return cross2(t[1]-t[0],t[2]-t[0])
def poly_area(p):
    if len(p)<3:return 0.
    return abs(sum(cross2(p[i],p[(i+1)%len(p)]) for i in range(len(p))))/2.
def clip(subject,clipper):
    out=[np.array(v,dtype=np.float64) for v in subject]
    for i in range(3):
        a,b=clipper[i],clipper[(i+1)%3];edge=b-a;inp=out;out=[]
        if not inp:break
        p=inp[-1];dp=cross2(edge,p-a)
        for q in inp:
            dq=cross2(edge,q-a);pin=dp>=-1e-14;qin=dq>=-1e-14
            if pin!=qin:
                den=dp-dq
                if abs(den)<1e-30:raise ValueError('unstable polygon clipping')
                out.append(p+(q-p)*(dp/den))
            if qin:out.append(q)
            p,dp=q,dq
    return out
def vecnorm(x):return np.linalg.norm(x,axis=-1)
def csr_rows(ptr,vals,row):return vals[int(ptr[row]):int(ptr[row+1])]
def node_chart(a,node,parent,face_chart):
    ptr=a['refinement_node_parent_face_indptr'];rows=a['refinement_node_parent_face_rows'];bc=a['refinement_node_parent_face_barycentric']
    lo,hi=map(int,ptr[node:node+2]);found=[np.asarray(bc[i],np.float64) for i in range(lo,hi) if int(rows[i])==int(parent)]
    found.extend(np.asarray(x,np.float64) for x in face_chart.get((int(node),int(parent)),[]))
    if not found:return None
    x=np.asarray(found[0],np.float64)
    if any(not np.allclose(x,y,rtol=0,atol=1e-12) for y in found[1:]):return None
    return x
def main():
    pins={'base':PIN,'topology':'9bbe5a000612ff48200ffec14c10a4642bb87b66d0b105c6b637cb4e7f4c704b','boundary':'71c062c31a2d45c7a62b6e72b439ffee9136f17cfe082848819e7b3c1d132766'}
    if sha(BASE)!=pins['base'] or sha(TOPO)!=pins['topology'] or sha(BOUND)!=pins['boundary']:raise ValueError('pinned input mismatch')
    a={}
    with np.load(BASE,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    topo=json.loads(TOPO.read_text());bound=json.loads(BOUND.read_text());aud=json.loads(AUD130.read_text())
    F=a['source_O_faces'].astype(np.int64);P=a['points_world_m'].astype(np.float64);N=len(P);M=len(F)
    parent=a['refinement_face_parent_face_row'].astype(np.int64);B=a['refinement_face_parent_barycentric'].astype(np.float64)
    target_inside=sorted(set(map(int,topo['inside_target_node_rows'])))
    all_inside={int(w['output_node']) for w in aud['inside_witnesses'] if w.get('classification')=='inside' and int(w.get('ambiguous',-1))==0}
    boundary_nodes=set(map(int,bound['boundary_node_ids']))
    # Preserve source-boundary and typed anchor nodes exactly; an edge collapse may not consume them.
    protected_nodes=set(map(int,a['coarse_boundary_edge_path_output_nodes'].reshape(-1)))
    protected_nodes.update(map(int,a['coarse_boundary_anchor_rows'].reshape(-1)))
    protected_nodes.update(map(int,a['typed_coarse_anchor_output_rows_018'].reshape(-1)))
    edge_faces=defaultdict(list);node_faces=defaultdict(list)
    for fi,t in enumerate(F):
        for n in t:node_faces[int(n)].append(fi)
        for j in range(3):
            u,v=sorted((int(t[j]),int(t[(j+1)%3])))
            edge_faces[(u,v)].append(fi)
    allchart=defaultdict(list)
    face_chart=defaultdict(list)
    for fi,p in enumerate(parent):
        allchart[int(p)].append(fi)
        for j,n in enumerate(F[fi]):face_chart[(int(n),int(p))].append(B[fi,j])
    # CSR sources for material identity checks.
    csrc=(a['face_source_ancestry_indptr'],a['face_source_ancestry_face_rows'])
    cdet=(a['face_fine_detail_ancestry_indptr'],a['face_fine_detail_ancestry_rows'])
    csub=(a['face_fine_subface_ancestry_indptr'],a['face_fine_subface_ancestry_rows'])
    scalar_keys=['face_kind','parent_coarse_face_row','parent_fine_detail_face_row','parent_source_face_marker','parent_fine_subface_row']
    def ids(csr,fr):return tuple(sorted(set(map(int,csr_rows(csr[0],csr[1],fr)))))
    results=[]
    for v in target_inside:
        if v<0 or v>=N:continue
        if int(a['refinement_node_kind'][v])!=1:continue
        # Endpoints of this exact refinement edge are the only source-faithful collapse survivors considered.
        endpoints=sorted(set(map(int,a['refinement_node_parent_edge_rows'][v])))
        for u in endpoints:
            edge=(min(v,u),max(v,u))
            fs=sorted(node_faces.get(v,[])); ef=sorted(edge_faces.get(edge,[]))
            rec={'inside_node':v,'survivor':u,'edge':list(edge),'edge_length_mm':float(np.linalg.norm(P[v]-P[u])*1000),'incident_faces':fs,'edge_faces':ef}
            reason=None
            if len(ef)!=2:reason='edge_not_two_manifold_faces'
            elif v in boundary_nodes:reason='removed_vertex_is_fixed_patch_boundary'
            elif v in protected_nodes:reason='removed_vertex_is_source_boundary_or_typed_anchor'
            elif u in all_inside:reason='survivor_is_inside_some_bank'
            elif not set(fs):reason='empty_star'
            if reason:
                rec['status']='REJECT';rec['reason']=reason;results.append(rec);continue
            # Triangle-mesh link condition.
            nv=set(map(int,F[fs].ravel()))-{v};nu=set(map(int,F[[i for i,t in enumerate(F) if u in t]].ravel()))-{u}
            actual=nv&nu
            opposite=set(map(int,F[ef].ravel()))-{v,u}
            if actual!=opposite:
                rec.update(status='REJECT',reason='link_condition_failure',link_intersection=sorted(actual),edge_opposites=sorted(opposite));results.append(rec);continue
            drop=set(ef);changed=[fi for fi in fs if fi not in drop]
            if not changed:rec.update(status='REJECT',reason='no_surviving_star_faces');results.append(rec);continue
            # Regenerate barycentric coordinates in each changed face's existing older affine chart.
            newB={};chart_reject=None
            for fi in changed:
                p=int(parent[fi]);bc=node_chart(a,u,p,face_chart)
                if bc is None:chart_reject=('survivor_missing_or_ambiguous_historical_chart',fi,p);break
                if np.min(bc)<-1e-12 or np.max(bc)>1+1e-12 or abs(float(bc.sum())-1)>1e-11:chart_reject=('survivor_chart_outside_simplex',fi,p);break
                bnew=B[fi].copy();where=np.flatnonzero(F[fi]==v)
                if len(where)!=1:chart_reject=('bad_inside_vertex_multiplicity',fi,p);break
                bnew[int(where[0])]=bc
                before=orient2(B[fi][:,1:3]);after=orient2(bnew[:,1:3])
                if before<=1e-12 or after<=1e-12:chart_reject=('nonpositive_legacy_chart_face',fi,p);break
                if np.min(bnew)<-1e-12 or np.max(bnew)>1+1e-12 or np.max(np.abs(bnew.sum(axis=1)-1))>1e-11:chart_reject=('new_legacy_barycentric_outside_simplex',fi,p);break
                newB[fi]=bnew
            if chart_reject:
                rec.update(status='REJECT',reason=chart_reject[0],witness={'face':chart_reject[1],'parent':chart_reject[2]});results.append(rec);continue
            # Float32 world shape checks on every affected surviving triangle.
            shape=[];world_reject=None
            for fi in changed:
                oldtri=F[fi];newtri=oldtri.copy();newtri[newtri==v]=u
                if len(set(map(int,newtri)))!=3:world_reject=('degenerate_vertex_identity',fi);break
                p0=P[oldtri];p1=P[newtri];n0=np.cross(p0[1]-p0[0],p0[2]-p0[0]);n1=np.cross(p1[1]-p1[0],p1[2]-p1[0]);a0=float(np.linalg.norm(n0)*.5);a1=float(np.linalg.norm(n1)*.5)
                cos=float(np.dot(n0,n1)/(np.linalg.norm(n0)*np.linalg.norm(n1))) if np.linalg.norm(n0)>0 and np.linalg.norm(n1)>0 else -1.
                row={'face':fi,'parent_chart':int(parent[fi]),'area_ratio':a1/a0 if a0 else 0.,'normal_cosine':cos,'old_area_mm2':a0*1e6,'new_area_mm2':a1*1e6}
                shape.append(row)
                if a1<=1e-15 or cos<=0:world_reject=('degenerate_or_inverted_world_face',fi);break
            if world_reject:
                rec.update(status='REJECT',reason=world_reject[0],witness={'face':world_reject[1]},shape=shape);results.append(rec);continue
            # Apply proposed chart triangles and require each historical chart remain a non-overlapping, area-complete patch.
            chart_updates={};coverage=[];chart_bad=None
            for p in sorted({int(parent[fi]) for fi in changed}):
                oldrows=allchart[p];newrows=[fi for fi in oldrows if fi not in drop]
                coords={fi:newB.get(fi,B[fi]) for fi in newrows}
                areas={fi:orient2(coords[fi][:,1:3])/2 for fi in newrows}
                if any(x<=1e-12 for x in areas.values()):chart_bad=('chart_contains_nonpositive_face',p);break
                old_area=sum(orient2(B[fi][:,1:3])/2 for fi in oldrows);new_area=sum(areas.values())
                if abs(old_area-new_area)>1e-10:chart_bad=('historical_chart_area_not_preserved',p);break
                changed_here=[fi for fi in changed if int(parent[fi])==p]
                # Only changed faces can overlap previously disjoint unchanged faces or each other.
                for fi in changed_here:
                    poly=coords[fi][:,1:3]
                    for fj in newrows:
                        if fj==fi:continue
                        if poly_area(clip(poly,coords[fj][:,1:3]))>1e-10:
                            chart_bad=('overlapping_material_faces_after_collapse',p,fi,fj);break
                    if chart_bad:break
                if chart_bad:break
                chart_updates.update({fi:coords[fi] for fi in changed_here})
                coverage.append({'parent_chart':p,'face_count_before':len(oldrows),'face_count_after':len(newrows),'area_before':old_area,'area_after':new_area,'area_abs_error':abs(old_area-new_area),'changed_faces':changed_here})
            if chart_bad:
                rec.update(status='REJECT',reason=chart_bad[0],witness=list(chart_bad[1:]),shape=shape);results.append(rec);continue
            # Compute exact (validator-compatible float64 clipping) positive-area overlap parents for changed faces.
            parents_by_face={};identity_by_face={};overlap_bad=None
            for fi in changed:
                p=int(parent[fi]);tri=newB[fi][:,1:3];area_new=poly_area(tri);pr=[];sumarea=0.
                for oldfi in allchart[p]:
                    ar=poly_area(clip(B[oldfi][:,1:3],tri))
                    if ar>1e-10:pr.append(oldfi);sumarea+=ar
                if not pr or abs(sumarea-area_new)>1e-10:
                    overlap_bad=('positive_overlap_coverage_failure',fi,p,area_new,sumarea,pr);break
                sids=sorted({x for oldfi in pr for x in ids(csrc,oldfi)})
                dids=sorted({x for oldfi in pr for x in ids(cdet,oldfi)})
                uids=sorted({x for oldfi in pr for x in ids(csub,oldfi)})
                vals={key:sorted(set(map(int,a[key][pr]))) for key in scalar_keys}
                if not sids or not dids or not uids or any(len(vals[k])!=1 for k in scalar_keys) or int(a['face_kind'][fi])!=2:
                    overlap_bad=('crosses_or_loses_typed_ancestry_identity',fi,p,pr,{'source':sids,'detail':dids,'subface':uids,'scalar':vals});break
                parents_by_face[fi]=sorted(pr)
                identity_by_face[fi]={'source':sids,'detail':dids,'subface':uids,'scalar':{k:v[0] for k,v in vals.items()}}
            if overlap_bad:
                rec.update(status='REJECT',reason=overlap_bad[0],witness=list(overlap_bad[1:]),shape=shape);results.append(rec);continue
            # Full output face table: collapse v into u and delete the two edge faces.
            kept_old=np.asarray([fi for fi in range(M) if fi not in drop],np.int64)
            candidate_F=F[kept_old].copy();old_to_new_face=np.full(M,-1,np.int64);old_to_new_face[kept_old]=np.arange(len(kept_old))
            for fi in changed:
                candidate_F[int(old_to_new_face[fi])][candidate_F[int(old_to_new_face[fi])]==v]=u
            if any(len(set(map(int,t)))!=3 for t in candidate_F):
                rec.update(status='REJECT',reason='degenerate_output_face',shape=shape);results.append(rec);continue
            dupe={};duplicate=None
            for fr,t in enumerate(candidate_F):
                k=tuple(sorted(map(int,t)))
                if k in dupe:duplicate=(dupe[k],fr,k);break
                dupe[k]=fr
            if duplicate:
                rec.update(status='REJECT',reason='duplicate_output_face',witness=duplicate,shape=shape);results.append(rec);continue
            # Prove the 90-node patch boundary edge set remains exact after collapse.
            patch_old=set(map(int,topo['final_patch_face_rows']))
            patch_candidate=[]
            for fi in sorted(patch_old):
                if fi in drop:continue
                t=F[fi].copy();t[t==v]=u;patch_candidate.append(t)
            def boundary_edges(tris):
                cnt=defaultdict(int)
                for t in tris:
                    for j in range(3):cnt[tuple(sorted((int(t[j]),int(t[(j+1)%3]))))]+=1
                return {e for e,n in cnt.items() if n==1}
            orig_b=boundary_edges(F[sorted(patch_old)])
            new_b=boundary_edges(patch_candidate)
            if orig_b!=new_b:
                rec.update(status='REJECT',reason='fixed_patch_boundary_edges_changed',old_boundary_count=len(orig_b),new_boundary_count=len(new_b),removed_boundary_edges=sorted(orig_b-new_b)[:12],added_boundary_edges=sorted(new_b-orig_b)[:12],shape=shape);results.append(rec);continue
            # Material ancestry arrays are filled later by the selected-candidate emitter.
            rec.update(status='PASS_CHEAP',reason=None,collapse_faces_removed=sorted(drop),changed_faces=changed,changed_face_parents={str(k):v for k,v in parents_by_face.items()},shape=shape,shape_summary={'min_area_ratio':min(x['area_ratio'] for x in shape),'min_normal_cosine':min(x['normal_cosine'] for x in shape),'min_new_area_mm2':min(x['new_area_mm2'] for x in shape)},legacy_chart_coverage=coverage,parents_by_face=parents_by_face,identity_by_face=identity_by_face,drop_faces=sorted(drop),new_face_table=candidate_F.tolist(),kept_old_face_rows=kept_old.tolist(),new_bary_by_old_face={str(k):v.tolist() for k,v in newB.items()})
            results.append(rec)
    passes=[r for r in results if r['status']=='PASS_CHEAP']
    # Preserve whole-batch rejection reasons; choose the shortest clear, chart-complete legal collapse.
    passes.sort(key=lambda r:(r['edge_length_mm'],-r['shape_summary']['min_area_ratio'],-r['shape_summary']['min_normal_cosine'],r['inside_node'],r['survivor']))
    if not passes:
        compact=[]
        for r in results:compact.append({k:v for k,v in r.items() if k not in ('new_face_table','kept_old_face_rows','new_bary_by_old_face','parents_by_face','identity_by_face')})
        fail={'schema':'numi.human.b43-16.edge-collapse-batch-screen.v1','status':'NO_CANDIDATE_PASSED_CHEAP_SCREENS','baseline_sha256':pins['base'],'edge_candidate_count':len(results),'edge_candidate_status_counts':{s:sum(x['status']==s for x in results) for s in sorted({x['status'] for x in results})},'failure_reason_counts':{s:sum(x.get('reason')==s for x in results) for s in sorted({x.get('reason') for x in results})},'candidates':compact}
        (D/'batch-screen-failure.json').write_text(json.dumps(fail,indent=2,sort_keys=True,default=lambda x:int(x) if isinstance(x,np.integer) else float(x) if isinstance(x,np.floating) else str(x))+'\n')
        print(json.dumps({'status':fail['status'],'edge_candidate_count':len(results),'status_counts':fail['edge_candidate_status_counts'],'reason_counts':fail['failure_reason_counts'],'report_path':str(D/'batch-screen-failure.json')},indent=2,sort_keys=True))
        return
    chosen=passes[0]
    # Rebuild emitted archive with one removed node and two removed faces.
    v=int(chosen['inside_node']);u=int(chosen['survivor']);drop=set(chosen['drop_faces']);changed=set(chosen['changed_faces']);keepf=np.asarray(chosen['kept_old_face_rows'],np.int64)
    Fnew=np.asarray(chosen['new_face_table'],np.int64); Bnew=B[keepf].copy()
    for oldfi in changed:Bnew[int(np.searchsorted(keepf,oldfi))]=np.asarray(chosen['new_bary_by_old_face'][str(oldfi)],np.float64)
    old_to_new_node=np.full(N,-1,np.int64);new_to_old_node=np.asarray([i for i in range(N) if i!=v],np.int64);old_to_new_node[new_to_old_node]=np.arange(N-1,dtype=np.int64)
    Fnew=old_to_new_node[Fnew]
    if np.any(Fnew<0):raise ValueError('output face still references removed vertex')
    # Filter all simple node/face row-aligned arrays; rebuild the relevant CSR arrays explicitly below.
    node_ptr_keys={'refinement_node_parent_face_indptr','source_prolongation_indptr','immediate_refinement_prolongation_indptr'}
    face_ptr_keys={'face_source_ancestry_indptr','face_fine_detail_ancestry_indptr','face_fine_subface_ancestry_indptr','refinement_face_parent_ancestry_indptr'}
    face_data_keys={'face_source_ancestry_face_rows','face_fine_detail_ancestry_rows','face_fine_subface_ancestry_rows','refinement_face_parent_ancestry_face_rows'}
    current={}
    for k,x in a.items():
        if k in node_ptr_keys|face_ptr_keys|face_data_keys:continue
        if x.ndim>=1 and x.shape[0]==N:
            current[k]=x[new_to_old_node].copy()
        elif x.ndim>=1 and x.shape[0]==M:
            current[k]=x[keepf].copy()
        else:
            current[k]=x.copy()
    # Current geometry/topology, legacy affine ancestry, and audit reference topology.
    current['points_world_m']=a['points_world_m'][new_to_old_node].copy()
    current['source_O_faces']=Fnew
    current['area_reference_037_source_O_faces']=Fnew.copy()
    current['refinement_face_parent_barycentric']=Bnew
    # Update current-node indices in source-boundary path records; the consumed midpoint was excluded above.
    for k in ('coarse_boundary_edge_path_output_nodes','coarse_boundary_anchor_rows','typed_coarse_anchor_output_rows_018'):
        if k in current:
            old=np.asarray(current[k],np.int64);mapped=old_to_new_node[old]
            if np.any(mapped<0):raise ValueError('removed node referenced by '+k)
            current[k]=mapped.astype(a[k].dtype,copy=False)
    # Rebuild node CSR rows, preserving each retained node's stored source/prolongation values exactly.
    def filter_node_csr(ptr_key,val_keys):
        ptr=a[ptr_key];vals=[a[k] for k in val_keys];outptr=[0];out=[[] for _ in vals]
        for old in new_to_old_node:
            lo,hi=map(int,ptr[old:old+2])
            for j,x in enumerate(vals):out[j].extend(x[lo:hi].tolist())
            outptr.append(len(out[0]))
        current[ptr_key]=np.asarray(outptr,dtype=ptr.dtype)
        for k,x in zip(val_keys,vals):current[k]=np.asarray(out[val_keys.index(k)],dtype=x.dtype)
    filter_node_csr('refinement_node_parent_face_indptr',['refinement_node_parent_face_rows','refinement_node_parent_face_barycentric'])
    filter_node_csr('source_prolongation_indptr',['source_prolongation_node_rows','source_prolongation_weights'])
    filter_node_csr('immediate_refinement_prolongation_indptr',['immediate_refinement_prolongation_node_rows','immediate_refinement_prolongation_weights'])
    # Rebuild per-face ancestry. Unchanged faces retain their exact old rows; changed faces use the full positive-overlap union.
    origin_to_candidate={int(old):j for j,old in enumerate(keepf)}
    def rebuild_face_csr(ptr_key,val_key,kind):
        ptr=a[ptr_key];val=a[val_key];rows=[];newptr=[0]
        for old in keepf:
            old=int(old)
            if old in changed:
                ps=chosen['parents_by_face'][old]
                if kind=='parent':q=[int(parent[old])]
                else:q=list(chosen['identity_by_face'][old][kind])
            else:q=list(map(int,csr_rows(ptr,val,old)))
            if kind=='parent' and not q:raise ValueError('empty legacy parent ancestry')
            rows.extend(q);newptr.append(len(rows))
        current[ptr_key]=np.asarray(newptr,dtype=ptr.dtype);current[val_key]=np.asarray(rows,dtype=val.dtype)
    rebuild_face_csr('face_source_ancestry_indptr','face_source_ancestry_face_rows','source')
    rebuild_face_csr('face_fine_detail_ancestry_indptr','face_fine_detail_ancestry_rows','detail')
    rebuild_face_csr('face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows','subface')
    rebuild_face_csr('refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows','parent')
    # All retained and changed faces stay in their declared historical parent chart; refresh changed scalar IDs from overlap ancestry.
    for oldfi in changed:
        nf=origin_to_candidate[int(oldfi)];ident=chosen['identity_by_face'][int(oldfi)]
        for k,val in ident['scalar'].items():current[k][nf]=val
    # Overlap CSR into the exact pinned 806 face domain.
    fptr=[0];frows=[]
    for old in keepf:
        old=int(old)
        q=chosen['parents_by_face'][old] if old in changed else [old]
        frows.extend(q);fptr.append(len(frows))
    current['remesh806_baseline_sha256']=np.asarray([pins['base']])
    current['remesh806_old_to_new_node_rows']=old_to_new_node
    current['remesh806_new_to_old_node_rows']=new_to_old_node
    current['remesh806_face_parent_indptr']=np.asarray(fptr,np.int64)
    current['remesh806_face_parent_face_rows']=np.asarray(frows,np.int64)
    # Mark deleted inside witness as explicitly unresolved; do not call it cleared by removal.
    current['remesh806_removed_inside_node_rows']=np.asarray([v],np.int64)
    current['remesh806_removed_inside_node_resolution']=np.asarray(['UNRESOLVED_PHYSICAL_CONTAINMENT'],dtype='<U40')
    out=D/'candidate.npz';report=D/'report.json'
    if out.exists() or report.exists():raise FileExistsError('private output already exists')
    np.savez_compressed(out,**current)
    outsha=sha(out)
    # Compact batch report drops large per-row materialization fields from unrelated rejected candidates.
    compact=[]
    for r in results:
        rr={k:v for k,v in r.items() if k not in ('new_face_table','kept_old_face_rows','new_bary_by_old_face','parents_by_face','identity_by_face')}
        compact.append(rr)
    final={'schema':'numi.human.b43-16.edge-collapse-remesh.v1','status':'PRIVATE_CANDIDATE_EMITTED_FOR_INDEPENDENT_CHECKS','baseline_sha256':pins['base'],'baseline_path':str(BASE),'topology_sha256':pins['topology'],'boundary_sha256':pins['boundary'],'audit130_sha256':sha(AUD130),'target_bank':[43,16],'target_inside_node_count':len(target_inside),'all_bank_inside_node_count':len(all_inside),'fixed_patch_boundary_nodes':len(boundary_nodes),'source_or_anchor_protected_node_count':len(protected_nodes),'edge_candidate_count':len(results),'edge_candidate_status_counts':{s:sum(x['status']==s for x in results) for s in sorted({x['status'] for x in results})},'selected_edge':{k:chosen[k] for k in ('inside_node','survivor','edge','edge_length_mm','collapse_faces_removed','changed_faces','shape_summary','legacy_chart_coverage')},'candidate_path':str(out),'candidate_sha256':outsha,'candidate_nodes':len(current['points_world_m']),'candidate_faces':len(current['source_O_faces']),'candidate_node_map_keys':['remesh806_old_to_new_node_rows','remesh806_new_to_old_node_rows'],'candidate_face_ancestry_keys':['remesh806_face_parent_indptr','remesh806_face_parent_face_rows'],'legacy_face_ancestry_domain':'preserved existing 41,745-face immediate parent chart; refinement_input_candidate_sha256 unchanged','removed_inside_node':{'row_in_806':v,'node_kind':int(a['node_kind'][v]),'refinement_node_kind':int(a['refinement_node_kind'][v]),'resolution':'UNRESOLVED_PHYSICAL_CONTAINMENT'},'candidate_checks':{'edge_link_condition':True,'patch_boundary_fixed':True,'source_boundary_and_anchors_protected':True,'legacy_material_chart_coverage_preserved':True,'legacy_chart_barycentric_refresh':True,'new_faces_nonzero_noninverted_world_area':True,'no_duplicate_faces':True,'source_detail_subface_ids_homogeneous_for_each_changed_face':True,'full_exact_bone_and_self_audit':'PENDING_PARENT_AUDIT','layer_transfer_and_positive_prism_checks':'PENDING_SIBLING_TRANSFER'},'batch_candidates':compact,'physical_admission':False,'accepted_physical_steps':0}
    report.write_text(json.dumps(final,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':final['status'],'candidate_count':len(results),'status_counts':final['edge_candidate_status_counts'],'selected_edge':final['selected_edge'],'candidate_path':str(out),'candidate_sha256':outsha,'report_path':str(report),'report_sha256':sha(report)},indent=2,sort_keys=True))
if __name__=='__main__':main()
