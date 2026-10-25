#!/Users/n/numi-human-prep-venv-20261005/bin/python
"""Read-only BFS orientation audit and curved fine-normal/source-depth proposals for batch 145."""
import csv,json,hashlib,importlib.util,sys,math
from pathlib import Path
from collections import defaultdict,deque,Counter
import numpy as np
from scipy.sparse import csr_matrix

W=Path('/Users/numi/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=W/'digit-source-diagnosis-110'
RUN=D/'source_path_batch_110_v2.py'
REPORT=D/'batch-145-curved-source-midpoints-004.json'
DETAIL=E/'bilateral-hand-functional-rebind-020/run-001/candidate.nhskin'
ROWS=E/'accepted-transfer-row-export-075/run-001/source-preflight/accepted-transfer-rows.csv'
COMPACT=W/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
TRANSFER=W/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002/Wnew-detail-to-084-nodes.npz'
OUT=D/'source-path-orientation-audit-112.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def directed_edge_sign(a,b):
    return 1 if a<b else -1
def orient_faces_by_edge_adjacency(faces):
    F=np.asarray(faces,dtype=np.int64)
    edge_to_faces=defaultdict(list)
    for fi,(a,b,c) in enumerate(F):
        for x,y in ((a,b),(b,c),(c,a)):
            edge_to_faces[tuple(sorted((int(x),int(y))))].append((fi,directed_edge_sign(int(x),int(y))))
    adj=[[] for _ in range(len(F))]
    boundary=0;nonmanifold=0
    for e,inc in edge_to_faces.items():
        if len(inc)==1:boundary+=1
        elif len(inc)>2:nonmanifold+=1
        if len(inc)==2:
            (a,sa),(b,sb)=inc
            # Flipped directed edges must be opposite: flip_b = -flip_a*sa*sb.
            factor=-sa*sb
            adj[a].append((b,factor,e));adj[b].append((a,factor,e))
    flips=np.zeros(len(F),dtype=np.int8);components=np.full(len(F),-1,dtype=np.int64);conflicts=[]
    comp=0
    for seed in range(len(F)):
        if flips[seed]:continue
        flips[seed]=1;components[seed]=comp;q=deque([seed])
        while q:
            u=q.popleft()
            for v,factor,e in adj[u]:
                want=int(flips[u])*factor
                if flips[v]==0:flips[v]=want;components[v]=comp;q.append(v)
                elif int(flips[v])!=want:conflicts.append({'face_rows':[u,v],'edge_rows':list(e)})
        comp+=1
    return flips,components,{'face_count':int(len(F)),'edge_count':len(edge_to_faces),'boundary_edges':boundary,'nonmanifold_edges':nonmanifold,'adjacency_winding_conflict_count':len(conflicts),'winding_conflicts':conflicts[:100],'component_count':comp}

def area_normals(points,faces,flips):
    P=np.asarray(points,dtype=np.float64);F=np.asarray(faces,dtype=np.int64)
    cross=np.cross(P[F[:,1]]-P[F[:,0]],P[F[:,2]]-P[F[:,0]])
    cross*=np.asarray(flips,dtype=np.float64)[:,None]
    vn=np.zeros_like(P)
    for k in range(3):np.add.at(vn,F[:,k],cross)
    norm=np.linalg.norm(vn,axis=1);good=norm>1e-30;vn[good]/=norm[good,None]
    fn=cross.copy();fnorm=np.linalg.norm(fn,axis=1);fgood=fnorm>1e-30;fn[fgood]/=fnorm[fgood,None]
    return vn,fn,cross

def parse_skin(path):
    raw=path.read_bytes()
    magic,abi,nb,nv,ni,_=__import__('struct').unpack_from('<8s5I',raw)
    vo=60+nb*36;io=vo+nv*56
    F=np.frombuffer(raw,dtype='<u4',count=ni,offset=io).reshape(-1,3).astype(np.int64,copy=True)
    return F,{'abi':int(abi),'binding_count':int(nb),'vertex_count':int(nv),'face_count':int(ni//3)}

def outward_sign_from_inward_dots(inward_dots, minimum_abs_dot=1e-9):
    """Choose BFS normal sign so O-to-I offsets point opposite the outward normal."""
    vals=np.asarray([float(x) for x in inward_dots if np.isfinite(x) and abs(float(x))>=minimum_abs_dot],dtype=np.float64)
    if not len(vals): return None, {'usable':0,'median_dot':None,'positive':0,'negative':0}
    med=float(np.median(vals))
    if abs(med)<minimum_abs_dot:return None,{'usable':int(len(vals)),'median_dot':med,'positive':int(np.sum(vals>0)),'negative':int(np.sum(vals<0))}
    sign=-1 if med>0 else 1
    return sign,{'usable':int(len(vals)),'median_dot':med,'positive':int(np.sum(vals>0)),'negative':int(np.sum(vals<0)),
       'chosen_outward_sign':sign,'agreement_fraction':float(np.mean((vals*sign)<0))}

def normal_scalar_layer_points(o_world, outward_normal, skin_depth_m, fat_depth_m):
    """Place existing scalar O-I and I-D depths along a validated local outward normal."""
    o=np.asarray(o_world,dtype=np.float64);n=np.asarray(outward_normal,dtype=np.float64)
    if o.shape!=(3,) or n.shape!=(3,) or not np.isfinite(o).all() or not np.isfinite(n).all():raise ValueError('nonfinite point/normal')
    nn=float(np.linalg.norm(n))
    if nn<=0:raise ValueError('zero normal')
    n=n/nn;skin=float(skin_depth_m);fat=float(fat_depth_m)
    if not np.isfinite(skin) or not np.isfinite(fat) or skin<=0 or fat<=0:raise ValueError('source layer depths must be positive')
    i=o-skin*n;d=i-fat*n
    return {'O_m':o.tolist(),'I_m':i.tolist(),'D_m':d.tolist(),'skin_depth_m':skin,'fat_depth_m':fat,'outward_normal':n.tolist()}

def main():
    runmodspec=importlib.util.spec_from_file_location('batch110',RUN);mod=importlib.util.module_from_spec(runmodspec);runmodspec.loader.exec_module(mod)
    report=json.load(open(REPORT));detail_faces,header=parse_skin(DETAIL)
    rows=list(csv.DictReader(ROWS.open(newline='')))
    detail_xyz=np.asarray([[float(r[f'detail_rest_world_{a}_m']) for a in 'xyz'] for r in rows],dtype=np.float64)
    active=np.asarray([int(r['active']) for r in rows],bool)
    selected=np.asarray([int(r['selected_face_row']) for r in rows],np.int64)
    with np.load(COMPACT,allow_pickle=False) as z:
        R=np.asarray(z['reference_points_m'],dtype=np.float64);C=np.asarray(z['current037_points_m_f32'])
        Fco=np.asarray(z['source_O_faces'],dtype=np.int64)
    with np.load(TRANSFER,allow_pickle=False) as z:
        Wmat=csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(int(x) for x in z['shape']))
    dcur=detail_xyz+Wmat.dot(C.astype(np.float64)-R)
    fflip,fcomp,fstats=orient_faces_by_edge_adjacency(detail_faces)
    cflip,ccomp,cstats=orient_faces_by_edge_adjacency(Fco)
    fvn_r,ffn_r,farea_r=area_normals(detail_xyz,detail_faces,fflip)
    fvn_c,ffn_c,farea_c=area_normals(dcur,detail_faces,fflip)
    cvn_r,_,_=area_normals(R,Fco,cflip)
    cvn_c,_,_=area_normals(C.astype(float),Fco,cflip)
    rowfacecomp=np.full(len(rows),-1,dtype=np.int64)
    for fi,tri in enumerate(detail_faces):
        for v in tri:
            if rowfacecomp[v]<0:rowfacecomp[v]=fcomp[fi]
            elif rowfacecomp[v]!=fcomp[fi]:rowfacecomp[v]=-2
    # Unique W one-hot anchors establish componentwise absolute outward parity.
    with np.load(TRANSFER,allow_pickle=False) as z:
        W=csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(int(x) for x in z['shape']))
    anchors=defaultdict(list)
    for dr in range(W.shape[0]):
        lo,hi=W.indptr[dr:dr+2]
        if hi-lo==1 and abs(float(W.data[lo])-1.0)<=1e-7:
            anchors[int(W.indices[lo])].append(dr)
    parity_rows=defaultdict(list)
    for coarse,ds in anchors.items():
        if len(ds)!=1:continue
        dr=ds[0];comp=int(rowfacecomp[dr])
        if comp<0 or coarse>=len(R):continue
        cr=float(np.dot(fvn_r[dr],cvn_r[coarse]))
        cc=float(np.dot(fvn_c[dr],cvn_c[coarse]))
        if math.isfinite(cr) and abs(cr)>=.25:
            parity_rows[comp].append({'coarse_output_row':coarse,'detail_row':dr,'reference_cosine':cr,'current037_cosine':cc})
    parity_by_comp={};parity_summary={}
    for comp,vals in parity_rows.items():
        signs=[1 if x['reference_cosine']>0 else -1 for x in vals]
        c=Counter(signs);chosen=c.most_common(1)[0][0]
        agree=sum(x==chosen for x in signs);fraction=agree/len(signs)
        parity_by_comp[int(comp)]=chosen
        parity_summary[str(comp)]={'anchor_count':len(vals),'sign_counts':{str(k):v for k,v in c.items()},
          'chosen_outward_parity':chosen,'agreement_fraction':fraction,
          'reference_cosine_quantiles':np.quantile([x['reference_cosine'] for x in vals],[0,.1,.5,.9,1]).tolist(),
          'current037_cosine_quantiles':np.quantile([x['current037_cosine'] for x in vals],[0,.1,.5,.9,1]).tolist(),
          'anchor_witnesses':vals[:20]}
    # Absolute outward sign is resolved independently from coarse winding: the source
    # O->I direction is known to point inward. Use only unique direct anchors on the
    # same orientable component and retain coarse-vs-fine winding parity separately.
    inward_dot_rows=defaultdict(list)
    for coarse,ds in anchors.items():
        if len(ds)!=1 or coarse+2411>=len(R):continue
        dr=ds[0];comp=int(rowfacecomp[dr])
        if comp<0:continue
        v=R[coarse+2411]-R[coarse]
        dot=float(np.dot(fvn_r[dr],v))
        if math.isfinite(dot) and abs(dot)>=1e-9:
            inward_dot_rows[comp].append({'coarse_output_row':coarse,'detail_row':dr,'raw_fine_normal_dot_I_minus_O_m':dot})
    outward_sign_by_comp={};layer_order_summary={}
    for comp,vals in inward_dot_rows.items():
        sign,stats=outward_sign_from_inward_dots([x['raw_fine_normal_dot_I_minus_O_m'] for x in vals])
        if sign is not None:outward_sign_by_comp[int(comp)]=sign
        layer_order_summary[str(comp)]={**stats,'anchor_count':len(vals),'fine_to_coarse_winding_parity':parity_by_comp.get(int(comp)),
          'anchor_witnesses':vals[:20]}
    edgefaces=defaultdict(list)
    for fi,(a,b,c) in enumerate(detail_faces):
        for x,y in ((a,b),(b,c),(c,a)):edgefaces[tuple(sorted((int(x),int(y))))].append(fi)
    rowsout=[];path_comp_issues=Counter()
    for e in report['edges']:
        path=list(map(int,e.get('path_detail_rows',[])))
        if len(path)<2:
            rowsout.append({'edge_output_rows':e['edge_output_rows'],'status':'no_path'});continue
        comps=[int(rowfacecomp[x]) for x in path]
        comp=next((x for x in comps if x>=0),-1)
        same_component=comp>=0 and all(x==comp for x in comps)
        if not same_component:path_comp_issues['path_component_mismatch']+=1
        parity=parity_by_comp.get(comp)
        outward_sign=outward_sign_by_comp.get(comp)
        k=int(e.get('midpoint_detail_edge_index',0));alpha=float(e.get('midpoint_edge_fraction',.5))
        a,b=path[k:k+2];inc=edgefaces.get(tuple(sorted((a,b))),[])
        nvec=np.zeros(3);nvec_c=np.zeros(3)
        for fi in inc:
            nvec+=farea_r[fi];nvec_c+=farea_c[fi]
        nr=np.linalg.norm(nvec);nc=np.linalg.norm(nvec_c)
        nout_r=outward_sign*nvec/nr if outward_sign is not None and nr>0 else None
        nout_c=outward_sign*nvec_c/nc if outward_sign is not None and nc>0 else None
        # Scalar source depths projected at original coarse endpoints onto their exact direct fine-anchor normal.
        u,v=map(int,e['edge_output_rows'])
        skin_ref=[];fat_ref=[];skin_cur=[];fat_cur=[];endpoint_detail_rows=[]
        chosen_endpoint_rows=e.get('endpoint_detail_rows',[None,None])
        for endpoint_index,coarse in enumerate((u,v)):
            dr_raw=chosen_endpoint_rows[endpoint_index] if endpoint_index<len(chosen_endpoint_rows) else None
            if dr_raw is None: endpoint_detail_rows.append(None);skin_ref.append(None);fat_ref.append(None);skin_cur.append(None);fat_cur.append(None);continue
            dr=int(dr_raw);endpoint_detail_rows.append(dr);cp=int(rowfacecomp[dr]);ep=outward_sign_by_comp.get(cp)
            en=ep*fvn_r[dr] if ep is not None else None
            enc=ep*fvn_c[dr] if ep is not None else None
            if en is None or enc is None:
                skin_ref.append(None);fat_ref.append(None);skin_cur.append(None);fat_cur.append(None);continue
            skin_ref.append(float(-np.dot(R[coarse+2411]-R[coarse],en)))
            fat_ref.append(float(-np.dot(R[coarse+4822]-R[coarse+2411],en)))
            skin_cur.append(float(-np.dot(C[coarse+2411].astype(float)-C[coarse].astype(float),enc)))
            fat_cur.append(float(-np.dot(C[coarse+4822].astype(float)-C[coarse+2411].astype(float),enc)))
        mid_ref=e.get('sample_point_world_m')
        base={'edge_output_rows':e['edge_output_rows'],'source_kind':e.get('source_kind'),
          'previous_compatibility_status':e['status'],'path_detail_rows':path,'path_component_ids':comps,
          'detail_component':comp,'fine_to_coarse_winding_parity_from_direct_onehot_anchors':parity,'global_outward_sign_from_O_to_I_layer_order':outward_sign,
          'same_orientable_component':same_component,'midpoint_detail_edge_rows':[a,b],
          'midpoint_edge_alpha':alpha,'registered_source_midpoint_m':mid_ref,
          'fine_incident_face_rows':inc,'orientation_faces_consistent':(fstats['adjacency_winding_conflict_count']==0),
          'reference_outward_normal':None if nout_r is None else nout_r.tolist(),
          'current037_outward_normal':None if nout_c is None else nout_c.tolist(),
          'source_skin_depth_reference_endpoint_m':skin_ref,'source_fat_depth_reference_endpoint_m':fat_ref,
          'source_skin_depth_current037_endpoint_m':skin_cur,'source_fat_depth_current037_endpoint_m':fat_cur,
          'scalar_depth_validity':'positive finite depths on both endpoints' if all(x is not None and x>0 for x in skin_ref+fat_ref+skin_cur+fat_cur) else 'unresolved/nonpositive at one or more endpoints'}
        if all(nout_r is not None for _ in [0]) and all(x is not None for x in skin_ref+fat_ref):
            t=.5;ts=(1-t)*skin_ref[0]+t*skin_ref[1];tf=(1-t)*fat_ref[0]+t*fat_ref[1]
            base['normal_scalar_source_proposal_reference']={**normal_scalar_layer_points(mid_ref,nout_r,ts,tf),
              'depths_are_interpolated_existing_scalar_fields':True}
        if all(nout_c is not None for _ in [0]) and all(x is not None for x in skin_cur+fat_cur) and e.get('midpoint_current037_world_m') is not None:
            t=.5;ts=(1-t)*skin_cur[0]+t*skin_cur[1];tf=(1-t)*fat_cur[0]+t*fat_cur[1];om=np.asarray(e['midpoint_current037_world_m'])
            base['normal_scalar_source_proposal_current037']={**normal_scalar_layer_points(om,nout_c,ts,tf),
              'depths_are_interpolated_existing_scalar_fields':True}
        rowsout.append(base)
    out={'status':'READ_ONLY_NHSKIN_ORIENTABILITY_AND_SCALAR_DEPTH_DIAGNOSTIC',
      'scope':'Audits actual NHSKIN face adjacency winding by BFS and anchors each connected component absolute sign from unique direct W one-hot detail/coarse O normal correspondences. Adds optional local fine-normal and existing scalar layer-depth proposals for all batch edges. No geometry or payload is changed; no bone/collision/volume gate is run.',
      'input_pins':{'batch_report_sha256':sha(REPORT),'batch_runner_sha256':sha(RUN),'detail_sha256':sha(DETAIL),'accepted_rows_sha256':sha(ROWS),'compact_sha256':sha(COMPACT),'W_sha256':sha(TRANSFER)},
      'detail_topology':fstats,'coarse_O_topology':cstats,'component_fine_to_coarse_winding_anchors':parity_summary,'component_outward_sign_from_layer_order':layer_order_summary,
      'path_edge_classification_counts':dict(Counter(x['scalar_depth_validity'] for x in rowsout)),
      'counts':{'batch_edges':len(rowsout),'same_component_paths':sum(x.get('same_orientable_component',False) for x in rowsout),
        'reference_outward_normals':sum(x.get('reference_outward_normal') is not None for x in rowsout),
        'current_outward_normals':sum(x.get('current037_outward_normal') is not None for x in rowsout),
        'normal_scalar_reference_proposals':sum('normal_scalar_source_proposal_reference' in x for x in rowsout),
        'normal_scalar_current_proposals':sum('normal_scalar_source_proposal_current037' in x for x in rowsout),
        'unresolved_previous_parity_edges':sum(x.get('previous_compatibility_status')=='candidate_source_path_orientation_unresolved' for x in rowsout)},
      'edges':rowsout,
      'limitations':['BFS proves topological orientability/winding consistency, not anatomical outwardness. Absolute outward sign here is inferred from the declared O-to-I layer direction at unique direct anchors; the independent fine-to-coarse face-winding parity is reported separately. Components without coherent layer-order anchors remain unresolved.','Scalar I/D construction is a source-field counterproposal only; every point must still pass exact complete-bank and downstream volume/material/self checks.','No fixed thickness is introduced: scalar depths are projected from existing endpoint layer offsets.'],
      'status_note':'A consistent reversed raw fine/coarse winding is not an inversion; local orientation continuity and global anchor parity are reported separately.'}
    OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'out':str(OUT),'detail_topology':fstats,'coarse_topology':cstats,'counts':out['counts'],'component_parity':{k:{q:v for q,v in val.items() if q!='anchor_witnesses'} for k,val in parity_summary.items()}},indent=2))
if __name__=='__main__':main()
