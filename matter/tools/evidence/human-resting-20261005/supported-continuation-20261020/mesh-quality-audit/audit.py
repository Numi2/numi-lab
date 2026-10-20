#!/usr/bin/env python3
"""Audit all six triangular-prism staircase orders on frozen whole-body skin/fat geometry."""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, math, re
from pathlib import Path
import numpy as np

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def read_block(text: str, name: str) -> str:
    m=re.search(re.escape(name)+r'\s*=\s*\{',text)
    if not m: raise ValueError(f'missing C++ array {name}')
    ends=[p for p in (text.find('\n};',m.end()),text.find('\n}};',m.end())) if p>=0]
    if not ends: raise ValueError(f'unclosed C++ array {name}')
    return text[m.end():min(ends)]

def int_array(text: str, name: str) -> np.ndarray:
    return np.fromiter((int(x) for x in re.findall(r'\d+u?',read_block(text,name))),dtype=np.int64)

def triples(text: str, name: str) -> np.ndarray:
    block=read_block(text,name)
    return np.asarray([[float(x) for x in row.split(',')] for row in re.findall(r'std::array<double,3>\{([^{}]+)\}',block)],dtype=np.float64)

def parity(p: tuple[int,int,int]) -> int:
    return sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))%2

def prisms(outer: np.ndarray, inner: np.ndarray, order: np.ndarray) -> np.ndarray:
    a=outer[order[:,:,0]]; b=outer[order[:,:,1]]; c=outer[order[:,:,2]]
    ia=inner[order[:,:,0]]; ib=inner[order[:,:,1]]; ic=inner[order[:,:,2]]
    return np.stack((np.stack((a,b,c,ic),axis=2),np.stack((a,b,ib,ic),axis=2),np.stack((a,ia,ib,ic),axis=2)),axis=2)

def metrics(tets: np.ndarray) -> dict[str,np.ndarray]:
    p0=tets[:,:,:,0,:]
    m=np.stack((tets[:,:,:,1,:]-p0,tets[:,:,:,2,:]-p0,tets[:,:,:,3,:]-p0),axis=-1)
    sv=np.linalg.svd(m,compute_uv=False); det=np.linalg.det(m); vol=np.abs(det)/6.0
    edge2=np.zeros_like(vol); lmax=np.zeros_like(vol)
    for i,j in itertools.combinations(range(4),2):
        d=tets[:,:,:,i,:]-tets[:,:,:,j,:]; d2=np.sum(d*d,axis=-1)
        edge2+=d2; lmax=np.maximum(lmax,np.sqrt(d2))
    mr=12.0*np.power(3.0*vol,2.0/3.0)/edge2
    return {'cond':sv[...,0]/sv[...,-1],'mr':mr,'sigma_min':sv[...,-1],'det':det,'eta':np.abs(det)/(lmax**3)}

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--face',type=int,default=3947)
    ap.add_argument('--header',type=Path)
    ap.add_argument('--header-source-label')
    ap.add_argument('--normalized-det-floor',type=float,default=1e-5)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args(); root=args.root.resolve()
    header=(args.header or root/'apps/NumiHumanWholeBodySkinReduction.hpp').resolve()
    prep=root/'apps/NumiHumanCoarseTissuePreparation.hpp'
    candidate_path=root/'tools/evidence/human-wholebody-skinfat-20261019/c296/candidate-024-safe.txt'
    coords_path=root/'tools/evidence/human-wholebody-skinfat-20261019/c296/admitted-layer-coordinates.csv'
    text=header.read_text()
    faces=np.asarray([tuple(map(int,x)) for x in re.findall(r'Face\{\s*(\d+)u,\s*(\d+)u,\s*(\d+)u\s*\}',read_block(text,'kWholeBodySkinFaces'))],dtype=np.int64)
    source_vertices=int_array(text,'kWholeBodySkinSourceVertices')
    source_faces=int_array(text,'kWholeBodySkinSourceFaces')
    frame_block=read_block(text,'kWholeBodySkinDonorFrames')
    reg=np.asarray([[float(v.strip().rstrip('f')) for v in m.group(3).split(',')] for m in re.finditer(r'WholeBodySkinDonorFrameV1\{\s*\d+u,\s*\{([^{}]+)\},\s*\{([^{}]+)\},\s*\{([^{}]+)\},\s*\{([^{}]+)\}\s*\}',frame_block)],dtype=np.float64)
    delta=triples(text,'kWholeBodySkinMechanicalWorldDeltas')
    interface=triples(text,'kWholeBodySkinInnerWorld'); deep=triples(text,'kWholeBodyFatDeepWorld'); outer=reg+delta
    if faces.shape!=(4619,3) or source_vertices.shape!=(2332,) or source_faces.shape!=(4619,) or not (outer.shape==interface.shape==deep.shape==(2332,3)):
        raise ValueError(f'unexpected source dimensions: faces {faces.shape}; maps {source_vertices.shape}/{source_faces.shape}; coords {outer.shape}/{interface.shape}/{deep.shape}')
    with coords_path.open(newline='') as f: rows=list(csv.DictReader(f))
    csv_points={k:np.asarray([[float(r[f'{prefix}_{a}']) for a in 'xyz'] for r in rows]) for k,prefix in [('outer','outer'),('interface','skin'),('deep','fat')]}
    coordinate_diffs={k:float(np.max(np.abs(p-csv_points[k]))) for k,p in [('outer',outer),('interface',interface),('deep',deep)]}
    lines=candidate_path.read_text().splitlines(); declared=list(map(int,lines[0].split())); source_map=np.asarray(list(map(int,lines[1].split())),dtype=np.int64)
    candidate=np.asarray([list(map(int,l.split())) for l in lines[2:2+declared[1]]],dtype=np.int64)
    if candidate.shape!=faces.shape: raise ValueError(f'candidate faces {candidate.shape}, expected {faces.shape}')
    source_map_equal=bool(np.array_equal(source_map,source_vertices))
    base_by_set={tuple(sorted(f)):i for i,f in enumerate(faces)}
    missing=sum(tuple(sorted(c)) not in base_by_set for c in candidate)
    order_changed=sum(tuple(c)!=tuple(faces[base_by_set[tuple(sorted(c))]]) for c in candidate if tuple(sorted(c)) in base_by_set)
    odd_flips=sum(parity(tuple(faces[base_by_set[tuple(sorted(c))]].tolist().index(v) for v in c)) for c in candidate if tuple(sorted(c)) in base_by_set)
    perms=list(itertools.permutations(range(3))); winding=np.asarray([parity(p)==0 for p in perms])
    order=np.stack([faces[:,p] for p in perms],axis=1)
    skin=metrics(prisms(outer,interface,order)); fat=metrics(prisms(interface,deep,order))
    both_cond=np.concatenate((skin['cond'],fat['cond']),axis=2)
    both_mr=np.concatenate((skin['mr'],fat['mr']),axis=2)
    both_sigma=np.concatenate((skin['sigma_min'],fat['sigma_min']),axis=2)
    per_order_cond=np.max(both_cond,axis=2); per_order_mr=np.min(both_mr,axis=2); per_order_sigma=np.min(both_sigma,axis=2)
    face_index=np.arange(len(faces)); sorted_choice=np.asarray([perms.index(tuple(np.argsort(f))) for f in faces])
    baseline=per_order_cond[face_index,sorted_choice]
    best_any=np.min(per_order_cond,axis=1)
    best_valid=np.min(np.where(winding[None,:],per_order_cond,np.inf),axis=1)
    def per_tet_counts(m: dict[str,np.ndarray],choice: np.ndarray) -> dict[str,int|float]:
        c=m['cond'][face_index[:,None],choice[:,None],:]
        return {'above_1e3':int((c>1e3).sum()),'above_1e4':int((c>1e4).sum()),'above_1e5':int((c>1e5).sum()),'max_condition':float(c.max())}
    fi=args.face
    if not 0<=fi<len(faces): raise ValueError(f'face index out of range: {fi}')
    exact_source=list(map(int,faces[fi])); mapped_source=[int(source_vertices[v]) for v in exact_source]
    witness=[]
    for pi,p in enumerate(perms):
        witness.append({'permutation':list(p),'local_vertex_order':list(map(int,faces[fi,list(p)])),'preserves_source_winding':bool(winding[pi]),
            'skin':{'max_condition':float(skin['cond'][fi,pi].max()),'min_mean_ratio':float(skin['mr'][fi,pi].min()),'min_sigma_m':float(skin['sigma_min'][fi,pi].min())},
            'fat':{'max_condition':float(fat['cond'][fi,pi].max()),'min_mean_ratio':float(fat['mr'][fi,pi].min()),'min_sigma_m':float(fat['sigma_min'][fi,pi].min())},
            'both_bands':{'max_condition':float(per_order_cond[fi,pi]),'min_mean_ratio':float(per_order_mr[fi,pi]),'min_sigma_m':float(per_order_sigma[fi,pi])}})
    baseidx=sorted_choice
    def eta_summary(points_outer: np.ndarray, points_inner: np.ndarray, label: str, layer: np.ndarray) -> dict[str,float|int]:
        q=layer.astype(np.float32).astype(np.float64); q0=q[:,:,:,0,:]
        qm=np.stack((q[:,:,:,1,:]-q0,q[:,:,:,2,:]-q0,q[:,:,:,3,:]-q0),axis=-1)
        det=np.abs(np.linalg.det(qm)); lmax=np.zeros_like(det)
        for i,j in itertools.combinations(range(4),2): lmax=np.maximum(lmax,np.linalg.norm(q[:,:,:,i,:]-q[:,:,:,j,:],axis=-1))
        eta=det/(lmax**3); base=eta[face_index[:,None],baseidx[:,None],:]
        return {'tets_all_six_orders':int(eta.size),'baseline_tets_below_floor':int((base<args.normalized_det_floor).sum()),'baseline_tets':int(base.size),'baseline_min':float(base.min()),'baseline_p01':float(np.quantile(base,.01)),'baseline_median':float(np.median(base))}
    # Tiny but nondegenerate 1 um radial shell on a 1 m semi-axis octahedron.
    octa=np.asarray([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],dtype=np.float64)
    inside=octa*(1.0-1e-6); ofaces=np.asarray([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
    oct_eta=[]
    for f in ofaces:
        a,b,c=map(int,f); local=np.asarray([[octa[a],octa[b],octa[c],inside[c]],[octa[a],octa[b],inside[b],inside[c]],[octa[a],inside[a],inside[b],inside[c]]],dtype=np.float64).astype(np.float32).astype(np.float64)
        for t in local:
            mat=np.stack((t[1]-t[0],t[2]-t[0],t[3]-t[0]),axis=-1); d=abs(np.linalg.det(mat)); l=max(np.linalg.norm(t[i]-t[j]) for i,j in itertools.combinations(range(4),2)); oct_eta.append(d/l**3)
    oct_eta=np.asarray(oct_eta)
    control_inside=octa*(1.0-1e-3); control_eta=[]
    for f in ofaces:
        a,b,c=map(int,f); local=np.asarray([[octa[a],octa[b],octa[c],control_inside[c]],[octa[a],octa[b],control_inside[b],control_inside[c]],[octa[a],control_inside[a],control_inside[b],control_inside[c]]],dtype=np.float64).astype(np.float32).astype(np.float64)
        for t in local:
            mat=np.stack((t[1]-t[0],t[2]-t[0],t[3]-t[0]),axis=-1); d=abs(np.linalg.det(mat)); l=max(np.linalg.norm(t[i]-t[j]) for i,j in itertools.combinations(range(4),2)); control_eta.append(d/l**3)
    control_eta=np.asarray(control_eta)
    candidate_order=np.asarray([perms.index(tuple(list(faces[base_by_set[tuple(sorted(c))]]).index(v) for v in c)) for c in candidate],dtype=np.int64)
    candidate_cond=per_order_cond[np.arange(len(candidate)),candidate_order]
    candidate_mr=per_order_mr[np.arange(len(candidate)),candidate_order]
    receipt={
      'schema':'whole_body_skin_fat_prism_order_quality_v1','basis':'Reference coordinates from the frozen header; triangle order evaluated with the existing three-tet staircase decomposition. Shape metrics use double arithmetic. Normalized determinant guard probe first quantizes each node coordinate to float32, then measures in double.',
      'inputs':{'header':{'path':args.header_source_label or (str(header.relative_to(root)) if header.is_relative_to(root) else str(header)),'sha256':sha(header)},'prism_builder':{'path':str(prep.relative_to(root)),'sha256':sha(prep)},'candidate':{'path':str(candidate_path.relative_to(root)),'sha256':sha(candidate_path)},'admitted_coordinates':{'path':str(coords_path.relative_to(root)),'sha256':sha(coords_path)},'embedded_source_payload_sha256':re.search(r'kWholeBodySkinSourcePayloadSHA256\[\]\s*=\s*"([0-9a-f]{64})"',text).group(1)},
      'ancestry':{'reduced_face_index':fi,'source_face_index':int(source_faces[fi]),'reduced_local_vertices':exact_source,'source_vertex_indices':mapped_source,'candidate_source_vertex_map_matches_header':source_map_equal,'candidate_faces_missing_from_header':int(missing),'candidate_face_orders_changed':int(order_changed),'candidate_odd_winding_flips':int(odd_flips)},
      'coordinate_receipt_max_abs_difference_m':coordinate_diffs,
      'baseline_sorted_id_decomposition':{'faces':len(faces),'per_face_max_condition_above_1e3':int((baseline>1e3).sum()),'per_face_max_condition_above_1e4':int((baseline>1e4).sum()),'max_per_face_condition':float(baseline.max()),'worst_face_index':int(baseline.argmax()),'skin_tet_condition_counts':per_tet_counts(skin,sorted_choice),'fat_tet_condition_counts':per_tet_counts(fat,sorted_choice)},
      'independent_best_order_lower_bound':{'all_six':{'faces_without_order_at_or_below_1e3':int((best_any>1e3).sum()),'faces_without_order_at_or_below_1e4':int((best_any>1e4).sum()),'max_best_condition':float(best_any.max()),'scope':'Independent per-face optimistic lower bound; shared-edge conformity is not enforced.'},'preserving_source_winding':{'faces_without_order_at_or_below_1e3':int((best_valid>1e3).sum()),'faces_without_order_at_or_below_1e4':int((best_valid>1e4).sum()),'max_best_condition':float(best_valid.max()),'scope':'Independent per-face optimistic lower bound; shared-edge conformity is not enforced.', 'unfixable_over_1e4':[{'face_index':int(i),'local_vertices':list(map(int,faces[i])),'best_condition':float(best_valid[i])} for i in np.flatnonzero(best_valid>1e4)]}},
      'topology_only_interpretation':{'positive_winding_faces_without_any_order_at_or_below_1e4':int((best_valid>1e4).sum()),'conclusion':'If this count is nonzero, ordering alone cannot put every face below 1e4, even before shared-edge conformity. A zero count is only necessary, not sufficient, for a globally conforming topology.'},
      'candidate_024_source_order':{'per_face_max_condition_above_1e3':int((candidate_cond>1e3).sum()),'per_face_max_condition_above_1e4':int((candidate_cond>1e4).sum()),'max_per_face_condition':float(candidate_cond.max()),'min_per_face_mean_ratio':float(candidate_mr.min()),'note':'Same face tuples/order as the header; these metrics model the file’s source order, while current runtime sorts each triangle by local vertex ID before decomposition.'},
      'face_witness':{'coordinates_m':{str(v):{'outer':outer[v].tolist(),'skin_fat_interface':interface[v].tolist(),'fat_deep':deep[v].tolist()} for v in exact_source},'six_orders':witness},
      'normalized_det_guard_probe':{'definition':'abs(det([p1-p0,p2-p0,p3-p0])) / Lmax^3; p are float32-quantized and differences/determinant/norms are evaluated in double. Sign/winding remains a separate check.','floor':args.normalized_det_floor,'skin':eta_summary(outer,interface,'skin',prisms(outer,interface,order)),'fat':eta_summary(interface,deep,'fat',prisms(interface,deep,order)),'1um_radial_shell_1m_semi_axis_octahedron':{'tets':int(len(oct_eta)),'below_floor':int((oct_eta<args.normalized_det_floor).sum()),'min':float(oct_eta.min()),'max':float(oct_eta.max())},'1mm_radial_shell_1m_semi_axis_octahedron_control':{'tets':int(len(control_eta)),'below_floor':int((control_eta<args.normalized_det_floor).sum()),'min':float(control_eta.min()),'max':float(control_eta.max())}},
      'audit_script_sha256':sha(Path(__file__).resolve())}
    out=args.output or root/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/mesh-quality-audit/receipt.json'
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(out),'header_sha256':receipt['inputs']['header']['sha256'],'baseline':receipt['baseline_sorted_id_decomposition'],'best_winding':receipt['independent_best_order_lower_bound']['preserving_source_winding'],'normalized_det':receipt['normalized_det_guard_probe']},indent=2))
if __name__=='__main__': main()
