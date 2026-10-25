import hashlib, importlib.util, itertools, json, math, os
from fractions import Fraction
from pathlib import Path
import numpy as np

BASE = Path('/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded')
INPUT_JSON = BASE / 'native-parent-transfer-085c-cross-audit-001/run-001/cross-material-witnesses.json'
OUT = BASE / 'native-parent-transfer-085c-cross-audit-001/reclassification-002'
OWNER = BASE / 'native-parent-transfer-085b/plane_filter_085.py'
STACKED = BASE / 'native-parent-transfer-085b/plane_filter_085b.py'
EXPECTED = {
    str(INPUT_JSON): 'a770c026dd8e1eeeb17ba6c47c8a79c3e574be91a6f1882976d83dd9e7cbc1f9',
    str(OWNER): '0ba4109b943f7e1b4312ca339f8e5f2f931b81413daca4074fa2be42f3b9c305',
    str(STACKED): 'ae3cbe1e9f79a580b3a0f22b5126a2d91f11293874f9c867f46a83e81d6f3f61',
}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def det3(a):
    return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])

def solve3(A,b):
    d=det3(A)
    if d == 0: return None
    out=[]
    for j in range(3):
        M=[list(row) for row in A]
        for i in range(3): M[i][j]=b[i]
        out.append(det3(M)/d)
    return tuple(out)

def exact_planes(v):
    # Unnormalized exact planes; orient every inequality so tet interior is <= 0.
    def sub(a,b): return tuple(a[i]-b[i] for i in range(3))
    def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    def dot(a,b): return sum((a[i]*b[i] for i in range(3)), Fraction(0))
    planes=[]
    for k in range(4):
        ids=[i for i in range(4) if i!=k]
        p0,p1,p2=(v[i] for i in ids)
        n=cross(sub(p1,p0),sub(p2,p0)); h=dot(n,p0)
        s=dot(n,v[k])-h
        if s == 0: raise ValueError('degenerate input tetrahedron')
        if s > 0: n=tuple(-x for x in n); h=-h
        planes.append((n,h))
    return planes

def exact_intersection_dimension(a_float,b_float):
    a=[tuple(Fraction.from_float(float(x)) for x in p) for p in a_float]
    b=[tuple(Fraction.from_float(float(x)) for x in p) for p in b_float]
    planes=exact_planes(a)+exact_planes(b)
    verts=set()
    for ids in itertools.combinations(range(8),3):
        A=[list(planes[i][0]) for i in ids]
        rhs=[planes[i][1] for i in ids]
        x=solve3(A,rhs)
        if x is None: continue
        if all(sum((n[j]*x[j] for j in range(3)),Fraction(0)) <= h for n,h in planes): verts.add(x)
    if not verts: return {'vertex_count':0,'affine_rank':-1,'classification':'empty'}
    ps=list(verts); p0=ps[0]
    vecs=[tuple(p[i]-p0[i] for i in range(3)) for p in ps[1:]]
    v1=next((v for v in vecs if any(x != 0 for x in v)),None)
    if v1 is None: rank=0
    else:
        cross12=None
        for v in vecs:
            c=(v1[1]*v[2]-v1[2]*v[1],v1[2]*v[0]-v1[0]*v[2],v1[0]*v[1]-v1[1]*v[0])
            if any(x != 0 for x in c): cross12=c; break
        if cross12 is None: rank=1
        elif any(sum((cross12[i]*v[i] for i in range(3)),Fraction(0)) != 0 for v in vecs): rank=3
        else: rank=2
    cls='positive_full_dimensional_binary64_geometry' if rank==3 else 'lower_dimensional_or_empty_binary64_geometry'
    return {'vertex_count':len(verts),'affine_rank':rank,'classification':cls}

# Verify the frozen evidence and owner implementations before evaluating witnesses.
for p,h in EXPECTED.items():
    got=sha(p)
    if got != h: raise SystemExit(f'hash pin failed: {p}: {got} != {h}')
raw=json.loads(INPUT_JSON.read_text())
scalar=loadmod('owner085',OWNER)
stacked=loadmod('owner085b',STACKED)
rows=[]
for w in raw['witnesses']:
    a=np.asarray(w['candidate_tet_reference_points_m'],dtype=np.float64)
    b=np.asarray(w['native_tet_reference_points_m'],dtype=np.float64)
    shift=a[0].copy()
    al=a-shift; bl=b-shift
    original_eps=stacked.tetra_intersection_volume_stacked(a,b,2e-10,2e-10)
    scalar_eps=scalar.tetra_intersection_volume(a,b,2e-10,2e-10)
    original_zero=stacked.tetra_intersection_volume_stacked(a,b,0.0,2e-10)
    local_eps=stacked.tetra_intersection_volume_stacked(al,bl,2e-10,2e-10)
    local_zero=stacked.tetra_intersection_volume_stacked(al,bl,0.0,2e-10)
    exact=exact_intersection_dimension(a,b)
    rows.append({
        'candidate_child_row':w['candidate_child_row'],'candidate_cell_uid':w['candidate_cell_uid'],
        'candidate_material_id':w['candidate_material_id'],'native_parent_row':w['native_parent_row'],
        'native_source_tet_row':w['native_source_tet_row'],'native_material_id':w['native_material_id'],
        'original_reported_overlap_m3':w['overlap_volume_m3'],
        'stacked_owner_volume_tol_2e-10_m3':original_eps,
        'scalar_owner_volume_tol_2e-10_m3':scalar_eps,
        'stacked_owner_volume_tol_0_m3':original_zero,
        'translated_local_volume_tol_2e-10_m3':local_eps,
        'translated_local_volume_tol_0_m3':local_zero,
        'relative_difference_to_reported':(original_eps-w['overlap_volume_m3'])/w['overlap_volume_m3'] if w['overlap_volume_m3'] else None,
        'exact_binary64_halfspace_intersection':exact,
        'candidate_tet_reference_points_m':w['candidate_tet_reference_points_m'],
        'native_tet_reference_points_m':w['native_tet_reference_points_m'],
        'nearest_candidate_I_PLC_face_row':w['candidate_I_PLC_nearest_face_row'],
        'nearest_candidate_I_PLC_distance_m':w['candidate_I_PLC_distance_from_overlap_centroid_m'],
        'nearest_candidate_interface_is_source_I_PLC':w['candidate_nearest_interface_face_is_source_I_PLC'],
        'source_overlap_centroid_m':w['intersection_centroid_m'],
        'source_overlap_vertices_from_tolerance_clip_m':w['intersection_points_m'],
    })

OUT.mkdir(parents=True,exist_ok=False)
report={
 'status':'completed_witness_only_reclassification_no_gate_waiver',
 'method':'For each of 22 already-retained cross-material pairs only: rerun pinned scalar and stacked 073 clip at owner tolerance and zero inside tolerance, rerun after a common local translation, and compute exact rational halfspace-intersection vertex affine rank using Fraction.from_float on the serialized binary64 input coordinates.',
 'inputs_sha256':EXPECTED,
 'source_report_pins':raw['inputs_sha256'],
 'candidate_pairs_revisited':len(rows),
 'no_full_scan':True,'no_threshold_or_tolerance_change_to_frozen_owner':True,
 'interpretation_limit':'Exact full-dimensional positive class means the serialized binary64 tets overlap with positive volume; lower-dimensional means the owner positive-volume result is caused by floating/tolerance/hull construction. This is diagnostic only and does not waive the original gate or establish physical admission.',
 'results':rows,
}
out=OUT/'witness-reclassification.json'
out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'output':str(out),'count':len(rows),'classification_counts':{c:sum(r['exact_binary64_halfspace_intersection']['classification']==c for r in rows) for c in sorted({r['exact_binary64_halfspace_intersection']['classification'] for r in rows})},'tolerance_zero_positive':sum(r['stacked_owner_volume_tol_0_m3']>0 for r in rows),'local_zero_positive':sum(r['translated_local_volume_tol_0_m3']>0 for r in rows),'max_original_replay_relerr':max(abs(r['relative_difference_to_reported']) for r in rows),'sha256':sha(out)},indent=2))
