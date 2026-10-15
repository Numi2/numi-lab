#!/usr/bin/env python3
"""Source-bound CPU audit for native selector-5 accepted-correction capture."""
from __future__ import annotations
import hashlib, importlib.util, json, math, pathlib, struct, subprocess, sys
import numpy as np
from scipy import sparse

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014')
CASE = BASE / 'solver/accepted-correction-recycle-first-control-284'
SNAP = BASE / 'snapshots/source-accepted-correction-regional-recycle-085'
OUT = BASE / 'analysis/accepted-correction-recycle-audit-001'
REPO = pathlib.Path('/Users/home/numi-cardboard-20261006')
PYTHON = pathlib.Path('/Users/home/.pyenv/versions/3.10.12/bin/python3')

def sha_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def sha(path: pathlib.Path) -> str: return sha_bytes(path.read_bytes())
def require(ok, message):
    if not ok: raise ValueError(message)
def fsum_dot(a,b): return math.fsum(float(x)*float(y) for x,y in zip(a,b))

def solve_lu_f32(matrix, pivots, rhs):
    """Scalar rounded FP32 forward/back substitution, matching kernel order."""
    n=len(pivots); y=np.zeros(n,dtype=np.float32); x=np.zeros(n,dtype=np.float32)
    for row in range(n):
        value=np.float32(rhs[int(pivots[row])])
        for col in range(row):
            product=np.float32(matrix[row,col]*y[col])
            value=np.float32(value-product)
        y[row]=value
    for row in range(n-1,-1,-1):
        value=np.float32(y[row])
        for col in range(row+1,n):
            product=np.float32(matrix[row,col]*x[col])
            value=np.float32(value-product)
        diagonal=np.float32(matrix[row,row])
        if not np.isfinite(value) or not np.isfinite(diagonal) or abs(float(diagonal)) <= 1e-30:
            raise ValueError(f'FP32 LU solve invalid at row {row}')
        x[row]=np.float32(value/diagonal)
    if not np.isfinite(x).all(): raise ValueError('FP32 LU solution nonfinite')
    return x

def solve_lu_f64(matrix, pivots, rhs):
    """FP64 substitution using the captured FP32 factors and exact row pivot."""
    n=len(pivots); y=[0.0]*n; x=[0.0]*n
    for row in range(n):
        y[row]=math.fsum([float(rhs[int(pivots[row])])] +
                         [-float(matrix[row,col])*y[col] for col in range(row)])
    for row in range(n-1,-1,-1):
        numer=math.fsum([y[row]] +
                        [-float(matrix[row,col])*x[col] for col in range(row+1,n)])
        diagonal=float(matrix[row,row])
        if not math.isfinite(numer) or not math.isfinite(diagonal) or abs(diagonal)<=1e-30:
            raise ValueError(f'FP64 LU solve invalid at row {row}')
        x[row]=numer/diagonal
    if not all(math.isfinite(v) for v in x): raise ValueError('FP64 LU solution nonfinite')
    return np.asarray(x,dtype=np.float64)

def matvec_fsum(matrix, vector):
    return np.asarray([fsum_dot(row,vector) for row in matrix],dtype=np.float64)

def inverse_action_audit(true_rap, lu_f32, pivots):
    n=len(pivots); rows=[]; f32_max=0.0; f64_max=0.0
    f32_max_abs=0.0; f64_max_abs=0.0; f32_max_relx=0.0; f64_max_relx=0.0
    f32_solutions=[]; f64_solutions=[]
    matrix_norm_inf=max(math.fsum(abs(float(x)) for x in row) for row in true_rap)
    for col in range(n):
        rhs=np.zeros(n,dtype=np.float64); rhs[col]=1.0
        x32=solve_lu_f32(lu_f32,pivots,rhs.astype(np.float32)).astype(np.float64)
        x64=solve_lu_f64(lu_f32,pivots,rhs)
        r32=matvec_fsum(true_rap,x32)-rhs
        r64=matvec_fsum(true_rap,x64)-rhs
        n32=float(np.linalg.norm(r32)); n64=float(np.linalg.norm(r64))
        den32=matrix_norm_inf*max(abs(float(x)) for x in x32)+1.0
        den64=matrix_norm_inf*max(abs(float(x)) for x in x64)+1.0
        berr32=max(abs(float(x)) for x in r32)/den32
        berr64=max(abs(float(x)) for x in r64)/den64
        exact=np.linalg.solve(true_rap,rhs)
        relx32=float(np.linalg.norm(x32-exact)/np.linalg.norm(exact))
        relx64=float(np.linalg.norm(x64-exact)/np.linalg.norm(exact))
        f32_max=max(f32_max,n32); f64_max=max(f64_max,n64)
        f32_max_abs=max(f32_max_abs,float(np.max(np.abs(r32))))
        f64_max_abs=max(f64_max_abs,float(np.max(np.abs(r64))))
        f32_max_relx=max(f32_max_relx,relx32); f64_max_relx=max(f64_max_relx,relx64)
        f32_solutions.append(x32); f64_solutions.append(x64)
        rows.append({'rhs_column':col,'fp32_true_rap_residual_l2':n32,
                     'fp32_true_rap_residual_max_abs':float(np.max(np.abs(r32))),
                     'fp32_backward_error_inf':berr32,'fp32_solution_relative_error_vs_true_rap':relx32,
                     'fp64_true_rap_residual_l2':n64,
                     'fp64_true_rap_residual_max_abs':float(np.max(np.abs(r64))),
                     'fp64_backward_error_inf':berr64,'fp64_solution_relative_error_vs_true_rap':relx64})
    return {'method':'all unit RHS columns; FP32 rounds each scalar multiply/subtract/divide in kernel order; FP64 uses math.fsum substitution with same captured FP32 LU and pivots; residuals evaluated against independently rebuilt nonsymmetric true RAP',
            'unit_rhs_count':n,'max_true_rap_residual_l2_fp32':f32_max,
            'max_true_rap_residual_max_abs_fp32':f32_max_abs,
            'max_backward_error_inf_fp32':max(x['fp32_backward_error_inf'] for x in rows),
            'max_solution_relative_error_vs_true_rap_fp32':f32_max_relx,
            'max_true_rap_residual_l2_fp64':f64_max,
            'max_true_rap_residual_max_abs_fp64':f64_max_abs,
            'max_backward_error_inf_fp64':max(x['fp64_backward_error_inf'] for x in rows),
            'max_solution_relative_error_vs_true_rap_fp64':f64_max_relx,
            'all_unit_rhs_relative_residual_below_existing_1e-3_local_solve_diagnostic':bool(max(f32_max,f64_max)<=1.0e-3),
            'max_absolute_difference_fp32_vs_fp64_solution':float(max(np.max(np.abs(a-b)) for a,b in zip(f32_solutions,f64_solutions))),
            'unit_rhs_results':rows}

def load_capture(manifest_path, payload_path):
    mraw=manifest_path.read_bytes(); m=json.loads(mraw); payload=payload_path.read_bytes()
    require(m.get('schema')=='numi.matter.regional_rap_capture.v1','unsupported capture schema')
    require(m.get('byte_order')=='little_endian','byte order mismatch')
    require(m.get('capture_point')=='after_regional_rap_and_graph_factor_barrier_before_fgmres_correction','capture-point mismatch')
    require(m.get('preconditioner_selector')==5 and m['graph_schwarz'].get('enabled') is True,'not selector-5 graph capture')
    require(m.get('root')==0 and m.get('iteration')==1,'selected root/iteration mismatch')
    require(m['payload']['byte_count']==len(payload) and m['payload']['sha256']==sha_bytes(payload),'payload hash/length mismatch')
    require(len(payload)<=32*1024*1024,'capture exceeds declared 32 MiB readback cap')
    arrays={}; cursor=0
    widths={'<f4':4,'<u4':4,'<f4x4':16,'<u4x4':16,'struct:NMFEMNodeStateGPU':80,'struct:NMMatterStatusGPU':48}
    for d in m['arrays']:
        name=d['name']; require(name not in arrays,'duplicate capture array')
        require(d['dtype'] in widths and widths[d['dtype']]==d['element_bytes'],'array dtype/element size mismatch')
        count=math.prod(d['shape']); nbytes=count*d['element_bytes']
        require(nbytes==d['byte_count'] and d['offset']==cursor,'array shape/length/offset discontinuity')
        raw=payload[cursor:cursor+nbytes]; require(sha_bytes(raw)==d['sha256'],f'array hash mismatch: {name}')
        arrays[name]=(d,raw); cursor+=nbytes
    require(cursor==len(payload),'unclaimed payload bytes')
    return m,arrays,sha_bytes(mraw),sha_bytes(payload)

def decode(arrays,name,shape=None):
    d,raw=arrays[name]
    if d['dtype'] in ('<f4','<f4x4'): dt='<f4'
    elif d['dtype'] in ('<u4','<u4x4'): dt='<u4'
    else: raise ValueError(f'cannot numeric-decode {name}')
    arr=np.frombuffer(raw,dtype=dt).copy()
    sh=tuple(shape if shape is not None else d['shape'])
    if d['dtype'].endswith('x4'): sh += (4,)
    return arr.reshape(sh)

def validate_identity(m,arrays,env=0):
    n=int(m['node_count']); t=int(m['tetrahedron_count']); e=int(m['environment_count']);
    nodes_raw=arrays['candidate_nodes'][1][env*n*80:(env+1)*n*80]
    state=np.frombuffer(nodes_raw,dtype='<f4').reshape(n,20)
    require(np.isfinite(state).all(),'candidate nodes nonfinite')
    tags=state[:,11]
    require(np.isin(tags,[0.0,1.0]).all(),'candidate fixed tags not exact binary')
    free=(tags==0.0).astype('u1')
    graph=m['graph_schwarz']
    require(sha_bytes(free.tobytes())==graph['source_free_mask_sha256'],'source free-mask digest mismatch')
    potential=np.ones(n,dtype='u1')
    require(sha_bytes(potential.tobytes())==graph['potential_node_mask_sha256'],'potential mask digest mismatch')
    digest=hashlib.sha256()
    for name in ['candidate_nodes','material_history','scheduler_exponents','deformation_high','deformation_low','algorithmic_tangent_cache','algorithmic_tangent_cache_valid']:
        d,raw=arrays[name]; width=len(raw)//e
        digest.update(raw[env*width:(env+1)*width])
    identity=next(x for x in m['candidate_linearization_sha256'] if x['environment']==env)
    require(digest.hexdigest()==identity['sha256'],'candidate linearization digest mismatch')
    require(sha_bytes(nodes_raw)==identity['candidate_nodes_sha256'],'candidate node digest mismatch')
    return state,free

def validate_graph_cpp(m,arrays,free):
    expected={'graph_patches':'patches.bin','graph_patch_nodes':'patch_nodes.bin','graph_rows':'rows.bin',
              'graph_node_patch_offsets':'node_patch_offsets.bin','graph_node_patch_values':'node_patch_values.bin','graph_node_coverage':'node_coverage.bin'}
    bindir=OUT/'graph_layout_output'; counts=json.loads(subprocess.check_output([str(OUT/'graph_layout'),str(OUT/'topology_input.u32'),str(OUT/'source_free_mask.u8'),str(bindir)],text=True))
    graph=m['graph_schwarz']
    for key,val in [('patch_count','patch_count'),('patch_node_count','patch_node_count'),('dof_count','dof_count'),('matrix_float_count','matrix_float_count'),('pivot_count','pivot_count')]:
        require(int(graph[key])==int(counts[val]),f'graph count mismatch {key}')
    require(int(graph['layout_checksum_fnv1a32'])==int(counts['checksum']),'graph layout checksum mismatch')
    checks={}
    for name,file in expected.items():
        raw=(bindir/file).read_bytes(); d,got=arrays[name][0],arrays[name][1]
        require(d['byte_count']==len(raw) and raw==got,f'exact source85 graph layout differs at {name}')
        checks[name]={'bytes':len(raw),'sha256':sha_bytes(raw),'exact_match':True}
    return {'status':'exact_source85_layout_match','counts':counts,'arrays':checks}

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    # Static-source identity from package exporter compiled against frozen085 headers/library.
    pkg=CASE/'output/compiled.nmatterpack'; mesh=CASE/'output/mesh.json';
    mesh_doc=json.loads(mesh.read_text()); conn=np.asarray(mesh_doc['tetrahedra'],dtype='<u4')
    words=np.concatenate([np.array([len(mesh_doc['nodes_m']),len(conn)],dtype='<u4'),conn.reshape(-1)])
    words.tofile(OUT/'topology_input.u32')
    raw_nodes=np.fromfile(OUT/'raw/nodes.bin',dtype='<f4').reshape(-1,20)
    static_free=(raw_nodes[:,11]==0).astype('u1'); static_free.tofile(OUT/'source_free_mask.u8')
    manifest,payload_arrays,mhash,phash=load_capture(CASE/'operator-capture.regional.json',CASE/'operator-capture.regional.bin')
    require(manifest['source_physics_fingerprint']==json.loads((OUT/'export.stdout.json').read_text())['physics_fingerprint'],'package physics fingerprint differs from capture')
    package_meta=json.loads((OUT/'export.stdout.json').read_text())
    require(package_meta['nodes']==manifest['node_count'] and package_meta['tetrahedra']==manifest['tetrahedron_count'],'package dimensions differ')
    topology_bytes=words.astype('<u4',copy=False).tobytes()
    require(sha_bytes(topology_bytes)==manifest['graph_schwarz']['source_topology_sha256'],'compiled topology digest differs')
    captured_state,free=validate_identity(manifest,payload_arrays,0)
    require(np.array_equal(free,static_free),'candidate constraints differ from static package source')
    graph_validation=validate_graph_cpp(manifest,payload_arrays,free)
    # Static package ABI decode: 96-byte tetrahedra, 80-byte nodes, 176-byte objects.
    n=int(manifest['node_count']); T=int(manifest['tetrahedron_count']); E=int(manifest['environment_count']); M=int(manifest['mode_count'])
    require(package_meta['tet_stride']==96 and package_meta['node_stride']==80 and package_meta['object_stride']==176,'unexpected packed ABI strides')
    twords=np.fromfile(OUT/'raw/tetrahedra.bin',dtype='<u4').reshape(T,24)
    tf=twords.view('<f4'); conn_static=twords[:,:4].astype(np.int64)
    require(np.array_equal(conn_static,conn),'compiled package connectivity differs from exact same-run mesh')
    inv=tf[:,4:16].reshape(T,3,4)[:,:,:3].astype(np.float64)
    volume=tf[:,7].astype(np.float64); quat=tf[:,20:24].astype(np.float64)
    object_index=twords[:,17].astype(np.int64)
    object_words=np.fromfile(OUT/'raw/objects.bin',dtype='<u4').reshape(-1,44)
    flags=object_words[object_index,2]
    mass_static=raw_nodes[:,3]
    mass=captured_state[:,3]
    require(np.array_equal(mass,mass_static),'captured nodal masses differ from exact compiled package')
    require(np.array_equal(captured_state[:,11],raw_nodes[:,11]),'candidate authored constraints differ from exact package')
    cache=decode(payload_arrays,'algorithmic_tangent_cache',(E,T,81))[0]
    cache_valid=decode(payload_arrays,'algorithmic_tangent_cache_valid',(E,T))[0]
    require(np.all(cache_valid==1),'one or more selected env0 cached tangents invalid')
    eff=decode(payload_arrays,'effective_P',(E,M,n))[0]
    ap=decode(payload_arrays,'AP',(E,M,n))[0]
    p=decode(payload_arrays,'P',(M,n))
    stored_factor=decode(payload_arrays,'pivoted_RAP',(E,M,M))[0]
    piv=decode(payload_arrays,'pivots',(E,M))[0].astype(int)
    factor_valid=decode(payload_arrays,'factor_valid',(E,))
    recycle=decode(payload_arrays,'accepted_correction_recycle_mode',(E,))
    require(factor_valid[0]==1,'env0 coarse factor invalid')
    require(sorted(piv.tolist())==list(range(M)),'coarse pivots are not a permutation')
    # Ensure captured graph map exactly occupies source85 layout; assemble A from same generation's native cache.
    source_tool=SNAP/'matter/tools/cardboard_cached_operator_audit.py'
    spec=importlib.util.spec_from_file_location('cacheaudit',source_tool); cacheaudit=importlib.util.module_from_spec(spec); spec.loader.exec_module(cacheaudit)
    rows=[]; cols=[]; values=[]
    for ti in range(T):
        ke=cacheaudit.local_tet_stiffness(inv[ti],float(volume[ti]),cache[ti].reshape(9,9).astype(np.float64),quat[ti],bool(flags[ti]&512))
        ids=conn_static[ti]
        dofs=(ids[:,None]*3+np.arange(3)).reshape(-1)
        for i,gi in enumerate(dofs):
            if free[gi//3]==0: continue
            for j,gj in enumerate(dofs):
                if free[gj//3]==0: continue
                v=float(ke[i,j])
                if v:
                    rows.append(int(gi)); cols.append(int(gj)); values.append(v)
    ndof=3*n
    K=sparse.coo_matrix((values,(rows,cols)),shape=(ndof,ndof)).tocsr(); K.sum_duplicates()
    free_dof=np.repeat(free.astype(bool),3)
    dt=float(manifest['dt_seconds'])
    A=sparse.diags(np.repeat(mass.astype(np.float64),3))+dt*dt*K
    AP_errors=[]; per_mode_energy=[]; predicted_AP=[]; native_AP=[]; mass_actions=[]; stiff_actions=[]
    for mode in range(M):
        x=eff[mode,:,:3].astype(np.float64).reshape(-1); x[~free_dof]=0.0
        actual=ap[mode,:,:3].astype(np.float64).reshape(-1)
        pred=np.asarray(A@x).reshape(-1); pred[~free_dof]=0.0; actual[~free_dof]=0.0
        mact=np.repeat(mass.astype(np.float64),3)*x
        kact=dt*dt*np.asarray(K@x).reshape(-1)
        err=pred-actual
        actual_norm=float(np.linalg.norm(actual)); pred_norm=float(np.linalg.norm(pred)); err_norm=float(np.linalg.norm(err))
        rel=err_norm/actual_norm if actual_norm>0 else (0.0 if err_norm==0 else None)
        AP_errors.append({'mode':mode,'predicted_Ap_l2':pred_norm,'native_AP_l2':actual_norm,'absolute_error_l2':err_norm,'relative_error_l2':rel,'max_abs_error':float(np.max(np.abs(err))),'mass_action_l2':float(np.linalg.norm(mact)),'dt2_stiffness_action_l2':float(np.linalg.norm(kact)),'energy_pAp_f64':fsum_dot(x,pred),'energy_pAPnative_f64':fsum_dot(x,actual)})
        predicted_AP.append(pred); native_AP.append(actual); mass_actions.append(mact); stiff_actions.append(kact)
    # Rebuild two RAPs independently: one from the captured native AP array,
    # another from the independently assembled A action. The latter is the
    # denominator for the inverse-action audit below.
    captured_rap=np.zeros((M,M),dtype=np.float64)
    true_rap=np.zeros((M,M),dtype=np.float64)
    for i in range(M):
        for j in range(M):
            pi=eff[i,:,:3].reshape(-1)
            captured_rap[i,j]=fsum_dot(pi,native_AP[j])
            true_rap[i,j]=fsum_dot(pi,predicted_AP[j])
    lu=stored_factor.astype(np.float64); L=np.tril(lu,-1)+np.eye(M); U=np.triu(lu)
    reconstructed=np.zeros((M,M),dtype=np.float64)
    for i in range(M):
        for j in range(M): reconstructed[i,j]=math.fsum(float(L[i,k])*float(U[k,j]) for k in range(M))
    pivoted=true_rap[piv,:]
    rap_error=reconstructed-pivoted
    rap_norm=float(np.linalg.norm(true_rap)); rap_rel=float(np.linalg.norm(rap_error)/max(rap_norm,1e-300))
    rap_capture_delta=true_rap-captured_rap
    inverse_audit=inverse_action_audit(true_rap,stored_factor.astype(np.float32),piv)
    base_rap=true_rap[:M-1,:M-1]
    full_singular=np.linalg.svd(true_rap,compute_uv=False)
    base_singular=np.linalg.svd(base_rap,compute_uv=False)
    schur=float(true_rap[M-1,M-1]-fsum_dot(
        true_rap[M-1,:M-1],np.linalg.solve(base_rap,true_rap[:M-1,M-1])))
    u_diagonal=np.diag(np.triu(stored_factor.astype(np.float64)))
    # Also inspect the recycle vector and base-mode Gram/energy to expose weak or redundant selected direction.
    base_gram=np.zeros((M-1,M-1),dtype=np.float64)
    for i in range(M-1):
        for j in range(M-1): base_gram[i,j]=fsum_dot(eff[i,:,:3].reshape(-1),eff[j,:,:3].reshape(-1))
    v=eff[M-1,:,:3].astype(np.float64).reshape(-1)
    rv={'mode_index':M-1,'selection_code':int(recycle[0]),'P_raw_norm':float(np.linalg.norm(p[M-1,:,:3])),'effective_norm':float(np.linalg.norm(v)),'projection_onto_effective_base_max_abs':max(abs(fsum_dot(eff[i,:,:3].reshape(-1),v)) for i in range(M-1)),'pAp_energy_from_rebuilt_A':AP_errors[-1]['energy_pAp_f64'],'native_AP_abs_error_l2':AP_errors[-1]['absolute_error_l2']}
    # Manifest/source bindings and detailed results.
    report={
      'schema':'numi.cardboard.accepted_correction_recycle_audit.v1','status':'captured_linearization_audited_not_full_solve',
      'case':str(CASE),'source_snapshot':str(SNAP),'prediction_sha256':sha(CASE/'prediction.json'),'receipt_sha256':sha(CASE/'receipt.json'),'stderr_sha256':sha(CASE/'stderr.log'),
      'capture_manifest_sha256':mhash,'capture_payload_sha256':phash,'compiled_package_sha256':sha(pkg),'same_world_comparison':{'263_mesh_sha256':sha(BASE/'solver/budgeted-graph-legacy-first-control-263/output/mesh.json'),'283_mesh_sha256':sha(BASE/'solver/recycle-source-default-first-control-283/output/mesh.json'),'284_mesh_sha256':sha(mesh),'all_three_meshes_bitwise_identical':len({sha(BASE/f'solver/{x}/output/mesh.json') for x in ['budgeted-graph-legacy-first-control-263','recycle-source-default-first-control-283','accepted-correction-recycle-first-control-284']})==1,'compiled_package_sha256_283':sha(BASE/'solver/recycle-source-default-first-control-283/output/compiled.nmatterpack'),'compiled_package_sha256_284':sha(pkg)},
      'metadata':{k:manifest[k] for k in ['source_physics_fingerprint','device_program_fingerprint','root','iteration','environment_count','node_count','tetrahedron_count','mode_count','linearization_generation','micro_generation','dt_seconds','tangent_cache_used']},
      'receipt':json.loads((CASE/'receipt.json').read_text()),'probe_result':json.loads((CASE/'output/result.json').read_text()),
      'package_export':package_meta,'source_file_hashes':{'runtime.mm':sha(SNAP/'matter/src/runtime.mm'),'fgmres.metalinc':sha(SNAP/'matter/src/metal/fgmres.metalinc'),'regional_graph_schwarz.hpp':sha(SNAP/'matter/src/regional_graph_schwarz.hpp'),'shared.h':sha(SNAP/'matter/include/numi/matter/shared.h'),'cached_operator_audit.py':sha(source_tool),'export.cpp':sha(OUT/'export.cpp'),'export':sha(OUT/'export'),'graph_layout.cpp':sha(OUT/'graph_layout.cpp'),'graph_layout':sha(OUT/'graph_layout')},
      'topology_validation':{'source_topology_sha256':manifest['graph_schwarz']['source_topology_sha256'],'source_free_mask_sha256':manifest['graph_schwarz']['source_free_mask_sha256'],'potential_node_mask_sha256':manifest['graph_schwarz']['potential_node_mask_sha256'],'graph_layout':graph_validation,'candidate_mask_matches_compiled_source':True},
      'static_state_validation':{'compiled_nodes':n,'compiled_tets':T,'mass_exact_match':True,'authored_constraint_mask_exact_match':True,'valid_tangent_count':int(np.count_nonzero(cache_valid)),'total_tangent_count':int(cache_valid.size),'material_state_stride':int(manifest['material_state_stride'])},
      'recycle':rv,'operator_assembly':{'equation':'A=M+dt^2*K, K=-d(internal_force)/d(v); exact compiled inverse-rest, volume, frames and masses plus captured same-generation tangent cache; fixed xyz eliminated','method':'independent FP64 assembly and math.fsum small reductions; no A symmetrization','AP_per_mode':AP_errors,'max_AP_relative_error_nonzero_columns':max((x['relative_error_l2'] or 0.0 for x in AP_errors),default=0.0),'max_AP_absolute_error_l2':max(x['absolute_error_l2'] for x in AP_errors),'coarse_RAP_replay':{'true_rap_from_independent_A_action':true_rap.tolist(),'captured_rap_from_native_AP':captured_rap.tolist(),'true_vs_captured_rap_max_abs':float(np.max(np.abs(rap_capture_delta))),'native_pivots_row_permutation':piv.tolist(),'factor_valid':int(factor_valid[0]),'lu_reconstruction_relative_error_against_true_rap':rap_rel,'lu_reconstruction_max_abs_error_against_true_rap':float(np.max(np.abs(rap_error))),'minimum_abs_U_diagonal':float(np.min(np.abs(u_diagonal))),'RAP_frobenius_norm':rap_norm,'base_effective_gram_max_deviation_from_identity':float(np.max(np.abs(base_gram-np.eye(M-1)))),'smallest_RAP_singular_value':float(full_singular[-1]),'largest_RAP_singular_value':float(full_singular[0]),'RAP_2norm_condition_estimate':float(full_singular[0]/full_singular[-1]),'base_without_recycle':{'mode_count':M-1,'smallest_singular_value':float(base_singular[-1]),'largest_singular_value':float(base_singular[0]),'condition_2norm':float(base_singular[0]/base_singular[-1])},'recycle_schur_complement':schur,'recycle_schur_fraction_of_diagonal':schur/float(true_rap[M-1,M-1]),'condition_increase_percent_from_recycle_mode':100.0*((full_singular[0]/full_singular[-1])/(base_singular[0]/base_singular[-1])-1.0),'unit_rhs_inverse_actions':inverse_audit}},
      'limits':['one captured root0/Newton1 linearization only; it is not a native preconditioner action or solve comparison','full requested step failed the strict residual certificate and accepted_steps=0','first Newton diagnostic indicates recycle selection, not full-cycle benefit','cached tangent A replay checks capture/operator consistency; it does not independently validate constitutive tangent accuracy','no physical/material calibration or mesh-convergence claim','selector5 reader v2 fixed-node check is incompatible with source085 dynamic graph layout: source85 appends disjoint patches for initially-fixed nodes; exact source-header layout is checked here instead']}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':report['status'],'recycle':rv,'max_AP_rel':report['operator_assembly']['max_AP_relative_error_nonzero_columns'],'RAP_factor_reconstruction_relative_error':rap_rel,'RAP_cond':report['operator_assembly']['coarse_RAP_replay']['RAP_2norm_condition_estimate'],'inverse_unit_rhs_max_residual_f32':inverse_audit['max_true_rap_residual_l2_fp32'],'inverse_unit_rhs_max_residual_f64':inverse_audit['max_true_rap_residual_l2_fp64'],'inverse_unit_rhs_max_rel_solution_error_f32':inverse_audit['max_solution_relative_error_vs_true_rap_fp32'],'inverse_unit_rhs_max_rel_solution_error_f64':inverse_audit['max_solution_relative_error_vs_true_rap_fp64']},indent=2))
if __name__=='__main__':
    try: run()
    except Exception as e: print(f'audit failed closed: {e}',file=sys.stderr); raise
