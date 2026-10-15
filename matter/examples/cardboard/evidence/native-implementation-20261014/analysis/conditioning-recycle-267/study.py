#!/usr/bin/env python3
"""CPU-only coarse-space enrichment sensitivity on captured Newton N63 A,b.

This is an offline diagnostic. It tests the current cumulative candidate
velocity increment available in the rich capture as a possible extra coarse
vector. It is NOT the prior Newton-62 line-search increment: production source
forms deltaVelocity = candidate velocity - accepted microstep velocity.
"""
from __future__ import annotations
import hashlib, json, math, signal, time, os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import linalg, sparse

BASE = Path('/Users/home/cardboard-implementation-20261014')
OUT = Path(__file__).resolve().parent / 'run-002'
AUDIT = BASE / 'analysis/control1-newton63-cached-operator-266'
CAPTURE_RUN = BASE / 'solver/sourcefit15-fg32-control1-newton63-capture-266'
P_DIR = CAPTURE_RUN / 'output/rejected-conditioning-000002'
P_MANIFEST = P_DIR / 'manifest.json'
GRAPH_JSON = Path(__file__).resolve().parent / 'graph-layout.json'
GRAPH_INPUT = Path(__file__).resolve().parent / 'layout-input.bin'
GRAPH_CPP = Path(__file__).resolve().parent / 'layout_from_source.cpp'
SOURCE = BASE / 'snapshots/source-runtime-diagnostic-return-repair-079'
GRAPH_HEADER = SOURCE / 'matter/src/regional_graph_schwarz.hpp'
FEM_METAL = SOURCE / 'matter/src/metal/fem.metalinc'
FGMRES_METAL = SOURCE / 'matter/src/metal/fgmres.metalinc'
MAX_SECONDS = 120.0
MAX_BYTES = 1 << 30
RESTART = 16
ITERATIONS = (32, 128, 512)
TOL = 1.0e-7


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def csr_sha(a: sparse.csr_matrix) -> str:
    h = hashlib.sha256()
    for item in (np.asarray(a.shape, dtype='<i8'),
                 np.asarray(a.indptr, dtype='<i8'),
                 np.asarray(a.indices, dtype='<i8'),
                 np.asarray(a.data, dtype='<f8')):
        h.update(item.tobytes())
    return h.hexdigest()




def dense_dot_fsum(a, b):
    return math.fsum(float(x) * float(y) for x, y in zip(np.ravel(a), np.ravel(b)))

def dense_matvec_fsum(a, x):
    a = np.asarray(a, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or x.shape != (a.shape[1],):
        raise ValueError('dense fsum matvec shape mismatch')
    return np.asarray([math.fsum(float(a[i,j]) * float(x[j]) for j in range(a.shape[1]))
                       for i in range(a.shape[0])], dtype=np.float64)

def dense_matmul_fsum(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('dense fsum matmul shape mismatch')
    return np.asarray([[math.fsum(float(a[i,k]) * float(b[k,j])
                                  for k in range(a.shape[1]))
                        for j in range(b.shape[1])] for i in range(a.shape[0])],
                       dtype=np.float64)

def stable_l2(v):
    return math.sqrt(math.fsum(float(x) * float(x) for x in np.ravel(v)))


def load_inputs():
    report = json.loads((AUDIT / 'report.json').read_text())
    matrix = sparse.load_npz(AUDIT / 'report.operator.npz').tocsr()
    vectors = np.load(AUDIT / 'report.vectors.npz')
    topology = np.load(AUDIT / 'report.topology.npz')
    pmanifest = json.loads(P_MANIFEST.read_text())
    graph = json.loads(GRAPH_JSON.read_text())
    state = np.asarray(vectors['captured_node_state_f32'], dtype=np.float32)
    free_nodes = np.asarray(vectors['free_node_ids'], dtype=np.int64)
    rhs = np.asarray(vectors['native_rhs_free_impulse_Ns'], dtype=np.float64)
    native = np.asarray(vectors['direction'], dtype=np.float64)
    nnode = state.shape[0]
    if report.get('capture_selection') != {'environment': 0, 'root': 1, 'newton_iteration': 63}:
        raise ValueError('unexpected exact linearization selection')
    if matrix.shape != (3 * len(free_nodes), 3 * len(free_nodes)) or rhs.shape != (matrix.shape[0],):
        raise ValueError('operator/RHS/free-node dimensions mismatch')
    if len(np.unique(free_nodes)) != len(free_nodes) or not np.array_equal(free_nodes, np.flatnonzero(state[:, 11] == 0.0)):
        raise ValueError('free-node map does not match captured candidate fixed tags')
    if nnode != int(topology['node_count']) or graph['node_count'] != nnode:
        raise ValueError('captured graph/layout node count mismatch')
    if graph['tetrahedron_count'] != len(topology['nodes']) or graph['environment_count'] != 2:
        raise ValueError('graph layout is not the captured two-environment topology')
    if graph['patch_count'] != len(graph['patches']):
        raise ValueError('layout patch count mismatch')
    for p in graph['patches']:
        if not p or any(int(node) < 0 or int(node) >= nnode for node in p):
            raise ValueError('invalid graph patch node identity')
        if len(set(p)) != len(p):
            raise ValueError('duplicate node in graph patch')
        tags = state[np.asarray(p, dtype=np.int64), 11]
        if np.any(tags != tags[0]):
            raise ValueError('source graph patch crosses captured free/fixed partition')
    Praw = np.fromfile(P_DIR / 'P.bin', dtype='<f4')
    mode_count = int(pmanifest['mode_count'])
    if Praw.size != mode_count * nnode * 4 or mode_count != 26:
        raise ValueError('source static basis byte shape mismatch')
    pfiles = {x['name']:x for x in pmanifest['files']}
    if sha(P_DIR / 'P.bin') != pfiles['P.bin']['sha256']:
        raise ValueError('source static P bytes fail manifest hash')
    Praw = Praw.reshape(mode_count, nnode, 4)
    if not np.isfinite(Praw).all() or np.max(np.abs(Praw[:, state[:,11] != 0, :3])) != 0.0:
        raise ValueError('source P has nonfinite entries or nonzero fixed-node support')
    if np.max(np.abs(Praw[:, :, 3])) != 0.0:
        raise ValueError('source P has nonzero reserved fourth component')
    # Runtime reported P == effective_P for both captured environments, so
    # the rejected diagnostic's static basis is usable as a source-mode
    # definition; its AP/RAP/factors are intentionally not consumed.
    EPraw = np.fromfile(P_DIR / 'effective_P.bin', dtype='<f4').reshape(2, mode_count, nnode, 4)
    if not np.array_equal(Praw, EPraw[0]) or not np.array_equal(EPraw[0], EPraw[1]):
        raise ValueError('captured static and effective P differ; refusing stale basis')
    pfree = np.transpose(Praw[:, free_nodes, :3], (1, 2, 0)).reshape(3*len(free_nodes), mode_count).astype(np.float64)
    orth_err = np.linalg.norm(dense_matmul_fsum(pfree.T, pfree) - np.eye(mode_count), ord=np.inf)
    if not np.isfinite(orth_err) or orth_err > 2e-4:
        raise ValueError(f'source basis not sufficiently orthonormal: {orth_err}')
    return report, matrix, vectors, topology, pmanifest, graph, state, free_nodes, rhs, native, pfree, orth_err


def build_preconditioner(A, Z, graph, free_nodes, nnode, memory_base):
    mode = Z.shape[1]
    if mode < 1 or mode > 32:
        raise ValueError('coarse basis exceeds source mode cap')
    Z32 = np.asarray(Z, dtype=np.float32)
    # A and all AP/RAP are freshly reconstructed from the selected N63 A;
    # rejected-output AP/RAP/factor files are never used.
    AP = np.asarray(A @ Z32.astype(np.float64), dtype=np.float32).astype(np.float64)
    RAP = np.asarray(dense_matmul_fsum(Z32.astype(np.float64).T, AP), dtype=np.float32)
    if not np.isfinite(AP).all() or not np.isfinite(RAP).all():
        raise ValueError('nonfinite reconstructed AP/RAP')
    cond = float(np.linalg.cond(RAP.astype(np.float64)))
    if not math.isfinite(cond):
        raise ValueError('nonfinite RAP condition estimate')
    try:
        rap_lu, rap_piv = linalg.lu_factor(RAP.astype(np.float64), check_finite=True)
    except Exception as exc:
        raise ValueError(f'RAP factorization failed: {exc}') from exc

    index = {int(node): i for i, node in enumerate(free_nodes)}
    coverage = np.zeros(nnode, dtype=np.uint32)
    patch_factors = []
    patch_nodes_free = []
    workspace = 0
    for pi, raw_nodes in enumerate(graph['patches']):
        nodes = np.asarray(raw_nodes, dtype=np.int64)
        tags = np.asarray([node not in index for node in nodes])
        if np.all(tags):
            # Initially fixed-only supplementary patches have zero output for
            # this captured free mask and do not affect the free operator.
            patch_nodes_free.append((pi, nodes, None))
            continue
        if np.any(tags):
            raise ValueError(f'patch {pi} crosses free/fixed partition')
        dofs = np.concatenate([3 * index[int(n)] + np.arange(3, dtype=np.int64) for n in nodes])
        local = np.asarray(A[dofs][:, dofs].toarray(), dtype=np.float64)
        if not np.isfinite(local).all():
            raise ValueError(f'patch {pi} matrix is nonfinite')
        workspace += 2 * local.nbytes + 4 * local.shape[0] * 4
        if memory_base + workspace > MAX_BYTES:
            raise MemoryError('captured inputs and dense local blocks exceed 1 GiB CPU limit')
        lu, piv = linalg.lu_factor(local, check_finite=True)
        if not np.isfinite(lu).all() or len(piv) != local.shape[0]:
            raise ValueError(f'patch {pi} factor invalid')
        coverage[nodes] += 1
        patch_factors.append((pi, nodes, dofs, lu, piv, local))
        patch_nodes_free.append((pi, nodes, dofs))
    if np.any(coverage[free_nodes] == 0):
        raise ValueError('captured free node outside graph patch coverage')

    def apply(r):
        r = np.asarray(r, dtype=np.float64)
        coarse_rhs = dense_matvec_fsum(Z32.astype(np.float64).T, r)
        coeff = linalg.lu_solve((rap_lu, rap_piv), coarse_rhs, check_finite=True)
        residual = r - dense_matvec_fsum(AP, coeff)
        graph_sum = np.zeros_like(r)
        for _pi, _nodes, dofs, lu, piv, _local in patch_factors:
            graph_sum[dofs] += linalg.lu_solve((lu, piv), residual[dofs], check_finite=False)
        for fi, node in enumerate(free_nodes):
            graph_sum[3*fi:3*fi+3] /= float(coverage[int(node)])
        return dense_matvec_fsum(Z32.astype(np.float64), coeff) + graph_sum
    meta = {
        'mode_count': mode,
        'RAP_condition_2norm': cond,
        'RAP_shape': list(RAP.shape),
        'free_patch_count': len(patch_factors),
        'source_fixed_patch_count': len(graph['patches']) - len(patch_factors),
        'source_free_patch_nodes': int(sum(len(x[1]) for x in patch_factors)),
        'max_patch_nodes': max(len(x[1]) for x in patch_factors),
        'patch_factor_precision': 'CPU float64 SciPy LU; no native factor equivalence claimed',
        'operator_basis_products': 'AP and RAP rebuilt from the selected captured A and basis for each variant',
        'graph_action': 'principal free-free A block solve per source free patch; sum and divide once by source coverage; fixed supplementary patches are zero-output under captured fixed mask',
        'estimated_dense_factor_workspace_bytes': int(workspace),
    }
    return apply, meta


def candidate_basis(Z, v):
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (Z.shape[0],) or not np.isfinite(v).all():
        raise ValueError('invalid candidate direction')
    raw_norm = float(np.linalg.norm(v))
    if raw_norm <= 0.0:
        raise ValueError('candidate direction is zero')
    q = v.copy()
    for _ in range(2):
        q -= dense_matvec_fsum(Z, dense_matvec_fsum(Z.T, q))
    orth_norm = float(np.linalg.norm(q))
    if not math.isfinite(orth_norm) or orth_norm <= 1e-10 * raw_norm:
        raise ValueError(f'candidate direction dependent on existing modes: {orth_norm/raw_norm}')
    q /= orth_norm
    return np.column_stack((Z, q)), {
        'candidate_raw_norm': raw_norm,
        'candidate_orthogonalized_norm': orth_norm,
        'candidate_removed_by_projection_fraction': 1.0 - orth_norm/raw_norm,
        'new_basis_gram_inf_error': float(np.linalg.norm(dense_matmul_fsum(np.column_stack((Z,q)).T, np.column_stack((Z,q))) - np.eye(Z.shape[1]+1), ord=np.inf)),
    }


def right_fgmres(A, b, M, restart, max_iterations, tol):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0.0: raise ValueError('zero captured RHS')
    rows = []
    total = 0
    cycle = 0
    t0 = time.monotonic()
    while total < max_iterations:
        r = b - A @ x
        beta = float(np.linalg.norm(r))
        rel = beta / bnorm
        rows.append({'iteration': total, 'cycle': cycle, 'true_residual_l2': beta, 'relative_true_residual': rel, 'stage':'restart'})
        if rel <= tol: break
        m = min(restart, max_iterations - total)
        V = np.zeros((n, m+1), dtype=np.float64)
        Z = np.zeros((n, m), dtype=np.float64)
        H = np.zeros((m+1, m), dtype=np.float64)
        V[:,0] = r / beta
        best = None
        used = 0
        for j in range(m):
            Z[:,j] = M(V[:,j])
            w = np.asarray(A @ Z[:,j], dtype=np.float64)
            for i in range(j+1):
                H[i,j] = np.dot(V[:,i], w)
                w -= H[i,j] * V[:,i]
            # One additional MGS pass limits CPU emulation drift; this is not
            # claimed bit-equivalent to production SIMD reductions.
            for i in range(j+1):
                corr = np.dot(V[:,i], w)
                H[i,j] += corr
                w -= corr * V[:,i]
            H[j+1,j] = np.linalg.norm(w)
            if H[j+1,j] > 0.0: V[:,j+1] = w / H[j+1,j]
            e = np.zeros(j+2); e[0] = beta
            y, *_ = np.linalg.lstsq(H[:j+2,:j+1], e, rcond=None)
            trial = x + dense_matvec_fsum(Z[:,:j+1], y)
            true_r = b - A @ trial
            true_norm = float(np.linalg.norm(true_r))
            total += 1; used = j+1
            rows.append({'iteration': total, 'cycle':cycle, 'true_residual_l2':true_norm,
                         'relative_true_residual':true_norm/bnorm,'stage':'arnoldi_candidate'})
            best = trial
            if true_norm / bnorm <= tol or H[j+1,j] == 0.0 or total >= max_iterations:
                break
        if best is None: break
        x = best
        cycle += 1
        if time.monotonic() - t0 > MAX_SECONDS:
            raise TimeoutError('study exceeded 120 second CPU cap')
    final_r = b - A @ x
    final = float(np.linalg.norm(final_r))
    return x, rows, {'iterations':total,'cycles':cycle,'relative_true_residual':final/bnorm,
                     'true_residual_l2':final,'correction_l2':float(np.linalg.norm(x)),
                     'converged_at_declared_offline_tol':final/bnorm<=tol,
                     'restart':restart,'max_iterations':max_iterations,'relative_tolerance':tol,
                     'elapsed_s':time.monotonic()-t0}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    timed = {'value':False}
    def alarm(*_):
        timed['value']=True; raise TimeoutError('120-second CPU wall cap')
    old = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        report,A,vectors,topo,pm,graph,state,free,b,native,Z,ortherr=load_inputs()
        # Candidate available in rich captured nodes: current candidate
        # velocity less the accepted microstep velocity (cumulative at the
        # timestep level, not the previous Newton step). Fixed nodes stay zero.
        delta = np.asarray(state[free,12:15], dtype=np.float32).astype(np.float64).reshape(-1)
        Zplus, enrich_meta = candidate_basis(Z, delta)
        # Independently retain low-mode alignment for diagnosis only.
        modes=np.load(AUDIT/'symmetric-part-modes.npz')
        low=np.asarray(modes['eigenvectors'][:,0],dtype=np.float64)
        mode_alignment=float(abs(np.dot(delta/np.linalg.norm(delta),low)))
        source_hashes={
            'report_json':sha(AUDIT/'report.json'),
            'operator_npz':sha(AUDIT/'report.operator.npz'),
            'vectors_npz':sha(AUDIT/'report.vectors.npz'),
            'topology_npz':sha(AUDIT/'report.topology.npz'),
            'offline_direct_npz':sha(AUDIT/'offline-direct-correction.npz'),
            'P_manifest':sha(P_MANIFEST),'P_bytes':sha(P_DIR/'P.bin'),
            'effective_P_bytes':sha(P_DIR/'effective_P.bin'),
            'graph_layout_json':sha(GRAPH_JSON),'graph_layout_input':sha(GRAPH_INPUT),
            'graph_layout_builder_cpp':sha(GRAPH_CPP),'graph_layout_builder_executable':sha(Path(__file__).resolve().parent/'layout_from_source'),
            'graph_source_header':sha(GRAPH_HEADER),'production_fem_metal':sha(FEM_METAL),
            'production_fgmres_metal':sha(FGMRES_METAL),
            'study_script':sha(Path(__file__).resolve()),
        }
        csr_hash=csr_sha(A)
        coverage=np.zeros(state.shape[0],dtype=np.uint32)
        for patch in graph['patches']: coverage[np.asarray(patch,dtype=np.int64)] += 1
        info={
            'binding':{'source_snapshot':str(SOURCE),'operator_report_sha256':source_hashes['report_json'],
                       'operator_selection':report.get('capture_selection'),
                       'native_run_prediction_sha256':report.get('binding',{}).get('prediction_sha256'),
                       'native_run_receipt_sha256':report.get('binding',{}).get('receipt_sha256'),
                       'P_manifest_source_physics_fingerprint':pm['source_physics_fingerprint'],
                       'P_manifest_device_program_fingerprint':pm['device_program_fingerprint'],
                       'operator_csr_canonical_sha256':csr_hash,'source_hashes':source_hashes},
            'captured_system':{'dofs':int(A.shape[0]),'nodes':int(state.shape[0]),'free_nodes':int(len(free)),
                'tetrahedra':int(len(topo['nodes'])),'rhs_l2':float(np.linalg.norm(b)),
                'native_correction_l2':float(np.linalg.norm(native)),
                'native_residual_relative':float(np.linalg.norm(b-A@native)/np.linalg.norm(b)),
                'matrix_nnz':int(A.nnz),'operator_unsymmetrized':True},
            'basis':{'legacy_mode_count':int(Z.shape[1]),'source_P_orthonormality_inf_error':ortherr,
                'source_static_P_equals_effective_P_both_envs':True,'RAP/AP_rebuilt_from_captured_A':True,
                'augmented_mode':{'field':'captured_node_state_f32[:,12:15] = deltaVelocity xyz',
                'semantics':'candidate velocity minus accepted microstep velocity at the captured nonlinear state; cumulative timestep candidate direction, not Newton-62 accepted correction',
                'fixed_node_support_zero':True,'absolute_alignment_to_lowest_symmetric_A_mode':mode_alignment,
                **enrich_meta}},
            'graph_layout':{'patch_count':len(graph['patches']),'patch_node_count':int(graph['patch_node_count']),
                'free_patch_count':int(sum(1 for p in graph['patches'] if state[np.asarray(p,dtype=np.int64),11][0]==0)),
                'fixed_patch_count':int(sum(1 for p in graph['patches'] if state[np.asarray(p,dtype=np.int64),11][0]!=0)),
                'free_coverage_minmax':[int(np.min(coverage[free])),int(np.max(coverage[free]))],
                'patch_nodes_limit':256,'anchor_stride':128},
            'method':'Offline CPU restarted right-FGMRES(16), exact captured unsymmetrized A and native b. Each variant rebuilds AP/RAP from its own basis and current A, uses source graph patch membership and double precision dense LU of captured A principal blocks; implementation is an idealized coarse-space diagnostic, not native selector-5 parity. Dense products use explicit math.fsum to avoid the known NumPy 2.2.5 matmul warning behavior; sparse A actions and SciPy factorizations remain checked finite.',
            'numerical_warnings':{'numpy_dense_matmul':'The first exploratory run-001 emitted divide-by-zero/overflow/underflow/invalid warnings on bounded finite dense matmul outputs; it is retained as inadmissible preliminary evidence. run-002 replaces explicit dense products with math.fsum and records any remaining warnings in stderr.'},
            'variants':{},
            'limitations':['The exact accepted Newton-62 correction is not included in capture 266; the tested added mode is cumulative candidate deltaVelocity, not that prior increment.',
                'CPU double-precision patch/coarse factorization is not the Metal FP32 factor/apply path.',
                'Reconstructed A is the audited captured linearization; this does not change or qualify the solver, physical trajectory, or paperboard model.'],
        }
        for label,basis in [('legacy26',Z),('legacy26_plus_cumulative_candidate_delta_velocity',Zplus)]:
            apply,meta=build_preconditioner(A,basis,graph,free,state.shape[0],A.data.nbytes+A.indices.nbytes+A.indptr.nbytes)
            vout={}
            for limit in ITERATIONS:
                x,hist,summary=right_fgmres(A,b,apply,RESTART,limit,TOL)
                vout[str(limit)]={'summary':summary,'history':hist,
                    'difference_from_native_current_correction_l2':float(np.linalg.norm(x-native)),
                    'direction_cosine_to_offline_direct':float(np.dot(x,np.load(AUDIT/'offline-direct-correction.npz')['direct'])/(np.linalg.norm(x)*np.linalg.norm(np.load(AUDIT/'offline-direct-correction.npz')['direct']))) if np.linalg.norm(x)>0 else 0}
            info['variants'][label]={'preconditioner':meta,'runs':vout}
            if time.monotonic()-start > MAX_SECONDS: raise TimeoutError('study exceeded total CPU cap')
        out=OUT/'report.json';out.write_text(json.dumps(info,sort_keys=True,indent=2,allow_nan=False)+'\n')
        (OUT/'source-hashes.json').write_text(json.dumps(source_hashes,sort_keys=True,indent=2)+'\n')
    except Exception as exc:
        (OUT/'failure.json').write_text(json.dumps({'failure':str(exc),'timed_out':timed['value'],'elapsed_s':time.monotonic()-start},sort_keys=True,indent=2)+'\n')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old)
    print(OUT/'report.json')
    return 0

if __name__=='__main__': raise SystemExit(main())
