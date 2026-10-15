#!/usr/bin/env python3
"""CPU-only CGLS conditioning study on the exact captured419 free-DOF CSR operator.

No regularization or operator alteration is performed.  Right scaling is only a
change of solve coordinates x=S*y.  FP32 arithmetic is a CPU recurrence model,
not a bitwise Metal implementation.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, struct
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy import sparse

BASE = Path('/Users/home/cardboard-implementation-20261014')
HERE = Path(__file__).resolve().parent
INPUT = BASE/'analysis/captured-base-state-cauchy-419-001'
CAP = BASE/'solver/two-stage-first-failure-rich-fd-419/operator-capture'
SOURCE = BASE/'snapshots/source-status5-vtccd-diagnostics-125'
ASSEMBLER = SOURCE/'matter/tools/cardboard_cached_operator_audit.py'
CGLS_SOURCE = BASE/'snapshots/source-bounded-native-cgls-mask-repair-142/matter/src/metal/base_cgls.metalinc'
CGLS_CPU = BASE/'analysis/native-base-cgls-mask-repair-142-001/cpu/cgls_reference.py'
PAYLOAD_BASE = BASE/'analysis/captured419-regularized-directions-001/base-replay-input.bin'
PAYLOAD_GEN = BASE/'analysis/captured419-truncated-cgls-directions-001/generate.py'
REPLAY_HELPER = BASE/'analysis/native-residual-direction-replay-419-001/build/native_residual_direction'
COMPILED_PACKAGE = BASE/'solver/two-stage-first-failure-rich-fd-419/output/compiled.nmatterpack'
SOURCE_BUILD = Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-125/source-build.json')
REGIONAL = BASE/'solver/two-stage-first-failure-rich-fd-419/output/rejected-conditioning-000002'
TARGETS=(32,64,128)
EXPORTS=(('none','fp64',64),('none','emulated_fp32',64),
         ('column','fp64',64),('column','fp64',128),
         ('column','emulated_fp32',64),('column','emulated_fp32',128),
         ('block3','fp64',64),('block3','fp64',128),
         ('block3','emulated_fp32',64),('block3','emulated_fp32',128))
HEADER=struct.Struct('<8s6IQfI')
EXPECTED_A='6b1d7f76b0a747cd28fa4ee53ef8b5316dd25ae390860a97242f1d91f63990ed'
EXPECTED_B='af22b09232b270e5cc15184d09cd8ed3d8f86b83bf08e765f510d6777fcaa8cb'
EXPECTED_CAPTURE='f2d42314cfee0428ecd285ebb9dbd5da1556624e3e92af1d12240fd8106b309a'
EXPECTED_CGLS=''


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def dot64(a,b):
    return math.fsum(float(x)*float(y) for x,y in zip(np.asarray(a).reshape(-1),np.asarray(b).reshape(-1)))
def norm64(x): return math.sqrt(max(0.0,dot64(x,x)))

def apply_s64(blocks, v, transpose=False):
    v=np.asarray(v,dtype=np.float64); out=np.empty_like(v)
    for bi, M in enumerate(blocks):
        M=M.T if transpose else M
        o=3*bi
        for i in range(3): out[o+i]=math.fsum(float(M[i,j])*float(v[o+j]) for j in range(3))
    return out

def build_scalings(A):
    n=A.shape[0]
    colnorm=np.sqrt(np.asarray(A.power(2).sum(axis=0),dtype=np.float64).reshape(-1))
    if not np.isfinite(colnorm).all() or np.any(colnorm<=0): raise ValueError('nonpositive/nonfinite column norm')
    col=1.0/colnorm
    blocks=[]; eig_min=float('inf'); eig_max=0.0
    for q in range(0,n,3):
        block=A[q:q+3,q:q+3].toarray()
        gram=block.T@block
        gram=0.5*(gram+gram.T)
        w,V=np.linalg.eigh(gram)
        eig_min=min(eig_min,float(w.min())); eig_max=max(eig_max,float(w.max()))
        if not np.isfinite(w).all() or w.min()<=max(1e-30,w.max()*1e-12):
            raise ValueError(f'ill-defined 3x3 right scale block {q//3}, eig={w.tolist()}')
        S=(V*(1.0/np.sqrt(w)))@V.T
        blocks.append(S)
    return {'none':None,'column':col,'block3':blocks}, {'column_scale_min':float(col.min()),'column_scale_max':float(col.max()),'block_gram_eigenvalue_min':eig_min,'block_gram_eigenvalue_max':eig_max,'block_count':len(blocks)}

def run64(A,b,kind,scale,targets):
    y=np.zeros(A.shape[1],np.float64); r=b.copy()
    applyS=(lambda x,transpose=False: x.copy()) if kind=='none' else (
        (lambda x,transpose=False: scale*x) if kind=='column' else
        (lambda x,transpose=False: apply_s64(scale,x,transpose)))
    s=applyS(np.asarray(A.T@r).reshape(-1),True); p=s.copy(); gamma=dot64(s,s)
    records={}; status='running'; breakdown=None
    for k in range(1,max(targets)+1):
        xdir=applyS(p)
        q=np.asarray(A@xdir).reshape(-1); denom=dot64(q,q)
        if not math.isfinite(gamma) or not math.isfinite(denom) or gamma<=0 or denom<=0:
            status='breakdown';breakdown={'iteration':k,'gamma':gamma,'denominator':denom};break
        alpha=gamma/denom
        y=y+alpha*p; r=r-alpha*q
        snew=applyS(np.asarray(A.T@r).reshape(-1),True); gn=dot64(snew,snew)
        if k in targets:
            x=applyS(y)
            true_r=b-np.asarray(A@x).reshape(-1)
            records[k]={'iterations':k,'status':'exhausted_at_checkpoint','recurrence_residual_l2':norm64(r),'true_unscaled_residual_l2':norm64(true_r),'true_residual_ratio':norm64(true_r)/norm64(b),'x_l2':norm64(x),'normal_residual_l2':math.sqrt(max(0.0,gn))}
        if not math.isfinite(gn) or gn<0:
            status='breakdown';breakdown={'iteration':k,'next_gamma':gn};break
        if gn==0:
            status='zero_normal_residual';break
        beta=gn/gamma
        if not math.isfinite(beta) or beta<=0:
            status='breakdown';breakdown={'iteration':k,'beta':beta};break
        p=snew+beta*p; s=snew; gamma=gn
    else: status='exhausted'
    return records, status, breakdown

def kahan_add(sumv, corr, value):
    adjusted=np.float32(value-corr); nxt=np.float32(sumv+adjusted); corr=np.float32(np.float32(nxt-sumv)-adjusted)
    return nxt,corr

def dot32_source(a,b,free_nodes,node_count):
    """Source-shaped 32-lane/Kahan/reduction dot on two-block full node layout."""
    a=np.asarray(a,dtype=np.float32).reshape(-1); b=np.asarray(b,dtype=np.float32).reshape(-1)
    node_map=np.full(node_count,-1,dtype=np.int32); node_map[free_nodes]=np.arange(len(free_nodes),dtype=np.int32)
    af=np.zeros((node_count,3),np.float32); bf=np.zeros_like(af)
    af[free_nodes]=a.reshape(-1,3); bf[free_nodes]=b.reshape(-1,3)
    lane_sum=np.zeros(32,np.float32); lane_corr=np.zeros(32,np.float32)
    for lane in range(32):
        local=lane
        while local<2*node_count:
            if local<node_count and node_map[local]>=0:
                av=af[local]; bv=bf[local]
                term=np.float32(np.float32(np.float32(av[0]*bv[0])+np.float32(av[1]*bv[1]))+np.float32(av[2]*bv[2]))
                lane_sum[lane],lane_corr[lane]=kahan_add(lane_sum[lane],lane_corr[lane],term)
            local+=32
    offset=16
    while offset:
        for lane in range(offset): lane_sum[lane]=np.float32(lane_sum[lane]+lane_sum[lane+offset])
        offset//=2
    return np.float32(lane_sum[0])

def apply_s32(kind, scale, v):
    v=np.asarray(v,dtype=np.float32)
    if kind=='none': return v.copy()
    if kind=='column': return np.asarray(scale*v,dtype=np.float32)
    out=np.empty_like(v)
    for bi,M in enumerate(scale):
        q=3*bi
        for i in range(3):
            t=np.float32(M[i,0]*v[q])+np.float32(M[i,1]*v[q+1])
            out[q+i]=np.float32(t+np.float32(M[i,2]*v[q+2]))
    return out

def run32(A,b,kind,scale,free_nodes,node_count,targets):
    A32=A.astype(np.float32).tocsr(); A32.sum_duplicates(); A32.sort_indices(); AT32=A32.transpose().tocsr(); AT32.sort_indices()
    rhs=np.asarray(b,dtype=np.float32); y=np.zeros(A.shape[1],np.float32); r=rhs.copy()
    s=apply_s32(kind,scale,np.asarray(AT32@r,dtype=np.float32)); p=s.copy(); gamma=dot32_source(s,s,free_nodes,node_count)
    records={}; status='running'; breakdown=None
    for k in range(1,max(targets)+1):
        q=np.asarray(A32@apply_s32(kind,scale,p),dtype=np.float32)
        denom=dot32_source(q,q,free_nodes,node_count)
        if not np.isfinite(gamma) or not np.isfinite(denom) or gamma<=0 or denom<=0:
            status='breakdown'; breakdown={'iteration':k,'gamma':float(gamma),'denominator':float(denom)}; break
        alpha=np.float32(gamma/denom)
        y=np.asarray(y+np.float32(alpha*p),dtype=np.float32)
        r=np.asarray(r-np.float32(alpha*q),dtype=np.float32)
        snew=apply_s32(kind,scale,np.asarray(AT32@r,dtype=np.float32))
        gn=dot32_source(snew,snew,free_nodes,node_count)
        if k in targets:
            x=apply_s32(kind,scale,y)
            true_r=b-np.asarray(A@x.astype(np.float64),dtype=np.float64).reshape(-1)
            records[k]={'iterations':k,'status':'exhausted_at_checkpoint','recurrence_residual_l2_fp32':float(np.sqrt(max(0.0,float(dot32_source(r,r,free_nodes,node_count))))),'true_unscaled_residual_l2':norm64(true_r),'true_residual_ratio':norm64(true_r)/norm64(b),'x_l2':norm64(x),'normal_residual_l2_fp32':float(np.sqrt(max(0.0,float(gn))))}
        if not np.isfinite(gn) or gn<0:
            status='breakdown';breakdown={'iteration':k,'next_gamma':float(gn)};break
        if gn==0: status='zero_normal_residual';break
        beta=np.float32(gn/gamma)
        if not np.isfinite(beta) or beta<=0:
            status='breakdown';breakdown={'iteration':k,'beta':float(beta)};break
        p=np.asarray(snew+np.float32(beta*p),dtype=np.float32); s=snew; gamma=gn
    else: status='exhausted'
    return records,status,breakdown

def make_payload(direction,free_nodes,node_count,base_payload,out_name):
    raw=bytearray(base_payload)
    magic,version,E,N,T,S,Q,fp,dt,reserved=HEADER.unpack_from(raw)
    if (magic,version,E,N,T,S,Q)!=(b'NMFDSWP1',1,2,882,1920,15,59): raise ValueError('native replay base header does not match source125 FEM scope')
    off=HEADER.size+E*N*80; nbytes=E*N*16
    if len(raw)!=2195272 or off!=141168 or nbytes!=28224: raise ValueError('native replay payload layout changed')
    full=np.zeros((E,N,4),dtype='<f4'); full[0,free_nodes,:3]=np.asarray(direction,dtype=np.float32).reshape(len(free_nodes),3)
    if np.any(full[1]) or np.any(full[:,:,3]) or np.any(full[0,np.setdiff1d(np.arange(N),free_nodes),:3]): raise ValueError('export direction violates quiet/fixed/w invariants')
    fullbytes=full.tobytes(order='C'); raw[off:off+nbytes]=fullbytes
    if raw[:off]!=base_payload[:off] or raw[off+nbytes:]!=base_payload[off+nbytes:]: raise ValueError('direction insertion changed a non-direction byte')
    p=HERE/'replay-inputs'/f'{out_name}.input.bin';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    d=HERE/'directions'/f'{out_name}.float32x4.bin';d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(fullbytes)
    return {'payload_path':str(p),'payload_sha256':sha(p),'payload_bytes':p.stat().st_size,'direction_path':str(d),'direction_sha256':sha(d),'direction_bytes':d.stat().st_size,'direction_offset':off,'direction_block_bytes':nbytes,'non_direction_bytes_unchanged':True,'fixed_nodes_zero':True,'quiet_environment_zero':True,'w_lane_zero':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE)
    args=ap.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    if out!=HERE: raise SystemExit('run_study.py is source-bound to its package directory; use the sealed package path')
    if (HERE/'report.json').exists() or (HERE/'manifest.json').exists(): raise SystemExit('refusing to overwrite an existing package')
    for p,h in [(INPUT/'A_free_csr.npz',EXPECTED_A),(INPUT/'base_rhs.float64le.bin',EXPECTED_B),(CAP,EXPECTED_CAPTURE)]:
        if sha(p)!=h: raise ValueError(f'input hash mismatch: {p}')
    A=sparse.load_npz(INPUT/'A_free_csr.npz').tocsr(); b=np.fromfile(INPUT/'base_rhs.float64le.bin',dtype='<f8')
    if A.shape!=(2538,2538) or A.nnz!=71046 or b.shape!=(2538,) or not np.isfinite(A.data).all() or not np.isfinite(b).all(): raise ValueError('unexpected/nonfinite source matrix')
    capreport=json.loads((INPUT/'report.json').read_text()); node_count=882; free_nodes=np.asarray(capreport['operator']['free_node_ids'],dtype=np.int64) if 'free_node_ids' in capreport['operator'] else None
    if free_nodes is None:
        # Obtain the exact free-node order from the sealed source125 assembly report payload.
        import importlib.util
        spec=importlib.util.spec_from_file_location('source125_cached_operator_audit',ASSEMBLER)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        header, rows, tets, integrity=mod.capture_records(CAP)
        mass, stiff, rebuilt, delta, metadata=mod.validate_and_assemble(header,rows,tets,integrity,0)
        diff=(rebuilt-A).tocsr(); diff.eliminate_zeros()
        if diff.nnz!=0 or not np.allclose(diff.data,0): raise ValueError('captured source assembler did not reproduce CSR')
        free_nodes=np.asarray(metadata['free_nodes'],dtype=np.int64)
        positions=np.asarray(metadata['node_positions_m'])
        if len(free_nodes)!=846 or positions.shape!=(882,3): raise ValueError('free-node/world layout mismatch')
    bnorm=norm64(b)
    scales,scale_meta=build_scalings(A)
    results={}
    for kind in ('none','column','block3'):
        results[kind]={}
        for prec in ('fp64','emulated_fp32'):
            if prec=='fp64': rec,status,breakdown=run64(A,b,kind,scales[kind],TARGETS)
            else:
                sc=scales[kind]
                if kind=='column': sc=np.asarray(sc,dtype=np.float32)
                elif kind=='block3': sc=np.asarray(sc,dtype=np.float32)
                rec,status,breakdown=run32(A,b,kind,sc,free_nodes,node_count,TARGETS)
            results[kind][prec]={'run_status':status,'breakdown':breakdown,'checkpoints':{str(k):v for k,v in rec.items()}}
    base=PAYLOAD_BASE.read_bytes(); payload_records={}
    for kind,prec,it in EXPORTS:
        rec=results[kind][prec]['checkpoints'].get(str(it))
        if rec is None: raise ValueError(f'no {kind}/{prec} direction at iteration {it}')
        # We retain each recurrence state for export by re-running once to its target.
        direction=direction_at(A,b,kind,scales[kind],prec,it,free_nodes,node_count)
        name=f'{kind}_{prec}_iter{it:03d}'
        payload_records[name]=make_payload(direction,free_nodes,node_count,base,name)
        exported=np.asarray(direction,dtype=np.float32).astype(np.float64)
        exported_r=b-np.asarray(A@exported).reshape(-1)
        rec['exported_payload_direction_l2']=norm64(exported)
        rec['exported_payload_true_unscaled_residual_l2']=norm64(exported_r)
        rec['exported_payload_true_residual_ratio']=norm64(exported_r)/bnorm
        rec['native_replay_payload']=payload_records[name]
        rec['payload_export_rounding']='Direction is serialized as FP32 NMFDSWP1 full-node float4; FP64 recurrence is rounded once for the native residual replay contract.'
    report={
      'schema':'numi.cardboard.captured419-cgls-right-preconditioning-study.v1',
      'classification':'CPU-only conditioning diagnostic on captured419 source125 unsymmetric free-DOF operator; no native GPU execution and no production solver qualification.',
      'source_bindings':{'operator_npz':{'path':str(INPUT/'A_free_csr.npz'),'sha256':sha(INPUT/'A_free_csr.npz')},'rhs':{'path':str(INPUT/'base_rhs.float64le.bin'),'sha256':sha(INPUT/'base_rhs.float64le.bin')},'rich_capture':{'path':str(CAP),'sha256':sha(CAP)},'source_snapshot':str(SOURCE),'source_assembler':{'path':str(ASSEMBLER),'sha256':sha(ASSEMBLER)},'CGLS_metal_source':{'path':str(CGLS_SOURCE),'sha256':sha(CGLS_SOURCE)},'CPU_CGLS_reference':{'path':str(CGLS_CPU),'sha256':sha(CGLS_CPU)},'prior_replay_generator':{'path':str(PAYLOAD_GEN),'sha256':sha(PAYLOAD_GEN)},'base_replay_payload':{'path':str(PAYLOAD_BASE),'sha256':sha(PAYLOAD_BASE)},'native_replay_helper':{'path':str(REPLAY_HELPER),'sha256':sha(REPLAY_HELPER)},'compiled_package':{'path':str(COMPILED_PACKAGE),'sha256':sha(COMPILED_PACKAGE)},'source_build_manifest':{'path':str(SOURCE_BUILD),'sha256':sha(SOURCE_BUILD)},'csr_shape':list(A.shape),'csr_nnz':int(A.nnz),'node_count':node_count,'free_node_count':len(free_nodes),'fixed_node_count':node_count-len(free_nodes),'rhs_l2_Ns':bnorm,'dt_seconds':9.999999747378752e-05},
      'operator_semantics':'Unmodified captured nonsymmetric A from source125: M + dt^2*K, K=-d(f_internal)/d(v); fixed DOFs already eliminated. All recurrence variants minimize ||A x-b||_2 without normal-matrix materialization.',
      'method':{'unknown':'x=S*y for right-preconditioned recurrences; final reported/exported direction is x','preconditioners':{'none':'S=I','column':'S[j,j]=1/||A[:,j]||_2','block3':'block-diagonal SPD S_n=(A_nn^T A_nn)^(-1/2), one 3x3 principal block per free FEM node, using eigendecomposition; no shift or regularization'},'iterations':list(TARGETS),'true_residual':'Recomputed as b-A*x in original unscaled operator with FP64 CSR action at each checkpoint; norms use math.fsum.','fp64':'FP64 CGLS recurrence and sparse actions; exported payload gets one required FP32 serialization rounding.','emulated_fp32':'CPU SciPy float32 CSR A/A^T actions; FP32 state/scaling/update arithmetic; CGLS dot uses source-shaped 32-lane Kahan accumulation and SIMD-tree sum over full 882-node backing layout. Not bitwise Metal: GPU matrix-free operator/reduction/action order differs.','native_cap':'Native CGLS is currently bounded to 32/64; 128 is CPU conditioning extrapolation only.'},
      'scale_diagnostics':scale_meta,'results':results,'native_replay_payloads':payload_records,
      'regional_graph_preconditioner':{'available_for_same_generation':False,'reason':'The 419 rich capture did not include a regional RAP sidecar under its aggregate readback budget. The later rejected-conditioning-000002 export has no graph-patch membership/factors and lacks a selected-Newton generation identity matching the 419 operator; its AP/RAP cannot be used as a same-generation preconditioner. No regional/graph result is fabricated.'},
      'limitations':['No GPU/native CGLS or production residual replay was run in this study. Payloads are prepared for the root-owned bounded native residual replay helper only.','Right scaling changes Krylov coordinates but not A; comparisons measure finite-iteration conditioning, not a new accepted solver policy.','The 3x3 block Gram scale is an offline, computed preconditioner and has not been admitted or used in production.','FP32 emulation is not Metal/native parity and reports true residual against unscaled FP64 A only.','No nonlinear Armijo, rollback, history, or full-coupon outcome is predicted by these linear residuals.','The graph/RAP option is unavailable at the exact selected generation, as documented above.'],
      'replay_contract':{'helper':str(REPLAY_HELPER),'compiled_package':str(COMPILED_PACKAGE),'command_template':f'{REPLAY_HELPER} {COMPILED_PACKAGE} <fresh replay-inputs/*.input.bin> <fresh output directory>','payload_header':'NMFDSWP1 v1; exact source125 2x882x4 float32 direction block replaced only at byte offset 141168, length 28224. Fixed rows, quiet environment, w lanes and all non-direction bytes are checked unchanged.'}}
    (HERE/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    readme=f'''# Captured 419 CGLS right-preconditioning study\n\nCPU-only conditioning study bound to captured source125 control 1 / Newton 48. It compares unpreconditioned CGLS, column right-scaling, and 3×3 per-node right-scaling on the same unsymmetrized 2538×2538 operator, without shifting or regularizing A. True residuals are always recomputed in the original unscaled A.\n\nThe 3×3 scale is S_n=(A_nnᵀA_nn)^(-1/2), computed from the 3×3 principal block of each free node. This is a diagnostic preconditioner; it has not been used in production. FP64 is a double-precision CPU recurrence. The FP32 variant uses float32 CPU CSR actions and source-shaped CGLS reductions but is not bitwise equivalent to Metal. Iteration 128 is outside the current native CGLS cap and is an offline diagnostic only.\n\nNative residual-replay payloads are included for both precisions at 64/128 iterations for the two right-scaled methods, plus unpreconditioned 64-iteration controls. Each payload preserves the exact NMFDSWP1 header, source/correction bytes, fixed nodes, quiet environment, and w lanes; only the direction block is replaced. Run them only with the root-owned helper command in report.json and fresh output directories.\n\nThe same-generation regional/graph RAP preconditioner is unavailable: the 419 rich capture omitted its sidecar under the aggregate capture budget, and the later terminal rejected-conditioning export has no graph patch map or matching selected-generation identity. No regional result is inferred from that later export.\n\nThis is linear conditioning evidence only. It does not establish native parity, nonlinear acceptance, a production policy, or coupon convergence. See report.json and manifest.json for complete numbers and hashes.\n'''
    (HERE/'README.md').write_text(readme)
    files={}
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name!='manifest.json': files[str(p.relative_to(HERE))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    manifest={'schema':'numi.cardboard.captured419-cgls-right-preconditioning-study-package.v1','report_sha256':sha(HERE/'report.json'),'source_capture_sha256':sha(CAP),'source_A_sha256':sha(INPUT/'A_free_csr.npz'),'source_b_sha256':sha(INPUT/'base_rhs.float64le.bin'),'files':files}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS_CPU_STUDY','report':str(HERE/'report.json'),'report_sha256':sha(HERE/'report.json'),'manifest':str(HERE/'manifest.json'),'manifest_sha256':sha(HERE/'manifest.json'),'results':{k:{p:{i:{'true_unscaled_residual_l2':v.get('true_unscaled_residual_l2'),'true_residual_ratio':v.get('true_residual_ratio')} for i,v in q['checkpoints'].items()} for p,q in d.items()} for k,d in results.items()},'payloads':{k:v['payload_sha256'] for k,v in payload_records.items()}},indent=2))

def direction_at(A,b,kind,scale,prec,it,free_nodes,node_count):
    if prec=='fp64':
        rec,_,_=run64(A,b,kind,scale,(it,))
        # run64 returns only metrics; use a second short solver exposing x
        return solve_direction64(A,b,kind,scale,it)
    sc=scale
    if kind=='column': sc=np.asarray(scale,dtype=np.float32)
    elif kind=='block3': sc=np.asarray(scale,dtype=np.float32)
    return solve_direction32(A,b,kind,sc,it,free_nodes,node_count)

def solve_direction64(A,b,kind,scale,it):
    applyS=(lambda x: x.copy()) if kind=='none' else ((lambda x: scale*x) if kind=='column' else (lambda x: apply_s64(scale,x,True)))
    applyST=(lambda x: x.copy()) if kind=='none' else ((lambda x: scale*x) if kind=='column' else (lambda x: apply_s64(scale,x,True)))
    y=np.zeros(A.shape[1],np.float64);r=b.copy();s=applyST(np.asarray(A.T@r).reshape(-1));p=s.copy();g=dot64(s,s)
    for _ in range(it):
        q=np.asarray(A@applyS(p)).reshape(-1);d=dot64(q,q)
        if g<=0 or d<=0: raise ValueError('FP64 CGLS broke down before export iteration')
        alpha=g/d;y=y+alpha*p;r=r-alpha*q;snew=applyST(np.asarray(A.T@r).reshape(-1));gn=dot64(snew,snew)
        if gn==0: return applyS(y)
        beta=gn/g;p=snew+beta*p;s=snew;g=gn
    return applyS(y)

def solve_direction32(A,b,kind,scale,it,free_nodes,node_count):
    A32=A.astype(np.float32).tocsr();A32.sum_duplicates();A32.sort_indices();AT=A32.T.tocsr();AT.sort_indices();rhs=np.asarray(b,np.float32); y=np.zeros(A.shape[1],np.float32);r=rhs.copy();s=apply_s32(kind,scale,np.asarray(AT@r,np.float32));p=s.copy();g=dot32_source(s,s,free_nodes,node_count)
    for _ in range(it):
        q=np.asarray(A32@apply_s32(kind,scale,p),np.float32);d=dot32_source(q,q,free_nodes,node_count)
        if not np.isfinite(g) or not np.isfinite(d) or g<=0 or d<=0: raise ValueError('FP32 CGLS broke down before export iteration')
        alpha=np.float32(g/d);y=np.asarray(y+np.float32(alpha*p),np.float32);r=np.asarray(r-np.float32(alpha*q),np.float32);sn=apply_s32(kind,scale,np.asarray(AT@r,np.float32));gn=dot32_source(sn,sn,free_nodes,node_count)
        if gn==0: return apply_s32(kind,scale,y)
        beta=np.float32(gn/g);p=np.asarray(sn+np.float32(beta*p),np.float32);s=sn;g=gn
    return apply_s32(kind,scale,y)

if __name__=='__main__': main()
