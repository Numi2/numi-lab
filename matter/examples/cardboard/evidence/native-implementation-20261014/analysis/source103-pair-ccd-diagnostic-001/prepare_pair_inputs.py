#!/usr/bin/env python3
"""Prepare a strict, source-bound pair of captured source-103 CCD inputs."""
from __future__ import annotations
import argparse, hashlib, json, math, struct
from pathlib import Path
import numpy as np

ROOT = Path('/Users/home/cardboard-implementation-20261014')
SOURCE = ROOT / 'snapshots/source-whole-mortar-proof-103'
CENTER = ROOT / 'native/first-fatal-center-359'
TRIAL = ROOT / 'native/first-fatal-private-trial-358'
PAIR = (92, 127, 283, 1)
EDGES = ((0, 1), (1, 2), (2, 0))
EPS32 = 1.1920928955078125e-7

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def strict_json(path: Path):
    def reject_constant(value):
        raise ValueError(f'non-standard JSON constant {value} in {path}')
    return json.loads(path.read_text(), parse_constant=reject_constant)

def read_capture(manifest_path: Path):
    manifest = strict_json(manifest_path)
    payload_path = manifest_path.parent / Path(manifest['payload']['file']).name
    raw = payload_path.read_bytes()
    if len(raw) != manifest['payload']['byte_count'] or sha(payload_path) != manifest['payload']['sha256']:
        raise ValueError(f'payload size/hash mismatch: {payload_path}')
    arrays = {}
    spans = []
    for row in manifest['arrays']:
        key = (row['name'], row.get('phase'))
        if key in arrays:
            raise ValueError(f'duplicate array key {key}')
        start, size = int(row['offset']), int(row['byte_count'])
        end = start + size
        if start < 0 or size < 0 or end > len(raw):
            raise ValueError(f'array bounds invalid for {key}')
        data = raw[start:end]
        if len(data) != size or sha_bytes(data) != row['sha256']:
            raise ValueError(f'array hash/size mismatch for {key}')
        arrays[key] = (row, data)
        spans.append((start, end, key))
    spans.sort()
    if any(spans[i][1] > spans[i+1][0] for i in range(len(spans)-1)):
        raise ValueError('capture array payload spans overlap')
    return manifest, payload_path, arrays

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def array(arrays, name, phase, dtype, shape):
    row, data = arrays[(name, phase)]
    value = np.frombuffer(data, dtype=dtype).copy()
    expected = int(np.prod(shape))
    if value.size != expected:
        raise ValueError(f'{name}/{phase} count mismatch: {value.size} != {expected}')
    return value.reshape(shape), row

def bits_f32(x):
    return [f'{int(v):08x}' for v in np.asarray(x, dtype='<f4').view('<u4').reshape(-1)]

def f32_json(x):
    arr = np.asarray(x, dtype='<f4')
    if not np.isfinite(arr).all():
        raise ValueError('non-finite input not admissible for standalone capture package')
    return [float(v) for v in arr.reshape(-1)]

def norm(v):
    return float(np.sqrt(np.dot(v, v)))

def closest_triangle(p, a, b, c):
    # Independent double-precision closest-point implementation (Ericson).
    ab, ac, ap = b-a, c-a, p-a
    d1, d2 = float(ab@ap), float(ac@ap)
    if d1 <= 0.0 and d2 <= 0.0: return a.copy(), np.array([1.,0.,0.])
    bp=p-b
    d3, d4 = float(ab@bp), float(ac@bp)
    if d3 >= 0.0 and d4 <= d3: return b.copy(), np.array([0.,1.,0.])
    vc=d1*d4-d3*d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v=d1/(d1-d3); return a+v*ab,np.array([1.-v,v,0.])
    cp=p-c; d5,d6=float(ab@cp),float(ac@cp)
    if d6 >= 0.0 and d5 <= d6: return c.copy(),np.array([0.,0.,1.])
    vb=d5*d2-d1*d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w=d2/(d2-d6); return a+w*ac,np.array([1.-w,0.,w])
    va=d3*d6-d5*d4
    if va <= 0.0 and d4-d3 >= 0.0 and d5-d6 >= 0.0:
        w=(d4-d3)/((d4-d3)+(d5-d6)); return b+w*(c-b),np.array([0.,1.-w,w])
    den=va+vb+vc
    if den == 0.0: raise ValueError('degenerate triangle has undefined interior closest point')
    v,w=vb/den,vc/den
    return a+v*ab+w*ac,np.array([1.-v-w,v,w])

def closest_segments(p0,p1,q0,q1):
    # Independent double-precision closest points on closed segments.
    u,v,w=p1-p0,q1-q0,p0-q0
    aa,ee,ff=float(u@u),float(v@v),float(v@w)
    if aa == 0.0 and ee == 0.0: s=t=0.0
    elif aa == 0.0: s=0.0;t=min(1.0,max(0.0,ff/ee))
    else:
        cc=float(u@w)
        if ee == 0.0: t=0.0;s=min(1.0,max(0.0,-cc/aa))
        else:
            bb=float(u@v);den=aa*ee-bb*bb
            s=min(1.0,max(0.0,(bb*ff-cc*ee)/den)) if den>0.0 else 0.0
            t=(bb*s+ff)/ee
            if t < 0.0: t=0.0;s=min(1.0,max(0.0,-cc/aa))
            elif t > 1.0: t=1.0;s=min(1.0,max(0.0,(bb-cc)/aa))
    return p0+s*u,q0+t*v,s,t

def closest_case(kind, points):
    if kind == 0:
        q,bary=closest_triangle(points[0],*points[1:4])
        return norm(points[0]-q), bary, q
    p,q,s,t=closest_segments(*points)
    return norm(p-q), np.array([s,t]), 0.5*(p+q)

def golden_min(fn, left, right, iterations=80):
    phi=(math.sqrt(5.0)-1.0)/2.0
    c=right-phi*(right-left);d=left+phi*(right-left)
    fc,fd=fn(c),fn(d)
    for _ in range(iterations):
        if fc <= fd:
            right,d,fd=d,c,fc;c=right-phi*(right-left);fc=fn(c)
        else:
            left,c,fc=c,d,fd;d=left+phi*(right-left);fd=fn(d)
    t=0.5*(left+right)
    return t,fn(t)

def cpu_reference(case, samples=4097):
    x0=np.asarray(case['points_start_m'],dtype=np.float64).reshape(4,3)
    x1=np.asarray(case['points_finish_m'],dtype=np.float64).reshape(4,3)
    kind=case['kind_id']; thick=case['thickness_m']
    eps=EPS32
    scale=max(thick,1.0e-12,float(np.max(np.abs(np.concatenate([x0,x1])))))
    roundoff=8.0*eps*scale
    effective=thick+roundoff
    def at(t):
        return closest_case(kind,x0+t*(x1-x0))[0]
    ts=np.linspace(0.0,1.0,samples)
    ds=np.array([at(float(t)) for t in ts])
    candidates=[(float(ds[0]),0.0),(float(ds[-1]),1.0)]
    for i in range(1,samples-1):
        if ds[i] <= ds[i-1] and ds[i] <= ds[i+1]:
            t,d=golden_min(at,float(ts[i-1]),float(ts[i+1]))
            candidates.append((d,t))
    min_d,min_t=min(candidates)
    # FP64 reimplementation of the production conservative-advance stepping
    # policy. Its result is a comparison trace, not a formal CCD certificate.
    displacements=x1-x0
    if kind == 0:
        # Production uses the sum of absolute endpoint displacement bounds.
        # A translation-invariant mathematical Lipschitz bound follows from
        # v_point - sum(beta_i v_i) = sum(beta_i (v_point-v_i)), beta_i>=0,
        # sum(beta_i)=1. This is only a hypothesis/reference value: it does
        # not include outward FP32 rounding needed to replace the production
        # certificate.
        speed=norm(displacements[0])+max(norm(displacements[i]) for i in range(1,4))
        relative_speed=max(norm(displacements[0]-displacements[i]) for i in range(1,4))
        maxiter=16
    else:
        # Segment-point velocities are convex combinations of endpoint
        # velocities; their difference is a convex combination of all four
        # endpoint-pair velocity differences.
        speed=max(norm(displacements[i]) for i in (0,1))+max(norm(displacements[i]) for i in (2,3))
        relative_speed=max(norm(displacements[i]-displacements[j])
                          for i in (0,1) for j in (2,3))
        maxiter=17
    speed_ratio=(speed/relative_speed) if relative_speed>0.0 else None
    trace=[]; t=0.0; terminal='iteration_exhausted'; certified=False; hit=False
    for it in range(maxiter):
        dist=at(t)
        rec={'iteration':it,'time':t,'distance_m':dist,'gap_to_effective_m':dist-effective,'production_absolute_displacement_bound_m':speed,'galilean_relative_displacement_bound_m':relative_speed}
        trace.append(rec)
        if dist <= effective:
            terminal='distance_within_effective_thickness';certified=True;hit=True;break
        if not speed > 1.0e-12:
            terminal='stationary_certified_miss';certified=True;break
        step=max((dist-effective)/speed,1.0e-6)
        rec['step'] = step
        t += step
        if t > 1.0 + 8.0*eps:
            terminal='advanced_past_end_certified_miss';certified=True;break
        t=min(t,1.0)
    return {
      'precision':'FP64 promotion of captured FP32 point inputs; independent closest-feature implementation',
      'sample_count':samples,
      'minimum_sampled_distance_m':float(np.min(ds)),
      'minimum_refined_distance_m':min_d,
      'minimum_refined_time':min_t,
      'minimum_distance_is_lower_bound':False,
      'nominal_thickness_m':thick,'source103_effective_thickness_m':effective,
      'roundoff_addition_m':roundoff,
      'speed_bound_comparison':{
        'production_absolute_endpoint_sum_m':speed,
        'galilean_relative_lipschitz_bound_m':relative_speed,
        'absolute_to_relative_ratio':speed_ratio,
        'interpretation':'Mathematical relative-motion bound over exact promoted captured FP32 endpoints. Hypothesis only; no outward FP32 error budget, no production certificate, and no claim that changing the bound is safe.'},
      'fp64_conservative_advance_comparison':{'certified':certified,'hit':hit,'terminal':terminal,'speed_bound_m':speed,'iterations':trace},
    }

def load_capture(tag, directory, manifest_name, node_name, node_phase, primitive_name, primitive_phase, scheduler_name, scheduler_phase, candidate_name, candidate_phase):
    manifest_path=directory/manifest_name
    manifest,payload,arrs=read_capture(manifest_path)
    dispatch,da=array(arrs,'dispatch_buffer',node_phase,'<u4',(48,))
    dispatch_f=dispatch.view('<f4')
    max_exp=int(dispatch[12]); base_dt=np.float32(dispatch_f[43])
    scheduler,sa=array(arrs,scheduler_name,scheduler_phase,'<u4',(int(dispatch[3]),20))
    exp=min(int(scheduler[0,1]),max_exp)
    dt=np.float32(base_dt/np.float32(1<<exp))
    if int(np.asarray(dt).view('<u4')) != int(manifest['dt_seconds_f32_bits_u32']):
        raise ValueError(f'{tag}: derived object timestep differs from capture metadata')
    nodes,na=array(arrs,node_name,node_phase,'<f4',(int(manifest['node_count']),20))
    primitives,pa=array(arrs,primitive_name,primitive_phase,'<u4',(192,16))
    candidates,ca=array(arrs,candidate_name,candidate_phase,'<u4',(int(manifest['deformable_contact_capacity']),4))
    if candidates[283].tolist()!=list(PAIR):
        raise ValueError(f'{tag}: candidate 283 identity does not match expected pair')
    left,right=primitives[92],primitives[127]
    for index,primitive in ((92,left),(127,right)):
        if int(primitive[4]) != index: raise ValueError(f'{tag}: primitive identity mismatch {index}')
        if int(primitive[5]) != 0: raise ValueError(f'{tag}: primitive {index} is not source triangle kind 0')
        if int(primitive[3]) >= int(dispatch[3]): raise ValueError('primitive object out of range')
    node_ids=[left[:3].astype(int).tolist(),right[:3].astype(int).tolist()]
    if any(n<0 or n>=int(manifest['node_count']) for row in node_ids for n in row):
        raise ValueError(f'{tag}: node index out of range')
    if int(left[3]) != int(right[3]): raise ValueError(f'{tag}: pair spans different objects/timesteps')
    # Ensure this scalar diagnostic is exactly the pair reaching the native candidate.
    thick=np.float32(dispatch_f[44])
    return {
      'tag':tag,'manifest_path':manifest_path,'manifest':manifest,'payload_path':payload,'arrays':arrs,
      'nodes':nodes,'primitives':primitives,'node_ids':node_ids,'dt':dt,'thickness':thick,
      'input_array_hashes':{'nodes':na['sha256'],'primitives':pa['sha256'],'candidate':ca['sha256'],'dispatch':da['sha256'],'scheduler':sa['sha256']},
      'capture_array_names':{'nodes':node_name,'primitive':primitive_name,'candidate':candidate_name,'scheduler':scheduler_name},
    }

def build_cases(capture):
    nodes=capture['nodes'];left_ids,right_ids=capture['node_ids'];dt=capture['dt'];thick=capture['thickness']
    xyz=[]
    for ids in (left_ids,right_ids):
        side=[]
        for node in ids:
            finish=nodes[node,0:3].astype('<f4',copy=True)
            velocity=nodes[node,4:7].astype('<f4',copy=True)
            # Reproduce source103's FP32 FEM start expression componentwise.
            product=np.multiply(np.float32(dt),velocity,dtype='<f4')
            start=np.subtract(finish,product,dtype='<f4')
            side.append({'start':start,'finish':finish,'velocity':velocity})
        xyz.append(side)
    cases=[]
    for direction in range(2):
        vertex_side=direction; triangle_side=1-direction
        for vertex in range(3):
            points=[xyz[vertex_side][vertex],*xyz[triangle_side]]
            cases.append(make_case(capture,0,direction,vertex,
              f'{capture["tag"]}_vt_direction{direction}_vertex{vertex}',points))
    for edge_a,(a0,a1) in enumerate(EDGES):
        for edge_b,(b0,b1) in enumerate(EDGES):
            points=[xyz[0][a0],xyz[0][a1],xyz[1][b0],xyz[1][b1]]
            cases.append(make_case(capture,1,edge_a,edge_b,
              f'{capture["tag"]}_ee_edge{edge_a}_edge{edge_b}',points))
    return cases

def make_case(capture,kind,feature_a,feature_b,name,point_records):
    points0=np.asarray([r['start'] for r in point_records],dtype='<f4')
    points1=np.asarray([r['finish'] for r in point_records],dtype='<f4')
    header=np.array([kind,capture['thickness'],feature_a,feature_b],dtype='<f4')
    records=np.zeros((9,4),dtype='<f4')
    records[0]=header
    records[1:5,:3]=points0;records[5:9,:3]=points1
    return {
      'name':name,'capture':capture['tag'],'kind_id':kind,'kind':'vertex_triangle' if kind==0 else 'edge_edge',
      'feature_a':feature_a,'feature_b':feature_b,'thickness_m':float(capture['thickness']),
      'points_start_m':f32_json(points0),'points_start_f32_bits':bits_f32(points0),
      'points_finish_m':f32_json(points1),'points_finish_f32_bits':bits_f32(points1),
      'float4_records':records,'record_f32_bits':bits_f32(records),
      'primitive_pair':[92,127],'candidate_slot':283,'candidate_identity':[92,127,283,1],
      'primitive_nodes':capture['node_ids'],'object_index':int(capture['primitives'][92,3]),
      'dt_seconds_f32':float(capture['dt']),'dt_f32_bits':f'{int(np.asarray(capture["dt"]).view("<u4")):08x}',
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'prepared')
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    c=load_capture('center359',CENTER,'center-capture.coupled.json','fem_candidate_nodes',None,'surface_primitives',None,'scheduler_state',None,'deformable_contact_candidates',None)
    t=load_capture('trial358',TRIAL,'private-trial.coupled.json','fem_armijo_candidate_nodes','candidate_before_private_contact_rebuild','surface_primitives_private_trial','private_contact_rebuild_before_free_body_terms','schedulers','candidate_before_private_contact_rebuild','deformable_candidates_private_trial','private_contact_rebuild_before_free_body_terms')
    if c['manifest']['source_physics_fingerprint'] != t['manifest']['source_physics_fingerprint'] or c['manifest']['device_program_fingerprint'] != t['manifest']['device_program_fingerprint']:
        raise ValueError('center/trial source or device fingerprints differ')
    if (c['manifest']['root'],c['manifest']['iteration']) != (1201,7) or (t['manifest']['root'],t['manifest']['iteration'],t['manifest']['trial']) != (1201,7,23):
        raise ValueError('wrong captured Newton/trial selection')
    # Trial node state before contact rebuild must be bit-identical to the same
    # endpoint state after the capture point; the two surfaces should retain
    # source identities and node IDs.
    post_nodes,_=array(t['arrays'],'fem_armijo_candidate_nodes','private_contact_rebuild_before_free_body_terms','<f4',(36,20))
    if not np.array_equal(t['nodes'].view('<u4'),post_nodes.view('<u4')):
        raise ValueError('trial candidate nodes changed during private contact rebuild')
    accepted_surface,_=array(t['arrays'],'surface_primitives_accepted_before_rebuild','candidate_before_private_contact_rebuild','<u4',(192,16))
    if not np.array_equal(t['primitives'][[92,127],:8],accepted_surface[[92,127],:8]):
        raise ValueError('trial pair surface primitive IDs/node ownership differ before/after rebuild')
    cases=build_cases(c)+build_cases(t)
    if len(cases)!=30 or sum(row['kind_id']==0 for row in cases)!=12 or sum(row['kind_id']==1 for row in cases)!=18:
        raise ValueError('case matrix must contain 12 VT and 18 EE calls across two captures')
    # Store the exact 9xfloat4 input rows expected by the standalone Metal kernel.
    raw=b''.join(np.asarray(row['float4_records'],dtype='<f4').tobytes(order='C') for row in cases)
    binary=out/'cases.bin';binary.write_bytes(raw)
    # Independent FP64 geometric reference and FP64 conservative-advance comparison.
    for row in cases: row['cpu_fp64_reference']=cpu_reference(row)
    capture_inputs=[]
    for cap in (c,t):
      capture_inputs.append({
       'tag':cap['tag'],'manifest':str(cap['manifest_path']),'manifest_sha256':sha(cap['manifest_path']),
       'payload':str(cap['payload_path']),'payload_sha256':sha(cap['payload_path']),
       'physics_fingerprint':cap['manifest']['source_physics_fingerprint'],
       'device_program_fingerprint':cap['manifest']['device_program_fingerprint'],
       'root':cap['manifest']['root'],'iteration':cap['manifest']['iteration'],'trial':cap['manifest'].get('trial'),
       'input_array_hashes':cap['input_array_hashes'],'array_names':cap['capture_array_names'],
       'pair':[92,127],'slot':283,'candidate_identity':[92,127,283,1],
       'primitive_nodes':cap['node_ids'],'dt_seconds_f32':float(cap['dt']),
       'dt_seconds_f32_bits':f'{int(np.asarray(cap["dt"]).view("<u4")):08x}',
       'thickness_m_f32':float(cap['thickness']),'thickness_f32_bits':f'{int(np.asarray(cap["thickness"]).view("<u4")):08x}',
      })
    prepared={
      'schema':'numi.cardboard.source103_pair_ccd_inputs.v1',
      'scope':'zero-transaction standalone direct production CCD instrumentation; prepared only, no GPU dispatch',
      'source_snapshot':str(SOURCE),'source_contact_sha256':sha(SOURCE/'matter/src/metal/contact.metalinc'),
      'source_common_sha256':sha(SOURCE/'matter/src/metal/common.metalinc'),
      'source_main_shader_sha256':sha(SOURCE/'matter/src/matter_kernels.metal'),
      'source_build_receipt':str(ROOT/'analysis/retained-firstfatal-captures103-001/359/source-build.json'),
      'source_build_receipt_sha256':sha(ROOT/'analysis/retained-firstfatal-captures103-001/359/source-build.json'),
      'capture_inputs':capture_inputs,
      'case_contract':{'total':30,'captures':['center359','trial358'],'per_capture':{'vertex_triangle':6,'edge_edge':9},'float4_records_per_case':9,'input_record_bytes':16,'case_bytes':144,'total_input_bytes':len(raw),'input_binary':'cases.bin','input_sha256':sha(binary),'order':'center359 six VT (direction 0 then 1, vertex 0..2), nine EE (edgeA outer, edgeB inner), then same order for trial358'},
      'cases':[{k:v for k,v in row.items() if k!='float4_records'} for row in cases],
      'native_kernel_contract':{'function':'nm_cardboard_source103_pair_ccd_diagnostic','source_entry':'diagnostic_pair.metal','production_functions':['vertexTriangleCCD','edgeEdgeCCD'],'outputs_per_case_float4_records':9,'record_layout':['production hit/certified/toi/separation','production normal xyz + parameter0','production witness point xyz + parameter1','production parameter2 + reserved + reserved + reserved','trace clone hit/certified/toi/separation','trace clone normal xyz + parameter0','trace clone witness point xyz + parameter1','trace parameter2 + iteration count + terminal code + effective thickness','trace speed bound + reserved + reserved + reserved'],'trace_policy':'A diagnostic-only clone records conservative-advance iteration records. It is not production authority; compare its summary outputs bitwise to direct production outputs during the later native replay.'},
      'limitations':['No Metal/GPU execution has been performed.','FP64 geometric and conservative-advance values are independent references, not formal continuous-collision certificates.','Input endpoints reproduce the source-103 FP32 FEM start expression finish - dt*velocity.','No physical transaction or accepted state is modified.'],
    }
    (out/'prepared.json').write_text(json.dumps(prepared,indent=2,allow_nan=False)+'\n')
    print('source103_pair_input_preparation=PASS')
    print(f'cases={len(cases)} vt={sum(r["kind_id"]==0 for r in cases)} ee={sum(r["kind_id"]==1 for r in cases)}')
    print(f'cases_bin_sha256={sha(binary)} bytes={len(raw)}')
    print(f'prepared_manifest_sha256={sha(out/"prepared.json")}')

if __name__=='__main__': main()
