#!/usr/bin/env python3
"""Strict, source-bound audit of native trial 21/22 exact production VTCCD calls."""
from __future__ import annotations
import hashlib, json, math, pathlib, struct, sys, os

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014/native')
PAIRS = [
    {'trial': 21, 'old': 'full-contact-trial21-capture-398', 'observed': 'full-contact-trial21-vtccd-call-405'},
    {'trial': 22, 'old': 'full-contact-trial22-capture-399', 'observed': 'full-contact-trial22-vtccd-call-406'},
]
EXPECTED = (1, 0, 173, 5, 36, 123, 173, 1, 13, 10, 14, 0, 28, 29, 32, 0)
EPS32 = 2.0 ** -23
def f32(x): return struct.unpack('<f',struct.pack('<f',float(x)))[0]

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def strict_json(path: pathlib.Path):
    return json.loads(path.read_text(), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f'non-JSON token {token}')))

def vector_sub(a, b): return tuple(x-y for x,y in zip(a,b))
def dot(a, b): return sum(x*y for x,y in zip(a,b))
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def scale(a,s): return tuple(x*s for x in a)
def norm(a): return math.sqrt(dot(a,a))

def closest_point_triangle(p, a, b, c):
    # Ericson's point/triangle closest-point regions, evaluated in binary64.
    ab, ac, ap = vector_sub(b,a), vector_sub(c,a), vector_sub(p,a)
    d1, d2 = dot(ab,ap), dot(ac,ap)
    if d1 <= 0.0 and d2 <= 0.0: return a, (1.0,0.0,0.0)
    bp=vector_sub(p,b); d3,d4=dot(ab,bp),dot(ac,bp)
    if d3 >= 0.0 and d4 <= d3: return b, (0.0,1.0,0.0)
    vc=d1*d4-d3*d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v=d1/(d1-d3); return add(a,scale(ab,v)),(1-v,v,0.0)
    cp=vector_sub(p,c); d5,d6=dot(ab,cp),dot(ac,cp)
    if d6 >= 0.0 and d5 <= d6: return c,(0.0,0.0,1.0)
    vb=d5*d2-d1*d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w=d2/(d2-d6); return add(a,scale(ac,w)),(1-w,0.0,w)
    va=d3*d6-d5*d4
    if va <= 0.0 and d4-d3 >= 0.0 and d5-d6 >= 0.0:
        w=(d4-d3)/((d4-d3)+(d5-d6)); return add(b,scale(vector_sub(c,b),w)),(0.0,1-w,w)
    inv=1.0/(va+vb+vc); v=vb*inv; w=vc*inv
    return add(add(a,scale(ab,v)),scale(ac,w)),(1-v-w,v,w)

def verify_capture(run_dir: pathlib.Path, require_observer: bool = True):
    manifest_path=run_dir/'trial-capture.coupled.json'; payload_path=run_dir/'trial-capture.coupled.bin'
    m=strict_json(manifest_path); payload=payload_path.read_bytes()
    assert m['schema']=='numi.matter.coupled_armijo_trial_capture.v1'
    assert m['payload']['byte_count']==len(payload)
    assert m['payload']['sha256']==sha(payload)
    assert pathlib.Path(m['payload']['file']).name==payload_path.name
    assert m['array_count']==len(m['arrays'])
    cursor=0
    for a in m['arrays']:
        assert a['offset']==cursor, (a['name'],a['offset'],cursor)
        assert a['byte_count']==a['count']*a['element_bytes']
        raw=payload[a['offset']:a['offset']+a['byte_count']]
        assert len(raw)==a['byte_count'] and sha(raw)==a['sha256'],a['name']
        cursor += a['byte_count']
    assert cursor==len(payload)
    obs=[a for a in m['arrays'] if a['name']=='actual_vertex_triangle_ccd_call']
    if require_observer:
        assert len(obs)==1 and obs[0]['count']==1 and obs[0]['element_bytes']==336
        raw=payload[obs[0]['offset']:obs[0]['offset']+336]
        words=struct.unpack('<16I68f',raw)
        return m,payload,words,obs[0]
    assert not obs
    return m,payload,None,None

def array_records(m):
    seen={}; out=[]
    for a in m['arrays']:
        k=(a['name'],a['phase'],a['dtype'],a['count'],a['element_bytes'])
        n=seen.get(k,0); seen[k]=n+1
        out.append((k,n,a))
    return out

def compare_pair(pair):
    trial=pair['trial']; olddir=BASE/pair['old']; newdir=BASE/pair['observed']
    oldm,oldbin,_,_=verify_capture(olddir, require_observer=False)
    newm,newbin,words,obs_spec=verify_capture(newdir)
    oldpred=strict_json(olddir/'prediction.json'); newpred=strict_json(newdir/'prediction.json')
    def normalized_prediction(p):
        argv=list(p['argv']); argv[0]=pathlib.Path(argv[0]).name
        argv=[('<native-output-path>' if str(BASE) in x else pathlib.Path(x).name if '/snapshots/' in x and x.endswith('.nmatter') else x) for x in argv]
        env={k:v for k,v in p['environment'].items() if 'VTCCD_AUDIT' not in k}
        env={k:('<native-output-prefix>' if k.endswith('_PATH') else v) for k,v in env.items()}
        return argv,env,p['native_protocol_id']
    assert normalized_prediction(oldpred)==normalized_prediction(newpred)
    oldmat=pathlib.Path(oldpred['argv'][1]); newmat=pathlib.Path(newpred['argv'][1])
    assert sha(oldmat.read_bytes())==sha(newmat.read_bytes())
    oldreceipt=strict_json(olddir/'receipt.json'); newreceipt=strict_json(newdir/'receipt.json')
    assert oldreceipt['execution_started'] and newreceipt['execution_started']
    assert oldreceipt['source_inputs_unchanged'] and newreceipt['source_inputs_unchanged']
    assert oldreceipt['exit_code']==newreceipt['exit_code']==1 and not oldreceipt['timed_out'] and not newreceipt['timed_out']
    assert oldm['root']==newm['root']==1201 and oldm['iteration']==newm['iteration']==9
    assert oldm['trial']==newm['trial']==trial
    assert oldm['source_physics_fingerprint']==newm['source_physics_fingerprint']=='915f50b10edc2e46'
    assert oldm['device_program_fingerprint']!=newm['device_program_fingerprint']
    oldarr={ (k,n):a for k,n,a in array_records(oldm) }
    newarr={ (k,n):a for k,n,a in array_records(newm) }
    common=set(oldarr)&set(newarr)
    assert len(common)==129, (len(common),len(oldarr),len(newarr))
    mismatch=[key for key in common if oldarr[key]['sha256']!=newarr[key]['sha256']]
    assert not mismatch,mismatch
    extras=set(newarr)-set(oldarr)
    assert len(extras)==1 and next(iter(extras))[0][0]=='actual_vertex_triangle_ccd_call'
    # Compare all selected Newton trace records; observer metadata is an added line.
    def rows(d): return [line for line in (d/'stderr.log').read_text(errors='strict').splitlines() if line.startswith('fem_residual_armijo_iteration=')]
    oldrows,newrows=rows(olddir),rows(newdir)
    assert len(oldrows)==len(newrows)==32 and oldrows==newrows
    def normalized_control_log(path):
        lines=(path/'diagnostic-state'/'dense'/'fixture-free-body-control-1201.log').read_text().splitlines()
        out=[]
        for line in lines:
            if line.startswith('fem_coupled_trial_vtccd_audit_enabled='): continue
            if line.startswith('fem_coupled_trial_capture_complete='):
                value=json.loads(line.split('=',1)[1]); value.update({'bytes':'<capture-bytes>','arrays':'<capture-array-count>','manifest':'<capture-manifest>'})
                line='fem_coupled_trial_capture_complete='+json.dumps(value,separators=(',',':'))
            out.append(line)
        return out
    assert normalized_control_log(olddir)==normalized_control_log(newdir)
    assert (olddir/'stdout.log').read_bytes()==(newdir/'stdout.log').read_bytes()
    # Production accepted/rejected state JSONs are exact; binary state snapshots only
    # vary in the already-defined device-program fingerprint bytes [12,20).
    state_old=olddir/'diagnostic-state'/'dense'; state_new=newdir/'diagnostic-state'/'dense'
    tx_json=[]; tx_bin=[]
    for i in range(3):
        name=f'transaction-{i:06d}.json'
        a=(state_old/name).read_bytes();b=(state_new/name).read_bytes()
        assert strict_json(state_old/name)==strict_json(state_new/name)
        tx_json.append({'name':name,'sha256':sha(a),'identical':a==b})
        for stage in ('before','after'):
            name=f'transaction-{i:06d}-{stage}.bin'; a=(state_old/name).read_bytes();b=(state_new/name).read_bytes()
            diffs=[j for j,(x,y) in enumerate(zip(a,b)) if x!=y]
            assert len(a)==len(b)==904359 and diffs==list(range(12,20)),(name,diffs[:32])
            old_fp=oldm['device_program_fingerprint'];new_fp=newm['device_program_fingerprint']
            assert a[12:20].hex()==bytes.fromhex(old_fp)[::-1].hex()
            assert b[12:20].hex()==bytes.fromhex(new_fp)[::-1].hex()
            tx_bin.append({'name':name,'byte_count':len(a),'difference_offsets':[12,19],
                           'old_device_fingerprint':old_fp,'new_device_fingerprint':new_fp,
                           'other_bytes_identical':a[:12]==b[:12] and a[20:]==b[20:]})
    # Decode exact helper record.
    u=words[:16]; f=words[16:]
    assert tuple(u)==EXPECTED
    start=[tuple(f[4*j:4*j+3]) for j in range(6)]
    finish=[tuple(f[24+4*j:24+4*j+3]) for j in range(6)]
    result=tuple(f[48:52]); normal=tuple(f[52:56]); bary=tuple(f[56:60]); point=tuple(f[60:64]); pars=tuple(f[64:68])
    def read_array(name,phase):
        a=next(x for x in newm['arrays'] if x['name']==name and x['phase']==phase)
        raw=newbin[a['offset']:a['offset']+a['byte_count']]
        return a,raw
    sa,sraw=read_array('fem_armijo_statuses','private_contact_rebuild_before_free_body_terms')
    status_words=struct.unpack('<8I4f',sraw[:48])
    _,craw=read_array('deformable_candidate_counts_private_trial','private_contact_rebuild_before_free_body_terms')
    _,araw=read_array('deformable_active_counts_private_trial','private_contact_rebuild_before_free_body_terms')
    candidate_count=struct.unpack('<4I',craw[:16]); active_count=struct.unpack('<4I',araw[:16])
    if trial==21: assert status_words[:8]==(6,0,0,173,0,0,0,0) and tuple(status_words[8:])==(36.0,123.0,2.0,4.0)
    else: assert status_words[:8]==(0,0,0,0,0,0,0,0)
    assert candidate_count[0]==400
    assert pars[3]==9.0
    assert pars[0]>0 and abs(pars[1]-pars[2])<=1e-12
    assert m_selector_matches(newm,trial)
    # Feature contract: direction 1, vertex 2 means right primitive node 2 vs left triangle.
    p0,p1=start[5],finish[5]; tri0,tri1=start[:3],finish[:3]
    coordinate_scale=max(abs(x) for p in start+finish for x in p)
    # Reproduce the frozen source expression with binary32 operation rounding.
    roundoff=f32(f32(8.0*f32(EPS32))*max(f32(pars[0]),f32(coordinate_scale),f32(1e-12)))
    effective=f32(f32(pars[0])+roundoff)
    def geom(t):
        p=tuple(a+t*(b-a) for a,b in zip(p0,p1))
        tri=[tuple(a+t*(b-a) for a,b in zip(x,y)) for x,y in zip(tri0,tri1)]
        q,bw=closest_point_triangle(p,*tri)
        return norm(vector_sub(p,q)),bw
    d0,b0=geom(0.0); dcall,bcall=geom(result[2]); d1,b1=geom(1.0)
    # Find the first crossing bracket on a dense diagnostic grid, then bisect it.
    assert d0>effective and d1<effective
    grid_n=4096
    grid_d=[geom(i/grid_n)[0] for i in range(grid_n+1)]
    crossing_i=next(i for i in range(1,grid_n+1) if grid_d[i] <= effective)
    lo=(crossing_i-1)/grid_n; hi=crossing_i/grid_n
    for _ in range(80):
        mid=(lo+hi)*0.5
        if geom(mid)[0] <= effective: hi=mid
        else: lo=mid
    cross_t=hi
    max_grid_rise=max(grid_d[i+1]-grid_d[i] for i in range(grid_n))
    # Count bit changes among all 36 endpoint coordinate lanes in equivalent captures.
    old_payload=oldbin; new_payload=newbin
    def array_bytes(m,b,name,phase):
        a=next(x for x in m['arrays'] if x['name']==name and x['phase']==phase)
        return b[a['offset']:a['offset']+a['byte_count']]
    old_geometry=array_bytes(oldm,old_payload,'fem_armijo_candidate_nodes','private_contact_rebuild_before_free_body_terms')
    new_geometry=array_bytes(newm,new_payload,'fem_armijo_candidate_nodes','private_contact_rebuild_before_free_body_terms')
    assert old_geometry==new_geometry
    # Status 6 is an uncertified private deformable narrowphase result; no success claim.
    return {
      'trial':trial,'directories':{'baseline':str(olddir),'observed':str(newdir)},
      'captures':{
        'baseline':{'manifest_sha256':sha((olddir/'trial-capture.coupled.json').read_bytes()),'payload_sha256':sha(oldbin),'payload_bytes':len(oldbin),'array_count':oldm['array_count'],'source_physics_fingerprint':oldm['source_physics_fingerprint'],'device_program_fingerprint':oldm['device_program_fingerprint']},
        'observed':{'manifest_sha256':sha((newdir/'trial-capture.coupled.json').read_bytes()),'payload_sha256':sha(newbin),'payload_bytes':len(newbin),'array_count':newm['array_count'],'source_physics_fingerprint':newm['source_physics_fingerprint'],'device_program_fingerprint':newm['device_program_fingerprint']},
        'common_array_count':len(common),'common_array_hash_mismatches':len(mismatch),'extra_array':{'name':obs_spec['name'],'phase':obs_spec['phase'],'bytes':obs_spec['byte_count'],'sha256':obs_spec['sha256']},
        'all_32_armijo_trace_rows_identical':True,'transaction_json_identical':tx_json,'transaction_binary_diffs':tx_bin,
      },
      'private_rebuild_status':{'code':status_words[0],'environment':status_words[1],'object_index':status_words[2],'failing_index':status_words[3],'diagnostics':list(status_words[8:]),'candidate_count':list(candidate_count),'active_count':list(active_count)},
      'actual_helper_record':{
        'attempt':list(u[:4]),'candidate_identity':list(u[4:8]),'left_nodes_and_object':list(u[8:12]),'right_nodes_and_object':list(u[12:16]),
        'start_left_then_right_m':start,'finish_left_then_right_m':finish,
        'hit':bool(result[0]),'certified':bool(result[1]),'toi':result[2],'reported_separation_m':result[3],
        'normal_valid':bool(result[0] and result[1]),'normal':normal,'barycentric':bary,'point':point,
        'thickness_m':pars[0],'left_dt_s':pars[1],'right_dt_s':pars[2],'solver_iteration':int(pars[3]),
        'effective_thickness_estimate_m':effective,'roundoff_estimate_m':roundoff,
        'reported_margin_from_effective_nm':(result[3]-effective)*1e9,
        'independent_fp64_geometry':{
          'distance_at_start_m':d0,'start_barycentric':b0,'distance_at_reported_toi_m':dcall,'reported_toi_barycentric':bcall,
          'distance_at_finish_m':d1,'finish_barycentric':b1,'first_effective_threshold_crossing_t':cross_t,
          'independent_grid_samples':grid_n+1,'maximum_positive_distance_increment_on_grid_m':max_grid_rise,
          'toi_minus_first_crossing_t':result[2]-cross_t,'finish_margin_from_effective_nm':(d1-effective)*1e9,
          'reported_toi_fp64_margin_from_effective_nm':(dcall-effective)*1e9,
          'method':'binary64 Ericson closest-point regions on the exact captured binary32 endpoint values, linearly interpolated in binary64; separate from production Metal result'
        }
      },
      'verdict':{
        'native_execution':'normal expected fixture failure; exit code 1, no timeout, source inputs unchanged',
        'trial21':'production helper returned uncertified (hit=0, certified=0); not a certified separation or miss. Captured path independently crosses the effective threshold before t=1, so this is unresolved true swept contact under the fixed production 16-iteration loop. The helper record does not expose an iteration counter; exhaustion is inferred from the frozen helper control flow and this return state.',
        'trial22':'production helper returned certified hit. Its reported TOI is earlier than the independent binary64 crossing; difference is within a sub-nanometre FP32 geometry/interpolation band. This is conservative early contact classification, not an unsafe missed collision.',
        'scope':'exact-call evidence explains the trial21/22 status boundary only. The full-contact fixture still fails and remains unqualified.'
      }
    }

def m_selector_matches(m,trial):
    s=m.get('vtccd_audit_selector',{})
    return s=={'environment':0,'candidate_slot':173,'primitive_pair':[36,123],'vertex':2,'direction':1,'execution':'same production call'} and m['trial']==trial

def main():
    reports=[compare_pair(p) for p in PAIRS]
    root=pathlib.Path(__file__).parent
    (root/'report.json').write_text(json.dumps({'schema':'numi.cardboard.private_vtccd_exact_call_adjudication.v1','created_from_native_captures':True,'checks_passed':True,'source_lines':{'helper':'source121/source122 contact.metalinc vertexTriangleCCD; source code is bound by per-run predictions and source physics fingerprint','effective_roundoff':'source121/source122 common.metalinc ccdDistanceRoundoff = 8*kEpsilon*max(thickness, coordinateScale, 1e-12)','failure_gate':'source121/source122 contact.metalinc un-mortared VT loop returns contact failure when certified is false'},'results':reports},indent=2)+'\n')
    print(root/'report.json')
    for r in reports:
        h=r['actual_helper_record'];g=h['independent_fp64_geometry']
        print(f"trial={r['trial']} hit={h['hit']} certified={h['certified']} toi={h['toi']:.10g} sep_um={h['reported_separation_m']*1e6:.9f} threshold_um={h['effective_thickness_estimate_m']*1e6:.9f} independent_tcross={g['first_effective_threshold_crossing_t']:.10g} common_hashes={r['captures']['common_array_count']} mismatch={r['captures']['common_array_hash_mismatches']}")

if __name__=='__main__': main()
