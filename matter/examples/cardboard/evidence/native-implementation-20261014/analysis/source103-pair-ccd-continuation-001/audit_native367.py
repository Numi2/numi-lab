#!/usr/bin/env python3
"""Independent CPU adjudication of source-103 native CCD diagnostic 367."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path('/Users/home/cardboard-implementation-20261014')
PACKAGE = ROOT / 'analysis/source103-pair-ccd-continuation-001'
NATIVE = ROOT / 'analysis/source103-pair-ccd-native-367'

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def strict_json(path: Path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def raw_bits(rows, label):
    if not isinstance(rows, list):
        raise ValueError(f'{label}: missing raw records')
    out=[]
    for i,row in enumerate(rows):
        if not isinstance(row,list) or len(row)!=4:
            raise ValueError(f'{label}[{i}]: bad float4 shape')
        words=[]
        for word in row:
            if not isinstance(word,str) or len(word)!=8:
                raise ValueError(f'{label}[{i}]: malformed float32 word')
            words.append(int(word,16))
        out.append(words)
    return out

def verify_native_result(result):
    cases=result['cases']
    if len(cases)!=30:
        raise ValueError('native result case count changed')
    mismatches=[]
    uncertified=[]
    by_kind={'vertex_triangle':{'count':0,'certified':0},
             'edge_edge':{'count':0,'certified':0}}
    for row in cases:
        prod=row['production_result']; clone=row['trace_clone_result']
        a=raw_bits(prod['raw_float4_bits'],row['name']+':production')
        b=raw_bits(clone['raw_float4_bits'],row['name']+':clone')
        equal=(a[:3]==b[:3] and a[3][0]==b[3][0])
        if equal != row.get('production_and_trace_clone_bitwise_equal'):
            raise ValueError(f'{row["name"]}: host bitwise result flag disagrees')
        if not equal: mismatches.append(row['name'])
        bucket=by_kind[row['kind']]
        bucket['count']+=1
        bucket['certified']+=int(bool(prod['certified']))
        if not prod['certified']:
            uncertified.append(row)
    if mismatches:
        raise ValueError(f'direct/clone mismatches: {mismatches}')
    if result.get('production_vs_trace_clone_all_bitwise_equal') is not True:
        raise ValueError('overall direct/clone flag not true')
    if result.get('physical_transactions_executed') != 0:
        raise ValueError('diagnostic unexpectedly executed physical transaction')
    return cases,by_kind,uncertified

def dot(a,b):
    return math.fsum(float(x)*float(y) for x,y in zip(a,b))

def sub(a,b): return [float(x)-float(y) for x,y in zip(a,b)]
def add(a,b): return [float(x)+float(y) for x,y in zip(a,b)]
def scale(a,s): return [float(x)*float(s) for x in a]

def closest_segments(p0,p1,q0,q1):
    u=sub(p1,p0); v=sub(q1,q0); w=sub(p0,q0)
    aa,cc=dot(u,u),dot(v,v); bb=dot(u,v); dd=dot(u,w); ee=dot(v,w)
    if aa == 0.0 and cc == 0.0:
        s=t=0.0
    elif aa == 0.0:
        s=0.0; t=min(1.0,max(0.0,ee/cc))
    elif cc == 0.0:
        t=0.0; s=min(1.0,max(0.0,-dd/aa))
    else:
        den=aa*cc-bb*bb
        s=min(1.0,max(0.0,(bb*ee-dd*cc)/den)) if den > 0.0 else 0.0
        t=(bb*s+ee)/cc
        if t<0.0:
            t=0.0; s=min(1.0,max(0.0,-dd/aa))
        elif t>1.0:
            t=1.0; s=min(1.0,max(0.0,(bb-dd)/aa))
    pp=add(p0,scale(u,s)); qq=add(q0,scale(v,t))
    delta=sub(pp,qq)
    return math.sqrt(dot(delta,delta)),s,t

def fp64_distance_trace(case, cap):
    x0=np.asarray(case['points_start_m'],dtype=np.float32).astype(np.float64).reshape(4,3)
    x1=np.asarray(case['points_finish_m'],dtype=np.float32).astype(np.float64).reshape(4,3)
    # Preserve captured endpoint identity: JSON values must round back to the
    # source FP32 words before promotion to the independent CPU calculation.
    if [f'{int(x):08x}' for x in x0.astype('<f4').view('<u4').reshape(-1)] != case['points_start_f32_bits']:
        raise ValueError('start endpoint JSON does not round-trip to captured FP32 words')
    if [f'{int(x):08x}' for x in x1.astype('<f4').view('<u4').reshape(-1)] != case['points_finish_f32_bits']:
        raise ValueError('finish endpoint JSON does not round-trip to captured FP32 words')
    eff=float(case['trace_clone_result']['effective_thickness_m'])
    speed=float(case['trace_clone_result']['speed_bound_m'])
    if not (math.isfinite(eff) and math.isfinite(speed) and speed>0.0):
        raise ValueError('selected case lacks finite positive bound/thickness')
    trace=[]; time=0.0; hit_index=None
    eps32=1.1920928955078125e-7
    for iteration in range(cap):
        points=x0 + time*(x1-x0)
        distance,s,t=closest_segments(points[0],points[1],points[2],points[3])
        trace.append({'iteration':iteration,'time':time,'distance_m':distance,
                      'gap_to_effective_thickness_m':distance-eff,'parameter0':s,'parameter1':t})
        if distance <= eff:
            hit_index=iteration
            break
        time += max((distance-eff)/speed,1.0e-6)
        if time > 1.0 + 8.0*eps32:
            break
    return {'evaluations':len(trace),'hit_candidate_index':hit_index,
            'iteration_exhausted':(len(trace)==cap and hit_index is None),
            'terminal_time_after_last_evaluation':time,'trace':trace}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--native-result',type=Path,default=NATIVE/'native-result.json')
    ap.add_argument('--receipt',type=Path,default=NATIVE/'receipt.json')
    ap.add_argument('--output',type=Path,default=PACKAGE/'native367-review.json')
    args=ap.parse_args()
    result=strict_json(args.native_result)
    receipt=strict_json(args.receipt)
    cases,counts,uncertified=verify_native_result(result)
    if result.get('source_snapshot') != str(ROOT/'snapshots/source-whole-mortar-proof-103'):
        raise ValueError('native result is not source-103 bound')
    if result.get('source_contact_sha256') != '427aca061ad2c6d2218b5eae0bc7620132a1ef9fff9f688390ccc96ca8a27ffc':
        raise ValueError('native result contact source binding mismatch')
    if receipt.get('commands',{}).get('native',{}).get('exit_code') != 0 or receipt.get('physical_transactions') != 0:
        raise ValueError('native receipt does not record successful zero-transaction diagnostic')
    if len(uncertified)!=1:
        raise ValueError(f'expected one uncertified result, found {len(uncertified)}')
    case=uncertified[0]
    if case['name']!='trial358_ee_edge0_edge0' or case['kind']!='edge_edge':
        raise ValueError(f'first uncertified result is not trial EE(0,0): {case["name"]}')
    if case['candidate_identity'] != [92,127,283,1] or case['primitive_pair'] != [92,127]:
        raise ValueError('uncertified case identity does not match captured failure')
    if case['production_result']['hit'] or case['production_result']['certified']:
        raise ValueError('selected direct production result is not unresolved')
    if case['trace_clone_result']['iteration_count'] != 17 or case['trace_clone_result']['terminal_code'] != 6:
        raise ValueError('selected baseline did not exhaust the 17-evaluation EE loop')
    trace=case['native_conservative_advance_trace']
    if len(trace)!=17 or any(row['iteration']!=i for i,row in enumerate(trace)):
        raise ValueError('native production trace is not the expected 17 contiguous samples')
    prod=case['production_result']; eff=case['trace_clone_result']['effective_thickness_m']
    gap=float(prod['separation_m'])-float(eff)
    candidate=fp64_distance_trace(case,33)
    if candidate['hit_candidate_index'] is None:
        raise ValueError('captured-input FP64 sample recurrence did not enter the effective band within 33 evaluations')
    ee01=next(r for r in cases if r['name']=='trial358_ee_edge0_edge1')
    report={
      'schema':'numi.cardboard.source103_native367_ee_exhaustion_review.v1',
      'status':'PASS',
      'source_snapshot':result['source_snapshot'],
      'bindings':{
        'native_result_sha256':sha(args.native_result),
        'native_receipt_sha256':sha(args.receipt),
        'native_audit_sha256':sha(NATIVE/'native-audit.json'),
        'prepared_manifest_sha256':sha(PACKAGE/'prepared/prepared.json'),
        'cases_bin_sha256':sha(PACKAGE/'prepared/cases.bin'),
        'source_contact_sha256':result['source_contact_sha256'],
      },
      'native_summary':{
        'case_count':len(cases),'direct_clone_bitwise_mismatches':0,
        'certification_by_kind':counts,
        'uncertified_case_count':len(uncertified),
        'physical_transactions':0,
      },
      'first_uncertified':{
        'case':case['name'],'query_kind':'edge_edge','edge_indices':[0,0],
        'primitive_pair':[92,127],'candidate_identity':[92,127,283,1],
        'root':1201,'newton_iteration':7,'line_search_trial':23,
        'production_hit':False,'production_certified':False,
        'native_iteration_count':17,'native_terminal_code':6,
        'last_native_sample':trace[-1],
        'effective_thickness_m':eff,'production_separation_m':prod['separation_m'],
        'distance_gap_m':gap,'distance_gap_nm':gap*1.0e9,
        'classification':'EE conservative-advance iteration exhaustion; not a certified miss and not itself a physical collision conclusion',
      },
      'neighboring_feature_control':{
        'case':ee01['name'],'hit':ee01['production_result']['hit'],
        'certified':ee01['production_result']['certified'],
        'toi':ee01['production_result']['toi'],
      },
      'fp64_captured_endpoint_recurrence':{
        'cap17':fp64_distance_trace(case,17),
        'cap33':candidate,
        'interpretation':'Independent FP64 closest-segment recurrence over endpoints round-tripped from captured FP32. It reaches the effective band by sample 28, but omits the production swept-axis checks and FP32 arithmetic; this supports only a bounded continuation experiment, not a native certification or physical result.',
      },
      'adjudication':'The only native uncertified pair query is EE(0,0); every VT query and all other pair queries certify. A continuation overlay may safely provide more conservative-advance evaluations while retaining the exact speed bound, thickness, distance, separating-axis checks, minimum step, and fail-closed exhaustion. Longer budget must never turn exhaustion into a miss.',
      'limitations':['Native source-103 baseline still exhausts at 17 evaluations.',
                    'No native run of the continuation overlay is included.',
                    'The FP64 recurrence is diagnostic and is not a formal CCD certificate.',
                    'No physical transactions were executed.'],
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'367_review=PASS output={args.output}')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'367_review=FAIL {exc}',file=sys.stderr)
        sys.exit(1)
