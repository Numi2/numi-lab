#!/usr/bin/env python3
"""CPU-only hash-bound recomputation of native 456/457 residual captures."""
from pathlib import Path
import csv, hashlib, json, math, struct

B = Path('/Users/home/cardboard-implementation-20261014')
PACKAGES = {
    '456': B / 'analysis/native-right-preconditioned-cgls-residual-replay-456',
    '457': B / 'analysis/native-block-inverse-cgls-residual-replay-457',
}
OUT = B / 'analysis/native456-right-preconditioning-adjudication-001'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_package(base):
    files = json.loads((base/'manifest.json').read_text())['files']
    errors = []
    for rel, meta in files.items():
        p = base/rel
        if not p.is_file() or p.stat().st_size != meta['bytes'] or sha(p) != meta['sha256']:
            errors.append(rel)
    if errors:
        raise ValueError(f'{base}: manifest mismatch: {errors[:5]}')
    pred = json.loads((base/'prediction.json').read_text())
    receipt = json.loads((base/'receipt.json').read_text())
    if receipt['physical_transactions'] != 0 or receipt['source_and_helper_inputs_unchanged'] is not True:
        raise ValueError(f'{base}: receipt invariants failed')
    if any(x['exit_code'] != 0 or x['timed_out'] for x in receipt['commands'].values()):
        raise ValueError(f'{base}: one or more calls did not complete successfully')
    return pred, receipt, len(files)

def parse_payload(path):
    raw = Path(path).read_bytes()
    magic,version,E,N,T,S,Q,fp,dt,reserved = struct.unpack_from('<8s6IQfI', raw, 0)
    if magic != b'NMFDSWP1' or version != 1 or (E,N,T,S,Q) != (2,882,1920,15,59):
        raise ValueError(f'{path}: unsupported replay payload layout')
    offsets={}; pos=48; nE=E*N; nT=E*T
    for name,count,size in [
        ('candidate',nE,80),('direction',nE,16),('rhs',nE,16),
        ('accepted_history',nT*S,4),('tangent',nT*81,4),
        ('tangent_valid',nT,4),('parameters',E*Q,4),
        ('captured_fresh_action',nE,16),('tet_audit',nT,88),('accepted_nodes',nE,80)]:
        end=pos+count*size; offsets[name]=raw[pos:end]; pos=end
    if pos != len(raw): raise ValueError(f'{path}: trailing or missing payload bytes')
    return {'sha256':sha(path),'bytes':len(raw),'dimensions':[E,N,T,S,Q],
            'world_fingerprint':fp,'dt':dt,'segments':offsets}

def summarize_sample(run_dir, row):
    k,side,alpha=int(row['sample']),int(row['side']),float(row['epsilon'])
    if row['status'] != [0,0] or row['history_unchanged'] is not True:
        raise ValueError(f'{run_dir}: failed sample status/history: {row}')
    np=run_dir/f'sample-{k}-{side}-nodes.bin'
    rp=run_dir/f'sample-{k}-{side}-residual.bin'
    fp=run_dir/f'sample-{k}-{side}-forces.bin'
    nd,rd,fd=np.read_bytes(),rp.read_bytes(),fp.read_bytes()
    if len(nd)!=2*882*80 or len(rd)!=2*882*16 or len(fd)!=2*1920*64:
        raise ValueError(f'{run_dir}: output record length mismatch for sample {k}/{side}')
    nodes=struct.unpack('<'+str(len(nd)//4)+'f',nd)
    residual=struct.unpack('<'+str(len(rd)//4)+'f',rd)
    forces=struct.unpack('<'+str(len(fd)//4)+'f',fd)
    if not all(math.isfinite(x) for x in nodes+residual+forces):
        raise ValueError(f'{run_dir}: nonfinite output at sample {k}/{side}')
    envs=[]; mask_words=[]
    for e in range(2):
        free_terms=[]; fixed_terms=[]; free=fixed=0; fixed_max=0.0
        for i in range(882):
            node=(e*882+i)*20
            constraint=nodes[node+11]  # NMFEMNodeStateGPU.restAndFixed.w
            is_fixed=(constraint==1.0 or constraint==3.0)
            mask_words.append(is_fixed)
            if constraint not in (0.0,1.0,3.0):
                raise ValueError(f'{run_dir}: unexpected constraint word {constraint}')
            rr=residual[(e*882+i)*4:(e*882+i)*4+3]
            sq=math.fsum(float(x)*float(x) for x in rr)
            if is_fixed:
                fixed+=1; fixed_terms.append(sq)
                fixed_max=max(fixed_max,*(abs(float(x)) for x in rr))
            else:
                free+=1; free_terms.append(sq)
        if (free,fixed)!=(846,36):
            raise ValueError(f'{run_dir}: unexpected free/fixed count {(free,fixed)}')
        envs.append({'free_nodes':free,'fixed_nodes':fixed,
                     'free_residual_l2_Ns':math.sqrt(math.fsum(free_terms)),
                     'all_rows_residual_l2_Ns':math.sqrt(math.fsum(free_terms)+math.fsum(fixed_terms)),
                     'fixed_residual_max_abs_Ns':fixed_max})
    return {'sample':k,'side':side,'alpha':alpha,'status':row['status'],
            'accepted_history_unchanged':row['history_unchanged'],'env0':envs[0],'env1':envs[1],
            'trial_nodes_sha256':sha(np),'residual_sha256':sha(rp),'element_forces_sha256':sha(fp),
            'constraint_mask_sha256':hashlib.sha256(bytes(mask_words)).hexdigest()}

def main():
    payload_hashes={}; run_summary={}; all_rows=[]; invariants={}; lineage={}
    all_inputs=[]
    for label,base in PACKAGES.items():
        pred,receipt,file_count=verify_package(base)
        if pred['schema']!='numi.cardboard.native_preconditioned_CGLS_residual_replay.v1':
            raise ValueError(f'{base}: schema mismatch')
        pld={}
        for name,cmd in pred['commands'].items():
            all_inputs.append((label,name,Path(cmd[2])))
            pld[name]=parse_payload(cmd[2])
        # Shared state/operator payloads must be identical; only CGLS direction is allowed to differ.
        keys=next(iter(pld.values()))['segments']
        segment_hashes={key:{hashlib.sha256(v['segments'][key]).hexdigest() for v in pld.values()} for key in keys}
        for key,h in segment_hashes.items():
            if key!='direction' and len(h)!=1:
                raise ValueError(f'{base}: non-direction input segment differs: {key}')
        if len(segment_hashes['direction'])<2:
            raise ValueError(f'{base}: no distinct proposed directions found')
        for name,p in pld.items():
            payload_hashes[f'{label}/{name}']=p['sha256']
        lineage[label]={'prediction_sha256':sha(base/'prediction.json'),'receipt_sha256':sha(base/'receipt.json'),
            'manifest_sha256':sha(base/'manifest.json'),'manifest_file_count':file_count,
            'source_build_sha256':pred['source125_build_sha256'],'helper_program_sha256':pred['helper_program_sha256'],
            'payload_dimensions':next(iter(pld.values()))['dimensions'],'world_fingerprint':next(iter(pld.values()))['world_fingerprint'],
            'dt_seconds':next(iter(pld.values()))['dt'],'non_direction_segment_sha256':{
                k:next(iter(v)) for k,v in segment_hashes.items() if k!='direction'},
            'direction_payload_sha256':{k:hashlib.sha256(v['segments']['direction']).hexdigest() for k,v in pld.items()},
            'physical_transactions':receipt['physical_transactions'],'source_inputs_unchanged':receipt['source_and_helper_inputs_unchanged'],
            'completed_native_and_preflight_calls':len(receipt['commands']),'timeouts':0}
        current={}
        for name in sorted(x.name for x in base.iterdir() if x.is_dir()):
            path=base/name
            rawrows=[json.loads(x) for x in (path/'residual-sweep.jsonl').read_text().splitlines()]
            if len(rawrows)!=15: raise ValueError(f'{path}: expected 15 alpha/sign rows, got {len(rawrows)}')
            got=[summarize_sample(path,r) for r in rawrows]
            # One zero-alpha row; seven positive and seven negative rows.
            if sum(r['side']==0 and r['alpha']==0 for r in got)!=1 or sum(r['side']==1 for r in got)!=7:
                raise ValueError(f'{path}: incomplete alpha/sign sweep')
            # Every sample must use the identical exact constraint mask.
            if len({r['constraint_mask_sha256'] for r in got})!=1:
                raise ValueError(f'{path}: constraint mask changed across samples')
            # This direction is env0-only; env1 nodes/residual stay byte-identical.
            env1_node_hash={r['trial_nodes_sha256'] for r in got}
            env1_res_hash={r['residual_sha256'] for r in got}
            # Whole buffers include env0; direct env1 slice equality is checked below.
            base_nodes=(path/'sample-0-0-nodes.bin').read_bytes()
            base_res=(path/'sample-0-0-residual.bin').read_bytes()
            base_n1=base_nodes[882*80:]; base_r1=base_res[882*16:]
            for r in rawrows:
                nd=(path/f"sample-{r['sample']}-{r['side']}-nodes.bin").read_bytes()
                rd=(path/f"sample-{r['sample']}-{r['side']}-residual.bin").read_bytes()
                if nd[882*80:]!=base_n1 or rd[882*16:]!=base_r1:
                    raise ValueError(f'{path}: env1 unexpectedly changed')
            for r in got:
                out={'package':label,'run':name,**r}
                all_rows.append(out)
            baseline=next(r['env0']['free_residual_l2_Ns'] for r in got if r['alpha']==0 and r['side']==0)
            positive=[r for r in got if r['side']==0 and r['alpha']>0]
            negative=[r for r in got if r['side']==1]
            best=min(positive,key=lambda r:r['env0']['free_residual_l2_Ns'])
            current[name]={'center_env0_l2_Ns':baseline,
                'positive_l2_Ns_by_alpha':{format(r['alpha'],'.8g'):r['env0']['free_residual_l2_Ns'] for r in positive},
                'negative_l2_Ns_by_alpha':{format(r['alpha'],'.8g'):r['env0']['free_residual_l2_Ns'] for r in negative},
                'best_tested_positive_alpha':best['alpha'],'best_positive_l2_Ns':best['env0']['free_residual_l2_Ns'],
                'best_positive_fractional_reduction_from_center':1-best['env0']['free_residual_l2_Ns']/baseline,
                'alpha1_positive_l2_Ns':next(r['env0']['free_residual_l2_Ns'] for r in positive if r['alpha']==1.0),
                'alpha_half_positive_l2_Ns':next(r['env0']['free_residual_l2_Ns'] for r in positive if r['alpha']==0.5),
                'alpha_quarter_positive_l2_Ns':next(r['env0']['free_residual_l2_Ns'] for r in positive if r['alpha']==0.25),
                'alpha2_positive_l2_Ns':next(r['env0']['free_residual_l2_Ns'] for r in positive if r['alpha']==2.0),
                'all_native_statuses_zero':all(r['status']==[0,0] for r in got),
                'all_history_markers_true':all(r['accepted_history_unchanged'] for r in got),
                'env1_buffer_unchanged':True,
                'all_sample_masks_846free_36fixed':True}
        run_summary[label]=current
        invariants[label]={'native_and_preflight_exit_codes_all_zero':True,'timeouts':0,'physical_transactions':0,
            'source_inputs_unchanged':True,'samples_per_run':15,'runs':len(current),'all_sample_status_codes_zero':True,
            'accepted_history_unchanged_all_samples':True,'fixed_mask':'restAndFixed.w == 1.0f or 3.0f','free_nodes_per_environment':846,'fixed_nodes_per_environment':36}
    # Compare every common non-direction segment across both packages.
    combined={}
    for lab,name,path in all_inputs:
        parsed=parse_payload(path)
        for key,data in parsed['segments'].items():
            combined.setdefault(key,set()).add(hashlib.sha256(data).hexdigest())
    for key,h in combined.items():
        if key!='direction' and len(h)!=1: raise ValueError(f'cross-package input mismatch in {key}')
    # Alpha-zero residual/node state must match across all methods/packages.
    center_res={}; center_nodes={}
    for lab,base in PACKAGES.items():
        for name in sorted(x.name for x in base.iterdir() if x.is_dir()):
            center_res[f'{lab}/{name}']=sha(base/name/'sample-0-0-residual.bin')
            center_nodes[f'{lab}/{name}']=sha(base/name/'sample-0-0-nodes.bin')
    if len(set(center_res.values()))!=1 or len(set(center_nodes.values()))!=1:
        raise ValueError('alpha-zero native baseline differs across replay inputs')
    summary={'schema':'numi.cardboard.native456_nonlinear_cgls_adjudication.v1','packages':lineage,
        'input_payload_sha256':payload_hashes,'input_common_segments_across_456_457':{k:next(iter(v)) for k,v in combined.items() if k!='direction'},
        'direction_segment_unique_hash_count':len(combined['direction']),
        'zero_alpha_residual_sha256':next(iter(center_res.values())),
        'zero_alpha_nodes_sha256':next(iter(center_nodes.values())),
        'invariants':invariants,'runs':run_summary}
    # outputs
    with (OUT/'residuals.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['package','run','sample','side','alpha','status_env0','status_env1','accepted_history_unchanged','env0_free_nodes','env0_fixed_nodes','env0_free_l2_Ns','env0_all_l2_Ns','env0_fixed_max_abs_Ns','env1_free_l2_Ns','env1_all_l2_Ns','env1_fixed_max_abs_Ns','trial_nodes_sha256','residual_sha256','element_forces_sha256','constraint_mask_sha256'])
        for r in all_rows:
            a,b=r['env0'],r['env1'];w.writerow([r['package'],r['run'],r['sample'],r['side'],format(r['alpha'],'.9g'),r['status'][0],r['status'][1],str(r['accepted_history_unchanged']).lower(),a['free_nodes'],a['fixed_nodes'],format(a['free_residual_l2_Ns'],'.17g'),format(a['all_rows_residual_l2_Ns'],'.17g'),format(a['fixed_residual_max_abs_Ns'],'.17g'),format(b['free_residual_l2_Ns'],'.17g'),format(b['all_rows_residual_l2_Ns'],'.17g'),format(b['fixed_residual_max_abs_Ns'],'.17g'),r['trial_nodes_sha256'],r['residual_sha256'],r['element_forces_sha256'],r['constraint_mask_sha256']])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'runs':{k:{n:{'alpha1':v['alpha1_positive_l2_Ns'],'alpha0.5':v['alpha_half_positive_l2_Ns'],'alpha0.25':v['alpha_quarter_positive_l2_Ns'],'alpha2':v['alpha2_positive_l2_Ns'],'best_alpha':v['best_tested_positive_alpha'],'best_l2':v['best_positive_l2_Ns']} for n,v in d.items()} for k,d in run_summary.items()},'rows':len(all_rows),'out':str(OUT)},indent=2))

if __name__=='__main__': main()
