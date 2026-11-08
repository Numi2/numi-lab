"""Descriptive accepted prefixes; failures remain invalid for qualification."""
from pathlib import Path
import csv,json,math
R=Path.cwd();B=R/'build/crow-beak-hold-build-01';A=R/'artifacts/crow-rl-beak-hold-capture-20261108'
results={};raw={}
for n in ('clamp','open','no-contact','zero-friction','chunk-eight','refinement'):
    rows=list(csv.DictReader((B/f'repair-{n}/trace.tsv').open(),delimiter='\t'));raw[n]=rows
    lanes=[]
    for lane in range(4):
        r=[{k:float(v) for k,v in row.items()} for row in rows if int(row['lane'])==lane]
        assert all(math.isfinite(v) for row in r for v in row.values())
        hold=[row for row in r if 64<=row['tick']<=95]
        anchor=next((row for row in r if row['tick']==31),None)
        metrics={'accepted_trace_samples':len(r),'last_published_tick':r[-1]['tick'],
            'minimum_z_m':min(x['z'] for x in r),'maximum_z_m':max(x['z'] for x in r),
            'minimum_upper_gap_m':min(x['upper_gap_m'] for x in r),'minimum_lower_gap_m':min(x['lower_gap_m'] for x in r)}
        if hold and anchor:
            metrics.update(minimum_lift_ticks64_95_m=min(x['z']-.266850 for x in hold),
                maximum_head_relative_slip_from_tick31_m=max(math.sqrt(sum((x['rel_'+a]-anchor['rel_'+a])**2 for a in 'xyz')) for x in hold),
                opposed_positive_normal_samples=sum(x['upper_Jn']>1e-9 and x['lower_Jn']>1e-9 for x in hold),
                sampled_hold_ticks=len(hold),
                primary_hold_prefix_pass=min(x['z']-.266850 for x in hold)>=.020 and max(math.sqrt(sum((x['rel_'+a]-anchor['rel_'+a])**2 for a in 'xyz')) for x in hold)<=.005 and sum(x['upper_Jn']>1e-9 and x['lower_Jn']>1e-9 for x in hold)>=16)
        lanes.append(metrics)
    assert all({k:v for k,v in row.items() if k!='lane'}=={k:v for k,v in rows[(i//4)*4].items() if k!='lane'} for i,row in enumerate(rows))
    results[n]={'lanes':lanes,'identical_lanes':True,'qualified':False}
matched={row['tick']+':'+row['lane']:row for row in raw['clamp']}
partition=all(row==matched.get(row['tick']+':'+row['lane']) for row in raw['chunk-eight'])
out={'schema':'crow-beak-contact-accepted-prefix-description-v1','results':results,
    'chunk_one_vs_eight_accepted_sampled_prefix_fields_exact':partition,
    'capability_qualified':False,'prediction_verdict':'inconclusive due native rejections; accepted clamp hold window passed its hold-only predicates, with release unqualified',
    'limits':'Descriptive prefixes only. Microstep impulse samples are not integrated forces. Four identical lanes are not independent replicates. Final checkpoints and completed release are absent. No change to production mechanics or policies.'}
(A/'partial-metrics.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'clamp_lane0':results['clamp']['lanes'][0],'sampled_prefix_partition_exact':partition,'qualified':False}))
