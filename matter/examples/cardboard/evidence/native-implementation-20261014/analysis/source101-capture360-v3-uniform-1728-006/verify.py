#!/usr/bin/env python3
"""Fail-closed integrity and gate verifier for output-006."""
import csv, hashlib, json, math, pathlib
HERE=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
MANIFEST=HERE/'manifest.json'
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    m=json.loads(MANIFEST.read_text())
    for rel,expect in m['files'].items():
        p=HERE/rel
        if not p.is_file() or sha(p)!=expect: raise SystemExit(f'file binding mismatch: {rel}')
    for name,item in m['inputs'].items():
        p=pathlib.Path(item['path'])
        if not p.is_file() or sha(p)!=item['sha256']: raise SystemExit(f'input binding mismatch: {name}')
    out=HERE/'output-006'; report=json.loads((out/'report.json').read_text())
    rows=[json.loads(x) for x in (out/'all1728_v3_rows.jsonl').read_text().splitlines() if x.strip()]
    ids=[int(r['tet_id']) for r in rows]
    expected_ids={int(r['tet_id']) for r in csv.DictReader(open(m['inputs']['directional_rows']['path'],newline=''))}
    if len(rows)!=1728 or len(set(ids))!=1728 or set(ids)!=expected_ids: raise SystemExit('row domain mismatch')
    counts={'rows':len(rows),'v3_all_center_and_endpoint_gates':0,'branch_stable':0,'within_1e-6':0,'exceeds_1e-6':0,'inconclusive':0}
    misses=[]
    for r in rows:
        c=next(x for x in r['continuation_v3']['epsilon_cases'] if x['target_deltaF_frobenius']==1e-7)
        if c['center_and_both_endpoints_v3_converged']: counts['v3_all_center_and_endpoint_gates']+=1
        if c['branch_signature_stable']: counts['branch_stable']+=1
        status=c['strict_1e-6_diagnostic']
        if status=='within_threshold_for_this_direction_and_step': counts['within_1e-6']+=1
        elif status=='exceeds_threshold_for_this_direction_and_step': counts['exceeds_1e-6']+=1; misses.append(int(r['tet_id']))
        else: counts['inconclusive']+=1
    if counts['rows']!=1728 or counts['v3_all_center_and_endpoint_gates']!=1728 or counts['branch_stable']!=1728:
        raise SystemExit('v3 projection or branch gate failed')
    if counts['within_1e-6']!=1727 or counts['exceeds_1e-6']!=1 or counts['inconclusive']!=0 or misses!=[1186]:
        raise SystemExit(f'unexpected threshold result: {counts}, misses={misses}')
    print(json.dumps({'status':'verified_bounded_diagnostic_with_one_threshold_exceedance','counts':counts,'exceed_tet_ids':misses},indent=2))
if __name__=='__main__': main()
