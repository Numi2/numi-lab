#!/usr/bin/env python3
"""Compare exact typed-bank pair inventories; no admission is implied."""
import argparse, hashlib, json
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def inventory(path,candidate_sha):
    d=json.loads(Path(path).read_text())
    if d.get('candidate_sha256')!=candidate_sha or d.get('complete') is not True or d.get('expected_bank_count')!=185 or d.get('bank_count')!=185:
        raise ValueError('audit incomplete or candidate mismatch')
    rows=d.get('banks',[])
    result={}
    for b in rows:
        key=(int(b['body']),int(b['stable_bone_id']))
        if key in result:raise ValueError('duplicate bank')
        value=b.get('result')
        if value is None:
            if int(b.get('skin_aabb_active',0)) and int(b.get('bank_aabb_active',0)):raise ValueError('missing active bank')
            pairs=set()
        else:
            raw=value['triangle_pairs']
            pairs={tuple(map(int,p)) for p in raw}
            if len(pairs)!=len(raw) or len(raw)!=int(value['count']):raise ValueError('pair inventory mismatch')
        result[key]=pairs
    if len(result)!=185:raise ValueError('wrong bank count')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('baseline-report','baseline-candidate','trial-report','trial-candidate','output'):
        p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('refusing overwrite')
    b=inventory(a.baseline_report,sha(a.baseline_candidate));t=inventory(a.trial_report,sha(a.trial_candidate))
    if set(b)!=set(t):raise ValueError('typed banks changed')
    rows=[]
    for k in sorted(b):
        gone=sorted(b[k]-t[k]);added=sorted(t[k]-b[k])
        if gone or added or k in ((43,16),(109,114),(65,99)):
            rows.append({'bank':list(k),'before':len(b[k]),'after':len(t[k]),'removed_pairs':len(gone),'added_pairs':len(added),'removed_ids':gone,'added_ids':added})
    result={'status':'DIAGNOSTIC_COMPARISON_ONLY','baseline_candidate_sha256':sha(a.baseline_candidate),'baseline_report_sha256':sha(a.baseline_report),
      'trial_candidate_sha256':sha(a.trial_candidate),'trial_report_sha256':sha(a.trial_report),
      'before_total':sum(map(len,b.values())),'after_total':sum(map(len,t.values())),'banks':rows,
      'native_admission':False,'accepted_physical_steps':0,'limitation':'Counts from a trial with known skin self-intersections; not an admissible geometry or containment result.'}
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'before':result['before_total'],'after':result['after_total'],'banks':[{'bank':x['bank'],'before':x['before'],'after':x['after'],'removed':x['removed_pairs'],'added':x['added_pairs']} for x in rows]}))
if __name__=='__main__':main()
