#!/usr/bin/env python3
"""Compare low/zero-capacity canonical rows over a shared processed prefix."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import hashlib
import numpy as np

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def canonical_rows(part_dir, meta, kind, prefix_end):
    with np.load(Path(part_dir)/meta['file'], allow_pickle=False) as z:
        offs=np.asarray(z['poly_offsets'],np.int64)
        tids=np.asarray(z['tet_ids'],np.int64)
        faces=np.asarray(z['face_ids'],np.int64)
        bary=np.asarray(z['polygon_barycentric'],np.float64)
        rows=np.asarray(z['insufficient_capacity_polygon_vertex_rows' if kind=='low' else 'zero_capacity_polygon_vertex_rows'],np.int64)
        if not len(rows): return []
        poly=np.searchsorted(offs[1:],rows,side='right')
        if np.any(poly>=len(tids)): raise ValueError(f'orphan {kind} rows in {meta["file"]}')
        if np.any(faces[poly]>=prefix_end): rows=rows[faces[poly]<prefix_end];poly=poly[faces[poly]<prefix_end]
        out=[]
        for r,p in zip(rows,poly):
            out.append((int(meta['bank_id']),int(faces[p]),int(tids[p]),np.asarray(bary[r],dtype='<f8').tobytes()))
        return out

def compare_runs(run009, run010, prefix_end):
    c9=json.loads((run009/'checkpoint.json').read_text())
    c10=json.loads((run010/'checkpoint.json').read_text())
    if int(c9['faces_processed'])!=prefix_end: raise ValueError('009 checkpoint is not the requested common prefix')
    if int(c10['faces_processed'])<prefix_end: raise ValueError('010 does not cover the full 009 prefix')
    p9=c9['parts'];p10=c10['parts']
    intervals9=[tuple(map(int,m['flat_face_range'])) for m in p9]
    intervals10=[]
    for m in p10:
        lo,hi=map(int,m['flat_face_range'])
        if lo<prefix_end:
            if hi>prefix_end: raise ValueError('010 part crosses common-prefix boundary; split comparator needed')
            intervals10.append((lo,hi))
    if intervals9!=intervals10: raise ValueError('processed part ranges differ, including zero-hit intervals')
    out={}
    for kind in ('low','zero'):
        a=Counter();b=Counter()
        for m in p9:
            a.update(canonical_rows(run009/'constraint-parts',m,kind,prefix_end))
        for m in p10:
            lo,hi=map(int,m['flat_face_range'])
            if lo<prefix_end: b.update(canonical_rows(run010/'constraint-parts',m,kind,prefix_end))
        missing=list((a-b).items());extra=list((b-a).items())
        out[kind]={
          'rows_009':sum(a.values()),'rows_010_prefix':sum(b.values()),
          'unique_009':len(a),'unique_010_prefix':len(b),
          'multiplicity_mismatches':sum(abs(n-(b[k])) for k,n in a.items() if b[k]!=n)+sum(b[k] for k in b.keys()-a.keys()),
          'missing_from_010':sum(n for _,n in missing),'extra_in_010':sum(n for _,n in extra),
          'sample_missing_keys':[{'bank_id':k[0],'face_id':k[1],'tet_id':k[2],'barycentric_f64_hex':k[3].hex(),'multiplicity':n} for k,n in missing[:5]],
          'sample_extra_keys':[{'bank_id':k[0],'face_id':k[1],'tet_id':k[2],'barycentric_f64_hex':k[3].hex(),'multiplicity':n} for k,n in extra[:5]],
          'exact_match':not missing and not extra,
        }
    return {
      'status':'PASS_EXACT_COMMON_PREFIX_ROWS' if all(out[k]['exact_match'] for k in out) else 'FAIL_COMMON_PREFIX_ROW_DIFFERENCE',
      'scope':'capacity-only pruning equivalence on common processed prefix; no full-domain or physical/native admission',
      'common_processed_face_prefix':[0,prefix_end],
      '009_checkpoint_sha256':sha256_file(run009/'checkpoint.json'),
      '010_checkpoint_sha256':sha256_file(run010/'checkpoint.json'),
      '009_parts':len(p9),'010_prefix_parts':len(intervals10),
      'zero_hit_face_ranges_covered':True,
      'comparisons':out,
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run009',type=Path,required=True);ap.add_argument('--run010',type=Path,required=True);ap.add_argument('--prefix-end',type=int,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();report=compare_runs(a.run009,a.run010,a.prefix_end)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    report['report_sha256']=sha256_file(a.output);print(json.dumps(report,indent=2,sort_keys=True))
    if report['status']!='PASS_EXACT_COMMON_PREFIX_ROWS':raise SystemExit(2)
if __name__=='__main__':main()
