#!/usr/bin/env python3
"""Verify captured419 linear-conditioning package and source input bindings."""
import hashlib, json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
BASE=P.parent
m=json.loads((P/'manifest.json').read_text())
r=json.loads((P/'report.json').read_text())
source=Path(m['input_package'])
source_m=source/'manifest.json'
assert hashlib.sha256(source_m.read_bytes()).hexdigest()==m['input_package_manifest_sha256']
assert hashlib.sha256((source/'report.json').read_bytes()).hexdigest()==m['input_report_sha256']
for x in m['input_files']:
    assert hashlib.sha256((source/x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']
    assert r['source_capture_bindings']['input_hashes'][x['path']]==x['sha256'],x['path']
for x in m['artifacts']:
    q=P/x['path']
    assert q.is_file() and q.stat().st_size==x['bytes'],x['path']
    assert hashlib.sha256(q.read_bytes()).hexdigest()==x['sha256'],x['path']
npz=np.load(P/'spectral_compact.npz')
s=npz['singular_values']; b=npz['rhs_left_coefficients']; d=npz['captured_delta_right_coefficients']
assert s.shape==(2538,) and b.shape==(2538,) and d.shape==(2538,)
assert np.isfinite(s).all() and np.isfinite(b).all() and np.isfinite(d).all()
assert np.all(s[:-1]>=s[1:]) and s[-1]>0
assert r['matrix']['shape']==[2538,2538] and r['matrix']['nnz']==71046
assert r['spectrum']['rank_at_numpy_tolerance']==2538
assert r['source_capture_bindings']['capture_sha256']==json.loads((source/'report.json').read_text())['input_bindings']['capture_sha256']
assert hashlib.sha256((P/'manifest.json').read_bytes()).hexdigest()==(P/'manifest.sha256').read_text().split()[0]
print('PASS: artifact hashes, exact capture input/report hashes, SVD arrays, rank metadata, and package manifest digest')
