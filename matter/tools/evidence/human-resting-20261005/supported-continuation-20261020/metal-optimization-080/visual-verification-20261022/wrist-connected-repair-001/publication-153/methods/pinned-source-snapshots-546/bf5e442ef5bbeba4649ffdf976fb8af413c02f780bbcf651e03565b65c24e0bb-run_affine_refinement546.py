#!/usr/bin/env python3
"""Hash-bound construction-only CLI for connected_refinement_adapter546."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ADAPTER_PATH = HERE / 'connected_refinement_adapter546.py'

def load_adapter():
    spec = importlib.util.spec_from_file_location('connected_refinement_adapter546_cli', ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load adapter {ADAPTER_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',required=True,type=Path)
    ap.add_argument('--candidate-sha256',required=True)
    ap.add_argument('--crossing-report',required=True,type=Path)
    ap.add_argument('--crossing-report-sha256',required=True)
    ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--freeze-face',type=int,action='append',default=[])
    ap.add_argument('--selection-exclusion-reason')
    a=ap.parse_args()
    if a.output_dir.exists():raise SystemExit(f'refuse to overwrite {a.output_dir}')
    pins={str(a.candidate):a.candidate_sha256,str(a.crossing_report):a.crossing_report_sha256,
          str(ADAPTER_PATH):sha(ADAPTER_PATH)}
    for path,expected in ((a.candidate,a.candidate_sha256),(a.crossing_report,a.crossing_report_sha256)):
        actual=sha(path)
        if actual!=expected:raise SystemExit(f'input SHA mismatch for {path}: {actual}')
    adapter=load_adapter()
    if pins[str(ADAPTER_PATH)]!=adapter.sha256_file(ADAPTER_PATH):raise SystemExit('adapter source changed before load')
    owner=adapter._load_owner270()
    pins[str(adapter.OWNER270)]=adapter.OWNER270_SHA256
    pins[str(owner.REFINER_PATH)]=owner.REFINER_SHA256
    pins[str(owner.P084_PATH)]=owner.P084_SHA256
    if any(sha(path)!=digest for path,digest in pins.items()):raise SystemExit('a pinned owner/input source changed before build')
    with np.load(a.candidate,allow_pickle=False) as z:source={k:z[k] for k in z.files}
    crossing=json.loads(a.crossing_report.read_text())
    required=('area_reference_037_points_world_m_f64','area_reference_037_source_O_faces')
    if any(k not in source for k in required):raise SystemExit('candidate lacks its matched 037 area reference')
    output,report=adapter.build_refined_arrays546(
        source,crossing,source[required[0]],source[required[1]],
        candidate_sha256=a.candidate_sha256,crossing_report_sha256=a.crossing_report_sha256,
        area_reference_sha256=a.candidate_sha256,
        frozen_face_rows=a.freeze_face,
        selection_exclusion_reason=a.selection_exclusion_reason)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    out_npz=a.output_dir/'candidate.npz';out_report=a.output_dir/'report.json'
    with out_npz.open('xb') as f:np.savez_compressed(f,**output)
    report.update({
        'input_candidate_path':str(a.candidate),'input_candidate_sha256':a.candidate_sha256,
        'crossing_report_path':str(a.crossing_report),'crossing_report_sha256':a.crossing_report_sha256,
        'area_reference_embedded_in_candidate_path':str(a.candidate),
        'area_reference_embedded_candidate_sha256':a.candidate_sha256,
        'output_candidate_path':str(out_npz),'output_candidate_sha256':sha(out_npz),
        'output_report_path':str(out_report),'native_admission':False,'accepted_physical_steps':0,
        'construction_only':True,
    })
    out_report.write_text(json.dumps(report,indent=2,allow_nan=False)+chr(10))
    for path,digest in pins.items():
        if sha(path)!=digest:raise SystemExit(f'pinned input/source changed during build: {path}')
    print(json.dumps({'status':report['status'],'output_candidate':str(out_npz),
                      'output_candidate_sha256':report['output_candidate_sha256'],
                      'output_report':str(out_report),'new_nodes':report['new_node_count']-report['old_node_count'],
                      'new_faces':report['new_face_count']-report['old_face_count'],
                      'native_admission':False,'accepted_physical_steps':0},indent=2))
if __name__=='__main__':main()
