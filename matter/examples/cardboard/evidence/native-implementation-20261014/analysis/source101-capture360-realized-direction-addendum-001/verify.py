#!/usr/bin/env python3
"""Verify exact bindings and row-level contract for the realized-direction addendum."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()
def need(ok,msg):
 if not ok: raise SystemExit('FAIL: '+msg)
def main():
 manifest=json.loads((HERE/'manifest.json').read_text())
 need(manifest['schema']=='numi.cardboard.source101-realized-direction-addendum.v1','manifest schema')
 for key,item in manifest['inputs'].items():
  p=Path(item['path']);need(p.is_file(),f'missing input {key}');need(sha(p)==item['sha256'],f'input SHA {key}')
 for key,item in manifest['artifacts'].items():
  p=HERE/item['path'];need(p.is_file(),f'missing artifact {key}');need(sha(p)==item['sha256'],f'artifact SHA {key}')
 rows=[json.loads(s) for s in (HERE/'output-001/realized_direction_rows.jsonl').read_text().splitlines() if s.strip()]
 report=json.loads((HERE/'output-001/report.json').read_text())
 need(report['status']=='completed' and report['selection']['rows_completed']==1728,'completed full row set')
 need(len(rows)==1728 and len({r['tet_id'] for r in rows})==1728,'unique 1728 rows')
 need(all(r['center_and_endpoints_v3_converged'] and r['captured_branch_signature_stable'] for r in rows),'source v3 projection/branch admission')
 need(all(math.isfinite(r['fd_vs_nominal_tangent_relative_l2']) and math.isfinite(r['fd_vs_realized_tangent_relative_l2']) for r in rows),'finite errors')
 need(sum(r['within_1e-6_nominal'] for r in rows)==1727,'nominal count remains package006 result')
 need(sum(r['within_1e-6_realized'] for r in rows)==1728,'realized-direction diagnostic count')
 need(report['metrics']['projection_and_recorded_branch_stable_count']==1728,'projection and recorded branch count')
 need(report['metrics']['fd_vs_nominal_tangent']['max_tet_id']==1186,'nominal max tet')
 need(report['metrics']['fd_vs_realized_tangent']['max_tet_id']==1515,'realized max tet')
 t=next(r for r in rows if r['tet_id']==1186)
 need(abs(t['fd_vs_nominal_tangent_relative_l2']-1.214904185971578e-6)<1e-15,'tet1186 nominal metric')
 need(abs(t['fd_vs_realized_tangent_relative_l2']-2.5607600684748406e-7)<1e-15,'tet1186 realized metric')
 need(manifest['verdict']['nominal_gate_replaced'] is False,'nominal gate must remain unchanged')
 need(manifest['verdict']['universal_noise_floor_claimed'] is False,'no universal noise floor')
 print('PASS: 1728 rows, immutable bindings, nominal gate preservation, and realized-direction diagnostics verified.')
if __name__=='__main__':main()
