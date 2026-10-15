#!/usr/bin/env python3
"""Overlap MP90 sweep at the prior FP64 epsilon 1e-7 for direct comparison."""
from pathlib import Path
import importlib.util, json, sys, hashlib, time
import numpy as np
BASE=Path('/Users/home/cardboard-implementation-20261014')
PKG=BASE/'analysis/medium-tangent-state-sensitivity-1634-002'
ORIG=BASE/'analysis/medium-tangent-state-sensitivity-1634-001/replay.py'
CAP=BASE/'solver/recycle-source-default-first-control-283/operator-capture'
NATIVE=BASE/'native/rationalized-all-paper-entry-points-300/output/paper-points-accurate.jsonl'
spec=importlib.util.spec_from_file_location('mp90_epfd',PKG/'mp90_endpoint_fd.py')
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
assert mod.sha(ORIG)==json.loads((BASE/'analysis/medium-tangent-state-sensitivity-1634-001/manifest.json').read_text())['artifacts']['replay.py']['sha256']
records=[json.loads(x) for x in CAP.read_text().splitlines() if x.strip()]
header=records[0]
rows={(x['environment'],x['tet_id']):x for x in records if x.get('record')=='tetrahedron'}
native=[json.loads(x) for x in NATIVE.read_text().splitlines() if x.strip()]
nmap={(x['environment'],x['tet_id']):x for x in native if x.get('record')=='point'}
start=time.time()
results=[mod.run_point(rows[(0,t)],header,nmap,['1e-7']) for t in (1634,1638)]
prior=json.loads((BASE/'analysis/medium-tangent-state-sensitivity-1634-001/report.json').read_text())
old={x['point']['tet_id']:x for x in prior['results']}
summary=[]
for x in results:
 tid=x['point']['tet_id']; cur=x['nine_direction_endpoint_sweep'][0]
 prev=old[tid]['nine_direction_check']['finite_difference_sweep'][-1]
 summary.append({'tet_id':tid,
  'mp90_epsilon_F_entry':'1e-7',
  'mp90_full_matrix_relative_error_vs_analytic':cur['central_matrix_relative_error_vs_MP90_implicit'],
  'prior_fp64_epsilon':prev['epsilon'],
  'prior_fp64_full_matrix_relative_error_vs_MP90':prev['relative_l2_vs_mp90'],
  'mp90_endpoint_projection_converged':cur['all_endpoint_projections_converged'],
  'mp90_branch_transition_columns':cur['branch_transition_columns'],
  'mp90_basis_fd_matrix':cur['central_matrix'],
  'endpoint_records':cur['endpoint_records']})
out={'schema':'numi.cardboard.medium-tangent-mp90-common-epsilon.v1',
 'classification':'Same-epsilon 1e-7 comparison between fresh MP90 endpoint re-projections and the already sealed FP64 finite-difference replay. Diagnostic only.',
 'bindings':{'mp90_package_report_sha256':mod.sha(PKG/'report.json'),
             'mp90_script_sha256':mod.sha(PKG/'mp90_endpoint_fd.py'),
             'capture_sha256':mod.sha(CAP),'native_jsonl_sha256':mod.sha(NATIVE),
             'prior_fp64_report_sha256':mod.sha(BASE/'analysis/medium-tangent-state-sensitivity-1634-001/report.json')},
 'result':summary,'elapsed_seconds':time.time()-start,
 'scope':'Two captured medium points, all nine matrix directions at epsilon 1e-7. Same captured accepted history and parameters; no physics source or GPU.'}
p=PKG/'common-epsilon-1e-7.json';p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'output':str(p),'sha256':mod.sha(p),'summary':[{'tet_id':x['tet_id'],'mp90_error':x['mp90_full_matrix_relative_error_vs_analytic'],'prior_fp64_error':x['prior_fp64_full_matrix_relative_error_vs_MP90'],'converged':x['mp90_endpoint_projection_converged']} for x in summary]},indent=2))
