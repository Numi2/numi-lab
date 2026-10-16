#!/usr/bin/env python3
"""Verify the sealed native526/527 CPU audit against its bound native evidence."""
import hashlib, json, pathlib, sys
base=pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
here=pathlib.Path(__file__).resolve().parent
report=json.loads((here/'report.json').read_text())
manifest=json.loads((here/'manifest.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
errors=[]
for item in manifest['inputs']:
    p=pathlib.Path(item['path'])
    if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
        errors.append('input mismatch: '+str(p))
for name,meta in manifest['outputs'].items():
    p=here/name
    if not p.is_file() or p.stat().st_size!=meta['bytes'] or sha(p)!=meta['sha256']:
        errors.append('output mismatch: '+name)
if not report.get('bindings_match_both_command_specs'): errors.append('source/build/material binding failed')
for run in ('526','527'):
    r=report['runs'][run]
    if not r['receipt_inventory_exact'] or not r['all_receipt_hashes_match']:
        errors.append(run+' receipt inventory mismatch')
    if not r['baseline_aliased_byte_equal']:
        errors.append(run+' baseline/alias outputs differ')
    if r['receipt']['exit_code']!=0 or r['receipt']['timed_out'] or r['receipt']['space_abort']:
        errors.append(run+' command receipt was not clean')
    t=r['transactions'][2]
    if t['before_cursor']!=t['after_cursor'] or t['changed_accepted_state_vectors']:
        errors.append(run+' rejected transaction changed accepted state')
a=report['runs']['526']['transactions'][2]['after_status']
if a['code']!=1 or a['failing_index']!=0 or a['diagnostics']!=[-3,-3,-3,-3]:
    errors.append('526 postCommit sentinel signature mismatch')
b=report['runs']['527']['transactions'][2]['after_status']
if b['code']!=10 or b['failing_index']!=18:
    errors.append('527 preDynamics failure signature mismatch')
if report['runs']['526']['trace_summary']['record_count']!=32 or report['runs']['526']['trace_summary']['healthy_pretrial_records']!=32:
    errors.append('526 Armijo trace health mismatch')
if report['runs']['527']['trace_summary']['first_nonzero_solver_status_newton']!=18 or report['runs']['527']['trace_summary']['first_nonzero_pretrial_newton']!=19:
    errors.append('527 Armijo failure ordering mismatch')
if report['common_command_binding']['differing_common_fields']:
    errors.append('shared source/build/material/common command fields differ')
if report['compiled_preconditioner_selection']['526']['line']!='selected_fem_preconditioner=regional_galerkin_contact selector=4':
    errors.append('526 compiled selector evidence mismatch')
if report['compiled_preconditioner_selection']['527']['line']!='selected_fem_preconditioner=scalar_diagonal selector=0':
    errors.append('527 compiled selector evidence mismatch')
print(json.dumps({'ok':not errors,'errors':errors,'report_sha256':sha(here/'report.json')},sort_keys=True))
sys.exit(1 if errors else 0)
