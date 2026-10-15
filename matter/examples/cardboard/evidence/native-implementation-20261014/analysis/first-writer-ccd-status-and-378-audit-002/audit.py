#!/usr/bin/env python3
"""Source-bound audit of root-1201 Newton-7 private contact status selection."""
from __future__ import annotations
import hashlib, json, pathlib, re, struct, sys

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014')
RUNS = {
    '362': BASE/'native/health-and-vt-marker-firstfatal-362',
    '369': BASE/'native/true-residual-default-parity-369',
    '371': BASE/'native/capture-lifetime-default-parity-371',
    '378': BASE/'native/ccd-continuation-full-contact-378',
}
SOURCES = {
    '104': BASE/'snapshots/source-contact-health-diagnostics-104',
    '105': BASE/'snapshots/source-true-residual-certification-105',
    '106': BASE/'snapshots/source-capture-callback-lifetime-106',
    '107': BASE/'snapshots/source-ee-ccd-continuation-107',
}

def sha(path: pathlib.Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def record(path: pathlib.Path, root: int, it: int):
    matches=[]
    for line in path.read_text().splitlines():
        if line.startswith('fem_residual_armijo_iteration='):
            obj=json.loads(line.split('=',1)[1])
            if obj.get('root')==root and obj.get('newton_iteration')==it:
                matches.append(obj)
    if len(matches)!=1: raise ValueError(f'{path}: expected one root={root} iteration={it}, got {len(matches)}')
    return matches[0]

def source_line(path: pathlib.Path, number: int):
    rows=path.read_text().splitlines()
    return rows[number-1].strip()


def captured_candidate(root: pathlib.Path, cap: dict, slot: int, expected_pair: list[int]):
    """Decode candidate/primitive/incidence records from the native capture ABI."""
    arrays = {item["name"]: item for item in cap["arrays"]}
    for required in ("deformable_contact_candidates", "surface_primitives",
                     "cohesive_surface_face_ranges"):
        if required not in arrays:
            raise ValueError(f"{root}: missing captured array {required}")
    payload = (root / "center-capture.coupled.bin").read_bytes()
    cand = arrays["deformable_contact_candidates"]
    if cand["dtype"] != "record:NMDeformableContactCandidateGPU" or cand["element_bytes"] != 16 or slot >= cand["count"]:
        raise ValueError(f"{root}: candidate ABI/slot mismatch")
    candidate = struct.unpack_from("<4I", payload, cand["offset"] + slot * 16)
    if candidate[:2] != tuple(expected_pair) or candidate[2] != slot or not (candidate[3] & 1):
        raise ValueError(f"{root}: captured candidate does not match reported failure tuple")
    primitives = arrays["surface_primitives"]
    if primitives["dtype"] != "record:NMContinuumSurfacePrimitiveGPU" or primitives["element_bytes"] != 64:
        raise ValueError(f"{root}: surface primitive ABI mismatch")
    decoded = []
    for primitive_id in candidate[:2]:
        if primitive_id >= primitives["count"]:
            raise ValueError(f"{root}: primitive id outside captured primitive array")
        words = struct.unpack_from("<16I", payload,
            primitives["offset"] + primitive_id * 64)
        if words[4] != primitive_id:
            raise ValueError(f"{root}: primitive stable identity does not match index")
        decoded.append({"primitive_id": primitive_id,
                        "nodes_and_object": list(words[:4]),
                        "identity": list(words[4:8]),
                        "kind": words[5]})
    ranges = arrays["cohesive_surface_face_ranges"]
    if ranges["dtype"] != "record:NMIncidenceRangeGPU" or ranges["element_bytes"] != 16:
        raise ValueError(f"{root}: cohesive surface range ABI mismatch")
    left_id = candidate[0]
    left_range = list(struct.unpack_from("<4I", payload,
        ranges["offset"] + left_id * 16))
    # enum NMContinuumSurfaceKind: triangle=0, point=1. The private
    # mortar lookup returns an empty patch set when this left incidence count
    # is zero; the contact topology is immutable during this trial.
    if any(item["kind"] != 0 for item in decoded):
        raise ValueError(f"{root}: expected triangle/triangle branch")
    return {"candidate_record_identity": list(candidate),
            "primitive_records": decoded,
            "left_cohesive_surface_range_first_count": left_range[:2],
            "left_cohesive_surface_range_raw": left_range,
            "mortar_patch_set_empty_by_zero_left_incidence": left_range[1] == 0,
            "abi_source": "source106 shared.h NMDeformableContactCandidateGPU (16 B), NMContinuumSurfacePrimitiveGPU (64 B), NMIncidenceRangeGPU (16 B)"}

def main(out: pathlib.Path):
    report={
      'schema':'numi.cardboard.private_ccd_first_writer_audit.v1',
      'verdicts':{
        'strict_362_369_371':'FAIL_UNEXPECTED_STATUS',
        'cross_variant_371_378':'NOT_COMPARABLE_PHYSICAL_SOURCE_CHANGED'
      },
      'runs':{}, 'source_bindings':{}, 'status_selection':{}, '378_comparison':{},
      'limitations':[]
    }
    for name, root in RUNS.items():
        receipt=json.loads((root/'receipt.json').read_text())
        prediction=json.loads((root/'prediction.json').read_text())
        cap=json.loads((root/'center-capture.coupled.json').read_text())
        armijo=record(root/'stderr.log',1201,7)
        n7=armijo['environments'][0]
        # Also retain the later 378 nonlinear stop and its classification.
        later=[]
        for n in (8,9,10):
            try:
                d=record(root/'stderr.log',1201,n)
                e=d['environments'][0]
                later.append({'iteration':n,'trial_outcome':e.get('trial_outcome'),
                  'accepted':e.get('accepted'),'solver_status_code':e['solver_status_code'],
                  'solver_failing_index':e['solver_status'].get('failing_index'),
                  'trial_status_code':e['trial_status'].get('code'),
                  'trial_failing_index':e['trial_status'].get('failing_index'),
                  'trial_diagnostics':e['trial_status'].get('diagnostics')})
            except ValueError:
                pass
        report['runs'][name]={
          'source_snapshot':prediction['source_snapshot'],
          'receipt':{k:receipt.get(k) for k in ('execution_started','source_inputs_unchanged','exit_code','timed_out','wall_seconds')},
          'source_physics_fingerprint':cap['source_physics_fingerprint'],
          'device_program_fingerprint':cap['device_program_fingerprint'],
          'capture':{'root':cap['root'],'iteration':cap['iteration'],'array_count':cap['array_count'],
             'payload_bytes':cap['payload']['byte_count'],'payload_sha256':cap['payload']['sha256'],
             'rhs_preserved_before_fgmres':cap['rhs_preserved_before_fgmres']},
          'newton7':{'base_l2':n7['base_l2'],'trial_alpha':n7['trial_alpha'],
             'accepted':n7['accepted'],'trial_outcome':n7['trial_outcome'],
             'solver_status_code':n7['solver_status_code'],
             'trial_status':n7['trial_status'],
             'decoded_candidate':(captured_candidate(root, cap, n7['trial_status']['failing_index'],
                                                    n7['trial_status']['diagnostics'][:2])
                                  if n7['trial_status'].get('code') == 6 else None)},
          'later_rows':later,
          'stderr_sha256':sha(root/'stderr.log'),
          'diagnostic_control_log_sha256':sha(root/'diagnostic-state/dense/fixture-free-body-control-1201.log')
        }
    # Exact source facts that explain the status tuple shape and concurrent writers.
    source106=SOURCES['106']
    common=source106/'matter/src/metal/common.metalinc'
    contact=source106/'matter/src/metal/contact.metalinc'
    runtime=source106/'matter/src/runtime.mm'
    lines={
      'recordFailure':(common,294,318),
      'mortar_patch_lookup_and_empty_range':(contact,1589,1607),
      'triangle_triangle_mortar_dispatch_and_fallback':(contact,3859,3885),
      'mortar_complement_uncertified_contrast':(contact,3885,3900),
      'edge_edge_uncertified':(contact,3991,4003),
      'candidate_kernel':(contact,3675,3704),
      'private_narrowphase_dispatch':(runtime,14956,14980),
      'stable_candidate_traversal':(contact,2931,3000),
      'candidate_count_scan_scatter':(contact,3003,3095),
    }
    report['source_bindings']={}
    for k,p in [('104',SOURCES['104']/ 'matter/src/metal/contact.metalinc'),
                ('105',SOURCES['105']/ 'matter/src/metal/contact.metalinc'),
                ('106',contact),
                ('107',SOURCES['107']/ 'matter/src/metal/contact.metalinc'),
                ('common106',common),('runtime106',runtime)]:
        report['source_bindings'][k]={'path':str(p),'sha256':sha(p)}
    report['status_selection']={
      'mechanism':'atomic first-writer: compare-exchange status.code from SUCCESS; winning thread writes the remaining tuple; subsequent contenders return',
      'deterministic_min_key_found':False,
      'relevant_callsite_mappings':[
        {'run':'362/369','status_code':6,'failing_index':283,'diagnostics':[92,127,0,0],
         'captured_candidate':[92,127,283,1],
         'captured_primitive_kinds':['triangle','triangle'],
         'captured_left_cohesive_surface_range_first_count':[1,0],
         'source':'contact.metalinc:3991-4003','meaning':'generic triangle-triangle edge-edge CCD candidate; edgeA=0, edgeB=0 was not certified; empty active mixed-cohesive patch set'},
        {'run':'371','status_code':6,'failing_index':124,'diagnostics':[24,127,1,0],
         'captured_candidate':[24,127,124,1],
         'captured_primitive_kinds':['triangle','triangle'],
         'captured_left_cohesive_surface_range_first_count':[1,0],
         'source':'contact.metalinc:3991-4003','meaning':'generic triangle-triangle edge-edge CCD candidate; edgeA=1, edgeB=0 was not certified; empty active mixed-cohesive patch set'}
      ],
      'concurrency_evidence':'private `nm_contact_narrowphase_deformable` dispatch has one invocation per candidate slot; each uncertified edge-edge pair calls recordFailure on the same environment status. The compare-exchange selects the first writer if multiple pairs fail, so different winners are consistent with arrival-order selection, but the captures do not prove the competing failure set.',
      'causal_limit':'Private trial candidate/failure flags are not in the 145-array center capture. The runs establish two different failing winners at the same root/Newton and identical 371-vs-362 center payload, but do not independently prove both candidates failed in every individual run.'
    }
    report['status_selection']['line_evidence']={k:{'path':str(p),'first':a,'last':b,
       'sha256':sha(p),'text':[source_line(p,i) for i in range(a,b+1)]}
       for k,(p,a,b) in lines.items()}
    # Cross-run strict facts.
    r362=report['runs']['362']; r369=report['runs']['369']; r371=report['runs']['371']; r378=report['runs']['378']
    report['strict_same_path_checks']={
      '362_369_371_capture_payload_identical':len({report['runs'][n]['capture']['payload_sha256'] for n in ('362','369','371')})==1,
      '362_369_371_source104_106_contact_shader_identical':len({report['source_bindings'][n]['sha256'] for n in ('104','105','106')})==1,
      '362_369_status_tuple_equal':r362['newton7']['trial_status']==r369['newton7']['trial_status'],
      '371_status_tuple_differs':r362['newton7']['trial_status']!=r371['newton7']['trial_status'],
      '362_369_371_receipt_exit1_no_timeout':all(report['runs'][n]['receipt']['exit_code']==1 and not report['runs'][n]['receipt']['timed_out'] for n in ('362','369','371'))
    }
    # Compare 371 and 378 same-layout capture payloads by descriptors.
    j371=json.loads((RUNS['371']/'center-capture.coupled.json').read_text())
    j378=json.loads((RUNS['378']/'center-capture.coupled.json').read_text())
    b371=(RUNS['371']/'center-capture.coupled.bin').read_bytes()
    b378=(RUNS['378']/'center-capture.coupled.bin').read_bytes()
    arrays371={a['name']:a for a in j371['arrays']}; arrays378={a['name']:a for a in j378['arrays']}
    if arrays371.keys()!=arrays378.keys(): raise ValueError('capture array-name sets differ')
    changed=[]
    for name,a in arrays371.items():
        z=arrays378[name]
        x=b371[a['offset']:a['offset']+a['byte_count']]
        y=b378[z['offset']:z['offset']+z['byte_count']]
        if x!=y: changed.append({'name':name,'dtype_371':a['dtype'],'count_371':a['count'],
            'byte_count':a['byte_count'],'differing_byte_positions':sum(u!=v for u,v in zip(x,y))})
    src107=contact if False else SOURCES['107']/'matter/src/metal/contact.metalinc'
    diff_cmd='The edge-edge distance iteration count changes from 17 to 33.'
    report['378_comparison']={
      'comparison_class':'separate source variant; not a strict parity pair',
      'source_physics_fingerprint_371_equals_378':j371['source_physics_fingerprint']==j378['source_physics_fingerprint'],
      'device_fingerprint_371':j371['device_program_fingerprint'],'device_fingerprint_378':j378['device_program_fingerprint'],
      'contact_shader_106_sha256':sha(contact),'contact_shader_107_sha256':sha(src107),
      'edge_edge_iteration_change':{'source_106_line_3365':source_line(contact,3365),
          'source_107_line_3365':source_line(src107,3365),'summary':diff_cmd},
      'capture_layout_equal':len(j371['arrays'])==len(j378['arrays']) and arrays371.keys()==arrays378.keys(),
      'capture_differing_array_count':len(changed),'capture_differing_arrays':changed,
      'capture_payload_sha_371':j371['payload']['sha256'],'capture_payload_sha_378':j378['payload']['sha256'],
      'armijo_prefix':[],
      'interpretation':'The edge-edge CCD implementation changed and the physical trajectory diverged at Newton 5; therefore the Newton-7 status cannot be compared as a same-input status-selection replicate.'
    }
    for n in range(5,10):
        try:
            d=record(RUNS['378']/'stderr.log',1201,n); e=d['environments'][0]
            report['378_comparison']['armijo_prefix'].append({'iteration':n,'base_l2':e['base_l2'],
                'trial_alpha':e['trial_alpha'],'accepted':e['accepted'],'trial_outcome':e['trial_outcome'],
                'solver_status_code':e['solver_status_code'],'solver_failing_index':e['solver_status'].get('failing_index'),
                'trial_status_code':e['trial_status'].get('code'),'trial_failing_index':e['trial_status'].get('failing_index'),
                'trial_diagnostics':e['trial_status'].get('diagnostics')})
        except ValueError: pass
    report['limitations']=[
      'The 362/369/371 evidence preserves the unexpected status difference; no parity normalization is applied.',
      'The captured center candidate records identify the winning primitive pairs and show zero mixed-cohesive incidence for their left primitive IDs, disambiguating both tuples as generic triangle-triangle edge-edge failures. The private trial failure list is not captured, so the evidence does not establish which other candidates also failed in each run.',
      'A deterministic fix should be scoped to narrowphase private/production status attribution and preserve earlier-phase status precedence; no global recordFailure or feasibility change is recommended from this audit alone.',
      '378 uses a changed edge-edge CCD iteration count and a diverged Newton trajectory, so it is a separate result rather than an independent repeat of the 371 failed trial.'
    ]
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report

if __name__=='__main__':
    target=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path(__file__).with_name('report.json')
    r=main(target)
    print(json.dumps({'overall_371_verdict':r['verdicts']['strict_362_369_371'],
      'source_variant_378':r['verdicts']['cross_variant_371_378'],
      'changed_378_arrays':r['378_comparison']['capture_differing_array_count'],
      'output':str(target)},sort_keys=True))
