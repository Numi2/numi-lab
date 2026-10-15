#!/usr/bin/env python3
"""Fail-closed source104 diagnostic-vs-source103 native output parity auditor.

This tool does not run native code. It compares an already completed 362-style
source104 run to the exact source103 Newton-7 center baseline (359). It admits
only the separately source-proven VT diagnostic marker and schema-3 health
classification fields; captures, accepted-history binaries, and all existing
numeric outputs must remain bitwise/equality identical.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path('/Users/home/cardboard-implementation-20261014')
SOURCE103 = ROOT / 'snapshots/source-whole-mortar-proof-103'
SOURCE104 = ROOT / 'snapshots/source-contact-health-diagnostics-104'
BASELINE = ROOT / 'native/first-fatal-center-359'
INTEGRATION = ROOT / 'analysis/source104-diagnostic-integration-001/integration.json'
HEALTH_MANIFEST = ROOT / 'analysis/source104-diagnostic-integration-001/health-manifest-count-erratum.json'
VT_MANIFEST = ROOT / 'analysis/source103-vt-failure-marker-001/manifest.json'
VT_CONSUMER_AUDIT = ROOT / 'analysis/source103-vt-failure-marker-001/consumer-audit.md'
TRACE_PREFIX = 'fem_residual_armijo_iteration='
TRACE_ENABLED_PREFIX = 'fem_residual_armijo_trace_enabled='
CAPTURE_COMPLETE_PREFIX = 'fem_coupled_capture_complete='


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path):
    def reject(token):
        raise ValueError(f'non-standard JSON token {token} in {path}')
    return json.loads(path.read_text(), parse_constant=reject)


def need(ok: bool, msg: str):
    if not ok:
        raise ValueError(msg)


def source_bindings():
    old = strict_json(SOURCE103 / 'frozen-source.json')['source_sha256']
    new = strict_json(SOURCE104 / 'frozen-source.json')['source_sha256']
    need(set(old) == set(new), 'source snapshots have different selected-file inventories')
    changed = {p: (old[p], new[p]) for p in old if old[p] != new[p]}
    expected = {
        'matter/src/metal/contact.metalinc': (
            '427aca061ad2c6d2218b5eae0bc7620132a1ef9fff9f688390ccc96ca8a27ffc',
            'b9ec3915b63bdee729ff0902c390e137e0f6b5d005d06e6a7136c534be3d510b'),
        'matter/src/metal/fem.metalinc': (
            'f4b5cd60dbe7ace7a54636746a00b5e7186d6c1f4890eb0718c138af9c753ae4',
            '742cf561ee1ed91e20ed797e664595f675302437a3d583f8c5768da6cb3d008e'),
        'matter/src/runtime.mm': (
            '836605dca3f6b264d5ee04a9f65b66997ecbdc39f60709039c7e1ac59becb087',
            '119148dedc593742dfe7f62c1b06c99e488b6c40a783aa839722f70d2da59c20'),
    }
    need(changed == expected, f'source delta differs from the three-file reviewed diagnostic overlay: {changed}')
    for snapshot, mapping in ((SOURCE103, old), (SOURCE104, new)):
        for rel, expected_hash in mapping.items():
            p = snapshot / rel
            need(p.is_file() and sha(p) == expected_hash, f'snapshot source hash mismatch: {snapshot}/{rel}')
    integration = strict_json(INTEGRATION)
    need(integration['snapshot'] == str(SOURCE104), 'integration manifest points at a different source104 snapshot')
    need(integration['base_snapshot'] == str(SOURCE103), 'integration base is not source103')
    need(integration['native_parity'] == 'pending', 'integration package was already adjudicated unexpectedly')
    health = strict_json(HEALTH_MANIFEST)
    need(health['diagnostic_only'] is True and health['production_mechanics_changed'] is False,
         'source104 health manifest does not declare diagnostic-only scope')
    vt = strict_json(VT_MANIFEST)
    need(vt['change']['new_diagnostic_w'] == 'float(3u + direction)', 'VT marker contract changed')
    need(vt['change']['resulting_vt_markers'] == [3, 4], 'VT marker range changed')
    return {
        'source_file_count': len(old), 'changed_source_files': changed,
        'source103_snapshot_manifest_sha256': sha(SOURCE103 / 'frozen-source.json'),
        'source104_snapshot_manifest_sha256': sha(SOURCE104 / 'frozen-source.json'),
        'source104_integration_sha256': sha(INTEGRATION),
        'source104_health_manifest_sha256': sha(HEALTH_MANIFEST),
        'source103_vt_marker_manifest_sha256': sha(VT_MANIFEST),
        'source103_vt_consumer_audit_sha256': sha(VT_CONSUMER_AUDIT),
    }


def validate_run(run: Path, expected_source: Path, label: str):
    receipt = strict_json(run / 'receipt.json')
    prediction = strict_json(run / 'prediction.json')
    need(receipt.get('schema') == 'numi.cardboard.native-run-receipt.v2', f'{label}: receipt schema mismatch')
    need(receipt.get('execution_started') is True, f'{label}: native execution did not start')
    need(receipt.get('source_inputs_unchanged') is True and receipt.get('changed_inputs') == [],
         f'{label}: source inputs changed during run')
    need(receipt.get('timed_out') is False, f'{label}: native run timed out')
    need(receipt.get('termination_errors') == [], f'{label}: run wrapper reports termination errors')
    need(prediction.get('schema') == 'numi.cardboard.native-prediction.v1', f'{label}: prediction schema mismatch')
    need(prediction.get('source_snapshot') == str(expected_source), f'{label}: wrong source snapshot binding')
    expected_material=expected_source/'matter/materials/nagasawa2013_medium_orthotropic.nmatter'
    env = prediction.get('environment', {})
    need(env.get('NM_FEM_COUPLED_CAPTURE_SELECTOR') == '1201:7', f'{label}: capture selector differs')
    argv = prediction.get('argv')
    need(isinstance(argv, list) and '--sparse-free-body-parity' in argv and
         '--diagnostic-state-output' in argv and '--extended-budget' in argv and
         '--compensated-material-arithmetic' in argv,
         f'{label}: expected full-contact source103/104 protocol flags missing')
    need(len(argv)>=2 and Path(argv[0]).name=='numi-matter-cardboard-full-contact-armijo-check',
         f'{label}: unexpected native executable identity')
    need(Path(argv[1]).resolve()==expected_material.resolve() and expected_material.is_file(),
         f'{label}: input material path is not the source-bound Nagasawa medium asset')
    build_inputs = prediction.get('external_inputs', [])
    need(any(Path(s).name == 'source-build.json' for s in build_inputs), f'{label}: missing source-build receipt input')
    # Validate the prediction's complete selected-source inventory against its sealed snapshot.
    inv = strict_json(expected_source / 'frozen-source.json')['source_sha256']
    input_hashes = prediction.get('inputs_sha256', {})
    for rel, expected_hash in inv.items():
        got = input_hashes.get('source/' + rel)
        need(got == expected_hash, f'{label}: run prediction source input mismatch: {rel}')
    validated_builds=[]
    for item in build_inputs:
        p = Path(item)
        # For source103 this is also retained under diagnostic-state/inputs; for source104
        # the run directory is the authoritative retained copy after completion.
        if p.exists():
            retained=p
        else:
            candidates=[run/'inputs/external'/p.name,
                        run/'diagnostic-state/inputs/external'/p.name]
            retained=next((q for q in candidates if q.is_file()),None)
        need(retained is not None and retained.is_file(),
             f'{label}: source-build input missing from run and retained inputs')
        external_keys=[k for k in input_hashes if k.startswith('external/') and k.endswith(p.name)]
        need(external_keys and all(input_hashes[k]==sha(retained) for k in external_keys),
             f'{label}: retained source-build receipt hash mismatch')
        build=strict_json(retained)
        need(build.get('success') is True and build.get('source_snapshot')==str(expected_source),
             f'{label}: source-build receipt does not bind the expected successful snapshot')
        validated_builds.append({'path':str(retained),'sha256':sha(retained),'record':build})
    return receipt, prediction, validated_builds


def normalized_argv(prediction):
    argv=prediction.get('argv')
    need(isinstance(argv,list) and len(argv)>=2,'prediction argv is malformed')
    out=['<executable>']
    i=1
    while i<len(argv):
        value=argv[i]
        if i==1 and Path(value).name == 'nagasawa2013_medium_orthotropic.nmatter':
            out.append('<source-material>')
        elif value=='--diagnostic-state-output':
            need(i+1<len(argv),'diagnostic-state-output value is missing')
            out.extend([value,'<diagnostic-state-dir>']); i+=1
        else:
            out.append(value)
        i+=1
    return out


def _artifact_hashes(builds, label):
    need(len(builds)==1, f'{label}: expected exactly one validated source-build receipt')
    artifacts=builds[0]['record'].get('artifacts_sha256')
    need(isinstance(artifacts,dict) and artifacts, f'{label}: source-build receipt lacks artifact hashes')
    by_name={}
    for path,digest in artifacts.items():
        by_name.setdefault(Path(path).name,[]).append(digest)
    return by_name


def compare_predictions(base_prediction,candidate_prediction,base_builds,candidate_builds):
    need(normalized_argv(base_prediction)==normalized_argv(candidate_prediction),
         'baseline/candidate native arguments differ after source/output path normalization')
    need(base_prediction.get('native_protocol_id')==candidate_prediction.get('native_protocol_id'),
         'native protocol ID differs')
    need(base_prediction.get('calibration_case_binding')==candidate_prediction.get('calibration_case_binding'),
         'calibration case binding differs')
    base_env=dict(base_prediction.get('environment',{})); cand_env=dict(candidate_prediction.get('environment',{}))
    need(base_env.get('NM_FEM_COUPLED_CAPTURE_SELECTOR')==cand_env.get('NM_FEM_COUPLED_CAPTURE_SELECTOR')=='1201:7',
         'capture selector differs')
    base_env['NM_FEM_COUPLED_CAPTURE_PATH']=cand_env['NM_FEM_COUPLED_CAPTURE_PATH']='<capture-prefix>'
    need(base_env==cand_env,'environment differs outside the capture output path')
    # The captured source inventory must be the same except for the reviewed three-file overlay.
    bh=base_prediction.get('inputs_sha256',{}); ch=candidate_prediction.get('inputs_sha256',{})
    need(set(bh)==set(ch),'run input inventory differs')
    allowed_source_keys={
      'source/matter/src/metal/contact.metalinc',
      'source/matter/src/metal/fem.metalinc',
      'source/matter/src/runtime.mm'}
    allowed_external={k for k in bh if k.startswith('external/') and k.endswith('source-build.json')}
    # The run input inventory also fingerprints the three products rebuilt from
    # the selected source snapshot. Permit those path hashes only when each exact
    # digest is independently present in that run's successful source-build receipt.
    runtime_artifacts={
      'runtime/lib/libmetalrobo.dylib':'libmetalrobo.dylib',
      'runtime/matter/numi-matter-cardboard-full-contact-armijo-check':'numi-matter-cardboard-full-contact-armijo-check',
      'runtime/matter/shaders/NumiMatter.metallib':'NumiMatter.metallib',
    }
    base_artifacts=_artifact_hashes(base_builds,'baseline')
    candidate_artifacts=_artifact_hashes(candidate_builds,'candidate')
    changed={k for k in bh if bh[k]!=ch[k]}
    runtime_keys=set(runtime_artifacts)
    need(changed <= allowed_source_keys | allowed_external | runtime_keys,
         f'run input hashes differ outside reviewed source/build provenance: {sorted(changed)}')
    need(allowed_source_keys <= changed,'expected reviewed source files were not rebound in candidate prediction')
    need(runtime_keys <= changed,'expected source-bound runtime products were not rebound in candidate prediction')
    for input_key,artifact_name in runtime_artifacts.items():
        old_matches=base_artifacts.get(artifact_name,[])
        new_matches=candidate_artifacts.get(artifact_name,[])
        need(len(old_matches)==1 and bh[input_key]==old_matches[0],
             f'baseline runtime input is not bound to its successful build artifact: {input_key}')
        need(len(new_matches)==1 and ch[input_key]==new_matches[0],
             f'candidate runtime input is not bound to its successful build artifact: {input_key}')
    return {'changed_input_hash_paths':sorted(changed),
            'changed_source_overlay_paths':sorted(changed & allowed_source_keys),
            'source_build_receipt_paths':sorted(allowed_external),
            'rebuilt_runtime_artifacts':{
                input_key:{'artifact_name':artifact_name,
                           'baseline_sha256':bh[input_key],
                           'candidate_sha256':ch[input_key],
                           'baseline_bound_by_source_build':True,
                           'candidate_bound_by_source_build':True}
                for input_key,artifact_name in runtime_artifacts.items()}}


def manifest_payload(run: Path, label: str):
    manifest_path = run / 'center-capture.coupled.json'
    payload_path = run / 'center-capture.coupled.bin'
    m = strict_json(manifest_path)
    need(m.get('schema') == 'numi.matter.coupled_newton_capture.v1', f'{label}: capture schema mismatch')
    need(m.get('capture_point') == 'after_fgmres_measure_and_human_direction_map_before_armijo_trial',
         f'{label}: capture point mismatch')
    need(m.get('root') == 1201 and m.get('iteration') == 7 and m.get('environment_count') == 1,
         f'{label}: selected capture identity mismatch')
    need(m.get('payload', {}).get('file') == str(payload_path), f'{label}: manifest payload path mismatch')
    need(payload_path.is_file(), f'{label}: missing capture payload')
    need(payload_path.stat().st_size == m['payload']['byte_count'], f'{label}: payload size mismatch')
    payload_bytes=payload_path.read_bytes()
    need(sha_bytes(payload_bytes) == m['payload']['sha256'], f'{label}: payload hash mismatch')
    need(len(m.get('arrays', [])) == m.get('array_count'), f'{label}: manifest array count mismatch')
    for a in m['arrays']:
        start = int(a['offset']); count = int(a['byte_count'])
        need(start >= 0 and count >= 0 and start + count <= payload_path.stat().st_size,
             f'{label}: array out of payload bounds: {a.get("name")}')
        raw = payload_bytes[start:start+count]
        need(sha_bytes(raw) == a['sha256'], f'{label}: array hash mismatch: {a.get("name")}')
    return manifest_path, payload_path, m


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare_capture(base: Path, cand: Path):
    bp, bb, bm = manifest_payload(base, '359')
    cp, cb, cm = manifest_payload(cand, '362')
    need(bb.read_bytes() == cb.read_bytes(), 'coupled center payload bytes differ')
    # Compare every declared payload array in order, including all production and accepted-history arrays.
    need(len(bm['arrays']) == len(cm['arrays']), 'capture array count changed')
    array_changes=[]
    for a, b in zip(bm['arrays'], cm['arrays']):
        for key in ('name','dtype','offset','byte_count','element_bytes','shape'):
            need(a.get(key) == b.get(key), f'capture array layout changed: {a.get("name")}:{key}')
        need(a['sha256'] == b['sha256'], f'capture numeric array hash changed: {a["name"]}')
    # Capture provenance is run-specific; only the executable program fingerprint and payload path
    # may differ as metadata. Physics fingerprint and all other manifest metadata remain exact.
    bm_norm = json.loads(json.dumps(bm)); cm_norm = json.loads(json.dumps(cm))
    bm_norm['payload']['file'] = cm_norm['payload']['file'] = '<capture-payload>'
    bm_norm['device_program_fingerprint'] = cm_norm['device_program_fingerprint'] = '<device-program>'
    need(bm_norm == cm_norm, 'coupled capture manifest differs outside run-specific payload path/program fingerprint')
    if bm['device_program_fingerprint'] != cm['device_program_fingerprint']:
        array_changes.append({'field':'device_program_fingerprint','baseline':bm['device_program_fingerprint'],
                              'candidate':cm['device_program_fingerprint'],'classification':'provenance-only'})
    need(bm['source_physics_fingerprint'] == cm['source_physics_fingerprint'],
         'source physics fingerprint changed')
    return {'array_count':len(bm['arrays']),'payload_bytes':bb.stat().st_size,
            'payload_sha256':sha(bb),'all_array_hashes_identical':True,
            'run_metadata_differences':array_changes,
            'baseline_manifest_sha256':sha(bp),'candidate_manifest_sha256':sha(cp)}


def compare_json_exact_with_marker(old, new, path='$', marker_changes=None):
    marker_changes = marker_changes if marker_changes is not None else []
    if type(old) is not type(new):
        raise ValueError(f'{path}: type differs ({type(old).__name__} vs {type(new).__name__})')
    if isinstance(old, dict):
        need(set(old) == set(new), f'{path}: JSON keys differ')
        for key in old:
            compare_json_exact_with_marker(old[key], new[key], f'{path}.{key}', marker_changes)
    elif isinstance(old, list):
        need(len(old) == len(new), f'{path}: JSON array length differs')
        for i, (a,b) in enumerate(zip(old,new)):
            compare_json_exact_with_marker(a,b,f'{path}[{i}]',marker_changes)
    elif (re.fullmatch(r'.*\.after\.statuses\[\d+\]\.diagnostics\[3\]',path) or
          re.fullmatch(r'.*\.trial_status\.diagnostics\[3\]',path)) and \
          isinstance(old,(int,float)) and isinstance(new,(int,float)) and old != new:
        need(old == 0 and new in (3,4), f'{path}: unexpected diagnostic.w delta {old} -> {new}')
        marker_changes.append({'path':path,'baseline':old,'candidate':new})
    else:
        need(old == new, f'{path}: value differs: {old!r} vs {new!r}')
    return marker_changes


def compare_snapshot_payload_bytes(old: bytes, new: bytes, name: str):
    # Raw diagnostic snapshots begin with a u32-length message, then
    # sourcePhysicsFingerprint and deviceProgramFingerprint u64s. The latter
    # changes with the independently rebuilt metallib; every remaining
    # persistent/scratch byte must remain exact.
    need(len(old)>=20 and len(new)==len(old),f'diagnostic snapshot size/extent differs: {name}')
    old_msg_len=int.from_bytes(old[0:4],'little'); new_msg_len=int.from_bytes(new[0:4],'little')
    need(old_msg_len==new_msg_len and 4+old_msg_len+16<=len(old),
         f'diagnostic snapshot scalar prefix is malformed: {name}')
    source_start=4+old_msg_len; device_start=source_start+8; suffix_start=device_start+8
    need(old[:device_start]==new[:device_start],
         f'diagnostic source fingerprint/message changed: {name}')
    old_device=int.from_bytes(old[device_start:suffix_start],'little')
    new_device=int.from_bytes(new[device_start:suffix_start],'little')
    need(old_device!=0 and new_device!=0,
         f'diagnostic device fingerprint is missing: {name}')
    need(old[suffix_start:]==new[suffix_start:],
         f'diagnostic payload differs outside rebuilt device fingerprint: {name}')
    return {'file':name,'offset':device_start,
            'baseline_device_program_fingerprint':f'{old_device:016x}',
            'candidate_device_program_fingerprint':f'{new_device:016x}',
            'all_remaining_payload_bytes_exact':True}


def trace_records_path(path: Path):
    text = path.read_text(errors='strict')
    records=[]; enabled=[]; complete=[]
    for line_no, line in enumerate(text.splitlines(),1):
        for prefix, target in ((TRACE_PREFIX,records),(TRACE_ENABLED_PREFIX,enabled),(CAPTURE_COMPLETE_PREFIX,complete)):
            pos=line.find(prefix)
            if pos >= 0:
                raw=line[pos+len(prefix):].strip()
                # Native failure summary may append text after JSON; these records are emitted as
                # their own complete lines, so require strict parsing of the suffix.
                target.append((line_no, strict_json_text(raw, f'{path}:{line_no}')))
    return path,text,records,enabled,complete


def trace_records(run: Path):
    return trace_records_path(run / 'stderr.log')


def strict_json_text(s: str, label: str):
    def reject(token): raise ValueError(f'{label}: invalid JSON token {token}')
    return json.loads(s, parse_constant=reject)


COMMON_SEMANTICS_103 = {
 'base_l2':'exact assembled free-node mechanical residual before this Newton correction, impulse units',
 'trial_l2':'selected residual if accepted, otherwise last tested residual',
 'trial_alpha':'selected alpha if accepted, otherwise last tested alpha',
 'environment_line_search_alpha':'environment-scoped alpha read by rigid apply; residual Armijo may update the FEM object alpha independently'}
COMMON_SEMANTICS_104 = {
 'base_l2':'raw recorded free-node metric in impulse units; interpret as the complete assembled mechanical residual only when base_health is healthy_at_armijo_entry',
 'trial_l2':'selected residual if accepted, otherwise last tested residual',
 'trial_alpha':'selected alpha if accepted, otherwise last tested alpha',
 'environment_line_search_alpha':'environment-scoped alpha read by rigid apply; residual Armijo may update the FEM object alpha independently',
 'base_status':'captured immediately before this Armijo trial loop',
 'trial_status':'last loop-pass scratch, not necessarily a tested private trial; after selected acceptance it may be an unused gated tail status'}
HEALTH_KEYS={'trial_outcome','last_trial_status_role','base_health','base_residual_interpretation','pretrial_status'}


def validate_health_record(r):
    need(r.get('schema') == 3, 'source104 Armijo trace schema must be 3')
    need(r.get('semantics') == COMMON_SEMANTICS_104, 'source104 schema-3 semantics wording changed')
    envs=r.get('environments'); need(isinstance(envs,list) and len(envs)==1,'expected one environment in selected fixture')
    for e in envs:
        need(HEALTH_KEYS <= set(e), 'schema-3 health fields missing')
        pre=e['pretrial_status']
        if pre is None:
            expected_health='unavailable_status_snapshot_missing'
        elif pre.get('code') != 0:
            expected_health='invalid_pretrial_solver_status'
        elif not isinstance(e.get('base_l2'),(int,float)) or not math.isfinite(e['base_l2']):
            expected_health='invalid_nonfinite_base_metric'
        else:
            expected_health='healthy_at_armijo_entry'
        need(e['base_health']==expected_health,'base_health disagrees with captured status/metric')
        expected_interp=('complete_assembled_free_dof_residual' if expected_health=='healthy_at_armijo_entry'
                         else 'raw_metric_only_base_not_qualified_as_complete_residual')
        need(e['base_residual_interpretation']==expected_interp,'base residual interpretation is inconsistent')
        accepted=e.get('accepted') is True; noop=e.get('certified_noop') is True
        status=e.get('trial_status') or {}
        status_code=status.get('code',0)
        if accepted: expected_outcome='certified_noop' if noop else 'selected_trial_accepted'
        elif pre is not None and pre.get('code') != 0: expected_outcome='blocked_by_pretrial_solver_status'
        elif status_code != 0: expected_outcome='last_tested_trial_rejected'
        else: expected_outcome='no_trial_accepted_status_clear'
        need(e['trial_outcome']==expected_outcome,'trial_outcome inconsistent with acceptance/status fields')
        count=e.get('line_search_raw',[None,None,None,None])[2]
        try: trial_count=float(count)
        except (TypeError,ValueError): trial_count=float('nan')
        no_trial=noop or not math.isfinite(trial_count) or trial_count <= 0
        selected_budget=24.0
        unused=accepted and not no_trial and trial_count < selected_budget
        expected_role=('no_private_trial_certified_noop' if noop else
                       'no_private_trial_reported' if no_trial and accepted else
                       'unused_tail_after_selected_accept' if unused else
                       'selected_trial_status' if accepted else 'last_tested_trial_status')
        need(e['last_trial_status_role']==expected_role,'last_trial_status_role inconsistent with actual loop count')


def compare_trace_files(base_path: Path, cand_path: Path, expected_root: int,
                        require_capture_complete: bool):
    _,base_text,base,base_enabled,base_complete=trace_records_path(base_path)
    _,cand_text,cand,cand_enabled,cand_complete=trace_records_path(cand_path)
    prefix_tokens=(TRACE_PREFIX,TRACE_ENABLED_PREFIX,CAPTURE_COMPLETE_PREFIX)
    base_plain=[line for line in base_text.splitlines() if not any(token in line for token in prefix_tokens)]
    cand_plain=[line for line in cand_text.splitlines() if not any(token in line for token in prefix_tokens)]
    need(base_plain==cand_plain,f'unstructured native log lines differ: {base_path.name}')
    need(len(base_enabled)==len(cand_enabled)==1,'Armijo trace-enabled record count changed')
    be,ce=base_enabled[0][1],cand_enabled[0][1]
    need(be.get('schema')==1 and ce.get('schema')==3,'unexpected trace-enabled schema versions')
    need({k:v for k,v in be.items() if k!='schema'} ==
         {k:v for k,v in ce.items() if k not in ('schema','base_status_capture')},
         'trace-enabled common fields differ')
    need(isinstance(ce.get('base_status_capture'),bool),'base status capture availability is not boolean')
    need(len(base)==len(cand)>0,'Armijo iteration record count changed')
    marker_changes=[]; numeric_records=0
    for (bl,b),(cl,c) in zip(base,cand):
        need(b.get('schema')==1 and c.get('schema')==3,'Armijo schema transition mismatch')
        need(b.get('root')==c.get('root')==expected_root and b.get('newton_iteration')==c.get('newton_iteration'),
             f'Armijo root/iteration differs at lines {bl}/{cl}')
        need(b.get('semantics')==COMMON_SEMANTICS_103,'source103 Armijo semantics not the reviewed baseline')
        need(c.get('semantics')==COMMON_SEMANTICS_104,'source104 Armijo semantics not the reviewed schema3 contract')
        common_old={k:v for k,v in b.items() if k not in ('schema','semantics')}
        common_new={k:v for k,v in c.items() if k not in ('schema','semantics')}
        need(set(common_new) == set(common_old), 'source104 top-level trace added unreviewed fields')
        need(len(common_old['environments'])==len(common_new['environments']),
             'source104 environment count changed')
        for new_env in common_new['environments']:
            need(HEALTH_KEYS <= set(new_env), 'schema-3 health fields missing from environment')
            need(set(new_env)-HEALTH_KEYS == set(common_old['environments'][0]),
                 'source104 environment added an unreviewed field')
        common_new['environments']=[{k:v for k,v in env.items() if k not in HEALTH_KEYS}
                                    for env in common_new['environments']]
        # Compare all pre-existing trace data exactly; diagnostic.w is allowed only at code-6 VT
        # failure records and only to the source-proven marker values 3 or 4.
        local_markers=[]
        compare_json_exact_with_marker(common_old,common_new,'trace',local_markers)
        for change in local_markers:
            env=b['environments'][0]
            old_status=env['trial_status']; new_status=c['environments'][0]['trial_status']
            need(old_status.get('code')==new_status.get('code')==6,
                 'VT marker delta does not occur on a contact-failure trial status')
            od=old_status['diagnostics']; nd=new_status['diagnostics']
            need(od[:3]==nd[:3] and nd[0]==92 and nd[1]==127 and 0<=nd[2]<=2,
                 'VT marker does not preserve the source-proven pair and vertex identity')
            marker_changes.extend(local_markers)
        env=c['environments'][0]
        validate_health_record(c)
        numeric_records += 1
    if require_capture_complete:
        # The center-capture completion line is observational provenance and must refer to same selection/size.
        need(len(base_complete)==len(cand_complete)==1,'coupled capture completion record count changed')
        bc=base_complete[0][1]; cc=cand_complete[0][1]
        for key in ('schema','root','iteration','bytes','arrays'):
            need(bc.get(key)==cc.get(key),f'capture completion value changed: {key}')
        completion={k:bc[k] for k in ('schema','root','iteration','bytes','arrays')}
    else:
        need(not base_complete and not cand_complete,'unexpected coupled-capture completion line in fixture log')
        completion=None
    return {'armijo_records':numeric_records,'trace_enabled_baseline_schema':1,
            'trace_enabled_candidate_schema':3,'base_status_capture':ce['base_status_capture'],
            'marker_changes':marker_changes,
            'unstructured_lines_exact':True,'capture_completion':completion}


def compare_trace_logs(base_run: Path, cand_run: Path):
    return compare_trace_files(base_run/'stderr.log',cand_run/'stderr.log',1201,True)


def compare_diagnostic_state(base: Path, cand: Path):
    bdir=base/'diagnostic-state/dense'; cdir=cand/'diagnostic-state/dense'
    need(bdir.is_dir() and cdir.is_dir(),'missing dense diagnostic state output')
    bf={p.name:p for p in bdir.iterdir() if p.is_file()}; cf={p.name:p for p in cdir.iterdir() if p.is_file()}
    need(set(bf)==set(cf), 'diagnostic-state file inventory differs')
    marker_changes=[]; binary_count=0; metadata_count=0; logs=[]; fingerprint_deltas=[]
    for name in sorted(bf):
        if name.endswith('.bin'):
            old=bf[name].read_bytes(); new=cf[name].read_bytes()
            fingerprint_deltas.append(compare_snapshot_payload_bytes(old,new,name))
            binary_count+=1
        elif name.startswith('transaction-') and name.endswith('.json'):
            old=strict_json(bf[name]); new=strict_json(cf[name])
            compare_json_exact_with_marker(old,new,name,marker_changes)
            metadata_count+=1
        elif name.endswith('.log'):
            match=re.fullmatch(r'fixture-free-body-control-(\d+)\.log',name)
            need(match is not None,'unclassified diagnostic log file')
            trace_report=compare_trace_files(bf[name],cf[name],int(match.group(1)),int(match.group(1))==1201)
            logs.append({'file':name,**trace_report})
        else:
            need(bf[name].read_bytes()==cf[name].read_bytes(),f'unclassified diagnostic state file differs: {name}')
    # A changed marker must be tied to actual code-6 status and source-proven pair identity.
    for change in marker_changes:
        # Recursive path includes a status array index. Validate the owning transaction JSON status context.
        match=re.fullmatch(r'(transaction-\d+\.json)\.after\.statuses\[(\d+)\]\.diagnostics\[3\]',change['path'])
        need(match is not None,'marker changed outside after-state status diagnostics.w')
        fname,index=match.group(1),int(match.group(2))
        need(fname in cf, f'marker context transaction not found: {fname}')
        tx=strict_json(cf[fname]); status_list=tx.get('after',{}).get('statuses',[])
        need(index < len(status_list),'marker status index is out of range')
        st=status_list[index]
        need(st.get('code')==6 and st.get('diagnostics',[None,None,None,None])[3] == change['candidate'],
             'diagnostic marker was not attached to the captured code-6 VT status')
        d=st['diagnostics']; need(d[0]==92 and d[1]==127 and 0<=d[2]<=2,
            'VT marker attached to unexpected primitive pair/vertex')
    return {'binary_files_exact':binary_count,'transaction_metadata_files':metadata_count,
            'marker_changes':marker_changes,'fixture_trace_logs':logs,
            'raw_snapshot_allowed_device_fingerprint_deltas':fingerprint_deltas,
            'all_production_and_accepted_history_payload_bytes_exact_except_device_fingerprint':True}


def audit(base_run: Path, cand_run: Path):
    bindings=source_bindings()
    br,bp,bb=validate_run(base_run,SOURCE103,'359'); cr,cp,cb=validate_run(cand_run,SOURCE104,'362')
    need(br['exit_code']==cr['exit_code'],'native exit code differs')
    prediction_comparison=compare_predictions(bp,cp,bb,cb)
    capture=compare_capture(base_run,cand_run)
    logs=compare_trace_logs(base_run,cand_run)
    diag=compare_diagnostic_state(base_run,cand_run)
    return {
      'schema':'numi.cardboard.source104_vs_source103_native_parity.v1',
      'status':'PASS_EXACT_PARITY_WITH_DECLARED_DIAGNOSTIC_DELTAS',
      'baseline_run':str(base_run),'candidate_run':str(cand_run),
      'baseline_receipt_sha256':sha(base_run/'receipt.json'),
      'candidate_receipt_sha256':sha(cand_run/'receipt.json'),
      'baseline_prediction_sha256':sha(base_run/'prediction.json'),
      'candidate_prediction_sha256':sha(cand_run/'prediction.json'),
      'exit_code':br['exit_code'],'timed_out':False,
      'source_bindings':bindings,'capture':capture,'armijo_trace':logs,
      'prediction_protocol_comparison':prediction_comparison,
      'diagnostic_state':diag,
      'limits':['This is same-build-output parity for a failed-prefix diagnostic run, not solver convergence or physical qualification.',
                'VT marker allowance is only code-6 diagnostics.w 0 -> 3/4 for primitive pair 92/127 and vertex 0..2.',
                'Schema-3 health fields are checked for internal consistency; pre-existing values remain exact.',
                'Post-fatal raw residual/status observations remain diagnostic, not qualified physical residuals.']}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline-run',type=Path,default=BASELINE)
    ap.add_argument('--candidate-run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ns=ap.parse_args()
    try:
        result=audit(ns.baseline_run.resolve(),ns.candidate_run.resolve())
        ns.output.parent.mkdir(parents=True,exist_ok=True)
        ns.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'status':result['status'],'output':str(ns.output),'armijo_records':result['armijo_trace']['armijo_records'],
                          'capture_arrays':result['capture']['array_count'],'binary_files':result['diagnostic_state']['binary_files_exact'],
                          'vt_marker_changes':len(result['diagnostic_state']['marker_changes'])},sort_keys=True))
        return 0
    except Exception as exc:
        result={'schema':'numi.cardboard.source104_vs_source103_native_parity.v1','status':'FAIL_CLOSED',
                'baseline_run':str(ns.baseline_run),'candidate_run':str(ns.candidate_run),'error':str(exc)}
        ns.output.parent.mkdir(parents=True,exist_ok=True)
        ns.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'status':'FAIL_CLOSED','error':str(exc),'output':str(ns.output)},sort_keys=True),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
