#!/usr/bin/env python3
"""Static source-bound checks for the bounded spec-diagnostic overlay 003."""
from pathlib import Path
import hashlib
import json
import re
import sys

CHECK_COUNT = 0

ROOT = Path('/Users/home/cardboard-implementation-20261014')
BASE = ROOT / 'snapshots/source-selected-private-trial-repair-094/matter/src/runtime.mm'
OVERLAY = ROOT / 'analysis/coupled-trial-capture-overlay-003/overlay/matter/src/runtime.mm'
CROSSWALK = ROOT / 'analysis/coupled-trial-capture-overlay-003/source291_capture_crosswalk.json'
CAPTURE291 = ROOT / 'analysis/retained-coupled-failure-capture-291/candidate-capture.coupled.json'
PACKAGE291 = ROOT / 'analysis/retained-coupled-failure-capture-291/package.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(ok, message):
    global CHECK_COUNT
    CHECK_COUNT += 1
    if not ok:
        raise AssertionError(message)

def main():
    base = BASE.read_text()
    text = OVERLAY.read_text()
    require(sha(BASE) == '2fbdf2c09e5e27fadf15f6503e56d3dc042215eabf1029b4ebd3aa08e143b2e3', 'wrong frozen-094 base')
    marker = '                std::uint64_t phaseBytes = 0u;'
    end_marker = '                if (present == 0u) {'
    base_start = base.index(marker, base.index('const auto appendTrialCapturePhase'))
    overlay_start = text.index(marker, text.index('const auto appendTrialCapturePhase'))
    base_end = base.index(end_marker, base_start)
    overlay_end = text.index(end_marker, overlay_start)
    # The only edit outside the validation block is the corrected buffer schema
    # for the candidate topology upload. Normalize it back to source 094 before
    # comparing the rest of the file byte-for-byte.
    corrected = (
        '                    push("fem_topology_nodes_candidate",\n'
        '                        "record:NMFEMTopologyNodeGPU", sizeof(NMFEMTopologyNodeGPU),\n'
        '                        state.femTopologyNodesCandidate);')
    original = (
        '                    push("fem_topology_nodes_candidate",\n'
        '                        "record:NMFEMNodeStateGPU", sizeof(NMFEMNodeStateGPU),\n'
        '                        state.femTopologyNodesCandidate);')
    require(text.count(corrected) == 1, 'candidate topology capture schema is not uniquely corrected')
    normalized = text.replace(corrected, original, 1)
    inverse_correct = (
        '                    push("coupled_inverse_statuses", "record:MRInverseMassStatusGPU",\n'
        '                        sizeof(MRInverseMassStatusGPU), state.coupledInverseStatuses);')
    inverse_base = (
        '                    push("coupled_inverse_statuses", "record:NMMatterStatusGPU",\n'
        '                        sizeof(NMMatterStatusGPU), state.coupledInverseStatuses);')
    require(text.count(inverse_correct) == 1, 'inverse-status capture schema is not uniquely corrected')
    normalized = normalized.replace(inverse_correct, inverse_base, 1)
    normalized_start = normalized.index(marker, normalized.index('const auto appendTrialCapturePhase'))
    normalized_end = normalized.index(end_marker, normalized_start)
    require(base[:base_start] == normalized[:normalized_start], 'unexpected prefix edits')
    require(base[base_end:] == normalized[normalized_end:], 'unexpected suffix edits')
    block = text[overlay_start:overlay_end]
    trial_start = text.index('const auto appendTrialContext')
    trial_end = text.index('const auto captureTrialMeasurement', trial_start)
    trial_context = text[trial_start:trial_end]
    require(corrected in trial_context, 'candidate topology capture must use NMFEMTopologyNodeGPU')
    require('push("coupled_inverse_statuses", "record:MRInverseMassStatusGPU",'
            in trial_context and 'sizeof(MRInverseMassStatusGPU), state.coupledInverseStatuses)' in trial_context,
            'inverse-status capture must match its actual allocation element type')
    require('privateScratch<MRInverseMassStatusGPU>' in base,
            'source 094 inverse-status allocation type changed')
    require('std::span<const NMFEMTopologyNodeGPU>(world.fem.topologyNodes)' in base,
            'source 094 topology-node allocation type changed')
    crosswalk = json.loads(CROSSWALK.read_text())
    require(sha(CAPTURE291) == crosswalk['source_capture_manifest_sha256'],
            'source-291 capture manifest hash mismatch')
    require(sha(PACKAGE291) == crosswalk['source_capture_package_sha256'],
            'source-291 capture package hash mismatch')
    capture = json.loads(CAPTURE291.read_text())
    actual = {row['name']: row for row in capture['arrays']}
    topo = actual['fem_topology_nodes_candidate']
    require((topo['dtype'], topo['count'], topo['element_bytes'], topo['byte_count']) ==
            ('record:NMFEMTopologyNodeGPU', 36, 16, 576),
            '291 topology candidate extent crosswalk changed')
    inverse = actual['coupled_inverse_statuses']
    require((inverse['byte_count'], inverse['element_bytes'], inverse['count']) ==
            (96, 4, 24), '291 legacy inverse-status byte extent changed')
    # Cross-check every literal trial-context spec whose name appears in the
    # complete 291 capture. The old whole-state capture mislabeled one buffer
    # as uint32; its source allocation is MRInverseMassStatusGPU and is checked
    # against its exact byte extent separately below.
    push_pattern = re.compile(
        r'push\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*sizeof\(([^)]+)\)')
    pushes = push_pattern.findall(trial_context)
    type_sizes = {}
    for row in capture['arrays']:
        if row['dtype'].startswith('record:'):
            type_sizes[row['dtype'][7:]] = row['element_bytes']
    type_sizes.update({'float': 4, 'std::uint32_t': 4, 'nm_float4': 16,
                       'nm_uint4': 16, 'MRInverseMassStatusGPU': 48})
    matched = [(name, dtype, cpp_type) for name, dtype, cpp_type in pushes
               if name in actual]
    require(len(matched) == 27, 'source-291 direct-name crosswalk changed')
    for name, dtype, cpp_type in matched:
        require(cpp_type in type_sizes, f'unknown record size in trial spec {name}: {cpp_type}')
        require((name == 'coupled_inverse_statuses' or
                 (actual[name]['dtype'] == dtype and
                  actual[name]['element_bytes'] == type_sizes[cpp_type])),
                f'trial spec disagrees with source-291 metadata: {name}')
    require((inverse['byte_count'] % type_sizes['MRInverseMassStatusGPU'] == 0) and
            type_sizes['MRInverseMassStatusGPU'] == 48,
            'legacy inverse-status bytes do not contain whole source allocation records')
    require('captureArrayLimit = 192u' in text and 'femCoupledCaptureCapBytes' in block, 'capture bounds changed')
    for field in ('source_bytes', 'element_bytes', 'phase_bytes_before',
                  'phase_bytes_after_saturating', 'planned_bytes_before',
                  'source_device_registry_id', 'expected_device_registry_id',
                  'failure_flags', 'invalid_spec_count', 'invalid_specs_reported',
                  'invalid_specs_truncated', 'metadata_limit_bytes'):
        require(field in block, f'missing bad-spec field: {field}')
    for flag in ('zero_element_bytes', 'non_integral_element_count',
                 'source_over_capture_cap', 'source_over_device_max_buffer_length',
                 'phase_sum_overflow', 'phase_sum_over_capture_cap',
                 'aggregate_sum_over_capture_cap', 'source_device_missing',
                 'source_device_mismatch'):
        require(f'flag("{flag}"' in block, f'missing invalid-extent reason: {flag}')
    require('invalidSpecMetadataLimit = 192u' in block, 'diagnostic spec-count bound changed')
    require('invalidSpecMetadataByteLimit = 128u * 1024u' in block, 'diagnostic metadata byte bound missing')
    require('femCoupledCaptureCapBytes' in block and 'captureArrayLimit = 192u' in text, 'capture limits missing')
    require('femCoupledTrialCaptureFailed = true;' in block, 'rejection is not latched')
    require('return true;' in block[block.index('if (invalidSpecCount != 0u) {'):], 'rejection must preserve the existing diagnostic-only return')
    require('copyFromBuffer' not in block, 'bad extent path must not blit')
    require(base != text, 'overlay equals base')
    output = {
        'schema': 'numi.matter.coupled_trial_capture_overlay_003_static_check.v1',
        'base_sha256': sha(BASE),
        'overlay_sha256': sha(OVERLAY),
        'crosswalk_sha256': sha(CROSSWALK),
        'source291_direct_spec_names_checked': 27,
        'metadata_spec_limit': 192,
        'metadata_byte_limit': 131072,
        'payload_limit_bytes': 33554432,
        'array_limit': 192,
        'checks': CHECK_COUNT,
        'result': 'PASS_STATIC_ONLY',
        'native_or_physical_qualification': False}
    print(json.dumps(output, sort_keys=True, separators=(',', ':')))

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('FAIL:', error, file=sys.stderr)
        raise SystemExit(1)
