#!/usr/bin/env python3
"""CPU-only source/evidence contract checks; does not launch a Metal kernel."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

BASE_CONTACT_SHA = 'b9ec3915b63bdee729ff0902c390e137e0f6b5d005d06e6a7136c534be3d510b'
BASE_CHECKER_SHA = 'bd2e0ef37446183c22e964a0517114f376e84e6fc85875c8f8946a92962e5528'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_body(source: str, signature: str, occurrence: int) -> str:
    starts = [m.start() for m in re.finditer(re.escape(signature), source)]
    if len(starts) <= occurrence:
        raise AssertionError(f'missing {signature} occurrence {occurrence}; found {len(starts)}')
    start = starts[occurrence]
    brace = source.find('{', start)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    raise AssertionError('unterminated function body')


def evidence_hash_source103_contact(evidence_path: Path) -> str:
    return json.loads(evidence_path.read_text())['source_artifacts']['source103_contact_source_sha256']

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--source103', type=Path, required=True)
    args = parser.parse_args()

    base_contact = args.base/'matter/src/metal/contact.metalinc'
    base_checker = args.base/'matter/tools/cardboard_contact_ccd_native_check.mm'
    overlay_contact = args.overlay/'matter/src/metal/contact.metalinc'
    overlay_checker = args.overlay/'matter/tools/cardboard_contact_ccd_native_check.mm'
    assert sha(base_contact) == BASE_CONTACT_SHA
    assert sha(base_checker) == BASE_CHECKER_SHA

    base_text = base_contact.read_text()
    overlay_text = overlay_contact.read_text()
    # The file has a helper definition and the production definition. Assert
    # that only production EE's loop bound changed; VT and all other bytes stay.
    production_old = function_body(base_text, 'inline bool edgeEdgeCCD(', 1)
    production_new = function_body(overlay_text, 'inline bool edgeEdgeCCD(', 1)
    assert production_old.count('iteration < 17u') == 1
    assert production_new.count('iteration < 33u') == 1
    assert production_old.replace('iteration < 17u', 'iteration < BOUNDu') == \
           production_new.replace('iteration < 33u', 'iteration < BOUNDu')
    assert overlay_text == base_text.replace(
        'for (uint iteration = 0u; iteration < 17u; ++iteration)',
        'for (uint iteration = 0u; iteration < 33u; ++iteration)', 1)
    assert 'iteration < 16u' in function_body(overlay_text, 'inline bool vertexTriangleCCD(', 1)
    evidence = json.loads(args.evidence.read_text())
    assert str(args.source103) == evidence['source_artifacts']['source103_contact_source']
    source103_text = args.source103.read_text()
    source103_production = function_body(source103_text, 'inline bool edgeEdgeCCD(', 1)
    assert sha(args.source103) == evidence_hash_source103_contact(args.evidence)
    assert production_old.replace('iteration < 17u', 'iteration < BOUNDu') == \
           source103_production.replace(re.search(r'iteration < \d+u', source103_production).group(0), 'iteration < BOUNDu')

    case = evidence['capture_case']
    comparison = evidence['native_source103_comparison']
    assert case['name'] == 'trial358_ee_edge0_edge0'
    assert case['primitive_pair'] == [92, 127]
    assert case['record_f32_bits'][0] == '3f800000'
    assert case['record_f32_bits'][1] == '3727c5ac'
    assert len(case['points_start_f32_bits']) == 12
    assert len(case['points_finish_f32_bits']) == 12
    assert comparison['iteration_bound_baseline'] == 17
    assert comparison['iteration_bound_continuation'] == 33
    assert comparison['baseline']['certified'] is False
    assert comparison['baseline']['hit'] is False
    assert comparison['baseline']['trace_rows'] == 17
    assert comparison['baseline']['terminal_iteration'] == 16
    assert comparison['continuation']['certified'] is True
    assert comparison['continuation']['hit'] is True
    assert comparison['continuation']['trace_rows'] == 18
    assert comparison['continuation']['terminal_iteration'] == 17
    assert comparison['baseline_and_continuation_input_corpus_match'] is True
    assert evidence['geometry_interpretation']['nominal_crossing_claimed'] is False

    checker = overlay_checker.read_text()
    assert 'capturedNearBandEdgeEdgeExhaustion(0.0f)' in checker
    assert 'capturedNearBandEdgeEdgeExhaustion(2.0f)' in checker
    assert 'expectRoundoffBandHit' in checker
    assert 'near_band_only_not_nominal_crossing' in checker
    assert 'contact-regional-all-pair-gate-092/slot129' in checker
    assert 'captured_ee_input_corpus_sha256' in checker
    assert 'full_contact_same_build_required' in checker
    # The C++ literals must recreate all 24 point words exactly.
    pairs = re.findall(r'pointFromBits\(0x([0-9a-f]+)u, 0x([0-9a-f]+)u, 0x([0-9a-f]+)u\)', checker)
    actual = [word for triple in pairs[-8:] for word in triple]
    expected = case['points_start_f32_bits'] + case['points_finish_f32_bits']
    assert actual == expected, (actual, expected)
    direct_words = ['3f800000', '3727c5ac', '00000000', '00000000']
    for group in (case['points_start_f32_bits'], case['points_finish_f32_bits']):
        for point in range(4):
            direct_words.extend(group[3 * point:3 * point + 3])
            direct_words.append('00000000')
    assert direct_words == case['record_f32_bits'], (direct_words, case['record_f32_bits'])
    assert hashlib.sha256(b''.join(bytes.fromhex(x)[::-1] for x in case['record_f32_bits'])).hexdigest() == \
           case['record_bytes_sha256']
    assert 'captured_ee_case_record_sha256' in checker
    assert 'sampled_minimum_is_clearance_proof\\\":false' in checker
    # Failure remains the historical measured fact; the new target case is a
    # roundoff-band certification expectation, not a nominal crossing gate.
    assert comparison['baseline']['separation_m'] > evidence['geometry_interpretation']['nominal_thickness_m']
    assert evidence['geometry_interpretation']['returned_separation_m'] <= \
           evidence['geometry_interpretation']['continuation_effective_thickness_m']
    assert evidence['geometry_interpretation']['independent_sampled_minimum_m'] < \
           evidence['geometry_interpretation']['nominal_thickness_m']
    for label in ('baseline', 'continuation'):
        native = comparison[label]
        raw = Path(evidence['source_artifacts'][label + '_native_result'])
        assert sha(raw) == evidence['source_artifacts'][label + '_native_result_sha256']
        parsed = json.loads(raw.read_text())
        assert parsed['cases_bin_sha256'] == comparison['source103_same_input_corpus_sha256']
        assert parsed['source_contact_sha256'] == native['source_contact_sha256']
        assert parsed['metallib_sha256'] == native['metallib_sha256']
        assert parsed['physical_transactions_executed'] == 0
    print(json.dumps({
        'status': 'pass',
        'scope': 'CPU source and captured-evidence contract only; no Metal dispatch',
        'base_contact_sha256': sha(base_contact),
        'base_checker_sha256': sha(base_checker),
        'overlay_contact_sha256': sha(overlay_contact),
        'overlay_checker_sha256': sha(overlay_checker),
        'captured_case': case['name'],
        'baseline17': 'uncertified at evaluation 16',
        'continuation33': 'certified at evaluation 17, roundoff band only',
    }, indent=2))

if __name__ == '__main__':
    main()
