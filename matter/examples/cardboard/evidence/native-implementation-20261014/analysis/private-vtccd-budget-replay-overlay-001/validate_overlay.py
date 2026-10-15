#!/usr/bin/env python3
"""CPU-only source/capture contract for the exact-helper budget replay overlay."""
from __future__ import annotations
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).parent
BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014/snapshots/source-native-tet858-vtccd-traces-122')
NATIVE = pathlib.Path('/Users/home/cardboard-implementation-20261014/native')
BASE_CONTACT_SHA = 'ea5a9caf6f47588258aeeb5e8be57168826acfb94a566ecbfea7e647ad50fb5c'
BASE_CMAKE_SHA = 'd64689e33063913c13493d6e621767124519c29432b45ee236deab01ca1c7448'
BASE_KERNELS_SHA = '37d3530284e2361b20f805409768c65446fc13186c50e94f286b73256debe168'
BUDGETS = [16, 32, 64, 128]

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def strict_json(path: pathlib.Path):
    return json.loads(path.read_text(), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))

def main() -> None:
    generator_path = ROOT/'generate_overlay.py'
    generator_namespace = {'__file__': str(generator_path), '__name__': '_overlay_generator'}
    exec(compile(generator_path.read_text(), str(generator_path), 'exec'), generator_namespace)
    base_path = BASE/'matter/src/metal/contact.metalinc'
    base_bytes = base_path.read_bytes()
    assert sha(base_bytes) == BASE_CONTACT_SHA
    expected = generator_namespace['transform'](base_bytes.decode('utf-8')).encode('utf-8')
    candidate_path = ROOT/'overlay/matter/src/metal/contact.metalinc'
    candidate_bytes = candidate_path.read_bytes()
    assert candidate_bytes == expected, 'candidate differs from deterministic source122 transformation'
    text = candidate_bytes.decode('utf-8')
    assert 'vertexTriangleCCDBudgetCore<16u, false>' in text
    assert 'for (uint iteration = 0u; iteration < IterationBudget; ++iteration)' in text
    for budget in BUDGETS:
        assert f'runVertexTriangleCCDBudgetReplay<{budget}u>' in text
    assert 'time_fraction' not in text  # trace ABI lane is positional and documented by the host checker
    assert text.count('kernel void nm_contact_vtccd_iteration_budget_replay(') == 1
    assert text.count('inline bool vertexTriangleCCD(') == 2  # forward declaration + original wrapper

    base_cmake = (BASE/'matter/CMakeLists.txt').read_text()
    assert sha(base_cmake.encode()) == BASE_CMAKE_SHA
    anchor = '    add_executable(numi-matter-cardboard-full-contact-armijo-check tools/cardboard_full_contact_armijo_check.mm)\n'
    block = '''    # Standalone exact-input VTCCD budget replay. This target is deliberately
    # not registered as a CTest because it consumes retained native capture files.
    add_executable(numi-matter-cardboard-vtccd-budget-replay-check
        tools/cardboard_vtccd_iteration_budget_check.mm)
    add_dependencies(numi-matter-cardboard-vtccd-budget-replay-check numi_matter_shaders)
    target_link_libraries(numi-matter-cardboard-vtccd-budget-replay-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-vtccd-budget-replay-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardboard-vtccd-budget-replay-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-vtccd-budget-replay-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
'''
    assert base_cmake.count(anchor) == 1
    cmake = (ROOT/'overlay/matter/CMakeLists.txt').read_text()
    assert cmake == base_cmake.replace(anchor, block + anchor, 1)
    assert cmake.count('numi-matter-cardboard-vtccd-budget-replay-check') == 6
    assert 'add_test(NAME matter.metal.cardboard_vtccd_budget_replay' not in cmake
    host = (ROOT/'overlay/matter/tools/cardboard_vtccd_iteration_budget_check.mm').read_text()
    assert 'constexpr std::array<std::uint32_t, 4> kBudgets{16u, 32u, 64u, 128u}' in host
    assert 'budget16 replay is not bitwise equal to the captured production result' in host
    assert 'gap_to_effective_thickness_m' in host

    base_kernels = (BASE/'matter/src/matter_kernels.metal').read_bytes()
    overlay_kernels = (ROOT/'overlay/matter/src/matter_kernels.metal').read_bytes()
    assert sha(base_kernels) == BASE_KERNELS_SHA and overlay_kernels == base_kernels

    # Bind both exact native inputs without rewriting their retained report.
    summaries = []
    for trial in (21, 22):
        run = NATIVE/f'full-contact-trial{trial}-vtccd-call-{384+trial}'
        manifest_path = run/'trial-capture.coupled.json'
        payload_path = run/'trial-capture.coupled.bin'
        manifest = strict_json(manifest_path)
        payload = payload_path.read_bytes()
        assert manifest['schema'] == 'numi.matter.coupled_armijo_trial_capture.v1'
        assert manifest['source_physics_fingerprint'] == '915f50b10edc2e46'
        assert manifest['device_program_fingerprint'] == '28f5c6c60a7f247f'
        assert manifest['payload']['byte_count'] == len(payload)
        assert manifest['payload']['sha256'] == sha(payload)
        specs = [a for a in manifest['arrays'] if a['name'] == 'actual_vertex_triangle_ccd_call']
        assert len(specs) == 1 and specs[0]['count'] == 1 and specs[0]['element_bytes'] == 336
        spec = specs[0]
        raw = payload[spec['offset']:spec['offset']+spec['byte_count']]
        assert len(raw) == 336 and sha(raw) == spec['sha256']
        words = json.loads(json.dumps(list(__import__('struct').unpack('<16I68f', raw))))
        assert words[:16] == [1,0,173,5,36,123,173,1,13,10,14,0,28,29,32,0]
        expected_result = words[64:68]
        assert (expected_result[0:2] == [0.0,0.0]) if trial == 21 else (expected_result[0:2] == [1.0,1.0])
        summaries.append({'trial':trial,'manifest_sha256':sha(manifest_path.read_bytes()),
            'payload_sha256':sha(payload),'observer_sha256':sha(raw),
            'observed_return':expected_result})
    assert summaries[0]['observer_sha256'] != summaries[1]['observer_sha256']
    print(json.dumps({'status':'PASS_CPU_SOURCE_AND_CAPTURE_CONTRACT',
        'base_contact_sha256':BASE_CONTACT_SHA,
        'candidate_contact_sha256':sha(candidate_bytes),
        'budgets':BUDGETS,'captures':summaries},sort_keys=True))

if __name__ == '__main__':
    main()
