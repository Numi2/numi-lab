from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
B = Path('/Users/home/cardboard-implementation-20261014')
S = B / 'snapshots/source-reference-jacobian-168'
T = B / 'snapshots/source-warm-cgls-refinement-169'
P = B / 'analysis/base-cgls-warm-start-refinement-overlay-159-003'
O = B / 'analysis/source169-warm-start-integration-001'
assert not T.exists() and not O.exists()
T.mkdir()
O.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P / 'source-overlay.json') == '804ce6bda0849792130d69996446ec18bf1ec7ed17af7e8509aba0e590072f52'
assert sha(P / 'overlay.patch') == '9a2abf35db908aa19b1fb2a4d017079eac2c730b0df03cdc9f2964f008aa04b6'
frozen = json.loads((S / 'frozen-source.json').read_text())
support = json.loads((S / 'configure-support-source.json').read_text())
files = dict(frozen['source_sha256'])
files.update(support['files'])
try:
    for name, expected in files.items():
        assert sha(S / name) == expected, name
        destination = T / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(S / name, destination)
    command = ['git', 'apply', str(P / 'overlay.patch')]
    applied = subprocess.run(command, cwd=T, capture_output=True)
    (O / 'patch.stdout.log').write_bytes(applied.stdout)
    (O / 'patch.stderr.log').write_bytes(applied.stderr)
    assert applied.returncode == 0, applied.stderr.decode()
    for name in ['cardboard_base_cgls_warm_start_check.metal', 'cardboard_base_cgls_warm_start_check.mm']:
        original = 'warm_start_check' + Path(name).suffix
        shutil.copy2(P / 'native' / original, T / 'matter/tools' / name)
    path = T / 'matter/tools/cardboard_solver_environment.py'
    text = path.read_text()
    old = '    "NM_FEM_BASE_CGLS_STATE_CAPTURE",\n'
    assert text.count(old) == 1
    text = text.replace(old, old + '    "NM_FEM_BASE_CGLS_WARM_START_REFINE",\n')
    text = text.replace('# Extracted from the combined source160-002 runtime and source157 probe',
                        '# Extracted from the combined source169 runtime and source167 probe')
    path.write_text(text)
    path = T / 'matter/tools/test_cardboard_solver_environment.py'
    text = path.read_text()
    assert text.count('self.assertEqual(len(env.NATIVE_ENV_KEYS), 49)') == 1
    path.write_text(text.replace('self.assertEqual(len(env.NATIVE_ENV_KEYS), 49)',
                                'self.assertEqual(len(env.NATIVE_ENV_KEYS), 50)'))
    cmake = T / 'matter/CMakeLists.txt'
    text = cmake.read_text()
    anchor = '    add_executable(numi-matter-fem-static-surface-map-native-check tools/fem_static_surface_map_native_check.mm)\n'
    assert text.count(anchor) == 1
    addition = '''    set(NUMI_CARDBOARD_WARM_CHECK_AIR "${NUMI_MATTER_SHADER_DIR}/CardboardWarmStartCheck.air")
    set(NUMI_CARDBOARD_WARM_CHECK_METALLIB "${NUMI_MATTER_SHADER_DIR}/CardboardWarmStartCheck.metallib")
    add_custom_command(
        OUTPUT "${NUMI_CARDBOARD_WARM_CHECK_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_MATTER_SHADER_DIR}"
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src/metal"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_base_cgls_warm_start_check.metal"
            -o "${NUMI_CARDBOARD_WARM_CHECK_AIR}"
        COMMAND xcrun -sdk macosx metallib "${NUMI_CARDBOARD_WARM_CHECK_AIR}"
            -o "${NUMI_CARDBOARD_WARM_CHECK_METALLIB}"
        DEPENDS tools/cardboard_base_cgls_warm_start_check.metal
            src/metal/common.metalinc src/metal/fem.metalinc
            src/metal/base_cgls.metalinc src/metal/base_cgls_runtime.metalinc
            src/metal/fgmres.metalinc include/numi/matter/shared.h
        VERBATIM)
    add_custom_target(numi_cardboard_warm_start_check_shaders
        DEPENDS "${NUMI_CARDBOARD_WARM_CHECK_METALLIB}")
    add_executable(numi-matter-cardboard-warm-start-check
        tools/cardboard_base_cgls_warm_start_check.mm)
    add_dependencies(numi-matter-cardboard-warm-start-check numi_cardboard_warm_start_check_shaders)
    target_include_directories(numi-matter-cardboard-warm-start-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(numi-matter-cardboard-warm-start-check PRIVATE "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_definitions(numi-matter-cardboard-warm-start-check PRIVATE
        NUMI_WARM_START_METALLIB="${NUMI_CARDBOARD_WARM_CHECK_METALLIB}")
    target_compile_options(numi-matter-cardboard-warm-start-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_warm_start_kernel_contract
            COMMAND numi-matter-cardboard-warm-start-check)
        set_tests_properties(matter.metal.cardboard_warm_start_kernel_contract PROPERTIES
            LABELS "matter;cardboard;solver;metal;synthetic" TIMEOUT 60 RUN_SERIAL TRUE)
    endif()
'''
    cmake.write_text(text.replace(anchor, addition + anchor))
    spec = importlib.util.spec_from_file_location('builder169', T / 'matter/tools/cardboard_source_bound_build.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    selected = builder.source_bindings(T)
    assert len(selected) == 804
    unchanged_support = {name: sha(T / name) for name in files if name not in selected}
    (T / 'frozen-source.json').write_text(json.dumps(dict(frozen, base_snapshot=str(S), source_sha256=selected,
        scope='Default-off warm-start CGLS refinement candidate plus native kernel checker. No native trajectory or feature-off parity qualification.'), indent=2) + '\n')
    (T / 'configure-support-source.json').write_text(json.dumps(dict(support, files=unchanged_support), indent=2) + '\n')
    changes = {name: digest for name, digest in selected.items() if frozen['source_sha256'].get(name) != digest}
    report = {'schema': 'numi.cardboard.warm-start-integrated-source.v1', 'source': str(T),
              'base_frozen_sha256': sha(S / 'frozen-source.json'), 'base_files_verified': len(files),
              'selected_source_count': len(selected), 'frozen_source_sha256': sha(T / 'frozen-source.json'),
              'component_manifest_sha256': sha(P / 'source-overlay.json'),
              'applied_patch_sha256': sha(P / 'overlay.patch'), 'changed_or_added': changes,
              'registered_native_controls': 50, 'gpu_execution': False,
              'qualification': 'Experimental conditioning candidate; physical operator and nonlinear gates unchanged. Candidate must start from0; old checkpoint environment digest cannot change.'}
    (O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
except Exception as error:
    (O / 'failure.json').write_text(json.dumps({'error': repr(error)}, indent=2) + '\n')
    raise
shutil.copyfile(__file__, O / 'prepare.py')
print(json.dumps(report))
