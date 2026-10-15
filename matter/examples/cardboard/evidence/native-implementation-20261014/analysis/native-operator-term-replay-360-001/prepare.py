from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess

B = Path('/Users/home/cardboard-implementation-20261014')
P = B / 'analysis/native-cached-operator-replay-360-003'
O = B / 'analysis/native-operator-term-replay-360-001'
O.mkdir(exist_ok=False)
(O/'build').mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'package-manifest.json') == '756f064c072cc292f99bc1ebefbfa4868a9fc37ff75541c26659f4be696f27e9'
old = (P/'cached_operator_replay.mm').read_text()
new = old
def replace(a, b):
    global new
    assert new.count(a) == 1, a[:100]
    new = new.replace(a, b)
replace('struct NativeResult {\n    std::vector<nm_float4> action;',
        'struct NativeResult {\n    std::vector<NMFEMElementVectorGPU> elements;\n    std::vector<nm_float4> action;')
replace('const std::uint32_t useCache) {', 'const std::uint32_t useCache,\n                            const bool massOnly = false) {')
replace('    // Production nodal gather: M*delta_v - dt^2*dF, with the authored fixed',
        '    // Isolate the unchanged gather mass term after checking the real bulk action.\n'
        '    // The normal cached/fresh paths never change this buffer.\n'
        '    if (massOnly) std::memset(elementB.contents, 0, elementB.length);\n\n'
        '    // Production nodal gather: M*delta_v - dt^2*dF, with the authored fixed')
replace('    NativeResult result;\n    result.action.assign',
        '    NativeResult result;\n'
        '    const auto* elementValues = static_cast<const NMFEMElementVectorGPU*>(elementB.contents);\n'
        '    result.elements.assign(elementValues, elementValues + std::size_t(eCount) * tCount);\n'
        '    result.action.assign')
replace('void writeReport(const std::string& path, const Payload& p,',
        'void writeElementFile(const std::string& path, const std::vector<NMFEMElementVectorGPU>& values) {\n'
        '    static_assert(sizeof(NMFEMElementVectorGPU) == 64u);\n'
        '    std::ofstream out(path, std::ios::binary);\n'
        '    need(out.good(), "cannot open element output file");\n'
        '    out.write(reinterpret_cast<const char*>(values.data()),\n'
        '        static_cast<std::streamsize>(values.size() * sizeof(NMFEMElementVectorGPU)));\n'
        '    need(out.good(), "failed writing element output file");\n'
        '}\n\n'
        'void writeReport(const std::string& path, const Payload& p,')
replace('            const Metrics env0 = metricsFor',
        '            NativeResult mass = evaluateNative(device, femOperator, gatherNodes,\n'
        '                world, p, contactEncoder, 1u, true);\n'
        '            const Metrics env0 = metricsFor')
replace('            writeReport(outputDir + "/replay-report.json", p, env0, env1);',
        '            writeElementFile(outputDir + "/cached-elements.float4x4.bin", cached.elements);\n'
        '            writeElementFile(outputDir + "/fresh-elements.float4x4.bin", fresh.elements);\n'
        '            writeFloat4File(outputDir + "/mass-only-action.float4.bin", mass.action);\n'
        '            writeReport(outputDir + "/replay-report.json", p, env0, env1);')
(O/'operator_terms_replay.mm').write_text(new)
m = json.loads((P/'package-manifest.json').read_text())
cmd = [v.replace(str(P/'cached_operator_replay.mm'), str(O/'operator_terms_replay.mm'))
       .replace(str(P/'build/cached_operator_replay'), str(O/'build/operator_terms_replay'))
       for v in m['compile']['command']]
with (O/'build/compile.stdout.log').open('x') as out, (O/'build/compile.stderr.log').open('x') as err:
    result = subprocess.run(cmd, stdout=out, stderr=err, timeout=120)
assert result.returncode == 0
preflight = [str(O/'build/operator_terms_replay'),
    str(B/'solver/reset-augmentation-first-failure-capture-360/output/compiled.nmatterpack'),
    str(P/'prepared/replay-input.bin'), '/tmp/cardboard-term-preflight-unused', '--preflight-only']
with (O/'preflight.stdout.log').open('x') as out, (O/'preflight.stderr.log').open('x') as err:
    check = subprocess.run(preflight, stdout=out, stderr=err, timeout=30)
assert check.returncode == 0
shutil.copy2(__file__, O/'prepare.py')
(O/'README.md').write_text('''# Native operator term replay

This diagnostic uses the exact source101 production Metal element operator and
nodal gather with the sealed capture360 payload from package003. Cached and
fresh normal paths are unchanged; raw four-node element directional forces are
also exported. A third call zeroes only its private element-output buffer before
the same gather, isolating mass. No constitutive or accepted history is changed.
No Matter transaction is executed. The mass-only control is an operator-term
diagnostic, not a physical model. Normal combined actions must exactly reproduce
native368 before interpreting the terms. Force sign is unchanged:
A(delta_v) = M(delta_v) - dt^2 * d(force)/d(delta_v).

Element arrays have little-endian float32 shape(2,1920,4,4), four node slots
with xyz forces and w pressure. Nodal arrays have shape(2,882,4). The native
binary accepts compiled-package, prepared-payload, output-directory; optional
--preflight-only performs immutable input checks without creating a Metal device.
Compilation and CPU preflight do not dispatch the diagnostic.
''')
files = {str(p.relative_to(O)): dict(bytes=p.stat().st_size, sha256=sha(p))
         for p in sorted(O.rglob('*')) if p.is_file()}
manifest = dict(schema='numi.cardboard.native-operator-term-package.v1',
    base_package=str(P), base_package_manifest_sha256=sha(P/'package-manifest.json'),
    source_snapshot=m['source_snapshot'] if 'source_snapshot' in m else m['inputs']['metallib']['path'],
    compile_argv=cmd, compile_exit_code=result.returncode,
    preflight_argv=preflight, preflight_exit_code=check.returncode,
    native_dispatched=False, physical_transactions=0, files=files,
    payload_sha256=sha(P/'prepared/replay-input.bin'),
    production_metallib_sha256=m['inputs']['metallib']['sha256'],
    created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(package=str(O), manifest_sha256=sha(O/'manifest.json'),
    binary_sha256=sha(O/'build/operator_terms_replay'), compile_exit_code=0, preflight_exit_code=0)))
