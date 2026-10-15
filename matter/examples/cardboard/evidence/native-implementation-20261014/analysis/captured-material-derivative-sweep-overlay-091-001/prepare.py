from pathlib import Path
import difflib, hashlib, json, shlex, subprocess

base = Path('/Users/home/cardboard-implementation-20261014')
source = base / 'snapshots/source-rationalized-paper-and-native-checkers-091'
relative = Path('matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm')
out = base / 'analysis/captured-material-derivative-sweep-overlay-091-001'
out.mkdir(exist_ok=False)
original = (source / relative).read_text()
helper = (base / 'derivative_sweep_helper.inc').read_text()
text = original.replace('\n}\n\nint main(', '\n' + helper + '\n}\n\nint main(')
assert text != original
needle = '    const bool native=nativeSelected||nativeAll;'
text = text.replace(needle, '''    const bool nativeSweep=argc==9&&std::string(argv[1])=="--native"&&std::string(argv[2])=="--derivative-sweep"&&std::string(argv[3])=="--expected-capture-sha256";
    const bool native=nativeSelected||nativeAll||nativeSweep;''')
text = text.replace('--selected-only|--all-points --expected', '--selected-only|--all-points|--derivative-sweep --expected')
text = text.replace('    if(cpu){\n', '    if(cpu){\n        need(derivativeSweepRecords(capture).size()==960u,"controlled sweep CPU admission failed");\n')
needle = '    Metal metal;\n'
replacement = '''    Metal metal;
    if(nativeSweep){
        runDerivativeSweep(metal,defaults,"default",capture,outdir,capSHA);
        runDerivativeSweep(metal,accurate,"accurate",capture,outdir,capSHA);
        std::cout<<"{\\"record\\":\\"end\\",\\"status\\":\\"controlled_derivative_diagnostic_complete\\",\\"physical_steps\\":0,\\"physical_validation\\":false}\\n";
        return 0;
    }
'''
assert text.count(needle) == 1
text = text.replace(needle, replacement)
target = out / relative.name
target.write_text(text)
(out / 'changes.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile='a/' + str(relative), tofile='b/' + str(relative))))
(out / 'helper.inc').write_text(helper)
(out / 'prepare.py').write_bytes(Path(__file__).read_bytes())
flags = Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-091/matter/CMakeFiles/numi-matter-cardboard-bilinear15-capture-diagnostic-check.dir/flags.make').read_text()
values = {line.split(' = ', 1)[0]: line.split(' = ', 1)[1] for line in flags.splitlines() if ' = ' in line}
command = ['/usr/bin/clang++', '-fsyntax-only']
for key in ('OBJCXX_DEFINES', 'OBJCXX_INCLUDES', 'OBJCXX_FLAGS'):
    command.extend(shlex.split(values[key]))
command.append(str(target))
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(out / 'syntax.log').write_text(result.stdout)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = dict(schema='numi.cardboard.observational-overlay.v1', base_snapshot=str(source),
    destination=str(relative), base_sha256=sha(source / relative), replacement_sha256=sha(target),
    patch_sha256=sha(out / 'changes.patch'), syntax_command=command, syntax_exit=result.returncode,
    selected_points=[567,568,569,1527,1528,1529,1634,1638], selected_environment=0,
    rows_per_policy=960, policies=['default','accurate'],
    epsilon_seconds=[1e-3,1e-4,1e-5,1e-6,1e-7,1e-8],
    production_changes=False, shader_changes=False, original_capture_modified=False,
    native_executed=False, qualification=False,
    limitation='Controlled constitutive point instrument only. Actual rounded input/output words retained. Repeats measure nondeterminism only. Independent branch, noise-floor and accuracy audit is required; no physical trajectory or assembly qualification.')
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(package=str(out), syntax_exit=result.returncode, replacement_sha256=sha(target))))
raise SystemExit(result.returncode)
