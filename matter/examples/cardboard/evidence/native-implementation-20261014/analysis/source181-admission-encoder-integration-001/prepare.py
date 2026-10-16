from pathlib import Path
import hashlib, importlib.util, json, shutil, sys
sys.dont_write_bytecode = True
B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'snapshots/source-checkpoint-warm-breakdown-173'
T = B / 'snapshots/source-admission-encoder-181'
O = B / 'analysis/source181-admission-encoder-integration-001'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not T.exists() and not O.exists()
f = json.loads((S / 'frozen-source.json').read_text())
c = json.loads((S / 'configure-support-source.json').read_text())
assert sha(S / 'frozen-source.json') == '6d7e76fdb07702b0b0727fc6610ae979c4e0e161ad92f6888036b1dfef3b53b3'
updates = ['matter/src/runtime.mm', 'matter/CMakeLists.txt', 'matter/tools/cardboard_probe.mm',
    'matter/tools/cardboard_solver_environment.py', 'matter/tools/test_cardboard_solver_environment.py',
    'matter/tools/test_cardboard_probe_argument_admission.py']
T.mkdir(); O.mkdir()
for name, digest in dict(c['files'], **f['source_sha256']).items():
    assert sha(S / name) == digest, name
    p = T / name
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(S / name, p)
for name in updates:
    p = T / name
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(R / name, p)
base = (S / 'matter/src/runtime.mm').read_text()
expected = base.replace('ownership->preDynamicsOpen = false;\n                return diagnostics;\n            }\n            if (femBaseCGLSPrimaryPolicyRequested',
    'ownership->preDynamicsOpen = false;\n                [encoder endEncoding];\n                encoder = nil;\n                return diagnostics;\n            }\n            if (femBaseCGLSPrimaryPolicyRequested', 1)
assert expected != base and (T / 'matter/src/runtime.mm').read_text() == expected
spec = importlib.util.spec_from_file_location('builder181', T / 'matter/tools/cardboard_source_bound_build.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
selected = m.source_bindings(T)
changed = {name: digest for name, digest in selected.items() if f['source_sha256'].get(name) != digest}
assert set(changed) == set(updates), changed.keys()
scope = 'Contact environment binding, CPU-only native argument admission, and balanced encoder cleanup on the already-rejected Cauchy configuration. No physical solver, tolerance or valid candidate update change; native validation pending.'
(T / 'frozen-source.json').write_text(json.dumps(dict(f, base_snapshot=str(S), source_sha256=selected, scope=scope), indent=2) + '\n')
(T / 'configure-support-source.json').write_text(json.dumps(dict(c, files={name:sha(T / name) for name in c['files'] if name not in selected}), indent=2) + '\n')
(O / 'report.json').write_text(json.dumps({'schema':'numi.cardboard.source181.integration.v1',
    'parent_frozen_sha256':sha(S / 'frozen-source.json'), 'frozen_sha256':sha(T / 'frozen-source.json'),
    'selected_files':len(selected), 'changed':changed, 'scope':scope, 'gpu_execution':False}, indent=2) + '\n')
shutil.copy2(__file__, O / 'prepare.py')
(O / 'manifest.json').write_text(json.dumps({'files_sha256':{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file()}}, indent=2) + '\n')
print(json.dumps({'frozen_sha256':sha(T / 'frozen-source.json'), 'manifest_sha256':sha(O / 'manifest.json'), 'selected_files':len(selected)}))
