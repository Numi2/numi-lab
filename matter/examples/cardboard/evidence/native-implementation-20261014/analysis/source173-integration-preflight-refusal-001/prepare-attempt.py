from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
B = Path('/Users/home/cardboard-implementation-20261014')
S = B / 'snapshots/source-warm-cgls-refinement-169'
T = B / 'snapshots/source-checkpoint-warm-breakdown-173'
O = B / 'analysis/source173-checkpoint-warm-integration-001'
COMPONENTS = [
    ('shipping-checkpoint-physical-arena-overlay-171-001', 'manifest.json', 'overlay.patch',
     '304273606d04d2367cc3d495f9c4565cbc2e60b6cc68ad5f75c2ef09bf5f9eaa',
     '7145aea2965145284dc65d76771bdc096b78f0484854c768d24f87a51f01759b'),
    ('base-cgls-warm-finite-breakdown-overlay-172-001', 'source-overlay.json', 'source-172.patch',
     'e21c032769e4118242c82352fa0307e2f8cc97c8998b7de1a429469293e388b1',
     'e11089cc392d746044cb3a72a24109f820583fc857db43c564e7b1894d970a48'),
]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not T.exists() and not O.exists()
    frozen = json.loads((S / 'frozen-source.json').read_text())
    support = json.loads((S / 'configure-support-source.json').read_text())
    assert sha(S / 'frozen-source.json') == 'f5054091c27a7e1add41630aae1b45ee6ecc3e6d4112ee85c931cc4c70f44ea4'
    components = []
    candidate_files = {}
    for directory, manifest_name, patch_name, manifest_sha, patch_sha in COMPONENTS:
        component = B / 'analysis' / directory
        manifest = component / manifest_name
        assert sha(manifest) == manifest_sha and sha(component / patch_name) == patch_sha
        names = sorted(str(p.relative_to(component / 'candidate'))
                       for p in (component / 'candidate').rglob('*') if p.is_file())
        # Both overlays are narrow and must not overlap.
        expected = {
            'shipping-checkpoint-physical-arena-overlay-171-001': {
                'matter/tools/cardboard_shipping_snapshot.hpp',
                'matter/tools/cardboard_shipping_snapshot_check.cpp'},
            'base-cgls-warm-finite-breakdown-overlay-172-001': {
                'matter/src/runtime.mm', 'matter/src/metal/base_cgls_runtime.metalinc',
                'matter/src/metal/base_cgls.metalinc',
                'matter/tools/cardboard_base_cgls_warm_start_check.mm'},
        }[directory]
        assert set(names) == expected
        for name in names:
            assert name not in candidate_files
            candidate_files[name] = component / 'candidate' / name
        components.append({'path': str(component), 'patch_name': patch_name,
                           'manifest_sha256': sha(manifest),
                           'patch_sha256': sha(component / patch_name),
                           'candidate_sha256': {name: sha(component / 'candidate' / name) for name in names}})
    files = dict(frozen['source_sha256'])
    files.update(support['files'])
    for name, expected in files.items():
        assert sha(S / name) == expected, name
    T.mkdir()
    O.mkdir()
    try:
        for name in files:
            destination = T / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(S / name, destination)
        for component in components:
            patch = Path(component['path']) / component['patch_name']
            command = ['git', 'apply', str(patch)]
            result = subprocess.run(command, cwd=T, capture_output=True)
            stem = Path(component['path']).name
            (O / (stem + '.stdout.log')).write_bytes(result.stdout)
            (O / (stem + '.stderr.log')).write_bytes(result.stderr)
            assert result.returncode == 0, result.stderr.decode()
        for name, candidate in candidate_files.items():
            assert sha(T / name) == sha(candidate), name
        spec = importlib.util.spec_from_file_location('builder173', T / 'matter/tools/cardboard_source_bound_build.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        selected = builder.source_bindings(T)
        assert len(selected) == 804
        changes = {name: digest for name, digest in selected.items()
                   if frozen['source_sha256'].get(name) != digest}
        assert set(changes) == set(candidate_files)
        unchanged_support = {name: sha(T / name) for name in files if name not in selected}
        scope = 'Exact physical-arena checkpoint sizes and default-off warm CGLS finite-breakdown decline; no native trajectory or physical qualification.'
        (T / 'frozen-source.json').write_text(json.dumps(dict(frozen, base_snapshot=str(S),
            source_sha256=selected, scope=scope), indent=2) + '\n')
        (T / 'configure-support-source.json').write_text(json.dumps(dict(support,
            files=unchanged_support), indent=2) + '\n')
        report = {'schema': 'numi.cardboard.source173.integrated-source.v1',
                  'source_snapshot': str(T), 'base_snapshot': str(S),
                  'base_frozen_sha256': sha(S / 'frozen-source.json'),
                  'base_files_verified': len(files), 'selected_source_count': len(selected),
                  'frozen_source_sha256': sha(T / 'frozen-source.json'),
                  'components': components, 'changed_or_added': changes,
                  'gpu_execution': False, 'qualification': scope}
        (O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    except Exception as error:
        (O / 'failure.json').write_text(json.dumps({'error': repr(error)}, indent=2) + '\n')
        raise
    shutil.copyfile(__file__, O / 'prepare.py')
    (O / 'manifest.json').write_text(json.dumps({'schema': 'numi.cardboard.source173.integration-manifest.v1',
        'files_sha256': {str(p.relative_to(O)): sha(p) for p in sorted(O.rglob('*')) if p.is_file()}}, indent=2) + '\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
