from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'snapshots/source-right-grip-telemetry-157'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overlay-spec', required=True)
    parser.add_argument('--attempt', choices=('001', '002', '003', '004'), default='004')
    args = parser.parse_args()
    spec_path = Path(args.overlay_spec).resolve()
    specs = json.loads(spec_path.read_text())
    out = B / ('analysis/source167-combined-integration-' + args.attempt)
    target = B / ('snapshots/source-diagnostic-and-resume-167' +
                  {'001': '', '002': '-clean', '003': '-verified', '004': '-final'}[args.attempt])
    assert not target.exists() and not out.exists()
    original = json.loads((S / 'frozen-source.json').read_text())
    support = json.loads((S / 'configure-support-source.json').read_text())
    assert sha(S / 'frozen-source.json') == '2976540a244ba9af686e8033783a3d18da1ac48b030a949b2109fe81108f7cc0'
    files = dict(original['source_sha256'])
    for name, expected in support['files'].items():
        assert name not in files
        files[name] = expected
    for name, expected in files.items():
        assert sha(S / name) == expected, name
    for spec in specs:
        package = Path(spec['package'])
        assert sha(package / spec['manifest']) == spec['manifest_sha256']
        assert sha(package / spec['patch']) == spec['patch_sha256']
        paths = set()
        for line in (package / spec['patch']).read_text().splitlines():
            if line.startswith(('--- ', '+++ ')):
                raw = line[4:].split('\t', 1)[0]
                if raw == '/dev/null':
                    continue
                assert raw.startswith(('a/', 'b/')), raw
                relative = Path(raw[2:])
                assert not relative.is_absolute() and '..' not in relative.parts
                paths.add(str(relative))
        assert paths == set(spec['paths']), (spec['package'], sorted(paths))
    out.mkdir()
    target.mkdir()
    rows = []
    try:
        for name, expected in files.items():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(S / name, destination)
            assert sha(destination) == expected
        for index, spec in enumerate(specs):
            patch = Path(spec['package']) / spec['patch']
            command = ['git', 'apply', str(patch)]
            result = subprocess.run(command, cwd=target, capture_output=True, text=True)
            (out / f'patch-{index:02d}.stdout.log').write_text(result.stdout)
            (out / f'patch-{index:02d}.stderr.log').write_text(result.stderr)
            rows.append(dict(spec=spec, command=command, returncode=result.returncode))
            assert result.returncode == 0, (patch, result.stdout, result.stderr)
        builder = target / 'matter/tools/cardboard_source_bound_build.py'
        module_spec = importlib.util.spec_from_file_location('source167_builder', builder)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        selected = module.source_bindings(target)
        all_files = {str(p.relative_to(target)): sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
        expected_files = set(files) | {path for spec in specs for path in spec['paths']}
        assert set(all_files) == expected_files, dict(
            extra=sorted(set(all_files) - expected_files),
            missing=sorted(expected_files - set(all_files)))
        configure = {name: expected for name, expected in all_files.items() if name not in selected}
        write(target / 'configure-support-source.json', dict(schema=support['schema'], files=configure,
              scope='Exact source157 configure support plus explicitly patched test files. No build products or evidence copied.'))
        write(target / 'frozen-source.json', dict(schema=original['schema'],
              parent_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip(),
              base_snapshot=str(S),
              scope='Integrated default-off diagnostic capture, environment binding, accepted coupon checkpoint, constitutive export and FP64 replay overlays. Native parity remains untested.',
              source_sha256=selected))
        write(out / 'report.json', dict(schema='numi.cardboard.combined-source-integration.v1',
              created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              base=str(S), target=str(target), overlay_spec_sha256=sha(spec_path), patches=rows,
              base_files_verified=len(files), selected_source_files=len(selected),
              configure_support_files=len(configure),
              frozen_source_sha256=sha(target / 'frozen-source.json'),
              configure_support_sha256=sha(target / 'configure-support-source.json'),
              changed_or_added={name: expected for name, expected in all_files.items() if files.get(name) != expected},
              removed=sorted(set(files) - set(all_files)),
              native_execution_performed=False, physics_qualification=False))
    except Exception as error:
        write(out / 'failure.json', dict(error=repr(error), patches=rows))
        raise
    shutil.copy2(spec_path, out / 'overlay-spec.json')
    shutil.copy2(Path(__file__), out / 'prepare.py')
    print(out / 'report.json')

if __name__ == '__main__':
    main()
