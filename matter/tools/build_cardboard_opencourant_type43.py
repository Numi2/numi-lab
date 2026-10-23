#!/usr/bin/env python3
"""Build the single pinned TYPE43 repair on the existing ARM64 MUMPS setup.

No dependency installation, solver tuning, or runtime overwrite. Reuses the
completed upstream build and creates a separately identified candidate runtime.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import time

from cardboard_opencourant import SOURCE, TYPE43_PATCH_SHA256, sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task-root', type=pathlib.Path, required=True)
    p.add_argument('--patch-directory', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    args = p.parse_args()
    task = args.task_root.resolve()
    source = task / 'upstream'
    output = args.output.resolve()
    setup = output / 'setup'
    setup.mkdir(parents=True, exist_ok=True)
    receipt_path = setup / 'build-receipt.json'
    if receipt_path.exists():
        raise ValueError('candidate receipt already exists; preserve it')
    patch = args.patch_directory / 'type43-implicit.patch'
    spec = json.loads((args.patch_directory / 'patch-identity.json').read_text())
    if sha(patch) != TYPE43_PATCH_SHA256 or spec['patch_sha256'] != TYPE43_PATCH_SHA256:
        raise ValueError('unrecognized source patch')
    git = lambda *a: subprocess.check_output(['git', '-C', str(source), *a], text=True).strip()
    if git('rev-parse', 'HEAD') != SOURCE:
        raise ValueError('upstream revision changed')
    changed = git('diff', '--name-only').splitlines()
    if not changed:
        for rel, hashes in spec['files'].items():
            if sha(source / rel) != hashes['before_sha256']:
                raise ValueError(f'upstream file changed: {rel}')
        subprocess.run(['git', '-C', str(source), 'apply', str(patch.resolve())], check=True)
    if set(git('diff', '--name-only').splitlines()) != set(spec['files']):
        raise ValueError('source contains changes beyond the pinned patch')
    if git('ls-files', '--others', '--exclude-standard'):
        raise ValueError('untracked source could enter the build')
    for rel, hashes in spec['files'].items():
        if sha(source / rel) != hashes['after_sha256']:
            raise ValueError(f'patched file changed: {rel}')
    build = source / 'engine/cbuild_engine_linuxa64_gf_ompi'
    command = ['/usr/bin/time', '-v', 'timeout', '--signal=TERM', '--kill-after=10',
               '300', 'cmake', '--build', str(build), '-j', '2']
    stamp = time.time()
    with (setup / 'candidate-build.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'build failed: {result.returncode}; log retained')
    runtime = output / 'runtime/OpenCourant'
    runtime.parent.mkdir(exist_ok=True)
    # Read-only hard links save space. The candidate Engine gets a new name;
    # no original library, config or executable is edited through these links.
    subprocess.run(['cp', '-al', str(task / 'runtime/OpenCourant'), str(runtime)], check=True)
    engine = runtime / 'exec/engine_type43'
    shutil.copy2(build / 'engine_linuxa64_gf_ompi', engine)
    for f in ('type43-implicit.patch', 'patch-identity.json'):
        shutil.copyfile(args.patch_directory / f, setup / f)
    (setup / 'source.diff').write_text(git('diff', '--binary') + '\n')
    flags = build / 'CMakeFiles/engine_linuxa64_gf_ompi.dir/flags.make'
    shutil.copyfile(flags, setup / 'flags.make')
    receipt = {'schema': 'cardboard.opencourant.type43.build.v1', 'source_commit': SOURCE,
               'patch_sha256': sha(patch), 'changed_sources': spec['files'],
               'engine_sha256': sha(engine), 'parent_build_receipt_sha256': sha(task / 'setup/build-receipt.json'),
               'source_diff_sha256': sha(setup / 'source.diff'), 'flags_sha256': sha(flags),
               'argv': command, 'elapsed_s': time.time() - stamp,
               'purpose': 'candidate TYPE43 initial stiffness, defined LAW117 restart slot and late-birth area/frame initialization; qualification separate'}
    write(receipt_path, receipt)
    identity = json.loads((task / 'setup/runtime-identity.json').read_text())
    identity['profile'] = 'arm64-mumps-type43'
    identity['files']['exec/engine_type43'] = sha(engine)
    identity['build'] = {'source_commit': SOURCE, 'patch_sha256': sha(patch),
                         'receipt_sha256': sha(receipt_path)}
    write(setup / 'runtime-identity.json', identity)
    print(json.dumps({'runtime': str(runtime), 'engine_sha256': sha(engine)}))


if __name__ == '__main__':
    main()
