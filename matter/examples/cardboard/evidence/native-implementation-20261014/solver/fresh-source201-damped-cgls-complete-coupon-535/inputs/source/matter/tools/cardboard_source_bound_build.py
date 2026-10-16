#!/usr/bin/env python3
"""Build in a fresh directory and bind the executed build to immutable inputs.

This complements the native run receipt: retaining a source tree beside an old
executable alone does not establish that the executable came from that tree.
No native executable or GPU work is launched here.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def source_bindings(root):
    files = []
    for directory in ('matter/include', 'matter/src', 'include', 'src',
                      'numi', 'cmake', 'schemas',
                      'matter/examples/cardboard/source-data'):
        files.extend(p for p in (root / directory).rglob('*') if p.is_file())
    for directory in ('matter/tools', 'matter/materials'):
        files.extend(p for p in (root / directory).iterdir() if p.is_file())
    files.extend(p for p in (root / 'CMakeLists.txt', root / 'matter/CMakeLists.txt',
                            root / 'matter/examples/cardboard/cardboard_shipping_300x200_protocol.json')
                 if p.is_file())
    return {str(p.relative_to(root)): digest(p) for p in sorted(set(files))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-snapshot', required=True)
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--cmake-source-relative', default='matter')
    parser.add_argument('--target', action='append', required=True)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--cmake-define', action='append', default=[])
    args = parser.parse_args()
    root = Path(args.source_snapshot).resolve()
    build = Path(args.build_dir).resolve()
    source = (root / args.cmake_source_relative).resolve()
    if not source.is_relative_to(root) or not (source / 'CMakeLists.txt').is_file():
        parser.error('CMake source must be a directory within the snapshot')
    if build.exists():
        parser.error('build directory must be new; existing objects cannot prove this build')
    if args.jobs < 1 or args.jobs > 32:
        parser.error('jobs must be between 1 and 32')
    before = source_bindings(root)
    build.mkdir(parents=True)
    record = {
        'schema': 'numi.cardboard.source-build.v1',
        'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_snapshot': str(root), 'cmake_source': str(source),
        'build_dir': str(build), 'targets': args.target,
        'source_sha256': before, 'platform': platform.platform(),
        'commands': [], 'success': False,
        'limitation': 'Compiler system headers and OS/driver are not archived.'}
    commands = [
        ['cmake', '-S', str(source), '-B', str(build),
         '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
         '-DPython3_EXECUTABLE=' + sys.executable] +
        ['-D' + value for value in args.cmake_define],
        ['cmake', '--build', str(build), '--target'] + args.target +
        ['-j', str(args.jobs)]]
    code = 1
    try:
        for index, command in enumerate(commands):
            record['commands'].append(command)
            with (build / f'provenance-command-{index}.log').open('wb') as output:
                process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            if process.returncode:
                code = process.returncode
                raise RuntimeError(f'build command {index} failed ({code})')
        after = source_bindings(root)
        record['changed_sources'] = sorted(k for k in set(before) | set(after)
                                           if before.get(k) != after.get(k))
        if record['changed_sources']:
            raise RuntimeError('source changed during build')
        binaries = []
        for target in args.target:
            candidates = [p for p in (build / target, build / 'matter' / target)
                          if p.is_file()]
            if len(candidates) != 1:
                raise RuntimeError(f'exact executable for target {target} is not identified')
            binaries.extend(candidates)
        for directory in (build, build / 'lib', build / 'matter', build / 'matter/lib'):
            binaries.extend(directory.glob('*.dylib'))
        binaries.extend((build / 'shaders').glob('*.metallib'))
        binaries.extend((build / 'matter/shaders').glob('*.metallib'))
        record['artifacts_sha256'] = {
            str(p.resolve()): digest(p) for p in sorted(set(binaries))}
        record['build_metadata_sha256'] = {
            p.name: digest(p) for p in
            (build / 'CMakeCache.txt', build / 'compile_commands.json') if p.is_file()}
        record['success'] = True
        code = 0
    except Exception as error:
        record['error'] = str(error)
    finally:
        record['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (build / 'source-build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'success': record['success'], 'exit_code': code,
                      'provenance': str(build / 'source-build.json')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
