#!/bin/bash
# Run inside the bounded cardboard-mumps ARM64 Lima guest.
# Installation only: mechanics source and algorithm remain upstream bytes.
set -euo pipefail
task_root="${1:?absolute task directory required}"
case "$task_root" in /*) ;; *) exit 2;; esac
test "$(uname -m)" = aarch64
source_root="$task_root/upstream"
test "$(git -C "$source_root" rev-parse HEAD)" = cb760c206e346eff12161a2268b134ef9d48a4ef
test -z "$(git -C "$source_root" diff HEAD --)"
test ! -e "$task_root/setup/build-receipt.json"
python3 - "$task_root" <<'PY'
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); prefix=root/'dependency-prefix'
libs=pathlib.Path('/usr/lib/aarch64-linux-gnu')
(prefix/'lib').mkdir(parents=True,exist_ok=False)
(prefix/'include').symlink_to('/usr/include')
# Upstream accepts absolute archive filenames. GNU ld also accepts shared
# objects at those paths. The aliases below are declared and hashed; there
# is no static-link claim. This uses the distribution's matching headers
# and shared libraries without editing upstream build or mechanics files.
aliases={'libdmumps.a':'libdmumps.so','libmumps_common.a':'libmumps_common.so',
         'libpord.a':'libpord.so','libscalapack.a':'libscalapack-openmpi.so',
         'liblapack.a':'liblapack.so','libtmglib.a':'libtmglib.so','librefblas.a':'libblas.so'}
items={}
for alias, name in aliases.items():
    target=(libs/name).resolve(strict=True)
    (prefix/'lib'/alias).symlink_to(target)
    items[alias]={'target':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
packages=subprocess.check_output(['dpkg-query','-W'],text=True)
(root/'setup/dpkg-packages.txt').write_text(packages)
(root/'setup/dependency-aliases.json').write_text(json.dumps(items,indent=2)+'\n')
PY
python3 "$source_root/Compiling_tools/script/load_extlib.py"
cd "$source_root/engine"
/usr/bin/time -v timeout --signal=TERM --kill-after=10 3600 \
    ./build_script.sh -arch=linuxa64_gf -release -prec=dp -nt=8 -mpi=ompi \
    -mpi-include=/usr/lib/aarch64-linux-gnu/openmpi/include \
    -mpi-libdir=/usr/lib/aarch64-linux-gnu \
    -mumps_root="$task_root/dependency-prefix" \
    -scalapack_root="$task_root/dependency-prefix/lib" \
    -lapack_root="$task_root/dependency-prefix/lib"
test -z "$(git -C "$source_root" diff HEAD --)"
test -x "$source_root/exec/engine_linuxa64_gf_ompi"
cp "$source_root/exec/engine_linuxa64_gf_ompi" "$task_root/runtime/OpenCourant/exec/engine_linuxa64_gf_ompi"
python3 - "$task_root" <<'PY'
import datetime,hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); setup=root/'setup'; source=root/'upstream'; runtime=root/'runtime/OpenCourant'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
receipt={'schema':'cardboard.opencourant.build.v1','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_commit':git('rev-parse','HEAD'),'tracked_source_diff':git('diff','HEAD','--'),
         'engine_sha256':sha(runtime/'exec/engine_linuxa64_gf_ompi'),
         'build_script_sha256':sha(root/'tools/build_cardboard_opencourant_mumps.sh'),
         'dependency_aliases_sha256':sha(setup/'dependency-aliases.json'),
         'dpkg_packages_sha256':sha(setup/'dpkg-packages.txt'),
         'compiler':subprocess.check_output(['gfortran','--version'],text=True),
         'extlib_manifest':json.loads((source/'EXTLIB_VERSION.json').read_text()),
         'limitations':'Distribution MUMPS/LAPACK/ScaLAPACK versions differ from upstream example recipe. Build completion is not backend admission.'}
(setup/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
identity=json.loads((setup/'packaged-runtime-identity.json').read_text())
identity.update(profile='arm64-mumps-build',build={'source_commit':receipt['source_commit'],
    'tracked_source_diff':receipt['tracked_source_diff'],'receipt_sha256':sha(setup/'build-receipt.json')})
identity['files']={str(p.relative_to(runtime)):sha(p) for p in sorted(runtime.rglob('*')) if p.is_file()}
(setup/'runtime-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
PY
