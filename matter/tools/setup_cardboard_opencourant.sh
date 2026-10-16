#!/bin/bash
# Run on the designated macmini. This installs no paid software or global Python packages.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
OC_TASK_ROOT="${1:?Usage: setup_cardboard_opencourant.sh ABSOLUTE_TASK_ROOT}"
case "$OC_TASK_ROOT" in /*) ;; *) echo 'Task root must be absolute' >&2; exit 2;; esac
mkdir -p "$OC_TASK_ROOT/setup"
if ! command -v limactl >/dev/null; then
    HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 brew install lima
fi
python3 - "$OC_TASK_ROOT" <<'PY'
import hashlib,json,pathlib,stat,sys,urllib.request,zipfile
root=pathlib.Path(sys.argv[1]); setup=root/'setup'
release='latest-20261011'; digest='ded9814e2460767353d591c89695ba1f76251291041943ab0ad73070a6c865f7'
package=setup/'OpenCourant_linuxa64.zip'
url=f'https://github.com/OpenCourant/OpenCourant/releases/download/{release}/{package.name}'
if not package.exists(): urllib.request.urlretrieve(url,package)
if hashlib.sha256(package.read_bytes()).hexdigest()!=digest: raise SystemExit('Package digest mismatch')
runtime=root/'runtime'/'OpenCourant'
if not runtime.exists():
    with zipfile.ZipFile(package) as z:
        for info in z.infolist():
            dest=(root/'runtime'/info.filename).resolve()
            if not dest.is_relative_to((root/'runtime').resolve()): raise SystemExit('Escaping archive member')
        z.extractall(root/'runtime')
for p in (runtime/'exec').iterdir():
    if p.is_file(): p.chmod(p.stat().st_mode | stat.S_IXUSR)
files={str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in sorted(runtime.rglob('*')) if p.is_file()}
# Bind extracted bytes to the authenticated archive, not merely to whatever
# happens to be on disk when setup is rerun.
with zipfile.ZipFile(package) as z:
    expected={str(pathlib.PurePosixPath(i.filename).relative_to('OpenCourant')):
              hashlib.sha256(z.read(i)).hexdigest()
              for i in z.infolist() if not i.is_dir()}
if files != expected: raise SystemExit('Extracted runtime differs from pinned archive')
(setup/'runtime-identity.json').write_text(json.dumps({'release':release,
    'source_commit':'cb760c206e346eff12161a2268b134ef9d48a4ef',
    'package_sha256':digest,'download_url':url,'files':files},indent=2)+'\n')
# JSON is valid YAML. Pin the same guest image used by the retained admission.
(setup/'lima-pinned.yaml').write_text(json.dumps({'images':[{
    'location':'https://cloud-images.ubuntu.com/releases/noble/release-20260926/ubuntu-24.04-server-cloudimg-arm64.img',
    'arch':'aarch64','digest':'sha256:1d6bffe64b848468ac97f821d369a4846d983de1800ccf6b5ec8853e85cefc55'}]},indent=2)+'\n')
PY
if limactl list --json | python3 -c 'import json,sys; rows=[json.loads(s) for s in sys.stdin if s.strip()]; sys.exit(0 if any(r["name"]=="cardboard-oc" for r in rows) else 1)'; then
    limactl list --json | python3 -c '
import json,sys
r=next(json.loads(s) for s in sys.stdin if json.loads(s)["name"]=="cardboard-oc")
assert r["arch"]=="aarch64" and r["memory"]<=12*1024**3
assert any(m["location"]==sys.argv[1] and m["writable"] for m in r["config"]["mounts"])
' "$OC_TASK_ROOT"
    limactl start --timeout=8m --tty=false cardboard-oc > "$OC_TASK_ROOT/setup/lima-start-replay.log" 2>&1
else
    limactl start --name=cardboard-oc --vm-type=vz --arch=aarch64 --cpus=8 \
        --memory=12 --disk=16 --containerd=none --mount-only="$OC_TASK_ROOT:w" \
        --timeout=8m --tty=false "$OC_TASK_ROOT/setup/lima-pinned.yaml" > "$OC_TASK_ROOT/setup/lima-start-replay.log" 2>&1
fi
limactl list --json > "$OC_TASK_ROOT/setup/vm.json"
limactl shell cardboard-oc -- bash -c '
set -eu
root="$1/runtime/OpenCourant"
export LD_LIBRARY_PATH="$root/extlib/hm_reader/linuxa64:$root/extlib/h3d/lib/linuxa64"
export RAD_CFG_PATH="$root/hm_cfg_files"
uname -a
ldd "$root/exec/starter_linuxa64_gf"
ldd "$root/exec/engine_linuxa64_gf"
"$root/exec/starter_linuxa64_gf" -version
"$root/exec/engine_linuxa64_gf" -version
' -- "$OC_TASK_ROOT" > "$OC_TASK_ROOT/setup/runtime-preflight.txt" 2>&1
echo "Runtime prepared. This is installation evidence, not backend admission."
