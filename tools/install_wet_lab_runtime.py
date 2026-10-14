#!/usr/bin/env python3
"""Install an immutable, verified local presentation runtime; retain scientific owners."""
import argparse,hashlib,json,os,shutil,sys,uuid
from pathlib import Path

NAMES=('virtual_wet_lab.py','wet_lab_shared.py','wet_lab_snapshot.py','wet_lab_cli.py','wet_lab_entry.py','biological_viewers.py','wet_lab_viewer_export.py')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--python',default=sys.executable);p.add_argument('--home',type=Path,default=Path.home());a=p.parse_args()
 source=Path(__file__).resolve().parents[1]/'python/metalrobo';base=a.home/'.numi/runtime/wet-lab';base.mkdir(parents=True,exist_ok=True)
 import subprocess
 subprocess.run([a.python,'-c','import numpy,safetensors,anndata,h5py,scipy,sklearn'],check=True)
 files={name:source/name for name in NAMES}
 for path in (source/'wet_lab').rglob('*'):
  if path.is_file() and not any(x in path.parts for x in ('node_modules','__pycache__')):files[str(path.relative_to(source))]=path
 hashes={name:digest(path) for name,path in files.items()};identifier=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16];dest=base/identifier
 if not dest.exists():
  staging=base/('.staging-'+uuid.uuid4().hex);staging.mkdir()
  for name,path in files.items():target=staging/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
  if any(digest(staging/name)!=sha for name,sha in hashes.items()):raise ValueError('Installed runtime differs from source')
  os.replace(staging,dest)
 manifest={'format':'numilab-installed-workbench/v1','runtime':str(dest),'python':str(Path(a.python).absolute()),'files':hashes,'source':str(source),'id':identifier}
 temp=base/'current.tmp';temp.write_text(json.dumps(manifest,indent=2));os.chmod(temp,0o600);os.replace(temp,base/'current.json')
 wrapper='''#!/bin/sh
set -eu
if [ "${1:-}" = "--numi-describe" ]; then
 printf 'Open and operate the retained Codex laboratory with shared edits and explicit observation reveal.\\n'
 exit 0
fi
exec python3 -c 'import json,os,sys;from pathlib import Path;c=json.loads((Path.home()/".numi/runtime/wet-lab/current.json").read_text());os.execv(c["python"],[c["python"],str(Path(c["runtime"])/("virtual_wet_lab.py" if sys.argv[1:] and sys.argv[1] in ("--vivo-root","--assay","--binary","--workspace","--catalog") else "wet_lab_cli.py")),*sys.argv[1:]])' "$@"
'''
 for directory in (a.home/'.numi/commands',a.home/'.config/numi/commands'):
  directory.mkdir(parents=True,exist_ok=True);target=directory/'wet-lab'
  if target.exists() and target.read_text()!=wrapper:
   backup=base/'launcher-backups';backup.mkdir(exist_ok=True);shutil.copy2(target,backup/(directory.parent.name+'-'+uuid.uuid4().hex+'.sh'))
  target.write_text(wrapper);os.chmod(target,0o755)
 print(json.dumps({'status':'installed-and-verified','runtimeID':identifier,'runtime':str(dest),'python':manifest['python'],'checkoutRequired':False,'nodeRequired':False},indent=2))
if __name__=='__main__':main()
