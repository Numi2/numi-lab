#!/usr/bin/env python3
from pathlib import Path
import subprocess, tempfile, shutil
pkg=Path(__file__).resolve().parent
base=Path('/Users/home/cardboard-implementation-20261014/analysis/armijo-all-trial-summary-overlay-001/overlay/matter/src/runtime.mm')
with tempfile.TemporaryDirectory(prefix='armijo-summary-002-') as td:
    root=Path(td)
    dest=root/'matter/src/runtime.mm'
    dest.parent.mkdir(parents=True)
    shutil.copyfile(base,dest)
    p=subprocess.run(['patch','--dry-run','-p1','-d',str(root),'-i',str(pkg/'overlay.patch')],capture_output=True,text=True)
    if p.returncode:
        raise SystemExit(p.stdout+p.stderr)
    print(p.stdout.strip() or 'checking patch...')
