#!/usr/bin/env python3
"""Compile-only Metal syntax check; never creates a device or dispatches work."""
import hashlib,json,shutil,subprocess,tempfile
from pathlib import Path
D=Path(__file__).resolve().parent
BASE=Path('/Users/home/cardboard-implementation-20261014/snapshots/source-portable-paper-reference-185')
def sha(b): return hashlib.sha256(b).hexdigest()
with tempfile.TemporaryDirectory(prefix='source186-metal-compile-') as td:
    root=Path(td)/'matter/src'
    shutil.copytree(BASE/'matter/src',root)
    shutil.copy2(D/'candidate/matter/src/metal/contact.metalinc',root/'metal/contact.metalinc')
    out=Path(td)/'mollifier.air'
    cmd=['xcrun','-sdk','macosx','metal','-c',str(D/'tests/mollifier_cases.metal'),'-I',str(root),'-I',str(BASE/'matter/include'),'-I',str(BASE/'include'),'-o',str(out)]
    r=subprocess.run(cmd,text=True,capture_output=True)
    air=out.read_bytes() if out.exists() else b''
    receipt={'schema':'numi.cardboard.source186_mollifier_metal_compile.v1','command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'air_byte_count':len(air),'air_sha256':sha(air) if air else None,'scope':'Metal compile only; no metallib link, Metal device creation, or GPU dispatch.'}
    (D/'metal-compile-receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(receipt,sort_keys=True))
    if r.returncode: raise SystemExit(r.returncode)
