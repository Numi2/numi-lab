#!/usr/bin/env python3
"""Frozen-pose native v15 hand/rod equality check; not a coupled-world test."""
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

fixture = Path(__file__).resolve().parent
base = (fixture / 'articulated-kkt-input.bin').read_bytes()
base_ref = json.loads((fixture / 'articulated-kkt-oracle.json').read_text())
pose_ref = json.loads((fixture / 'pose-kkt-oracle.json').read_text())
assert hashlib.sha256(base).hexdigest() == base_ref['input_sha256']
for elbow, wrist in [('0.6', '0.525'), ('1.2', '1.05'), ('1.8', '1.575')]:
    hand = json.loads(subprocess.run([sys.argv[2], elbow, wrist], capture_output=True, text=True, check=True).stdout)
    assert hand['nv'] == 46
    assert max(abs(x-y) for x,y in zip(hand['point'],pose_ref[elbow]['point'])) < 1e-7
    packed = bytearray()
    for case in range(4):
        values = list(struct.unpack_from('<5624f', base, case * 22496))
        for i in range(46):
            for j in range(46):
                values[(27+i)*73+27+j] = hand['M'][i*46+j] / (0.000125**2)
        for axis in range(3):
            for j in range(46):
                values[5402+axis*73+27+j] = -hand['J'][axis*46+j]
        packed.extend(struct.pack('<5624f', *values))
    with tempfile.TemporaryDirectory(prefix='crow-v15-pose-kkt-') as folder:
        source = Path(folder) / 'input.bin'
        source.write_bytes(packed)
        result = json.loads(subprocess.run([sys.argv[1], str(source)], capture_output=True, text=True, check=True).stdout)
    assert len(result['cases']) == 4
    mapped = {}
    for case, (out, ref) in enumerate(zip(result['cases'], base_ref['cases'])):
        z = struct.unpack_from('<5624f', packed, case * 22496)
        A, b, J, target = z[:5329], z[5329:5402], z[5402:5621], z[5621:5624]
        x, impulse = out['velocity'], out['impulse']
        assert out['path'] == 3 and len(x) == 73 and len(impulse) == 3 and all(map(math.isfinite,x+impulse))
        residual = max(abs(math.fsum(A[i*73+j]*x[j] for j in range(73))-b[i]-math.fsum(J[a*73+i]*impulse[a] for a in range(3))) for i in range(73))
        equality = max(abs(math.fsum(J[a*73+j]*x[j] for j in range(73))-target[a]) for a in range(3))
        difference = max(abs(a-b) for a,b in zip(x,pose_ref[elbow]['coordinates'][case]))
        assert residual < .01 and equality < 1e-9 and difference < 1e-6, (elbow,ref['name'],residual,equality,difference)
        mapped[ref['name']] = [x[j] for j in ref['inverse_permutation']] + x[27:]
        print(elbow,wrist,ref['name'],'force',residual,'equality',equality,'reference',difference)
    for name in ('rest','incoming'):
        assert max(abs(a-b) for a,b in zip(mapped[name+'_forward'],mapped[name+'_reverse'])) < 1e-6
print('passed frozen-pose v15 bilateral checks; not a live world or flight test')
