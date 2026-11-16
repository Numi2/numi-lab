import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

fixture = Path(__file__).resolve().parent
blob = (fixture / 'articulated-kkt-input.bin').read_bytes()
oracle = json.loads((fixture / 'articulated-kkt-oracle.json').read_text())
assert hashlib.sha256(blob).hexdigest() == oracle['input_sha256']
live_run = subprocess.run([sys.argv[2]], capture_output=True, text=True, check=True)
live = json.loads(live_run.stdout)
assert live['nv'] == 46
run = subprocess.run([sys.argv[1], str(fixture / 'articulated-kkt-input.bin')],
                     capture_output=True, text=True, check=True)
cases = json.loads(run.stdout)['cases']
assert len(cases) == len(oracle['cases']) == 4
mapped = {}
for index, (out, ref) in enumerate(zip(cases, oracle['cases'])):
    raw = struct.unpack_from('<5624f', blob, index * 22496)
    A, b, J, target = raw[:5329], raw[5329:5402], raw[5402:5621], raw[5621:5624]
    # Ensure the retained matrix/Jacobian is still the actual v15 model.
    for i in range(46):
        for j in range(46):
            expected = live['M'][i*46+j] / (0.000125**2)
            assert abs(A[(27+i)*73+27+j]-expected) <= 1e-6*max(1.0,abs(expected))
    for axis in range(3):
        for i in range(46):
            expected = -live['J'][axis*46+i]
            assert abs(J[axis*73+27+i]-expected) <= 1e-6*max(1.0,abs(expected))
    x, impulse = out['velocity'], out['impulse']
    assert out['path'] == 3 and len(x) == 73 and len(impulse) == 3
    assert all(math.isfinite(v) for v in x + impulse)
    residual = max(abs(math.fsum(A[i*73+j]*x[j] for j in range(73))-b[i]-
                       math.fsum(J[a*73+i]*impulse[a] for a in range(3))) for i in range(73))
    equality = max(abs(math.fsum(J[a*73+j]*x[j] for j in range(73))-target[a]) for a in range(3))
    difference = max(abs(a-b) for a,b in zip(x, ref['reference']))
    assert residual < 0.01 and equality < 1e-9 and difference < 1e-6, (ref['name'], residual, equality, difference)
    mapped[ref['name']] = [x[j] for j in ref['inverse_permutation']] + list(x[27:])
    print(ref['name'], 'force_residual', residual, 'attachment_error_m', equality, 'difference', difference)
for name in ('rest', 'incoming'):
    assert max(abs(a-b) for a,b in zip(mapped[name+'_forward'], mapped[name+'_reverse'])) < 1e-6
print('native quality exact bilateral reference passed; not mixed-contact or MetalWorld qualification')
