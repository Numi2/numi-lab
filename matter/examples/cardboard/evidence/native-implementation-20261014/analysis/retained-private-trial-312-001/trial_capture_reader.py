"""Validate private-trial readback bytes without treating them as a checkpoint."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

CAP = 32 * 1024 * 1024
ARRAY_CAP = 192
BOUNDS = {'pre': 71, 'post': 109, 'both': 136}

def reject_constant(token):
    raise ValueError('nonstandard JSON constant: ' + token)

def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError('invalid nonnegative integer: ' + name)
    return value

class Capture:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if self.path.stat().st_size > 1024 * 1024:
            raise ValueError('manifest exceeds 1 MiB')
        self.manifest = m = json.loads(self.path.read_text(), parse_constant=reject_constant)
        if m.get('schema') != 'numi.matter.coupled_armijo_trial_capture.v1':
            raise ValueError('unsupported capture schema')
        if m.get('byte_order') != 'little_endian':
            raise ValueError('unsupported byte order')
        phase = m.get('selected_phase')
        if phase not in BOUNDS:
            raise ValueError('unsupported selected phase')
        if m.get('max_payload_bytes') != CAP or m.get('max_arrays') != ARRAY_CAP:
            raise ValueError('unexpected production bounds')
        count = integer(m.get('array_count'), 'array_count')
        arrays = m.get('arrays')
        if not isinstance(arrays, list) or len(arrays) != count or not 0 < count <= BOUNDS[phase]:
            raise ValueError('array count or phase bound mismatch')
        p = m['payload']
        size = integer(p.get('byte_count'), 'payload.byte_count')
        if not 0 < size <= CAP:
            raise ValueError('payload exceeds bound or is empty')
        # Production metadata records its original absolute path. Restoration
        # must resolve only the manifest-adjacent basename, never that old path.
        name = Path(p['file']).name
        if not name or name in ('.', '..') or not name.endswith('.coupled.bin'):
            raise ValueError('invalid adjacent payload basename')
        self.payload_path = self.path.parent / name
        if self.payload_path.is_symlink() or self.payload_path.stat().st_size != size:
            raise ValueError('payload extent or symlink mismatch')
        self.payload = self.payload_path.read_bytes()
        if hashlib.sha256(self.payload).hexdigest() != p.get('sha256'):
            raise ValueError('payload hash mismatch')
        self.arrays = {}
        ranges = []
        scalar_bytes = {'float32': 4, 'uint32': 4, 'float32x4': 16,
                        'uint32x4': 16, 'byte': 1}
        for a in arrays:
            key = (a.get('phase'), a.get('name'))
            if not all(isinstance(v, str) and v for v in key) or key in self.arrays:
                raise ValueError('invalid or duplicate phase/name')
            offset = integer(a.get('offset'), 'offset')
            n = integer(a.get('count'), 'count')
            stride = integer(a.get('element_bytes'), 'element_bytes')
            length = integer(a.get('byte_count'), 'byte_count')
            if not n or not stride or n * stride != length or offset + length > size:
                raise ValueError('array extent mismatch')
            dtype = a.get('dtype', '')
            if dtype in scalar_bytes:
                if stride != scalar_bytes[dtype]:
                    raise ValueError('scalar element stride mismatch')
            elif not isinstance(dtype, str) or not dtype.startswith('record:'):
                raise ValueError('unsupported raw type')
            data = self.payload[offset:offset + length]
            if hashlib.sha256(data).hexdigest() != a.get('sha256'):
                raise ValueError('array hash mismatch')
            self.arrays[key] = a
            ranges.append((offset, offset + length))
        ranges.sort()
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError('overlapping array extents')

    def raw(self, name, phase):
        a = self.arrays[(phase, name)]
        return self.payload[a['offset']:a['offset'] + a['byte_count']]

    def f32(self, name, phase):
        raw = self.raw(name, phase)
        if len(raw) % 4:
            raise ValueError('unaligned raw float view')
        return np.frombuffer(raw, dtype='<f4')

    def summary(self):
        phases = sorted({key[0] for key in self.arrays})
        return dict(schema=self.manifest['schema'], root=self.manifest['root'],
            iteration=self.manifest['iteration'], trial=self.manifest['trial'],
            selected_phase=self.manifest['selected_phase'], array_count=len(self.arrays),
            payload_bytes=len(self.payload), payload_sha256=hashlib.sha256(self.payload).hexdigest(),
            adjacent_payload=str(self.payload_path),
            phase_counts={p: sum(key[0] == p for key in self.arrays) for p in phases},
            accepted_checkpoint=False, physical_validation=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    print(json.dumps(Capture(args.manifest).summary(), indent=2, allow_nan=False))
