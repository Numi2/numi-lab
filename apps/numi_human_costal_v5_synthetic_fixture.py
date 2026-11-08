#!/usr/bin/env python3
"""Build an explicitly synthetic NHCART1/NHTBIND2 v5 admission fixture.

The required fourteen region identifiers describe decoder coverage only, not
anatomical provenance. All 112 tetrahedra are separate; both selected rigid
body IDs are artificial test owners. Never use this fixture as Human anatomy.
"""
import argparse
import hashlib
from pathlib import Path
import struct

p = argparse.ArgumentParser()
p.add_argument('rigid', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--sternal-body', type=int, default=20)
p.add_argument('--rib-body', type=int, default=21)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
rigid = a.rigid.read_bytes()
if rigid[:8] != b'NHRIGID2':
    raise SystemExit('invalid rigid magic')
body_count = struct.unpack_from('<I', rigid, 8 + 3*4)[0]
if min(a.sternal_body, a.rib_body) < 0 or max(a.sternal_body, a.rib_body) >= body_count or a.sternal_body == a.rib_body:
    raise SystemExit('invalid distinct owners')
registration = b'{"synthetic_fixture":true,"anatomical_registration":false}\n'
(a.output / 'synthetic-registration.json').write_bytes(registration)
member = ['FJ3239','FJ3242','FJ3245','FJ3248','FJ3251','FJ3254','FJ3255',
          'FJ3333','FJ3335','FJ3337','FJ3339','FJ3341','FJ3343','FJ3345']
nodes, tets, regions = [], [], []
for region in range(14):
    start_n, start_t = len(nodes), len(tets)
    for tetra in range(8):
        x = 0.0025 * tetra
        y = 0.003 * (region % 7)
        z = 0.003 * (region // 7)
        points = [(x,y,z), (x+0.002,y,z), (x,y+0.002,z), (x,y,z+0.002)]
        base = len(nodes)
        for j, point in enumerate(points):
            flags = 1 if tetra < 4 else 2
            nodes.append(struct.pack('<4fIII', *point, 1e-6, flags, region, 0))
        tets.append(struct.pack('<5I', base, base+1, base+2, base+3, region))
    regions.append(struct.pack('<8s10I4f', member[region].encode(), region//7,
        region%7+1, start_n, 32, start_t, 8, 16, 16, 4, 4,
        1e-8, 1e-8, .002, 0))
header = struct.pack('<8s6I2f32s', b'NHCART1\0', 1, 14, len(nodes), len(tets),
                     len(nodes), 0, 1000., .004, hashlib.sha256(b'synthetic-no-atlas').digest())
cartilage = header + b''.join(regions) + b''.join(nodes) + b''.join(tets)
(a.output / 'synthetic.nhcartilage').write_bytes(cartilage)
transform = (1.,0.,0.,0.1, 0.,1.,0.,1., 0.,0.,1.,-0.2, 0.,0.,0.,1.)
binding = (struct.pack('<8s6I', b'NHTBIND2', 2, 14, body_count, len(nodes), len(tets), 0)
           + hashlib.sha256(cartilage).digest() + hashlib.sha256(rigid).digest()
           + hashlib.sha256(registration).digest() + struct.pack('<16d', *transform))
binding += b''.join(struct.pack('<3I', 0xffffffff, a.sternal_body, a.rib_body) for _ in range(14))
binding += b''.join(struct.pack('<I', a.sternal_body if i%32<16 else a.rib_body) for i in range(len(nodes)))
(a.output / 'synthetic.nhtbind').write_bytes(binding)
print(f'body_count={body_count} nodes={len(nodes)} tets={len(tets)} owners={a.sternal_body},{a.rib_body} synthetic=true')
