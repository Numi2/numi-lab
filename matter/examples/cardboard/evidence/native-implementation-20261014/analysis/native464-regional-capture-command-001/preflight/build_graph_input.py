#!/usr/bin/env python3
"""Extract source graph inputs from one exact compiled.nmatterpack."""
from __future__ import annotations
import hashlib, json, struct
from pathlib import Path

PACK = Path('/Users/home/cardboard-implementation-20261014/solver/cgls-block-right-disabled-parity-462/output/compiled.nmatterpack')
OUT = Path(__file__).resolve().parent
raw = PACK.read_bytes()
if len(raw) < 56:
    raise SystemExit('short package header')
magic, version, endian, abi, section_count, physics, fingerprint, header_hash = struct.unpack_from('<16sIIIIQQQ', raw, 0)
if magic != b'NUMIMATTERPKG\0\0\0' or endian != 0x01020304:
    raise SystemExit('bad package identity')
# The source uses 64-bit FNV-1a for package/section hashing.
def fnv64(buf: bytes, seed: int = 1469598103934665603) -> int:
    h = seed
    for b in buf:
        h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h
header_zero = bytearray(raw[:56]); header_zero[48:56] = bytes(8)
if fnv64(header_zero) != header_hash:
    raise SystemExit('package header hash mismatch')
off = 56
sections = {}
for _ in range(section_count):
    if off + 32 > len(raw): raise SystemExit('truncated section header')
    sid, elem, count, nbytes, content_hash = struct.unpack_from('<IIQQQ', raw, off)
    off += 32
    if nbytes != elem * count or off + nbytes > len(raw):
        raise SystemExit(f'bad section extent {sid}')
    content = raw[off:off+nbytes]
    if fnv64(content) != content_hash:
        raise SystemExit(f'section hash mismatch {sid}')
    if sid in sections: raise SystemExit(f'duplicate section {sid}')
    sections[sid] = {'element_size': elem, 'element_count': count, 'byte_count': nbytes,
                     'content_hash_u64': f'{content_hash:016x}', 'offset': off, 'content': content}
    off += nbytes
if off != len(raw): raise SystemExit('trailing bytes')
# NMMatterDispatchGPU section is one 176-byte record. The fields are u32s in
# shared.h; the first 16 u32 words through parameterCount contain FEM counts.
d = sections[1]
if (d['element_size'], d['element_count']) != (192,1):
    raise SystemExit(f'unexpected dispatch section: {d["element_size"]}/{d["element_count"]}')
words = struct.unpack_from('<40I', d['content'], 0)
envs, objects, materials, params = words[2], words[3], words[4], words[5]
nodes, tets, material_stride = words[8], words[9], words[21]
if (envs, nodes, tets, material_stride) != (2,882,1920,15):
    raise SystemExit(f'compiled dimensions differ from intended case: E={envs} N={nodes} T={tets} stride={material_stride}')
node = sections[19]; tet = sections[20]
if (node['element_size'],node['element_count']) != (80,nodes) or (tet['element_size'],tet['element_count']) != (96,tets):
    raise SystemExit('FEM node/tetrahedron record layout mismatch')
# NMFEMNodeStateGPU.restAndFixed.w is record float lane 11 (byte 44).
free = bytearray(nodes)
for i in range(nodes):
    fixed = struct.unpack_from('<f', node['content'], i*80+44)[0]
    if fixed not in (0.0,1.0): raise SystemExit(f'nonbinary fixed tag node={i} value={fixed!r}')
    free[i] = 1 if fixed == 0.0 else 0
conn = []
for i in range(tets):
    row = struct.unpack_from('<4I', tet['content'], i*96)
    if any(x >= nodes for x in row): raise SystemExit(f'bad tet node id tet={i}')
    conn.extend(row)
# Binary ABI for the exact C++ layout builder: E,N,T, then T*4 indices, N mask bytes.
binout = bytearray(struct.pack('<III', envs, nodes, tets))
binout.extend(struct.pack('<' + 'I'*len(conn), *conn))
binout.extend(free)
(OUT/'actual-graph-input.bin').write_bytes(binout)
inventory = {
 'schema':'numi.cardboard.regional-graph-preflight-input.v1',
 'compiled_package':str(PACK),'compiled_package_sha256':hashlib.sha256(raw).hexdigest(),
 'compiled_package_bytes':len(raw),'package_version':version,'matter_abi':abi,
 'physics_fingerprint_u64_hex':f'{physics:016x}','package_fingerprint_u64_hex':f'{fingerprint:016x}',
 'dispatch':{'environment_count':envs,'object_count':objects,'material_count':materials,
             'parameter_count':params,'node_count':nodes,'tetrahedron_count':tets,
             'material_state_stride':material_stride},
 'sections':{str(k):{k2:v2 for k2,v2 in v.items() if k2!='content'} for k,v in sections.items()},
 'fixed_mask_encoding':'raw u8, 1=free, 0=fixed; source tag restAndFixed.w read from byte offset 44 of 80-byte NMFEMNodeStateGPU',
 'free_nodes':sum(free),'fixed_nodes':nodes-sum(free),
 'tetrahedron_encoding':'little-endian u32x4 first 16 bytes of each 96-byte NMTetrahedronGPU',
 'source_topology_sha256':hashlib.sha256(struct.pack('<II', nodes, tets)+struct.pack('<' + 'I'*len(conn), *conn)).hexdigest(),
 'source_free_mask_sha256':hashlib.sha256(free).hexdigest(),
 'potential_node_mask_sha256':hashlib.sha256(bytes([1])*nodes).hexdigest(),
 'graph_input_sha256':hashlib.sha256(binout).hexdigest(),
 'graph_input_bytes':len(binout),
}
(OUT/'compiled-package-topology-inventory.json').write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:inventory[k] for k in ('compiled_package_sha256','dispatch','free_nodes','fixed_nodes','graph_input_sha256','graph_input_bytes')},indent=2))
