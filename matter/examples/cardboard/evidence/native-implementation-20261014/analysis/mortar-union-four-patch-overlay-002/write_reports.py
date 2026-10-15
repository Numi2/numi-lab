from pathlib import Path
import hashlib,json,re
root=Path('/Users/home/cardboard-implementation-20261014')
pkg=root/'analysis/mortar-union-four-patch-overlay-002'
snap=root/'snapshots/source-rationalized-paper-and-native-checkers-091'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
text=(pkg/'hma_parent_pair_multiplicity.stdout.txt').read_text().strip()
lines=text.splitlines()
first=lines[0]
histlines=lines[1:]
fields=dict(item.split('=',1) for item in first.split())
hist={}
max_records=[]
for line in histlines:
    if line.startswith('multiplicity_'):
        k,v=line.split('=',1); hist[int(k.removeprefix('multiplicity_'))]=int(v)
    elif line.startswith('max_record_source='):
        d=dict(item.split('=',1) for item in line.split()); max_records.append(d)
headers=['matter/tools/cardboard_shipping_protocol.hpp','matter/tools/cardboard_shipping_closure_mesh.hpp','matter/tools/cardboard_starch_interface_mesh.hpp','matter/tools/cardboard_box_blank.hpp','matter/tools/cardboard_shipping_allocation_preflight.cpp']
summary={
 'schema':'numi.cardboard.shipping-hma-parent-pair-multiplicity.v1',
 'evidence_class':'exact CPU reconstruction using source091 shipping blank, starch duplication, and HMA mortar mesh builders; no native assembly or constitutive execution',
 'source_snapshot':str(snap),
 'source_snapshot_manifest_sha256':sha(snap/'frozen-source.json'),
 'source_headers_sha256':{rel:sha(snap/rel) for rel in headers},
 'analysis_program':{'path':str(pkg/'hma_parent_pair_multiplicity.cpp'),'sha256':sha(pkg/'hma_parent_pair_multiplicity.cpp'),'binary_sha256':sha(pkg/'hma_parent_pair_multiplicity'),'stdout_sha256':sha(pkg/'hma_parent_pair_multiplicity.stdout.txt')},
 'command':'xcrun --sdk macosx clang++ -std=c++20 -O2 -I <source091>/matter/tools -I <source091>/matter/include -I <source091>/include hma_parent_pair_multiplicity.cpp -o hma_parent_pair_multiplicity && ./hma_parent_pair_multiplicity',
 'geometry_inputs':{
  'inside_dimensions_m':[.300,.200,.200], 'manufacturer_joint_width_m':.025, 'slot_kerf_m':.001,
  'subdivisions_per_band':16, 'board_recipe_id':'literature2009', 'nx_per_pitch':8,
  'thickness_slices':1, 'engineering_offsets_m':[0,0,0],
  'starch_density_kg_m3_by_region':{'liner':766,'medium':679,'glue':1500},
  'hma_footprint_width_m':.010, 'hma_activation_gap_placeholder_m':.0001,
  'note':'Matches cardboard_shipping_allocation_preflight.cpp construction. Density values validate starch topology only and do not alter HMA face enumeration.'},
 'topology':{k:int(v) for k,v in fields.items() if v.isdigit()},
 'canonical_parent_primitive_semantics':'Each triangle side is keyed as 4*tetIndex + oppositeCorner(faceNodes), then the two side IDs are sorted. This matches the full-contact checker and canonical primitive-pair source grouping.',
 'multiplicity_histogram_combined_starch_plus_hma':{str(k):v for k,v in sorted(hist.items())},
 'max_pair_records':max_records,
 'verdict':'Under this exact shipping-source geometry and current builder, the maximum combined starch+HMA mortar record multiplicity is four; five same-parent patches are not needed by this geometry. Other resolutions/configurations remain unverified.'}
(pkg/'shipping_hma_parent_pair_multiplicity.json').write_text(json.dumps(summary,indent=2)+'\n')
readme=f'''# Bounded same-parent mortar union overlay (source 091)

This package implements a narrow shared-boundary repair for the private mortar-complement CCD path, raises the explicit same-canonical-parent-pair patch ceiling from three to four, and adds a bounded 4-patch admission/CCD fixture. The patch is against frozen source snapshot `source-rationalized-paper-and-native-checkers-091` and is in `mortar-union-four-patch-092.patch`.

The repair retains each covered cell's face/patch identity. It does not skip a covered-cell pair wholesale. For a pair of distinct mortar face IDs, it derives the candidate common edge from each authored face's mapped coordinates on both parent primitives. It filters a single vertex/triangle or edge/edge CCD witness only when both witness barycentric points lie on that common edge, their edge parameters match, and relative normal displacement is non-approaching within the source epsilon. Inward normal motion, a mismatched edge parameter, off-edge points, and interior witnesses remain contact candidates. Both production narrowphase and private Armijo sweep callers use the same code path.

The CPU predicate check is `cpu_boundary_contract_check.py`; its output is `cpu_boundary_contract_check.json`. It uses the exact float32 sector and midpoint maps authored by the fixture, with native clip/overlap/edge tolerances recorded. It verifies shared-edge endpoints/interior, separating/tangential/inward motion, parameter mismatch, off-edge, positive-area cyclic cross-mapping interior, and 4-triangle tiling. It is CPU-only and does not execute Metal CCD.

The C++ source/compiler check and shader build pass. The CPU-only executable reports source admission success with zero native FEM transactions. Native private-sweep and full-contact GPU tests have not been run by this worker. Native run 301 remains preserved and only reports an aggregate closure error; it did not capture an exact first pair/witness, so the pair attribution remains a CPU reconstruction. Do not treat compile/CPU tests as a native or physical qualification.

## Current shipping geometry ceiling

`shipping_hma_parent_pair_multiplicity.json` is a separate source-bound CPU reconstruction of the current default shipping source. It uses the actual default design (300×200×200 mm inside, 25 mm manufacturer joint, 1 mm slot kerf, 16 subdivisions per band, literature2009 recipe), 8 corrugation intervals per pitch and one through-thickness slice, then performs starch node duplication and the source HMA closure builder. Starch and HMA faces are grouped together by the same canonical parent primitive IDs used by the compiler.

The resulting mesh has 528,444 nodes, 1,477,728 tets, 97,984 starch paired faces, and 162,851 HMA mortar patches across 9 HMA groups. Combined records group into 183,033 canonical primitive pairs with multiplicity histogram 1:106,021, 2:76,227, 3:780, 4:5; no pair exceeds four. This supports a four-patch capacity for this exact geometry only. It does not prove actual glue-law behavior, assembled-box execution, other mesh resolutions, or any altered HMA footprint. Five-patch input remains rejected.

## Reproduce

From the overlay directory:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 cpu_boundary_contract_check.py
cmake --build build-source/build-overlay --target numi-matter-cardboard-full-contact-armijo-check -j2
build-source/build-overlay/matter/numi-matter-cardboard-full-contact-armijo-check build-source/matter/materials/hajali2009_liner_hill_asymmetric_c1_transfer.nmatter --cpu-only
```

The exact shipping pair-multiplicity program is compiled against frozen source 091 headers using the command in its JSON report. `manifest.json` binds source, patch, executable, metallib, CPU outputs, and hashes. No native GPU run is included.
'''
(pkg/'README.md').write_text(readme)
print(json.dumps({'shipping_report_sha256':sha(pkg/'shipping_hma_parent_pair_multiplicity.json'),'readme_sha256':sha(pkg/'README.md')},indent=2))
