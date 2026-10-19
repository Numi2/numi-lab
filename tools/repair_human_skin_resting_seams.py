#!/usr/bin/env python3
"""Apply the ten reviewed, orientation-preserving edge flips to the pinned NHSKIN shell."""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
from pathlib import Path

SOURCE_DEFAULT = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/bilateral-hand-skin-repair-20261018-001/source-hand-wrist-preserved-025/bodyparts3d-myosim-skinned-shell.nhskin")
FIXTURE_DEFAULT = Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-001/candidate-fixture-001.txt")
CSV_DEFAULT = Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/native-native-root-pose-003/registered-skin.csv")
EXPECTED_SOURCE_SHA = "2d6b0870d1ba0dd6b0944b6a0175ca0304f0392d9ec4458521938a3a5766ad68"
EXPECTED_FIXTURE_SHA = "4fa1146b9d153ed2bd63f19bf27214ece09e9001773fb36d9328d5fb8e331396"
EXPECTED_CSV_SHA = "4dd63097a035f5f385712307f2336ef6fd46c964d1b93a2e85d84193ac12d327"
CANDIDATE_SHA = "d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04"
FLIPS = [
    {"edge":[40733,55305],"faces":[68004,68005],"new":[[55303,55305,55306],[55303,55306,40733]]},
    {"edge":[43508,55416],"faces":[71300,71301],"new":[[55413,55416,55419],[55413,55419,43508]]},
    {"edge":[45162,55483],"faces":[74617,74618],"new":[[55484,55483,55480],[55484,55480,45162]]},
    {"edge":[9872,55254],"faces":[19569,19570],"new":[[55256,9872,9874],[55256,9874,55254]]},
    {"edge":[54950,55333],"faces":[69149,69150],"new":[[55331,55333,55335],[55331,55335,54950]]},
    {"edge":[55000,55437],"faces":[73682,73683],"new":[[55436,55437,55438],[55436,55438,55000]]},
    {"edge":[7645,55158],"faces":[14898,14899],"new":[[55157,55158,55159],[55157,55159,7645]]},
    {"edge":[54950,55322],"faces":[69153,69154],"new":[[55321,55322,55323],[55321,55323,54950]]},
    {"edge":[46305,55295],"faces":[77824,77825],"new":[[55294,46305,55296],[55294,55296,55295]]},
    {"edge":[46582,55323],"faces":[78385,78386],"new":[[55322,46582,55324],[55322,55324,55323]]},
]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_fixture(path: Path):
    vals = list(map(int, path.read_text().split()))
    pos = 0
    vertex_count, face_count = vals[:2]
    pos = 2
    source_vertices = vals[pos:pos + vertex_count]
    pos += vertex_count
    faces = [vals[pos + 3*i:pos + 3*i + 3] for i in range(face_count)]
    pos += 3 * face_count
    source_faces = vals[pos:pos + face_count]
    pos += face_count
    assert pos == len(vals) and len(set(source_vertices)) == vertex_count
    assert len(set(source_faces)) == face_count
    return source_vertices, faces, source_faces

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_DEFAULT,
                        help="optional exact compact fixture to verify face/source ancestry")
    parser.add_argument("--registered-csv", type=Path, default=CSV_DEFAULT,
                        help="optional pinned registered-pose table recorded in the manifest")
    parser.add_argument("--no-fixture", action="store_true", help="apply to the pinned source by face ID without compact-fixture validation")
    args = parser.parse_args()
    source_path = args.source.resolve()
    output_dir = args.output_dir.resolve()
    fixture_path = None if args.no_fixture else args.fixture.resolve()
    csv_path = args.registered_csv.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assert source_path.is_file() and sha(source_path) == EXPECTED_SOURCE_SHA, "source asset missing or hash mismatch"
    assert fixture_path is None or fixture_path.is_file(), "fixture missing"
    if fixture_path is not None:
        assert sha(fixture_path) == EXPECTED_FIXTURE_SHA, "fixture hash mismatch"
    csv_sha = sha(csv_path) if csv_path.is_file() else None
    if csv_sha is not None:
        assert csv_sha == EXPECTED_CSV_SHA, "registered CSV hash mismatch"

    src = source_path.read_bytes()
    magic, abi, binding_count, vertex_count, index_count, flags = struct.unpack_from("<8s5I", src, 0)
    assert magic == b"NHSKIN1\0" and abi == 5 and binding_count == 86
    assert vertex_count == 61950 and index_count == 326961
    assert len(src) == 60 + binding_count*36 + vertex_count*56 + index_count*4 + vertex_count*86*4
    index_offset = 60 + binding_count*36 + vertex_count*56
    original_indices = list(struct.unpack_from(f"<{index_count}I", src, index_offset))
    face_count = index_count // 3
    planned = {fid: tuple(tri) for flip in FLIPS for fid, tri in zip(flip["faces"], flip["new"])}
    assert len(planned) == 20

    fixture_sha = None
    if fixture_path is not None:
        sv, compact_faces, source_faces = load_fixture(fixture_path)
        assert len(sv) == 54551 and max(sv) < vertex_count
        assert len(source_faces) == face_count and max(source_faces) < face_count
        compact_by_source = {source:i for i, source in enumerate(sv)}
        for compact_face, source_face in enumerate(source_faces):
            expected = tuple(sv[v] for v in compact_faces[compact_face])
            source_tri = tuple(original_indices[3*source_face:3*source_face+3])
            assert expected == planned.get(source_face, source_tri), (compact_face, source_face)
        for flip in FLIPS:
            assert all(v in compact_by_source for tri in flip["new"] for v in tri)
        fixture_sha = sha(fixture_path)

    changed = {}
    for flip in FLIPS:
        face_ids = flip["faces"]
        assert all(0 <= f < face_count for f in face_ids)
        old = [tuple(original_indices[3*f:3*f+3]) for f in face_ids]
        assert set(old[0]) | set(old[1]) == set(flip["new"][0]) | set(flip["new"][1])
        assert sum(tuple(sorted(e)) == tuple(sorted(flip["edge"]))
                   for tri in old for e in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0]))) == 2
        for fid, tri in zip(face_ids, flip["new"]):
            assert fid not in changed and len(set(tri)) == 3 and all(0 <= v < vertex_count for v in tri)
            changed[fid] = tuple(tri)

    candidate = bytearray(src)
    for fid, tri in changed.items():
        struct.pack_into("<3I", candidate, index_offset + 12*fid, *tri)
    candidate_path = output_dir / source_path.name
    assert candidate_path.resolve() != source_path
    candidate_bytes = bytes(candidate)
    if candidate_path.exists():
        assert candidate_path.read_bytes() == candidate_bytes, "refusing to overwrite a different candidate"
    else:
        candidate_path.write_bytes(candidate_bytes)
    assert sha(candidate_path) == CANDIDATE_SHA, "candidate bytes differ from reviewed output"
    assert candidate_bytes[:index_offset] == src[:index_offset]
    assert candidate_bytes[index_offset + index_count*4:] == src[index_offset + index_count*4:]

    new_indices = list(struct.unpack_from(f"<{index_count}I", candidate_bytes, index_offset))
    changed_faces = [f for f in range(face_count)
                     if original_indices[3*f:3*f+3] != new_indices[3*f:3*f+3]]
    assert sorted(changed_faces) == sorted(changed)
    def edge_inventory(indices):
        use = {}
        direction = {}
        degenerate = []
        for fi in range(face_count):
            a,b,c = indices[3*fi:3*fi+3]
            if len({a,b,c}) != 3:
                degenerate.append(fi)
            for x,y in ((a,b),(b,c),(c,a)):
                key = tuple(sorted((x,y)))
                use[key] = use.get(key,0) + 1
                direction[key] = direction.get(key,0) + (1 if (x,y)==key else -1)
        return use, direction, degenerate
    original_use, _, original_degenerate = edge_inventory(original_indices)
    candidate_use, candidate_direction, candidate_degenerate = edge_inventory(new_indices)
    assert not original_degenerate and not candidate_degenerate
    assert max(original_use.values()) <= 2 and max(candidate_use.values()) <= 2
    original_boundary = {k for k,v in original_use.items() if v == 1}
    candidate_boundary = {k for k,v in candidate_use.items() if v == 1}
    assert candidate_boundary == original_boundary
    orientation_conflicts = [k for k,v in candidate_direction.items() if candidate_use[k] == 2 and v != 0]
    assert not orientation_conflicts

    ancestry = []
    for flip in FLIPS:
        for fid, tri in zip(flip["faces"], flip["new"]):
            ancestry.append({"candidate_source_face_id":fid,
                             "parent_source_face_ids":flip["faces"],
                             "flipped_edge_source_vertex_ids":flip["edge"],
                             "candidate_triangle_source_vertex_ids":tri})
    manifest = {
        "candidate_asset":str(candidate_path), "candidate_sha256":sha(candidate_path),
        "source_asset":str(source_path), "source_sha256":sha(source_path),
        "fixture":str(fixture_path) if fixture_path else None, "fixture_sha256":fixture_sha,
        "registered_csv":str(csv_path) if csv_sha else None, "registered_csv_sha256":csv_sha,
        "header_binding_vertex_normal_weight_bytes_identical":True,
        "only_changed_region":"triangle index records for 20 faces in seven local seam patches",
        "changed_source_face_ids":sorted(changed), "face_count":face_count,
        "vertex_count":vertex_count, "binding_count":binding_count, "index_count":index_count,
        "degenerate_faces":len(candidate_degenerate),
        "edge_use_counts":{str(k):sum(v==k for v in candidate_use.values()) for k in sorted(set(candidate_use.values()))},
        "boundary_edges":len(candidate_boundary), "boundary_set_unchanged":candidate_boundary==original_boundary,
        "orientation_conflicts":len(orientation_conflicts), "ancestry":ancestry
    }
    (output_dir/"repair-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))

if __name__ == "__main__":
    main()
