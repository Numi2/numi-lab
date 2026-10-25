#!/usr/bin/env python3
"""Bounded source-identity comparison for 027 O rows; no geometry repair."""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np

E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D = E / "wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001"
OUT = E / "current-027-detailed-skin-lineage-001"
inputs = {
    "candidate_027": E / "o12-outward-repair-027-f32-gated/candidate.npz",
    "baseline_014": E / "baseline-depth-transition-014/candidate.npz",
    "embedding_036": D / "generated/embedding.csv",
    "nodes_036": D / "generated/nodes.csv",
    "faces_036": D / "generated/faces.csv",
    "registered_128_csv": D / "provenance/registered-skin-128-candidate.csv",
    "registered_128_payload": E / "left-ankle-rest-preserving-weight-repair-128/candidate.nhskin",
    "base_294_payload": Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin"),
    "D36_inputs": D / "provenance/build_candidate_rebind_inputs.json",
    "D36_report": D / "D36-remap-report.json",
    "D36_builder": D / "provenance/build_candidate_rebind.py",
    "lane_ancestry": Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001/lane-ancestry-core.json"),
    "physical_preparation_header_D36": D / "source-overlay/apps/NumiHumanCoarseTissuePreparation.hpp",
    "renderer_transfer_header_D36": D / "source-overlay/apps/NumiHumanAcceptedFEMSkinTransfer.hpp",
    "coarse_source_consumer_header_D36": D / "source-overlay/apps/NumiHumanWholeBodySkinEmbeddingConsumer.hpp",
    "resting_visual_header_D36": D / "source-overlay/apps/NumiHumanRestingVisual.hpp",
}
expected = {
    "candidate_027": "ccfeb27e6e939249e4c5146f77722bf2477d1f8377300bde8cc82086f44bbe1a",
    "baseline_014": "f5b0e33933ddf0867715a877ade75279659034362ae8f586435ccbf00609f42c",
    "embedding_036": "715e685cb75c9b160fa81abe6e1ed9ae2c85cc85551f1ac6dfe0bdf3d0fa4847",
    "nodes_036": "5ae63af6c1f43fc38a53422452b685b6b1644c890b6b88d9d108cef64a208a11",
    "faces_036": "64b481ba0e65f4787211a1f2a0fac54bdc10b4b35e812ed30dbee6ae11d28a3c",
    "registered_128_payload": "4fd80f449b2c4e6a94b0808c2e54a5735684711de1f7fc1c34e893c2494250a5",
    "base_294_payload": "d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04",
    "D36_inputs": "44d0b41c12a48d41e98cf39562b045649c99e62ea2cc74a5eaf2eae5477cbde0",
    "D36_report": "e6aa24a7cc475aa828767fadd026169fd08c273e68802e987f04187848352d89",
    "D36_builder": "4b91de03d6eef0d383983e3a617a68567b517d6f87a2786c95cd916a31d713c9",
    "registered_128_csv": "ba19f07304c36e4c3cdcd13068bbd66589bedbd51f22621f947dd0873858a3d6",
    "lane_ancestry": "3d2886415b86c1d8f44c30b9aaebd97eed8feecc86f1c76df794e8b428cc87db",
    "physical_preparation_header_D36": "0f185acc77dc67a49587ba2a20b7873fb6cfba14cdfc18cff30198d579a18c62",
    "renderer_transfer_header_D36": "b7f23c1d9f7279e05f995fcc1cb1d417675f3ffac845220e42ffc8204cf2a72d",
    "coarse_source_consumer_header_D36": "fdcec56d0c7af908995d3b4e4719bd2a59fa5aa835cd0efb081d29b448621efb",
    "resting_visual_header_D36": "92c3ba5b1cab09aee31871d46ba8eee74277392953872f3cbf037476ca958a1e",
}
def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()
actual = {k: sha(p) for k,p in inputs.items()}
assert all(actual[k] == v for k,v in expected.items()), {k:(actual[k],expected.get(k)) for k in actual if actual[k] != expected.get(k)}
base=np.load(inputs["baseline_014"],allow_pickle=False)
cand=np.load(inputs["candidate_027"],allow_pickle=False)
assert base["points_world_f32"].shape == cand["points_world_f32"].shape == (7233,3)
assert base["tetrahedra"].shape == cand["tetrahedra"].shape == (28632,4)
assert np.array_equal(base["tetrahedra"],cand["tetrahedra"])
N=2411
p0=base["points_world_f32"].reshape(3,N,3)
p1=cand["points_world_f32"].reshape(3,N,3)
changed=[np.flatnonzero(np.any(p0[layer]!=p1[layer],axis=1)) for layer in range(3)]
max_move_mm=[float(np.max(np.linalg.norm(p1[i]-p0[i],axis=1))*1000.0) for i in range(3)]
with inputs["embedding_036"].open(newline="") as f: emb=list(csv.DictReader(f))
with inputs["registered_128_csv"].open(newline="") as f: reg=list(csv.DictReader(f))
assert len(emb)==N and len(reg)==61950
assert [int(r["row"]) for r in emb]==list(range(N))
assert [int(r["vertex"]) for r in reg]==list(range(len(reg)))
source_vertices=[int(emb[i]["source_vertex"]) for i in changed[0]]
assert all(0<=i<len(reg) for i in source_vertices)
donors=Counter(int(reg[i]["donor"]) for i in source_vertices)
# Reconstruct the D36 source O-face set from the same indexed 014 tetrahedra.
from itertools import combinations
freq=Counter()
for tet in base["tetrahedra"]:
    for face in combinations(map(int,tet),3):
        if max(face)<N: freq[tuple(sorted(face))]+=1
source_o_faces={f for f,c in freq.items() if c==1}
with inputs["faces_036"].open(newline="") as f:
    d36_o_faces={tuple(sorted((int(r["a"]),int(r["b"]),int(r["c"])))) for r in csv.DictReader(f)}
assert len(source_o_faces)==len(d36_o_faces)==4777
assert source_o_faces==d36_o_faces
# Current source checkout is deliberately only hash-pinned as a worktree observation.
import subprocess
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
head=subprocess.check_output(["git","-C",str(R),"rev-parse","HEAD"],text=True).strip()
status=subprocess.check_output(["git","-C",str(R),"status","--short","--","apps/NumiHumanCoarseTissuePreparation.hpp","apps/NumiHumanRestingVisual.hpp"],text=True).strip()
current_code={
 "apps/NumiHumanCoarseTissuePreparation.hpp": sha(R/"apps/NumiHumanCoarseTissuePreparation.hpp"),
 "apps/NumiHumanAcceptedFEMSkinTransfer.hpp": sha(R/"apps/NumiHumanAcceptedFEMSkinTransfer.hpp"),
 "apps/NumiHumanWholeBodySkinEmbeddingConsumer.hpp": sha(R/"apps/NumiHumanWholeBodySkinEmbeddingConsumer.hpp"),
 "apps/NumiHumanRestingVisual.hpp": sha(R/"apps/NumiHumanRestingVisual.hpp"),
}
report={
 "schema":"numi.human.current-027-detailed-skin-lineage.v1",
 "status":"BOUNDED_LINEAGE_ONLY_NO_PHYSICAL_TRANSFER",
 "inputs":{k:{"path":str(inputs[k]),"sha256":actual[k]} for k in inputs},
 "candidate_vs_baseline":{
   "same_point_and_tet_shapes":True,"same_tetrahedra_exact":True,
   "rows_per_layer":N,"changed_rows_by_layer":[int(len(x)) for x in changed],
   "max_displacement_mm_by_layer":max_move_mm,
   "changed_O_rows_local":list(map(int,changed[0])),
   "changed_O_rows_direct_D36_source_vertices":dict(zip(map(str,map(int,changed[0])),source_vertices)),
   "changed_O_donor_core_counts":{str(k):v for k,v in sorted(donors.items())},
 },
 "surface_identity":{
   "D36_generated_O_faces":len(d36_o_faces),"014_tetrahedral_O_boundary_faces":len(source_o_faces),
   "unordered_source_index_face_sets_equal":True,
 },
 "source_lineage_scope":{
   "D36_candidate_payload_tag":"128",
   "D36_base_payload":"294",
   "D36_embedding_direction":"coarse O rows to source vertex or source face+barycentric ancestry",
   "D36_embedding_rows":len(emb),
   "D36_direct_source_vertex_rows":sum(0<=int(r["source_vertex"])<len(reg) for r in emb),
   "D36_source_face_bary_rows":sum(int(r["source_vertex"])>=len(reg) for r in emb),
   "027_changes_have_direct_D36_source_vertex_ancestry":True,
   "027_npz_contains_only_points_tets_material_ids_no_ancestry_or_transfer_rows":True,
 },
 "physical_owner":{
   "entrypoint":"NumiHumanCoarseTissuePreparation.hpp buildRegisteredSkinFatCandidateLayers -> appendRegisteredSkinFatFEMFromLayers/appendRegisteredSkinFatFEM",
   "requires":"full detailed skin/fat candidate positions and source ancestry/bindings/material/thickness data; this bounded 027-to-D36 map supplies no detailed vertex prolongation and no I/D ancestry",
   "AcceptedFEMSkinTransfer":"maps accepted coarse O displacement to detailed rows for presentation/rendering; not a physical NHTISS/FEM rest-geometry update",
   "makeWholeBodySkinSourceConsumerRows":"coarse-row source ancestry consumer; not an inverse detailed-vertex displacement map",
   "missing_for_transfer":["a validated detailed294 vertex-to-027-O displacement/prolongation map including anchored outside rows and interface closure","I/D layer ancestry or an explicit physical layer rebuild policy","a current-294-bound physical candidate construction and geometry/mass/material gates"],
 },
 "current_checkout_observation":{"head":head,"relevant_worktree_status":status,"file_sha256":current_code},
 "qualification":{"physical_qualification":False,"native_candidate_admission":False,"region_labels":"typed donor/joint IDs only; no coordinate-based anatomical region labels"},
}
(OUT/"lineage-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(OUT/"lineage-report.json"),"changed_rows_by_layer":report["candidate_vs_baseline"]["changed_rows_by_layer"],"max_displacement_mm_by_layer":max_move_mm,"O_faces":len(source_o_faces),"donor_core_counts":dict(donors),"current_head":head},indent=2))
