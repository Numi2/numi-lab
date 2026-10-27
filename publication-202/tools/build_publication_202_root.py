from pathlib import Path
import hashlib,json,shutil
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D");S=D/"surface-clearance-546/component-sat-selector-638";W=S/"per-bank-workers-20261027";T=D/"publication-202-stage-001"
T.mkdir(exist_ok=False)
def add(src,rel):
 src=Path(src);assert src.is_file(),src;dst=T/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def optional(src,rel):
 if Path(src).is_file():add(src,rel)
add(S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz","inputs/common-parent-7f2.npz")
add(S/"partial-midpoint-continuation-root-001/same-parent-bone-selector653-provenance-001/candidate.npz","inputs/ten-flips-308.npz")
add(W/"b65-99/root-layer-guard-002/backtrack-0p25/candidate.npz","inputs/baseline-a69.npz")
R=W/"root-topology-a69-safe-subset-001"
for name in ["candidate.npz","rebuild_subset.py","preflight.json","shape-preflight.json","layer-identity-comparison.json","full-static-comparison-653-root.json","compare_full.py","transfer_layers.py","test_reflected_layer_offsets.py"]:add(R/name,"six-flips/"+name)
for name in ["audit-130/summary.json","audit-159/report.json","layer-diagnostic/report.json","layer-diagnostic/skin-fat-layers-541.npz","layer-reflection-diagnostic-001/report.json"]:add(R/name,"six-flips/"+name)
add(W/"root-topology-a69-rebase-001/layer-identity-comparison.json","six-flips/ten-flips-layer-comparison.json")
add(W/"root-topology-d071-selector-001/selector-report.json","six-flips/next-selector-report.json")
N=W/"b109-114/attempt-025-a69-full-audits-001"
for name in ["static-construction-increment.json","static-construction-node-patch.npz","a69-full-audit-comparison.json","audit-130/summary.json","audit-159/report.json"]:add(N/name,"bank-109/"+name)
add(W/"b109-114/attempt-024-a69-layer-fit-backtrack-001/alpha-0p015625/candidate.npz","bank-109/candidate-468.npz")
add(W/"b61-97/root-layer-guard-004/candidate-private.npz","bank-61/candidate-c65.npz")
A=W/"b43-16/attempt-024-witness-constrained-fit-001"
for name in ["report.json","layer-transfer-check.json","exact-self-check.json","scoped-43-16-crossings.json"]:add(A/name,"bank-43/failed-fit/"+name)
A=W/"b43-16/attempt-025-witness-cluster-topology-001"
for name in ["witness-cluster-preflight.json","cluster_witness_components.py","build_component0_flip.py","component-00-oneflip/report.json","component-00-oneflip/candidate-private.npz","component-00-oneflip/layer-virtual-topology-check.json","component-00-oneflip/layer_virtual_topology_check.py"]:add(A/name,"bank-43/cluster-patch/"+name)
optional(A/"component-00-oneflip/exact-topology-delta.json","bank-43/cluster-patch/component-00-oneflip/exact-topology-delta.json")
for name in ["other-witness-rings-3-8.json","ring_boundary_check_3_8.py"]:optional(A/name,"bank-43/cluster-patch/"+name)
A=W/"b65-99/attempt-019-exact-witness-layer-guard-001/scoped-exact-run-001"
add(A/"final-receipt.json","bank-65/failed-fit/final-receipt.json")
A=W/"b65-99/attempt-021-boundary-expansion-001"
for name in ["run-001/expansion-receipt.json","expand_boundary_by_measured_witnesses.py","check_expanded_boundary.py"]:add(A/name,"bank-65/expanded-boundary/"+name)
# Include final optional receipts and compact source files from bounded workers.
for root,folder in [(W/"b65-99/attempt-022-boundary-owner-flip-001","bank-65/owner-flip"),(W/"b43-16/attempt-025-witness-cluster-topology-001","bank-43/cluster-extra"),(W/"b43-16/attempt-026-patch-cdt-owner-001","bank-43/cdt-feasibility")]:
 if root.is_dir():
  for p in root.rglob("*"):
   if p.is_file() and p.suffix in (".json",".py",".npz") and p.stat().st_size<12000000 and "__pycache__" not in p.parts:
    rel=p.relative_to(root)
    if folder=="bank-43/cluster-extra" and len(rel.parts)<=2:continue
    dst=T/folder/rel
    if not dst.exists():add(p,str(Path(folder)/rel))
for root in sorted((W/"b109-114").glob("attempt-026*")):
 if root.is_dir():
  for p in root.rglob("*"):
   if p.is_file() and p.suffix in (".json",".py",".npz") and p.stat().st_size<12000000 and "__pycache__" not in p.parts:add(p,str(Path("bank-109/remesh-feasibility")/root.name/p.relative_to(root)))
for root in sorted((W/"b65-99").glob("attempt-023*")):
 if root.is_dir():
  for p in root.rglob("*"):
   if p.is_file() and p.suffix in (".json",".py",".npz") and p.stat().st_size<12000000 and "__pycache__" not in p.parts:add(p,str(Path("bank-65/remesh-feasibility")/root.name/p.relative_to(root)))
for root in sorted((W/"b43-16").glob("attempt-027*")):
 if root.is_dir():
  for p in root.rglob("*"):
   if p.is_file() and p.suffix in (".json",".py",".npz") and p.stat().st_size<12000000 and "__pycache__" not in p.parts:add(p,str(Path("bank-43/current-preflight")/root.name/p.relative_to(root)))
# The merge worker writes a self-contained receipt if its candidate qualifies.
for root in sorted(W.glob("combined-d071-b109-b61-*")):
 if root.is_dir():
  for p in root.rglob("*"):
   if p.is_file() and p.suffix in (".json",".py",".npz") and p.stat().st_size<12000000 and "__pycache__" not in p.parts:add(p,str(Path("combined-position")/root.name/p.relative_to(root)))
add(D/"build_publication_202_root.py","tools/build_publication_202_root.py")
add(D/"publish_publication_202_root.py","tools/publish_publication_202_root.py")
readme="""# Numi Human hand clearance: six-flip construction checkpoint\n\nThis is offline, reference-pose construction evidence. It is not an admitted skin, an integrated native simulation, a complete anatomy, a natural-shape certificate, or clinical validation.\n\nThe frozen a69 baseline had 983 raw child-triangle/bone-triangle intersections, 651 mapped parent/bone triangle identities, zero exact nonadjacent skin self pairs, and 125 sampled skin-node/bone inside identities. The pinned candidate d071f04fa64ce602f08dbf7859c45b4518ef2032ac92af0c1100d69dbdd59826 changes six same-parent edges, yielding 977 raw pairs while retaining the same 651 parent identities, 0 skin self pairs and 125 sampled inside identities. Exact pair-identity results and all 185 typed bank checks are included. Fewer raw pairs after retriangulation do not prove that a parent crossing or penetration is eliminated.\n\nChanged faces passed the local original/proposed layer orientation, Float32 nonpositive and normalized determinant checks, and unchanged faces gained no new detected failure identities. This is not full physical qualification: the transferred offset-prism diagnostic has 7,612 skin and 8,530 fat authored-orientation failures, two nonpositive Float32 tets per layer, missing conservative donor cell-volume and mass ancestry, and 45 skin boundary edges. No accepted physical steps, dynamic pose checks, mass conservation, complete solid containment, global layer collisions, or native scene admission are shown.\n\nThe small 109/114 position candidate 468254c2d44472bbcaf91b86f968536477fa2847b8581bc69003c3eb2395e0e7 made 24 inside samples shallower by at least 0.000300 mm but did not reduce raw or parent crossings. The 61/97 candidate c65 also preserves the same crossing and inside identities while reducing one sampled depth. Their individual receipts and source inputs are included. Combined candidate 806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756 preserves d071 crossing and inside identities, has zero surface self pairs, and retains the sampled depth improvements. A comparison bug that called a selected-triangle-plane value a signed distance was corrected to use classification-signed nearest_distance_m; all 13,181 shared rows passed the displacement-bound check, and the superseded plane metric remains labeled diagnostic evidence. It is not admitted to physical simulation.\n\nThe 43/16 connected-component flip improved triangle shape but retained invalid layer tets with worse signed sums and did not improve exact crossing identities; it remains rejected evidence. The 65/99 expanded boundary and failed fit are retained. A normal-reflection diagnostic increased authored failures from 7,612 to 7,630 for skin and 8,530 to 8,600 for fat; it was not admitted.\n\nUse earlier independently verified evidence archives P200 (7d7d8d42c7ad4486dda5399b7d7a2045cf7b836b) and P201 (ba47ffb4e78333787f09417fc055120a2a430648) for prior attempts and larger pinned inputs. Work was performed on SSH Numi; owner checkout 868486402da2ff0a208bf60c267df0928e2f91e0 had unrelated dirty changes, which were not modified. This branch is evidence only and does not change product main.\n\nThe exterior derives from BodyParts3D FJ2810, © The Database Center for Life Science, CC Attribution 4.0 International, https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html and https://creativecommons.org/licenses/by/4.0/ . It was registered, retopologized, refined and locally displaced. The scene manifest declares MyoSim Apache-2.0 for registration and musculoskeletal source. This is a mixed-source reference assembly, not measurements of one person.\n"""
(T/"README.md").write_text(readme)
files=sorted(p for p in T.rglob("*") if p.is_file());members=[{"path":p.relative_to(T).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
(T/"publication-manifest.json").write_text(json.dumps({"version":1,"status":"offline construction evidence only","accepted_physical_steps":0,"geometry_admitted":False,"members":members},indent=2)+"\n")
print("staged",len(members),"files",sum(m["bytes"] for m in members),"bytes")
