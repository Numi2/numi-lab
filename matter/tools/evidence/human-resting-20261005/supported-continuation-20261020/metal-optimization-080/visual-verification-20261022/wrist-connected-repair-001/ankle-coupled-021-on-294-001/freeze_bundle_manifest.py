#!/usr/bin/env python3
"""Hash-bound inventory for the isolated ankle candidate bundle."""
import hashlib, json
from pathlib import Path
E=Path(__file__).resolve().parent
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
W=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
EXT={
"raw_021_npz":(W/"ankle-coupled-geometry-weight-021/ankle-coupled-021-candidate.npz","ea5184dcc4652c318f0ad818dfcb135f95ed569953719727625881d6ce3a0aec"),
"exact_index_map":(W/"ankle-coupled-geometry-weight-021/ankle-patch-069i-to-294-exact-index-map.npz","39011d487170e7b535295887aa96c73d485db20f96a0edf1999faf912749192d"),
"exact_correspondence":(W/"ankle-coupled-geometry-weight-021/ankle-patch-069i-to-294-exact-correspondence.json","52f9dc4010a17c78a7ac341632270886011e00a4a90006125d138d7ea119ef05"),
"upstream_069i_payload":(W/"finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin","a848a41368aeb0dda8af579fe0ea417bbf7ec7db641033b5ba917302189fe1bc"),
"base_294_payload":(Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin"),"d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04"),
"main_294_reduction_header":(Path("/Users/n/numi-human-separation-20261018/C/wholebody-integration-295/build-wholebody-016/source/apps/NumiHumanWholeBodySkinReduction.hpp"),"a0e6d69706611257556c74ab7815ca2605f1f3c5f5530d9dd3472ecf482f5200"),
"reference_snapshot":(W/"native-registered-skin-pose-clean-014/registered-reference-elevation-010-014/registered-skin-poses.json","f4b375a3c1476b9e4dcf506fc4001afa8ab4079070644af7b7d473cb43366818"),
"dorsiflexion_snapshot":(W/"native-registered-joint-reference-pairs-016/ankle-dorsiflexion-020-ref010/registered-skin-poses.json","04de51e3e5ec9e6c736aab3c1217aceb4f6780254da16070e7d8278dbebfd425"),
"plantarflexion_snapshot":(W/"native-registered-joint-reference-pairs-016/ankle-plantarflexion-020-ref010/registered-skin-poses.json","37cee243eef596013c1b71fa6862334b8dae343fe7df60b966026b05dfc1243f"),
"compileworld_link_033":(Path("/tmp/numi-human-joint-owner-build-033/lib/libmetalrobo.dylib"),"b4c83d0df4edec0cc00fb40853158499a9d34d90e315a63f9af554b065657149"),
"source_021_solid_occupancy_receipt":(W/"ankle-solid-occupancy-root-036/solid-occupancy.json","8e9a30b92e6c024e4c8a785669b01e105026d02fded69e29885875af20450c6e"),
}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
errors=[]; external=[]
for name,(p,want) in EXT.items():
 if not p.is_file(): errors.append("missing "+name+": "+str(p)); continue
 got=sha(p)
 if got!=want: errors.append("hash mismatch "+name+": "+got)
 external.append({"label":name,"path":str(p),"sha256":got,"expected_sha256":want})
manifest=E/"bundle-manifest.json"
internal=[{"path":str(p.relative_to(E)),"size_bytes":p.stat().st_size,"sha256":sha(p)}
 for p in sorted(E.rglob("*")) if p.is_file() and p!=manifest and "__pycache__" not in p.parts and p.suffix!=".pyc"]
data={"schema":"numi.human.ankle-021-on-294-bundle-manifest.v1","candidate_status":"proposal_only_not_active_or_dynamically_admitted","root":str(E),"reproduction_command":str(E/"native-compileworld-candidate-001/run_cpu_checks.sh"),"internal_files":internal,"external_inputs_and_build_artifacts":external,"errors":errors}
manifest.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
if errors: raise SystemExit("\n".join(errors))
print(json.dumps({"manifest":str(manifest),"internal_file_count":len(internal),"external_file_count":len(external),"sha256":sha(manifest),"status":"hashes_verified"},indent=2))
