import json, math, struct
from pathlib import Path
out=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/common-sub-float32-root-local-frame-001")
p=out/"local_frame_precision_report.json"
d=json.loads(p.read_text())
def f32_from_bits(x): return struct.unpack("<f",struct.pack("<I",int(x)))[0]
for r in d["results"]:
    pinbits=r["pinned_world_f32_root_bits_from_retained_witness"][0]
    pin=tuple(f32_from_bits(x) for x in pinbits)
    deltas=[]
    for bits in r["body_local_root_round_trip_world_f32_bits"]:
        q=tuple(f32_from_bits(x) for x in bits)
        deltas.append(math.sqrt(sum((q[i]-pin[i])**2 for i in range(3))))
    r["world_f32_root_delta_from_pinned_exact_root_m"]=deltas
    r["world_f32_max_delta_from_pinned_exact_root_m"]=max(deltas,default=None)
    if r["bank"]=="43/16":
        r["world_f32_reemission_outcome"]="roots become distinct by one x-coordinate ULP; one root is displaced from the pinned shared Float32 point"
    else:
        r["world_f32_reemission_outcome"]="both roots collapse to the pinned shared Float32 point"
d["status"]="LOCAL_BODY_FRAME_PRESERVES_ROOT_DISTINCTION_BUT_WORLD_EMISSION_IS_NOT_SAFE"
d["conclusion"]="Body-local Float32 re-intersection preserves two distinct roots in both tested pairs and changes segment lengths by less than 0.25 percent. On transform back to accepted world Float32, 65/99 collapses again; 43/16 separates by one world-float ULP, moving one root away from the pinned shared Float32 point. The three source skin vertices in each tested face round-trip to their original world Float32 bits, but rounded local root coordinates are not exactly incident on the quantized triangles, so emitting them requires changing both surface triangulations. The existing local Metal ABI carries point influences for support queries, not the conforming triangle topology required here. A safe owner-integrated remesh needs a local-frame topology/transform contract; this probe changes no geometry or public format."
d["candidate_written"]=False
d["geometry_changed"]=False
d["full_or_all_bank_audit_run"]=False
d["native_admission"]=False
d["accepted_physical_steps"]=0
p.write_text(json.dumps(d,indent=2)+"\n")
print(json.dumps(d,indent=2))
