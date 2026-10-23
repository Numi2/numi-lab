from pathlib import Path
import hashlib,json
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
D=E/"sparse-bone-contact-135"; O=D/"source-overlay"
shader="matter/src/metal/numi_human.metalinc"; host="matter/src/numi_human.mm"
s=(R/shader).read_text(); start=s.index("kernel void nm_numi_human_assemble_fem_body_contact_loads("); end=s.index("\nkernel void ",start+1);old=s[start:end]
new=old.replace("const uint total = dispatch.environmentCount * dispatch.femNodeCount;", "// Admission guarantees one FEM/body-contact sample per slave node.\n    // A sample-domain launch therefore has disjoint writes, without atomics.\n    const uint total = dispatch.environmentCount * dispatch.femBodyContactSampleCount;").replace("const uint environment = global / dispatch.femNodeCount;\n    const uint node = global - environment * dispatch.femNodeCount;", "const uint environment = global / dispatch.femBodyContactSampleCount;\n    const uint index = global - environment * dispatch.femBodyContactSampleCount;\n    const NMNumiHumanFEMBodyContactSampleGPU sample = samples[index];\n    const uint node = sample.slaveNode;\n    if (node >= dispatch.femNodeCount || !validBody(sample.bodyIndex, dispatch)) return;")
a=new.index("    float3 nodeForce = float3(0.0f);");b=new.index("        float3 slavePoint;",a)
new=new[:a]+new[b:]
a=new.index("        nodeForce += normal * (pressure * contactArea);")
new=new[:a]+"        externalForces[nodeBase + node].xyz += normal * (pressure * contactArea);\n}\n"
# Remove obsolete loop indentation, leaving function statements conventionally aligned.
lines=new.splitlines(); new="\n".join(x[4:] if x.startswith("        ") else x for x in lines)+"\n"
s=s[:start]+new+s[end:]
h=(R/host).read_text(); before="state_->femBodyContactForcePipeline,\n                state_->environmentCount * state_->nodeLoads.size(),";after="state_->femBodyContactForcePipeline,\n                state_->environmentCount * state_->femBodyContactSolverSamples.size(),"
assert h.count(before)==1;h=h.replace(before,after)
for name,data in [(shader,s),(host,h)]:
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
# Keep the pre-change GPU kernel as an independent oracle in the private coupon only.
q=O/"tests/numi_human_fem_body_triangle_contact";q.mkdir(parents=True,exist_ok=True)
for p in (R/"tests/numi_human_fem_body_triangle_contact").iterdir():
 if p.is_file(): (q/p.name).write_bytes(p.read_bytes())
p=q/"fem_body_triangle_contact_coupon.metal";p.write_text(p.read_text()+"\nnamespace numi_matter_metal {\n"+old.replace("nm_numi_human_assemble_fem_body_contact_loads(","nm_numi_human_assemble_fem_body_contact_loads_dense_oracle(")+"\n}\n")
p=q/"fem_body_triangle_contact_coupon.mm";c=p.read_text();c=c.replace("encodeOwner(@\"nm_numi_human_assemble_fem_body_contact_loads\",\n                    count,ownerForceBuffer);","encodeOwner(@\"nm_numi_human_assemble_fem_body_contact_loads\",\n                    dispatch.femBodyContactSampleCount,ownerForceBuffer);")
p.write_text(c)
# Original math and attachment coupon dependencies are unchanged.
q2=O/"tests/numi_human_fem_surface_attachment";q2.mkdir(parents=True,exist_ok=True)
for p in (R/"tests/numi_human_fem_surface_attachment").iterdir():
 if p.is_file(): (q2/p.name).write_bytes(p.read_bytes())
D.mkdir(parents=True,exist_ok=True)
(D/"source-preparation.json").write_text(json.dumps({"scope":"one contact thread per admitted unique slave; no contact law/state changes","original_shader_sha256":hashlib.sha256((R/shader).read_bytes()).hexdigest(),"source_overlay":str(O)},indent=2))
print(D)
