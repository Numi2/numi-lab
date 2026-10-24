from pathlib import Path
import json,hashlib,shutil
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");N=E/"native-candidate020-conforming-prism-030";A=N/"apps"
shutil.copytree(A,N/"apps-attempt-005")
shutil.copy2(N/"candidate-identity.json",N/"candidate-identity-attempt-005.json")
shutil.copy2(N/"app-integration.diff",N/"app-integration-attempt-005.diff")
p=A/"numilab_human_myosim_visual_probe.mm";s=p.read_text()
x='"BodyParts3D opt-in exact identity remap diagnostic",'
assert s.count(x)==1;s=s.replace(x,'paired.name,')
x='                identityTopology.referenceSourceIdentity=materialIdentity;\n                identityTopology.materialSourceIdentity=materialIdentity;'
assert s.count(x)==1
s=s.replace(x,r'''
                std::string prismSourceBytes;
                prismSourceBytes.append(reinterpret_cast<const char*>(materialIdentity.data()),sizeof(materialIdentity));
                prismSourceBytes.append(reinterpret_cast<const char*>(identityTets.data()),identityTets.size()*sizeof(identityTets.front()));
                prismSourceBytes.append(reinterpret_cast<const char*>(identityParentOffsets.data()),identityParentOffsets.size()*sizeof(identityParentOffsets.front()));
                prismSourceBytes.append(reinterpret_cast<const char*>(identityParentRows.data()),identityParentRows.size()*sizeof(identityParentRows.front()));
                prismSourceBytes.append(reinterpret_cast<const char*>(identityParentFractions.data()),identityParentFractions.size()*sizeof(identityParentFractions.front()));
                prismSourceBytes.append(prism030::receiptSHA256);
                const auto prismSourceDigest=loadedKneeSHA256(prismSourceBytes.data(),prismSourceBytes.size());
                std::array<std::uint64_t,4u> prismSourceIdentity{};
                for(unsigned word=0;word<4u;++word)for(unsigned byte=0;byte<8u;++byte)
                    prismSourceIdentity[word]=(prismSourceIdentity[word]<<8u)|prismSourceDigest[8u*word+byte];
                identityTopology.referenceSourceIdentity=prismSourceIdentity;
                identityTopology.materialSourceIdentity=prismSourceIdentity;
''')
x='                source.objects.back()=std::move(identityWorld.objects.back());'
assert s.count(x)==1
s=s.replace(x,r'''
                // Preserve the existing physical object descriptor, including name,
                // boundary/attachment settings, velocities and contact policy.
                // Only the explicitly validated FEM topology/reference payload changes.
                auto replacement=paired;
                auto& generated=identityWorld.objects.back();
                replacement.tetrahedra=std::move(generated.tetrahedra);
                replacement.femNodes=std::move(generated.femNodes);
                replacement.femReferenceNodes=std::move(generated.femReferenceNodes);
                replacement.femReferenceSourceIdentity=generated.femReferenceSourceIdentity;
                replacement.femMaterialSourceIdentity=generated.femMaterialSourceIdentity;
                replacement.femMaterialIndices=std::move(generated.femMaterialIndices);
                replacement.femContactNodes=std::move(generated.femContactNodes);
                source.objects.back()=std::move(replacement);
                source.materials=std::move(identityWorld.materials);
''')
p.write_text(s)
(N/"object-binding-fix.json").write_text(json.dumps({"reason":"native001 stopped before physical stepping because diagnostic name did not match accepted boundary object owner","prior_run":"native-1ms-001","source_snapshot":"apps-attempt-005","change":"retain paired ObjectSource descriptor and install only validated FEM payload; unique reference/material source identity includes child topology and CSR; use unpaired-owner material density parameters","physics_gates_relaxed":False},indent=2)+"\n")
