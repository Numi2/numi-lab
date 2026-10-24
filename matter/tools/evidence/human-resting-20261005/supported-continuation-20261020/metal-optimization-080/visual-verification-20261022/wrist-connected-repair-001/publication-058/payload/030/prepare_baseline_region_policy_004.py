from pathlib import Path
import json,hashlib,shutil,difflib
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");N=E/"native-candidate020-conforming-prism-030";A=N/"apps"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
shutil.copytree(A,N/"apps-attempt-003")
shutil.copy2(N/"candidate-identity.json",N/"candidate-identity-attempt-003.json")
shutil.copy2(N/"app-integration.diff",N/"app-integration-attempt-003.diff")
p=A/"numilab_human_myosim_visual_probe.mm";s=p.read_text()
start=s.index('                std::vector<std::uint32_t> parentTetDonor(tetCount,NM_INVALID_INDEX);')
end=s.index('                identityTets.assign(prism030::childTetrahedra.begin()',start)
s=s[:start]+r'''
                // Preserve baseline regional identities by asking the existing
                // paired compiler, including original-parent and refined-face rules.
                const auto baselineBound=[&](std::uint32_t vertex,std::uint32_t donor) {
                    return vertex<skin->vertices.size()&&dominantSkinDonorBody(*skin,vertex)==donor;
                };
                const auto baselineParent=[&](std::uint32_t body) {
                    return body<model.bodies.size()?model.bodies[body].parentBody:NM_INVALID_INDEX;
                };
                const auto baselineEmbedded=[&](std::uint32_t row,std::uint32_t donor) {
                    return row<wholeBodySkinSourceRows.dominantBody.size()&&
                        wholeBodySkinSourceRows.dominantBody[row]==donor;
                };
                RegisteredSkinFatReductionSourceLabels baselineLabels;
                require(buildRegisteredSkinFatReductionSourceLabels(region,paired,
                            wholeBodySkinNodesPerLayer,model.bodies.size(),
                            baselineBound,baselineParent,baselineLabels,tissueError,
                            skinBasisMaxFacesPerRegion,std::span<const std::uint32_t>(skinBasisSplitTets),
                            skinBasisPatchCount,baselineEmbedded,
                            std::span<const std::uint32_t>(kWholeBodySkinMechanicalFaceIdentities),
                            explicitTopology?&*explicitTopology:nullptr,skinMaterialIndex,fatMaterialIndex,
                            (partitionedReductionDiagnostic||partitionedPreconditionerRequested)),
                        "prism030 baseline regional source validation rejected: "+tissueError);
                const auto parentTetDonor=baselineLabels.tetrahedronSourceDonorBodyIndices;
                require(parentTetDonor.size()==tetCount,"prism030 baseline donor label extent mismatch");
''' +s[end:]
# Reuse the existing basis IDs for current operators; only changed cells inherit
# their actual largest-overlap old cell's basis. No whole-body regrouping.
anchor='                skinFatCaptureNodeRegionLabels=prismLabels.sourceNodeDonorBodyIndices;'
assert s.count(anchor)==1
s=s.replace(anchor,r'''
                std::size_t changedBasisRows=0u;
                for(std::size_t ti=0;ti<tetCount;++ti) {
                    const auto parent=prism030::dominantParent[ti];
                    prismLabels.tetrahedronBasisRegionLabels[ti]=
                        baselineLabels.tetrahedronBasisRegionLabels[parent];
                    if(prismLabels.tetrahedronBasisRegionLabels[ti]!=
                            baselineLabels.tetrahedronBasisRegionLabels[ti])++changedBasisRows;
                }
                std::cout<<"prism030_baseline_basis_reuse changed_rows="<<changedBasisRows
                    <<" all_other_basis_ids_exact=1"<<std::endl;
''' +anchor)
p.write_text(s)
(N/"baseline-region-policy-fix.json").write_text(json.dumps({"status":"IMPLEMENTED_PENDING_BUILD","policy":"baseline existing paired source compiler supplies parent donor IDs and basis IDs; largest actual overlap supplies changed-cell representative","old_preflight":"source-preflight-003","source_snapshot":"apps-attempt-003"},indent=2))
