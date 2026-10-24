from pathlib import Path
import shutil,json,difflib
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");N=E/"native-candidate020-conforming-prism-030";A=N/"apps"
shutil.copytree(A,N/"apps-attempt-004")
shutil.copy2(N/"candidate-identity.json",N/"candidate-identity-attempt-004.json")
shutil.copy2(N/"app-integration.diff",N/"app-integration-attempt-004.diff")
p=A/"NumiHumanSkinFatReductionLabels.hpp";old=p.read_text();s=old
x='    if(basisByStableFace.size()>numi::matter::kHumanPartitionedBasisMaximumRegionCount)'
assert s.count(x)==1
s=s.replace(x,'''    std::set<std::uint32_t> emittedBasisLabels;
    for(const auto& [face,label]:basisByStableFace) {
        (void)face;emittedBasisLabels.insert(label);
    }
    if(emittedBasisLabels.size()>numi::matter::kHumanPartitionedBasisMaximumRegionCount)''')
p.write_text(s)
(N/"basis-capacity-count-fix.diff").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="024/NumiHumanSkinFatReductionLabels.hpp",tofile="030/NumiHumanSkinFatReductionLabels.hpp")))
p=A/"numilab_human_myosim_visual_probe.mm";s=p.read_text();old='                            skinBasisMaxFacesPerRegion,verifyEmbedded),'
assert s.count(old)==1
s=s.replace(old,'''                            skinBasisPatchCount!=0u?skinBasisMaxFacesPerRegion:
                                static_cast<std::uint32_t>(region.faces.size()),verifyEmbedded),''')
p.write_text(s)
(N/"basis-admission-policy.json").write_text(json.dumps({"scope":"Source labels only; no capacity or physical gate relaxation","capacity":"distinct actual emitted basis labels, not faces","baseline_policy":"split_skin_patch_count=0 keeps one existing donor region per group; no requested subdivision","unchanged_cell_basis_policy":"exact baseline IDs reused","full_human_native_validation":"pending005"},indent=2))
