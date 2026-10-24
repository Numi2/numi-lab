from pathlib import Path
import numpy as np,json,hashlib,shutil,difflib,subprocess,time,shlex
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/"native-candidate020-mechanical-source-024"
C=E/"native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003"
N=E/"native-candidate020-conforming-prism-030"
if (N/"candidate-identity.json").exists():raise SystemExit("refuse to overwrite an existing prepared or frozen candidate")
N.mkdir(exist_ok=True);A=N/"apps";shutil.copytree(D/"apps",A,dirs_exist_ok=True);B=N/"build-001";B.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(C/"candidate.npz")=="640b7bc56013748b26d5e4e12790f85a472602ee6a4732f5773243df6e2147cc"
assert sha(C/"report.json")=="cba3dabe05563ce5b6c7e659c395b577236e734395ce6edd5b21620536d1fb26"
a=np.load(C/"candidate.npz")
assert np.array_equal(np.flatnonzero(np.any(a["tetrahedra"]!=a["old_tetrahedra"],axis=1)),a["changed_tet_rows"])
assert np.array_equal(a["material_ids"],a["old_material_ids"])
offsets=a["parent_csr_offsets"];parents=a["parent_csr_old_tet_rows"];fractions=a["parent_csr_old_volume_fractions"]
dominant=np.array([parents[offsets[i]+np.argmax(fractions[offsets[i]:offsets[i+1]])] for i in range(len(a["tetrahedra"]))])
def arr(name,ctype,values):
 vals=np.asarray(values)
 if vals.ndim==2:
  body=",\n".join("    std::array<std::uint32_t,4>{"+",".join(str(int(v))+"u" for v in row)+"}" for row in vals)
  return "inline constexpr std::array<std::array<std::uint32_t,4>,"+str(len(vals))+"> "+name+"{{\n"+body+"\n}};\n"
 body=",\n".join("    "+(format(float(v),".17g") if ctype=="double" else str(int(v))+"u") for v in vals)
 return "inline constexpr std::array<"+ctype+","+str(len(vals))+"> "+name+"{{\n"+body+"\n}};\n"
header="#pragma once\n#include <array>\n#include <cstdint>\nnamespace numi::human::prism030 {\n"
for name,ctype,key in [
 ("childTetrahedra","", "tetrahedra"),("oldTetrahedra","", "old_tetrahedra"),
 ("parentOffsets","std::uint32_t","parent_csr_offsets"),("parentRows","std::uint32_t","parent_csr_old_tet_rows"),
 ("parentFractions","double","parent_csr_old_volume_fractions")]:
 header+=arr(name,ctype,a[key])
header+=arr("dominantParent","std::uint32_t",dominant)
header+='inline constexpr const char* receiptSHA256="'+sha(C/"report.json")+'";\n}\n'
hp=A/"NumiHumanConformingPrism030.hpp";hp.write_text(header)
identity={"schema":"numi.human.skin.attachment-candidate-identity.d38.v1","candidate_id":"candidate020-d37-conforming-prism-030","base_candidate_id":"candidate020-d37-current-unpaired163-owner-integration-038","base_identity_sha256":sha(D/"D38-candidate-identity.json"),"candidate_npz_sha256":sha(C/"candidate.npz"),"overlap_verification_report_sha256":sha(C/"report.json"),"topology_header_sha256":sha(hp),"scope":"16-cell conforming topology replacement through existing unpaired source owner; exact unchanged nodes/boundary; conservative CSR; not admitted","physical_qualification":False}
ip=N/"candidate-identity.json";ip.write_text(json.dumps(identity,indent=2,sort_keys=True)+"\n")
p=A/"numilab_human_myosim_visual_probe.mm";old=p.read_text();s=old
def rep(x,y):
 global s
 assert s.count(x)==1,(x[:100],s.count(x))
 s=s.replace(x,y)
rep('#include "NumiHumanUnpairedSkinFatVolume.hpp"','#include "NumiHumanUnpairedSkinFatVolume.hpp"\n#include "NumiHumanConformingPrism030.hpp"')
rep("candidate020-d37-current-unpaired163-owner-integration-038",identity["candidate_id"])
rep("ad204691d202ee3455cb28653c297fca8a3094bcce6a838e069343d315f6f39b",sha(ip))
# The private candidate always exercises the existing unpaired owner. Keep the
# original exact identity test in its parent snapshot; do not mutate that record.
rep('if(identityPreflightEnv!=nullptr&&std::string_view(identityPreflightEnv)=="1") {',
    'if(true) { // private prism030 candidate, guarded by exact candidate ID and SHA above')
anchor='                seedIdentityFromReport(preparedFat,"fat");'
rep(anchor,anchor+'''
                namespace prism030=numi::human::prism030;
                require(identityTets.size()==prism030::oldTetrahedra.size()&&
                        std::equal(identityTets.begin(),identityTets.end(),prism030::oldTetrahedra.begin()),
                        "prism030 parent connectivity differs from the pinned D37 source");
                const auto parentTets=identityTets;
                const auto parentStableFaces=identityStableFaces;
                std::vector<std::uint32_t> parentTetDonor(tetCount,NM_INVALID_INDEX);
                for(std::size_t ti=0;ti<tetCount;++ti) {
                    std::map<std::uint32_t,unsigned> counts;
                    for(auto node:parentTets[ti])++counts[region.donorBodyIndices[node%surfaceCount]];
                    auto chosen=counts.begin();
                    for(auto it=counts.begin();it!=counts.end();++it)
                        if(it->second>chosen->second)chosen=it;
                    parentTetDonor[ti]=chosen->first;
                }
                identityTets.assign(prism030::childTetrahedra.begin(),prism030::childTetrahedra.end());
''')
rep('std::vector<std::uint32_t> identityParentOffsets(tetCount+1u);',
    'std::vector<std::uint32_t> identityParentOffsets(prism030::parentOffsets.begin(),prism030::parentOffsets.end());')
rep('std::vector<std::uint32_t> identityParentRows(tetCount);',
    'std::vector<std::uint32_t> identityParentRows(prism030::parentRows.begin(),prism030::parentRows.end());')
rep('std::vector<double> identityParentFractions(tetCount,1.0);',
    'std::vector<double> identityParentFractions(prism030::parentFractions.begin(),prism030::parentFractions.end());')
rep('                    identityParentOffsets[ti]=ti;identityParentRows[ti]=ti;\n','')
rep('                identityParentOffsets[tetCount]=static_cast<std::uint32_t>(tetCount);\n','')
rep('identityTopology.childDominantParentTetrahedra=identityParentRows;',
    'identityTopology.childDominantParentTetrahedra=prism030::dominantParent;')
rep('identityTopology.parentTetrahedra=identityTets;','identityTopology.parentTetrahedra=parentTets;')
rep('identityTopology.parentStableSourceFaceIdentities=identityStableFaces;',
    'identityTopology.parentStableSourceFaceIdentities=parentStableFaces;')
rep('identityTopology.parentDonorBodyIndices=identityTetDonor;','identityTopology.parentDonorBodyIndices=parentTetDonor;')
rep('identityTopology.ancestryVerificationReceiptSHA256=identityReceipt;',
    'identityTopology.ancestryVerificationReceiptSHA256=prism030::receiptSHA256;')
start=s.index('                bool exactIdentityTets=')
end=s.index('\n            }\n            if(partitionedReductionDiagnostic||',start)
new=r'''
                require(identityWorld.objects.size()==source.objects.size()&&
                            identityWorld.objects.back().femNodes==paired.femNodes&&
                            identityWorld.objects.back().femMaterialIndices==paired.femMaterialIndices&&
                            identityWorld.objects.back().femContactNodes==paired.femContactNodes,
                        "prism030 changed exact nodes, materials or contact node ownership");
                require(std::abs(identityPreparedSkin.volumeM3-preparedSkin.volumeM3)<=1.0e-8*preparedSkin.volumeM3&&
                            std::abs(identityPreparedFat.volumeM3-preparedFat.volumeM3)<=1.0e-8*preparedFat.volumeM3&&
                            identityPreparedSkin.massDebitKg==preparedSkin.massDebitKg&&
                            identityPreparedFat.massDebitKg==preparedFat.massDebitKg,
                        "prism030 changed regional material volume or mass debit");
                // Existing callers consume one combined nodal donor ledger in
                // preparedSkin, while retaining separate skin/fat volume and ancestry.
                std::map<std::uint32_t,DonorMassNode> combined;
                for(const auto* report:{&identityPreparedSkin,&identityPreparedFat})
                    for(const auto& mass:report->donorMassNodes) {
                        const auto found=combined.find(mass.nodeIndex);
                        if(found==combined.end())combined.emplace(mass.nodeIndex,mass);
                        else {
                            require(found->second.donorBody==mass.donorBody&&
                                    found->second.localPosition==mass.localPosition,
                                    "prism030 per-material nodal donor identities disagree");
                            found->second.massKg+=mass.massKg;
                        }
                    }
                require(combined.size()==volumeCount,"prism030 donor ledger does not cover every node");
                double combinedMass=0.0;
                identityPreparedSkin.donorMassNodes.clear();
                for(const auto& [node,mass]:combined) {
                    (void)node;combinedMass+=mass.massKg;
                    identityPreparedSkin.donorMassNodes.push_back(mass);
                }
                require(std::abs(combinedMass-skinDebit-fatDebit)<=1.0e-8*(skinDebit+fatDebit),
                        "prism030 combined nodal mass differs from regional debits");
                identityPreparedSkin.layerThicknessMinM=preparedSkin.layerThicknessMinM;
                identityPreparedSkin.layerThicknessMaxM=preparedSkin.layerThicknessMaxM;
                identityPreparedFat.layerThicknessMinM=preparedFat.layerThicknessMinM;
                identityPreparedFat.layerThicknessMaxM=preparedFat.layerThicknessMaxM;
                // Build source labels while every topology span is alive.
                const auto verifyBound=[&](std::uint32_t sourceVertex,std::uint32_t donor) {
                    return sourceVertex<skin->vertices.size()&&dominantSkinDonorBody(*skin,sourceVertex)==donor;
                };
                const auto parentAt=[&](std::uint32_t body) {
                    return body<model.bodies.size()?model.bodies[body].parentBody:NM_INVALID_INDEX;
                };
                const auto verifyEmbedded=[&](std::uint32_t row,std::uint32_t donor) {
                    return row<volumeCount&&
                        nodeDonor[row]==donor&&
                        (row%surfaceCount)<wholeBodySkinSourceRows.dominantBody.size()&&
                        wholeBodySkinSourceRows.dominantBody[row%surfaceCount]==donor;
                };
                RegisteredSkinFatReductionSourceLabels prismLabels;
                require(buildRegisteredSkinFatUnpairedReductionSourceLabels(region,
                            identityWorld.objects.back(),wholeBodySkinBoundaryNodeMap,identityTopology,
                            model.bodies.size(),verifyBound,parentAt,prismLabels,tissueError,
                            skinBasisMaxFacesPerRegion,verifyEmbedded),
                        "prism030 unpaired source labels rejected: "+tissueError);
                skinFatCaptureNodeRegionLabels=prismLabels.sourceNodeDonorBodyIndices;
                skinFatCaptureTetRegionLabels=prismLabels.tetrahedronRegionLabels;
                skinFatReductionSourceLabels=std::move(prismLabels);
                source.objects.back()=std::move(identityWorld.objects.back());
                preparedSkin=std::move(identityPreparedSkin);
                preparedFat=std::move(identityPreparedFat);
                explicitTopology.reset();
                std::cout<<"whole_body_skin_fat_conforming_prism=PASS"
                    <<" nodes="<<identityReport.volumeNodeCount<<" tets="<<identityReport.tetrahedronCount
                    <<" changed_tets=16 reference_m3="<<(preparedSkin.volumeM3+preparedFat.volumeM3)
                    <<" mass_kg="<<combinedMass<<" ancestry_receipt_sha256="<<prism030::receiptSHA256
                    <<" source_world_mutated=1 physical_qualification=not_claimed"<<std::endl;
'''
# Use the donor-row concrete type from the existing vector, avoiding a new API.
new=new.replace('std::map<std::uint32_t,DonorMassNode>',
                'std::map<std::uint32_t,typename decltype(identityPreparedSkin.donorMassNodes)::value_type>')
s=s[:start]+new+s[end:]
rep('            if(partitionedReductionDiagnostic||\n               partitionedPreconditionerRequested||endpointCaptureRequested) {',
    '            if(!skinFatReductionSourceLabels&&(partitionedReductionDiagnostic||\n               partitionedPreconditionerRequested||endpointCaptureRequested)) {')
p.write_text(s)
(N/"app-integration.diff").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="024/app.mm",tofile="030/app.mm")))
for label in ("compile","link"):
 cmd=json.loads((D/"build-003"/(label+"-command.json")).read_text())
 cmd=[str(A/x[len(str(D/"apps"))+1:]) if x.startswith(str(D/"apps")+"/") else
      str(A) if x==str(D/"apps") else
      str(B/Path(x).name) if x.startswith(str(D/"build-003")+"/") else x for x in cmd]
 (B/(label+"-command.json")).write_text(json.dumps(cmd,indent=2)+"\n")
 t=time.monotonic()
 with (B/(label+".log")).open("wb") as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 result={"returncode":r.returncode,"wall_seconds":time.monotonic()-t}
 (B/(label+"-result.json")).write_text(json.dumps(result,indent=2)+"\n");print(label,result,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
deps=shlex.split((B/"candidate.d").read_text().split(":",1)[1].replace("\\\n",""))
expected=[x.name for x in A.iterdir() if x.is_file() and x.suffix==".hpp"]
for name in expected:
 got=[Path(x) for x in deps if x.endswith("/"+name)]
 assert got==[A/name],(name,got)
pins={x:sha(x) for x in deps if Path(x).is_file() and x.startswith("/Users/")}
(B/"dependency-certification.json").write_text(json.dumps({"status":"PASS_EXACT_OVERLAY","input_hashes":pins,"binary_sha256":sha(B/"numi-human-native"),"physical_qualification":False},indent=2)+"\n")
print("binary",sha(B/"numi-human-native"),flush=True)
