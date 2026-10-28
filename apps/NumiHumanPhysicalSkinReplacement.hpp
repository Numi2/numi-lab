#pragma once

#include "NumiHumanPhysicalSkinChildMassMapping.hpp"
#include "NumiHumanCompiledSkinFatMassCapture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct RegisteredSkinFatDonorMomentDelta {
    std::uint32_t donorBodyIndex=NM_INVALID_INDEX;
    double oldSourceMassKg=0.0;
    double newSourceMassKg=0.0;
    double sourceMassDeltaKg=0.0;
    std::array<double,3> oldSourceFirstMomentKgM{};
    std::array<double,3> newSourceFirstMomentKgM{};
    std::array<double,3> sourceFirstMomentDeltaKgM{};
};

struct RegisteredSkinFatReplacementAudit {
    std::uint32_t replacedObjectIndex=NM_INVALID_INDEX;
    std::uint32_t firstGlobalFEMNode=NM_INVALID_INDEX;
    double oldSkinDebitKg=0.0;
    double newSkinDebitKg=0.0;
    double oldFatDebitKg=0.0;
    double newFatDebitKg=0.0;
    bool comparedSourceDonorClosure=false;
    bool comparedVerifiedCookedParentBaseline=false;
    bool preservedExactIdentityContactPolicy=false;
    double maximumPreparedToCookedNodeMassErrorKg=0.0;
    double maximumPreparedToCookedNodeMassRelativeError=0.0;
    std::vector<RegisteredSkinFatDonorMomentDelta> sourceRawToCookedDeltas;
    bool inheritedSkinThicknessMetadata=false;
    bool inheritedFatThicknessMetadata=false;
    std::array<std::uint64_t,4> oldReferenceSourceIdentity{};
    std::array<std::uint64_t,4> newReferenceSourceIdentity{};
    std::array<std::uint64_t,4> oldMaterialSourceIdentity{};
    std::array<std::uint64_t,4> newMaterialSourceIdentity{};
    std::vector<RegisteredSkinFatDonorMomentDelta> sourceDonorDeltas;
    std::vector<RegisteredSkinFatDonorMomentDelta> nodeFrameOwnerDeltas;
};

namespace registered_skin_fat_replacement_detail {

struct DonorMoment {
    double massKg=0.0;
    std::array<double,3> firstMomentKgM{};
};

inline bool finite(const DonorMoment& value) {
    if(!std::isfinite(value.massKg)||value.massKg<0.0)return false;
    return std::all_of(value.firstMomentKgM.begin(),value.firstMomentKgM.end(),
        [](double x){return std::isfinite(x);});
}

inline bool closeMass(double a,double b) {
    return std::isfinite(a)&&std::isfinite(b)&&
        std::abs(a-b)<=1.0e-6*std::max({std::abs(a),std::abs(b),1.0e-12});
}

template<class Rows>
bool addRows(std::map<std::uint32_t,DonorMoment>& result,const Rows& rows,std::string& error) {
    for(const auto& row:rows) {
        if(row.donorBody==NM_INVALID_INDEX||!std::isfinite(row.massKg)||row.massKg<=0.0)
            {error="replacement donor ledger has an invalid body or mass";return false;}
        auto& total=result[row.donorBody];
        total.massKg+=row.massKg;
        for(unsigned axis=0;axis<3u;++axis) {
            const double position=row.localPosition[axis];
            if(!std::isfinite(position)) {
                error="replacement donor ledger has a nonfinite local point";return false;
            }
            total.firstMomentKgM[axis]+=row.massKg*position;
        }
        if(!finite(total)) {
            error="replacement donor ledger overflows its local mass moment";return false;
        }
    }
    return true;
}

inline bool sourceDonorMoments(const PreparedOrganicFEM& skin,const PreparedOrganicFEM& fat,
                               std::map<std::uint32_t,DonorMoment>& result,std::string& error) {
    const bool hasSource=!skin.donorMassContributions.empty()||!fat.donorMassContributions.empty();
    if(hasSource) {
        if(!addRows(result,skin.donorMassContributions,error)||
           !addRows(result,fat.donorMassContributions,error))return false;
    } else if(!addRows(result,skin.donorMassNodes,error)||
              !addRows(result,fat.donorMassNodes,error))return false;
    return true;
}

inline bool ownerDonorMoments(const PreparedOrganicFEM& skin,const PreparedOrganicFEM& fat,
                              std::map<std::uint32_t,DonorMoment>& result,std::string& error) {
    return addRows(result,skin.donorMassNodes,error)&&addRows(result,fat.donorMassNodes,error);
}

inline bool sameDonorMasses(const std::map<std::uint32_t,DonorMoment>& oldRows,
                            const std::map<std::uint32_t,DonorMoment>& newRows) {
    if(oldRows.size()!=newRows.size())return false;
    for(const auto& [donor,oldTotal]:oldRows) {
        const auto it=newRows.find(donor);
        if(it==newRows.end()||!closeMass(oldTotal.massKg,it->second.massKg))return false;
    }
    return true;
}

inline bool sameSourceClosures(const std::vector<ExplicitUnpairedVolumeSourceMassClosure>& oldRows,
                               const std::vector<ExplicitUnpairedVolumeSourceMassClosure>& newRows) {
    std::map<std::uint32_t,std::array<double,3>> oldByDonor,newByDonor;
    for(const auto& row:oldRows) {
        auto& total=oldByDonor[row.donorBodyIndex];
        total[0]+=row.inputSourceMassKg;
        total[1]+=row.cookedParentMassKg;
        total[2]+=row.childSourceMassKg;
    }
    for(const auto& row:newRows) {
        auto& total=newByDonor[row.donorBodyIndex];
        total[0]+=row.inputSourceMassKg;
        total[1]+=row.cookedParentMassKg;
        total[2]+=row.childSourceMassKg;
    }
    if(oldByDonor.size()!=newByDonor.size())return false;
    for(const auto& [donor,oldMass]:oldByDonor) {
        const auto it=newByDonor.find(donor);
        if(it==newByDonor.end())return false;
        for(unsigned i=0;i<3u;++i)if(!closeMass(oldMass[i],it->second[i]))return false;
    }
    return true;
}

inline bool densityDefault(const numi::matter::MaterialProgram& material,double& value) {
    const auto found=std::find_if(material.parameters.begin(),material.parameters.end(),
        [](const auto& parameter){return parameter.name=="density";});
    if(found==material.parameters.end()||!std::isfinite(found->defaultValue)||found->defaultValue<=0.0)
        return false;
    value=found->defaultValue;
    return true;
}

inline bool sourceMaterialMasses(const numi::matter::ObjectSource& object,
                                 const numi::matter::WorldSource& world,
                                 std::map<std::uint32_t,double>& totals,std::string& error) {
    if(object.representation!=numi::matter::Representation::fem||
       object.tetrahedra.empty()||object.femNodes.empty()||
       object.femMaterialIndices.size()!=object.tetrahedra.size()||
       (!object.femTetrahedronMassesKg.empty()&&
        object.femTetrahedronMassesKg.size()!=object.tetrahedra.size())) {
        error="replacement object has incomplete FEM topology/material mass rows";return false;
    }
    if(object.femTetrahedronMassesKg.empty()&&
       std::any_of(object.femTetrahedronMassSourceIdentity.begin(),
                   object.femTetrahedronMassSourceIdentity.end(),[](auto word){return word!=0u;})) {
        error="replacement object has a mass-source identity without per-tetrahedron masses";return false;
    }
    const auto& points=object.femReferenceNodes.size()==object.femNodes.size()?
        object.femReferenceNodes:object.femNodes;
    for(std::size_t cell=0;cell<object.tetrahedra.size();++cell) {
        const auto materialIndex=object.femMaterialIndices[cell];
        if(materialIndex>=world.materials.size()) {
            error="replacement object references an absent FEM material";return false;
        }
        double cellMass=0.0;
        if(!object.femTetrahedronMassesKg.empty()) {
            cellMass=object.femTetrahedronMassesKg[cell];
        } else {
            double density=0.0;
            if(!densityDefault(world.materials[materialIndex],density)) {
                error="replacement object material has no finite positive density default";return false;
            }
            const auto& tet=object.tetrahedra[cell].nodes;
            for(auto node:tet)if(node>=points.size()) {
                error="replacement tetrahedron references an absent FEM point";return false;
            }
            const double volume=std::abs(tetVolume(points[tet[0]],points[tet[1]],points[tet[2]],points[tet[3]]));
            if(!std::isfinite(volume)||volume<=0.0) {
                error="replacement tetrahedron has invalid reference volume";return false;
            }
            cellMass=density*volume;
        }
        if(!std::isfinite(cellMass)||cellMass<=0.0) {
            error="replacement tetrahedron mass is nonpositive or nonfinite";return false;
        }
        totals[materialIndex]+=cellMass;
        if(!std::isfinite(totals[materialIndex])) {
            error="replacement material cell-mass total overflowed";return false;
        }
    }
    return true;
}

inline bool selectedMaterialPair(const numi::matter::ObjectSource& object,
                                 std::uint32_t skinMaterialIndex,std::uint32_t fatMaterialIndex) {
    if(object.materialIndex!=skinMaterialIndex||object.femMaterialIndices.empty())return false;
    bool skin=false,fat=false;
    for(auto material:object.femMaterialIndices) {
        if(material==skinMaterialIndex)skin=true;
        else if(material==fatMaterialIndex)fat=true;
        else return false;
    }
    return skin&&fat;
}

inline bool finiteVector(const std::array<double,3>& value) {
    return std::all_of(value.begin(),value.end(),[](double x){return std::isfinite(x);});
}

inline bool preserveThicknessMetadata(const PreparedOrganicFEM& oldReport,
                                     PreparedOrganicFEM& nextReport,
                                     const RegisteredSurface& surface) {
    if(!std::isfinite(oldReport.layerThicknessMinM)||
       !std::isfinite(oldReport.layerThicknessMaxM)||
       oldReport.layerThicknessMinM<0.0||
       oldReport.layerThicknessMaxM<oldReport.layerThicknessMinM||
       (oldReport.layerThicknessMinM==0.0&&oldReport.layerThicknessMaxM==0.0))
        return false;
    nextReport.layerThicknessM=oldReport.layerThicknessM;
    nextReport.layerThicknessMinM=oldReport.layerThicknessMinM;
    nextReport.layerThicknessMaxM=oldReport.layerThicknessMaxM;
    const bool sameSurfacePayload=oldReport.meshSHA256==nextReport.meshSHA256&&
        oldReport.payloadSHA256==nextReport.payloadSHA256;
    if(sameSurfacePayload&&oldReport.layerThicknessBySurfaceVertexM.size()==surface.worldPoints.size())
        nextReport.layerThicknessBySurfaceVertexM=oldReport.layerThicknessBySurfaceVertexM;
    nextReport.layerThicknessProvenance=oldReport.layerThicknessProvenance.empty()
        ?"inherited from prior report; candidate thickness not remeasured"
        :oldReport.layerThicknessProvenance+"; inherited from prior report; candidate thickness not remeasured";
    return true;
}

inline bool identityPresent(const std::array<std::uint64_t,4>& identity) {
    return std::any_of(identity.begin(),identity.end(),[](auto word){return word!=0u;});
}

inline bool closeDensityForIdentityClone(double a,double b) {
    return std::isfinite(a)&&std::isfinite(b)&&
        std::abs(a-b)<=1.0e-12*std::max({std::abs(a),std::abs(b),1.0});
}

inline std::vector<RegisteredSkinFatDonorMomentDelta> momentDeltas(
        const std::map<std::uint32_t,DonorMoment>& oldRows,
        const std::map<std::uint32_t,DonorMoment>& newRows) {
    std::set<std::uint32_t> donors;
    for(const auto& [donor,row]:oldRows){(void)row;donors.insert(donor);}
    for(const auto& [donor,row]:newRows){(void)row;donors.insert(donor);}
    std::vector<RegisteredSkinFatDonorMomentDelta> result;
    result.reserve(donors.size());
    for(const auto donor:donors) {
        const auto oldIt=oldRows.find(donor),newIt=newRows.find(donor);
        const DonorMoment oldValue=oldIt==oldRows.end()?DonorMoment{}:oldIt->second;
        const DonorMoment newValue=newIt==newRows.end()?DonorMoment{}:newIt->second;
        RegisteredSkinFatDonorMomentDelta delta;
        delta.donorBodyIndex=donor;
        delta.oldSourceMassKg=oldValue.massKg;
        delta.newSourceMassKg=newValue.massKg;
        delta.sourceMassDeltaKg=newValue.massKg-oldValue.massKg;
        delta.oldSourceFirstMomentKgM=oldValue.firstMomentKgM;
        delta.newSourceFirstMomentKgM=newValue.firstMomentKgM;
        for(unsigned axis=0;axis<3u;++axis)
            delta.sourceFirstMomentDeltaKgM[axis]=newValue.firstMomentKgM[axis]-oldValue.firstMomentKgM[axis];
        result.push_back(delta);
    }
    return result;
}

} // namespace registered_skin_fat_replacement_detail

// Transactionally replace one existing final explicit skin/fat FEM object.
// The appended builder runs only on a copy containing the original world prefix.
// On any failed gate the world, prepared outputs, report and audit are untouched.
[[nodiscard]] inline bool replaceLastRegisteredSkinFatUnpairedVolume(
        numi::matter::WorldSource& world,
        const std::size_t oldObjectIndex,
        const RegisteredSurface& surface,
        const numi::human::SkinBoundaryNodeMap& boundaryMap,
        const ExplicitUnpairedVolumeTopology& topology,
        const std::uint32_t skinMaterialIndex,
        const std::string_view skinMaterialName,
        const std::string_view skinMaterialProvenance,
        const double skinDonorDebitKg,
        const std::uint32_t fatMaterialIndex,
        const std::string_view fatMaterialName,
        const std::string_view fatMaterialProvenance,
        const double fatDonorDebitKg,
        const std::size_t sourceBodyCount,
        const std::string_view objectName,
        const PreparedOrganicFEM& oldSkin,
        const PreparedOrganicFEM& oldFat,
        const ExplicitUnpairedVolumeReport* oldReport,
        PreparedOrganicFEM& newSkin,
        PreparedOrganicFEM& newFat,
        ExplicitUnpairedVolumeReport& newReport,
        RegisteredSkinFatReplacementAudit& audit,
        std::string& error) {
    using namespace registered_skin_fat_replacement_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(world.objects.empty()||oldObjectIndex>=world.objects.size()||
       oldObjectIndex+1u!=world.objects.size())
        return fail("replacement target must be the final source object");
    if(skinMaterialIndex==fatMaterialIndex||skinMaterialIndex>=world.materials.size()||
       fatMaterialIndex>=world.materials.size()||objectName.empty())
        return fail("replacement material/object identity is invalid");
    const auto& oldObject=world.objects[oldObjectIndex];
    if(oldObject.representation!=numi::matter::Representation::fem)
        return fail("replacement target is not an FEM source object");
    if(oldObject.name!=objectName||!selectedMaterialPair(oldObject,skinMaterialIndex,fatMaterialIndex))
        return fail("replacement target name/material ownership does not match the requested skin/fat object");
    if(oldObject.femNodes.empty()||oldObject.tetrahedra.empty()||
       oldObject.femReferenceNodes.size()!=oldObject.femNodes.size())
        return fail("replacement target has incomplete FEM reference rows");
    if(!identityPresent(topology.referenceSourceIdentity)||!identityPresent(topology.materialSourceIdentity))
        return fail("replacement topology reference/material source identities are not pinned");

    std::uint64_t globalNodeBase=0u;
    for(std::size_t i=0;i<oldObjectIndex;++i) {
        const auto count=world.objects[i].femNodes.size();
        if(count>std::numeric_limits<std::uint32_t>::max()-globalNodeBase)
            return fail("replacement prefix FEM node base overflows the source format");
        globalNodeBase+=count;
    }
    if(globalNodeBase>std::numeric_limits<std::uint32_t>::max())
        return fail("replacement prefix FEM node base exceeds the source format");
    if(oldSkin.firstGlobalFEMNode!=globalNodeBase)
        return fail("replacement old skin global node base mismatch: actual="+
            std::to_string(oldSkin.firstGlobalFEMNode)+" expected="+std::to_string(globalNodeBase));
    if(oldSkin.materialIndex!=skinMaterialIndex)
        return fail("replacement old skin prepared material index mismatch: actual="+
            std::to_string(oldSkin.materialIndex)+" expected="+std::to_string(skinMaterialIndex));
    if(oldFat.materialIndex!=fatMaterialIndex)
        return fail("replacement old fat prepared material index mismatch: actual="+
            std::to_string(oldFat.materialIndex)+" expected="+std::to_string(fatMaterialIndex));
    if(world.materials[skinMaterialIndex].name!=skinMaterialName)
        return fail("replacement skin material program name mismatch: world="+
            world.materials[skinMaterialIndex].name+" requested="+std::string(skinMaterialName));
    if(world.materials[fatMaterialIndex].name!=fatMaterialName)
        return fail("replacement fat material program name mismatch: world="+
            world.materials[fatMaterialIndex].name+" requested="+std::string(fatMaterialName));
    if(!std::isfinite(oldSkin.massDebitKg)||oldSkin.massDebitKg<=0.0||
       !std::isfinite(oldFat.massDebitKg)||oldFat.massDebitKg<=0.0)
        return fail("replacement old prepared skin/fat debits must be finite and positive: skin="+
            std::to_string(oldSkin.massDebitKg)+" fat="+std::to_string(oldFat.massDebitKg));
    if(!std::isfinite(skinDonorDebitKg)||skinDonorDebitKg<=0.0||
       !std::isfinite(fatDonorDebitKg)||fatDonorDebitKg<=0.0)
        return fail("replacement requested skin/fat debits must be finite and positive: skin="+
            std::to_string(skinDonorDebitKg)+" fat="+std::to_string(fatDonorDebitKg));
    if(!closeMass(oldSkin.massDebitKg,skinDonorDebitKg))
        return fail("replacement skin donor debit mismatch: old="+
            std::to_string(oldSkin.massDebitKg)+" requested="+std::to_string(skinDonorDebitKg));
    if(!closeMass(oldFat.massDebitKg,fatDonorDebitKg))
        return fail("replacement fat donor debit mismatch: old="+
            std::to_string(oldFat.massDebitKg)+" requested="+std::to_string(fatDonorDebitKg));
    const std::string_view replacementSkinProvenance=oldSkin.materialProvenance.empty()
        ?skinMaterialProvenance:std::string_view(oldSkin.materialProvenance);
    const std::string_view replacementFatProvenance=oldFat.materialProvenance.empty()
        ?fatMaterialProvenance:std::string_view(oldFat.materialProvenance);

    std::map<std::uint32_t,double> oldCellMassByMaterial;
    if(!sourceMaterialMasses(oldObject,world,oldCellMassByMaterial,error))return false;
    if(!oldCellMassByMaterial.contains(skinMaterialIndex)||!oldCellMassByMaterial.contains(fatMaterialIndex)||
       !closeMass(oldCellMassByMaterial[skinMaterialIndex],oldSkin.massDebitKg)||
       !closeMass(oldCellMassByMaterial[fatMaterialIndex],oldFat.massDebitKg))
        return fail("replacement target cell masses do not close the existing per-material debit");

    double skinDensityBefore=0.0,fatDensityBefore=0.0;
    if(!densityDefault(world.materials[skinMaterialIndex],skinDensityBefore)||
       !densityDefault(world.materials[fatMaterialIndex],fatDensityBefore))
        return fail("replacement source material density defaults are not finite and positive");

    std::map<std::uint32_t,DonorMoment> oldSourceDonors,oldOwnerDonors;
    if(!sourceDonorMoments(oldSkin,oldFat,oldSourceDonors,error)||
       !ownerDonorMoments(oldSkin,oldFat,oldOwnerDonors,error))return false;
    if(oldSourceDonors.empty()||oldOwnerDonors.empty())
        return fail("replacement target has no source-donor or node-frame mass ledger");
    const bool oldHasExplicitSourceLedger=!oldSkin.donorMassContributions.empty()||
        !oldFat.donorMassContributions.empty();

    numi::matter::WorldSource scratch=world;
    scratch.objects.resize(oldObjectIndex);
    PreparedOrganicFEM candidateSkin,candidateFat;
    ExplicitUnpairedVolumeReport candidateReport;
    std::string candidateError;
    if(!appendRegisteredSkinFatUnpairedVolume(scratch,surface,boundaryMap,topology,
        skinMaterialIndex,skinMaterialName,replacementSkinProvenance,skinDonorDebitKg,
        fatMaterialIndex,fatMaterialName,replacementFatProvenance,fatDonorDebitKg,
        sourceBodyCount,objectName,candidateSkin,candidateFat,candidateReport,candidateError))
        return fail("replacement candidate failed the registered volume builder: "+candidateError);
    if(scratch.objects.size()!=oldObjectIndex+1u)
        return fail("replacement builder did not emit exactly one final FEM object");
    auto& candidateObject=scratch.objects.back();
    if(candidateObject.representation!=numi::matter::Representation::fem||
       candidateObject.name!=oldObject.name||
       !selectedMaterialPair(candidateObject,skinMaterialIndex,fatMaterialIndex))
        return fail("replacement builder changed the final object name or skin/fat FEM material ownership");
    if(candidateObject.femReferenceSourceIdentity!=topology.referenceSourceIdentity||
       candidateObject.femMaterialSourceIdentity!=topology.materialSourceIdentity)
        return fail("replacement builder did not install the pinned new reference/material source identities");
    // This changes constitutive input semantics and belongs to the existing
    // owner policy, not to the generated topology. Retain it across replacement.
    candidateObject.femReferenceDisplacementGradient=oldObject.femReferenceDisplacementGradient;
    // Supplying explicit I/D sidecars must not silently change contact policy
    // for a byte-identical physical mesh. Changed geometry/topology retains
    // the independent-volume builder's canonical contact policy.
    bool exactPhysicalIdentity=candidateObject.femNodes==oldObject.femNodes&&
        candidateObject.femReferenceNodes==oldObject.femReferenceNodes&&
        candidateObject.femMaterialIndices==oldObject.femMaterialIndices&&
        candidateObject.tetrahedra.size()==oldObject.tetrahedra.size();
    if(exactPhysicalIdentity)
        for(std::size_t cell=0u;cell<oldObject.tetrahedra.size();++cell)
            if(candidateObject.tetrahedra[cell].nodes!=oldObject.tetrahedra[cell].nodes) {
                exactPhysicalIdentity=false;break;
            }
    if(exactPhysicalIdentity) {
        candidateObject.deformableContact=oldObject.deformableContact;
        candidateObject.deformableSelfContact=oldObject.deformableSelfContact;
    }
    if(candidateSkin.firstGlobalFEMNode!=globalNodeBase||
       candidateFat.firstGlobalFEMNode!=globalNodeBase)
        return fail("replacement candidate global node base mismatch: skin="+
            std::to_string(candidateSkin.firstGlobalFEMNode)+" fat="+
            std::to_string(candidateFat.firstGlobalFEMNode)+" expected="+
            std::to_string(globalNodeBase));
    if(candidateSkin.materialIndex!=skinMaterialIndex||candidateSkin.materialName!=world.materials[skinMaterialIndex].name)
        return fail("replacement candidate skin prepared material does not match the owning world program");
    if(candidateFat.materialIndex!=fatMaterialIndex||candidateFat.materialName!=world.materials[fatMaterialIndex].name)
        return fail("replacement candidate fat prepared material does not match the owning world program");
    if(candidateSkin.materialProvenance!=replacementSkinProvenance||
       candidateFat.materialProvenance!=replacementFatProvenance)
        return fail("replacement candidate did not preserve existing skin/fat material provenance");
    if(!closeMass(candidateSkin.massDebitKg,oldSkin.massDebitKg))
        return fail("replacement candidate skin debit mismatch: old="+
            std::to_string(oldSkin.massDebitKg)+" new="+std::to_string(candidateSkin.massDebitKg));
    if(!closeMass(candidateFat.massDebitKg,oldFat.massDebitKg))
        return fail("replacement candidate fat debit mismatch: old="+
            std::to_string(oldFat.massDebitKg)+" new="+std::to_string(candidateFat.massDebitKg));
    double skinDensityAfter=0.0,fatDensityAfter=0.0;
    const bool identityClone=topology.referenceSourceIdentity==oldObject.femReferenceSourceIdentity&&
        topology.materialSourceIdentity==oldObject.femMaterialSourceIdentity;
    const auto densityPreserved=[&](double before,double after){
        return before==after||(identityClone&&closeDensityForIdentityClone(before,after));
    };
    if(!densityDefault(scratch.materials[skinMaterialIndex],skinDensityAfter)||
       !densityDefault(scratch.materials[fatMaterialIndex],fatDensityAfter)||
       !densityPreserved(skinDensityBefore,skinDensityAfter)||
       !densityPreserved(fatDensityBefore,fatDensityAfter))
        return fail("replacement candidate would change an existing skin/fat material density default");

    std::map<std::uint32_t,double> newCellMassByMaterial;
    if(!sourceMaterialMasses(candidateObject,scratch,newCellMassByMaterial,error))return false;
    if(!newCellMassByMaterial.contains(skinMaterialIndex)||!newCellMassByMaterial.contains(fatMaterialIndex)||
       !closeMass(newCellMassByMaterial[skinMaterialIndex],oldCellMassByMaterial[skinMaterialIndex])||
       !closeMass(newCellMassByMaterial[fatMaterialIndex],oldCellMassByMaterial[fatMaterialIndex]))
        return fail("replacement candidate per-material cell masses differ from the existing source totals");

    std::map<std::uint32_t,DonorMoment> newSourceDonors,newOwnerDonors;
    if(!sourceDonorMoments(candidateSkin,candidateFat,newSourceDonors,error)||
       !ownerDonorMoments(candidateSkin,candidateFat,newOwnerDonors,error))return false;
    const bool newHasExplicitSourceLedger=!candidateSkin.donorMassContributions.empty()||
        !candidateFat.donorMassContributions.empty();
    if(oldHasExplicitSourceLedger&&!newHasExplicitSourceLedger)
        return fail("replacement candidate discarded the existing explicit source-donor ledger");
    if(!sameDonorMasses(oldSourceDonors,newSourceDonors))
        return fail("replacement candidate physical source-donor mass totals differ");
    const double oldSourceTotal=[&](){
        double total=0.0;for(const auto& [donor,row]:oldSourceDonors){(void)donor;total+=row.massKg;}return total;
    }();
    const double newSourceTotal=[&](){
        double total=0.0;for(const auto& [donor,row]:newSourceDonors){(void)donor;total+=row.massKg;}return total;
    }();
    if(!closeMass(oldSourceTotal,oldSkin.massDebitKg+oldFat.massDebitKg)||
       !closeMass(newSourceTotal,candidateSkin.massDebitKg+candidateFat.massDebitKg))
        return fail("replacement source-donor mass does not close the combined tissue debit");

    const bool inheritedSkinThickness=preserveThicknessMetadata(oldSkin,candidateSkin,surface);
    const bool inheritedFatThickness=preserveThicknessMetadata(oldFat,candidateFat,surface);
    RegisteredSkinFatReplacementAudit candidateAudit;
    candidateAudit.preservedExactIdentityContactPolicy=exactPhysicalIdentity;
    candidateAudit.inheritedSkinThicknessMetadata=inheritedSkinThickness;
    candidateAudit.inheritedFatThicknessMetadata=inheritedFatThickness;
    candidateAudit.replacedObjectIndex=static_cast<std::uint32_t>(oldObjectIndex);
    candidateAudit.firstGlobalFEMNode=static_cast<std::uint32_t>(globalNodeBase);
    candidateAudit.oldSkinDebitKg=oldSkin.massDebitKg;
    candidateAudit.newSkinDebitKg=candidateSkin.massDebitKg;
    candidateAudit.oldFatDebitKg=oldFat.massDebitKg;
    candidateAudit.newFatDebitKg=candidateFat.massDebitKg;
    candidateAudit.oldReferenceSourceIdentity=oldObject.femReferenceSourceIdentity;
    candidateAudit.newReferenceSourceIdentity=candidateObject.femReferenceSourceIdentity;
    candidateAudit.oldMaterialSourceIdentity=oldObject.femMaterialSourceIdentity;
    candidateAudit.newMaterialSourceIdentity=candidateObject.femMaterialSourceIdentity;
    if(oldReport) {
        if(!oldReport->sourceMassClosure.empty()) {
            if(!sameSourceClosures(oldReport->sourceMassClosure,candidateReport.sourceMassClosure))
                return fail("replacement candidate per-material source-donor totals differ from the prior report");
            candidateAudit.comparedSourceDonorClosure=true;
        }
        if(oldReport->usesPerChildMassLedger) {
            if(!candidateReport.usesPerChildMassLedger||
               !closeMass(oldReport->skinParentMassKg,candidateReport.skinParentMassKg)||
               !closeMass(oldReport->fatParentMassKg,candidateReport.fatParentMassKg)||
               !closeMass(oldReport->skinChildMassKg,candidateReport.skinChildMassKg)||
               !closeMass(oldReport->fatChildMassKg,candidateReport.fatChildMassKg))
                return fail("replacement candidate parent/child material mass totals differ from the prior report");
        }
    }
    candidateAudit.sourceDonorDeltas=momentDeltas(oldSourceDonors,newSourceDonors);
    candidateAudit.nodeFrameOwnerDeltas=momentDeltas(oldOwnerDonors,newOwnerDonors);
    for(const auto& delta:candidateAudit.sourceDonorDeltas)
        if(!finiteVector(delta.oldSourceFirstMomentKgM)||
           !finiteVector(delta.newSourceFirstMomentKgM)||
           !finiteVector(delta.sourceFirstMomentDeltaKgM))
            return fail("replacement source-donor local first moment is nonfinite");
    for(const auto& delta:candidateAudit.nodeFrameOwnerDeltas)
        if(!finiteVector(delta.oldSourceFirstMomentKgM)||
           !finiteVector(delta.newSourceFirstMomentKgM)||
           !finiteVector(delta.sourceFirstMomentDeltaKgM))
            return fail("replacement node-frame owner local first moment is nonfinite");

    // Commit only the final object and caller-visible prepared/report records.
    world.objects.back()=candidateObject;
    newSkin=std::move(candidateSkin);
    newFat=std::move(candidateFat);
    newReport=std::move(candidateReport);
    audit=std::move(candidateAudit);
    error.clear();
    return true;
}

namespace registered_skin_fat_replacement_detail {

// Legacy paired reports precede Float32 cooking. Compare a remeshed child to
// the actual physical parent ledger, after independently recooking and checking
// every imported parent row. Keep the raw-to-cooked discrepancy in the audit.
inline bool prepareVerifiedCookedParentBaseline(
        const numi::matter::WorldSource& world,const std::size_t objectIndex,
        const PhysicalSkinChildMassMappingBundle& bundle,
        const PreparedOrganicFEM& oldSkin,const PreparedOrganicFEM& oldFat,
        PreparedOrganicFEM& baselineSkin,PreparedOrganicFEM& baselineFat,
        CompiledSkinFatMassCaptureReport& report,
        std::vector<RegisteredSkinFatDonorMomentDelta>& rawToCooked,
        bool& used,std::string& error) {
    baselineSkin=oldSkin;baselineFat=oldFat;used=false;
    if(!oldSkin.donorMassContributions.empty()||!oldFat.donorMassContributions.empty())
        return true;
    const auto fail=[&](std::string message) {error=std::move(message);return false;};
    // Existing exact/source-ledger replacements need no cook reconciliation.
    // Only the legacy raw-to-physical discrepancy requires the independent
    // compiler proof below; the ordinary replacement gates remain mandatory.
    std::map<std::uint32_t,DonorMoment> existing,imported;
    if(!sourceDonorMoments(oldSkin,oldFat,existing,error))return false;
    for(const auto& row:bundle.parent.sourceMassContributions) {
        auto& moment=imported[row.donorBodyIndex];
        moment.massKg+=row.massKg;
        for(unsigned axis=0;axis<3u;++axis)
            moment.firstMomentKgM[axis]+=row.massKg*row.donorLocalPosition[axis];
    }
    if(sameDonorMasses(existing,imported))return true;
    if(objectIndex>=world.objects.size())
        return fail("cooked-parent baseline has no original source object");
    const auto& object=world.objects[objectIndex];
    const auto& parent=bundle.parent;
    const auto count=object.femNodes.size();
    if(count==0u||parent.referenceWorldPoints!=object.femReferenceNodes||
       parent.tetrahedra.size()!=object.tetrahedra.size()||
       parent.materialIndices!=object.femMaterialIndices)
        return fail("cooked-parent baseline reference/topology/material source does not match the imported parent");
    for(std::size_t cell=0;cell<object.tetrahedra.size();++cell)
        if(parent.tetrahedra[cell]!=object.tetrahedra[cell].nodes)
            return fail("cooked-parent baseline tetrahedron rows differ from the original source");

    // Scene constraints are not finalized at this preparation point. Cook the
    // unchanged physical object with the unchanged material programs; restore
    // and independently check its original global node namespace below.
    std::uint64_t globalBase=0u;
    for(std::size_t row=0u;row<objectIndex;++row)
        globalBase+=world.objects[row].femNodes.size();
    if(globalBase>std::numeric_limits<std::uint32_t>::max()||
       count>std::numeric_limits<std::uint32_t>::max()-globalBase||
       oldSkin.firstGlobalFEMNode!=globalBase)
        return fail("cooked-parent baseline source global node prefix is invalid");
    numi::matter::WorldSource massSource;
    massSource.materials=world.materials;
    massSource.objects.push_back(object);
    numi::matter::CompileOptions options;options.maximumRateExponent=0u;
    const auto compiled=numi::matter::compileWorld(massSource,options);
    if(!compiled.succeeded()) {
        std::string diagnostics;
        for(const auto& diagnostic:compiled.diagnostics) {
            if(diagnostics.size()>2048u)break;
            diagnostics+="; "+diagnostic.message;
        }
        return fail("cooked-parent baseline could not compile the unchanged source object"+diagnostics);
    }
    if(compiled.world.objects.size()!=1u)
        return fail("cooked-parent baseline compiler omitted the source object");
    const auto& descriptor=compiled.world.objects.front();
    const auto first=descriptor.stateOffset;
    if(parent.currentWorldPoints.size()!=count||parent.compiledLumpedNodeMassKg.size()!=count||
       parent.compiledGlobalNodeIndices.size()!=count||
       parent.nodeFrameOwnerBodyIndices.size()!=count||parent.nodeFrameOwnerLocalPoints.size()!=count||
       std::uint64_t(first)+count>compiled.world.fem.nodes.size())
        return fail("cooked-parent baseline native node packet is incomplete");
    std::map<std::uint32_t,NumiHumanTissueMassNode> byNode;
    for(const auto* prepared:{&oldSkin,&oldFat})for(const auto& row:prepared->donorMassNodes) {
        auto [found,inserted]=byNode.emplace(row.nodeIndex,row);
        if(!inserted) {
            if(found->second.donorBody!=row.donorBody||
               norm(sub(found->second.localPosition,row.localPosition))>1.0e-7)
                return fail("cooked-parent baseline source node has ambiguous donor ownership");
            found->second.massKg+=row.massKg;
        }
    }
    if(byNode.size()!=count)
        return fail("cooked-parent baseline prepared reports do not cover the original node domain");
    std::vector<NumiHumanTissueMassNode> preparedRows;
    preparedRows.reserve(count);
    for(const auto& [node,row]:byNode) {
        if(node<globalBase||std::uint64_t(node)-globalBase>=count)
            return fail("cooked-parent baseline prepared node is outside the original global domain");
        auto local=row;
        local.nodeIndex=static_cast<std::uint32_t>(std::uint64_t(node)-globalBase)+first;
        preparedRows.push_back(local);
    }
    for(std::size_t node=0;node<count;++node) {
        const auto& state=compiled.world.fem.nodes[first+node].positionAndMass;
        const V3 current{double(state.x),double(state.y),double(state.z)};
        if(parent.currentWorldPoints[node]!=current||
           parent.compiledLumpedNodeMassKg[node]!=double(state.w)||
           parent.compiledGlobalNodeIndices[node]!=globalBase+node)
            return fail("cooked-parent baseline compiled position/mass/global node differs from the captured parent");
    }
    std::vector<double> capturedMass;
    std::vector<ExplicitUnpairedVolumeSourceMassContribution> capturedSource;
    if(!captureCompiledSkinFatParentMassLedger(massSource,compiled.world,0u,
           parent.nodeFrameOwnerBodyIndices,parent.nodeFrameOwnerLocalPoints,preparedRows,
           capturedMass,capturedSource,report,error))
        return false;
    if(capturedMass!=parent.tetrahedronMassKg||
       capturedSource.size()!=parent.sourceMassContributions.size())
        return fail("cooked-parent baseline captured cell/source masses differ from the imported parent");
    for(std::size_t row=0;row<capturedSource.size();++row) {
        const auto& actual=capturedSource[row];
        const auto& expected=parent.sourceMassContributions[row];
        if(actual.parentNodeIndex!=expected.parentNodeIndex||
           actual.materialIndex!=expected.materialIndex||
           actual.donorBodyIndex!=expected.donorBodyIndex||
           actual.massKg!=expected.massKg||actual.donorLocalPosition!=expected.donorLocalPosition)
            return fail("cooked-parent baseline source row differs from the imported physical ledger");
    }
    std::map<std::uint32_t,DonorMoment> raw,cooked;
    if(!sourceDonorMoments(oldSkin,oldFat,raw,error))return false;
    baselineSkin.donorMassContributions.clear();
    baselineFat.donorMassContributions.clear();
    for(const auto& row:capturedSource)
        baselineSkin.donorMassContributions.push_back({
            .nodeIndex=static_cast<std::uint32_t>(globalBase)+row.parentNodeIndex,.donorBody=row.donorBodyIndex,
            .massKg=row.massKg,.localPosition=row.donorLocalPosition});
    if(!sourceDonorMoments(baselineSkin,baselineFat,cooked,error))return false;
    rawToCooked=momentDeltas(raw,cooked);
    used=true;error.clear();return true;
}

} // namespace registered_skin_fat_replacement_detail

// Bundle-aware transactional owner. The path digest binds the imported O/I/D
// surfaces, exact boundary and attachment rows, source domains, node ancestry,
// and cell labels. The existing replacement then replays the conservative
// transport receipt and performs all geometry, mass, donor, and thickness checks.
[[nodiscard]] inline bool replaceLastPreparedRegisteredSkinFatUnpairedVolume(
        numi::matter::WorldSource& world,
        const std::size_t oldObjectIndex,
        const PhysicalSkinChildMassMappingBundle& bundle,
        const std::uint32_t skinMaterialIndex,
        const std::string_view skinMaterialName,
        const std::string_view skinMaterialProvenance,
        const double skinDonorDebitKg,
        const std::uint32_t fatMaterialIndex,
        const std::string_view fatMaterialName,
        const std::string_view fatMaterialProvenance,
        const double fatDonorDebitKg,
        const std::size_t sourceBodyCount,
        const std::string_view objectName,
        const PreparedOrganicFEM& oldSkin,
        const PreparedOrganicFEM& oldFat,
        const ExplicitUnpairedVolumeReport* oldReport,
        PreparedOrganicFEM& newSkin,
        PreparedOrganicFEM& newFat,
        ExplicitUnpairedVolumeReport& newReport,
        RegisteredSkinFatReplacementAudit& audit,
        std::string& error) {
    if(!physical_skin_child_mass_mapping_detail::
           validatePreparedPhysicalSkinChildMassMappingPath(bundle,error))return false;
    PreparedOrganicFEM baselineSkin,baselineFat;
    CompiledSkinFatMassCaptureReport capture;
    std::vector<RegisteredSkinFatDonorMomentDelta> rawToCooked;
    bool usedCookedBaseline=false;
    if(!registered_skin_fat_replacement_detail::prepareVerifiedCookedParentBaseline(
           world,oldObjectIndex,bundle,oldSkin,oldFat,baselineSkin,baselineFat,
           capture,rawToCooked,usedCookedBaseline,error))return false;
    const auto topology=bundle.topologyView();
    if(!replaceLastRegisteredSkinFatUnpairedVolume(
        world,oldObjectIndex,bundle.outerSkin,bundle.boundaryMap,topology,
        skinMaterialIndex,skinMaterialName,skinMaterialProvenance,skinDonorDebitKg,
        fatMaterialIndex,fatMaterialName,fatMaterialProvenance,fatDonorDebitKg,
        sourceBodyCount,objectName,baselineSkin,baselineFat,oldReport,
        newSkin,newFat,newReport,audit,error))return false;
    audit.comparedVerifiedCookedParentBaseline=usedCookedBaseline;
    audit.maximumPreparedToCookedNodeMassErrorKg=capture.maximumPreparedDonorMassClosureErrorKg;
    audit.maximumPreparedToCookedNodeMassRelativeError=capture.maximumPreparedDonorMassClosureRelativeError;
    audit.sourceRawToCookedDeltas=std::move(rawToCooked);
    return true;
}

} // namespace metalrobo::human::coarse_tissue

