#pragma once

// Transactional file-level import for the existing combined TetGen,
// registered node-map, and facet-ancestry CSV assets. The caller supplies
// already pinned parent and registered-surface state; this helper does not
// admit a physical child volume or start a simulation.
#include "NumiHumanPhysicalSkinChildTetGen.hpp"
#include "NumiHumanPhysicalSkinChildRegisteredVolumeCSV.hpp"
#include "NumiHumanPhysicalSkinChildRegisteredFacetMaps.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>

namespace metalrobo::human::coarse_tissue {

struct PhysicalSkinRegisteredVolumeAssetFiles {
    std::filesystem::path tetgenNode;
    std::filesystem::path tetgenElements;
    std::filesystem::path childLabels;
    std::filesystem::path combinedNodeMapCSV; // existing plain CSV after one-time decompression
    std::filesystem::path facetAncestryCSV;   // existing quoted-ancestry CSV
    // Order: node, elements, labels, combined node map, facet ancestry.
    std::array<std::string,5> expectedSHA256{};
};

struct PhysicalSkinRegisteredVolumeFileImportReport {
    PhysicalSkinChildTetGenImportReport tetgen;
    PhysicalSkinCombinedNodeMapCSVReport nodeMap;
    PhysicalSkinRegisteredFacetCSVReport facetAncestry;
    PhysicalSkinRegisteredVolumeImportReport registeredVolume;
    PhysicalSkinRegisteredFacetMapReport facetMaps;
    std::array<std::string,5> inputSHA256{};
    bool physicalAdmission=false;
};

namespace physical_skin_registered_volume_integration_detail {
inline bool pinnedFile(const std::filesystem::path& path,const std::string& expected,
                       std::string& actual,std::string& error) {
    if(path.empty()||!physical_skin_child_mass_detail::validSHA256(expected)){
        error="registered volume import requires an explicit path and lowercase SHA-256 pin";
        return false;
    }
    if(!physical_skin_child_mass_detail::sha256File(path,actual,error))return false;
    if(actual!=expected){
        error="registered volume import file hash differs from its declared pin: "+path.string();
        return false;
    }
    return true;
}
}

[[nodiscard]] inline bool importPhysicalSkinRegisteredVolumeFiles(
        const PhysicalSkinRegisteredVolumeAssetFiles& files,
        const std::string_view coordinateFrameSHA256,
        const std::size_t sourceBodyCount,
        const double maximumMassTransportDistanceM,
        const double maximumParentRegistrationDistanceM,
        const std::string_view canonicalDetailedSourceDomain,
        const std::size_t detailedSourceVertexCount,
        const std::size_t detailedSourceFaceCount,
        const PhysicalSkinRegisteredSurfaceCSVSet& registered,
        const std::span<const PhysicalSkinStableDeepAttachmentRequest> deepAttachments,
        const std::span<const PhysicalSkinRegisteredBodyPose> allBodyCurrentPoses,
        PhysicalSkinChildMassMappingBundle& outputBundle,
        PhysicalSkinRegisteredVolumeImportSidecars& outputSidecars,
        PhysicalSkinRegisteredVolumeFileImportReport& outputReport,
        std::string& error,
        const std::span<const PhysicalSkinSourceDomainRegistration> sourceDomains={}) {
    using namespace physical_skin_registered_volume_integration_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(files.combinedNodeMapCSV.extension()==".gz"||files.facetAncestryCSV.extension()==".gz")
        return fail("registered volume import accepts plain existing CSV files after one-time decompression");
    if(sourceBodyCount==0u||!std::isfinite(maximumParentRegistrationDistanceM)||
       maximumParentRegistrationDistanceM<=0.0)
        return fail("registered volume import has an invalid body domain or parent registration bound");

    // All parsing and compilation happens in local state. No caller output is
    // changed unless TetGen, node maps, O/I/D import, and facet maps all pass.
    PhysicalSkinChildMassMappingBundle nextBundle=outputBundle;
    PhysicalSkinChildTetGenImportReport tetgenReport;
    if(!loadPhysicalSkinChildTetGenReference(files.tetgenNode,files.tetgenElements,files.childLabels,
            coordinateFrameSHA256,sourceBodyCount,maximumMassTransportDistanceM,
            nextBundle,tetgenReport,error))
        return false;
    const std::array<std::string,3> tetgenHashes{
        tetgenReport.nodeFileSHA256,tetgenReport.elementFileSHA256,tetgenReport.labelFileSHA256};
    for(unsigned i=0;i<3u;++i)
        if(!physical_skin_child_mass_detail::validSHA256(tetgenHashes[i])||tetgenHashes[i]!=files.expectedSHA256[i])
            return fail("registered volume import TetGen file hash differs from its declared pin: "+
                std::array<std::filesystem::path,3>{files.tetgenNode,files.tetgenElements,files.childLabels}[i].string());

    std::string mapHashBefore,mapHashAfter;
    if(!pinnedFile(files.combinedNodeMapCSV,files.expectedSHA256[3],mapHashBefore,error))return false;
    std::ifstream nodeMapInput(files.combinedNodeMapCSV);
    if(!nodeMapInput)return fail("cannot open plain combined TetGen node-map CSV");
    PhysicalSkinCombinedNodeMapCSVReport nodeMapReport;
    std::vector<PhysicalSkinRegisteredVolumeNodeRow> nodeRows;
    if(!readPhysicalSkinCombinedNodeMapCSV(nodeMapInput,nextBundle.referenceWorldPoints.size(),
            sourceBodyCount,nodeRows,nodeMapReport,error))
        return false;
    if(!pinnedFile(files.combinedNodeMapCSV,files.expectedSHA256[3],mapHashAfter,error)||
       mapHashAfter!=mapHashBefore)
        return fail("combined TetGen node-map CSV changed while it was being imported");

    std::string facetHashBefore,facetHashAfter;
    if(!pinnedFile(files.facetAncestryCSV,files.expectedSHA256[4],facetHashBefore,error))return false;
    std::ifstream facetInput(files.facetAncestryCSV);
    if(!facetInput)return fail("cannot open existing registered facet-ancestry CSV");
    PhysicalSkinRegisteredFacetCSVReport facetCSVReport;
    std::vector<PhysicalSkinAuthoredHeadWallRow> authoredWalls;
    if(!readPhysicalSkinAuthoredHeadWallsCSV(facetInput,nextBundle.childTetrahedra.size(),
            nextBundle.referenceWorldPoints.size(),authoredWalls,facetCSVReport,error))
        return false;
    if(!pinnedFile(files.facetAncestryCSV,files.expectedSHA256[4],facetHashAfter,error)||
       facetHashAfter!=facetHashBefore)
        return fail("facet-ancestry CSV changed while it was being imported");
    const auto& surfaces=registered.surfaces;
    const auto expectedRegisteredFacetRows=surfaces[0].faces.size()+
        2u*surfaces[1].faces.size()+surfaces[2].faces.size();
    if(facetCSVReport.registeredRows!=expectedRegisteredFacetRows)
        return fail("facet-ancestry CSV registered O/I/D row count differs from exact registered surfaces");

    PhysicalSkinRegisteredVolumeImportSidecars nextSidecars;
    PhysicalSkinRegisteredVolumeImportReport registeredReport;
    if(!compilePhysicalSkinRegisteredVolumeImport(nextBundle,registered,nodeRows,deepAttachments,
            allBodyCurrentPoses,sourceBodyCount,maximumParentRegistrationDistanceM,
            canonicalDetailedSourceDomain,detailedSourceVertexCount,detailedSourceFaceCount,
            nextSidecars,registeredReport,error,sourceDomains))
        return false;
    PhysicalSkinRegisteredFacetMapReport facetMapReport;
    if(!preparePhysicalSkinRegisteredFacetMaps(nextBundle,authoredWalls,15u,16u,facetMapReport,error))
        return false;

    PhysicalSkinRegisteredVolumeFileImportReport measured;
    measured.tetgen=std::move(tetgenReport);measured.nodeMap=nodeMapReport;
    measured.facetAncestry=facetCSVReport;measured.registeredVolume=registeredReport;
    measured.facetMaps=facetMapReport;
    measured.inputSHA256={measured.tetgen.nodeFileSHA256,measured.tetgen.elementFileSHA256,
        measured.tetgen.labelFileSHA256,mapHashAfter,facetHashAfter};
    measured.physicalAdmission=false;
    outputBundle=std::move(nextBundle);outputSidecars=std::move(nextSidecars);
    outputReport=std::move(measured);error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
