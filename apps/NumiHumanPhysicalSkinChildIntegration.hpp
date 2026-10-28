#pragma once

// Strict app-side bridge for an externally authored TetGen child that is an
// exact identity re-encoding of the currently registered skin/fat parent.
// Changed topology needs its own explicit O/I/D, node-frame, and ancestry maps.
#include "NumiHumanPhysicalSkinChildTetGen.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct PhysicalSkinChildIdentityImportFiles {
    std::filesystem::path parentMassCSV;
    std::filesystem::path parentLabelsCSV;
    std::filesystem::path childNodeFile;
    std::filesystem::path childElementFile;
    std::filesystem::path childLabelsCSV;
};

struct PhysicalSkinChildIdentityImportReport {
    PhysicalSkinChildTetGenImportReport tetgen;
    std::string coordinateFrameSHA256;
    std::string importedTetGenMeshSHA256;
    std::string exactLiveChildMeshSHA256;
    std::size_t capturedParentCurrentExactF32Rows=0u;
    std::size_t capturedParentReferenceExactF64Rows=0u;
};

namespace physical_skin_child_integration_detail {

inline bool validIdentityFiles(const PhysicalSkinChildIdentityImportFiles& files,
                               std::string& error) {
    if (files.parentMassCSV.empty() || files.parentLabelsCSV.empty() ||
        files.childNodeFile.empty() || files.childElementFile.empty() ||
        files.childLabelsCSV.empty()) {
        error = "identity child import requires explicit parent mass/label and child node/element/label paths";
        return false;
    }
    return true;
}

inline bool makeIdentityLayerSurface(const RegisteredSurface& outer,
                                     const std::span<const V3> layerWorld,
                                     const std::span<const V3> layerLocal,
                                     const std::string_view role,
                                     RegisteredSurface& output,
                                     std::string& error) {
    if (outer.worldPoints.empty() || layerWorld.size() != outer.worldPoints.size() ||
        layerLocal.size() != outer.worldPoints.size() || outer.faces.empty()) {
        error = "identity child layer surface does not match the registered outer surface rows";
        return false;
    }
    RegisteredSurface next = outer;
    next.name += " " + std::string(role) + " identity boundary";
    next.worldPoints.assign(layerWorld.begin(), layerWorld.end());
    next.donorLocalPoints.assign(layerLocal.begin(), layerLocal.end());
    if (!outer.sourceEmbeddingResidualWorldM.empty()) {
        if (outer.sourceEmbeddingResidualWorldM.size() != outer.worldPoints.size()) {
            error = "identity child source embedding residual rows are incomplete";
            return false;
        }
        for (std::size_t row = 0; row < next.sourceEmbeddingResidualWorldM.size(); ++row)
            next.sourceEmbeddingResidualWorldM[row] = add(outer.sourceEmbeddingResidualWorldM[row],
                sub(next.worldPoints[row], outer.worldPoints[row]));
    }
    if (!outwardVertexNormals(next.worldPoints, next.faces, next.outwardWorldNormals, error))
        return false;
    output = std::move(next);
    error.clear();
    return true;
}

template<class T>
inline void copyRows(std::vector<T>& output, const std::span<const T> rows) {
    output.assign(rows.begin(), rows.end());
}

inline bool initializeIdentitySidecars(
        PhysicalSkinChildMassMappingBundle& bundle,
        const RegisteredSurface& outer,
        const RegisteredSurface& inner,
        const RegisteredSurface& deep,
        const numi::human::SkinBoundaryNodeMap& boundaryMap,
        const ExplicitUnpairedVolumeTopology& topology,
        const double maximumTransportDistanceM,
        std::string& error) {
    const auto fail = [&](std::string message) { error = std::move(message); return false; };
    if (!bundle.currentWorldPoints.empty() || !bundle.referenceWorldPoints.empty() ||
        !bundle.childTetrahedra.empty() || !bundle.childMaterialIndices.empty())
        return fail("identity sidecar initialization requires an empty child-table bundle");
    const std::size_t nodeCount = topology.currentWorldPoints.size();
    const std::size_t tetCount = topology.childTetrahedra.size();
    if (nodeCount == 0u || tetCount == 0u || topology.referenceWorldPoints.size() != nodeCount ||
        topology.childMaterialIndices.size() != tetCount || topology.childDonorBodyIndices.size() != tetCount ||
        topology.childParentSurfaceFaces.size() != tetCount || topology.childStableSourceFaceIdentities.size() != tetCount ||
        topology.childRegisteredLayerFacetSurfaceFaces.size() != 4u * tetCount ||
        topology.childRegisteredLayerFacetSourceFaceIdentities.size() != 4u * tetCount ||
        topology.childAuthoredHeadSidewallFacetFlags.size() != 4u * tetCount ||
        topology.childAuthoredHeadSidewallStableFaceIdentities.size() != 4u * tetCount ||
        topology.volumeNodeDonorBodyIndices.size() != nodeCount ||
        topology.volumeNodeDonorLocalPoints.size() != nodeCount ||
        topology.volumeNodeDonorLocalToWorldLinear.size() != nodeCount ||
        topology.volumeNodeDonorOriginWorld.size() != nodeCount ||
        topology.registeredBodyLocalToWorldLinear.empty() ||
        topology.registeredBodyOriginWorld.size() != topology.registeredBodyLocalToWorldLinear.size() ||
        topology.volumeNodeSourceVertexIndices.size() != nodeCount ||
        topology.volumeNodeSourceFaceIndices.size() != nodeCount ||
        topology.volumeNodeSourceBarycentric.size() != nodeCount ||
        topology.volumeNodeParentTetrahedra.size() != nodeCount ||
        topology.volumeNodeParentBarycentric.size() != nodeCount ||
        topology.volumeNodeParentResidualWorldM.size() != nodeCount ||
        topology.retainedOriginalNodeCount != nodeCount ||
        boundaryMap.volumeNodeCount != nodeCount || outer.worldPoints.empty() ||
        inner.worldPoints.size() != outer.worldPoints.size() || deep.worldPoints.size() != outer.worldPoints.size() ||
        inner.faces != outer.faces || deep.faces != outer.faces ||
        !std::isfinite(maximumTransportDistanceM) || maximumTransportDistanceM <= 0.0)
        return fail("identity sidecar topology or O/I/D surface arrays are incomplete or inconsistent");
    if (topology.referenceSourceIdentity == std::array<std::uint64_t, 4>{} ||
        topology.materialSourceIdentity == std::array<std::uint64_t, 4>{})
        return fail("identity sidecar source/material identities are not pinned");

    bundle.outerSkin = outer;
    bundle.independentSkinInner = inner;
    bundle.independentFatDeep = deep;
    bundle.boundaryMap = boundaryMap;
    copyRows(bundle.childParentSurfaceFaces, topology.childParentSurfaceFaces);
    copyRows(bundle.childRegisteredLayerFacetSurfaceFaces, topology.childRegisteredLayerFacetSurfaceFaces);
    copyRows(bundle.childRegisteredLayerFacetSourceFaceIdentities,
             topology.childRegisteredLayerFacetSourceFaceIdentities);
    copyRows(bundle.childAuthoredHeadSidewallFacetFlags, topology.childAuthoredHeadSidewallFacetFlags);
    copyRows(bundle.childAuthoredHeadSidewallStableFaceIdentities,
             topology.childAuthoredHeadSidewallStableFaceIdentities);
    copyRows(bundle.volumeNodeDonorBodyIndices, topology.volumeNodeDonorBodyIndices);
    copyRows(bundle.volumeNodeDonorLocalPoints, topology.volumeNodeDonorLocalPoints);
    copyRows(bundle.volumeNodeDonorLocalToWorldLinear, topology.volumeNodeDonorLocalToWorldLinear);
    copyRows(bundle.volumeNodeDonorOriginWorld, topology.volumeNodeDonorOriginWorld);
    copyRows(bundle.registeredBodyLocalToWorldLinear, topology.registeredBodyLocalToWorldLinear);
    copyRows(bundle.registeredBodyOriginWorld, topology.registeredBodyOriginWorld);
    bundle.detailedSourceVertexCount = topology.detailedSourceVertexCount;
    bundle.detailedSourceFaceCount = topology.detailedSourceFaceCount;
    copyRows(bundle.volumeNodeSourceVertexIndices, topology.volumeNodeSourceVertexIndices);
    copyRows(bundle.volumeNodeSourceFaceIndices, topology.volumeNodeSourceFaceIndices);
    copyRows(bundle.volumeNodeSourceBarycentric, topology.volumeNodeSourceBarycentric);
    bundle.retainedOriginalNodeCount = topology.retainedOriginalNodeCount;
    copyRows(bundle.volumeNodeParentTetrahedra, topology.volumeNodeParentTetrahedra);
    copyRows(bundle.volumeNodeParentBarycentric, topology.volumeNodeParentBarycentric);
    copyRows(bundle.volumeNodeParentResidualWorldM, topology.volumeNodeParentResidualWorldM);
    bundle.referenceSourceIdentity = topology.referenceSourceIdentity;
    bundle.materialSourceIdentity = topology.materialSourceIdentity;
    bundle.maximumSpatialMassTransportDistanceM = maximumTransportDistanceM;
    error.clear();
    return true;
}

inline std::string identityCoordinateFrameSHA256(
        const PhysicalSkinParentMassPacket& parent,
        const ExplicitUnpairedVolumeTopology& expected) {
    const std::size_t nodeCount = expected.currentWorldPoints.size();
    if (nodeCount == 0u || expected.referenceWorldPoints.size() != nodeCount ||
        parent.currentWorldPoints.size() != nodeCount || parent.referenceWorldPoints.size() != nodeCount ||
        expected.volumeNodeDonorBodyIndices.size() != nodeCount ||
        expected.volumeNodeDonorLocalPoints.size() != nodeCount ||
        expected.volumeNodeDonorLocalToWorldLinear.size() != nodeCount ||
        expected.volumeNodeDonorOriginWorld.size() != nodeCount ||
        expected.registeredBodyLocalToWorldLinear.size() != expected.registeredBodyOriginWorld.size())
        return {};
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("live-native-identity-child.coordinate-frame.v2");
    hash.string(parent.parentMassPacketSHA256);
    hash.string(parent.parentSourceContributionSHA256);
    hash.string(parent.parentTopologySHA256);
    hash.string(parent.parentLabelSHA256);
    hash.u64(nodeCount);
    for (std::size_t node = 0; node < nodeCount; ++node) {
        hash.u32(parent.compiledGlobalNodeIndices[node]);
        hash.u32(parent.nodeFrameOwnerBodyIndices[node]);
        hash.u32(expected.volumeNodeDonorBodyIndices[node]);
        for (const auto value : expected.currentWorldPoints[node]) hash.real(value);
        for (const auto value : expected.referenceWorldPoints[node]) hash.real(value);
        for (const auto value : parent.currentWorldPoints[node]) hash.real(value);
        for (const auto value : parent.referenceWorldPoints[node]) hash.real(value);
        for (const auto value : parent.nodeFrameOwnerLocalPoints[node]) hash.real(value);
        for (const auto value : expected.volumeNodeDonorLocalPoints[node]) hash.real(value);
        for (const auto value : expected.volumeNodeDonorOriginWorld[node]) hash.real(value);
        for (const auto value : expected.volumeNodeDonorLocalToWorldLinear[node]) hash.real(value);
        if (!expected.volumeNodeSourceVertexIndices.empty())
            hash.u32(expected.volumeNodeSourceVertexIndices[node]);
        if (!expected.volumeNodeSourceFaceIndices.empty())
            hash.u32(expected.volumeNodeSourceFaceIndices[node]);
        if (!expected.volumeNodeSourceBarycentric.empty())
            for (const auto value : expected.volumeNodeSourceBarycentric[node]) hash.real(value);
        if (!expected.volumeNodeParentTetrahedra.empty())
            hash.u32(expected.volumeNodeParentTetrahedra[node]);
        if (!expected.volumeNodeParentBarycentric.empty())
            for (const auto value : expected.volumeNodeParentBarycentric[node]) hash.real(value);
        if (!expected.volumeNodeParentResidualWorldM.empty())
            for (const auto value : expected.volumeNodeParentResidualWorldM[node]) hash.real(value);
    }
    hash.u64(expected.registeredBodyLocalToWorldLinear.size());
    for (std::size_t body = 0; body < expected.registeredBodyLocalToWorldLinear.size(); ++body) {
        for (const auto value : expected.registeredBodyLocalToWorldLinear[body]) hash.real(value);
        for (const auto value : expected.registeredBodyOriginWorld[body]) hash.real(value);
    }
    return hash.finish();
}

inline std::string exactLiveChildMeshSHA256(
        const PhysicalSkinChildMassMappingBundle& bundle,
        const std::string_view importedTetGenMeshSHA256) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("live-native-identity-child.mesh.v2");
    hash.string(importedTetGenMeshSHA256);
    hash.string(bundle.regionLabelSHA256);
    hash.string(bundle.coordinateFrameSHA256);
    hash.u64(bundle.referenceSourceIdentity.size());
    for (const auto value : bundle.referenceSourceIdentity) hash.u64(value);
    hash.u64(bundle.materialSourceIdentity.size());
    for (const auto value : bundle.materialSourceIdentity) hash.u64(value);
    const auto surface = [&](const RegisteredSurface& value) {
        hash.string(value.name);hash.string(value.provider);hash.string(value.sourceMember);
        hash.string(value.sourceMeshSHA256);hash.string(value.sourcePayloadSHA256);
        hash.u32(value.stableIdentifier);hash.u32(value.sourceBodyIndex);hash.u32(value.sourceLayer);
        hash.u32(value.aggregateIdentity?1u:0u);
        hash.u64(value.worldPoints.size());
        for (std::size_t row=0;row<value.worldPoints.size();++row) {
            for (const auto coordinate : value.worldPoints[row]) hash.real(coordinate);
            if (row<value.donorLocalPoints.size())
                for (const auto coordinate : value.donorLocalPoints[row]) hash.real(coordinate);
            if (row<value.donorBodyIndices.size()) hash.u32(value.donorBodyIndices[row]);
            if (row<value.sourceVertexIndices.size()) hash.u32(value.sourceVertexIndices[row]);
            if (row<value.sourceEmbeddingFaceIndices.size()) {
                hash.u32(value.sourceEmbeddingFaceIndices[row]);
                for (const auto coordinate : value.sourceEmbeddingBarycentric[row]) hash.real(coordinate);
            }
            if (row<value.sourceEmbeddingResidualWorldM.size())
                for (const auto coordinate : value.sourceEmbeddingResidualWorldM[row]) hash.real(coordinate);
            if (row<value.worldToDonorLocalLinear.size())
                for (const auto coefficient : value.worldToDonorLocalLinear[row]) hash.real(coefficient);
            if (row<value.outwardWorldNormals.size())
                for (const auto coordinate : value.outwardWorldNormals[row]) hash.real(coordinate);
        }
        hash.u64(value.faces.size());
        for (std::size_t row=0;row<value.faces.size();++row) {
            for (const auto node : value.faces[row]) hash.u32(node);
            if (row<value.sourceFaceIndices.size()) hash.u32(value.sourceFaceIndices[row]);
        }
    };
    surface(bundle.outerSkin);surface(bundle.independentSkinInner);surface(bundle.independentFatDeep);
    hash.u64(bundle.boundaryMap.volumeNodeCount);
    for (const auto node : bundle.boundaryMap.outer) hash.u32(node);
    for (const auto node : bundle.boundaryMap.skinInner) hash.u32(node);
    for (const auto node : bundle.boundaryMap.fatDeep) hash.u32(node);
    hash.u64(bundle.currentWorldPoints.size());
    for (const auto& point : bundle.currentWorldPoints)
        for (const auto value : point) hash.real(value);
    hash.u64(bundle.referenceWorldPoints.size());
    for (const auto& point : bundle.referenceWorldPoints)
        for (const auto value : point) hash.real(value);
    hash.u64(bundle.childTetrahedra.size());
    for (std::size_t cell = 0; cell < bundle.childTetrahedra.size(); ++cell) {
        for (const auto node : bundle.childTetrahedra[cell]) hash.u32(node);
        hash.u32(bundle.childMaterialIndices[cell]);
        hash.u32(bundle.childDonorBodyIndices[cell]);
        hash.u32(bundle.childAnatomicalRegionIndices[cell]);
        hash.u32(bundle.childStableSourceFaceIdentities[cell]);
        hash.u32(bundle.childParentSurfaceFaces[cell]);
    }
    for (const auto value : bundle.childRegisteredLayerFacetSurfaceFaces) hash.u32(value);
    for (const auto value : bundle.childRegisteredLayerFacetSourceFaceIdentities) hash.u32(value);
    for (const auto value : bundle.childAuthoredHeadSidewallFacetFlags) hash.u32(value);
    for (const auto value : bundle.childAuthoredHeadSidewallStableFaceIdentities) hash.u32(value);
    hash.u64(bundle.volumeNodeDonorBodyIndices.size());
    for (std::size_t node=0;node<bundle.volumeNodeDonorBodyIndices.size();++node) {
        hash.u32(bundle.volumeNodeDonorBodyIndices[node]);
        for (const auto value : bundle.volumeNodeDonorLocalPoints[node]) hash.real(value);
        for (const auto value : bundle.volumeNodeDonorLocalToWorldLinear[node]) hash.real(value);
        for (const auto value : bundle.volumeNodeDonorOriginWorld[node]) hash.real(value);
        hash.u32(bundle.volumeNodeSourceVertexIndices[node]);
        hash.u32(bundle.volumeNodeSourceFaceIndices[node]);
        for (const auto value : bundle.volumeNodeSourceBarycentric[node]) hash.real(value);
        hash.u32(bundle.volumeNodeParentTetrahedra[node]);
        for (const auto value : bundle.volumeNodeParentBarycentric[node]) hash.real(value);
        for (const auto value : bundle.volumeNodeParentResidualWorldM[node]) hash.real(value);
    }
    hash.u64(bundle.registeredBodyLocalToWorldLinear.size());
    for (std::size_t body=0;body<bundle.registeredBodyLocalToWorldLinear.size();++body) {
        for (const auto value : bundle.registeredBodyLocalToWorldLinear[body]) hash.real(value);
        for (const auto value : bundle.registeredBodyOriginWorld[body]) hash.real(value);
    }
    hash.u64(bundle.detailedSourceVertexCount);hash.u64(bundle.detailedSourceFaceCount);
    hash.u64(bundle.retainedOriginalNodeCount);
    return hash.finish();
}

} // namespace physical_skin_child_integration_detail

[[nodiscard]] inline bool loadPhysicalSkinChildIdentityImport(
        const PhysicalSkinChildIdentityImportFiles& files,
        const std::size_t sourceBodyCount,
        const ExplicitUnpairedVolumeTopology& expectedLiveParent,
        PhysicalSkinChildMassMappingBundle& bundle,
        PhysicalSkinChildIdentityImportReport& report,
        std::string& error,
        const double maximumTransportDistanceM = 1.0e-6) {
    using namespace physical_skin_child_integration_detail;
    if (!validIdentityFiles(files, error)) return false;
    if (sourceBodyCount == 0u || !std::isfinite(maximumTransportDistanceM) ||
        maximumTransportDistanceM <= 0.0) {
        error = "identity child import source-body domain or transport bound is invalid";
        return false;
    }
    if (!loadCompiledSkinFatParentMassPacketCSV(files.parentMassCSV, bundle.parent, error) ||
        !loadCompiledSkinFatParentLabelsCSV(files.parentLabelsCSV, bundle.parent, error))
        return false;
    PhysicalSkinChildIdentityImportReport measured;
    measured.coordinateFrameSHA256 = identityCoordinateFrameSHA256(bundle.parent, expectedLiveParent);
    if (measured.coordinateFrameSHA256.empty()) {
        error = "live identity child coordinate frame is incomplete";
        return false;
    }
    if (!loadPhysicalSkinChildTetGenReference(files.childNodeFile, files.childElementFile,
            files.childLabelsCSV, measured.coordinateFrameSHA256, sourceBodyCount,
            maximumTransportDistanceM, bundle, measured.tetgen, error))
        return false;
    measured.importedTetGenMeshSHA256 = bundle.childMeshSHA256;
    report = std::move(measured);
    error.clear();
    return true;
}

[[nodiscard]] inline bool validatePhysicalSkinChildIdentityImport(
        PhysicalSkinChildMassMappingBundle& bundle,
        const ExplicitUnpairedVolumeTopology& expected,
        PhysicalSkinChildIdentityImportReport& report,
        std::string& error) {
    const auto fail = [&](std::string message) { error = std::move(message); return false; };
    const auto sameRows = [](const auto& actual, const auto expectedRows) {
        return actual.size() == expectedRows.size() &&
            std::equal(actual.begin(), actual.end(), expectedRows.begin());
    };
    const std::size_t nodeCount=bundle.referenceWorldPoints.size();
    const std::size_t cellCount=bundle.childTetrahedra.size();
    if (nodeCount==0u||cellCount==0u||
        bundle.childParentSurfaceFaces.size()!=cellCount||
        bundle.childRegisteredLayerFacetSurfaceFaces.size()!=4u*cellCount||
        bundle.childRegisteredLayerFacetSourceFaceIdentities.size()!=4u*cellCount||
        bundle.childAuthoredHeadSidewallFacetFlags.size()!=4u*cellCount||
        bundle.childAuthoredHeadSidewallStableFaceIdentities.size()!=4u*cellCount||
        bundle.volumeNodeDonorBodyIndices.size()!=nodeCount||
        bundle.volumeNodeDonorLocalPoints.size()!=nodeCount||
        bundle.volumeNodeDonorLocalToWorldLinear.size()!=nodeCount||
        bundle.volumeNodeDonorOriginWorld.size()!=nodeCount||
        bundle.volumeNodeSourceVertexIndices.size()!=nodeCount||
        bundle.volumeNodeSourceFaceIndices.size()!=nodeCount||
        bundle.volumeNodeSourceBarycentric.size()!=nodeCount||
        bundle.volumeNodeParentTetrahedra.size()!=nodeCount||
        bundle.volumeNodeParentBarycentric.size()!=nodeCount||
        bundle.volumeNodeParentResidualWorldM.size()!=nodeCount||
        bundle.registeredBodyLocalToWorldLinear.size()!=bundle.registeredBodyOriginWorld.size())
        return fail("identity import is missing exact node/cell ancestry, boundary-facet, or donor-frame sidecars");
    if (bundle.currentWorldPoints.size() != expected.currentWorldPoints.size() ||
        bundle.referenceWorldPoints.size() != expected.referenceWorldPoints.size() ||
        bundle.parent.currentWorldPoints.size() != expected.currentWorldPoints.size() ||
        bundle.parent.referenceWorldPoints.size() != expected.referenceWorldPoints.size())
        return fail("imported child node rows do not cover the live identity parent");
    for (std::size_t node = 0; node < expected.referenceWorldPoints.size(); ++node) {
        const auto& reference = expected.referenceWorldPoints[node];
        const V3 emitted{static_cast<double>(static_cast<float>(reference[0])),
                         static_cast<double>(static_cast<float>(reference[1])),
                         static_cast<double>(static_cast<float>(reference[2]))};
        if (bundle.referenceWorldPoints[node] != emitted ||
            bundle.currentWorldPoints[node] != emitted)
            return fail("imported TetGen node row is not the IEEE-F32 emission of live reference row " +
                std::to_string(node));
        const auto pointText = [](const V3& point) {
            std::ostringstream text;
            text << std::setprecision(17) << '[' << point[0] << ',' << point[1] << ',' << point[2] << ']';
            return text.str();
        };
        const auto f32Point = [](const V3& point) {
            return V3{static_cast<double>(static_cast<float>(point[0])),
                      static_cast<double>(static_cast<float>(point[1])),
                      static_cast<double>(static_cast<float>(point[2]))};
        };
        const auto emittedCurrent = f32Point(expected.currentWorldPoints[node]);
        if (bundle.parent.currentWorldPoints[node] != emittedCurrent) {
            return fail("captured parent current row is not the exact IEEE-F32 emission of live row " +
                std::to_string(node) + ": captured=" + pointText(bundle.parent.currentWorldPoints[node]) +
                " live_f64=" + pointText(expected.currentWorldPoints[node]) +
                " live_f32=" + pointText(emittedCurrent));
        }
        ++report.capturedParentCurrentExactF32Rows;
        if (bundle.parent.referenceWorldPoints[node] != expected.referenceWorldPoints[node]) {
            const auto emittedReference = f32Point(expected.referenceWorldPoints[node]);
            return fail("captured parent reference row differs from exact live reference row " +
                std::to_string(node) + ": captured=" + pointText(bundle.parent.referenceWorldPoints[node]) +
                " live_f64=" + pointText(expected.referenceWorldPoints[node]) +
                " live_f32=" + pointText(emittedReference));
        }
        ++report.capturedParentReferenceExactF64Rows;
    }
    if (!sameRows(bundle.childTetrahedra, expected.childTetrahedra) ||
        !sameRows(bundle.childMaterialIndices, expected.childMaterialIndices) ||
        !sameRows(bundle.childDonorBodyIndices, expected.childDonorBodyIndices) ||
        !sameRows(bundle.childParentSurfaceFaces, expected.childParentSurfaceFaces) ||
        !sameRows(bundle.childStableSourceFaceIdentities, expected.childStableSourceFaceIdentities))
        return fail("imported child cells or explicit labels differ from the live identity parent rows");
    if (!sameRows(bundle.childTetrahedra, bundle.parent.tetrahedra) ||
        !sameRows(bundle.childMaterialIndices, bundle.parent.materialIndices) ||
        !sameRows(bundle.childDonorBodyIndices, bundle.parent.donorBodyIndices) ||
        !sameRows(bundle.childAnatomicalRegionIndices, bundle.parent.anatomicalRegionIndices) ||
        !sameRows(bundle.childStableSourceFaceIdentities, bundle.parent.stableSourceFaceIdentities))
        return fail("imported child identity rows do not exactly match the captured parent topology and labels");
    // The file is a Float32 import representation. Downstream compilation owns
    // the exact live doubles; never replace the parent reference/current state
    // with the rounded importer coordinates.
    bundle.currentWorldPoints.assign(expected.currentWorldPoints.begin(),expected.currentWorldPoints.end());
    bundle.referenceWorldPoints.assign(expected.referenceWorldPoints.begin(),expected.referenceWorldPoints.end());
    report.exactLiveChildMeshSHA256 = physical_skin_child_integration_detail::
        exactLiveChildMeshSHA256(bundle,report.importedTetGenMeshSHA256);
    if (!physical_skin_child_mass_detail::validSHA256(report.exactLiveChildMeshSHA256))
        return fail("exact live identity child mesh fingerprint could not be produced");
    bundle.childMeshSHA256 = report.exactLiveChildMeshSHA256;
    bundle.coordinateFrameSHA256 = physical_skin_child_integration_detail::
        identityCoordinateFrameSHA256(bundle.parent,expected);
    error.clear();
    return true;
}

} // namespace metalrobo::human::coarse_tissue
