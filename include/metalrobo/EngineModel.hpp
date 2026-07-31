#pragma once

#include "metalrobo/Collision.hpp"
#include "metalrobo/ConstraintIR.hpp"
#include "metalrobo/engine_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace metalrobo {

// Canonical, compiled engine model. Unlike the v0 Franka Model, generalized
// configuration and velocity storage are explicitly different (nq != nv).
// Runtime buffers are separate; this object contains immutable topology and a
// complete reset state.
struct EngineModel {
    MRWorldGPU world{};
    std::vector<MRArticulationGPU> articulations;
    std::vector<MRJointDescriptorGPU> joints;
    // Exactly one authoritative record per generalized velocity coordinate,
    // stored in global v order.
    std::vector<MRDofPropertiesGPU> dofs;
    // Optional measured or explicitly authored actuator truth. An empty
    // vector means the mechanism has no identified profile and preserves the
    // authoritative DoF effort limits. When present it is global-v ordered
    // and contains inactive zero records for unactuated coordinates.
    std::vector<MRActuatorProfileGPU> actuatorProfiles;
    std::vector<MRBodyPropertiesGPU> bodies;
    std::vector<MRShapeGPU> shapes;
    std::vector<MRMaterialGPU> materials;
    // Cooker-owned immutable geometry arenas. MRShapeGPU::geometryOffset
    // indexes geometryHeaders; every nested offset is relative to the
    // corresponding typed arena below.
    std::vector<MRGeometryHeaderGPU> geometryHeaders;
    std::vector<mr_float4> geometryVertices;
    std::vector<std::uint32_t> geometryIndices;
    std::vector<MRConvexFaceGPU> convexFaces;
    std::vector<MRConvexHalfEdgeGPU> convexHalfEdges;
    std::vector<MRMeshBVHNodeGPU> meshBvhNodes;
    std::vector<MRMeshTriangleGPU> meshTriangles;
    // Canonical collider-index pairs removed before broadphase compilation.
    // SRDF disabled-collision pairs and calibrated task exclusions live here
    // so every runtime and world-pack fingerprint sees identical filtering.
    std::vector<CollisionPairExclusion> collisionExclusions;
    // Immutable authored mechanism program. Contact blocks remain dynamic,
    // while limits, equality/loop rows, gears, tendons, RCM constraints, and
    // calibrated bounded friction travel with the model and world pack.
    ConstraintIR constraintProgram;
    std::vector<float> defaultQ;
    std::vector<float> defaultV;
    // Semantic identities are canonical compiler input, not runtime lookup
    // data. Importers preserve authored names here so TaskPack compilation can
    // resolve them once into stable body, joint, DoF, and collider indices.
    // Empty vectors are permitted for legacy programmatic models that are
    // never bound to a task; non-empty vectors must exactly match topology.
    std::vector<std::string> bodyNames;
    std::vector<std::string> jointNames;
    std::vector<std::string> dofNames;
    std::vector<std::string> shapeNames;
    std::string name;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

// Minimal floating/free-body scene used to prove the canonical nq=7, nv=6
// path independently from the fixed-base Franka compatibility runtime.
[[nodiscard]] EngineModel makeFreeSphereEngineModel();

} // namespace metalrobo
