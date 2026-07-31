#include "metalrobo/EngineModelComposer.hpp"

#include "metalrobo/ConstraintIR.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset =
    1469598103934665603ull;
constexpr std::uint64_t kFnvPrime =
    1099511628211ull;

bool finite(const mr_float4 value) {
    return
        std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
}

EngineModelComposeDiagnostics fail(
    EngineModelComposeDiagnostics diagnostics,
    const EngineModelComposeStatus status,
    std::string message,
    const std::uint32_t component = MR_INVALID_INDEX
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    diagnostics.firstFailingComponent = component;
    return diagnostics;
}

bool checkedOffset(
    const std::uint32_t value,
    const std::uint64_t offset,
    std::uint32_t& result
) {
    if (value == MR_INVALID_INDEX) {
        result = MR_INVALID_INDEX;
        return true;
    }
    if (offset + value >
        std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    result = static_cast<std::uint32_t>(offset + value);
    return true;
}

bool checkedCount(
    const std::size_t size,
    std::uint32_t& result
) {
    if (size >
        std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    result = static_cast<std::uint32_t>(size);
    return true;
}

std::uint32_t& component(
    mr_uint4& value,
    const std::size_t index
) {
    switch (index) {
    case 0u:
        return value.x;
    case 1u:
        return value.y;
    case 2u:
        return value.z;
    default:
        return value.w;
    }
}

std::uint64_t hashBytes(
    std::uint64_t hash,
    const void* data,
    const std::size_t size
) {
    const auto* bytes =
        static_cast<const unsigned char*>(data);
    for (std::size_t index = 0u; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash;
}

ConstraintIRStableKey namespacedKey(
    const std::string_view instanceId,
    const ConstraintIRStableKey& source
) {
    const std::array<std::uint64_t, 2> seeds{
        kFnvOffset,
        kFnvOffset ^ 0x9e3779b97f4a7c15ull,
    };
    std::array<std::uint64_t, 2> hashes = seeds;
    for (std::size_t lane = 0u; lane < hashes.size(); ++lane) {
        hashes[lane] = hashBytes(
            hashes[lane],
            instanceId.data(),
            instanceId.size()
        );
        const std::uint32_t separator =
            0x434f4d50u + static_cast<std::uint32_t>(lane);
        hashes[lane] = hashBytes(
            hashes[lane],
            &separator,
            sizeof(separator)
        );
        hashes[lane] = hashBytes(
            hashes[lane],
            &source,
            sizeof(source)
        );
    }
    ConstraintIRStableKey result{};
    result.words[0] =
        static_cast<std::uint32_t>(hashes[0]);
    result.words[1] =
        static_cast<std::uint32_t>(hashes[0] >> 32u);
    result.words[2] =
        static_cast<std::uint32_t>(hashes[1]);
    result.words[3] =
        static_cast<std::uint32_t>(hashes[1] >> 32u);
    return result;
}

struct ComponentOffsets {
    std::uint32_t articulation = 0u;
    std::uint32_t joint = 0u;
    std::uint32_t q = 0u;
    std::uint32_t v = 0u;
    std::uint32_t body = 0u;
    std::uint32_t shape = 0u;
    std::uint32_t material = 0u;
    std::uint32_t geometry = 0u;
    std::uint32_t vertex = 0u;
    std::uint32_t index = 0u;
    std::uint32_t face = 0u;
    std::uint32_t halfEdge = 0u;
    std::uint32_t bvh = 0u;
    std::uint32_t triangle = 0u;
    std::uint32_t island = 0u;
    std::uint32_t event = 0u;
};

struct OwnedConstraint {
    ConstraintIRStableKey key{};
    std::uint32_t type = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t islandIndex = 0u;
    std::uint32_t eventSlot = kConstraintIRInvalidIndex;
    std::vector<ConstraintIREndpoint> endpoints;
    std::vector<ConstraintIRRow> rows;
    std::optional<ConstraintIRCone> cone;
    std::vector<float> warmImpulses;
};

bool appendConstraintProgram(
    const EngineModel& source,
    const std::string_view instanceId,
    const ComponentOffsets& offsets,
    std::vector<OwnedConstraint>& constraints
) {
    for (const ConstraintIRBlock& block :
         source.constraintProgram.blocks) {
        OwnedConstraint owned;
        owned.key = namespacedKey(instanceId, block.key);
        owned.type = block.type;
        owned.flags = block.flags;
        if (!checkedOffset(
                block.islandIndex,
                offsets.island,
                owned.islandIndex
            )) {
            return false;
        }
        if (!checkedOffset(
                block.eventSlot,
                offsets.event,
                owned.eventSlot
            )) {
            return false;
        }
        owned.endpoints.assign(
            source.constraintProgram.endpoints.begin() +
                block.endpointOffset,
            source.constraintProgram.endpoints.begin() +
                block.endpointOffset +
                block.endpointCount
        );
        for (ConstraintIREndpoint& endpoint :
             owned.endpoints) {
            if (endpoint.jacobianKind ==
                constraintIRJacobianGeneralized) {
                if (!checkedOffset(
                        endpoint.objectIndex,
                        offsets.v,
                        endpoint.objectIndex
                    ) ||
                    !checkedOffset(
                        endpoint.articulationIndex,
                        offsets.articulation,
                        endpoint.articulationIndex
                    )) {
                    return false;
                }
                if ((endpoint.flags &
                     constraintIREndpointQIndexValid) != 0u &&
                    !checkedOffset(
                        endpoint.linkIndex,
                        offsets.q,
                        endpoint.linkIndex
                    )) {
                    return false;
                }
            } else {
                if (endpoint.role !=
                        constraintIREndpointWorld &&
                    !checkedOffset(
                        endpoint.objectIndex,
                        offsets.body,
                        endpoint.objectIndex
                    )) {
                    return false;
                }
                if (!checkedOffset(
                        endpoint.articulationIndex,
                        offsets.articulation,
                        endpoint.articulationIndex
                    ) ||
                    !checkedOffset(
                        endpoint.linkIndex,
                        offsets.body,
                        endpoint.linkIndex
                    )) {
                    return false;
                }
            }
        }
        owned.rows.assign(
            source.constraintProgram.rows.begin() +
                block.rowOffset,
            source.constraintProgram.rows.begin() +
                block.rowOffset +
                block.dimension
        );
        owned.warmImpulses.assign(
            source.constraintProgram.warmImpulses.begin() +
                block.impulseOffset,
            source.constraintProgram.warmImpulses.begin() +
                block.impulseOffset +
                block.dimension
        );
        if (block.coneIndex != kConstraintIRInvalidIndex) {
            owned.cone =
                source.constraintProgram.cones[block.coneIndex];
        }
        constraints.push_back(std::move(owned));
    }
    return true;
}

bool appendGeometry(
    const EngineModel& source,
    const ComponentOffsets& offsets,
    EngineModel& output
) {
    for (const MRGeometryHeaderGPU& sourceHeader :
         source.geometryHeaders) {
        MRGeometryHeaderGPU header = sourceHeader;
        if (!checkedOffset(
                header.vertexOffset,
                offsets.vertex,
                header.vertexOffset
            ) ||
            !checkedOffset(
                header.indexOffset,
                offsets.index,
                header.indexOffset
            ) ||
            !checkedOffset(
                header.faceOffset,
                offsets.face,
                header.faceOffset
            ) ||
            !checkedOffset(
                header.halfEdgeOffset,
                offsets.halfEdge,
                header.halfEdgeOffset
            ) ||
            !checkedOffset(
                header.bvhOffset,
                offsets.bvh,
                header.bvhOffset
            ) ||
            !checkedOffset(
                header.triangleOffset,
                offsets.triangle,
                header.triangleOffset
            ) ||
            !checkedOffset(
                header.materialOffset,
                offsets.material,
                header.materialOffset
            )) {
            return false;
        }
        output.geometryHeaders.push_back(header);
    }
    output.geometryVertices.insert(
        output.geometryVertices.end(),
        source.geometryVertices.begin(),
        source.geometryVertices.end()
    );
    for (const std::uint32_t sourceIndex :
         source.geometryIndices) {
        std::uint32_t index = 0u;
        if (!checkedOffset(
                sourceIndex,
                offsets.vertex,
                index
            )) {
            return false;
        }
        output.geometryIndices.push_back(index);
    }
    for (const MRConvexFaceGPU& sourceFace :
         source.convexFaces) {
        MRConvexFaceGPU face = sourceFace;
        if (!checkedOffset(
                face.firstHalfEdge,
                offsets.halfEdge,
                face.firstHalfEdge
            )) {
            return false;
        }
        output.convexFaces.push_back(face);
    }
    for (const MRConvexHalfEdgeGPU& sourceEdge :
         source.convexHalfEdges) {
        MRConvexHalfEdgeGPU edge = sourceEdge;
        if (!checkedOffset(
                edge.originVertex,
                offsets.vertex,
                edge.originVertex
            ) ||
            !checkedOffset(
                edge.twinHalfEdge,
                offsets.halfEdge,
                edge.twinHalfEdge
            ) ||
            !checkedOffset(
                edge.nextHalfEdge,
                offsets.halfEdge,
                edge.nextHalfEdge
            ) ||
            !checkedOffset(
                edge.faceIndex,
                offsets.face,
                edge.faceIndex
            )) {
            return false;
        }
        output.convexHalfEdges.push_back(edge);
    }
    output.meshBvhNodes.insert(
        output.meshBvhNodes.end(),
        source.meshBvhNodes.begin(),
        source.meshBvhNodes.end()
    );
    for (const MRMeshTriangleGPU& sourceTriangle :
         source.meshTriangles) {
        MRMeshTriangleGPU triangle = sourceTriangle;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            if (!checkedOffset(
                    component(
                        triangle.verticesAndFeature,
                        axis
                    ),
                    offsets.vertex,
                    component(
                        triangle.verticesAndFeature,
                        axis
                    )
                ) ||
                !checkedOffset(
                    component(
                        triangle.adjacencyAndEdges,
                        axis
                    ),
                    offsets.triangle,
                    component(
                        triangle.adjacencyAndEdges,
                        axis
                    )
                )) {
                return false;
            }
        }
        if (!checkedOffset(
                triangle.materialAndFlags.x,
                offsets.material,
                triangle.materialAndFlags.x
            )) {
            return false;
        }
        output.meshTriangles.push_back(triangle);
    }
    return true;
}

} // namespace

EngineModelComposeDiagnostics composeEngineModels(
    const std::span<const EngineModelComponent> components,
    EngineModel& output,
    const EngineModelComposeConfig& config
) {
    EngineModelComposeDiagnostics diagnostics;
    diagnostics.componentCount =
        static_cast<std::uint32_t>(
            std::min<std::size_t>(
                components.size(),
                std::numeric_limits<std::uint32_t>::max()
            )
        );
    if (components.empty() ||
        config.name.empty() ||
        !finite(config.gravityAndTimestep) ||
        !(config.gravityAndTimestep.w > 0.0f) ||
        !finite(config.solverScales) ||
        config.solverType > MR_SOLVER_LEGACY_PROJECTED ||
        config.frictionConeType > MR_FRICTION_CONE_PYRAMID_8 ||
        config.contactsPerPair == 0u) {
        return fail(
            std::move(diagnostics),
            EngineModelComposeStatus::invalidConfiguration,
            "composition configuration is invalid"
        );
    }

    try {
        std::set<std::string, std::less<>> identities;
        EngineModel staged;
        staged.name = config.name;
        std::vector<OwnedConstraint> constraints;
        std::uint64_t islandOffset = 0u;
        std::uint64_t eventOffset = 0u;

        for (std::uint32_t componentIndex = 0u;
             componentIndex < components.size();
             ++componentIndex) {
            const EngineModelComponent& component =
                components[componentIndex];
            if (component.model == nullptr ||
                component.instanceId.empty()) {
                return fail(
                    std::move(diagnostics),
                    EngineModelComposeStatus::invalidComponent,
                    "component model or stable instance identity is empty",
                    componentIndex
                );
            }
            if (!identities.emplace(
                    component.instanceId
                ).second) {
                return fail(
                    std::move(diagnostics),
                    EngineModelComposeStatus::duplicateInstanceId,
                    "component instance identities are not unique",
                    componentIndex
                );
            }
            std::string reason;
            if (!component.model->valid(&reason)) {
                return fail(
                    std::move(diagnostics),
                    EngineModelComposeStatus::invalidComponent,
                    "invalid component model: " + reason,
                    componentIndex
                );
            }
            const EngineModel& source = *component.model;
            ComponentOffsets offsets;
            if (!checkedCount(
                    staged.articulations.size(),
                    offsets.articulation
                ) ||
                !checkedCount(staged.joints.size(), offsets.joint) ||
                !checkedCount(staged.defaultQ.size(), offsets.q) ||
                !checkedCount(staged.defaultV.size(), offsets.v) ||
                !checkedCount(staged.bodies.size(), offsets.body) ||
                !checkedCount(staged.shapes.size(), offsets.shape) ||
                !checkedCount(staged.materials.size(), offsets.material) ||
                !checkedCount(
                    staged.geometryHeaders.size(),
                    offsets.geometry
                ) ||
                !checkedCount(
                    staged.geometryVertices.size(),
                    offsets.vertex
                ) ||
                !checkedCount(
                    staged.geometryIndices.size(),
                    offsets.index
                ) ||
                !checkedCount(staged.convexFaces.size(), offsets.face) ||
                !checkedCount(
                    staged.convexHalfEdges.size(),
                    offsets.halfEdge
                ) ||
                !checkedCount(
                    staged.meshBvhNodes.size(),
                    offsets.bvh
                ) ||
                !checkedCount(
                    staged.meshTriangles.size(),
                    offsets.triangle
                ) ||
                islandOffset >
                    std::numeric_limits<std::uint32_t>::max() ||
                eventOffset >
                    std::numeric_limits<std::uint32_t>::max()) {
                return fail(
                    std::move(diagnostics),
                    EngineModelComposeStatus::capacityOverflow,
                    "component offsets exceed the 32-bit engine ABI",
                    componentIndex
                );
            }
            offsets.island =
                static_cast<std::uint32_t>(islandOffset);
            offsets.event =
                static_cast<std::uint32_t>(eventOffset);

            for (const MRArticulationGPU& sourceArticulation :
                 source.articulations) {
                MRArticulationGPU articulation =
                    sourceArticulation;
                if (!checkedOffset(
                        articulation.rootBody,
                        offsets.body,
                        articulation.rootBody
                    ) ||
                    !checkedOffset(
                        articulation.firstBody,
                        offsets.body,
                        articulation.firstBody
                    ) ||
                    !checkedOffset(
                        articulation.firstJoint,
                        offsets.joint,
                        articulation.firstJoint
                    ) ||
                    !checkedOffset(
                        articulation.qOffset,
                        offsets.q,
                        articulation.qOffset
                    ) ||
                    !checkedOffset(
                        articulation.vOffset,
                        offsets.v,
                        articulation.vOffset
                    )) {
                    return fail(
                        std::move(diagnostics),
                        EngineModelComposeStatus::capacityOverflow,
                        "articulation rebase overflow",
                        componentIndex
                    );
                }
                if (!checkedOffset(
                        articulation.solverGroup,
                        offsets.articulation,
                        articulation.solverGroup
                    )) {
                    return fail(
                        std::move(diagnostics),
                        EngineModelComposeStatus::capacityOverflow,
                        "solver group rebase overflow",
                        componentIndex
                    );
                }
                staged.articulations.push_back(articulation);
            }
            for (const MRJointDescriptorGPU& sourceJoint :
                 source.joints) {
                MRJointDescriptorGPU joint = sourceJoint;
                if (!checkedOffset(
                        joint.parentBody,
                        offsets.body,
                        joint.parentBody
                    ) ||
                    !checkedOffset(
                        joint.childBody,
                        offsets.body,
                        joint.childBody
                    ) ||
                    !checkedOffset(
                        joint.qOffset,
                        offsets.q,
                        joint.qOffset
                    ) ||
                    !checkedOffset(
                        joint.vOffset,
                        offsets.v,
                        joint.vOffset
                    )) {
                    return fail(
                        std::move(diagnostics),
                        EngineModelComposeStatus::capacityOverflow,
                        "joint rebase overflow",
                        componentIndex
                    );
                }
                staged.joints.push_back(joint);
            }
            for (const MRDofPropertiesGPU& sourceDof :
                 source.dofs) {
                MRDofPropertiesGPU dof = sourceDof;
                if (!checkedOffset(
                        dof.articulationIndex,
                        offsets.articulation,
                        dof.articulationIndex
                    ) ||
                    !checkedOffset(
                        dof.jointIndex,
                        offsets.joint,
                        dof.jointIndex
                    ) ||
                    !checkedOffset(
                        dof.qIndex,
                        offsets.q,
                        dof.qIndex
                    ) ||
                    !checkedOffset(
                        dof.vIndex,
                        offsets.v,
                        dof.vIndex
                    )) {
                    return fail(
                        std::move(diagnostics),
                        EngineModelComposeStatus::capacityOverflow,
                        "DoF rebase overflow",
                        componentIndex
                    );
                }
                staged.dofs.push_back(dof);
            }
            if (!source.actuatorProfiles.empty() &&
                staged.actuatorProfiles.empty() &&
                offsets.v != 0u) {
                staged.actuatorProfiles.resize(offsets.v);
                for (std::uint32_t index = 0u;
                     index < offsets.v;
                     ++index) {
                    staged.actuatorProfiles[index].identity.x =
                        index;
                }
            }
            if (!staged.actuatorProfiles.empty() ||
                !source.actuatorProfiles.empty()) {
                for (std::uint32_t local = 0u;
                     local < source.dofs.size();
                     ++local) {
                    MRActuatorProfileGPU profile{};
                    if (!source.actuatorProfiles.empty()) {
                        profile =
                            source.actuatorProfiles[local];
                    } else {
                        profile.identity.x = local;
                    }
                    if (!checkedOffset(
                            profile.identity.x,
                            offsets.v,
                            profile.identity.x
                        )) {
                        return fail(
                            std::move(diagnostics),
                            EngineModelComposeStatus::
                                capacityOverflow,
                            "actuator profile rebase overflow",
                            componentIndex
                        );
                    }
                    staged.actuatorProfiles.push_back(
                        profile
                    );
                }
            }
            for (const MRBodyPropertiesGPU& sourceBody :
                 source.bodies) {
                MRBodyPropertiesGPU body = sourceBody;
                if (!checkedOffset(
                        body.articulationIndex,
                        offsets.articulation,
                        body.articulationIndex
                    ) ||
                    !checkedOffset(
                        body.parentBody,
                        offsets.body,
                        body.parentBody
                    ) ||
                    !checkedOffset(
                        body.inboundJoint,
                        offsets.joint,
                        body.inboundJoint
                    )) {
                    return fail(
                        std::move(diagnostics),
                        EngineModelComposeStatus::capacityOverflow,
                        "body rebase overflow",
                        componentIndex
                    );
                }
                staged.bodies.push_back(body);
            }
            staged.materials.insert(
                staged.materials.end(),
                source.materials.begin(),
                source.materials.end()
            );
            if (!appendGeometry(source, offsets, staged)) {
                return fail(
                    std::move(diagnostics),
                    EngineModelComposeStatus::capacityOverflow,
                    "geometry arena rebase overflow",
                    componentIndex
                );
            }
            for (const MRShapeGPU& sourceShape :
                 source.shapes) {
                MRShapeGPU shape = sourceShape;
                if (!checkedOffset(
                        shape.bodyIndex,
                        offsets.body,
                        shape.bodyIndex
                    ) ||
                    !checkedOffset(
                        shape.materialIndex,
                        offsets.material,
                        shape.materialIndex
                    ) ||
                    (shape.geometryCount != 0u &&
                     !checkedOffset(
                         shape.geometryOffset,
                         offsets.geometry,
                         shape.geometryOffset
                     ))) {
                    return fail(
                        std::move(diagnostics),
                        EngineModelComposeStatus::capacityOverflow,
                        "shape rebase overflow",
                        componentIndex
                    );
                }
                staged.shapes.push_back(shape);
            }
            for (const CollisionPairExclusion& sourceExclusion :
                 source.collisionExclusions) {
                std::uint32_t first = 0u;
                std::uint32_t second = 0u;
                if (!checkedOffset(
                        sourceExclusion.colliderA,
                        offsets.shape,
                        first
                    ) ||
                    !checkedOffset(
                        sourceExclusion.colliderB,
                        offsets.shape,
                        second
                    )) {
                    return fail(
                        std::move(diagnostics),
                        EngineModelComposeStatus::capacityOverflow,
                        "collision exclusion rebase overflow",
                        componentIndex
                    );
                }
                staged.collisionExclusions.push_back({
                    .colliderA = first,
                    .colliderB = second,
                });
            }
            if (!appendConstraintProgram(
                    source,
                    component.instanceId,
                    offsets,
                    constraints
                )) {
                return fail(
                    std::move(diagnostics),
                    EngineModelComposeStatus::capacityOverflow,
                    "ConstraintIR rebase overflow",
                    componentIndex
                );
            }
            staged.defaultQ.insert(
                staged.defaultQ.end(),
                source.defaultQ.begin(),
                source.defaultQ.end()
            );
            staged.defaultV.insert(
                staged.defaultV.end(),
                source.defaultV.begin(),
                source.defaultV.end()
            );
            islandOffset += source.world.islandCapacity;
            std::uint32_t maximumEvent = 0u;
            bool hasEvent = false;
            for (const ConstraintIRBlock& block :
                 source.constraintProgram.blocks) {
                if (block.eventSlot !=
                    kConstraintIRInvalidIndex) {
                    maximumEvent = std::max(
                        maximumEvent,
                        block.eventSlot
                    );
                    hasEvent = true;
                }
            }
            eventOffset += hasEvent
                ? static_cast<std::uint64_t>(maximumEvent) + 1u
                : 0u;
        }

        std::vector<ConstraintIRSourceBlock> sources;
        sources.reserve(constraints.size());
        for (const OwnedConstraint& constraint : constraints) {
            sources.push_back({
                .key = constraint.key,
                .type = constraint.type,
                .flags = constraint.flags,
                .islandIndex = constraint.islandIndex,
                .eventSlot = constraint.eventSlot,
                .endpoints = constraint.endpoints,
                .rows = constraint.rows,
                .cone = constraint.cone,
                .warmImpulses = constraint.warmImpulses,
            });
        }
        ConstraintIRCompilationResult compiled =
            compileConstraintIR(sources);
        if (!compiled.succeeded()) {
            return fail(
                std::move(diagnostics),
                EngineModelComposeStatus::
                    constraintCompilationFailure,
                "composed ConstraintIR compilation failed: " +
                    compiled.diagnostics.message
            );
        }
        staged.constraintProgram = std::move(compiled.ir);
        std::ranges::sort(
            staged.collisionExclusions,
            [](const CollisionPairExclusion& left,
               const CollisionPairExclusion& right) {
                return
                    std::pair{
                        left.colliderA,
                        left.colliderB,
                    } <
                    std::pair{
                        right.colliderA,
                        right.colliderB,
                    };
            }
        );
        staged.collisionExclusions.erase(
            std::ranges::unique(
                staged.collisionExclusions,
                {},
                [](const CollisionPairExclusion& value) {
                    return std::pair{
                        value.colliderA,
                        value.colliderB,
                    };
                }
            ).begin(),
            staged.collisionExclusions.end()
        );

        MRWorldGPU& world = staged.world;
        world.abiVersion = MR_ENGINE_ABI_VERSION;
        if (!checkedCount(staged.bodies.size(), world.bodyCount) ||
            !checkedCount(
                staged.articulations.size(),
                world.articulationCount
            ) ||
            !checkedCount(staged.joints.size(), world.jointCount) ||
            !checkedCount(staged.shapes.size(), world.shapeCount) ||
            !checkedCount(
                staged.materials.size(),
                world.materialCount
            ) ||
            !checkedCount(staged.defaultQ.size(), world.nq) ||
            !checkedCount(staged.defaultV.size(), world.nv)) {
            return fail(
                std::move(diagnostics),
                EngineModelComposeStatus::capacityOverflow,
                "composed model counts exceed the 32-bit engine ABI"
            );
        }
        const std::uint64_t logicalPairs =
            static_cast<std::uint64_t>(world.shapeCount) *
            (world.shapeCount > 0u ? world.shapeCount - 1u : 0u) /
            2u;
        const std::uint64_t contacts =
            logicalPairs * config.contactsPerPair;
        const std::uint64_t constraintCapacity =
            contacts +
            staged.constraintProgram.blocks.size() +
            config.constraintHeadroom;
        if (logicalPairs >
                std::numeric_limits<std::uint32_t>::max() ||
            contacts >
                std::numeric_limits<std::uint32_t>::max() ||
            constraintCapacity >
                std::numeric_limits<std::uint32_t>::max() ||
            islandOffset >
                std::numeric_limits<std::uint32_t>::max()) {
            return fail(
                std::move(diagnostics),
                EngineModelComposeStatus::capacityOverflow,
                "derived world capacities exceed the 32-bit engine ABI"
            );
        }
        world.pairCapacity =
            static_cast<std::uint32_t>(
                std::max<std::uint64_t>(logicalPairs, 1u)
            );
        world.contactCapacity =
            static_cast<std::uint32_t>(
                std::max<std::uint64_t>(contacts, 8u)
            );
        world.constraintCapacity =
            static_cast<std::uint32_t>(
                std::max<std::uint64_t>(
                    constraintCapacity,
                    8u
                )
            );
        world.islandCapacity =
            static_cast<std::uint32_t>(
                std::max<std::uint64_t>(
                    islandOffset,
                    std::max<std::uint32_t>(
                        world.bodyCount,
                        1u
                    )
                )
            );
        world.solverType = config.solverType;
        world.frictionConeType =
            config.frictionConeType;
        world.gravityAndTimestep =
            config.gravityAndTimestep;
        world.solverScales = config.solverScales;

        std::string reason;
        if (!staged.valid(&reason)) {
            return fail(
                std::move(diagnostics),
                EngineModelComposeStatus::invalidComposedModel,
                "composed model is invalid: " + reason
            );
        }
        diagnostics.articulationCount =
            world.articulationCount;
        diagnostics.bodyCount = world.bodyCount;
        diagnostics.shapeCount = world.shapeCount;
        diagnostics.geometryCount =
            static_cast<std::uint32_t>(
                staged.geometryHeaders.size()
            );
        diagnostics.constraintBlockCount =
            static_cast<std::uint32_t>(
                staged.constraintProgram.blocks.size()
            );
        output = std::move(staged);
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return fail(
            std::move(diagnostics),
            EngineModelComposeStatus::allocationFailure,
            "model composition allocation failed"
        );
    }
}

const char* engineModelComposeStatusName(
    const EngineModelComposeStatus status
) noexcept {
    switch (status) {
    case EngineModelComposeStatus::success:
        return "success";
    case EngineModelComposeStatus::invalidConfiguration:
        return "invalid_configuration";
    case EngineModelComposeStatus::invalidComponent:
        return "invalid_component";
    case EngineModelComposeStatus::duplicateInstanceId:
        return "duplicate_instance_id";
    case EngineModelComposeStatus::capacityOverflow:
        return "capacity_overflow";
    case EngineModelComposeStatus::constraintCompilationFailure:
        return "constraint_compilation_failure";
    case EngineModelComposeStatus::invalidComposedModel:
        return "invalid_composed_model";
    case EngineModelComposeStatus::allocationFailure:
        return "allocation_failure";
    }
    return "unknown";
}

} // namespace metalrobo
