#include "metalrobo/HeterogeneousWorld.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

using Vec3 = std::array<double, 3>;

bool setReason(std::string* reason, std::string message) {
    if (reason != nullptr) {
        *reason = std::move(message);
    }
    return false;
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const mr_float4 value) {
    return
        std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
}

bool finite(const Vec3& value) {
    return std::ranges::all_of(value, [](const double item) {
        return finite(item);
    });
}

Vec3 add(const Vec3& left, const Vec3& right) {
    return {
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    };
}

Vec3 subtract(const Vec3& left, const Vec3& right) {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}

Vec3 multiply(const Vec3& value, const double scale) {
    return {
        value[0] * scale,
        value[1] * scale,
        value[2] * scale,
    };
}

Vec3 cross(const Vec3& left, const Vec3& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

double norm(const Vec3& value) {
    return std::sqrt(
        value[0] * value[0] +
        value[1] * value[1] +
        value[2] * value[2]
    );
}

Vec3 rotate(const mr_float4 quaternion, const Vec3& value) {
    const Vec3 imaginary{
        quaternion.x,
        quaternion.y,
        quaternion.z,
    };
    return add(
        value,
        multiply(
            cross(
                imaginary,
                add(
                    cross(imaginary, value),
                    multiply(value, quaternion.w)
                )
            ),
            2.0
        )
    );
}

bool validSceneState(
    const MRBodyPropertiesGPU& properties,
    const MRBodyStateGPU& state
) {
    const double quaternionNorm =
        static_cast<double>(state.orientation.x) *
            state.orientation.x +
        static_cast<double>(state.orientation.y) *
            state.orientation.y +
        static_cast<double>(state.orientation.z) *
            state.orientation.z +
        static_cast<double>(state.orientation.w) *
            state.orientation.w;
    return
        properties.articulationIndex == MR_INVALID_INDEX &&
        state.flagsAndIndices[0] == properties.motionType &&
        state.flagsAndIndices[1] == MR_INVALID_INDEX &&
        finite(state.position) &&
        finite(state.orientation) &&
        finite(state.linearVelocityAndInverseMass) &&
        finite(state.angularVelocity) &&
        finite(state.inverseInertiaWorldRow0) &&
        finite(state.inverseInertiaWorldRow1) &&
        finite(state.inverseInertiaWorldRow2) &&
        std::abs(quaternionNorm - 1.0) <= 2.0e-4 &&
        (
            properties.motionType == MR_MOTION_DYNAMIC
            ? state.linearVelocityAndInverseMass.w > 0.0f
            : state.linearVelocityAndInverseMass.w == 0.0f
        );
}

bool validRodState(
    const DiscreteElasticRodModel& model,
    const DiscreteElasticRodState& state
) {
    DiscreteElasticRodEnergy energy;
    return evaluateDiscreteElasticRodEnergy(
        model,
        state,
        energy
    ).succeeded();
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

template <typename T>
std::uint64_t hashPod(std::uint64_t hash, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    return hashBytes(hash, &value, sizeof(value));
}

template <typename T>
std::uint64_t hashVector(
    std::uint64_t hash,
    const std::vector<T>& values
) {
    static_assert(std::is_trivially_copyable_v<T>);
    const std::uint64_t count = values.size();
    hash = hashPod(hash, count);
    return values.empty()
        ? hash
        : hashBytes(
              hash,
              values.data(),
              values.size() * sizeof(T)
          );
}

std::uint64_t hashString(
    std::uint64_t hash,
    const std::string& value
) {
    const std::uint64_t count = value.size();
    hash = hashPod(hash, count);
    return value.empty()
        ? hash
        : hashBytes(hash, value.data(), value.size());
}

std::uint64_t hashModel(
    std::uint64_t hash,
    const EngineModel& model
) {
    hash = hashPod(hash, model.world);
    hash = hashVector(hash, model.articulations);
    hash = hashVector(hash, model.joints);
    hash = hashVector(hash, model.dofs);
    hash = hashVector(hash, model.actuatorProfiles);
    hash = hashVector(hash, model.bodies);
    hash = hashVector(hash, model.shapes);
    hash = hashVector(hash, model.materials);
    hash = hashVector(hash, model.geometryHeaders);
    hash = hashVector(hash, model.geometryVertices);
    hash = hashVector(hash, model.geometryIndices);
    hash = hashVector(hash, model.convexFaces);
    hash = hashVector(hash, model.convexHalfEdges);
    hash = hashVector(hash, model.meshBvhNodes);
    hash = hashVector(hash, model.meshTriangles);
    hash = hashVector(hash, model.collisionExclusions);
    hash = hashPod(hash, model.constraintProgram.abiVersion);
    hash = hashVector(hash, model.constraintProgram.blocks);
    hash = hashVector(hash, model.constraintProgram.endpoints);
    hash = hashVector(hash, model.constraintProgram.rows);
    hash = hashVector(hash, model.constraintProgram.cones);
    hash = hashVector(hash, model.constraintProgram.warmImpulses);
    hash = hashVector(hash, model.defaultQ);
    hash = hashVector(hash, model.defaultV);
    return hashString(hash, model.name);
}

std::uint64_t hashRod(
    std::uint64_t hash,
    const HeterogeneousRodProgram& rod
) {
    hash = hashString(hash, rod.instanceId);
    hash = hashString(hash, rod.model.name);
    hash = hashString(hash, rod.model.fidelityBoundary);
    hash = hashPod(hash, rod.model.radius);
    hash = hashVector(hash, rod.model.restPositions);
    hash = hashVector(hash, rod.model.restTwists);
    hash = hashVector(hash, rod.model.restLengths);
    hash = hashVector(hash, rod.model.nodeMasses);
    hash = hashVector(hash, rod.model.edgeRotationalInertias);
    hash = hashVector(hash, rod.model.stretchStiffness);
    hash = hashVector(hash, rod.model.bendStiffness);
    hash = hashVector(hash, rod.model.twistStiffness);
    hash = hashVector(hash, rod.defaultState.positions);
    hash = hashVector(hash, rod.defaultState.velocities);
    hash = hashVector(hash, rod.defaultState.twists);
    hash = hashVector(hash, rod.defaultState.twistRates);
    hash = hashPod(hash, rod.stepConfig.timestep);
    hash = hashPod(hash, rod.stepConfig.gravity);
    hash = hashPod(hash, rod.stepConfig.solverIterations);
    hash = hashPod(hash, rod.stepConfig.constraintTolerance);
    hash = hashPod(hash, rod.stepConfig.linearDamping);
    hash = hashPod(hash, rod.stepConfig.twistDamping);
    hash = hashPod(hash, rod.stepConfig.derivativeStep);
    const std::uint32_t selfCollision =
        rod.stepConfig.enableSelfCollision ? 1u : 0u;
    hash = hashPod(hash, selfCollision);
    hash = hashPod(hash, rod.stepConfig.selfCollisionMargin);
    hash = hashPod(
        hash,
        rod.stepConfig.selfCollisionCompliance
    );
    hash = hashPod(hash, rod.collision.materialIndex);
    hash = hashPod(hash, rod.collision.collisionGroup);
    hash = hashPod(hash, rod.collision.collisionMask);
    hash = hashPod(hash, rod.collision.topologyGeneration);
    hash = hashPod(hash, rod.collision.contactOffset);
    hash = hashPod(hash, rod.collision.restOffset);
    const std::uint32_t toolCollision =
        rod.collision.enableToolCollision ? 1u : 0u;
    const std::uint32_t ccd =
        rod.collision.enableCCD ? 1u : 0u;
    hash = hashPod(hash, toolCollision);
    hash = hashPod(hash, ccd);
    hash = hashVector(hash, rod.attachments);
    return hashVector(hash, rod.rigidBindings);
}

HeterogeneousWorldComposeDiagnostics fail(
    HeterogeneousWorldComposeDiagnostics diagnostics,
    const HeterogeneousWorldComposeStatus status,
    std::string message,
    const std::uint32_t component = MR_INVALID_INDEX
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    diagnostics.firstFailingComponent = component;
    return diagnostics;
}

EngineModel makeNeedleSceneModel(
    const CurvedSutureNeedleAsset& needle,
    const mr_float4 gravityAndTimestep
) {
    EngineModel model;
    model.name = "dynamic_curved_suture_needle_scene";
    model.bodies.push_back(needle.rigid.body);
    MRBodyPropertiesGPU& body = model.bodies.front();
    body.articulationIndex = MR_INVALID_INDEX;
    body.parentBody = MR_INVALID_INDEX;
    body.inboundJoint = MR_INVALID_INDEX;
    model.materials.push_back(needle.rigid.material);
    model.shapes = needle.rigid.shapes;
    for (MRShapeGPU& shape : model.shapes) {
        shape.bodyIndex = 0u;
        shape.materialIndex = 0u;
    }
    MRWorldGPU& world = model.world;
    world.abiVersion = MR_ENGINE_ABI_VERSION;
    world.bodyCount = 1u;
    world.shapeCount =
        static_cast<std::uint32_t>(model.shapes.size());
    world.materialCount = 1u;
    const std::uint64_t shapeCount = model.shapes.size();
    const std::uint64_t pairCount =
        shapeCount * (shapeCount - 1u) / 2u;
    world.pairCapacity = static_cast<std::uint32_t>(
        std::max<std::uint64_t>(pairCount, 1u)
    );
    world.contactCapacity = static_cast<std::uint32_t>(
        std::max<std::uint64_t>(4u * pairCount, 8u)
    );
    world.constraintCapacity =
        std::max<std::uint32_t>(world.contactCapacity, 8u);
    world.islandCapacity = 1u;
    world.solverType = MR_SOLVER_THROUGHPUT_PGS;
    world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    world.gravityAndTimestep = gravityAndTimestep;
    world.solverScales = {
        1.0e-7f,
        1.0e-9f,
        2.0f,
        1.0e-5f,
    };
    return model;
}

} // namespace

std::uint64_t heterogeneousWorldFingerprint(
    const HeterogeneousWorld& world
) noexcept {
    try {
        std::uint64_t hash = kFnvOffset;
        hash = hashPod(hash, world.formatVersion);
        hash = hashModel(hash, world.model);
        hash = hashVector(hash, world.sceneBodyIndices);
        hash = hashVector(hash, world.defaultSceneBodies);
        const std::uint64_t rodCount = world.rods.size();
        hash = hashPod(hash, rodCount);
        for (const HeterogeneousRodProgram& rod : world.rods) {
            hash = hashRod(hash, rod);
        }
        const std::uint64_t componentCount =
            world.componentInstanceIds.size();
        hash = hashPod(hash, componentCount);
        for (const std::string& id :
             world.componentInstanceIds) {
            hash = hashString(hash, id);
        }
        return hash;
    } catch (...) {
        return 0u;
    }
}

bool HeterogeneousWorld::valid(std::string* reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (formatVersion != kHeterogeneousWorldFormatVersion) {
        return setReason(
            reason,
            "heterogeneous world format version is unsupported"
        );
    }
    std::string modelReason;
    if (!model.valid(&modelReason)) {
        return setReason(
            reason,
            "heterogeneous model is invalid: " + modelReason
        );
    }
    std::vector<std::uint32_t> expectedSceneBodies;
    for (std::uint32_t body = 0u;
         body < model.bodies.size();
         ++body) {
        if (model.bodies[body].articulationIndex ==
            MR_INVALID_INDEX) {
            expectedSceneBodies.push_back(body);
        }
    }
    if (sceneBodyIndices != expectedSceneBodies ||
        defaultSceneBodies.size() != sceneBodyIndices.size()) {
        return setReason(
            reason,
            "scene body packing does not match model ownership"
        );
    }
    for (std::size_t local = 0u;
         local < sceneBodyIndices.size();
         ++local) {
        if (!validSceneState(
                model.bodies[sceneBodyIndices[local]],
                defaultSceneBodies[local]
            )) {
            return setReason(
                reason,
                "scene reset state is inconsistent with body properties"
            );
        }
    }
    std::set<std::string> identities;
    for (const std::string& id : componentInstanceIds) {
        if (id.empty() || !identities.insert(id).second) {
            return setReason(
                reason,
                "component instance identities are empty or duplicated"
            );
        }
    }
    for (const HeterogeneousRodProgram& rod : rods) {
        if (rod.instanceId.empty() ||
            !identities.insert(rod.instanceId).second ||
            !rod.model.valid(&modelReason) ||
            !validRodState(rod.model, rod.defaultState) ||
            rod.attachments.size() !=
                rod.rigidBindings.size()) {
            return setReason(
                reason,
                "rod program identity, model, state, or binding count is invalid"
            );
        }
        if (!finite(rod.stepConfig.timestep) ||
            !(rod.stepConfig.timestep > 0.0) ||
            !finite(rod.stepConfig.gravity) ||
            rod.stepConfig.solverIterations == 0u ||
            !finite(rod.stepConfig.constraintTolerance) ||
            !(rod.stepConfig.constraintTolerance > 0.0) ||
            !finite(rod.stepConfig.linearDamping) ||
            rod.stepConfig.linearDamping < 0.0 ||
            !finite(rod.stepConfig.twistDamping) ||
            rod.stepConfig.twistDamping < 0.0 ||
            !finite(rod.stepConfig.derivativeStep) ||
            !(rod.stepConfig.derivativeStep > 0.0) ||
            !finite(rod.stepConfig.selfCollisionMargin) ||
            rod.stepConfig.selfCollisionMargin < 0.0 ||
            !finite(
                rod.stepConfig.selfCollisionCompliance
            ) ||
            rod.stepConfig.selfCollisionCompliance < 0.0) {
            return setReason(
                reason,
                "rod step or self-contact configuration is invalid"
            );
        }
        if (rod.collision.materialIndex >=
                model.materials.size() ||
            rod.collision.topologyGeneration == 0u ||
            !finite(rod.collision.contactOffset) ||
            rod.collision.contactOffset < 0.0 ||
            !finite(rod.collision.restOffset) ||
            rod.collision.restOffset < 0.0 ||
            rod.collision.restOffset >
                rod.collision.contactOffset ||
            (
                rod.collision.enableToolCollision &&
                (
                    rod.collision.collisionGroup == 0u ||
                    rod.collision.collisionMask == 0u
                )
            )) {
            return setReason(
                reason,
                "rod collision configuration is invalid"
            );
        }
        std::set<std::uint32_t> nodes;
        std::set<std::uint32_t> bodies;
        for (std::size_t index = 0u;
             index < rod.attachments.size();
             ++index) {
            const DiscreteRodAttachment& attachment =
                rod.attachments[index];
            const DiscreteRodRigidAttachmentBinding& binding =
                rod.rigidBindings[index];
            if (attachment.nodeIndex >=
                    rod.model.restPositions.size() ||
                !nodes.insert(attachment.nodeIndex).second ||
                !finite(attachment.targetPosition) ||
                !finite(attachment.targetVelocity) ||
                !finite(attachment.compliance) ||
                attachment.compliance < 0.0 ||
                !finite(binding.localAnchor)) {
                return setReason(
                    reason,
                    "rod attachment payload is invalid"
                );
            }
            if (binding.bodyIndex ==
                kDiscreteRodNoRigidBody) {
                continue;
            }
            if (binding.bodyIndex >=
                    defaultSceneBodies.size() ||
                !bodies.insert(binding.bodyIndex).second ||
                defaultSceneBodies[binding.bodyIndex].
                        flagsAndIndices[0] !=
                    MR_MOTION_DYNAMIC) {
                return setReason(
                    reason,
                    "rod rigid binding does not name a unique dynamic scene body"
                );
            }
            const MRBodyStateGPU& body =
                defaultSceneBodies[binding.bodyIndex];
            const Vec3 offset = rotate(
                body.orientation,
                binding.localAnchor
            );
            const Vec3 expectedPosition = add(
                {
                    body.position.x,
                    body.position.y,
                    body.position.z,
                },
                offset
            );
            const Vec3 expectedVelocity = add(
                {
                    body.linearVelocityAndInverseMass.x,
                    body.linearVelocityAndInverseMass.y,
                    body.linearVelocityAndInverseMass.z,
                },
                cross(
                    {
                        body.angularVelocity.x,
                        body.angularVelocity.y,
                        body.angularVelocity.z,
                    },
                    offset
                )
            );
            if (norm(subtract(
                    attachment.targetPosition,
                    expectedPosition
                )) > 2.0e-6 ||
                norm(subtract(
                    attachment.targetVelocity,
                    expectedVelocity
                )) > 2.0e-6) {
                return setReason(
                    reason,
                    "rod reset target disagrees with its rigid anchor"
                );
            }
        }
    }
    const std::uint64_t expectedFingerprint =
        heterogeneousWorldFingerprint(*this);
    if (fingerprint == 0u ||
        fingerprint != expectedFingerprint) {
        return setReason(
            reason,
            "heterogeneous world fingerprint mismatch"
        );
    }
    return true;
}

HeterogeneousWorldComposeDiagnostics
composeHeterogeneousWorld(
    const std::span<const HeterogeneousWorldComponent> components,
    const std::span<const HeterogeneousRodProgram> rods,
    HeterogeneousWorld& output,
    const EngineModelComposeConfig& config
) {
    HeterogeneousWorldComposeDiagnostics diagnostics;
    diagnostics.componentCount =
        static_cast<std::uint32_t>(components.size());
    diagnostics.rodCount =
        static_cast<std::uint32_t>(rods.size());
    if (components.empty()) {
        return fail(
            std::move(diagnostics),
            HeterogeneousWorldComposeStatus::
                invalidConfiguration,
            "heterogeneous world requires at least one component"
        );
    }
    try {
        std::vector<EngineModelComponent> modelComponents;
        modelComponents.reserve(components.size());
        std::vector<MRBodyStateGPU> sceneStates;
        std::vector<std::string> componentIds;
        componentIds.reserve(components.size());
        for (std::uint32_t componentIndex = 0u;
             componentIndex < components.size();
             ++componentIndex) {
            const HeterogeneousWorldComponent& component =
                components[componentIndex];
            if (component.model == nullptr ||
                component.instanceId.empty()) {
                return fail(
                    std::move(diagnostics),
                    HeterogeneousWorldComposeStatus::
                        invalidComponent,
                    "heterogeneous component is empty",
                    componentIndex
                );
            }
            std::size_t expectedSceneStates = 0u;
            for (const MRBodyPropertiesGPU& body :
                 component.model->bodies) {
                expectedSceneStates +=
                    body.articulationIndex == MR_INVALID_INDEX
                    ? 1u
                    : 0u;
            }
            if (component.defaultSceneBodies.size() !=
                expectedSceneStates) {
                return fail(
                    std::move(diagnostics),
                    HeterogeneousWorldComposeStatus::
                        invalidSceneState,
                    "component scene-state count disagrees with topology",
                    componentIndex
                );
            }
            std::size_t localScene = 0u;
            for (const MRBodyPropertiesGPU& body :
                 component.model->bodies) {
                if (body.articulationIndex != MR_INVALID_INDEX) {
                    continue;
                }
                if (!validSceneState(
                        body,
                        component.defaultSceneBodies[
                            localScene++
                        ]
                    )) {
                    return fail(
                        std::move(diagnostics),
                        HeterogeneousWorldComposeStatus::
                            invalidSceneState,
                        "component scene reset is invalid",
                        componentIndex
                    );
                }
            }
            modelComponents.push_back({
                .model = component.model,
                .instanceId = component.instanceId,
            });
            sceneStates.insert(
                sceneStates.end(),
                component.defaultSceneBodies.begin(),
                component.defaultSceneBodies.end()
            );
            componentIds.emplace_back(component.instanceId);
        }

        EngineModel composed;
        const EngineModelComposeDiagnostics modelDiagnostics =
            composeEngineModels(
                modelComponents,
                composed,
                config
            );
        if (!modelDiagnostics.succeeded()) {
            return fail(
                std::move(diagnostics),
                HeterogeneousWorldComposeStatus::
                    modelCompositionFailure,
                modelDiagnostics.message,
                modelDiagnostics.firstFailingComponent
            );
        }

        HeterogeneousWorld staged;
        staged.model = std::move(composed);
        staged.defaultSceneBodies = std::move(sceneStates);
        staged.rods.assign(rods.begin(), rods.end());
        staged.componentInstanceIds =
            std::move(componentIds);
        for (std::uint32_t body = 0u;
             body < staged.model.bodies.size();
             ++body) {
            if (staged.model.bodies[body].articulationIndex ==
                MR_INVALID_INDEX) {
                staged.sceneBodyIndices.push_back(body);
            }
        }
        // Scene state is composed in component order, which is also the
        // canonical non-articulated body order produced by the model
        // composer. Rebase the optional identity field to the composed global
        // body index so the persistent world can validate and consume the
        // state without accepting stale component-local identities.
        for (std::size_t localScene = 0u;
             localScene < staged.defaultSceneBodies.size();
             ++localScene) {
            staged.defaultSceneBodies[localScene]
                .flagsAndIndices[2] =
                staged.sceneBodyIndices[localScene];
        }
        staged.fingerprint =
            heterogeneousWorldFingerprint(staged);
        std::string reason;
        if (!staged.valid(&reason)) {
            return fail(
                std::move(diagnostics),
                HeterogeneousWorldComposeStatus::invalidWorld,
                std::move(reason)
            );
        }
        diagnostics.articulationCount =
            static_cast<std::uint32_t>(
                staged.model.articulations.size()
            );
        diagnostics.sceneBodyCount =
            static_cast<std::uint32_t>(
                staged.defaultSceneBodies.size()
            );
        diagnostics.fingerprint = staged.fingerprint;
        output = std::move(staged);
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return fail(
            std::move(diagnostics),
            HeterogeneousWorldComposeStatus::allocationFailure,
            "heterogeneous world allocation failed"
        );
    } catch (const std::exception& exception) {
        return fail(
            std::move(diagnostics),
            HeterogeneousWorldComposeStatus::invalidWorld,
            exception.what()
        );
    }
}

HeterogeneousWorldComposeDiagnostics
makeDualDvrkPsmNeedleThreadHeterogeneousWorld(
    HeterogeneousWorld& output,
    const DualPsmNeedleThreadWorldConfig& config
) {
    try {
        const DualPsmNeedleThreadWorld surgical =
            makeDualDvrkPsmNeedleThreadWorld(config);
        EngineModel needleModel = makeNeedleSceneModel(
            surgical.needle,
            surgical.robots.model.world.gravityAndTimestep
        );
        std::string reason;
        if (!needleModel.valid(&reason)) {
            HeterogeneousWorldComposeDiagnostics diagnostics;
            return fail(
                std::move(diagnostics),
                HeterogeneousWorldComposeStatus::invalidComponent,
                "needle scene model is invalid: " + reason,
                1u
            );
        }
        const std::array<MRBodyStateGPU, 1> needleStates{
            surgical.needleState,
        };
        const std::array<HeterogeneousWorldComponent, 2>
            components{{
                {
                    .model = &surgical.robots.model,
                    .instanceId = "dual_psm",
                },
                {
                    .model = &needleModel,
                    .instanceId = "curved_needle",
                    .defaultSceneBodies = needleStates,
                },
            }};
        HeterogeneousRodProgram thread;
        thread.instanceId = "suture_thread";
        thread.model = surgical.threadModel;
        thread.defaultState = surgical.threadState;
        thread.stepConfig.timestep =
            surgical.robots.model.world.gravityAndTimestep.w;
        thread.stepConfig.gravity = {
            surgical.robots.model.world.gravityAndTimestep.x,
            surgical.robots.model.world.gravityAndTimestep.y,
            surgical.robots.model.world.gravityAndTimestep.z,
        };
        thread.stepConfig.enableSelfCollision = true;
        thread.attachments.assign(
            surgical.attachments.begin(),
            surgical.attachments.end()
        );
        thread.rigidBindings.assign(
            surgical.rigidBindings.begin(),
            surgical.rigidBindings.end()
        );
        const std::array<HeterogeneousRodProgram, 1> rods{
            std::move(thread),
        };
        EngineModelComposeConfig composeConfig;
        composeConfig.name =
            "dual_psm_curved_needle_thread_world";
        composeConfig.gravityAndTimestep =
            surgical.robots.model.world.gravityAndTimestep;
        return composeHeterogeneousWorld(
            components,
            rods,
            output,
            composeConfig
        );
    } catch (const std::bad_alloc&) {
        HeterogeneousWorldComposeDiagnostics diagnostics;
        return fail(
            std::move(diagnostics),
            HeterogeneousWorldComposeStatus::allocationFailure,
            "surgical heterogeneous world allocation failed"
        );
    } catch (const std::exception& exception) {
        HeterogeneousWorldComposeDiagnostics diagnostics;
        return fail(
            std::move(diagnostics),
            HeterogeneousWorldComposeStatus::invalidWorld,
            exception.what()
        );
    }
}

const char* heterogeneousWorldComposeStatusName(
    const HeterogeneousWorldComposeStatus status
) noexcept {
    switch (status) {
    case HeterogeneousWorldComposeStatus::success:
        return "success";
    case HeterogeneousWorldComposeStatus::invalidConfiguration:
        return "invalid_configuration";
    case HeterogeneousWorldComposeStatus::invalidComponent:
        return "invalid_component";
    case HeterogeneousWorldComposeStatus::invalidSceneState:
        return "invalid_scene_state";
    case HeterogeneousWorldComposeStatus::invalidRodProgram:
        return "invalid_rod_program";
    case HeterogeneousWorldComposeStatus::modelCompositionFailure:
        return "model_composition_failure";
    case HeterogeneousWorldComposeStatus::invalidWorld:
        return "invalid_world";
    case HeterogeneousWorldComposeStatus::allocationFailure:
        return "allocation_failure";
    }
    return "unknown";
}

} // namespace metalrobo
