#include "numi/matter/detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <numbers>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace numi::matter {
namespace {

constexpr std::uint32_t kKnownMatterFlags =
    NM_MATTER_DETERMINISTIC |
    NM_MATTER_CONTACT |
    NM_MATTER_ADAPTIVE |
    NM_MATTER_IDENTIFICATION |
    NM_MATTER_MIXED_FEM |
    NM_MATTER_MULTIPHYSICS |
    NM_MATTER_MUTATION |
    NM_MATTER_LEARNED_MATERIAL |
    NM_MATTER_IPC;
constexpr std::uint32_t kKnownMaterialFlags =
    NM_MATERIAL_HAS_STATE |
    NM_MATERIAL_HAS_DISSIPATION |
    NM_MATERIAL_HAS_IMPLICIT_STATE;
constexpr std::uint32_t kKnownObjectFlags =
    NM_OBJECT_ACTIVE |
    NM_OBJECT_TWO_WAY_COUPLED |
    NM_OBJECT_ADAPTIVE |
    NM_OBJECT_IDENTIFIABLE |
    NM_OBJECT_MIXED_FEM |
    NM_OBJECT_MULTIPHYSICS |
    NM_OBJECT_MUTABLE_TOPOLOGY |
    NM_OBJECT_DISABLE_SELF_CONTACT;
constexpr std::uint32_t kKnownRigidFlags =
    NM_RIGID_ARTICULATED |
    NM_RIGID_DYNAMIC |
    NM_RIGID_PUNCTURE_TIP |
    NM_RIGID_SUTURE_STRAND |
    NM_RIGID_PUNCTURE_DILATOR;

[[nodiscard]] bool finite4(const nm_float4 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z) && std::isfinite(value.w);
}

[[nodiscard]] bool nearlyEqual(
    const float left,
    const float right,
    const float multiplier = 64.0f
) noexcept {
    const float scale = std::max({std::abs(left), std::abs(right), 1.0f});
    return std::abs(left - right) <=
        multiplier * std::numeric_limits<float>::epsilon() * scale;
}

[[nodiscard]] bool rangeWithin(
    const std::uint32_t first,
    const std::uint32_t count,
    const std::size_t capacity
) noexcept {
    return first <= capacity && count <= capacity - first;
}

[[nodiscard]] bool product3(
    const std::uint32_t x,
    const std::uint32_t y,
    const std::uint32_t z,
    std::size_t& result
) noexcept {
    if (x == 0u || y == 0u || z == 0u) {
        return false;
    }
    const std::uint64_t product =
        static_cast<std::uint64_t>(x) * y * z;
    if (product > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    result = static_cast<std::size_t>(product);
    return true;
}

class LayoutValidator {
public:
    LayoutValidator(const CompiledWorld& world, std::string* error)
        : world_(world), error_(error) {
        if (error_ != nullptr) {
            error_->clear();
        }
    }

    [[nodiscard]] bool run() {
        return validateDispatch() &&
            validateMixedAuthorities() &&
            validateProgramsAndMaterials() &&
            validateObjectsAndTopology() &&
            validateHumanAttachments() &&
            validateContact() &&
            validateAdaptiveAndSchedulers() &&
            validateIdentification() &&
            validateFingerprint();
    }

private:
    [[nodiscard]] bool fail(const std::string_view message) {
        if (error_ != nullptr && error_->empty()) {
            *error_ = std::string(message);
        }
        return false;
    }

    [[nodiscard]] bool failIndexed(
        const std::string_view role,
        const std::size_t index,
        const std::string_view message
    ) {
        if (error_ != nullptr && error_->empty()) {
            *error_ = std::string(role) + " " + std::to_string(index) +
                ": " + std::string(message);
        }
        return false;
    }

    [[nodiscard]] bool validateDispatch() {
        const NMMatterDispatchGPU& dispatch = world_.dispatch;
        if (dispatch.abiVersion != NM_MATTER_ABI_VERSION) {
            return fail("Matter dispatch ABI version is unsupported");
        }
        if ((dispatch.flags & ~kKnownMatterFlags) != 0u) {
            return fail("Matter dispatch contains unknown flags");
        }
        if (dispatch.environmentCount == 0u || dispatch.materialCount == 0u) {
            return fail("Matter dispatch requires environments and materials");
        }
        if (dispatch.materialCount != world_.materials.size() ||
            dispatch.parameterCount != world_.parameters.size() ||
            dispatch.stateInitialCount != world_.stateInitials.size() ||
            dispatch.objectCount != world_.objects.size() ||
            dispatch.particleCount != world_.mpm.particles.size() ||
            dispatch.gridNodeCount != world_.mpm.nodes.size() ||
            dispatch.mpmGridCount != world_.mpm.grids.size() ||
            dispatch.mpmBlockCount != world_.mpm.blocks.size() ||
            dispatch.mpmBlockLookupCount != world_.mpm.blockLookup.size() ||
            dispatch.femNodeCount != world_.fem.nodes.size() ||
            dispatch.tetrahedronCount != world_.fem.tetrahedra.size() ||
            dispatch.surfaceFaceCount != world_.fem.surfaceFaces.size() ||
            dispatch.mixedMaterialCount != world_.mixedMaterials.size() ||
            dispatch.fieldBoundaryCount != world_.fem.fieldBoundaries.size() ||
            dispatch.cohesiveFaceCount != world_.fem.cohesiveFaces.size() ||
            dispatch.mutationCommandCount != world_.fem.mutationCommands.size() ||
            dispatch.learnedMaterialCount != world_.learnedMaterials.size() ||
            dispatch.learnedLayerCount != world_.learnedLayers.size() ||
            dispatch.learnedWeightCount != world_.learnedWeights.size() ||
            dispatch.topologyNodeCapacity != world_.fem.topologyNodes.size() ||
            dispatch.punctureChannelCount != world_.fem.punctureChannels.size() ||
            dispatch.femCapacityCount != world_.fem.capacities.size() ||
            dispatch.femHumanAttachmentCount !=
                world_.fem.humanAttachments.size() ||
            dispatch.rigidProxyCount != world_.contact.rigidProxies.size() ||
            dispatch.contactPairCount != world_.contact.pairs.size()) {
            return fail("Matter dispatch counts disagree with cooked arenas");
        }
        const std::uint64_t expectedEventStride =
            static_cast<std::uint64_t>(NM_EVENT_CLASS_COUNT) *
            dispatch.objectCount;
        const bool attachmentJacobianStrideFits =
            detail::femHumanAttachmentPointJacobianStrideFits(
                dispatch.femHumanAttachmentCount,
                dispatch.rigidGeneralizedCapacity);
        const std::uint32_t expectedAttachmentJacobianStride =
            attachmentJacobianStrideFits
            ? detail::femHumanAttachmentPointJacobianStride(
                  dispatch.femHumanAttachmentCount,
                  dispatch.rigidGeneralizedCapacity)
            : 0u;
        if (dispatch.maximumRateExponent > NM_MAX_RATE_EXPONENT ||
            dispatch.maximumParticlesPerBlock !=
                NM_MPM_MAX_PARTICLES_PER_BLOCK ||
            dispatch.materialStateStride > NM_MAX_MATERIAL_STATE ||
            expectedEventStride >
                std::numeric_limits<std::uint32_t>::max() ||
            dispatch.eventStride != expectedEventStride ||
            dispatch.femHumanAttachmentCount >
                NM_MATTER_MAX_HUMAN_ATTACHMENT_POINTS ||
            !attachmentJacobianStrideFits ||
            dispatch.femHumanAttachmentPointJacobianStride !=
                expectedAttachmentJacobianStride) {
            return fail("Matter dispatch exceeds a fixed GPU capacity");
        }
        std::uint64_t expectedActiveMPMNodeCapacity = 0u;
        for (const NMContinuumObjectGPU& object : world_.objects) {
            if (object.representation != NM_REPRESENTATION_MPM) continue;
            expectedActiveMPMNodeCapacity += std::min<std::uint64_t>(
                object.auxiliaryCount,
                static_cast<std::uint64_t>(object.stateCount) *
                    NM_MPM_STENCIL_WIDTH);
        }
        if (expectedActiveMPMNodeCapacity > dispatch.gridNodeCount ||
            dispatch.mpmActiveNodeCapacity !=
                expectedActiveMPMNodeCapacity) {
            return fail(
                "active MPM Krylov capacity disagrees with particle support"
            );
        }
        if ((dispatch.identificationCandidateCount & 1u) != 0u ||
            dispatch.identificationCandidateCount >
                dispatch.environmentCount) {
            return fail("Matter identification candidates violate antithetic pairing");
        }
        if (!finite4(dispatch.gravityAndTimestep) ||
            !(dispatch.gravityAndTimestep.w > 0.0f) ||
            !finite4(dispatch.numericalLimits) ||
            !(dispatch.numericalLimits.x >= 0.0f) ||
            !(dispatch.numericalLimits.y > 0.0f) ||
            !(dispatch.numericalLimits.z > 0.0f) ||
            !(dispatch.numericalLimits.w > 0.0f)) {
            return fail("Matter dispatch contains invalid numerical limits");
        }
        const bool hasContact = !world_.contact.pairs.empty();
        const bool hasAdaptive = std::ranges::any_of(
            world_.objects,
            [](const NMContinuumObjectGPU& object) {
                return (object.flags & NM_OBJECT_ADAPTIVE) != 0u;
            }
        );
        if (((dispatch.flags & NM_MATTER_CONTACT) != 0u) != hasContact ||
            ((dispatch.flags & NM_MATTER_ADAPTIVE) != 0u) != hasAdaptive ||
            ((dispatch.flags & NM_MATTER_IDENTIFICATION) != 0u) !=
                !world_.identification.empty()) {
            return fail("Matter dispatch feature flags disagree with cooked programs");
        }
        return true;
    }

    [[nodiscard]] bool validateHumanAttachments() {
        std::vector<bool> claimedNodes(world_.fem.nodes.size(), false);
        std::set<std::uint32_t> stableIdentifiers;
        std::uint32_t previousNode = 0u;
        bool hasPreviousNode = false;
        for (std::size_t index = 0u;
             index < world_.fem.humanAttachments.size();
             ++index) {
            const NMFEMHumanAttachmentGPU& attachment =
                world_.fem.humanAttachments[index];
            const std::uint32_t node = attachment.identity.x;
            const std::uint32_t body = attachment.identity.y;
            const std::uint32_t object = attachment.identity.z;
            const std::uint32_t stableIdentifier = attachment.identity.w;
            if (node >= world_.fem.nodes.size() ||
                object >= world_.objects.size() ||
                femNodeOwners_[node] != object ||
                world_.objects[object].representation !=
                    NM_REPRESENTATION_FEM ||
                body == NM_INVALID_INDEX ||
                stableIdentifier == 0u ||
                stableIdentifier == NM_INVALID_INDEX ||
                claimedNodes[node] ||
                !stableIdentifiers.insert(stableIdentifier).second ||
                (hasPreviousNode && node <= previousNode) ||
                !finite4(attachment.localPoint) ||
                attachment.localPoint.w != 0.0f ||
                world_.fem.nodes[node].restAndFixed.w != 2.0f ||
                world_.fem.nodes[node].positionAndMass.w <= 0.0f ||
                world_.fem.nodes[node].velocityAndInverseMass.w <= 0.0f ||
                (world_.fem.topologyNodes[node].identity.w &
                    NM_TOPOLOGY_ACTIVE) == 0u) {
                return failIndexed(
                    "FEM Human attachment",
                    index,
                    "identity, local point, ordering, or node constraint is invalid"
                );
            }
            claimedNodes[node] = true;
            previousNode = node;
            hasPreviousNode = true;
        }
        for (std::size_t node = 0u; node < world_.fem.nodes.size(); ++node) {
            const float marker = world_.fem.nodes[node].restAndFixed.w;
            if ((marker != 0.0f && marker != 1.0f && marker != 2.0f) ||
                ((marker == 2.0f) != claimedNodes[node])) {
                return failIndexed(
                    "FEM node",
                    node,
                    "constraint marker disagrees with Human attachments"
                );
            }
        }
        return true;
    }

    [[nodiscard]] bool validateMixedAuthorities() {
        const NMMixedSolverGPU& solver = world_.mixedSolver;
        if (solver.nonlinearIterations.x == 0u ||
            solver.nonlinearIterations.y == 0u ||
            solver.nonlinearIterations.z < solver.nonlinearIterations.y ||
            solver.nonlinearIterations.w == 0u ||
            solver.executionBudgets.x == 0u ||
            solver.executionBudgets.x >
                NM_MIXED_FIELD_SMOOTHER_MAX_PASSES ||
            solver.executionBudgets.y == 0u ||
            solver.executionBudgets.z != 0u ||
            solver.executionBudgets.w != 0u ||
            !finite4(solver.residualTolerances) ||
            solver.residualTolerances.x < 0.0f ||
            solver.residualTolerances.y < 0.0f ||
            solver.residualTolerances.z < 0.0f ||
            solver.residualTolerances.w < 0.0f ||
            !finite4(solver.contactAcceptance) ||
            solver.contactAcceptance.x < 0.0f ||
            solver.contactAcceptance.x >= 1.0f ||
            solver.contactAcceptance.y != 0.0f ||
            solver.contactAcceptance.z != 0.0f ||
            solver.contactAcceptance.w != 0.0f ||
            !finite4(solver.regularization) ||
            !finite4(solver.globalization) ||
            !(solver.regularization.x > 0.0f) ||
            !(solver.regularization.y > 0.0f) ||
            !(solver.regularization.z >= solver.regularization.y) ||
            solver.regularization.w < 0.0f ||
            !(solver.globalization.x > 0.0f &&
              solver.globalization.x < 1.0f) ||
            !(solver.globalization.y > 0.0f) ||
            !(solver.globalization.z > 0.0f &&
              solver.globalization.z < 0.5f) ||
            solver.globalization.w < 0.0f ||
            solver.globalization.w > 1.0f) {
            return fail("mixed solver policy is invalid");
        }
        if (world_.mixedMaterials.size() != world_.materials.size() ||
            world_.fem.capacities.size() != world_.objects.size() ||
            world_.fem.fields.size() != world_.fem.nodes.size() ||
            world_.fem.topologyNodes.size() != world_.fem.nodes.size()) {
            return fail("mixed FEM arenas do not match their owning topology");
        }
        std::uint64_t tetrahedronCapacity = 0u;
        for (std::size_t index = 0u; index < world_.mixedMaterials.size(); ++index) {
            const NMMixedMaterialGPU material = world_.mixedMaterials[index];
            if (!finite4(material.mechanics) || !finite4(material.thermal) ||
                !finite4(material.porous) || !finite4(material.electrical) ||
                !finite4(material.fibre) || !finite4(material.coupling) ||
                !(material.mechanics.x > 0.0f) ||
                material.mechanics.y < 0.0f ||
                material.mechanics.z < 0.0f || material.mechanics.z > 1.0f ||
                material.thermal.x < 0.0f || material.thermal.y < 0.0f ||
                material.porous.x < 0.0f || material.porous.y < 0.0f ||
                material.electrical.x < 0.0f || material.electrical.y < 0.0f ||
                material.electrical.z < 0.0f || material.electrical.w < 0.0f) {
                return failIndexed("mixed material", index, "coefficients are invalid");
            }
        }
        for (const NMFEMCapacityGPU capacity : world_.fem.capacities) {
            tetrahedronCapacity += capacity.topology.y;
        }
        if (tetrahedronCapacity != world_.dispatch.topologyTetrahedronCapacity) {
            return fail("topology tetrahedron capacity disagrees with dispatch");
        }
        for (std::size_t node = 0u; node < world_.fem.fields.size(); ++node) {
            if (!finite4(world_.fem.fields[node].primary) ||
                !finite4(world_.fem.fields[node].secondary) ||
                !(world_.fem.fields[node].primary.y > 0.0f) ||
                world_.fem.fields[node].secondary.x < 0.0f ||
                world_.fem.fields[node].secondary.x > 1.0f) {
                return failIndexed("FEM field", node, "initial state is invalid");
            }
        }
        for (std::size_t learned = 0u;
             learned < world_.learnedMaterials.size();
             ++learned) {
            const NMLearnedMaterialGPU descriptor = world_.learnedMaterials[learned];
            if (!rangeWithin(descriptor.layout.x, descriptor.layout.y,
                    world_.learnedLayers.size()) ||
                !rangeWithin(descriptor.layout.z, descriptor.layout.w,
                    world_.learnedWeights.size()) ||
                descriptor.identity.y >= world_.materials.size() ||
                descriptor.identity.z != NM_LEARNED_SOFTPLUS ||
                descriptor.identity.x < 4u ||
                descriptor.identity.x > NM_LEARNED_MAX_INVARIANTS ||
                !finite4(descriptor.policy) || !(descriptor.policy.x > 0.0f) ||
                !(descriptor.policy.y > 0.0f) || descriptor.policy.z < 0.0f) {
                return failIndexed("learned material", learned, "layout is invalid");
            }
            std::uint32_t previousWidth = 0u;
            for (std::uint32_t local = 0u; local < descriptor.layout.y; ++local) {
                const NMLearnedLayerGPU layer =
                    world_.learnedLayers[descriptor.layout.x + local];
                const std::uint64_t inputCount =
                    static_cast<std::uint64_t>(layer.layout.x) * layer.layout.y;
                const std::uint64_t recurrentCount =
                    static_cast<std::uint64_t>(previousWidth) * layer.layout.y;
                if (layer.layout.x != descriptor.identity.x ||
                    layer.layout.y == 0u ||
                    layer.layout.y > NM_LEARNED_MAX_WIDTH ||
                    layer.layout.z < descriptor.layout.z ||
                    layer.routing.x != layer.layout.z + inputCount ||
                    layer.layout.w != layer.routing.x + recurrentCount ||
                    layer.layout.w + layer.layout.y >
                        descriptor.layout.z + descriptor.layout.w) {
                    return failIndexed("learned layer",
                        descriptor.layout.x + local, "weight routing is invalid");
                }
                for (std::uint64_t offset = 0u;
                     offset < inputCount + recurrentCount;
                     ++offset) {
                    const float value = world_.learnedWeights[
                        layer.layout.z + offset
                    ];
                    if (!std::isfinite(value) || value < 0.0f) {
                        return failIndexed("learned layer",
                            descriptor.layout.x + local,
                            "convex paths must be finite and nonnegative");
                    }
                }
                for (std::uint32_t bias = 0u; bias < layer.layout.y; ++bias) {
                    if (!std::isfinite(world_.learnedWeights[layer.layout.w + bias])) {
                        return failIndexed("learned layer",
                            descriptor.layout.x + local, "bias is not finite");
                    }
                }
                previousWidth = layer.layout.y;
            }
            if (previousWidth != 1u) {
                return failIndexed("learned material", learned,
                    "network output is not scalar");
            }
        }
        return true;
    }

    [[nodiscard]] bool validateProgram(
        const std::uint32_t programIndex,
        const std::size_t materialIndex
    ) {
        if (programIndex >= world_.scalarPrograms.size() ||
            materialIndex >= world_.materials.size()) {
            return fail("scalar program owner is invalid");
        }
        const NMScalarProgramGPU& program =
            world_.scalarPrograms[programIndex];
        const NMMaterialGPU& material = world_.materials[materialIndex];
        if (program.flags != 0u || program.instructionCount == 0u ||
            program.maximumStack == 0u ||
            program.maximumStack > NM_EXPRESSION_STACK_CAPACITY ||
            !rangeWithin(
                program.firstInstruction,
                program.instructionCount,
                world_.instructions.size()
            )) {
            return failIndexed(
                "scalar program",
                programIndex,
                "descriptor is outside the bytecode arena"
            );
        }
        std::uint32_t stackDepth = 0u;
        std::uint32_t maximumDepth = 0u;
        for (std::uint32_t local = 0u;
             local < program.instructionCount;
             ++local) {
            const std::size_t instructionIndex =
                static_cast<std::size_t>(program.firstInstruction) + local;
            const NMExpressionInstructionGPU& instruction =
                world_.instructions[instructionIndex];
            if (instruction.reserved != 0u ||
                !finite4(instruction.immediate) ||
                instruction.immediate.y != 0.0f ||
                instruction.immediate.z != 0.0f ||
                instruction.immediate.w != 0.0f) {
                return failIndexed(
                    "expression instruction",
                    instructionIndex,
                    "reserved or immediate fields are invalid"
                );
            }
            const auto requireStack = [&](const std::uint32_t count) {
                return stackDepth >= count;
            };
            switch (instruction.opcode) {
            case NM_EXPR_CONSTANT:
                ++stackDepth;
                break;
            case NM_EXPR_PARAMETER:
                if (instruction.index >= material.parameterCount) {
                    return failIndexed(
                        "expression instruction",
                        instructionIndex,
                        "parameter index exceeds its material range"
                    );
                }
                ++stackDepth;
                break;
            case NM_EXPR_STATE:
            case NM_EXPR_NEXT_STATE:
                if (instruction.index >= material.stateCount) {
                    return failIndexed(
                        "expression instruction",
                        instructionIndex,
                        "state index exceeds its material range"
                    );
                }
                ++stackDepth;
                break;
            case NM_EXPR_F:
            case NM_EXPR_DF:
            case NM_EXPR_RATE:
                if (instruction.index >= 9u) {
                    return failIndexed(
                        "expression instruction",
                        instructionIndex,
                        "tensor component is outside [0, 8]"
                    );
                }
                ++stackDepth;
                break;
            case NM_EXPR_DT:
            case NM_EXPR_TEMPERATURE:
                if (instruction.index != 0u) {
                    return failIndexed(
                        "expression instruction",
                        instructionIndex,
                        "scalar runtime input has a nonzero index"
                    );
                }
                ++stackDepth;
                break;
            case NM_EXPR_NEGATE:
            case NM_EXPR_LOG:
            case NM_EXPR_EXP:
            case NM_EXPR_EXPM1_MINUS_X:
            case NM_EXPR_SQRT:
            case NM_EXPR_ABS:
            case NM_EXPR_POW_INTEGER:
                if (!requireStack(1u)) {
                    return failIndexed(
                        "expression instruction",
                        instructionIndex,
                        "unary operator underflows the stack"
                    );
                }
                break;
            case NM_EXPR_ADD:
            case NM_EXPR_SUBTRACT:
            case NM_EXPR_MULTIPLY:
            case NM_EXPR_DIVIDE:
            case NM_EXPR_MIN:
            case NM_EXPR_MAX:
                if (!requireStack(2u)) {
                    return failIndexed(
                        "expression instruction",
                        instructionIndex,
                        "binary operator underflows the stack"
                    );
                }
                --stackDepth;
                break;
            case NM_EXPR_CLAMP:
                if (!requireStack(3u)) {
                    return failIndexed(
                        "expression instruction",
                        instructionIndex,
                        "clamp underflows the stack"
                    );
                }
                stackDepth -= 2u;
                break;
            default:
                return failIndexed(
                    "expression instruction",
                    instructionIndex,
                    "opcode is unknown"
                );
            }
            maximumDepth = std::max(maximumDepth, stackDepth);
            if (maximumDepth > NM_EXPRESSION_STACK_CAPACITY) {
                return failIndexed(
                    "scalar program",
                    programIndex,
                    "stack exceeds the shared ABI capacity"
                );
            }
        }
        if (stackDepth != 1u || maximumDepth != program.maximumStack) {
            return failIndexed(
                "scalar program",
                programIndex,
                "compiled stack contract is not canonical"
            );
        }
        return true;
    }

    [[nodiscard]] bool validateProgramsAndMaterials() {
        std::size_t expectedInstruction = 0u;
        for (std::size_t index = 0u;
             index < world_.scalarPrograms.size();
             ++index) {
            const NMScalarProgramGPU& program = world_.scalarPrograms[index];
            if (program.firstInstruction != expectedInstruction) {
                return failIndexed(
                    "scalar program",
                    index,
                    "instruction ranges overlap or contain a gap"
                );
            }
            expectedInstruction += program.instructionCount;
        }
        if (expectedInstruction != world_.instructions.size()) {
            return fail("scalar program ranges do not cover the instruction arena");
        }

        std::vector<std::size_t> owners(
            world_.scalarPrograms.size(),
            std::numeric_limits<std::size_t>::max()
        );
        std::size_t parameterCursor = 0u;
        std::size_t stateCursor = 0u;
        std::size_t programCursor = 0u;
        std::uint32_t maximumStateCount = 0u;
        const auto claim = [&] (
            const std::size_t materialIndex,
            const std::uint32_t first,
            const std::uint32_t count,
            const std::string_view role
        ) -> bool {
            if (first != programCursor ||
                !rangeWithin(first, count, world_.scalarPrograms.size())) {
                return failIndexed(
                    "material",
                    materialIndex,
                    std::string(role) + " program range is not canonical"
                );
            }
            for (std::uint32_t local = 0u; local < count; ++local) {
                const std::size_t index =
                    static_cast<std::size_t>(first) + local;
                if (owners[index] != std::numeric_limits<std::size_t>::max()) {
                    return failIndexed(
                        "scalar program",
                        index,
                        "program is aliased by multiple material roles"
                    );
                }
                owners[index] = materialIndex;
            }
            programCursor += count;
            return true;
        };

        for (std::size_t index = 0u; index < world_.materials.size(); ++index) {
            const NMMaterialGPU& material = world_.materials[index];
            if ((material.flags & ~kKnownMaterialFlags) != 0u ||
                material.constitutiveKind >
                    NM_CONSTITUTIVE_POLYCONVEX_ICNN ||
                material.projectionKind >
                    NM_MATERIAL_PROJECTION_DRUCKER_PRAGER ||
                material.localNewtonIterations > 16u) {
                return failIndexed(
                    "material",
                    index,
                    "kind, flags, projection, or reserved fields are invalid"
                );
            }
            if (material.parameterOffset != parameterCursor ||
                !rangeWithin(
                    material.parameterOffset,
                    material.parameterCount,
                    world_.parameters.size()
                )) {
                return failIndexed(
                    "material",
                    index,
                    "parameter arena is aliased or non-contiguous"
                );
            }
            parameterCursor += material.parameterCount;
            if (material.stateCount > NM_MAX_MATERIAL_STATE ||
                material.stateInitialOffset != stateCursor ||
                !rangeWithin(
                    material.stateInitialOffset,
                    material.stateCount,
                    world_.stateInitials.size()
                )) {
                return failIndexed(
                    "material",
                    index,
                    "initial-state arena is aliased or non-contiguous"
                );
            }
            stateCursor += material.stateCount;
            maximumStateCount = std::max(
                maximumStateCount,
                material.stateCount
            );
            const bool hasState = material.stateCount != 0u;
            const bool hasDissipation =
                (material.flags & NM_MATERIAL_HAS_DISSIPATION) != 0u;
            const bool hasImplicit =
                (material.flags & NM_MATERIAL_HAS_IMPLICIT_STATE) != 0u;
            for (std::uint32_t state = 0u; state < material.stateCount; ++state) {
                if (((material.stateTransferMask >> (2u * state)) & 3u) == 3u) {
                    return failIndexed("material", index,
                        "state transfer policy is invalid");
                }
            }
            if (((material.flags & NM_MATERIAL_HAS_STATE) != 0u) != hasState ||
                (hasState && material.stateUpdateProgramOffset ==
                    NM_INVALID_INDEX) ||
                (!hasState && material.stateUpdateProgramOffset !=
                    NM_INVALID_INDEX)) {
                return failIndexed(
                    "material",
                    index,
                    "state flags disagree with state program ownership"
                );
            }
            if (!claim(index, material.stressProgramOffset, 9u, "stress") ||
                !claim(index, material.tangentProgramOffset, 9u, "tangent")) {
                return false;
            }
            if (hasDissipation) {
                if (!claim(
                        index,
                        material.viscousStressProgramOffset,
                        9u,
                        "viscous stress"
                    ) ||
                    !claim(
                        index,
                        material.viscousTangentProgramOffset,
                        9u,
                        "viscous tangent"
                    )) {
                    return false;
                }
            } else if (material.viscousStressProgramOffset !=
                    NM_INVALID_INDEX ||
                material.viscousTangentProgramOffset != NM_INVALID_INDEX ||
                material.dissipationProgram != NM_INVALID_INDEX) {
                return failIndexed(
                    "material",
                    index,
                    "dissipation program ranges exist without a dissipation flag"
                );
            }
            if (hasState && !claim(
                    index,
                    material.stateUpdateProgramOffset,
                    material.stateCount,
                    "state update"
                )) {
                return false;
            }
            if (hasImplicit) {
                const std::uint32_t square =
                    material.stateCount * material.stateCount;
                if (!hasState || material.localNewtonIterations == 0u ||
                    !claim(index, material.implicitResidualProgramOffset,
                        material.stateCount, "implicit residual") ||
                    !claim(index, material.implicitJacobianProgramOffset,
                        square, "implicit Jacobian") ||
                    !claim(index, material.implicitDeformationProgramOffset,
                        material.stateCount, "implicit deformation action") ||
                    !claim(index, material.stressStateDerivativeProgramOffset,
                        9u * material.stateCount, "stress-state derivative")) {
                    return false;
                }
            } else if (material.implicitResidualProgramOffset != NM_INVALID_INDEX ||
                material.implicitJacobianProgramOffset != NM_INVALID_INDEX ||
                material.implicitDeformationProgramOffset != NM_INVALID_INDEX ||
                material.stressStateDerivativeProgramOffset != NM_INVALID_INDEX) {
                return failIndexed("material", index,
                    "implicit program ranges exist without an implicit-state flag");
            }
            if (hasDissipation) {
                if (!claim(
                        index,
                        material.dissipationProgram,
                        1u,
                        "dissipation"
                    )) {
                    return false;
                }
            }
            if (material.validityProgram != NM_INVALID_INDEX &&
                !claim(index, material.validityProgram, 1u, "validity")) {
                return false;
            }
            if (!finite4(material.bulk) ||
                !finite4(material.interfaceResponse) ||
                !finite4(material.inelastic) ||
                !finite4(material.validity) ||
                !(material.bulk.x > 0.0f) ||
                material.interfaceResponse.x < 0.0f ||
                material.interfaceResponse.y < 0.0f ||
                material.interfaceResponse.z < 0.0f ||
                material.interfaceResponse.z > 1.0f ||
                !(material.validity.x > 0.0f) ||
                !(material.validity.y > material.validity.x) ||
                !(material.validity.z > 0.0f) ||
                !(material.validity.w > 0.0f)) {
                return failIndexed(
                    "material",
                    index,
                    "physical limits are nonfinite or nonphysical"
                );
            }
            if (material.projectionKind != NM_MATERIAL_PROJECTION_GENERIC &&
                (material.stateCount != 10u || !(material.bulk.z > 0.0f) ||
                 !(material.bulk.w + 2.0f * material.bulk.z / 3.0f > 0.0f) ||
                 material.inelastic.x < 0.0f ||
                 material.inelastic.y < 0.0f ||
                 material.inelastic.z < 0.0f)) {
                return failIndexed(
                    "material", index,
                    "specialized plastic projection state or parameters are invalid"
                );
            }
        }
        if (parameterCursor != world_.parameters.size() ||
            stateCursor != world_.stateInitials.size() ||
            programCursor != world_.scalarPrograms.size() ||
            maximumStateCount != world_.dispatch.materialStateStride) {
            return fail("material arenas are not covered exactly once");
        }
        for (std::size_t index = 0u; index < owners.size(); ++index) {
            if (owners[index] == std::numeric_limits<std::size_t>::max() ||
                !validateProgram(
                    static_cast<std::uint32_t>(index),
                    owners[index]
                )) {
                return false;
            }
        }
        for (std::size_t index = 0u; index < world_.parameters.size(); ++index) {
            const nm_float4 value = world_.parameters[index].valueAndBounds;
            const bool logarithmic = value.w == 1.0f;
            if (!finite4(value) ||
                (value.w != 0.0f && !logarithmic) ||
                value.y > value.x || value.x > value.z ||
                (logarithmic && !(value.y > 0.0f))) {
                return failIndexed(
                    "parameter",
                    index,
                    "value, bounds, or logarithmic flag are invalid"
                );
            }
        }
        for (std::size_t index = 0u; index < world_.stateInitials.size(); ++index) {
            if (!std::isfinite(world_.stateInitials[index])) {
                return failIndexed(
                    "state initializer",
                    index,
                    "value is nonfinite"
                );
            }
        }
        return true;
    }

    [[nodiscard]] bool validateMPMGrid(
        const std::size_t gridIndex,
        const std::size_t objectIndex,
        std::size_t& blockCursor,
        std::size_t& lookupCursor
    ) {
        const NMMPMGridGPU& grid = world_.mpm.grids[gridIndex];
        const NMContinuumObjectGPU& object = world_.objects[objectIndex];
        if (grid.nodeMinimumAndObject.w !=
                static_cast<nm_i32>(objectIndex) ||
            grid.nodeDimensionsAndOffset.w != object.auxiliaryOffset ||
            grid.blockMinimumAndOffset.w < 0 ||
            static_cast<std::size_t>(grid.blockMinimumAndOffset.w) !=
                blockCursor ||
            grid.blockDimensionsAndLookup.w != lookupCursor ||
            !finite4(grid.metrics) || !(grid.metrics.x > 0.0f) ||
            !(grid.metrics.y > 0.0f) ||
            !nearlyEqual(grid.metrics.x * grid.metrics.y, 1.0f) ||
            !nearlyEqual(grid.metrics.z, 1.5f) ||
            grid.metrics.w != 0.0f) {
            return failIndexed(
                "MPM grid",
                gridIndex,
                "geometry, object binding, or sparse offsets are invalid"
            );
        }
        std::size_t nodeCount = 0u;
        std::size_t blockCount = 0u;
        if (!product3(
                grid.nodeDimensionsAndOffset.x,
                grid.nodeDimensionsAndOffset.y,
                grid.nodeDimensionsAndOffset.z,
                nodeCount
            ) ||
            !product3(
                grid.blockDimensionsAndLookup.x,
                grid.blockDimensionsAndLookup.y,
                grid.blockDimensionsAndLookup.z,
                blockCount
            ) ||
            nodeCount != object.auxiliaryCount ||
            !rangeWithin(
                grid.nodeDimensionsAndOffset.w,
                static_cast<std::uint32_t>(nodeCount),
                world_.mpm.nodes.size()
            ) ||
            blockCursor > world_.mpm.blocks.size() ||
            blockCursor >
                static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max()
                ) ||
            blockCount >
                static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max()
                ) ||
            blockCount > world_.mpm.blocks.size() - blockCursor ||
            lookupCursor > world_.mpm.blockLookup.size() ||
            blockCount > world_.mpm.blockLookup.size() - lookupCursor) {
            return failIndexed(
                "MPM grid",
                gridIndex,
                "dense dimensions exceed a cooked arena"
            );
        }
        const std::uint32_t dimX = grid.nodeDimensionsAndOffset.x;
        const std::uint32_t dimY = grid.nodeDimensionsAndOffset.y;
        for (std::size_t local = 0u; local < nodeCount; ++local) {
            const std::uint32_t x = static_cast<std::uint32_t>(local % dimX);
            const std::uint32_t yz = static_cast<std::uint32_t>(local / dimX);
            const std::uint32_t y = yz % dimY;
            const std::uint32_t z = yz / dimY;
            const NMGridNodeStateGPU& node = world_.mpm.nodes[
                static_cast<std::size_t>(object.auxiliaryOffset) + local
            ];
            const float expectedX = float(grid.nodeMinimumAndObject.x + int(x)) *
                grid.metrics.x;
            const float expectedY = float(grid.nodeMinimumAndObject.y + int(y)) *
                grid.metrics.x;
            const float expectedZ = float(grid.nodeMinimumAndObject.z + int(z)) *
                grid.metrics.x;
            if (!finite4(node.positionAndMass) ||
                !finite4(node.velocityAndInverseMass) ||
                !finite4(node.forceAndEnergy) ||
                !finite4(node.candidateVelocity) ||
                !nearlyEqual(node.positionAndMass.x, expectedX) ||
                !nearlyEqual(node.positionAndMass.y, expectedY) ||
                !nearlyEqual(node.positionAndMass.z, expectedZ)) {
                return failIndexed(
                    "MPM node",
                    static_cast<std::size_t>(object.auxiliaryOffset) + local,
                    "position or state is inconsistent with its grid"
                );
            }
        }
        std::vector<bool> seen(blockCount, false);
        const std::uint32_t blockDimX = grid.blockDimensionsAndLookup.x;
        const std::uint32_t blockDimY = grid.blockDimensionsAndLookup.y;
        for (std::size_t local = 0u; local < blockCount; ++local) {
            const std::uint32_t blockIndex =
                world_.mpm.blockLookup[lookupCursor + local];
            if (blockIndex < blockCursor ||
                blockIndex >= blockCursor + blockCount) {
                return failIndexed(
                    "MPM block lookup",
                    lookupCursor + local,
                    "entry points outside its grid block range"
                );
            }
            const std::size_t relative = blockIndex - blockCursor;
            if (seen[relative]) {
                return failIndexed(
                    "MPM block lookup",
                    lookupCursor + local,
                    "entry aliases an earlier block"
                );
            }
            seen[relative] = true;
            const NMMPMBlockGPU& block = world_.mpm.blocks[blockIndex];
            const std::uint32_t x = static_cast<std::uint32_t>(local % blockDimX);
            const std::uint32_t yz = static_cast<std::uint32_t>(local / blockDimX);
            const std::uint32_t y = yz % blockDimY;
            const std::uint32_t z = yz / blockDimY;
            if (block.identity.z != gridIndex ||
                block.identity.w != objectIndex ||
                block.coordinateAndLookup.w != static_cast<nm_i32>(local) ||
                block.coordinateAndLookup.x !=
                    grid.blockMinimumAndOffset.x + int(x) ||
                block.coordinateAndLookup.y !=
                    grid.blockMinimumAndOffset.y + int(y) ||
                block.coordinateAndLookup.z !=
                    grid.blockMinimumAndOffset.z + int(z)) {
                return failIndexed(
                    "MPM block",
                    blockIndex,
                    "identity, coordinate, or dense lookup is inconsistent"
                );
            }
        }
        blockCursor += blockCount;
        lookupCursor += blockCount;
        return true;
    }

    [[nodiscard]] bool validateIncidence(
        const std::span<const NMIncidenceRangeGPU> ranges,
        const std::span<const std::uint32_t> incidence,
        const std::span<const std::uint32_t> expectedOwners,
        const std::size_t sourceCapacity,
        const std::string_view role
    ) {
        if (ranges.size() != expectedOwners.size()) {
            return fail(std::string(role) + " range count is invalid");
        }
        std::size_t cursor = 0u;
        for (std::size_t owner = 0u; owner < ranges.size(); ++owner) {
            const NMIncidenceRangeGPU& range = ranges[owner];
            if (range.first != cursor || range.reserved != 0u ||
                range.objectIndex != expectedOwners[owner] ||
                !rangeWithin(range.first, range.count, incidence.size())) {
                return failIndexed(
                    role,
                    owner,
                    "range is non-contiguous or has the wrong owner"
                );
            }
            std::uint32_t previous = 0u;
            bool hasPrevious = false;
            for (std::uint32_t local = 0u; local < range.count; ++local) {
                const std::uint32_t source =
                    incidence[static_cast<std::size_t>(range.first) + local];
                if (source >= sourceCapacity ||
                    (hasPrevious && source <= previous)) {
                    return failIndexed(
                        role,
                        owner,
                        "source indices are invalid or not strictly ordered"
                    );
                }
                previous = source;
                hasPrevious = true;
            }
            cursor += range.count;
        }
        if (cursor != incidence.size()) {
            return fail(std::string(role) + " ranges do not cover incidence");
        }
        return true;
    }

    [[nodiscard]] bool validateObjectsAndTopology() {
        std::size_t particleCursor = 0u;
        std::size_t gridCursor = 0u;
        std::size_t nodeCursor = 0u;
        std::size_t blockCursor = 0u;
        std::size_t lookupCursor = 0u;
        std::size_t femNodeCursor = 0u;
        std::size_t tetrahedronCursor = 0u;
        mpmNodeOwners_.assign(world_.mpm.nodes.size(), NM_INVALID_INDEX);
        femNodeOwners_.assign(world_.fem.nodes.size(), NM_INVALID_INDEX);

        for (std::size_t index = 0u; index < world_.objects.size(); ++index) {
            const NMContinuumObjectGPU& object = world_.objects[index];
            if (object.materialIndex >= world_.materials.size() ||
                (object.flags & ~kKnownObjectFlags) != 0u ||
                (object.flags & NM_OBJECT_ACTIVE) == 0u ||
                object.schedulerIndex != index ||
                object.topologyGeneration == 0u ||
                object.solver.x > object.solver.y ||
                object.solver.y > world_.dispatch.maximumRateExponent ||
                object.solver.z != 0u ||
                object.solver.w != 0u ||
                !finite4(object.fidelity) ||
                !(object.fidelity.x > 0.0f) ||
                object.fidelity.y < 0.0f ||
                object.fidelity.z < 0.0f ||
                object.fidelity.w < 0.0f ||
                object.representation > NM_REPRESENTATION_FEM) {
                return failIndexed(
                    "continuum object",
                    index,
                    "descriptor or solver contract is invalid"
                );
            }
            if ((object.flags & NM_OBJECT_ADAPTIVE) != 0u) {
                if (object.rigidBinding >= world_.contact.rigidProxies.size()) {
                    return failIndexed(
                        "continuum object",
                        index,
                        "adaptive object has no rigid binding"
                    );
                }
            } else if (object.rigidBinding != NM_INVALID_INDEX &&
                object.rigidBinding >= world_.contact.rigidProxies.size()) {
                return failIndexed(
                    "continuum object",
                    index,
                    "rigid binding is outside the proxy arena"
                );
            }

            if (object.representation == NM_REPRESENTATION_MPM) {
                if (object.stateOffset != particleCursor ||
                    object.elementOffset != gridCursor ||
                    object.elementCount != 1u ||
                    object.auxiliaryOffset != nodeCursor ||
                    object.stateCount == 0u ||
                    !rangeWithin(
                        object.stateOffset,
                        object.stateCount,
                        world_.mpm.particles.size()
                    ) ||
                    !rangeWithin(
                        object.auxiliaryOffset,
                        object.auxiliaryCount,
                        world_.mpm.nodes.size()
                    )) {
                    return failIndexed(
                        "continuum object",
                        index,
                        "MPM state, grid, or node range is not canonical"
                    );
                }
                if (!validateMPMGrid(
                        gridCursor,
                        index,
                        blockCursor,
                        lookupCursor
                    )) {
                    return false;
                }
                for (std::uint32_t local = 0u;
                     local < object.stateCount;
                     ++local) {
                    const std::size_t particleIndex =
                        static_cast<std::size_t>(object.stateOffset) + local;
                    const NMParticleStateGPU& particle =
                        world_.mpm.particles[particleIndex];
                    if (!finite4(particle.positionAndMass) ||
                        !finite4(particle.velocityAndReferenceVolume) ||
                        !finite4(particle.deformationRow0) ||
                        !finite4(particle.deformationRow1) ||
                        !finite4(particle.deformationRow2) ||
                        !finite4(particle.affineRow0) ||
                        !finite4(particle.affineRow1) ||
                        !finite4(particle.affineRow2) ||
                        !finite4(particle.referenceAndTemperature) ||
                        !(particle.positionAndMass.w > 0.0f) ||
                        !(particle.velocityAndReferenceVolume.w > 0.0f) ||
                        !(particle.referenceAndTemperature.w > 0.0f) ||
                        particle.identity.x != index ||
                        particle.identity.y != object.materialIndex ||
                        particle.identity.z != object.topologyGeneration ||
                        (particle.identity.w & NM_OBJECT_ACTIVE) == 0u) {
                        return failIndexed(
                            "MPM particle",
                            particleIndex,
                            "state or object identity is invalid"
                        );
                    }
                }
                for (std::uint32_t local = 0u;
                     local < object.auxiliaryCount;
                     ++local) {
                    mpmNodeOwners_[
                        static_cast<std::size_t>(object.auxiliaryOffset) + local
                    ] = static_cast<std::uint32_t>(index);
                }
                particleCursor += object.stateCount;
                ++gridCursor;
                nodeCursor += object.auxiliaryCount;
            } else if (object.representation == NM_REPRESENTATION_FEM) {
                if (object.stateOffset != femNodeCursor ||
                    object.elementOffset != tetrahedronCursor ||
                    object.stateCount == 0u || object.elementCount == 0u ||
                    object.auxiliaryOffset != 0u ||
                    object.auxiliaryCount != 0u ||
                    !rangeWithin(
                        object.stateOffset,
                        object.stateCount,
                        world_.fem.nodes.size()
                    ) ||
                    !rangeWithin(
                        object.elementOffset,
                        object.elementCount,
                        world_.fem.tetrahedra.size()
                    )) {
                    return failIndexed(
                        "continuum object",
                        index,
                        "FEM node or element range is not canonical"
                    );
                }
                for (std::uint32_t local = 0u;
                     local < object.stateCount;
                     ++local) {
                    const std::size_t nodeIndex =
                        static_cast<std::size_t>(object.stateOffset) + local;
                    const NMFEMNodeStateGPU& node = world_.fem.nodes[nodeIndex];
                    if (!finite4(node.positionAndMass) ||
                        !finite4(node.velocityAndInverseMass) ||
                        !finite4(node.restAndFixed) ||
                        !finite4(node.deltaVelocity) ||
                        node.positionAndMass.w < 0.0f ||
                        node.velocityAndInverseMass.w < 0.0f ||
                        (node.positionAndMass.w > 0.0f &&
                         node.velocityAndInverseMass.w > 0.0f &&
                         !nearlyEqual(
                             node.positionAndMass.w *
                                 node.velocityAndInverseMass.w,
                             1.0f,
                             256.0f
                         ))) {
                        return failIndexed(
                            "FEM node",
                            nodeIndex,
                            "state or lumped mass is invalid"
                        );
                    }
                    femNodeOwners_[nodeIndex] =
                        static_cast<std::uint32_t>(index);
                }
                for (std::uint32_t local = 0u;
                     local < object.elementCount;
                     ++local) {
                    const std::size_t tetrahedronIndex =
                        static_cast<std::size_t>(object.elementOffset) + local;
                    const NMTetrahedronGPU& tetrahedron =
                        world_.fem.tetrahedra[tetrahedronIndex];
                    const std::array<std::uint32_t, 4> nodes{
                        tetrahedron.nodes.x,
                        tetrahedron.nodes.y,
                        tetrahedron.nodes.z,
                        tetrahedron.nodes.w,
                    };
                    if (tetrahedron.identity.x != object.materialIndex ||
                        tetrahedron.identity.y != index ||
                        tetrahedron.identity.z != object.topologyGeneration ||
                        !finite4(tetrahedron.inverseRestRow0) ||
                        !finite4(tetrahedron.inverseRestRow1) ||
                        !finite4(tetrahedron.inverseRestRow2)) {
                        return failIndexed(
                            "FEM tetrahedron",
                            tetrahedronIndex,
                            "identity or rest operator is invalid"
                        );
                    }
                    const bool active =
                        (tetrahedron.identity.w & NM_OBJECT_ACTIVE) != 0u;
                    if (!active) {
                        // Mutable FEM objects cook canonical dormant arena
                        // slots. They acquire nodes/rest operators only inside
                        // an accepted topology transaction, so treating them
                        // as authored elements makes every growable world
                        // impossible to validate or package.
                        if (tetrahedron.identity.w != 0u ||
                            tetrahedron.inverseRestRow0.w != 0.0f ||
                            tetrahedron.inverseRestRow1.w != 0.0f ||
                            tetrahedron.inverseRestRow2.w != 0.0f) {
                            return failIndexed(
                                "FEM tetrahedron",
                                tetrahedronIndex,
                                "dormant arena slot is not canonical"
                            );
                        }
                        continue;
                    }
                    if (!(tetrahedron.inverseRestRow0.w > 0.0f)) {
                        return failIndexed(
                            "FEM tetrahedron",
                            tetrahedronIndex,
                            "active rest operator has nonpositive volume"
                        );
                    }
                    const std::uint64_t nodeEnd =
                        static_cast<std::uint64_t>(object.stateOffset) +
                        object.stateCount;
                    for (std::size_t node = 0u; node < nodes.size(); ++node) {
                        if (nodes[node] < object.stateOffset ||
                            static_cast<std::uint64_t>(nodes[node]) >= nodeEnd ||
                            std::count(nodes.begin(), nodes.end(), nodes[node]) !=
                                1) {
                            return failIndexed(
                                "FEM tetrahedron",
                                tetrahedronIndex,
                                "node indices are outside the object or repeated"
                            );
                        }
                    }
                }
                femNodeCursor += object.stateCount;
                tetrahedronCursor += object.elementCount;
            } else {
                if (object.stateOffset != 0u || object.stateCount != 0u ||
                    object.elementOffset != 0u || object.elementCount != 0u ||
                    object.auxiliaryOffset != 0u ||
                    object.auxiliaryCount != 0u) {
                    return failIndexed(
                        "continuum object",
                        index,
                        "rigid representation owns continuum arena entries"
                    );
                }
            }
        }
        if (particleCursor != world_.mpm.particles.size() ||
            gridCursor != world_.mpm.grids.size() ||
            nodeCursor != world_.mpm.nodes.size() ||
            blockCursor != world_.mpm.blocks.size() ||
            lookupCursor != world_.mpm.blockLookup.size() ||
            femNodeCursor != world_.fem.nodes.size() ||
            tetrahedronCursor != world_.fem.tetrahedra.size()) {
            return fail("continuum object ranges do not cover cooked topology");
        }

        if (!validateIncidence(
                world_.mpm.nodeRanges,
                world_.mpm.nodeIncidence,
                mpmNodeOwners_,
                world_.mpm.stencils.size(),
                "MPM node incidence"
            ) ||
            !validateIncidence(
                world_.fem.nodeRanges,
                world_.fem.nodeIncidence,
                femNodeOwners_,
                world_.fem.tetrahedra.size(),
                "FEM node incidence"
            )) {
            return false;
        }
        for (std::size_t node = 0u; node < world_.fem.nodeRanges.size(); ++node) {
            const NMIncidenceRangeGPU& range = world_.fem.nodeRanges[node];
            for (std::uint32_t local = 0u; local < range.count; ++local) {
                const std::uint32_t tetrahedronIndex =
                    world_.fem.nodeIncidence[
                        static_cast<std::size_t>(range.first) + local
                    ];
                const NMTetrahedronGPU& tetrahedron =
                    world_.fem.tetrahedra[tetrahedronIndex];
                if (tetrahedron.nodes.x != node &&
                    tetrahedron.nodes.y != node &&
                    tetrahedron.nodes.z != node &&
                    tetrahedron.nodes.w != node) {
                    return failIndexed(
                        "FEM node incidence",
                        node,
                        "tetrahedron does not contain its owner node"
                    );
                }
            }
        }
        for (std::size_t faceIndex = 0u;
             faceIndex < world_.fem.surfaceFaces.size(); ++faceIndex) {
            const NMFEMSurfaceFaceGPU& face =
                world_.fem.surfaceFaces[faceIndex];
            if (face.adjacency.x >= world_.fem.tetrahedra.size() ||
                face.adjacency.z >= world_.objects.size() ||
                face.sides.x >= 4u || face.sides.w != 0u ||
                (face.adjacency.w & NM_TOPOLOGY_ACTIVE) == 0u ||
                (face.adjacency.y != NM_INVALID_INDEX &&
                 (face.adjacency.y >= world_.fem.tetrahedra.size() ||
                  face.sides.y >= 4u))) {
                return failIndexed(
                    "FEM surface face", faceIndex,
                    "adjacency or side descriptor is invalid");
            }
            const NMTetrahedronGPU& first =
                world_.fem.tetrahedra[face.adjacency.x];
            if (first.identity.y != face.adjacency.z ||
                (face.adjacency.y != NM_INVALID_INDEX &&
                 world_.fem.tetrahedra[face.adjacency.y].identity.y !=
                    face.adjacency.z)) {
                return failIndexed(
                    "FEM surface face", faceIndex,
                    "adjacent tetrahedron belongs to another object");
            }
        }
        for (std::size_t stencil = 0u;
             stencil < world_.mpm.stencils.size();
             ++stencil) {
            const NMMPMStencilGPU& value = world_.mpm.stencils[stencil];
            if (value.particleIndex >= world_.mpm.particles.size() ||
                value.nodeIndex >= world_.mpm.nodes.size() ||
                value.localSlot >= NM_MPM_STENCIL_WIDTH ||
                value.reserved != 0u ||
                !finite4(value.gradientAndWeight) ||
                mpmNodeOwners_[value.nodeIndex] !=
                    world_.mpm.particles[value.particleIndex].identity.x) {
                return failIndexed(
                    "MPM stencil",
                    stencil,
                    "binding or weight is invalid"
                );
            }
        }
        return true;
    }

    [[nodiscard]] bool validateContact() {
        const std::size_t unifiedNodeCount =
            world_.mpm.nodes.size() + world_.fem.nodes.size();
        std::vector<std::uint32_t> unifiedOwners;
        unifiedOwners.reserve(unifiedNodeCount);
        unifiedOwners.insert(
            unifiedOwners.end(),
            mpmNodeOwners_.begin(),
            mpmNodeOwners_.end()
        );
        unifiedOwners.insert(
            unifiedOwners.end(),
            femNodeOwners_.begin(),
            femNodeOwners_.end()
        );
        std::vector<std::uint32_t> rigidOwners(
            world_.contact.rigidProxies.size(),
            NM_INVALID_INDEX
        );
        std::map<std::uint32_t, std::uint32_t> freeBodyIndices;
        std::set<std::uint32_t> claimedFreeBodyIndices;
        std::uint64_t articulatedProxyCount = 0u;
        for (std::size_t index = 0u;
             index < world_.contact.rigidProxies.size();
             ++index) {
            const NMRigidProxyGPU& proxy = world_.contact.rigidProxies[index];
            const bool articulated =
                (proxy.flags & NM_RIGID_ARTICULATED) != 0u;
            articulatedProxyCount += articulated ? 1u : 0u;
            const bool dynamic = (proxy.flags & NM_RIGID_DYNAMIC) != 0u;
            const bool punctureTip =
                (proxy.flags & NM_RIGID_PUNCTURE_TIP) != 0u;
            const bool strand =
                (proxy.flags & NM_RIGID_SUTURE_STRAND) != 0u;
            const bool punctureDilator =
                (proxy.flags & NM_RIGID_PUNCTURE_DILATOR) != 0u;
            const float capsuleDx =
                proxy.localExtent.x - proxy.localCenterAndRadius.x;
            const float capsuleDy =
                proxy.localExtent.y - proxy.localCenterAndRadius.y;
            const float capsuleDz =
                proxy.localExtent.z - proxy.localCenterAndRadius.z;
            const float capsuleLengthSquared =
                capsuleDx * capsuleDx + capsuleDy * capsuleDy +
                capsuleDz * capsuleDz;
            const float orientationNorm =
                proxy.localOrientation.x * proxy.localOrientation.x +
                proxy.localOrientation.y * proxy.localOrientation.y +
                proxy.localOrientation.z * proxy.localOrientation.z +
                proxy.localOrientation.w * proxy.localOrientation.w;
            if (proxy.shapeKind > NM_RIGID_ARC ||
                proxy.materialIndex >= world_.materials.size() ||
                (proxy.flags & ~kKnownRigidFlags) != 0u ||
                (articulated && dynamic) ||
                (strand &&
                    (articulated || dynamic || punctureTip ||
                     punctureDilator ||
                     proxy.shapeKind != NM_RIGID_CAPSULE ||
                     proxy.bodyIndex == NM_INVALID_INDEX ||
                     proxy.sceneBodyIndex == NM_INVALID_INDEX ||
                     proxy.bodyIndex == proxy.sceneBodyIndex ||
                     proxy.adaptiveObjectIndex != NM_INVALID_INDEX)) ||
                (punctureDilator &&
                    (proxy.bodyIndex == NM_INVALID_INDEX ||
                     (proxy.shapeKind != NM_RIGID_CAPSULE &&
                      proxy.shapeKind != NM_RIGID_ARC))) ||
                (punctureTip &&
                    (proxy.shapeKind != NM_RIGID_CAPSULE ||
                     proxy.bodyIndex == NM_INVALID_INDEX ||
                     !(proxy.localCenterAndRadius.w > 0.0f) ||
                     !(capsuleLengthSquared > 1.0e-18f))) ||
                ((articulated || dynamic) &&
                    proxy.bodyIndex == NM_INVALID_INDEX) ||
                (dynamic && proxy.sceneBodyIndex == NM_INVALID_INDEX) ||
                (!strand && !dynamic &&
                    proxy.sceneBodyIndex != NM_INVALID_INDEX) ||
                !finite4(proxy.localCenterAndRadius) ||
                !finite4(proxy.localExtent) ||
                !finite4(proxy.localOrientation) ||
                !(orientationNorm > 1.0e-12f) ||
                (dynamic &&
                    proxy.generalizedFreeBodyIndex == NM_INVALID_INDEX) ||
                (!dynamic &&
                    proxy.generalizedFreeBodyIndex != NM_INVALID_INDEX) ||
                proxy.reserved2 != 0u) {
                return failIndexed(
                    "rigid proxy",
                    index,
                    "shape, binding, material, or geometry is invalid"
                );
            }
            if (dynamic) {
                const auto [owner, inserted] = freeBodyIndices.emplace(
                    proxy.bodyIndex, proxy.generalizedFreeBodyIndex);
                if ((!inserted && owner->second !=
                        proxy.generalizedFreeBodyIndex) ||
                    (inserted && !claimedFreeBodyIndices.insert(
                        proxy.generalizedFreeBodyIndex).second)) {
                    return failIndexed(
                        "rigid proxy",
                        index,
                        "free-body generalized ownership is not canonical"
                    );
                }
            }
            if (proxy.shapeKind == NM_RIGID_PLANE) {
                const float normalNorm =
                    proxy.localCenterAndRadius.x *
                        proxy.localCenterAndRadius.x +
                    proxy.localCenterAndRadius.y *
                        proxy.localCenterAndRadius.y +
                    proxy.localCenterAndRadius.z *
                        proxy.localCenterAndRadius.z;
                if (!(normalNorm > 1.0e-12f)) {
                    return failIndexed(
                        "rigid proxy",
                        index,
                        "plane normal is degenerate"
                    );
                }
            } else if (!(proxy.localCenterAndRadius.w > 0.0f)) {
                return failIndexed(
                    "rigid proxy",
                    index,
                    "curved shape radius is nonpositive"
                );
            }
            if (proxy.shapeKind == NM_RIGID_BOX &&
                (!(proxy.localExtent.x > 0.0f) ||
                 !(proxy.localExtent.y > 0.0f) ||
                 !(proxy.localExtent.z > 0.0f))) {
                return failIndexed(
                    "rigid proxy",
                    index,
                    "box half extent is nonpositive"
                );
            }
            if (proxy.shapeKind == NM_RIGID_ARC &&
                (!(proxy.localExtent.x > 0.0f) ||
                 !(proxy.localExtent.z > 0.0f) ||
                 proxy.localExtent.z > 2.0f * std::numbers::pi_v<float>)) {
                return failIndexed(
                    "rigid proxy",
                    index,
                    "arc centreline radius or angular sweep is invalid"
                );
            }
            if (proxy.adaptiveObjectIndex != NM_INVALID_INDEX) {
                if (proxy.adaptiveObjectIndex >= world_.objects.size() ||
                    (world_.objects[proxy.adaptiveObjectIndex].flags &
                        NM_OBJECT_ADAPTIVE) == 0u) {
                    return failIndexed(
                        "rigid proxy",
                        index,
                        "adaptive owner is invalid"
                    );
                }
            }
        }
        for (std::uint32_t index = 0u;
             index < claimedFreeBodyIndices.size(); ++index) {
            if (!claimedFreeBodyIndices.contains(index))
                return fail("free-body generalized indices are not compact");
        }
        const bool hasArticulatedCandidateAuthority =
            articulatedProxyCount != 0u ||
            !world_.fem.humanAttachments.empty();
        const std::uint64_t freeCapacity =
            static_cast<std::uint64_t>(freeBodyIndices.size()) * 6u;
        if (world_.dispatch.rigidGeneralizedCapacity < freeCapacity) {
            return fail("rigid generalized capacities disagree with coupled candidate ownership");
        }
        const std::uint64_t articulatedCapacity =
            world_.dispatch.rigidGeneralizedCapacity - freeCapacity;
        if ((!hasArticulatedCandidateAuthority &&
             (articulatedCapacity != 0u ||
              world_.dispatch.rigidQCapacity != 0u)) ||
            (hasArticulatedCandidateAuthority &&
             (articulatedCapacity == 0u ||
              articulatedCapacity > NM_MATTER_MAX_ARTICULATED_DOFS ||
              world_.dispatch.rigidQCapacity == 0u ||
              world_.dispatch.rigidQCapacity >
                  NM_MATTER_MAX_ARTICULATED_Q))) {
            return fail("rigid generalized capacities disagree with coupled candidate ownership");
        }

        std::vector<std::uint32_t> pairNodeCounts(
            world_.contact.pairs.size(), 0u
        );
        std::vector<std::uint32_t> pairRigidCounts(
            world_.contact.pairs.size(), 0u
        );
        for (std::size_t index = 0u;
             index < world_.contact.pairs.size();
             ++index) {
            const NMContactPairGPU& pair = world_.contact.pairs[index];
            if (pair.continuumNode >= unifiedNodeCount ||
                pair.rigidProxy >= world_.contact.rigidProxies.size() ||
                pair.objectIndex >= world_.objects.size() ||
                pair.materialInterface >= world_.materials.size() ||
                unifiedOwners[pair.continuumNode] != pair.objectIndex ||
                world_.contact.rigidProxies[pair.rigidProxy].materialIndex !=
                    pair.materialInterface ||
                world_.contact.rigidProxies[pair.rigidProxy]
                        .adaptiveObjectIndex == pair.objectIndex) {
                return failIndexed(
                    "contact pair",
                    index,
                    "node, proxy, object, or material binding is invalid"
                );
            }
        }
        if (!validateIncidence(
                world_.contact.nodeRanges,
                world_.contact.nodeIncidence,
                unifiedOwners,
                world_.contact.pairs.size(),
                "contact node incidence"
            ) ||
            !validateIncidence(
                world_.contact.rigidRanges,
                world_.contact.rigidIncidence,
                rigidOwners,
                world_.contact.pairs.size(),
                "contact rigid incidence"
            )) {
            return false;
        }
        for (std::size_t node = 0u;
             node < world_.contact.nodeRanges.size();
             ++node) {
            const NMIncidenceRangeGPU& range =
                world_.contact.nodeRanges[node];
            for (std::uint32_t local = 0u; local < range.count; ++local) {
                const std::uint32_t pairIndex =
                    world_.contact.nodeIncidence[
                        static_cast<std::size_t>(range.first) + local
                    ];
                if (world_.contact.pairs[pairIndex].continuumNode != node) {
                    return failIndexed(
                        "contact node incidence",
                        node,
                        "pair does not reference its owner node"
                    );
                }
                ++pairNodeCounts[pairIndex];
            }
        }
        for (std::size_t proxy = 0u;
             proxy < world_.contact.rigidRanges.size();
             ++proxy) {
            const NMIncidenceRangeGPU& range =
                world_.contact.rigidRanges[proxy];
            for (std::uint32_t local = 0u; local < range.count; ++local) {
                const std::uint32_t pairIndex =
                    world_.contact.rigidIncidence[
                        static_cast<std::size_t>(range.first) + local
                    ];
                if (world_.contact.pairs[pairIndex].rigidProxy != proxy) {
                    return failIndexed(
                        "contact rigid incidence",
                        proxy,
                        "pair does not reference its owner proxy"
                    );
                }
                ++pairRigidCounts[pairIndex];
            }
        }
        for (std::size_t pair = 0u; pair < pairNodeCounts.size(); ++pair) {
            if (pairNodeCounts[pair] != 1u || pairRigidCounts[pair] != 1u) {
                return failIndexed(
                    "contact pair",
                    pair,
                    "pair is not owned exactly once by both incidence tables"
                );
            }
        }
        return true;
    }

    [[nodiscard]] bool validateAdaptiveAndSchedulers() {
        if (world_.adaptive.size() != world_.objects.size() ||
            world_.schedulers.size() != world_.objects.size()) {
            return fail("adaptive or scheduler arena does not match object count");
        }
        std::vector<bool> adaptiveProxySeen(
            world_.contact.rigidProxies.size(), false
        );
        for (std::size_t index = 0u; index < world_.objects.size(); ++index) {
            const NMContinuumObjectGPU& object = world_.objects[index];
            const NMAdaptiveStateGPU& adaptive = world_.adaptive[index];
            const NMSchedulerStateGPU& scheduler = world_.schedulers[index];
            if (adaptive.activeRepresentation > NM_REPRESENTATION_FEM ||
                adaptive.requestedRepresentation > NM_REPRESENTATION_FEM ||
                adaptive.activeRepresentation != object.representation ||
                adaptive.requestedRepresentation != object.representation ||
                adaptive.stableFrames != 0u || adaptive.flags != 0u ||
                !finite4(adaptive.massAndError) ||
                !finite4(adaptive.centerAndRadius) ||
                !finite4(adaptive.referenceCenter) ||
                !finite4(adaptive.linearVelocityAndAngularSpeed) ||
                !finite4(adaptive.angularVelocityAndMinimumJ) ||
                !finite4(adaptive.orientation) ||
                !finite4(adaptive.inverseInertiaRow0) ||
                !finite4(adaptive.inverseInertiaRow1) ||
                !finite4(adaptive.inverseInertiaRow2)) {
                return failIndexed(
                    "adaptive state",
                    index,
                    "initial representation or physical state is invalid"
                );
            }
            if ((object.flags & NM_OBJECT_ADAPTIVE) != 0u) {
                if (object.rigidBinding >= adaptiveProxySeen.size() ||
                    adaptiveProxySeen[object.rigidBinding]) {
                    return failIndexed(
                        "continuum object",
                        index,
                        "adaptive rigid binding is absent or aliased"
                    );
                }
                adaptiveProxySeen[object.rigidBinding] = true;
                const NMRigidProxyGPU& proxy =
                    world_.contact.rigidProxies[object.rigidBinding];
                if ((proxy.flags & NM_RIGID_DYNAMIC) == 0u ||
                    (proxy.flags & NM_RIGID_ARTICULATED) != 0u ||
                    proxy.adaptiveObjectIndex != index) {
                    return failIndexed(
                        "continuum object",
                        index,
                        "adaptive fallback is not a uniquely owned free body"
                    );
                }
            }
            if (scheduler.baseExponent != object.solver.x ||
                scheduler.activeExponent != object.solver.x ||
                scheduler.requestedExponent != object.solver.x ||
                scheduler.quietFrames != 0u ||
                !finite4(scheduler.physical) ||
                !finite4(scheduler.numerical) ||
                !finite4(scheduler.thresholds) ||
                !finite4(scheduler.timing) ||
                scheduler.thresholds.x < 0.0f ||
                scheduler.thresholds.y < 0.0f ||
                scheduler.thresholds.z < 0.0f ||
                scheduler.thresholds.w < 0.0f ||
                scheduler.numerical.w !=
                    (object.representation == NM_REPRESENTATION_RIGID
                        ? 0.0f : 1.0f)) {
                return failIndexed(
                    "scheduler state",
                    index,
                    "initial cadence or thresholds are invalid"
                );
            }
        }
        return true;
    }

    [[nodiscard]] bool validateIdentification() {
        std::vector<bool> parameterSeen(world_.parameters.size(), false);
        for (std::size_t index = 0u;
             index < world_.identification.size();
             ++index) {
            const NMIdentificationDistributionGPU& distribution =
                world_.identification[index];
            const std::uint32_t materialIndex = distribution.identity.x;
            const std::uint32_t localParameter = distribution.identity.y;
            const std::uint32_t globalParameter = distribution.identity.z;
            if (materialIndex >= world_.materials.size() ||
                localParameter >=
                    world_.materials[materialIndex].parameterCount ||
                globalParameter !=
                    world_.materials[materialIndex].parameterOffset +
                        localParameter ||
                globalParameter >= world_.parameters.size() ||
                distribution.identity.w > 1u ||
                parameterSeen[globalParameter] ||
                !finite4(distribution.momentsAndBounds) ||
                !finite4(distribution.update) ||
                !(distribution.momentsAndBounds.y > 0.0f) ||
                distribution.momentsAndBounds.z >
                    distribution.momentsAndBounds.x ||
                distribution.momentsAndBounds.x >
                    distribution.momentsAndBounds.w ||
                distribution.update.x < 0.0f ||
                distribution.update.x > 1.0f ||
                !(distribution.update.y > 0.0f) ||
                !(distribution.update.z > 0.0f) ||
                (distribution.update.w != 0.0f &&
                 distribution.update.w != 1.0f) ||
                distribution.update.w !=
                    world_.parameters[globalParameter].valueAndBounds.w ||
                distribution.identity.w !=
                    static_cast<std::uint32_t>(distribution.update.w)) {
                return failIndexed(
                    "identification distribution",
                    index,
                    "parameter ownership or moments are invalid"
                );
            }
            parameterSeen[globalParameter] = true;
        }
        return true;
    }

    [[nodiscard]] bool validateFingerprint() {
        if (world_.physicsFingerprint == 0u) {
            return fail("Matter source-physics fingerprint is missing");
        }
        if (world_.fingerprint == 0u ||
            world_.fingerprint != compiledWorldFingerprint(world_)) {
            return fail("Matter world fingerprint does not match canonical sections");
        }
        return true;
    }

    const CompiledWorld& world_;
    std::string* error_ = nullptr;
    std::vector<std::uint32_t> mpmNodeOwners_;
    std::vector<std::uint32_t> femNodeOwners_;
};

} // namespace

bool validateCompiledWorldLayout(
    const CompiledWorld& world,
    std::string* error
) {
    try {
        return LayoutValidator(world, error).run();
    } catch (const std::bad_alloc&) {
        if (error != nullptr) {
            *error = "insufficient memory while validating Matter topology";
        }
        return false;
    } catch (...) {
        if (error != nullptr) {
            *error = "unexpected failure while validating Matter topology";
        }
        return false;
    }
}

} // namespace numi::matter
