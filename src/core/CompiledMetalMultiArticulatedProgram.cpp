#include "metalrobo/MetalMultiArticulatedConstraints.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

MetalMultiArticulatedConstraintDiagnostics fail(
    MetalMultiArticulatedConstraintDiagnostics diagnostics,
    const MetalMultiArticulatedConstraintStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool supportedTopology(
    const EngineModel& model,
    const std::size_t articulationIndex
) {
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    if ((articulation.rootType != MR_ROOT_FIXED &&
         articulation.rootType != MR_ROOT_FLOATING) ||
        articulation.bodyCount == 0u ||
        articulation.bodyCount > MR_ARTICULATED_ABA_MAX_BODIES ||
        articulation.nv == 0u ||
        articulation.nv > MR_ARTICULATED_ABA_MAX_DOFS ||
        articulation.nq > MR_ARTICULATED_ABA_MAX_Q ||
        articulation.jointCount + 1u != articulation.bodyCount) {
        return false;
    }
    for (std::uint32_t localJoint = 0u;
         localJoint < articulation.jointCount;
         ++localJoint) {
        const std::uint32_t type = model.joints[
            articulation.firstJoint + localJoint
        ].jointType;
        if (type != MR_JOINT_FIXED &&
            type != MR_JOINT_REVOLUTE &&
            type != MR_JOINT_CONTINUOUS &&
            type != MR_JOINT_PRISMATIC) {
            return false;
        }
    }
    return true;
}

bool compileJacobian(
    const EngineModel& model,
    std::vector<float>& jacobian
) {
    const ConstraintIR& program = model.constraintProgram;
    jacobian.assign(
        program.rows.size() * model.world.nv,
        0.0f
    );
    for (const ConstraintIRBlock& block : program.blocks) {
        if (block.type == MR_CONSTRAINT_CONTACT) {
            return false;
        }
        for (std::uint32_t local = 0u;
             local < block.endpointCount;
             ++local) {
            const ConstraintIREndpoint& endpoint =
                program.endpoints[
                    block.endpointOffset + local
                ];
            const std::uint32_t localRow =
                endpoint.flags &
                constraintIREndpointRowMask;
            if (endpoint.jacobianKind !=
                    constraintIRJacobianGeneralized ||
                endpoint.objectIndex >= model.world.nv ||
                localRow >= block.dimension ||
                !std::isfinite(endpoint.axis.x)) {
                return false;
            }
            float& coefficient = jacobian[
                (block.rowOffset + localRow) *
                    model.world.nv +
                endpoint.objectIndex
            ];
            coefficient += endpoint.axis.x;
            if (!std::isfinite(coefficient)) {
                return false;
            }
        }
    }
    for (std::size_t row = 0u;
         row < program.rows.size();
         ++row) {
        bool nonzero = false;
        for (std::uint32_t dof = 0u;
             dof < model.world.nv;
             ++dof) {
            nonzero =
                nonzero ||
                jacobian[row * model.world.nv + dof] != 0.0f;
        }
        if (!nonzero) {
            return false;
        }
    }
    return true;
}

std::uint64_t hashBytes(
    std::uint64_t hash,
    const void* data,
    const std::size_t bytes
) {
    const auto* values =
        static_cast<const unsigned char*>(data);
    for (std::size_t index = 0u; index < bytes; ++index) {
        hash ^= values[index];
        hash *= kFnvPrime;
    }
    return hash;
}

template <typename T>
std::uint64_t hashSpan(
    std::uint64_t hash,
    const std::span<const T> values
) {
    const std::uint64_t count = values.size();
    hash = hashBytes(hash, &count, sizeof(count));
    if (!values.empty()) {
        hash = hashBytes(hash, values.data(), values.size_bytes());
    }
    return hash;
}

std::uint64_t fingerprint(
    const CompiledMetalMultiArticulatedProgram& program
) {
    std::uint64_t hash = kFnvOffset;
    const EngineModel& model = program.model();
    hash = hashBytes(
        hash,
        model.name.data(),
        model.name.size()
    );
    hash = hashBytes(hash, &model.world, sizeof(model.world));
    hash = hashSpan<MRArticulationGPU>(
        hash,
        model.articulations
    );
    hash = hashSpan<MRJointDescriptorGPU>(hash, model.joints);
    hash = hashSpan<MRDofPropertiesGPU>(hash, model.dofs);
    hash = hashSpan<MRBodyPropertiesGPU>(hash, model.bodies);
    hash = hashSpan<ConstraintIRBlock>(
        hash,
        model.constraintProgram.blocks
    );
    hash = hashSpan<ConstraintIREndpoint>(
        hash,
        model.constraintProgram.endpoints
    );
    hash = hashSpan<ConstraintIRRow>(
        hash,
        model.constraintProgram.rows
    );
    hash = hashSpan<float>(
        hash,
        model.constraintProgram.warmImpulses
    );
    const std::uint64_t schedule =
        program.abaSchedule().fingerprint;
    hash = hashBytes(hash, &schedule, sizeof(schedule));
    hash = hashSpan<float>(
        hash,
        program.generalizedJacobian()
    );
    hash = hashSpan<std::uint32_t>(
        hash,
        program.rowChunkOffsets()
    );
    hash = hashSpan<std::uint32_t>(
        hash,
        program.rowChunkCounts()
    );
    return hash == 0u ? 1u : hash;
}

} // namespace

bool CompiledMetalMultiArticulatedProgram::valid()
    const noexcept {
    return
        fingerprint_ != 0u &&
        !model_.articulations.empty() &&
        !model_.constraintProgram.rows.empty() &&
        abaSchedule_.fingerprint != 0u &&
        generalizedJacobian_.size() ==
            static_cast<std::size_t>(model_.world.nv) *
                model_.constraintProgram.rows.size() &&
        !rowChunkOffsets_.empty() &&
        rowChunkOffsets_.size() == rowChunkCounts_.size();
}

const EngineModel&
CompiledMetalMultiArticulatedProgram::model() const noexcept {
    return model_;
}

const ParallelABASchedule&
CompiledMetalMultiArticulatedProgram::abaSchedule()
    const noexcept {
    return abaSchedule_;
}

std::span<const float>
CompiledMetalMultiArticulatedProgram::generalizedJacobian()
    const noexcept {
    return generalizedJacobian_;
}

std::span<const std::uint32_t>
CompiledMetalMultiArticulatedProgram::rowChunkOffsets()
    const noexcept {
    return rowChunkOffsets_;
}

std::span<const std::uint32_t>
CompiledMetalMultiArticulatedProgram::rowChunkCounts()
    const noexcept {
    return rowChunkCounts_;
}

std::uint32_t
CompiledMetalMultiArticulatedProgram::rowCount()
    const noexcept {
    return valid()
        ? static_cast<std::uint32_t>(
              model_.constraintProgram.rows.size()
          )
        : 0u;
}

std::uint64_t
CompiledMetalMultiArticulatedProgram::fingerprint()
    const noexcept {
    return valid() ? fingerprint_ : 0u;
}

MetalMultiArticulatedConstraintDiagnostics
compileMetalMultiArticulatedProgram(
    const EngineModel& model,
    CompiledMetalMultiArticulatedProgram& output
) {
    MetalMultiArticulatedConstraintDiagnostics diagnostics{};
    try {
        std::string reason;
        if (!model.valid(&reason)) {
            return fail(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::invalidModel,
                "invalid EngineModel: " + reason
            );
        }
        if (model.articulations.empty()) {
            return fail(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    unsupportedTopology,
                "compiled program requires at least one articulation"
            );
        }
        const std::size_t rowCount =
            model.constraintProgram.rows.size();
        if (rowCount == 0u ||
            rowCount > MR_GENERALIZED_CONSTRAINT_MAX_ROWS) {
            return fail(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    unsupportedConstraint,
                "generalized row count is outside the Metal bucket"
            );
        }
        for (std::size_t articulation = 0u;
             articulation < model.articulations.size();
             ++articulation) {
            if (!supportedTopology(model, articulation)) {
                return fail(
                    std::move(diagnostics),
                    MetalMultiArticulatedConstraintStatus::
                        unsupportedTopology,
                    "an articulation exceeds the Metal ABA bucket"
                );
            }
        }

        CompiledMetalMultiArticulatedProgram staged;
        staged.model_ = model;
        const ParallelABAScheduleDiagnostics scheduleDiagnostics =
            compileParallelABASchedule(
                staged.model_,
                staged.abaSchedule_
            );
        if (!scheduleDiagnostics.succeeded()) {
            return fail(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    unsupportedTopology,
                "parallel ABA schedule compilation failed: " +
                    scheduleDiagnostics.message
            );
        }
        if (!compileJacobian(
                staged.model_,
                staged.generalizedJacobian_
            )) {
            return fail(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    unsupportedConstraint,
                "ConstraintIR contains a contact, spatial endpoint, "
                "or zero generalized row"
            );
        }
        for (std::size_t rowOffset = 0u;
             rowOffset < rowCount;
             rowOffset +=
                 MR_ARTICULATED_INVERSE_MASS_MAX_RHS) {
            staged.rowChunkOffsets_.push_back(
                static_cast<std::uint32_t>(rowOffset)
            );
            staged.rowChunkCounts_.push_back(
                static_cast<std::uint32_t>(
                    std::min(
                        std::size_t{
                            MR_ARTICULATED_INVERSE_MASS_MAX_RHS
                        },
                        rowCount - rowOffset
                    )
                )
            );
        }
        staged.fingerprint_ = fingerprint(staged);
        diagnostics.layout.dispatch.rowCount =
            static_cast<std::uint32_t>(rowCount);
        diagnostics.layout.dispatch.nv = model.world.nv;
        diagnostics.layout.inverseMassDispatches.resize(
            staged.rowChunkOffsets_.size() *
                staged.model_.articulations.size()
        );
        output = std::move(staged);
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return fail(
            std::move(diagnostics),
            MetalMultiArticulatedConstraintStatus::
                metalBufferFailure,
            "host allocation failed while compiling the "
            "multi-articulation program"
        );
    } catch (const std::exception& exception) {
        return fail(
            std::move(diagnostics),
            MetalMultiArticulatedConstraintStatus::invalidModel,
            exception.what()
        );
    }
}

} // namespace metalrobo
