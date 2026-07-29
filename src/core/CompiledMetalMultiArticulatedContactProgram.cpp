#include "metalrobo/MetalMultiArticulatedContact.hpp"

#include <cstddef>
#include <cstdint>
#include <new>
#include <span>
#include <string>
#include <utility>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

MetalMultiArticulatedContactDiagnostics fail(
    MetalMultiArticulatedContactDiagnostics diagnostics,
    const MetalMultiArticulatedContactStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
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
    return values.empty()
        ? hash
        : hashBytes(
              hash,
              values.data(),
              values.size_bytes()
          );
}

std::uint64_t fingerprint(
    const EngineModel& model,
    const ParallelABASchedule& schedule
) {
    std::uint64_t hash = kFnvOffset;
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
    hash = hashSpan<float>(hash, model.defaultQ);
    hash = hashSpan<float>(hash, model.defaultV);
    hash = hashBytes(
        hash,
        &schedule.fingerprint,
        sizeof(schedule.fingerprint)
    );
    return hash == 0u ? 1u : hash;
}

} // namespace

bool CompiledMetalMultiArticulatedContactProgram::valid()
    const noexcept {
    return fingerprint_ != 0u &&
        !model_.articulations.empty() &&
        schedule_.fingerprint != 0u &&
        schedule_.articulations.size() ==
            model_.articulations.size();
}

const EngineModel&
CompiledMetalMultiArticulatedContactProgram::model()
    const noexcept {
    return model_;
}

const ParallelABASchedule&
CompiledMetalMultiArticulatedContactProgram::abaSchedule()
    const noexcept {
    return schedule_;
}

std::uint64_t
CompiledMetalMultiArticulatedContactProgram::fingerprint()
    const noexcept {
    return valid() ? fingerprint_ : 0u;
}

MetalMultiArticulatedContactDiagnostics
compileMetalMultiArticulatedContactProgram(
    const EngineModel& model,
    CompiledMetalMultiArticulatedContactProgram& output
) {
    MetalMultiArticulatedContactDiagnostics diagnostics;
    try {
        std::string reason;
        if (!model.valid(&reason)) {
            return fail(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::invalidModel,
                "invalid EngineModel: " + reason
            );
        }
        if (model.articulations.empty()) {
            return fail(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    unsupportedTopology,
                "contact program requires at least one articulation"
            );
        }
        for (const MRArticulationGPU& articulation :
             model.articulations) {
            if (articulation.bodyCount == 0u ||
                articulation.bodyCount >
                    MR_ARTICULATED_OPERATOR_MAX_BODIES ||
                articulation.bodyCount >
                    MR_ARTICULATED_ABA_MAX_BODIES ||
                articulation.nv == 0u ||
                articulation.nv >
                    MR_ARTICULATED_OPERATOR_MAX_DOFS ||
                articulation.nv >
                    MR_ARTICULATED_ABA_MAX_DOFS ||
                articulation.nq >
                    MR_ARTICULATED_ABA_MAX_Q) {
                return fail(
                    std::move(diagnostics),
                    MetalMultiArticulatedContactStatus::
                        unsupportedTopology,
                    "an articulation exceeds the Metal contact bucket"
                );
            }
        }

        CompiledMetalMultiArticulatedContactProgram staged;
        staged.model_ = model;
        const auto scheduleDiagnostics =
            compileParallelABASchedule(
                staged.model_,
                staged.schedule_
            );
        if (!scheduleDiagnostics.succeeded()) {
            return fail(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    unsupportedTopology,
                "parallel ABA schedule compilation failed: " +
                    scheduleDiagnostics.message
            );
        }
        staged.fingerprint_ = fingerprint(
            staged.model_,
            staged.schedule_
        );
        if (!staged.valid()) {
            return fail(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    internalFailure,
                "compiled contact program is internally invalid"
            );
        }
        output = std::move(staged);
        diagnostics.published = true;
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return fail(
            std::move(diagnostics),
            MetalMultiArticulatedContactStatus::
                metalBufferFailure,
            "contact program allocation failed"
        );
    }
}

} // namespace metalrobo
