#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace metalrobo::human::observer {

struct AcceptedGeneralizedForceSummary {
    // CPU double-precision sums in local-v order. Muscle rows are the
    // already-collected MyoSim generalized-force rows. Tendon rows are
    // source-to-distributed-transfer corrections, not complete tendon loads.
    std::vector<double> muscleRowsByDof;
    std::vector<double> tendonCorrectionsByDof;
};

inline bool summarizeAcceptedGeneralizedForceRows(
    const std::span<const float> muscleRows,
    const std::span<const float> postConsumerAggregateSlice,
    const std::span<const float> tendonCorrectionRows,
    const std::size_t muscleCount,
    const std::size_t tendonBindingCount,
    const std::size_t dofCount,
    AcceptedGeneralizedForceSummary& summary,
    std::string_view& error
) {
    summary = {};
    error = {};
    if (dofCount == 0u || muscleCount == 0u) {
        error = "muscle and dof counts must be nonzero";
        return false;
    }
    if (muscleCount > std::numeric_limits<std::size_t>::max() / dofCount ||
        tendonBindingCount >
            std::numeric_limits<std::size_t>::max() / dofCount) {
        error = "force-row element count overflows size_t";
        return false;
    }
    if (muscleRows.size() != muscleCount * dofCount) {
        error = "muscle force row shape does not match [muscle][dof]";
        return false;
    }
    if (postConsumerAggregateSlice.size() != dofCount) {
        error = "aggregate force slice does not match dof count";
        return false;
    }
    if (tendonCorrectionRows.size() != tendonBindingCount * dofCount) {
        error = "tendon correction row shape does not match [binding][dof]";
        return false;
    }
    const auto finite = [](const std::span<const float> values) {
        for (const float value : values) {
            if (!std::isfinite(value)) return false;
        }
        return true;
    };
    if (!finite(muscleRows) || !finite(postConsumerAggregateSlice) ||
        !finite(tendonCorrectionRows)) {
        error = "force observer input contains a non-finite value";
        return false;
    }

    summary.muscleRowsByDof.assign(dofCount, 0.0);
    summary.tendonCorrectionsByDof.assign(dofCount, 0.0);
    for (std::size_t muscle = 0u; muscle < muscleCount; ++muscle) {
        const std::size_t row = muscle * dofCount;
        for (std::size_t dof = 0u; dof < dofCount; ++dof) {
            summary.muscleRowsByDof[dof] +=
                static_cast<double>(muscleRows[row + dof]);
        }
    }
    for (std::size_t binding = 0u;
         binding < tendonBindingCount; ++binding) {
        const std::size_t row = binding * dofCount;
        for (std::size_t dof = 0u; dof < dofCount; ++dof) {
            summary.tendonCorrectionsByDof[dof] +=
                static_cast<double>(tendonCorrectionRows[row + dof]);
        }
    }
    for (const double value : summary.muscleRowsByDof) {
        if (!std::isfinite(value)) {
            summary = {};
            error = "summed muscle force rows are non-finite";
            return false;
        }
    }
    for (const double value : summary.tendonCorrectionsByDof) {
        if (!std::isfinite(value)) {
            summary = {};
            error = "summed tendon correction rows are non-finite";
            return false;
        }
    }
    return true;
}

}  // namespace metalrobo::human::observer
