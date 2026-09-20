#pragma once

#include "metalrobo/NumiHumanMuscleEquilibrium.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {

// Immutable shared program: nv*nv row-major FP32 K followed by nv rest
// coordinates. The native owner maps each scalar DoF through its source qIndex.
// It is not an external force vector or a constant equilibrium-reaction preload.
// Calibration and the applicability of this linear law remain source concerns.
inline bool validateNumiHumanPassiveJointProgram(
    const std::span<const float> program, const std::size_t nv,
    std::string& error
) {
    const auto fail = [&error](const char* message) {
        error = message;
        return false;
    };
    if (nv < 6u || nv > 160u || program.size() != nv * (nv + 1u))
        return fail("passive joint program has an invalid shape");
    if (!std::all_of(program.begin(), program.end(),
            [](float value) { return std::isfinite(value); }))
        return fail("passive joint program is non-finite");
    std::vector<std::size_t> active;
    for (std::size_t row = 0u; row < nv; ++row) {
        const double diagonal = program[row * nv + row];
        if (diagonal < 0.0)
            return fail("passive joint stiffness has a negative diagonal");
        for (std::size_t column = 0u; column < nv; ++column) {
            const float value = program[row * nv + column];
            if ((row < 6u || column < 6u) && value != 0.0f)
                return fail("passive joint law cannot apply a floating-root wrench");
            if (value != program[column * nv + row])
                return fail("passive joint stiffness must be symmetric");
            if (diagonal == 0.0 && value != 0.0f)
                return fail("zero passive stiffness diagonal has a nonzero coupling");
        }
        if (diagonal > 0.0) active.push_back(row);
    }
    // Test the actual uploaded FP32 matrix, in FP64 normalized coordinates.
    // Pivoting admits null directions without adding artificial stiffness.
    const std::size_t n = active.size();
    std::vector<double> schur(n * n, 0.0);
    for (std::size_t row = 0u; row < n; ++row)
        for (std::size_t column = 0u; column < n; ++column) {
            const auto r = active[row], c = active[column];
            schur[row*n+column] = static_cast<double>(program[r*nv+c]) /
                (std::sqrt(static_cast<double>(program[r*nv+r])) *
                 std::sqrt(static_cast<double>(program[c*nv+c])));
        }
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(n, std::size_t{1}));
    for (std::size_t k = 0u; k < n; ++k) {
        std::size_t pivot = k;
        for (std::size_t i = k+1u; i < n; ++i)
            if (schur[i*n+i] > schur[pivot*n+pivot]) pivot = i;
        if (pivot != k) {
            for (std::size_t i = 0u; i < n; ++i)
                std::swap(schur[k*n+i], schur[pivot*n+i]);
            for (std::size_t i = 0u; i < n; ++i)
                std::swap(schur[i*n+k], schur[i*n+pivot]);
        }
        const double diagonal = schur[k*n+k];
        if (diagonal <= tolerance) {
            for (std::size_t i = k; i < n; ++i)
                for (std::size_t j = k; j < n; ++j)
                    if (!std::isfinite(schur[i*n+j]) ||
                        std::abs(schur[i*n+j]) > tolerance)
                        return fail("passive joint stiffness is not positive semidefinite");
            break;
        }
        for (std::size_t i = k+1u; i < n; ++i)
            for (std::size_t j = i; j < n; ++j) {
                const double value = schur[j*n+i] -
                    schur[i*n+k] * schur[j*n+k] / diagonal;
                schur[j*n+i] = schur[i*n+j] = value;
            }
    }
    error.clear();
    return true;
}

// Compile the existing offline coupling records, not an independent tissue
// model. Multiple contributions may share a matrix entry. A source coordinate
// must have one rest position, so force is the gradient of one bounded energy.
// Failure never mutates the previously accepted output.
inline bool compileNumiHumanPassiveJointProgram(
    const std::span<const NumiHumanPassiveCoordinateCoupling> couplings,
    const std::size_t nv, std::vector<float>& output, std::string& error
) {
    if (nv < 6u || nv > 160u) {
        error = "passive joint program has an invalid DoF count";
        return false;
    }
    std::vector<double> matrix(nv*nv, 0.0), rest(nv, 0.0);
    std::vector<bool> assigned(nv, false);
    for (const auto& coupling : couplings) {
        const auto row = coupling.targetDofIndex, column = coupling.sourceDofIndex;
        if (row < 6u || column < 6u || row >= nv || column >= nv ||
            !std::isfinite(coupling.stiffness) ||
            !std::isfinite(coupling.sourceRestPosition)) {
            error = "passive source coupling is non-finite or not internal-scalar-owned";
            return false;
        }
        if (coupling.stiffness == 0.0) continue;
        if (assigned[column] && rest[column] != coupling.sourceRestPosition) {
            error = "passive source coordinate has conflicting rest positions";
            return false;
        }
        assigned[column] = true;
        rest[column] = coupling.sourceRestPosition;
        matrix[row*nv+column] += coupling.stiffness;
    }
    std::vector<float> candidate(nv*(nv+1u), 0.0f);
    for (std::size_t i = 0u; i < candidate.size(); ++i) {
        const double value = i < nv*nv ? matrix[i] : rest[i-nv*nv];
        if (!std::isfinite(value) ||
            std::abs(value) > static_cast<double>(std::numeric_limits<float>::max())) {
            error = "passive joint coefficient is not representable in FP32";
            return false;
        }
        candidate[i] = static_cast<float>(value);
        if (value != 0.0 && candidate[i] == 0.0f) {
            error = "passive joint coefficient underflows FP32";
            return false;
        }
    }
    if (!validateNumiHumanPassiveJointProgram(candidate, nv, error)) return false;
    output = std::move(candidate);
    return true;
}

} // namespace metalrobo
