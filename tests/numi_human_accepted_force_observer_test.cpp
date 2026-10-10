#include "metalrobo/numi_human_accepted_force_observer.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace {
void expect(const bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}
}

int main() {
    using metalrobo::human::observer::AcceptedGeneralizedForceSummary;
    using metalrobo::human::observer::summarizeAcceptedGeneralizedForceRows;
    constexpr std::array<float, 6u> muscleRows{
        1.25f, -2.0f, 3.5f,
        2.75f, 5.0f, -4.5f,
    };
    // The post-consumer slice intentionally differs from the row sum: an
    // optional owner may add a correction into this arena after the MyoSim
    // reduction. The observer must preserve it separately, not call it net.
    constexpr std::array<float, 3u> aggregate{91.0f, 92.0f, 93.0f};
    constexpr std::array<float, 6u> tendonRows{
        0.5f, 1.0f, -2.0f,
        -0.25f, 0.5f, 3.0f,
    };
    AcceptedGeneralizedForceSummary summary;
    std::string_view error;
    expect(summarizeAcceptedGeneralizedForceRows(
               muscleRows, aggregate, tendonRows, 2u, 2u, 3u,
               summary, error),
           "valid force row layout is accepted");
    expect(summary.muscleRowsByDof.size() == 3u &&
               summary.muscleRowsByDof[0] == 4.0 &&
               summary.muscleRowsByDof[1] == 3.0 &&
               summary.muscleRowsByDof[2] == -1.0,
           "muscle rows sum by local DoF in muscle-major order");
    expect(summary.tendonCorrectionsByDof.size() == 3u &&
               summary.tendonCorrectionsByDof[0] == 0.25 &&
               summary.tendonCorrectionsByDof[1] == 1.5 &&
               summary.tendonCorrectionsByDof[2] == 1.0,
           "tendon correction rows sum by local DoF in binding-major order");

    const std::array<float, 5u> shortMuscleRows{};
    expect(!summarizeAcceptedGeneralizedForceRows(
               shortMuscleRows, aggregate, tendonRows, 2u, 2u, 3u,
               summary, error) &&
               error == "muscle force row shape does not match [muscle][dof]",
           "truncated muscle ownership fails closed");
    const std::array<float, 2u> shortAggregate{};
    expect(!summarizeAcceptedGeneralizedForceRows(
               muscleRows, shortAggregate, tendonRows, 2u, 2u, 3u,
               summary, error),
           "aggregate slice must have one value per DoF");
    const std::array<float, 5u> shortTendonRows{};
    expect(!summarizeAcceptedGeneralizedForceRows(
               muscleRows, aggregate, shortTendonRows, 2u, 2u, 3u,
               summary, error),
           "tendon correction ownership must match binding count and nv");
    expect(!summarizeAcceptedGeneralizedForceRows(
               muscleRows, aggregate, tendonRows, 0u, 2u, 3u,
               summary, error),
           "missing muscle records are not silently summarized");
    expect(!summarizeAcceptedGeneralizedForceRows(
               muscleRows, aggregate, tendonRows,
               std::numeric_limits<std::size_t>::max(), 0u, 2u,
               summary, error) &&
               error == "force-row element count overflows size_t",
           "force row count overflow fails before shape or allocation");

    auto nonfiniteMuscle = muscleRows;
    nonfiniteMuscle[2] = std::numeric_limits<float>::quiet_NaN();
    expect(!summarizeAcceptedGeneralizedForceRows(
               nonfiniteMuscle, aggregate, tendonRows, 2u, 2u, 3u,
               summary, error),
           "non-finite muscle rows are rejected");
    auto nonfiniteAggregate = aggregate;
    nonfiniteAggregate[1] = std::numeric_limits<float>::infinity();
    expect(!summarizeAcceptedGeneralizedForceRows(
               muscleRows, nonfiniteAggregate, tendonRows, 2u, 2u, 3u,
               summary, error),
           "non-finite post-consumer aggregate slice is rejected");
    auto nonfiniteTendon = tendonRows;
    nonfiniteTendon[4] = std::numeric_limits<float>::quiet_NaN();
    expect(!summarizeAcceptedGeneralizedForceRows(
               muscleRows, aggregate, nonfiniteTendon, 2u, 2u, 3u,
               summary, error),
           "non-finite tendon corrections are rejected");

    expect(summarizeAcceptedGeneralizedForceRows(
               std::span<const float>(muscleRows.data(), 6u), aggregate,
               std::span<const float>{}, 2u, 0u, 3u, summary, error) &&
               summary.tendonCorrectionsByDof ==
                   std::vector<double>{0.0, 0.0, 0.0},
           "zero tendon bindings produce a zero correction vector");
    return 0;
}
