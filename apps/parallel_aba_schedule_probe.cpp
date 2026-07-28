#include "metalrobo/EngineModelComposer.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/ParallelABASchedule.hpp"
#include "metalrobo/SurgicalWorld.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    try {
        const metalrobo::DualPsmWorld dualPsm =
            metalrobo::makeDualDvrkPsmWorld();
        const metalrobo::EngineModel g1 =
            metalrobo::makeUnitreeG1EngineModel();
        const std::array components{
            metalrobo::EngineModelComponent{
                .model = &dualPsm.model,
                .instanceId = "surgical_cell",
            },
            metalrobo::EngineModelComponent{
                .model = &g1,
                .instanceId = "mobile_operator",
            },
        };
        metalrobo::EngineModel heterogeneous;
        const auto composeDiagnostics =
            metalrobo::composeEngineModels(
                components,
                heterogeneous
            );
        require(
            composeDiagnostics.succeeded(),
            "failed to compose heterogeneous schedule probe"
        );
        require(
            heterogeneous.articulations.size() == 3u,
            "probe did not compose three articulations"
        );

        metalrobo::ParallelABASchedule schedule;
        const auto diagnostics =
            metalrobo::compileParallelABASchedule(
                heterogeneous,
                schedule
            );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"schedule compilation failed: "} +
                diagnostics.message
            );
        }
        std::string reason;
        require(schedule.valid(&reason), reason.c_str());
        require(
            schedule.articulations.size() == 3u,
            "compiled schedule lost an articulation"
        );
        require(
            diagnostics.maximumDepth > 0u &&
                diagnostics.maximumLevelWidth > 0u,
            "compiled schedule has empty frontier diagnostics"
        );

        std::uint32_t branchingArticulations = 0u;
        std::uint32_t reverseReductions = 0u;
        for (const MRParallelABAArticulationGPU& articulation :
             schedule.articulations) {
            branchingArticulations +=
                (articulation.flags &
                 MR_PARALLEL_ABA_BRANCHING) != 0u
                ? 1u
                : 0u;
            for (std::uint32_t levelIndex = 0u;
                 levelIndex < articulation.reverseLevelCount;
                 ++levelIndex) {
                reverseReductions += schedule.levels[
                    articulation.reverseLevelOffset + levelIndex
                ].parentReductionCount;
            }
        }
        require(
            branchingArticulations > 0u &&
                reverseReductions > 0u,
            "probe did not exercise parent-owned branch reduction"
        );

        metalrobo::ParallelABASchedule replay;
        const auto replayDiagnostics =
            metalrobo::compileParallelABASchedule(
                heterogeneous,
                replay
            );
        require(
            replayDiagnostics.succeeded() &&
                replay.fingerprint == schedule.fingerprint &&
                replay.levelBodies == schedule.levelBodies &&
                replay.childIndices == schedule.childIndices,
            "parallel ABA cooking is not deterministic"
        );

        const std::uint64_t sentinelFingerprint =
            schedule.fingerprint;
        const std::size_t sentinelLevels = schedule.levels.size();
        metalrobo::EngineModel invalid = heterogeneous;
        invalid.articulations[0].rootBody =
            invalid.articulations[0].firstBody +
            invalid.articulations[0].bodyCount;
        const auto rejected =
            metalrobo::compileParallelABASchedule(
                invalid,
                schedule
            );
        require(
            !rejected.succeeded() &&
                schedule.fingerprint == sentinelFingerprint &&
                schedule.levels.size() == sentinelLevels,
            "failed schedule compilation was not transactional"
        );

        std::cout
            << "parallel_aba_schedule=ok"
            << " articulations=" << schedule.articulations.size()
            << " bodies=" << heterogeneous.bodies.size()
            << " forward_reverse_levels="
            << schedule.levels.size()
            << " max_depth=" << diagnostics.maximumDepth
            << " max_width=" << diagnostics.maximumLevelWidth
            << " branching_articulations="
            << branchingArticulations
            << " parent_reductions=" << reverseReductions
            << " fingerprint=" << schedule.fingerprint
            << " deterministic=yes transactional=yes\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "parallel_aba_schedule=failed reason="
            << exception.what() << '\n';
        return 1;
    }
}
