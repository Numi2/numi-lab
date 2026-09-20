#include "metalrobo/MultiArticulatedWorld.hpp"
#include "metalrobo/SurgicalWorld.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    try {
        const metalrobo::DualPsmWorld world =
            metalrobo::makeDualDvrkPsmWorld();
        std::string reason;
        require(
            world.model.valid(&reason),
            "dual PSM model is invalid: " + reason
        );
        require(
            world.model.articulations.size() == 2u &&
                world.model.world.nq == 30u &&
                world.model.world.nv == 28u &&
                world.model.bodies.size() == 18u &&
                world.model.shapes.size() ==
                    2u * metalrobo::kSurgicalPSMShapeCount,
            "dual PSM packed dimensions changed"
        );
        require(
            world.model.constraintProgram.blocks.size() == 14u &&
                world.model.constraintProgram.rows.size() == 14u &&
                world.metadata.baseLockBlockCount == 12u &&
                world.metadata.jawCouplingBlockCount == 2u,
            "dual PSM generalized mechanism program is incomplete"
        );
        require(
            world.model.defaultQ[
                world.metadata.qOffsets[0]
            ] == -0.18f &&
                world.model.defaultQ[
                    world.metadata.qOffsets[1]
                ] == 0.18f,
            "dual PSM base placements were not preserved"
        );

        std::vector<double> q(
            world.model.defaultQ.begin(),
            world.model.defaultQ.end()
        );
        std::vector<double> v(
            world.model.defaultV.begin(),
            world.model.defaultV.end()
        );
        const std::uint32_t leftJawA =
            world.metadata.firstJawVelocity[0];
        const std::uint32_t leftJawB =
            world.metadata.secondJawVelocity[0];
        const std::uint32_t rightJawA =
            world.metadata.firstJawVelocity[1];
        const std::uint32_t rightJawB =
            world.metadata.secondJawVelocity[1];
        v[leftJawA] = 0.4;
        v[leftJawB] = 0.1;
        v[rightJawA] = -0.3;
        v[rightJawB] = 0.2;
        std::vector<double> force(world.model.world.nv, 0.0);
        metalrobo::MultiArticulationFactorCache cache;
        metalrobo::MultiArticulatedWorldConfig config;
        config.dynamics.timestep =
            world.model.world.gravityAndTimestep.w;
        config.solverIterations = 128u;
        config.solverTolerance = 1.0e-8;
        config.constraintResidual.residualTolerance =
            1.0e-7;
        const auto diagnostics =
            metalrobo::stepMultiArticulatedWorldCpu(
                world.model,
                q,
                v,
                force,
                {},
                cache,
                config
            );
        require(
            diagnostics.succeeded(),
            "dual PSM multi-articulation step failed"
        );
        require(
            std::abs(v[leftJawA] + v[leftJawB]) < 2.0e-7 &&
                std::abs(v[rightJawA] + v[rightJawB]) < 2.0e-7,
            "dual PSM jaw gear rows did not enforce symmetry"
        );
        for (std::uint32_t arm = 0u; arm < 2u; ++arm) {
            const std::uint32_t root =
                world.metadata.vOffsets[arm];
            for (std::uint32_t axis = 0u;
                 axis < 6u;
                 ++axis) {
                require(
                    std::abs(v[root + axis]) < 2.0e-7,
                    "dual PSM base-lock row leaked velocity"
                );
            }
        }

        std::vector<double> rejectedQ = q;
        std::vector<double> rejectedV = v;
        rejectedV[0] =
            std::numeric_limits<double>::quiet_NaN();
        metalrobo::MultiArticulationFactorCache rejectedCache =
            cache;
        const auto rejected =
            metalrobo::stepMultiArticulatedWorldCpu(
                world.model,
                rejectedQ,
                rejectedV,
                force,
                {},
                rejectedCache,
                config
            );
        require(
            !rejected.succeeded() &&
                rejectedQ == q &&
                rejectedCache.generation == cache.generation,
            "dual PSM failure was not transactional"
        );

        metalrobo::DualPsmNeedleThreadWorldConfig surgicalConfig;
        surgicalConfig.threadNodeCount = 9u;
        surgicalConfig.threadLengthM = 0.12;
        const metalrobo::DualPsmNeedleThreadWorld surgical =
            metalrobo::makeDualDvrkPsmNeedleThreadWorld(
                surgicalConfig
            );
        require(
            surgical.robots.model.articulations.size() == 2u &&
                surgical.robots.model.constraintProgram.blocks.size() ==
                    14u &&
                surgical.needle.rigid.shapes.size() ==
                    surgicalConfig.needle.arcSegments &&
                surgical.threadModel.restPositions.size() == 9u &&
                surgical.threadState.positions.front() ==
                    surgical.metadata.swageAnchorWorld &&
                surgical.threadContactMaterial.friction.x == 0.18f &&
                surgical.threadContactMaterial.friction.y == 0.12f &&
                surgical.attachments[0].targetPosition ==
                    surgical.metadata.swageAnchorWorld &&
                surgical.attachments.size() == 1u &&
                surgical.attachments[0].compliance == 0.0 &&
                surgical.metadata.hardSwagedThreadNodeCount == 1u &&
                surgical.metadata.threadBoundaryNodeCount == 2u &&
                surgical.rigidBindings[0].bodyIndex ==
                    surgical.metadata.needleSceneBodyIndex &&
                surgical.tangentBindings.size() == 1u &&
                surgical.tangentBindings[0].edgeIndex == 0u &&
                surgical.tangentBindings[0].bodyIndex ==
                    surgical.metadata.needleSceneBodyIndex &&
                surgical.rigidBindings[0].localAnchor ==
                    surgical.metadata.swageAnchorLocal &&
                surgical.tangentBindings[0].localTangent ==
                    surgicalConfig.threadExitDirectionLocal &&
                surgical.tangentBindings[0].complianceRadPerNm == 0.0 &&
                surgical.metadata.swageTangentComplianceRadPerNm == 0.0 &&
                surgical.twistBindings.size() == 1u &&
                surgical.twistBindings[0].edgeIndex == 0u &&
                surgical.twistBindings[0].bodyIndex ==
                    surgical.metadata.needleSceneBodyIndex &&
                surgical.twistBindings[0].localMaterialDirector ==
                    surgical.metadata.swageMaterialDirectorLocal &&
                surgical.twistBindings[0].complianceRadPerNm == 0.0 &&
                surgical.needleState.flagsAndIndices[0] ==
                    MR_MOTION_DYNAMIC &&
                surgical.needleState.
                        linearVelocityAndInverseMass.w >
                    0.0f,
            "dual PSM needle-thread composition is incomplete"
        );

        std::cout
            << "dual_psm_world=ok"
            << " articulations="
            << world.model.articulations.size()
            << " nq=" << world.model.world.nq
            << " nv=" << world.model.world.nv
            << " bodies=" << world.model.bodies.size()
            << " shapes=" << world.model.shapes.size()
            << " mechanism_blocks="
            << world.model.constraintProgram.blocks.size()
            << " solver_iterations="
            << diagnostics.solverIterations
            << " residual="
            << diagnostics.residual.maximumNaturalResidual
            << " needle_shapes="
            << surgical.needle.rigid.shapes.size()
            << " thread_nodes="
            << surgical.threadModel.restPositions.size()
            << " swage_bindings="
            << surgical.attachments.size() +
                   surgical.tangentBindings.size() +
                   surgical.twistBindings.size()
            << " swage_binding=hard_root_two_axis_tangent_material_frame"
            << " tangent_compliance_rad_per_nm="
            << surgical.tangentBindings[0].complianceRadPerNm
            << " torsional_compliance_rad_per_nm="
            << surgical.twistBindings[0].complianceRadPerNm
            << " hidden_grasp=no"
            << " transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "dual_psm_world=failed reason=\""
            << error.what()
            << "\"\n";
        return 1;
    }
}
