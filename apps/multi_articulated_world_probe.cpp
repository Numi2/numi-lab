#include "metalrobo/MultiArticulatedWorld.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool bitwiseEqual(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    return left.size() == right.size() &&
        (left.empty() ||
         std::memcmp(
             left.data(),
             right.data(),
             left.size() * sizeof(double)
         ) == 0);
}

metalrobo::EngineModel makeTwoFreeBodies() {
    metalrobo::EngineModel model =
        metalrobo::makeFreeSphereEngineModel();
    model.name = "two_free_articulations";

    MRArticulationGPU second =
        model.articulations.front();
    second.rootBody = 2u;
    second.firstBody = 2u;
    second.qOffset = 7u;
    second.vOffset = 6u;
    model.articulations.push_back(second);

    MRBodyPropertiesGPU secondBody = model.bodies[1];
    secondBody.articulationIndex = 1u;
    model.bodies.push_back(secondBody);

    for (std::uint32_t local = 0u; local < 6u; ++local) {
        MRDofPropertiesGPU dof = model.dofs[local];
        dof.articulationIndex = 1u;
        dof.qIndex = local < 3u ? 7u + local : MR_INVALID_INDEX;
        dof.vIndex = 6u + local;
        model.dofs.push_back(dof);
    }

    MRShapeGPU secondShape = model.shapes[1];
    secondShape.bodyIndex = 2u;
    secondShape.slotGeneration = 2u;
    model.shapes.push_back(secondShape);

    const std::array<float, 7> secondQ{
        2.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    model.defaultQ.insert(
        model.defaultQ.end(),
        secondQ.begin(),
        secondQ.end()
    );
    model.defaultV.resize(12u, 0.0F);
    model.world.bodyCount =
        static_cast<mr_u32>(model.bodies.size());
    model.world.articulationCount =
        static_cast<mr_u32>(model.articulations.size());
    model.world.shapeCount =
        static_cast<mr_u32>(model.shapes.size());
    model.world.nq =
        static_cast<mr_u32>(model.defaultQ.size());
    model.world.nv =
        static_cast<mr_u32>(model.defaultV.size());

    metalrobo::ConstraintIRRow gearRow{};
    gearRow.timeConstant = 0.02F;
    const std::array<metalrobo::ConstraintIREndpoint, 2>
        endpoints{
            metalrobo::makeConstraintIRGeneralizedEndpoint(
                0u, 0u, 0u, 0u, 1.0F
            ),
            metalrobo::makeConstraintIRGeneralizedEndpoint(
                1u, 7u, 6u, 0u, -1.0F
            ),
        };
    const std::array<metalrobo::ConstraintIRRow, 1>
        rows{gearRow};
    const std::array<float, 1> warm{0.0F};
    const std::array<metalrobo::ConstraintIRSourceBlock, 1>
        sources{
            metalrobo::ConstraintIRSourceBlock{
                .key = {{1u, 2u, 0u, 0u}},
                .type = MR_CONSTRAINT_GEAR,
                .islandIndex = 0u,
                .endpoints = endpoints,
                .rows = rows,
                .warmImpulses = warm,
            },
        };
    auto compiled = metalrobo::compileConstraintIR(sources);
    require(
        compiled.succeeded(),
        "cross-articulation gear compilation failed"
    );
    model.constraintProgram = std::move(compiled.ir);
    return model;
}

} // namespace

int main() {
    try {
        metalrobo::EngineModel model = makeTwoFreeBodies();
        std::string reason;
        require(
            model.valid(&reason),
            "two-articulation model is invalid: " + reason
        );

        std::vector<double> q(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        std::vector<double> v(model.world.nv, 0.0);
        std::vector<double> force(model.world.nv, 0.0);
        force[0] = 1.0;

        metalrobo::MultiArticulatedWorldConfig config;
        config.dynamics.gravity = {0.0, 0.0, 0.0};
        config.dynamics.timestep = 1.0e-3;
        config.dynamics.applyBodyDamping = false;
        config.constraintEvaluation.timestep =
            config.dynamics.timestep;
        config.constraintEvaluation.minimumRegularization =
            1.0e-8;
        config.solverIterations = 96u;
        config.solverTolerance = 1.0e-10;
        config.constraintResidual.residualTolerance =
            2.0e-6;

        metalrobo::MultiArticulationFactorCache cache;
        const auto diagnostics =
            metalrobo::stepMultiArticulatedWorldCpu(
                model,
                q,
                v,
                force,
                {},
                cache,
                config
            );
        require(
            diagnostics.succeeded(),
            std::string("multi-articulation step failed: ") +
                metalrobo::multiArticulatedWorldStatusName(
                    diagnostics.status
                )
        );
        require(
            cache.factors.size() == 2u &&
            diagnostics.articulationCount == 2u &&
            diagnostics.constraintBlockCount == 1u &&
            diagnostics.constraintRowCount == 1u &&
            diagnostics.residual.maximumNaturalResidual <
                2.0e-6,
            "multi-articulation evidence is incomplete"
        );
        require(
            std::abs(v[0] - v[6]) < 2.0e-6 &&
            v[0] > 0.0 &&
            v[6] > 0.0 &&
            std::abs(v[0] - 5.0e-4) < 2.0e-6,
            "cross-articulation impulse did not couple equal masses: " +
                std::to_string(v[0]) + "/" +
                std::to_string(v[6])
        );

        const std::vector<double> acceptedQ = q;
        const std::vector<double> acceptedV = v;
        force[0] = std::numeric_limits<double>::infinity();
        const auto failed =
            metalrobo::stepMultiArticulatedWorldCpu(
                model,
                q,
                v,
                force,
                {},
                cache,
                config
            );
        require(
            !failed.succeeded() &&
            bitwiseEqual(q, acceptedQ) &&
            bitwiseEqual(v, acceptedV),
            "failed multi-articulation step mutated state"
        );

        std::cout
            << "multi_articulated_world=ok"
            << " articulations="
            << diagnostics.articulationCount
            << " rows=" << diagnostics.constraintRowCount
            << " iterations=" << diagnostics.solverIterations
            << " velocity_a=" << v[0]
            << " velocity_b=" << v[6]
            << " residual="
            << diagnostics.residual.maximumNaturalResidual
            << " factor_residual="
            << diagnostics.maximumFactorResidual
            << " transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "multi-articulated world probe failed: "
            << error.what() << '\n';
        return 1;
    }
}
