#include "metalrobo/EngineModelComposer.hpp"

#include "metalrobo/GeometryCooker.hpp"
#include "metalrobo/MultiArticulatedWorld.hpp"
#include "metalrobo/SurgicalWorld.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

metalrobo::EngineModel makeConvexFreeModel(
    const std::string& name
) {
    metalrobo::EngineModel model =
        metalrobo::makeFreeSphereEngineModel();
    model.name = name;
    const std::array<mr_float4, 4> vertices{{
        {-0.08f, -0.08f, -0.08f, 1.0f},
        {0.08f, -0.08f, 0.08f, 1.0f},
        {-0.08f, 0.08f, 0.08f, 1.0f},
        {0.08f, 0.08f, -0.08f, 1.0f},
    }};
    const std::array<std::uint32_t, 12> indices{{
        0u, 2u, 1u,
        0u, 1u, 3u,
        1u, 2u, 3u,
        2u, 0u, 3u,
    }};
    const auto cooked = metalrobo::cookConvexGeometry(
        model,
        vertices,
        indices
    );
    require(
        cooked.succeeded(),
        "convex component cook failed: " + cooked.message
    );
    MRShapeGPU convex{};
    convex.bodyIndex = 1u;
    convex.shapeType = MR_SHAPE_CONVEX;
    convex.materialIndex = 0u;
    convex.collisionGroup = 1u;
    convex.collisionMask = ~0u;
    convex.slotGeneration = 2u;
    convex.geometryOffset = cooked.geometryIndex;
    convex.geometryCount = 1u;
    convex.localRotation = {0.0f, 0.0f, 0.0f, 1.0f};
    convex.dimensions = {1.0f, 1.0f, 1.0f, 0.0f};
    convex.contactRestAndBoundingRadius = {
        0.005f,
        0.0f,
        0.14f,
        0.0f,
    };
    model.shapes.push_back(convex);
    model.world.shapeCount =
        static_cast<std::uint32_t>(model.shapes.size());
    model.world.pairCapacity = 3u;
    model.world.contactCapacity = 12u;
    model.world.constraintCapacity = 16u;
    std::string reason;
    require(
        model.valid(&reason),
        "convex component is invalid: " + reason
    );
    return model;
}

} // namespace

int main() {
    try {
        const metalrobo::EngineModel first =
            makeConvexFreeModel("free_convex_a");
        const metalrobo::EngineModel second =
            makeConvexFreeModel("free_convex_b");
        const metalrobo::DualPsmWorld dual =
            metalrobo::makeDualDvrkPsmWorld();
        const std::array<metalrobo::EngineModelComponent, 3>
            components{{
                {&first, "object_a"},
                {&second, "object_b"},
                {&dual.model, "dual_psm"},
            }};
        metalrobo::EngineModel composed;
        metalrobo::EngineModelComposeConfig config;
        config.name = "heterogeneous_dual_psm_two_convex";
        const auto diagnostics =
            metalrobo::composeEngineModels(
                components,
                composed,
                config
            );
        require(
            diagnostics.succeeded(),
            "heterogeneous composition failed: " +
                diagnostics.message
        );
        std::string reason;
        require(
            composed.valid(&reason),
            "heterogeneous output is invalid: " + reason
        );
        require(
            composed.articulations.size() == 4u &&
                composed.world.nq == 44u &&
                composed.world.nv == 40u &&
                composed.bodies.size() == 22u &&
                composed.shapes.size() == 46u &&
                composed.materials.size() == 4u &&
                composed.geometryHeaders.size() == 2u &&
                composed.constraintProgram.blocks.size() == 14u,
            "heterogeneous packed counts changed"
        );
        require(
            composed.shapes[2].geometryOffset == 0u &&
                composed.shapes[5].geometryOffset == 1u &&
                composed.geometryHeaders[1].vertexOffset ==
                    first.geometryVertices.size() &&
                composed.geometryHeaders[1].halfEdgeOffset ==
                    first.convexHalfEdges.size(),
            "nested geometry arenas were not rebased"
        );
        for (const auto& endpoint :
             composed.constraintProgram.endpoints) {
            require(
                endpoint.articulationIndex >= 2u &&
                    endpoint.objectIndex >= 12u,
                "composed generalized endpoint was not rebased"
            );
        }

        std::vector<double> q(
            composed.defaultQ.begin(),
            composed.defaultQ.end()
        );
        std::vector<double> v(
            composed.defaultV.begin(),
            composed.defaultV.end()
        );
        std::vector<double> force(composed.world.nv, 0.0);
        metalrobo::MultiArticulationFactorCache cache;
        metalrobo::MultiArticulatedWorldConfig stepConfig;
        stepConfig.dynamics.timestep =
            config.gravityAndTimestep.w;
        stepConfig.solverIterations = 128u;
        stepConfig.solverTolerance = 1.0e-8;
        stepConfig.constraintResidual.residualTolerance =
            1.0e-7;
        const auto step =
            metalrobo::stepMultiArticulatedWorldCpu(
                composed,
                q,
                v,
                force,
                {},
                cache,
                stepConfig
            );
        require(
            step.succeeded(),
            "heterogeneous multi-articulation step failed"
        );

        metalrobo::EngineModel replay;
        const auto replayDiagnostics =
            metalrobo::composeEngineModels(
                components,
                replay,
                config
            );
        require(
            replayDiagnostics.succeeded() &&
                replay.constraintProgram.blocks.size() ==
                    composed.constraintProgram.blocks.size(),
            "heterogeneous replay failed"
        );
        for (std::size_t block = 0u;
             block < composed.constraintProgram.blocks.size();
             ++block) {
            require(
                metalrobo::constraintIRKeyEqual(
                    replay.constraintProgram.blocks[block].key,
                    composed.constraintProgram.blocks[block].key
                ),
                "composed stable key replay changed"
            );
        }

        metalrobo::EngineModel sentinel =
            metalrobo::makeFreeSphereEngineModel();
        const std::string sentinelName = sentinel.name;
        const std::array<metalrobo::EngineModelComponent, 2>
            duplicate{{
                {&first, "duplicate"},
                {&second, "duplicate"},
            }};
        const auto rejected =
            metalrobo::composeEngineModels(
                duplicate,
                sentinel,
                config
            );
        require(
            !rejected.succeeded() &&
                sentinel.name == sentinelName &&
                sentinel.articulations.size() == 1u,
            "failed heterogeneous composition published partial state"
        );

        std::cout
            << "engine_model_composer=ok"
            << " components=" << diagnostics.componentCount
            << " articulations=" << diagnostics.articulationCount
            << " bodies=" << diagnostics.bodyCount
            << " shapes=" << diagnostics.shapeCount
            << " geometries=" << diagnostics.geometryCount
            << " constraints="
            << diagnostics.constraintBlockCount
            << " factor_residual="
            << step.maximumFactorResidual
            << " deterministic=yes transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "engine_model_composer=failed reason=\""
            << error.what()
            << "\"\n";
        return 1;
    }
}
