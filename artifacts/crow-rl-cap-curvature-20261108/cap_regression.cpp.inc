float verifyCappedNewtonClosure() {
    float maximumOptimality = 0.0f;
    for (const auto specification :
         std::array<std::array<float, 2>, 5>{{
             {0.25f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f},
             {4.0f, 0.0f}, {0.25f, 0.125f},
         }}) {
        const float radius = specification[0];
        const float adhesion = specification[1];
        const float normal = radius - adhesion;
        const std::array<float, 3> scales{1.0f, 1.0f, 1.0f};
        const std::array blocks{
            coneBlock(0u, scales, 1.0f, adhesion, normal, 9001u),
        };
        const std::array<float, 9> mass{
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        };
        const std::array<float, 3> bias{-10.0f, 3.0f, 4.0f};
        const std::array<float, 3> freeVelocity{
            -normal, 0.6f * radius, 0.8f * radius,
        };
        const std::array<float, 3> warmVelocity{0.0f, 0.0008f, -0.0006f};
        const std::array<float, 3> warmImpulses{};
        for (const bool pcg : {false, true}) {
            metalrobo::MetalUnifiedQualityProblem problem;
            problem.problemCount = 1u;
            problem.generalizedVelocityCount = 3u;
            problem.rowCount = 3u;
            problem.blocks = blocks;
            problem.dynamics = mass;
            problem.jacobian = mass;
            problem.bias = bias;
            problem.freeVelocity = freeVelocity;
            problem.warmVelocity = warmVelocity;
            problem.warmImpulses = warmImpulses;

            metalrobo::MetalUnifiedQualityConfig config;
            config.maximumNewtonIterations = 1u;
            config.optimalityTolerance = 1.0e-6f;
            config.directMaximumGeneralizedVelocities = pcg ? 1u : 64u;
            metalrobo::MetalUnifiedQualityResult result;
            const auto diagnostics = metalrobo::solveMetalUnifiedQuality(
                problem, result, config
            );
            require(
                diagnostics.succeeded() && result.statuses.size() == 1u &&
                    result.velocity.size() == 3u &&
                    result.statuses[0].newtonIterations == 1u,
                "capped-ball Newton derivative closure failed: radius=" +
                    std::to_string(radius) + " pcg=" + std::to_string(pcg) +
                    " " + diagnostics.message
            );
            // Independent FP64 circular-ball projection at the published
            // velocity. The warm perturbation is tangent to its active cap,
            // so this also checks the curvature used by direct and PCG.
            const double w1 = double(bias[1]) + result.velocity[1];
            const double w2 = double(bias[2]) + result.velocity[2];
            const double tangent = std::hypot(w1, w2);
            require(tangent > radius, "capped-ball control left its branch");
            const std::array<double, 3> impulse{
                normal, -double(radius) * w1 / tangent,
                -double(radius) * w2 / tangent,
            };
            double gradient = 0.0, dynamics = 0.0, constraint = 0.0;
            for (std::size_t dof = 0u; dof < 3u; ++dof) {
                const double action =
                    double(result.velocity[dof]) - freeVelocity[dof];
                gradient = std::max(gradient, std::abs(action - impulse[dof]));
                dynamics = std::max(dynamics, std::abs(action));
                constraint = std::max(constraint, std::abs(impulse[dof]));
            }
            const double residual = gradient / (1.0 + dynamics + constraint);
            require(
                std::isfinite(residual) &&
                    residual <= config.optimalityTolerance,
                "capped-ball published velocity failed independent residual"
            );
            maximumOptimality = std::max(
                maximumOptimality, static_cast<float>(residual)
            );
        }
    }
    return maximumOptimality;
}

