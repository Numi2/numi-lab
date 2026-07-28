#include "metalrobo/EngineModel.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

int main() {
    try {
        metalrobo::EngineModel model =
            metalrobo::makeFreeSphereEngineModel();
        std::string reason;
        if (!model.valid(&reason)) {
            throw std::runtime_error(reason);
        }
        if (model.world.nq != 7u || model.world.nv != 6u ||
            model.articulations.size() != 1u ||
            model.articulations.front().rootType != MR_ROOT_FLOATING) {
            throw std::runtime_error("floating/free-body mapping regressed");
        }

        metalrobo::ConstraintIRRow boundedRow{};
        boundedRow.impulseLower = -0.25F;
        boundedRow.impulseUpper = 0.25F;
        const std::array<metalrobo::ConstraintIREndpoint, 1>
            boundedEndpoints{
                metalrobo::makeConstraintIRGeneralizedEndpoint(
                    0u,
                    0u,
                    0u,
                    0u,
                    1.0F
                ),
            };
        const std::array<metalrobo::ConstraintIRRow, 1>
            boundedRows{boundedRow};
        const std::array<float, 1> boundedWarm{0.0F};
        const std::array<metalrobo::ConstraintIRSourceBlock, 1>
            boundedSources{
                metalrobo::ConstraintIRSourceBlock{
                    .key = {{1u, 0u, 0u, 0u}},
                    .type = MR_CONSTRAINT_DRY_FRICTION,
                    .islandIndex = 0u,
                    .endpoints = boundedEndpoints,
                    .rows = boundedRows,
                    .warmImpulses = boundedWarm,
                },
            };
        auto compiledProgram =
            metalrobo::compileConstraintIR(boundedSources);
        if (!compiledProgram.succeeded()) {
            throw std::runtime_error(
                "mechanism program compilation failed"
            );
        }
        model.constraintProgram =
            std::move(compiledProgram.ir);
        if (!model.valid(&reason)) {
            throw std::runtime_error(
                "model-owned constraint program failed: " +
                reason
            );
        }

        metalrobo::EngineModel broken = model;
        broken.defaultQ[6] = 0.5f;
        if (broken.valid(&reason) ||
            reason != "floating-root quaternion is not normalized") {
            throw std::runtime_error(
                "invalid quaternion was not rejected transactionally"
            );
        }

        broken = model;
        broken.world.contactCapacity = 0u;
        if (broken.valid(&reason) ||
            reason != "all production capacities must be explicit") {
            throw std::runtime_error(
                "missing production capacity was not rejected"
            );
        }

        broken = model;
        broken.dofs[3].qIndex = 3u;
        if (broken.valid(&reason) ||
            reason !=
                "floating-root DoF ownership or properties are invalid") {
            throw std::runtime_error(
                "quaternion-rate scalar mapping was not rejected"
            );
        }

        broken = model;
        broken.dofs[0].flags |= MR_DOF_FLAG_ACTUATED;
        if (broken.valid(&reason) ||
            reason !=
                "floating-root DoF ownership or properties are invalid") {
            throw std::runtime_error(
                "implicitly actuated floating root was not rejected"
            );
        }

        broken = model;
        broken.articulations[0].qOffset =
            std::numeric_limits<mr_u32>::max();
        broken.articulations[0].nq = 1u;
        if (broken.valid(&reason) ||
            reason != "articulation range or root is invalid") {
            throw std::runtime_error(
                "wrapping generalized range was not rejected"
            );
        }

        broken = model;
        broken.constraintProgram.endpoints[0].objectIndex =
            model.world.nv;
        if (broken.valid(&reason) ||
            reason !=
                "constraint generalized endpoint ownership "
                "is invalid") {
            throw std::runtime_error(
                "invalid mechanism ownership was not rejected"
            );
        }

        std::cout
            << "model=\"" << model.name << "\""
            << " abi=" << model.world.abiVersion
            << " bodies=" << model.world.bodyCount
            << " articulations=" << model.world.articulationCount
            << " nq=" << model.world.nq
            << " nv=" << model.world.nv
            << " root=floating"
            << " free_body=yes"
            << " invalid_quaternion_rejected=yes"
            << " dof_mapping_rejected=yes"
            << " passive_root_enforced=yes"
            << " capacity_preflight=yes"
            << " wrapping_range_rejected=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_engine_model_probe: " << error.what() << '\n';
        return 1;
    }
}
