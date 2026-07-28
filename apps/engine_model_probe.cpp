#include "metalrobo/EngineModel.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

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
        broken.articulations[0].qOffset =
            std::numeric_limits<mr_u32>::max();
        broken.articulations[0].nq = 1u;
        if (broken.valid(&reason) ||
            reason != "articulation range or root is invalid") {
            throw std::runtime_error(
                "wrapping generalized range was not rejected"
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
            << " capacity_preflight=yes"
            << " wrapping_range_rejected=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_engine_model_probe: " << error.what() << '\n';
        return 1;
    }
}
