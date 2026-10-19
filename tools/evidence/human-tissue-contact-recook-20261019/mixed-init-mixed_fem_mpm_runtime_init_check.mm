#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace nm = numi::matter;
static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
static bool near(float a, float b) {
    return std::isfinite(a) && std::abs(a-b) <= 1.0e-6f;
}
int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            require(argc == 2, "usage: mixed_fem_mpm_runtime_init_check <matter-source-dir>");
            const std::filesystem::path matter = argv[1];
            auto femMaterial = nm::parseMatterFile(
                matter / "materials/starch2007_finite_glue_elastic.nmatter");
            auto mpmMaterial = nm::parseMatterFile(
                matter / "materials/silicone.nmatter");
            require(femMaterial.succeeded(), "FEM material parse failed");
            require(mpmMaterial.succeeded(), "MPM material parse failed");

            nm::WorldSource source;
            source.environmentCount = 1u;
            source.frameTimestep = 0.001;
            source.gravity = {0.0, 0.0, 0.0};
            source.compactImmutableFEMSurfaces = true;
            source.materials = {femMaterial.material, mpmMaterial.material};

            nm::ObjectSource fem;
            fem.name = "runtime-init-fem";
            fem.materialIndex = 0u;
            fem.representation = nm::Representation::fem;
            fem.mixedFEM = false;
            fem.twoWayCoupling = false;
            fem.femNodes = {{{0,0,0}}, {{0.01,0,0}}, {{0,0.01,0}},
                            {{0,0,0.01}}, {{0,0,-0.01}}};
            fem.tetrahedra = {{{0,1,2,3}}, {{0,2,1,4}}};
            source.objects.push_back(fem);

            nm::ObjectSource mpm;
            mpm.name = "runtime-init-mpm";
            mpm.materialIndex = 1u;
            mpm.representation = nm::Representation::mpm;
            mpm.twoWayCoupling = false;
            mpm.characteristicLength = 0.02;
            mpm.mpmGridMinimum = {0.0,0.0,0.0};
            mpm.mpmGridMaximum = {0.06,0.06,0.06};
            mpm.particles.push_back(nm::ParticleSource{
                {0.03,0.03,0.03}, {0.0,0.0,0.0}, 0.001, 1.0e-6, true});
            source.objects.push_back(mpm);

            auto cooked = nm::compileWorld(source);
            if (!cooked.succeeded()) {
                std::string detail;
                for (const auto& d : cooked.diagnostics) detail += d.message + "; ";
                throw std::runtime_error("mixed world compile failed: " + detail);
            }
            const auto& dispatch = cooked.world.dispatch;
            require(dispatch.femNodeCount == 5u, "cooked FEM node count mismatch");
            require(dispatch.tetrahedronCount == 2u, "cooked FEM tet count mismatch");
            require(dispatch.surfaceFaceCount == 6u, "cooked compact exterior face count mismatch");
            require(dispatch.particleCount == 1u, "cooked MPM particle count mismatch");
            require(dispatch.mpmGridCount == 1u, "cooked MPM grid count mismatch");

            nm::Runtime runtime;
            nm::RuntimeConfiguration config;
            config.environmentCount = 1u;
            config.captureEvents = true;
            config.captureDiagnostics = false;
            config.automaticIdentification = false;
            config.adaptiveTransfer = false;
            const auto initialized = runtime.initialize(cooked.world, config);
            require(initialized.encoded && runtime.valid(),
                    "mixed FEM/MPM Runtime.initialize failed: " + initialized.message);

            const auto snapshot = runtime.snapshot();
            require(snapshot.available, "initial mixed-state snapshot unavailable: " + snapshot.message);
            require(snapshot.femNodes.size() == dispatch.femNodeCount,
                    "runtime snapshot did not retain FEM state");
            require(snapshot.particles.size() == dispatch.particleCount,
                    "runtime snapshot did not retain MPM particle state");
            require(snapshot.mpmActiveNodeCounts.size() == 1u,
                    "runtime snapshot omitted MPM active-node count state");
            require(snapshot.controlStep == 0u && snapshot.physicsSubstep == 0u,
                    "initialization advanced the simulation cursor");
            require(near(snapshot.femNodes[1].positionAndMass.x, 0.01f) &&
                    near(snapshot.femNodes[1].positionAndMass.y, 0.0f),
                    "FEM initial state changed during initialization");
            require(near(snapshot.particles[0].positionAndMass.x, 0.03f) &&
                    near(snapshot.particles[0].positionAndMass.y, 0.03f) &&
                    near(snapshot.particles[0].positionAndMass.z, 0.03f),
                    "MPM initial state changed during initialization");
            std::cout << "{\"result\":\"PASS\",\"scope\":\"Runtime.initialize plus initial snapshot only; no encode/evolution\","
                      << "\"fem_nodes\":" << snapshot.femNodes.size()
                      << ",\"fem_tets\":" << dispatch.tetrahedronCount
                      << ",\"compact_faces\":" << dispatch.surfaceFaceCount
                      << ",\"mpm_particles\":" << snapshot.particles.size()
                      << ",\"mpm_active_count_records\":" << snapshot.mpmActiveNodeCounts.size()
                      << ",\"control_step\":" << snapshot.controlStep
                      << ",\"physics_substep\":" << snapshot.physicsSubstep
                      << "}\n";
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "FAIL " << e.what() << "\n";
            return 1;
        }
    }
}
