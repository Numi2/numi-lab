#include "numi/matter/surgical_tissue.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef NUMI_SYNTHETIC_SKIN_MATERIAL
#define NUMI_SYNTHETIC_SKIN_MATERIAL ""
#endif

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double parameterValue(
    const numi::matter::MaterialProgram& material,
    const std::string& name
) {
    const auto found = std::find_if(
        material.parameters.begin(), material.parameters.end(),
        [&](const auto& parameter) { return parameter.name == name; }
    );
    require(found != material.parameters.end(), "skin material parameter absent");
    return found->defaultValue;
}

} // namespace

int main() {
    try {
        auto parsed = numi::matter::parseMatterFile(
            NUMI_SYNTHETIC_SKIN_MATERIAL
        );
        require(parsed.succeeded(), "synthetic skin material did not parse");
        require(parsed.material.name == "synthetic_skin_wound",
                "skin material identity changed");
        require(parameterValue(parsed.material, "density") == 1050.0,
                "synthetic density changed");
        require(parameterValue(parsed.material, "mu") == 25000.0,
                "synthetic shear modulus changed");
        require(parameterValue(parsed.material, "lambda") == 250000.0,
                "synthetic volumetric modulus changed");

        const numi::matter::SyntheticSkinWoundSpec spec;
        require(parameterValue(parsed.material, "density") ==
                    spec.densityKgPerM3,
                "synthetic material and specimen densities differ");
        auto coupon = numi::matter::makeSyntheticSkinWoundCoupon(0u, spec);
        require(coupon.object.name == "synthetic_skin_wound_coupon",
                "skin geometry identity changed");
        require(coupon.object.representation == numi::matter::Representation::fem &&
                coupon.object.twoWayCoupling &&
                coupon.object.mutationPolicy.enabled,
                "skin specimen lost the coupled FEM or puncture capability");
        require(coupon.metadata.nodeCount == coupon.object.femNodes.size() &&
                coupon.metadata.tetrahedronCount == coupon.object.tetrahedra.size() &&
                coupon.metadata.minimumRestTetrahedronVolumeM3 > 0.0,
                "skin specimen topology is incomplete");
        require(!coupon.metadata.incisionLipNodePairs.empty(),
                "skin wound has no separate incision lips");
        double maximumLipGapErrorM = 0.0;
        for (const auto& pair : coupon.metadata.incisionLipNodePairs) {
            require(pair[0] != pair[1] &&
                    pair[0] < coupon.object.femNodes.size() &&
                    pair[1] < coupon.object.femNodes.size(),
                    "wound lip ownership is invalid");
            const auto& lower = coupon.object.femNodes[pair[0]];
            const auto& upper = coupon.object.femNodes[pair[1]];
            const double gap = std::hypot(
                upper[0] - lower[0], upper[1] - lower[1],
                upper[2] - lower[2]
            );
            maximumLipGapErrorM = std::max(
                maximumLipGapErrorM,
                std::abs(gap - spec.incisionGapM)
            );
            require(
                std::find(coupon.object.femContactNodes.begin(),
                          coupon.object.femContactNodes.end(), pair[0]) !=
                    coupon.object.femContactNodes.end() &&
                std::find(coupon.object.femContactNodes.begin(),
                          coupon.object.femContactNodes.end(), pair[1]) !=
                    coupon.object.femContactNodes.end(),
                "wound lip is missing from the contact boundary"
            );
        }
        require(maximumLipGapErrorM <= 1.0e-12,
                "authored wound gap differs from requested geometry");

        numi::matter::WorldSource source;
        source.frameTimestep = 1.0 / 2000.0;
        source.contactSlop = 1.0e-5;
        source.deterministic = true;
        source.materials.push_back(std::move(parsed.material));
        source.objects.push_back(std::move(coupon.object));
        auto compiled = numi::matter::compileWorld(source, {});
        require(compiled.succeeded(), "synthetic skin world did not compile");
        std::string error;
        require(numi::matter::validateCompiledWorldLayout(
                    compiled.world, &error),
                "synthetic skin world layout is invalid");
        require(compiled.world.fem.nodes.size() >= coupon.metadata.nodeCount &&
                compiled.world.fem.tetrahedra.size() >=
                    coupon.metadata.tetrahedronCount,
                "skin mesh is absent from the cooked world");

        std::cout << std::setprecision(12)
                  << "synthetic_skin_wound_coupon=compiled"
                  << " nodes=" << coupon.metadata.nodeCount
                  << " tetrahedra=" << coupon.metadata.tetrahedronCount
                  << " lip_pairs=" << coupon.metadata.incisionLipNodePairs.size()
                  << " authored_gap_m=" << spec.incisionGapM
                  << " max_gap_error_m=" << maximumLipGapErrorM
                  << " world_fingerprint=" << compiled.world.fingerprint
                  << " physical_calibration=none"
                  << " stitch_executed=false\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "synthetic_skin_wound_coupon=failed reason=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
