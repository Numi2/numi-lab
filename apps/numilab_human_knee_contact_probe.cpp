#include "metalrobo/NumiHumanKnee.hpp"
#include "metalrobo/NumiHumanKneeContact.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Point = std::array<double, 3u>;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::byte> readBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.is_open(), "Open Knee contact payload did not open");
    const std::streamsize size = input.tellg();
    require(size > 0, "Open Knee contact payload is empty");
    input.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    require(input.good(), "Open Knee contact payload read failed");
    return bytes;
}

Point subtract(const Point& a, const Point& b) {
    return {a[0u] - b[0u], a[1u] - b[1u], a[2u] - b[2u]};
}

Point cross(const Point& a, const Point& b) {
    return {
        a[1u] * b[2u] - a[2u] * b[1u],
        a[2u] * b[0u] - a[0u] * b[2u],
        a[0u] * b[1u] - a[1u] * b[0u],
    };
}

double dot(const Point& a, const Point& b) {
    return a[0u] * b[0u] + a[1u] * b[1u] + a[2u] * b[2u];
}

double length(const Point& value) { return std::sqrt(dot(value, value)); }

double triangleArea(
    const std::array<std::uint32_t, 3u>& face,
    const std::span<const Point> points
) {
    return 0.5 * length(cross(
        subtract(points[face[1u]], points[face[0u]]),
        subtract(points[face[2u]], points[face[0u]])));
}

double tetrahedronVolume(
    const std::array<std::uint32_t, 4u>& tetrahedron,
    const std::span<const Point> points
) {
    return std::abs(dot(
        subtract(points[tetrahedron[1u]], points[tetrahedron[0u]]),
        cross(subtract(points[tetrahedron[2u]], points[tetrahedron[0u]]),
              subtract(points[tetrahedron[3u]], points[tetrahedron[0u]])))) /
        6.0;
}

bool articular(const metalrobo::NumiHumanKneeRegionKind kind) {
    return kind == metalrobo::NumiHumanKneeRegionKind::cartilage ||
        kind == metalrobo::NumiHumanKneeRegionKind::meniscus;
}

double vectorLength(const Point& value) { return length(value); }

std::uint64_t mix(std::uint64_t hash, const void* bytes, std::size_t count) {
    const auto* values = static_cast<const std::uint8_t*>(bytes);
    for (std::size_t index = 0u; index < count; ++index) {
        hash ^= values[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

std::uint64_t resultFingerprint(
    const metalrobo::NumiHumanKneeContactResult& result
) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const Point& force : result.nodalForcesNewtons)
        hash = mix(hash, force.data(), sizeof(force));
    for (const auto& pair : result.pairs) {
        hash = mix(hash, &pair.activeSampleCount,
                   sizeof(pair.activeSampleCount));
        hash = mix(hash, &pair.minimumGapChangeMeters,
                   sizeof(pair.minimumGapChangeMeters));
        hash = mix(hash, &pair.maximumPressurePascals,
                   sizeof(pair.maximumPressurePascals));
        hash = mix(hash, &pair.contactAreaSquareMeters,
                   sizeof(pair.contactAreaSquareMeters));
        hash = mix(hash, &pair.normalForceNewtons,
                   sizeof(pair.normalForceNewtons));
        hash = mix(hash, &pair.storedEnergyJoules,
                   sizeof(pair.storedEnergyJoules));
    }
    hash = mix(hash, result.forceResidualNewtons.data(),
               sizeof(result.forceResidualNewtons));
    hash = mix(hash, result.momentResidualNewtonMeters.data(),
               sizeof(result.momentResidualNewtonMeters));
    hash = mix(hash, &result.forceL1Newtons,
               sizeof(result.forceL1Newtons));
    hash = mix(hash, &result.storedEnergyJoules,
               sizeof(result.storedEnergyJoules));
    return hash;
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2,
                "usage: metalrobo_numilab_human_knee_contact_probe OPEN_KNEE_NHKNEE1");
        const std::vector<std::byte> bytes = readBytes(argv[1]);
        metalrobo::NumiHumanKneePayload knee;
        const auto decoded = metalrobo::decodeNumiHumanKneePayload(bytes, knee);
        require(decoded.succeeded(),
                std::string("Open Knee contact payload rejected: ") +
                    metalrobo::numiHumanKneeStatusName(decoded.status));
        std::vector<Point> restNodes;
        restNodes.reserve(knee.nodes.size());
        for (const auto& node : knee.nodes)
            restNodes.push_back({node.restWorld[0u], node.restWorld[1u],
                                 node.restWorld[2u]});

        std::vector<double> regionVolumes(knee.regions.size(), 0.0);
        std::vector<double> meniscusContactAreas(knee.regions.size(), 0.0);
        for (std::uint32_t regionIndex = 0u;
             regionIndex < knee.regions.size(); ++regionIndex) {
            const auto& region = knee.regions[regionIndex];
            for (std::uint32_t local = 0u;
                 local < region.tetrahedronCount; ++local) {
                regionVolumes[regionIndex] += tetrahedronVolume(
                    knee.tetrahedra[region.firstTetrahedron + local],
                    restNodes);
            }
        }
        for (const auto& pair : knee.surfacePairs) {
            const auto& master = knee.surfaces[pair.masterSurface];
            const auto& slave = knee.surfaces[pair.slaveSurface];
            if (!articular(knee.regions[master.regionIndex].kind) ||
                !articular(knee.regions[slave.regionIndex].kind)) continue;
            for (const auto surfaceIndex :
                 {pair.masterSurface, pair.slaveSurface}) {
                const auto& surface = knee.surfaces[surfaceIndex];
                if (knee.regions[surface.regionIndex].kind !=
                    metalrobo::NumiHumanKneeRegionKind::meniscus) continue;
                for (std::uint32_t local = 0u;
                     local < surface.faceCount; ++local)
                    meniscusContactAreas[surface.regionIndex] += triangleArea(
                        knee.faces[surface.firstFace + local], restNodes);
            }
        }

        std::vector<metalrobo::NumiHumanKneeContactRegionMaterial> materials;
        double minimumMeniscusThickness =
            std::numeric_limits<double>::infinity();
        double maximumMeniscusThickness = 0.0;
        for (std::uint32_t regionIndex = 0u;
             regionIndex < knee.regions.size(); ++regionIndex) {
            const auto kind = knee.regions[regionIndex].kind;
            if (!articular(kind)) continue;
            metalrobo::NumiHumanKneeContactMaterial material;
            if (kind == metalrobo::NumiHumanKneeRegionKind::cartilage) {
                // KneeHub/Open Knee elastic-foundation reference values.
                material = {
                    .elasticModulusPascals = 12.0e6,
                    .poissonRatio = 0.45,
                    .thicknessMeters = 0.003,
                };
            } else {
                require(regionVolumes[regionIndex] > 0.0 &&
                            meniscusContactAreas[regionIndex] > 0.0,
                        "Open Knee meniscus thickness measure is unavailable");
                // V / mean(top area, bottom area) = 2V/(top+bottom).
                const double thickness = 2.0 * regionVolumes[regionIndex] /
                    meniscusContactAreas[regionIndex];
                require(std::isfinite(thickness) && thickness >= 0.001 &&
                            thickness <= 0.015,
                        "Open Knee geometry-derived meniscus thickness is implausible");
                material = {
                    .elasticModulusPascals = 20.0e6,
                    .poissonRatio = 0.30,
                    .thicknessMeters = thickness,
                };
                minimumMeniscusThickness = std::min(
                    minimumMeniscusThickness, thickness);
                maximumMeniscusThickness = std::max(
                    maximumMeniscusThickness, thickness);
            }
            materials.push_back({
                .regionIndex = regionIndex,
                .material = material,
            });
        }

        metalrobo::NumiHumanKneeContactModel model;
        const auto built = metalrobo::buildNumiHumanKneeArticularContactModel(
            knee, restNodes, materials, model);
        require(built.succeeded(),
                std::string("Open Knee contact model failed: ") +
                    metalrobo::numiHumanKneeContactStatusName(built.status) +
                    " " + built.message);
        require(model.pairs.size() == 7u && !model.samples.empty(),
                "Open Knee articular coverage is incomplete");

        constexpr std::uint32_t stepCount = 65u;
        constexpr double peakClosureMeters = 0.00005;
        double previousLoadingForce = 0.0;
        double peakForce = 0.0;
        double peakPressure = 0.0;
        double peakEnergy = 0.0;
        double maximumForceResidual = 0.0;
        double maximumMomentResidual = 0.0;
        std::uint64_t peakFingerprint = 0u;
        metalrobo::NumiHumanKneeContactResult peakResult;
        for (std::uint32_t step = 0u; step < stepCount; ++step) {
            const std::uint32_t half = (stepCount - 1u) / 2u;
            const double fraction = step <= half
                ? static_cast<double>(step) / static_cast<double>(half)
                : static_cast<double>(stepCount - 1u - step) /
                    static_cast<double>(half);
            const double closure = peakClosureMeters * fraction;
            metalrobo::NumiHumanKneeContactResult result;
            const auto evaluated = metalrobo::evaluateNumiHumanKneeContact(
                model, restNodes, closure, result);
            require(evaluated.succeeded(),
                    std::string("Open Knee contact evaluation failed: ") +
                        evaluated.message);
            const double totalNormalForce = 0.5 * result.forceL1Newtons;
            if (step <= half) {
                require(totalNormalForce + 1.0e-9 >= previousLoadingForce,
                        "Open Knee contact loading response is non-monotonic");
                previousLoadingForce = totalNormalForce;
            }
            maximumForceResidual = std::max(
                maximumForceResidual, vectorLength(result.forceResidualNewtons));
            maximumMomentResidual = std::max(
                maximumMomentResidual,
                vectorLength(result.momentResidualNewtonMeters));
            if (step == half) {
                peakForce = totalNormalForce;
                peakEnergy = result.storedEnergyJoules;
                peakResult = result;
                for (const auto& pair : result.pairs) {
                    require(pair.activeSampleCount > 0u &&
                                pair.normalForceNewtons > 0.0 &&
                                pair.maximumPressurePascals > 0.0 &&
                                pair.contactAreaSquareMeters > 0.0,
                            "Open Knee peak contact pair is inactive");
                    peakPressure = std::max(
                        peakPressure, pair.maximumPressurePascals);
                }
                peakFingerprint = resultFingerprint(result);
                metalrobo::NumiHumanKneeContactResult replay;
                const auto replayed = metalrobo::evaluateNumiHumanKneeContact(
                    model, restNodes, closure, replay);
                require(replayed.succeeded() &&
                            resultFingerprint(replay) == peakFingerprint,
                        "Open Knee contact peak replay is not bitwise");
            }
            if (step == 0u || step == stepCount - 1u) {
                require(result.forceL1Newtons == 0.0 &&
                            result.storedEnergyJoules == 0.0,
                        "Open Knee unloaded contact state did not restore");
            }
        }
        require(std::isfinite(peakForce) && peakForce > 0.0 &&
                    std::isfinite(peakPressure) && peakPressure > 0.0 &&
                    std::isfinite(peakEnergy) && peakEnergy > 0.0,
                "Open Knee contact peak response is invalid");
        const double energyRelativeError = std::abs(
            peakEnergy - 0.5 * peakForce * peakClosureMeters) /
            std::max(1.0e-16, peakEnergy);
        require(std::isfinite(energyRelativeError) &&
                    energyRelativeError <= 1.0e-12,
                "Open Knee contact elastic work/energy identity failed");
        const double residualScale = std::max(1.0, peakForce);
        require(maximumForceResidual <= 1.0e-10 * residualScale &&
                    maximumMomentResidual <= 1.0e-10 * residualScale,
                "Open Knee contact action-reaction balance failed");

        std::cout << std::setprecision(17)
                  << "numi_human_knee_contact=passed"
                  << " side="
                  << (knee.side == metalrobo::NumiHumanKneeSide::left
                          ? "left" : "right_mirrored")
                  << " articular_pairs=" << model.pairs.size()
                  << " contact_samples=" << model.samples.size()
                  << " sustained_steps=" << stepCount
                  << " peak_closure_m=" << peakClosureMeters
                  << " peak_normal_force_n=" << peakForce
                  << " peak_pressure_pa=" << peakPressure
                  << " peak_energy_j=" << peakEnergy
                  << " energy_relative_error=" << energyRelativeError
                  << " max_force_residual_n=" << maximumForceResidual
                  << " max_moment_residual_nm=" << maximumMomentResidual
                  << " meniscus_thickness_min_m="
                  << minimumMeniscusThickness
                  << " meniscus_thickness_max_m="
                  << maximumMeniscusThickness
                  << " replay=bitwise restore=verified"
                  << " evidence_level=preflight"
                  << " live_human_coupling=false"
                  << " nonpenetration_solve=false\n";
        require(peakResult.pairs.size() == model.pairs.size(),
                "Open Knee peak pair result coverage drifted");
        for (std::uint32_t pairIndex = 0u;
             pairIndex < model.pairs.size(); ++pairIndex) {
            const auto& pair = model.pairs[pairIndex];
            const auto& peak = peakResult.pairs[pairIndex];
            const auto& sourcePair = knee.surfacePairs[pair.sourcePairIndex];
            std::cout << "articular_pair=" << pair.name
                      << " master_surface="
                      << knee.surfaces[sourcePair.masterSurface].name
                      << " slave_surface="
                      << knee.surfaces[sourcePair.slaveSurface].name
                      << " samples=" << pair.sampleCount
                      << " area_m2=" << pair.tributaryAreaSquareMeters
                      << " foundation_stiffness_pa_per_m="
                      << pair.effectiveFoundationStiffnessPascalsPerMeter
                      << " peak_active_samples=" << peak.activeSampleCount
                      << " peak_pressure_pa=" << peak.maximumPressurePascals
                      << " peak_normal_force_n=" << peak.normalForceNewtons
                      << " peak_energy_j=" << peak.storedEnergyJoules
                      << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "numi_human_knee_contact=failed error=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
