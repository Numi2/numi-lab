#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace numi::cardboard::damage_volume {

inline constexpr double kPaperDamageVolumeThreshold = 0.5;

struct PaperConstituentDamageVolume {
    double damageVolumeIntegral = 0.0;
    double referenceVolumeM3 = 0.0;
    double volumeAtOrAboveThresholdM3 = 0.0;
    std::uint64_t tetrahedronCount = 0u;

    bool addTetrahedron(double referenceVolume,
                        const std::array<double, 4>& planeAreas,
                        const std::array<double, 4>& planeDamage,
                        double threshold) {
        if (!(std::isfinite(referenceVolume) && referenceVolume > 0.0 &&
              std::isfinite(threshold) && threshold >= 0.0 && threshold <= 1.0) ||
            tetrahedronCount == std::numeric_limits<std::uint64_t>::max()) {
            return false;
        }
        double totalArea = 0.0;
        double areaWeightedDamage = 0.0;
        for (std::size_t plane = 0u; plane < planeAreas.size(); ++plane) {
            const double area = planeAreas[plane];
            const double damage = planeDamage[plane];
            if (!(std::isfinite(area) && area > 0.0 &&
                  std::isfinite(damage) && damage >= 0.0 && damage <= 1.0)) {
                return false;
            }
            totalArea += area;
            areaWeightedDamage += area * damage;
            if (!(std::isfinite(totalArea) && std::isfinite(areaWeightedDamage))) {
                return false;
            }
        }
        if (!(totalArea > 0.0)) return false;
        const double tetrahedronIndex = areaWeightedDamage / totalArea;
        const double nextDamageVolume =
            damageVolumeIntegral + referenceVolume * tetrahedronIndex;
        const double nextReferenceVolume = referenceVolumeM3 + referenceVolume;
        const double nextThresholdVolume = volumeAtOrAboveThresholdM3 +
            (tetrahedronIndex >= threshold ? referenceVolume : 0.0);
        if (!(std::isfinite(tetrahedronIndex) && tetrahedronIndex >= 0.0 &&
              tetrahedronIndex <= 1.0 &&
              std::isfinite(nextDamageVolume) &&
              std::isfinite(nextReferenceVolume) &&
              std::isfinite(nextThresholdVolume))) {
            return false;
        }
        damageVolumeIntegral = nextDamageVolume;
        referenceVolumeM3 = nextReferenceVolume;
        volumeAtOrAboveThresholdM3 = nextThresholdVolume;
        ++tetrahedronCount;
        return true;
    }

    bool finish(double& volumeWeightedDamageIndex,
                double& volumeFractionAtOrAboveThreshold) const {
        if (tetrahedronCount == 0u ||
            !(std::isfinite(damageVolumeIntegral) &&
              std::isfinite(referenceVolumeM3) && referenceVolumeM3 > 0.0 &&
              std::isfinite(volumeAtOrAboveThresholdM3) &&
              volumeAtOrAboveThresholdM3 >= 0.0 &&
              volumeAtOrAboveThresholdM3 <= referenceVolumeM3)) {
            return false;
        }
        volumeWeightedDamageIndex = damageVolumeIntegral / referenceVolumeM3;
        volumeFractionAtOrAboveThreshold =
            volumeAtOrAboveThresholdM3 / referenceVolumeM3;
        return std::isfinite(volumeWeightedDamageIndex) &&
            volumeWeightedDamageIndex >= 0.0 &&
            volumeWeightedDamageIndex <= 1.0 &&
            std::isfinite(volumeFractionAtOrAboveThreshold) &&
            volumeFractionAtOrAboveThreshold >= 0.0 &&
            volumeFractionAtOrAboveThreshold <= 1.0;
    }
};

} // namespace numi::cardboard::damage_volume
