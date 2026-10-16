#pragma once

#include "cardboard_box_assembly_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace numi::cardboard {

struct ShippingAcceptedBounds {
    BoxVec3 minimum{};
    BoxVec3 maximum{};
    bool valid = false;
};

inline ShippingAcceptedBounds measureShippingAcceptedBounds(
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t nodeCount,
    const std::uint32_t environment = 0u) {
    if (nodeCount == 0u || environment >
            std::numeric_limits<std::size_t>::max() / nodeCount ||
        acceptedNodes.size() <
            (static_cast<std::size_t>(environment) + 1u) * nodeCount)
        throw std::invalid_argument("shipping accepted-node bounds input is truncated");
    ShippingAcceptedBounds result;
    result.minimum = {std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity()};
    result.maximum = {-std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity()};
    const std::size_t base = static_cast<std::size_t>(environment) * nodeCount;
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        const BoxVec3 position = boxAcceptedPosition(acceptedNodes[base + node]);
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z))
            throw std::invalid_argument("shipping accepted-node bounds are nonfinite");
        result.minimum.x = std::min(result.minimum.x, position.x);
        result.minimum.y = std::min(result.minimum.y, position.y);
        result.minimum.z = std::min(result.minimum.z, position.z);
        result.maximum.x = std::max(result.maximum.x, position.x);
        result.maximum.y = std::max(result.maximum.y, position.y);
        result.maximum.z = std::max(result.maximum.z, position.z);
    }
    result.valid = result.maximum.x > result.minimum.x &&
        result.maximum.y > result.minimum.y &&
        result.maximum.z > result.minimum.z;
    if (!result.valid)
        throw std::invalid_argument("shipping accepted-node bounds are degenerate");
    return result;
}

struct ShippingCompressionIncrement {
    double displacementM = 0.0;
    double timestepSeconds = 0.0;
};

// Body-weight transfer is a separate physical test from a drop. Gravity stays
// enabled while an upward, transaction-scoped actuator force decreases to
// zero; a following accepted zero-wrench step is the actuator release.
struct ShippingGravityTransferProtocol {
    double durationSeconds = 0.5;
    double timestepSeconds = 5.0e-4;
    double postReleaseObservationSeconds = 0.25;
};

inline std::uint32_t shippingGravityTransferStepCount(
    const ShippingGravityTransferProtocol& protocol) {
    if (!(std::isfinite(protocol.durationSeconds) &&
          protocol.durationSeconds > 0.0 &&
          std::isfinite(protocol.timestepSeconds) &&
          protocol.timestepSeconds > 0.0 &&
          std::isfinite(protocol.postReleaseObservationSeconds) &&
          protocol.postReleaseObservationSeconds >= 0.0))
        throw std::invalid_argument(
            "shipping gravity-transfer timing must be finite and positive");
    const double steps = std::ceil(
        protocol.durationSeconds / protocol.timestepSeconds);
    if (!(std::isfinite(steps) && steps >= 1.0 &&
          steps <= static_cast<double>(std::numeric_limits<std::uint32_t>::max())))
        throw std::invalid_argument(
            "shipping gravity-transfer step count exceeds native capacity");
    return static_cast<std::uint32_t>(steps);
}

inline double shippingGravityTransferSupportFraction(
    const std::uint32_t step,
    const std::uint32_t stepCount) {
    if (stepCount == 0u || step >= stepCount)
        throw std::invalid_argument(
            "shipping gravity-transfer step is outside its schedule");
    // Midpoint quadrature for a linear force ramp. It avoids an endpoint
    // discontinuity at the last loaded interval; zero force is then submitted
    // explicitly by the separate actuator-release transaction.
    return 1.0 - (static_cast<double>(step) + 0.5) /
        static_cast<double>(stepCount);
}

inline double shippingGravityTransferSupportForceY(
    const double massKg,
    const double gravityYMetresPerSecondSquared,
    const double supportFraction) {
    if (!(std::isfinite(massKg) && massKg > 0.0 &&
          std::isfinite(gravityYMetresPerSecondSquared) &&
          gravityYMetresPerSecondSquared < 0.0 &&
          std::isfinite(supportFraction) && supportFraction >= 0.0 &&
          supportFraction <= 1.0))
        throw std::invalid_argument(
            "shipping support force requires positive mass, downward gravity, and a unit fraction");
    return -massKg * gravityYMetresPerSecondSquared * supportFraction;
}

inline double shippingActuatorWorkJ(
    const std::array<double, 3>& forceNewtons,
    const std::array<double, 3>& acceptedDisplacementMetres) {
    double work = 0.0;
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        if (!(std::isfinite(forceNewtons[axis]) &&
              std::isfinite(acceptedDisplacementMetres[axis])))
            throw std::invalid_argument(
                "shipping actuator work contains a nonfinite component");
        work += forceNewtons[axis] * acceptedDisplacementMetres[axis];
    }
    return work;
}

struct ShippingLinearMomentumBalance {
    std::array<double, 3> deltaMomentumKgMPerS{};
    std::array<double, 3> gravityImpulseNs{};
    std::array<double, 3> actuatorImpulseNs{};
    std::array<double, 3> nativeContactImpulseNs{};
    std::array<double, 3> residualNs{};
    double normalizedResidual = std::numeric_limits<double>::infinity();
};

inline ShippingLinearMomentumBalance shippingLinearMomentumBalance(
    const std::array<double, 3>& momentumBefore,
    const std::array<double, 3>& momentumAfter,
    const std::array<double, 3>& gravityMPerS2,
    const double massKg,
    const std::array<double, 3>& actuatorForceNewtons,
    const std::array<double, 3>& contactImpulseNs,
    const double timestepSeconds) {
    if (!(std::isfinite(massKg) && massKg > 0.0 &&
          std::isfinite(timestepSeconds) && timestepSeconds > 0.0))
        throw std::invalid_argument(
            "shipping momentum balance requires positive mass and timestep");
    ShippingLinearMomentumBalance result;
    double residual2 = 0.0;
    double scale2 = 0.0;
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        const std::array<double, 5> inputs{
            momentumBefore[axis], momentumAfter[axis], gravityMPerS2[axis],
            actuatorForceNewtons[axis], contactImpulseNs[axis]};
        for (const double value : inputs)
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    "shipping momentum balance contains a nonfinite component");
        result.deltaMomentumKgMPerS[axis] =
            momentumAfter[axis] - momentumBefore[axis];
        result.gravityImpulseNs[axis] = massKg * gravityMPerS2[axis] *
            timestepSeconds;
        result.actuatorImpulseNs[axis] = actuatorForceNewtons[axis] *
            timestepSeconds;
        result.nativeContactImpulseNs[axis] = contactImpulseNs[axis];
        result.residualNs[axis] = result.deltaMomentumKgMPerS[axis] -
            result.gravityImpulseNs[axis] - result.actuatorImpulseNs[axis] -
            contactImpulseNs[axis];
        residual2 += result.residualNs[axis] * result.residualNs[axis];
        scale2 += result.deltaMomentumKgMPerS[axis] *
                result.deltaMomentumKgMPerS[axis] +
            result.gravityImpulseNs[axis] * result.gravityImpulseNs[axis] +
            result.actuatorImpulseNs[axis] * result.actuatorImpulseNs[axis] +
            contactImpulseNs[axis] * contactImpulseNs[axis];
    }
    result.normalizedResidual = std::sqrt(residual2) /
        std::max(1.0e-12, std::sqrt(scale2));
    return result;
}

struct ShippingCompressionForceSample {
    std::uint64_t sampleIndex = 0u;
    double strokeM = 0.0;
    double forceN = 0.0;
    double contactSamples = 0.0;
};

struct ShippingCompressionSensitivity {
    bool valid = false;
    double commonStrokeM = 0.0;
    double nominalWorkJ = 0.0;
    double halfSpeedWorkJ = 0.0;
    double relativeWorkDifference = std::numeric_limits<double>::infinity();
    double maximumForceDifferenceNormalizedByNominalPeak =
        std::numeric_limits<double>::infinity();
    double rmsForceDifferenceNormalizedByNominalCurve =
        std::numeric_limits<double>::infinity();
};

inline double shippingCompressionRelativePeakDifference(
    const double nominalPeakForceN,
    const double halfSpeedPeakForceN) {
    if (!(std::isfinite(nominalPeakForceN) && nominalPeakForceN > 0.0 &&
          std::isfinite(halfSpeedPeakForceN) && halfSpeedPeakForceN > 0.0))
        return std::numeric_limits<double>::infinity();
    return std::abs(halfSpeedPeakForceN - nominalPeakForceN) /
        nominalPeakForceN;
}

inline bool shippingCompressionPeakEquivalent(
    const double nominalPeakForceN,
    const double halfSpeedPeakForceN,
    const double tolerance = 0.02) {
    return std::isfinite(tolerance) && tolerance > 0.0 &&
        shippingCompressionRelativePeakDifference(
            nominalPeakForceN, halfSpeedPeakForceN) <= tolerance;
}

inline double shippingCompressionForceAtStroke(
    const std::span<const ShippingCompressionForceSample> curve,
    const double strokeM) {
    if (curve.size() < 2u || !std::isfinite(strokeM) ||
        strokeM < curve.front().strokeM || strokeM > curve.back().strokeM)
        throw std::invalid_argument("compression curve interpolation is outside its samples");
    const auto upper = std::lower_bound(curve.begin(), curve.end(), strokeM,
        [](const ShippingCompressionForceSample& sample, const double stroke) {
            return sample.strokeM < stroke;
        });
    if (upper == curve.begin()) return upper->forceN;
    if (upper == curve.end()) return curve.back().forceN;
    if (upper->strokeM == strokeM) return upper->forceN;
    const auto& lower = *(upper - 1);
    const double width = upper->strokeM - lower.strokeM;
    if (!(width > 0.0))
        throw std::invalid_argument("compression curve stroke samples are not strictly increasing");
    const double fraction = (strokeM - lower.strokeM) / width;
    return lower.forceN + fraction * (upper->forceN - lower.forceN);
}

inline ShippingCompressionSensitivity compareShippingCompressionCurves(
    const std::span<const ShippingCompressionForceSample> nominal,
    const std::span<const ShippingCompressionForceSample> halfSpeed,
    const double nominalReferencePeakForceN = 0.0) {
    ShippingCompressionSensitivity result;
    const auto validate = [](const auto curve) {
        if (curve.size() < 2u || curve.front().strokeM != 0.0)
            throw std::invalid_argument("compression sensitivity needs complete curves from zero stroke");
        double previousStroke = -1.0;
        std::uint64_t previousIndex = 0u;
        for (std::size_t index = 0u; index < curve.size(); ++index) {
            const auto& sample = curve[index];
            if (!(std::isfinite(sample.strokeM) && sample.strokeM >= 0.0 &&
                  sample.strokeM > previousStroke &&
                  (index == 0u ? sample.sampleIndex == 0u :
                                 sample.sampleIndex > previousIndex) &&
                  std::isfinite(sample.forceN) && sample.forceN >= 0.0 &&
                  std::isfinite(sample.contactSamples) &&
                  ((index == 0u && sample.strokeM == 0.0 &&
                    sample.forceN == 0.0 && sample.contactSamples == 0.0) ||
                   (index > 0u && sample.contactSamples > 0.0))))
                throw std::invalid_argument("compression sensitivity curve contains an invalid sample");
            previousStroke = sample.strokeM;
            previousIndex = sample.sampleIndex;
        }
    };
    validate(nominal);
    validate(halfSpeed);
    result.commonStrokeM = std::min(nominal.back().strokeM,
                                    halfSpeed.back().strokeM);
    if (!(result.commonStrokeM > 0.0))
        throw std::invalid_argument("compression sensitivity has no common positive stroke");

    std::vector<double> knots{0.0, result.commonStrokeM};
    for (const auto& sample : nominal)
        if (sample.strokeM > 0.0 && sample.strokeM < result.commonStrokeM)
            knots.push_back(sample.strokeM);
    for (const auto& sample : halfSpeed)
        if (sample.strokeM > 0.0 && sample.strokeM < result.commonStrokeM)
            knots.push_back(sample.strokeM);
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());

    double nominalSquaredIntegral = 0.0;
    double differenceSquaredIntegral = 0.0;
    double nominalPeak = nominalReferencePeakForceN;
    if (!(std::isfinite(nominalPeak) && nominalPeak >= 0.0))
        throw std::invalid_argument("compression nominal reference peak is invalid");
    double maxDifference = 0.0;
    if (nominalPeak == 0.0)
        for (const auto& sample : nominal)
            if (sample.strokeM <= result.commonStrokeM)
                nominalPeak = std::max(nominalPeak, sample.forceN);
    for (std::size_t index = 1u; index < knots.size(); ++index) {
        const double a = knots[index - 1u];
        const double b = knots[index];
        const double nominalA = shippingCompressionForceAtStroke(nominal, a);
        const double nominalB = shippingCompressionForceAtStroke(nominal, b);
        const double halfA = shippingCompressionForceAtStroke(halfSpeed, a);
        const double halfB = shippingCompressionForceAtStroke(halfSpeed, b);
        const double differenceA = halfA - nominalA;
        const double differenceB = halfB - nominalB;
        const double width = b - a;
        result.nominalWorkJ += 0.5 * (nominalA + nominalB) * width;
        result.halfSpeedWorkJ += 0.5 * (halfA + halfB) * width;
        nominalSquaredIntegral += width / 3.0 *
            (nominalA * nominalA + nominalA * nominalB + nominalB * nominalB);
        differenceSquaredIntegral += width / 3.0 *
            (differenceA * differenceA + differenceA * differenceB +
             differenceB * differenceB);
        maxDifference = std::max({maxDifference, std::abs(differenceA),
                                  std::abs(differenceB)});
    }
    if (!(nominalPeak > 0.0) || !(result.nominalWorkJ > 0.0) ||
        !(nominalSquaredIntegral > 0.0))
        throw std::invalid_argument("compression sensitivity nominal curve has no positive work/force");
    result.relativeWorkDifference = std::abs(
        result.halfSpeedWorkJ - result.nominalWorkJ) / result.nominalWorkJ;
    result.maximumForceDifferenceNormalizedByNominalPeak =
        maxDifference / nominalPeak;
    result.rmsForceDifferenceNormalizedByNominalCurve =
        std::sqrt(differenceSquaredIntegral / nominalSquaredIntegral);
    result.valid = std::isfinite(result.relativeWorkDifference) &&
        std::isfinite(result.maximumForceDifferenceNormalizedByNominalPeak) &&
        std::isfinite(result.rmsForceDifferenceNormalizedByNominalCurve);
    return result;
}

inline bool shippingCompressionSensitivityPassed(
    const ShippingCompressionSensitivity& comparison,
    const double workTolerance = 0.02,
    const double fullCurveTolerance = 0.05) {
    if (!comparison.valid ||
        !(std::isfinite(workTolerance) && workTolerance > 0.0) ||
        !(std::isfinite(fullCurveTolerance) && fullCurveTolerance > 0.0))
        return false;
    return comparison.relativeWorkDifference <= workTolerance &&
        comparison.maximumForceDifferenceNormalizedByNominalPeak <=
            fullCurveTolerance;
}

struct ShippingQualifiedCompressionPeak {
    ShippingCompressionForceSample peak{};
    std::uint64_t qualifyingSampleIndex = 0u;
    double qualifyingDropFraction = 0.0;
};

// Streaming selector for the first measured local peak. A peak is not
// returned on the rising edge: it requires a rising predecessor, two
// consecutive declining accepted contact samples, and the configured drop
// from the candidate peak. The 1% default is an explicit protocol criterion,
// not a material parameter.
class ShippingFirstCompressionPeakDetector {
public:
    explicit ShippingFirstCompressionPeakDetector(
        const double requiredDropFraction = 0.01,
        const std::uint32_t requiredConsecutiveDeclines = 2u)
        : requiredDropFraction_(requiredDropFraction),
          requiredConsecutiveDeclines_(requiredConsecutiveDeclines) {
        if (!(std::isfinite(requiredDropFraction_) &&
              requiredDropFraction_ > 0.0 && requiredDropFraction_ < 1.0) ||
            requiredConsecutiveDeclines_ < 2u)
            throw std::invalid_argument(
                "first compression peak qualification settings are invalid");
    }

    std::optional<ShippingQualifiedCompressionPeak> observeAccepted(
        const double strokeM,
        const double forceN,
        const double contactSamples) {
        if (!(std::isfinite(strokeM) && strokeM >= 0.0) ||
            !(std::isfinite(forceN) && forceN >= 0.0) ||
            !(std::isfinite(contactSamples) && contactSamples >= 0.0))
            throw std::invalid_argument(
                "compression peak observation contains invalid values");
        if (qualified_) return qualified_;
        if (!(contactSamples > 0.0) || !(forceN > 0.0)) {
            // Do not join two disconnected contact episodes into one peak.
            havePrevious_ = false;
            havePreviousPrevious_ = false;
            hasCandidate_ = false;
            consecutiveDeclines_ = 0u;
            return std::nullopt;
        }

        ShippingCompressionForceSample current{
            sampleCount_++, strokeM, forceN, contactSamples};
        if (hasCandidate_) {
            if (current.forceN > candidate_.forceN) {
                candidate_ = current;
                consecutiveDeclines_ = 0u;
            } else if (havePrevious_ && current.forceN < previous_.forceN) {
                ++consecutiveDeclines_;
            } else {
                consecutiveDeclines_ = 0u;
            }
            const double drop = 1.0 - current.forceN / candidate_.forceN;
            if (consecutiveDeclines_ >= requiredConsecutiveDeclines_ &&
                drop >= requiredDropFraction_) {
                qualified_ = ShippingQualifiedCompressionPeak{
                    candidate_, current.sampleIndex, drop};
                return qualified_;
            }
        } else if (havePreviousPrevious_ && havePrevious_ &&
                   previous_.forceN > previousPrevious_.forceN &&
                   current.forceN < previous_.forceN) {
            candidate_ = previous_;
            hasCandidate_ = true;
            consecutiveDeclines_ = 1u;
        }
        previousPrevious_ = previous_;
        havePreviousPrevious_ = havePrevious_;
        previous_ = current;
        havePrevious_ = true;
        return std::nullopt;
    }

    [[nodiscard]] bool qualified() const noexcept {
        return qualified_.has_value();
    }
    [[nodiscard]] const std::optional<ShippingQualifiedCompressionPeak>&
    result() const noexcept { return qualified_; }
    [[nodiscard]] const ShippingCompressionForceSample& candidate() const noexcept {
        return candidate_;
    }
    [[nodiscard]] bool hasCandidate() const noexcept { return hasCandidate_; }
    [[nodiscard]] std::uint64_t sampleCount() const noexcept { return sampleCount_; }

private:
    double requiredDropFraction_ = 0.01;
    std::uint32_t requiredConsecutiveDeclines_ = 2u;
    std::uint64_t sampleCount_ = 0u;
    ShippingCompressionForceSample previousPrevious_{};
    ShippingCompressionForceSample previous_{};
    ShippingCompressionForceSample candidate_{};
    bool havePreviousPrevious_ = false;
    bool havePrevious_ = false;
    bool hasCandidate_ = false;
    std::uint32_t consecutiveDeclines_ = 0u;
    std::optional<ShippingQualifiedCompressionPeak> qualified_;
};

inline std::vector<ShippingCompressionIncrement> makeShippingCompressionSchedule(
    const double insideHeightM,
    const ShippingCompressionProtocol& protocol) {
    if (!(std::isfinite(insideHeightM) && insideHeightM > 0.0) ||
        !(std::isfinite(protocol.platenSpeedMPerS) && protocol.platenSpeedMPerS > 0.0) ||
        !(std::isfinite(protocol.incrementM) && protocol.incrementM > 0.0) ||
        !(std::isfinite(protocol.targetStrokeFractionOfInsideHeight) &&
          protocol.targetStrokeFractionOfInsideHeight > 0.0 &&
          protocol.targetStrokeFractionOfInsideHeight < 1.0))
        throw std::invalid_argument("shipping compression protocol is invalid");
    const double target = insideHeightM *
        protocol.targetStrokeFractionOfInsideHeight;
    const std::uint64_t count = static_cast<std::uint64_t>(
        std::ceil(target / protocol.incrementM));
    if (count == 0u || count > 100000u)
        throw std::length_error("shipping compression schedule exceeds its bounded step count");
    std::vector<ShippingCompressionIncrement> result;
    result.reserve(static_cast<std::size_t>(count));
    double consumed = 0.0;
    for (std::uint64_t step = 0u; step < count; ++step) {
        const double displacement = std::min(
            protocol.incrementM, target - consumed);
        if (!(std::isfinite(displacement) && displacement > 0.0))
            throw std::logic_error("shipping compression schedule generated an invalid increment");
        result.push_back({displacement,
                          displacement / protocol.platenSpeedMPerS});
        consumed += displacement;
    }
    if (std::abs(consumed - target) >
        8.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, target))
        throw std::logic_error("shipping compression schedule misses the requested stroke");
    return result;
}

inline double shippingImpactMassFromPeakForce(
    const double peakForceN,
    const double ratio,
    const double gravityMPerS2 = 9.80665) {
    if (!(std::isfinite(peakForceN) && peakForceN > 0.0) ||
        !(std::isfinite(ratio) && ratio > 0.0) ||
        !(std::isfinite(gravityMPerS2) && gravityMPerS2 > 0.0))
        throw std::invalid_argument("Fpeak-derived shipping impact mass needs positive finite inputs");
    const double mass = ratio * peakForceN / gravityMPerS2;
    if (!(std::isfinite(mass) && mass > 0.0))
        throw std::overflow_error("Fpeak-derived shipping impact mass is invalid");
    return mass;
}

inline std::array<double, 3> shippingImpactStartCenter(
    const ShippingAcceptedBounds& bounds,
    const double halfHeightM,
    const double dropHeightM,
    const double offsetZM) {
    if (!bounds.valid || !(std::isfinite(halfHeightM) && halfHeightM > 0.0) ||
        !(std::isfinite(dropHeightM) && dropHeightM > 0.0) ||
        !std::isfinite(offsetZM))
        throw std::invalid_argument("shipping impact pose inputs are invalid");
    const double centerX = 0.5 * (bounds.minimum.x + bounds.maximum.x);
    const double centerZ = 0.5 * (bounds.minimum.z + bounds.maximum.z) + offsetZM;
    const double centerY = bounds.maximum.y + dropHeightM + halfHeightM;
    if (!std::isfinite(centerX) || !std::isfinite(centerY) ||
        !std::isfinite(centerZ))
        throw std::overflow_error("shipping impact start pose is nonfinite");
    return {centerX, centerY, centerZ};
}

} // namespace numi::cardboard
