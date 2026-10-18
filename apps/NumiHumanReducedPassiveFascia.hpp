#pragma once

// CPU-only cooking of the existing Matter passive-ligament representation for
// the admitted two-layer, ordered fascia payload used by this caller: each layer
// has four first-band anchors followed by four second-band anchors, and the full
// node span is [all lower-layer nodes, all upper-layer nodes]. This is not a
// general fascia mesher. The caller must first establish exact source ancestry,
// region identity, and payload provenance. This cooks force rows, not FEM mass.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace numi::human::reduced_passive_fascia {

inline constexpr double kFitMinimumStretch = 1.005;
inline constexpr double kFitMaximumStretch = 1.12;
inline constexpr unsigned kFitSampleCount = 24u;
inline constexpr double kMaximumPeakNormalizedFitError = 0.03;

struct Vec3 { double x = 0.0, y = 0.0, z = 0.0; };
struct Node {
    Vec3 position;
    std::uint32_t sourceVertexIndex = 0u;
    std::uint32_t endpointFlags = 0u; // bit 0: first band, bit 1: second band
};
struct Tetrahedron { std::array<std::uint32_t, 4u> node{}; };
struct Material {
    double densityKgPerCubicMeter = 0.0;
    double matrixShearPascals = 0.0;
    double fiberK1Pascals = 0.0;
    double fiberK2 = 0.0;
    double tensionSmoothing = 0.0;
};
struct FiberLaw {
    double c3Pascals = 0.0;
    double c4 = 0.0;
    double c5Pascals = 0.0;
    double maximumStretch = kFitMaximumStretch;
};
struct Strip {
    std::uint32_t layer = 0u;
    std::uint32_t firstNode = 0u;
    std::uint32_t firstSourceVertex = 0u;
    std::uint32_t secondNode = 0u;
    std::uint32_t secondSourceVertex = 0u;
    double restLengthMeters = 0.0;
    double sourceVolumeCubicMeters = 0.0;
    double areaSquareMeters = 0.0;
    double fiberAlignment = 0.0;
    FiberLaw law;
    // Both errors are divided by the peak target stress, not pointwise stress.
    double peakNormalizedRmsError = 0.0;
    double peakNormalizedMaximumError = 0.0;
};
struct Result {
    std::vector<Strip> strips;
    double sourceVolumeCubicMeters = 0.0;
    double sourceMassKg = 0.0;
    // The reduced rows are massless force elements. Source mass remains with
    // the original rigid donors; this value is deliberately fixed at zero.
    double distributedStripMassKg = 0.0;
};

inline double tetrahedronVolume(const std::span<const Node> nodes,
                                const Tetrahedron& tet) noexcept {
    const auto& a = nodes[tet.node[0]].position;
    const auto& b = nodes[tet.node[1]].position;
    const auto& c = nodes[tet.node[2]].position;
    const auto& d = nodes[tet.node[3]].position;
    const double bx=b.x-a.x, by=b.y-a.y, bz=b.z-a.z;
    const double cx=c.x-a.x, cy=c.y-a.y, cz=c.z-a.z;
    const double dx=d.x-a.x, dy=d.y-a.y, dz=d.z-a.z;
    const double determinant =
        bx*(cy*dz-cz*dy) - by*(cx*dz-cz*dx) + bz*(cx*dy-cy*dx);
    return std::abs(determinant) / 6.0;
}

inline double sourceAxialStress(const Material& material,
                                double stretch,
                                double fiberCosineSquared) noexcept {
    const double inverseStretch = 1.0 / stretch;
    const double i4 = stretch * stretch * fiberCosineSquared +
        inverseStretch * (1.0 - fiberCosineSquared);
    const double x = i4 - 1.0;
    const double root = std::sqrt(x*x + material.tensionSmoothing*material.tensionSmoothing);
    const double positive = 0.5 * (x + root - material.tensionSmoothing);
    const double positiveDerivative = 0.5 * (1.0 + x / root);
    const double di4 = 2.0 * stretch * fiberCosineSquared -
        inverseStretch * inverseStretch * (1.0 - fiberCosineSquared);
    return material.matrixShearPascals *
               (stretch - inverseStretch*inverseStretch) +
        material.fiberK1Pascals * positive *
            std::exp(material.fiberK2 * positive * positive) *
            positiveDerivative * di4;
}

// evaluateOwnerStress must evaluate the existing passive-ligament owner law:
// bool(law, restLength, area, currentStretch, stressPascals). `output` is assigned
// only after every validation and fit succeeds, so a failed cook preserves it.
template <class EvaluateOwnerStress>
bool cook(std::span<const Node> nodes,
          std::span<const Tetrahedron> tetrahedra,
          const Material& material,
          Vec3 fiberAxis,
          EvaluateOwnerStress&& evaluateOwnerStress,
          Result& output,
          std::string& error) {
    error.clear();
    auto fail = [&](const char* message) {
        error = message;
        return false;
    };
    if (nodes.size() < 12u || nodes.size() % 2u != 0u || tetrahedra.empty())
        return fail("source region needs an even node count and tetrahedra");
    if (!(std::isfinite(material.densityKgPerCubicMeter) && material.densityKgPerCubicMeter > 0.0 &&
          std::isfinite(material.matrixShearPascals) && material.matrixShearPascals > 0.0 &&
          std::isfinite(material.fiberK1Pascals) && material.fiberK1Pascals > 0.0 &&
          std::isfinite(material.fiberK2) && material.fiberK2 > 0.0 &&
          std::isfinite(material.tensionSmoothing) && material.tensionSmoothing > 0.0))
        return fail("source material parameters are invalid");
    const double axisLength = std::hypot(fiberAxis.x, fiberAxis.y, fiberAxis.z);
    if (!(std::isfinite(axisLength) && axisLength > 1.0e-12))
        return fail("source fiber axis is invalid");
    fiberAxis.x /= axisLength; fiberAxis.y /= axisLength; fiberAxis.z /= axisLength;

    const std::uint32_t half = static_cast<std::uint32_t>(nodes.size() / 2u);
    std::array<std::vector<std::uint32_t>, 2u> firstByLayer, secondByLayer;
    for (std::uint32_t i = 0u; i < nodes.size(); ++i) {
        const auto flags = nodes[i].endpointFlags & 3u;
        if (flags == 3u) return fail("endpoint node has ambiguous attachment flags");
        const std::uint32_t layer = i < half ? 0u : 1u;
        if (flags == 1u) firstByLayer[layer].push_back(i);
        if (flags == 2u) secondByLayer[layer].push_back(i);
        const auto p = nodes[i].position;
        if (!(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)))
            return fail("source node position is non-finite");
    }

    std::vector<double> nodalVolume(nodes.size(), 0.0);
    double sourceVolume = 0.0;
    for (const auto& tet : tetrahedra) {
        for (unsigned c = 0u; c < 4u; ++c) {
            if (tet.node[c] >= nodes.size()) return fail("tetrahedron node is out of range");
            for (unsigned j = 0u; j < c; ++j)
                if (tet.node[c] == tet.node[j]) return fail("tetrahedron repeats a node");
        }
        const double volume = tetrahedronVolume(nodes, tet);
        if (!(std::isfinite(volume) && volume > 1.0e-15))
            return fail("source tetrahedron is degenerate");
        sourceVolume += volume;
        for (const auto node : tet.node) nodalVolume[node] += 0.25 * volume;
    }
    if (!(std::isfinite(sourceVolume) && sourceVolume > 0.0))
        return fail("source volume is invalid");

    std::vector<Strip> strips;
    strips.reserve(8u);
    std::vector<std::int32_t> nodeStrip(nodes.size(), -1);
    for (std::uint32_t layer = 0u; layer < 2u; ++layer) {
        const auto& first = firstByLayer[layer];
        const auto& second = secondByLayer[layer];
        if (first.size() != 4u || second.size() != 4u)
            return fail("each source layer must have four anchors per band");
        std::array<std::uint32_t, 4u> permutation{0u,1u,2u,3u}, best{};
        double bestCost = std::numeric_limits<double>::infinity();
        do {
            double cost = 0.0;
            for (unsigned i = 0u; i < 4u; ++i) {
                const auto& a = nodes[first[i]].position;
                const auto& b = nodes[second[permutation[i]]].position;
                const double x=a.x-b.x, y=a.y-b.y, z=a.z-b.z;
                cost += x*x + y*y + z*z;
            }
            if (cost < bestCost) { bestCost = cost; best = permutation; }
        } while (std::next_permutation(permutation.begin(), permutation.end()));
        for (unsigned i = 0u; i < 4u; ++i) {
            const auto a = first[i], b = second[best[i]];
            const auto& pa = nodes[a].position;
            const auto& pb = nodes[b].position;
            const double x=pb.x-pa.x, y=pb.y-pa.y, z=pb.z-pa.z;
            const double length = std::sqrt(x*x + y*y + z*z);
            if (!(std::isfinite(length) && length > 1.0e-4))
                return fail("paired fascia strip endpoints coincide");
            const auto index = static_cast<std::int32_t>(strips.size());
            strips.push_back(Strip{.layer=layer, .firstNode=a,
                .firstSourceVertex=nodes[a].sourceVertexIndex, .secondNode=b,
                .secondSourceVertex=nodes[b].sourceVertexIndex,
                .restLengthMeters=length});
            nodeStrip[a] = index; nodeStrip[b] = index;
        }
    }

    for (std::uint32_t node = 0u; node < nodes.size(); ++node) {
        if (nodeStrip[node] >= 0) {
            strips[static_cast<std::size_t>(nodeStrip[node])].sourceVolumeCubicMeters +=
                nodalVolume[node];
            continue;
        }
        const std::uint32_t layer = node < half ? 0u : 1u;
        const auto& x = nodes[node].position;
        double bestDistance = std::numeric_limits<double>::infinity();
        std::size_t bestStrip = strips.size();
        for (std::size_t si = 0u; si < strips.size(); ++si) {
            const auto& strip = strips[si];
            if (strip.layer != layer) continue;
            const auto& a = nodes[strip.firstNode].position;
            const auto& b = nodes[strip.secondNode].position;
            const double dx=b.x-a.x, dy=b.y-a.y, dz=b.z-a.z;
            const double denominator=dx*dx+dy*dy+dz*dz;
            const double u=std::clamp(((x.x-a.x)*dx+(x.y-a.y)*dy+(x.z-a.z)*dz)/
                                      denominator, 0.0, 1.0);
            const double ex=x.x-(a.x+u*dx), ey=x.y-(a.y+u*dy), ez=x.z-(a.z+u*dz);
            const double distance=ex*ex+ey*ey+ez*ez;
            if (distance < bestDistance) { bestDistance=distance; bestStrip=si; }
        }
        if (!(bestStrip < strips.size() && std::isfinite(bestDistance)))
            return fail("unanchored source node has no strip in its layer");
        strips[bestStrip].sourceVolumeCubicMeters += nodalVolume[node];
    }

    double distributedVolume = 0.0;
    for (const auto& strip : strips) distributedVolume += strip.sourceVolumeCubicMeters;
    if (!(strips.size() == 8u &&
          std::abs(distributedVolume - sourceVolume) <=
              std::max(1.0e-14, sourceVolume * 1.0e-10)))
        return fail("strip assignment does not conserve source volume");

    constexpr unsigned fitCandidates = 799u;
    std::array<double, kFitSampleCount> target{}, basis{};
    for (auto& strip : strips) {
        strip.areaSquareMeters = strip.sourceVolumeCubicMeters / strip.restLengthMeters;
        const auto& a = nodes[strip.firstNode].position;
        const auto& b = nodes[strip.secondNode].position;
        const double alignment = std::clamp(std::abs(
            ((b.x-a.x)*fiberAxis.x + (b.y-a.y)*fiberAxis.y +
             (b.z-a.z)*fiberAxis.z) / strip.restLengthMeters), 0.0, 1.0);
        strip.fiberAlignment = alignment;
        double peak = 0.0;
        for (unsigned j = 0u; j < kFitSampleCount; ++j) {
            const double stretch = kFitMinimumStretch +
                (kFitMaximumStretch-kFitMinimumStretch) *
                double(j) / double(kFitSampleCount-1u);
            target[j] = sourceAxialStress(material, stretch, alignment*alignment);
            peak = std::max(peak, std::abs(target[j]));
        }
        if (!(std::isfinite(peak) && peak > 0.0))
            return fail("source axial stress curve has no positive scale");
        double bestError = std::numeric_limits<double>::infinity();
        for (unsigned candidate = 0u; candidate < fitCandidates; ++candidate) {
            const double c4 = 0.05 + 0.025 * double(candidate);
            double aa = 0.0, ay = 0.0;
            for (unsigned j = 0u; j < kFitSampleCount; ++j) {
                const double stretch = kFitMinimumStretch +
                    (kFitMaximumStretch-kFitMinimumStretch) *
                    double(j) / double(kFitSampleCount-1u);
                basis[j] = std::expm1(c4 * (stretch-1.0));
                aa += basis[j]*basis[j]; ay += basis[j]*target[j];
            }
            if (!(aa > 0.0)) continue;
            const double c3 = ay / aa;
            double error2 = 0.0;
            for (unsigned j = 0u; j < kFitSampleCount; ++j) {
                const double residual = c3*basis[j]-target[j];
                error2 += residual*residual;
            }
            if (c3 > 0.0 && error2 < bestError) {
                bestError = error2;
                strip.law.c3Pascals = c3;
                strip.law.c4 = c4;
            }
        }
        strip.law.maximumStretch = kFitMaximumStretch;
        strip.law.c5Pascals = strip.law.c3Pascals * strip.law.c4 *
            std::exp(strip.law.c4*(kFitMaximumStretch-1.0));
        if (!(std::isfinite(strip.areaSquareMeters) && strip.areaSquareMeters > 0.0 &&
              std::isfinite(strip.law.c3Pascals) && strip.law.c3Pascals > 0.0 &&
              std::isfinite(strip.law.c4) && strip.law.c4 > 0.0 &&
              std::isfinite(strip.law.c5Pascals) && strip.law.c5Pascals > 0.0))
            return fail("fitted strip material is invalid");
        double error2 = 0.0, maxError = 0.0;
        for (unsigned j = 0u; j < kFitSampleCount; ++j) {
            const double stretch = kFitMinimumStretch +
                (kFitMaximumStretch-kFitMinimumStretch) *
                double(j) / double(kFitSampleCount-1u);
            double stress = 0.0;
            if (!evaluateOwnerStress(strip.law, strip.restLengthMeters,
                                     strip.areaSquareMeters, stretch, stress) ||
                !std::isfinite(stress))
                return fail("existing passive-ligament evaluator rejected fitted strip");
            const double residual = stress - target[j];
            error2 += residual*residual;
            maxError = std::max(maxError, std::abs(residual));
        }
        strip.peakNormalizedRmsError = std::sqrt(error2 / kFitSampleCount) / peak;
        strip.peakNormalizedMaximumError = maxError / peak;
        if (!(std::isfinite(strip.peakNormalizedRmsError) &&
              std::isfinite(strip.peakNormalizedMaximumError) &&
              strip.peakNormalizedMaximumError <= kMaximumPeakNormalizedFitError))
            return fail("fitted strip exceeds peak-normalized source-curve error bound");
    }

    Result result;
    result.strips = std::move(strips);
    result.sourceVolumeCubicMeters = sourceVolume;
    result.sourceMassKg = sourceVolume * material.densityKgPerCubicMeter;
    result.distributedStripMassKg = 0.0;
    if (!(std::isfinite(result.sourceMassKg) && result.sourceMassKg > 0.0))
        return fail("source fascia mass is invalid");
    output = std::move(result);
    return true;
}

} // namespace numi::human::reduced_passive_fascia
