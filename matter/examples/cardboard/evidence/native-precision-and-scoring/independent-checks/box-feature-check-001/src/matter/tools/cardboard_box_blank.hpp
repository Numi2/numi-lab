#pragma once

// Source-described FEFCO 0201 corrugated-box blank geometry. The sheet uses
// the existing finite-glue, sinusoidal cross-section as its laminate section;
// the blank outline and slots are cut in that 3D volume. Score lines are
// explicit layout metadata only: no weakened hinge, crushing, damage, or
// folded node positions are authored here. All dimensions are SI metres.

#include "cardboard_glue_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace numi::cardboard {

struct FEFCO0201BlankConfig {
    // Dieline score-to-score dimensions. These are not inside dimensions;
    // score allowance depends on board and score process and is not inferred.
    double panelLength = 0.0;
    double panelWidth = 0.0;
    double wallHeight = 0.0;
    double manufacturerJointWidth = 0.0;
    // A positive idealized cut kerf separates adjacent flap material. The
    // source describes paired cuts/removal but provides no nominal kerf.
    double slotKerf = 0.0;
    std::uint32_t subdivisionsPerBand = 1u;
};

struct BoxBlankPoint {
    double x = 0.0; // around the four box panels and manufacturer joint
    double y = 0.0; // bottom flap, body wall, top flap
    double z = 0.0; // board caliper direction
};

struct BoxBlankTetrahedron {
    std::array<std::uint32_t, 4> nodes{};
    CellMaterial material = CellMaterial::liner;
    double frameAngle = 0.0;
    std::uint32_t panelBand = 0u;
};

enum class BoxScoreKind : std::uint32_t {
    manufacturerJointRoot = 0u,
    bodyCorner = 1u,
    bottomFlap = 2u,
    topFlap = 3u,
};

struct BoxScoreLine {
    BoxScoreKind kind = BoxScoreKind::bodyCorner;
    std::uint32_t panelIndex = 0u;
    BoxBlankPoint first{};
    BoxBlankPoint second{};
    double targetFoldDegrees = 90.0;
    bool physicallyScored = false;
};

struct BoxSlot {
    std::uint32_t panelBoundary = 0u;
    bool top = false;
    double xLeft = 0.0;
    double xRight = 0.0;
    double yStart = 0.0;
    double yEnd = 0.0;
};

struct BoxBlankPanel {
    std::uint32_t index = 0u;
    double xStart = 0.0;
    double xEnd = 0.0;
    double yStart = 0.0;
    double yEnd = 0.0;
    bool longWall = false;
};

enum class BoxAssemblyFeatureKind : std::uint32_t {
    panel = 0u,
    scoreLine = 1u,
    manufacturerJoint = 2u,
    boxEnd = 3u,
};

struct BoxAssemblyFeatureReference {
    BoxAssemblyFeatureKind kind = BoxAssemblyFeatureKind::panel;
    std::uint32_t index = 0u;
};

inline constexpr std::uint32_t kNoBoxScoreLineIndex =
    std::numeric_limits<std::uint32_t>::max();

inline const char* boxAssemblyFeatureKindName(
    const BoxAssemblyFeatureKind kind) noexcept {
    switch (kind) {
        case BoxAssemblyFeatureKind::panel: return "panel";
        case BoxAssemblyFeatureKind::scoreLine: return "score_line";
        case BoxAssemblyFeatureKind::manufacturerJoint: return "manufacturer_joint";
        case BoxAssemblyFeatureKind::boxEnd: return "box_end";
    }
    return "unknown";
}

inline const char* boxAssemblyFeatureIndexSemantics(
    const BoxAssemblyFeatureKind kind) noexcept {
    switch (kind) {
        case BoxAssemblyFeatureKind::panel:
            return "zero-based index into blank.panels";
        case BoxAssemblyFeatureKind::scoreLine:
            return "zero-based index into blank.scoreLines";
        case BoxAssemblyFeatureKind::manufacturerJoint:
            return "0 = the single tab region x=[0, joint_width]";
        case BoxAssemblyFeatureKind::boxEnd:
            return "0 = bottom end; 1 = top end";
    }
    return "unknown feature index semantics";
}

struct BoxAssemblyAction {
    enum class Kind : std::uint32_t {
        erectWallPanel = 0u,
        bondManufacturerJoint = 1u,
        foldFlap = 2u,
        closeEndWithTapeOrAdhesive = 3u,
    };

    Kind kind = Kind::erectWallPanel;
    BoxAssemblyFeatureReference feature{};
    // Wall-erection targets name both the panel and its score-line hinge.
    // Other actions use the feature reference itself as their score/region ID.
    std::uint32_t hingeScoreLineIndex = kNoBoxScoreLineIndex;
    double targetAngleDegrees = 0.0;
    const char* stage = "";
    bool executed = false;
};

struct BoxAssemblySchedule {
    std::vector<BoxAssemblyAction> actions;
    const char* executionState = "targets-authored-not-executed";
};

struct FEFCO0201Blank {
    FEFCO0201BlankConfig config;
    double boardCaliper = 0.0;
    double perimeterSpan = 0.0;
    double blankHeight = 0.0;
    double topFlapDepth = 0.0;
    double bottomScoreY = 0.0;
    double topScoreY = 0.0;
    std::vector<BoxBlankPoint> points;
    std::vector<BoxBlankTetrahedron> tetrahedra;
    std::array<double, 3> volumeByMaterial{};
    std::vector<BoxBlankPanel> panels;
    std::vector<BoxScoreLine> scoreLines;
    std::vector<BoxSlot> slots;
    std::array<double, 3> clippedSectionAreaByMaterial{};
    // Integrated section area times the available y span in each x slab;
    // the section already runs around the full blank perimeter.
    std::array<double, 3> plannedVolumeByMaterial{};
    BoxAssemblySchedule assembly;
    std::string geometryStatus =
        "flat, preformed corrugated blank; score centerlines are metadata only";
};

namespace cardboard_box_detail {

constexpr double kCoordinateScale = 1.0e12;
constexpr double kGeometryTolerance = 2.0e-12;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct Key2 {
    std::int64_t x = 0;
    std::int64_t z = 0;
    friend bool operator<(const Key2& a, const Key2& b) {
        return a.x < b.x || (a.x == b.x && a.z < b.z);
    }
};

struct Key3 {
    std::uint32_t sectionPoint = 0u;
    std::uint32_t station = 0u;
    friend bool operator<(const Key3& a, const Key3& b) {
        return a.sectionPoint < b.sectionPoint ||
            (a.sectionPoint == b.sectionPoint && a.station < b.station);
    }
};

struct SectionTriangle {
    std::array<std::uint32_t, 3> points{};
    CellMaterial material = CellMaterial::liner;
    double frameAngle = 0.0;
    std::uint32_t slab = 0u;
};

class SectionPointRegistry {
public:
    std::uint32_t add(const GlueMeshPoint& point) {
        const Key2 key{
            static_cast<std::int64_t>(std::llround(point.x * kCoordinateScale)),
            static_cast<std::int64_t>(std::llround(point.z * kCoordinateScale)),
        };
        const auto [it, inserted] = indices_.emplace(
            key, static_cast<std::uint32_t>(points_.size()));
        if (inserted) points_.push_back(point);
        return it->second;
    }

    [[nodiscard]] const std::vector<GlueMeshPoint>& points() const {
        return points_;
    }

private:
    std::map<Key2, std::uint32_t> indices_;
    std::vector<GlueMeshPoint> points_;
};

inline double signedArea2(const GlueMeshPoint& a,
                          const GlueMeshPoint& b,
                          const GlueMeshPoint& c) {
    return (b.x - a.x) * (c.z - a.z) -
        (b.z - a.z) * (c.x - a.x);
}

inline std::vector<GlueMeshPoint> clipVertical(
    const std::vector<GlueMeshPoint>& polygon,
    const double boundary,
    const bool keepGreater) {
    std::vector<GlueMeshPoint> output;
    if (polygon.empty()) return output;
    const auto inside = [&](const GlueMeshPoint& point) {
        return keepGreater ? point.x >= boundary - kGeometryTolerance
                           : point.x <= boundary + kGeometryTolerance;
    };
    const auto intersect = [&](const GlueMeshPoint& a,
                               const GlueMeshPoint& b) {
        const double denominator = b.x - a.x;
        if (std::abs(denominator) <= 1.0e-20)
            return GlueMeshPoint{boundary, a.z};
        const double t = std::clamp((boundary - a.x) / denominator, 0.0, 1.0);
        return GlueMeshPoint{boundary, a.z + t * (b.z - a.z)};
    };
    GlueMeshPoint previous = polygon.back();
    bool previousInside = inside(previous);
    for (const GlueMeshPoint& current : polygon) {
        const bool currentInside = inside(current);
        if (currentInside != previousInside)
            output.push_back(intersect(previous, current));
        if (currentInside) output.push_back(current);
        previous = current;
        previousInside = currentInside;
    }
    return output;
}

inline std::vector<GlueMeshPoint> clipSectionTriangle(
    const ConformingGlueMesh& section,
    const GlueMeshTriangle& triangle,
    const double left,
    const double right) {
    std::vector<GlueMeshPoint> polygon;
    polygon.reserve(5u);
    for (const std::uint32_t point : triangle.points)
        polygon.push_back(section.points.at(point));
    polygon = clipVertical(polygon, left, true);
    polygon = clipVertical(polygon, right, false);
    if (polygon.size() < 3u) return {};

    // Consecutive duplicate points arise when a cut coincides with an
    // existing section vertex; remove them before fan triangulation.
    std::vector<GlueMeshPoint> clean;
    clean.reserve(polygon.size());
    for (const auto& point : polygon) {
        if (clean.empty() || std::hypot(point.x - clean.back().x,
                                       point.z - clean.back().z) > 1.0e-13)
            clean.push_back(point);
    }
    if (clean.size() > 2u &&
        std::hypot(clean.front().x - clean.back().x,
                   clean.front().z - clean.back().z) <= 1.0e-13)
        clean.pop_back();
    return clean.size() >= 3u ? clean : std::vector<GlueMeshPoint>{};
}

inline double signedSixVolume(const BoxBlankPoint& a,
                              const BoxBlankPoint& b,
                              const BoxBlankPoint& c,
                              const BoxBlankPoint& d) {
    const std::array<double, 3> ab{b.x - a.x, b.y - a.y, b.z - a.z};
    const std::array<double, 3> ac{c.x - a.x, c.y - a.y, c.z - a.z};
    const std::array<double, 3> ad{d.x - a.x, d.y - a.y, d.z - a.z};
    return ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) -
        ab[1] * (ac[0] * ad[2] - ac[2] * ad[0]) +
        ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]);
}

inline void orientPositive(BoxBlankTetrahedron& tetrahedron,
                           const std::vector<BoxBlankPoint>& points) {
    const double sixVolume = signedSixVolume(
        points[tetrahedron.nodes[0]], points[tetrahedron.nodes[1]],
        points[tetrahedron.nodes[2]], points[tetrahedron.nodes[3]]);
    if (!std::isfinite(sixVolume) || std::abs(sixVolume) <= 1.0e-24)
        throw std::runtime_error("box blank generated a degenerate tetrahedron");
    if (sixVolume < 0.0)
        std::swap(tetrahedron.nodes[0], tetrahedron.nodes[1]);
}

inline std::vector<double> extrusionStations(
    const double start,
    const double end,
    const std::uint32_t subdivisions) {
    std::vector<double> result;
    result.reserve(static_cast<std::size_t>(subdivisions) + 1u);
    for (std::uint32_t index = 0u; index <= subdivisions; ++index)
        result.push_back(start + (end - start) *
            static_cast<double>(index) / subdivisions);
    return result;
}

inline bool slabIsRemoved(const double midpoint,
                          const std::vector<std::pair<double, double>>& slots) {
    for (const auto& [left, right] : slots)
        if (midpoint > left && midpoint < right) return true;
    return false;
}

} // namespace cardboard_box_detail

inline BoxAssemblySchedule makeFEFCO0201AssemblySchedule(
    const FEFCO0201Blank& blank) {
    BoxAssemblySchedule schedule;
    // The turns and bonds are source-described targets, not solved motions.
    // The order closes the tube, bonds its manufacturer's joint, then closes
    // bottom and top ends with inner-flap-before-outer-flap sequencing.
    for (std::uint32_t score = 0u; score < blank.scoreLines.size(); ++score) {
        const auto& line = blank.scoreLines[score];
        if (line.kind == BoxScoreKind::manufacturerJointRoot ||
            line.kind == BoxScoreKind::bodyCorner) {
            const std::uint32_t panel =
                line.kind == BoxScoreKind::manufacturerJointRoot
                    ? 0u : line.panelIndex + 1u;
            schedule.actions.push_back({
                BoxAssemblyAction::Kind::erectWallPanel,
                {BoxAssemblyFeatureKind::panel, panel},
                score,
                line.targetFoldDegrees,
                "form four-wall tube from flat blank",
                false,
            });
        }
    }
    schedule.actions.push_back({
        BoxAssemblyAction::Kind::bondManufacturerJoint,
        {BoxAssemblyFeatureKind::manufacturerJoint, 0u},
        kNoBoxScoreLineIndex,
        0.0,
        "overlap tab and bond the manufacturer's joint",
        false,
    });
    // Panel indices 1 and 3 have score width W; indices 0 and 2 are the
    // lengthwise panels whose flaps close last in a conventional RSC sequence.
    for (const bool top : {false, true}) {
        for (const std::uint32_t panel : {1u, 3u, 0u, 2u}) {
            const BoxScoreKind flapScoreKind = top
                ? BoxScoreKind::topFlap : BoxScoreKind::bottomFlap;
            const auto score = std::find_if(
                blank.scoreLines.begin(), blank.scoreLines.end(),
                [=](const BoxScoreLine& line) {
                    return line.kind == flapScoreKind && line.panelIndex == panel;
                });
            if (score == blank.scoreLines.end())
                throw std::invalid_argument(
                    "assembly schedule cannot resolve a flap score-line reference");
            const auto scoreIndex = static_cast<std::uint32_t>(
                std::distance(blank.scoreLines.begin(), score));
            schedule.actions.push_back({
                BoxAssemblyAction::Kind::foldFlap,
                {BoxAssemblyFeatureKind::scoreLine, scoreIndex},
                kNoBoxScoreLineIndex,
                top ? -90.0 : 90.0,
                top ? "close top flap" : "close bottom flap",
                false,
            });
        }
        schedule.actions.push_back({
            BoxAssemblyAction::Kind::closeEndWithTapeOrAdhesive,
            {BoxAssemblyFeatureKind::boxEnd, top ? 1u : 0u},
            kNoBoxScoreLineIndex,
            0.0,
            top ? "seal top flaps" : "seal bottom flaps",
            false,
        });
    }
    return schedule;
}

inline FEFCO0201Blank buildFEFCO0201Blank(
    const FEFCO0201BlankConfig& config,
    const ConformingGlueMesh& section) {
    using namespace cardboard_box_detail;
    const auto positive = [](const double value) {
        return std::isfinite(value) && value > 0.0;
    };
    if (!positive(config.panelLength) || !positive(config.panelWidth) ||
        !positive(config.wallHeight) || !positive(config.manufacturerJointWidth) ||
        !positive(config.slotKerf) || config.subdivisionsPerBand == 0u ||
        config.subdivisionsPerBand > 64u)
        throw std::invalid_argument("FEFCO 0201 dimensions, slot kerf, and subdivisions must be positive");
    const double perimeter = config.manufacturerJointWidth +
        2.0 * (config.panelLength + config.panelWidth);
    const double flapDepth = 0.5 * config.panelWidth;
    if (!(config.slotKerf < 0.25 * std::min(config.panelLength,
                                           config.panelWidth)))
        throw std::invalid_argument("slot kerf is too wide for the smallest panel");
    if (section.points.empty() || section.triangles.empty() ||
        std::abs(section.config.length - perimeter) > 1.0e-10 * perimeter)
        throw std::invalid_argument("cross-section must span the full FEFCO blank perimeter");
    if (!positive(section.config.caliper))
        throw std::invalid_argument("cross-section board caliper is invalid");

    FEFCO0201Blank blank;
    blank.config = config;
    blank.boardCaliper = section.config.caliper;
    blank.perimeterSpan = perimeter;
    blank.blankHeight = config.wallHeight + config.panelWidth;
    blank.topFlapDepth = flapDepth;
    blank.bottomScoreY = flapDepth;
    blank.topScoreY = flapDepth + config.wallHeight;

    const std::array<double, 5> panelEdges{
        config.manufacturerJointWidth,
        config.manufacturerJointWidth + config.panelLength,
        config.manufacturerJointWidth + config.panelLength + config.panelWidth,
        config.manufacturerJointWidth + 2.0 * config.panelLength + config.panelWidth,
        perimeter,
    };
    for (std::uint32_t panel = 0u; panel < 4u; ++panel) {
        blank.panels.push_back({
            panel,
            panelEdges[panel],
            panelEdges[panel + 1u],
            flapDepth,
            flapDepth + config.wallHeight,
            panel == 0u || panel == 2u,
        });
    }

    // The non-extended manufacturer's joint is a body-height-only tab. The
    // four internal score axes comprise its root plus three wall corners.
    blank.scoreLines.push_back({
        BoxScoreKind::manufacturerJointRoot, 0u,
        {config.manufacturerJointWidth, flapDepth, 0.0},
        {config.manufacturerJointWidth, flapDepth + config.wallHeight, 0.0},
        90.0, false,
    });
    for (std::uint32_t boundary = 1u; boundary <= 3u; ++boundary) {
        const double x = panelEdges[boundary];
        blank.scoreLines.push_back({
            BoxScoreKind::bodyCorner, boundary - 1u,
            {x, flapDepth, 0.0},
            {x, flapDepth + config.wallHeight, 0.0},
            90.0, false,
        });
    }
    for (std::uint32_t panel = 0u; panel < 4u; ++panel) {
        const auto& bounds = blank.panels[panel];
        blank.scoreLines.push_back({
            BoxScoreKind::bottomFlap, panel,
            {bounds.xStart, flapDepth, 0.0},
            {bounds.xEnd, flapDepth, 0.0},
            90.0, false,
        });
        blank.scoreLines.push_back({
            BoxScoreKind::topFlap, panel,
            {bounds.xStart, flapDepth + config.wallHeight, 0.0},
            {bounds.xEnd, flapDepth + config.wallHeight, 0.0},
            -90.0, false,
        });
    }

    // Three internal cuts separate neighboring flap pairs at both ends. A
    // non-extended glue tab carries no flaps, so the tab edges are cut by the
    // outer blank contour rather than by an additional interior slot.
    std::vector<std::pair<double, double>> slotIntervals;
    for (std::uint32_t boundary = 1u; boundary <= 3u; ++boundary) {
        const double center = panelEdges[boundary];
        const double left = center - 0.5 * config.slotKerf;
        const double right = center + 0.5 * config.slotKerf;
        slotIntervals.emplace_back(left, right);
        blank.slots.push_back({
            boundary - 1u, false, left, right, 0.0, flapDepth,
        });
        blank.slots.push_back({
            boundary - 1u, true, left, right,
            flapDepth + config.wallHeight, blank.blankHeight,
        });
    }

    std::vector<double> slabEdges{0.0, perimeter, config.manufacturerJointWidth};
    for (std::uint32_t boundary = 1u; boundary <= 3u; ++boundary)
        slabEdges.push_back(panelEdges[boundary]);
    for (const auto& [left, right] : slotIntervals) {
        slabEdges.push_back(left);
        slabEdges.push_back(right);
    }
    std::sort(slabEdges.begin(), slabEdges.end());
    slabEdges.erase(std::unique(slabEdges.begin(), slabEdges.end(),
        [](const double a, const double b) {
            return std::abs(a - b) <= 1.0e-13;
        }), slabEdges.end());

    SectionPointRegistry sectionRegistry;
    std::vector<SectionTriangle> sectionTriangles;
    for (std::uint32_t slab = 0u; slab + 1u < slabEdges.size(); ++slab) {
        const double left = slabEdges[slab];
        const double right = slabEdges[slab + 1u];
        for (const auto& sourceTriangle : section.triangles) {
            const auto polygon = clipSectionTriangle(
                section, sourceTriangle, left, right);
            if (polygon.size() < 3u) continue;
            std::vector<std::uint32_t> ids;
            ids.reserve(polygon.size());
            for (const auto& point : polygon) ids.push_back(sectionRegistry.add(point));
            for (std::size_t fan = 1u; fan + 1u < ids.size(); ++fan) {
                std::array<std::uint32_t, 3> triangle{
                    ids[0], ids[fan], ids[fan + 1u],
                };
                const double area2 = signedArea2(
                    sectionRegistry.points()[triangle[0]],
                    sectionRegistry.points()[triangle[1]],
                    sectionRegistry.points()[triangle[2]]);
                if (std::abs(area2) <= 1.0e-22) continue;
                if (area2 < 0.0) std::swap(triangle[1], triangle[2]);
                sectionTriangles.push_back({
                    triangle, sourceTriangle.material,
                    sourceTriangle.frameAngle, slab,
                });
                blank.clippedSectionAreaByMaterial[
                    static_cast<std::size_t>(sourceTriangle.material)] +=
                    0.5 * std::abs(area2);
            }
        }
    }
    if (sectionTriangles.empty())
        throw std::runtime_error("cross-section clipping removed the entire blank");
    for (std::size_t material = 0u;
         material < blank.clippedSectionAreaByMaterial.size(); ++material) {
        const double expected = section.crossSectionAreaByMaterial[material];
        if (std::abs(blank.clippedSectionAreaByMaterial[material] - expected) >
            2.0e-10 * std::max(expected, 1.0e-18))
            throw std::runtime_error(
                "box clipping changed cross-section material area");
    }

    std::vector<double> stations;
    const auto appendBand = [&](const double start, const double end) {
        auto band = extrusionStations(
            start, end, config.subdivisionsPerBand);
        if (!stations.empty()) band.erase(band.begin());
        stations.insert(stations.end(), band.begin(), band.end());
    };
    appendBand(0.0, flapDepth);
    appendBand(flapDepth, flapDepth + config.wallHeight);
    appendBand(flapDepth + config.wallHeight, blank.blankHeight);

    std::vector<double> slabMidpoints(slabEdges.size() - 1u);
    for (std::size_t slab = 0u; slab < slabMidpoints.size(); ++slab)
        slabMidpoints[slab] = 0.5 * (slabEdges[slab] + slabEdges[slab + 1u]);

    std::vector<std::array<double, 3>> sectionAreaBySlab(slabMidpoints.size());
    for (const auto& triangle : sectionTriangles) {
        const auto& a = sectionRegistry.points()[triangle.points[0]];
        const auto& b = sectionRegistry.points()[triangle.points[1]];
        const auto& c = sectionRegistry.points()[triangle.points[2]];
        sectionAreaBySlab[triangle.slab][
            static_cast<std::size_t>(triangle.material)] +=
            0.5 * std::abs(signedArea2(a, b, c));
    }
    for (std::size_t slab = 0u; slab < slabMidpoints.size(); ++slab) {
        const double x = slabMidpoints[slab];
        const bool carriesFlaps = x >= config.manufacturerJointWidth &&
            !slabIsRemoved(x, slotIntervals);
        const double availableY = config.wallHeight +
            (carriesFlaps ? config.panelWidth : 0.0);
        for (std::size_t material = 0u;
             material < blank.plannedVolumeByMaterial.size(); ++material)
            blank.plannedVolumeByMaterial[material] +=
                sectionAreaBySlab[slab][material] * availableY;
    }

    std::map<Key3, std::uint32_t> volumePointIndices;
    const auto pointAt = [&](const std::uint32_t sectionPoint,
                             const std::uint32_t station) {
        const Key3 key{sectionPoint, station};
        const auto found = volumePointIndices.find(key);
        if (found != volumePointIndices.end()) return found->second;
        const auto& sourcePoint = sectionRegistry.points().at(sectionPoint);
        if (blank.points.size() >= std::numeric_limits<std::uint32_t>::max())
            throw std::length_error("box blank exceeds uint32 node capacity");
        const auto index = static_cast<std::uint32_t>(blank.points.size());
        blank.points.push_back({sourcePoint.x, stations.at(station), sourcePoint.z});
        volumePointIndices.emplace(key, index);
        return index;
    };

    const auto addPrism = [&](const SectionTriangle& sectionTriangle,
                              const std::uint32_t lowerStation,
                              const std::uint32_t upperStation) {
        auto base = sectionTriangle.points;
        std::sort(base.begin(), base.end());
        const std::uint32_t a = pointAt(base[0], lowerStation);
        const std::uint32_t b = pointAt(base[1], lowerStation);
        const std::uint32_t c = pointAt(base[2], lowerStation);
        const std::uint32_t A = pointAt(base[0], upperStation);
        const std::uint32_t B = pointAt(base[1], upperStation);
        const std::uint32_t C = pointAt(base[2], upperStation);
        const std::array<std::array<std::uint32_t, 4>, 3> split{{
            {a, b, c, C},
            {a, b, B, C},
            {a, A, B, C},
        }};
        for (const auto& nodes : split) {
            BoxBlankTetrahedron tetrahedron{
                nodes, sectionTriangle.material,
                sectionTriangle.frameAngle, sectionTriangle.slab,
            };
            orientPositive(tetrahedron, blank.points);
            const double volume = signedSixVolume(
                blank.points[tetrahedron.nodes[0]],
                blank.points[tetrahedron.nodes[1]],
                blank.points[tetrahedron.nodes[2]],
                blank.points[tetrahedron.nodes[3]]) / 6.0;
            blank.volumeByMaterial[static_cast<std::size_t>(tetrahedron.material)] += volume;
            blank.tetrahedra.push_back(tetrahedron);
        }
    };

    const std::uint32_t topBandEnd = config.subdivisionsPerBand;
    const std::uint32_t bodyBandEnd = 2u * config.subdivisionsPerBand;
    for (std::uint32_t band = 0u; band + 1u < stations.size(); ++band) {
        const double midpointY = 0.5 * (stations[band] + stations[band + 1u]);
        const bool inBottomFlap = band < topBandEnd;
        const bool inBody = band >= topBandEnd && band < bodyBandEnd;
        for (const auto& triangle : sectionTriangles) {
            const double midpointX = slabMidpoints[triangle.slab];
            if (inBody) {
                addPrism(triangle, band, band + 1u);
                continue;
            }
            if (midpointX < config.manufacturerJointWidth ||
                slabIsRemoved(midpointX, slotIntervals))
                continue;
            if (inBottomFlap && midpointY < flapDepth + kGeometryTolerance)
                addPrism(triangle, band, band + 1u);
            else if (!inBottomFlap && midpointY > flapDepth + config.wallHeight -
                     kGeometryTolerance)
                addPrism(triangle, band, band + 1u);
        }
    }

    if (blank.tetrahedra.empty())
        throw std::runtime_error("FEFCO 0201 blank contains no tetrahedra");
    for (std::size_t material = 0u;
         material < blank.volumeByMaterial.size(); ++material) {
        const double expected = blank.plannedVolumeByMaterial[material];
        if (std::abs(blank.volumeByMaterial[material] - expected) >
            2.0e-9 * std::max(expected, 1.0e-18))
            throw std::runtime_error(
                "tetrahedral volume differs from per-slab blank integration");
    }
    blank.assembly = makeFEFCO0201AssemblySchedule(blank);
    return blank;
}

} // namespace numi::cardboard
