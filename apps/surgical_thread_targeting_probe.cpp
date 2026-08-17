#include "metalrobo/SurgicalThreadTargeting.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Point = metalrobo::SurgicalThreadTargetPoint;

double dot(const Point& left, const Point& right) {
    return left[0] * right[0] +
        left[1] * right[1] +
        left[2] * right[2];
}

Point cross(const Point& left, const Point& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

double length(const Point& value) {
    return std::sqrt(dot(value, value));
}

Point subtract(const Point& left, const Point& right) {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool samePoint(const Point& left, const Point& right) {
    return left == right;
}

} // namespace

int main() {
    try {
        std::vector<Point> threadNodes;
        threadNodes.reserve(101u);
        for (std::uint32_t node = 0u; node <= 100u; ++node) {
            threadNodes.push_back({
                1.0e-3 * static_cast<double>(node),
                0.0,
                5.0e-3,
            });
        }
        std::vector<Point> tissueNodes{
            {-1.0e-2, -2.0e-2, 0.0},
            {1.1e-1, -2.0e-2, 0.0},
            {1.1e-1, 2.0e-2, 0.0},
            {-1.0e-2, 2.0e-2, 0.0},
        };
        const std::vector<metalrobo::SurgicalThreadSurfaceTriangle>
            tissueTriangles{{0u, 1u, 2u}, {0u, 2u, 3u}};
        const std::vector<metalrobo::SurgicalThreadObstacleCapsule>
            obstacleCapsules{{
                .firstM = {5.0e-2, -5.0e-3, 5.0e-3},
                .secondM = {5.0e-2, 5.0e-3, 5.0e-3},
                .radiusM = 3.0e-3,
            }};
        const metalrobo::SurgicalThreadTargetingSpec spec{
            .threadRadiusM = 1.0e-4,
            .jawEnvelopeRadiusM = 1.0e-3,
            .jawContactLengthM = 8.0e-3,
            .minimumArcLengthFromSwageM = 1.5e-2,
            .minimumFreeTailLengthM = 2.0e-2,
            .preferredArcLengthFromSwageM = 5.0e-2,
            .maximumCenterlineDeviationM = 2.0e-5,
            .maximumTurningAngleRad = 1.0e-3,
            .minimumTissueClearanceM = 3.0e-3,
            .minimumObstacleClearanceM = 1.0e-3,
            .preferredApproachDirection = {0.0, 0.0, 1.0},
        };

        const auto diagnostics =
            metalrobo::selectSurgicalThreadGraspTarget(
                threadNodes,
                tissueNodes,
                tissueTriangles,
                obstacleCapsules,
                spec
            );
        require(
            diagnostics.succeeded(),
            "valid post-bite target rejected: " + std::string(
                metalrobo::surgicalThreadTargetStatusName(
                    diagnostics.status
                )
            )
        );
        const auto replay =
            metalrobo::selectSurgicalThreadGraspTarget(
                threadNodes,
                tissueNodes,
                tissueTriangles,
                obstacleCapsules,
                spec
            );
        require(replay.succeeded(), "deterministic replay rejected");
        require(
            diagnostics.target.centerEdge == replay.target.centerEdge &&
                diagnostics.target.windowFirstNode ==
                    replay.target.windowFirstNode &&
                diagnostics.target.windowLastNode ==
                    replay.target.windowLastNode &&
                samePoint(
                    diagnostics.target.centerM,
                    replay.target.centerM
                ) &&
                samePoint(
                    diagnostics.target.railDirection,
                    replay.target.railDirection
                ) &&
                samePoint(
                    diagnostics.target.separationDirection,
                    replay.target.separationDirection
                ) &&
                samePoint(
                    diagnostics.target.approachDirection,
                    replay.target.approachDirection
                ) &&
                diagnostics.target.score == replay.target.score,
            "target selection was not bit-repeatable"
        );

        const auto& target = diagnostics.target;
        const Point rightHandedApproach = cross(
            target.railDirection,
            target.separationDirection
        );
        const double frameError = std::max({
            std::abs(length(target.railDirection) - 1.0),
            std::abs(length(target.separationDirection) - 1.0),
            std::abs(length(target.approachDirection) - 1.0),
            std::abs(dot(
                target.railDirection,
                target.separationDirection
            )),
            std::abs(dot(
                target.railDirection,
                target.approachDirection
            )),
            std::abs(dot(
                target.separationDirection,
                target.approachDirection
            )),
            length(subtract(
                rightHandedApproach,
                target.approachDirection
            )),
        });
        require(frameError <= 1.0e-12, "jaw frame is not orthonormal");
        require(
            target.arcLengthFromSwageM >=
                    spec.minimumArcLengthFromSwageM &&
                target.freeTailLengthM >=
                    spec.minimumFreeTailLengthM &&
                target.centerlineWindowLengthM >=
                    spec.jawContactLengthM,
            "selected target violated its arc-length window"
        );
        require(
            target.minimumTissueClearanceM >=
                    spec.minimumTissueClearanceM &&
                target.minimumObstacleClearanceM >=
                    spec.minimumObstacleClearanceM,
            "selected target violated clearance constraints"
        );
        const auto surfaceClearance =
            metalrobo::evaluateSurgicalThreadJawSurfaceClearance(
                target.centerM,
                target.railDirection,
                spec.jawContactLengthM,
                spec.jawEnvelopeRadiusM,
                tissueNodes,
                tissueTriangles
            );
        require(
            surfaceClearance.succeeded() &&
                surfaceClearance.minimumEnvelopeClearanceM ==
                    target.minimumTissueClearanceM,
            "standalone jaw-surface clearance disagrees with selection"
        );
        Point penetratingCenter = target.centerM;
        penetratingCenter[2] = 5.0e-4;
        const auto penetratingClearance =
            metalrobo::evaluateSurgicalThreadJawSurfaceClearance(
                penetratingCenter,
                target.railDirection,
                spec.jawContactLengthM,
                spec.jawEnvelopeRadiusM,
                tissueNodes,
                tissueTriangles
            );
        require(
            penetratingClearance.succeeded() &&
                penetratingClearance.minimumEnvelopeClearanceM < 0.0,
            "penetrating jaw envelope was not measured as unsafe"
        );
        require(
            std::abs(target.arcLengthFromSwageM -
                     spec.preferredArcLengthFromSwageM) > 4.0e-3,
            "needle obstacle did not displace the preferred grasp"
        );

        auto closeTissueNodes = tissueNodes;
        for (Point& node : closeTissueNodes) {
            node[2] = 4.8e-3;
        }
        const auto closeTissueDiagnostics =
            metalrobo::selectSurgicalThreadGraspTarget(
                threadNodes,
                closeTissueNodes,
                tissueTriangles,
                obstacleCapsules,
                spec
            );
        require(
            closeTissueDiagnostics.status ==
                metalrobo::SurgicalThreadTargetStatus::
                    noAccessibleSegment,
            "unsafe tissue approach was not rejected"
        );

        const std::array<
            metalrobo::SurgicalThreadSurfaceTriangle,
            1u
        > malformedTriangles{{{0u, 1u, 99u}}};
        const auto malformedDiagnostics =
            metalrobo::selectSurgicalThreadGraspTarget(
                threadNodes,
                tissueNodes,
                malformedTriangles,
                obstacleCapsules,
                spec
            );
        require(
            malformedDiagnostics.status ==
                metalrobo::SurgicalThreadTargetStatus::invalidTopology,
            "malformed tissue surface was not rejected"
        );

        const std::vector<metalrobo::SurgicalThreadChannelCapsule>
            twoTractChannels{
                {
                    .firstM = {2.05e-2, -1.0e-3, 5.0e-3},
                    .secondM = {2.05e-2, 1.0e-3, 5.0e-3},
                    .radiusM = 2.0e-4,
                    .tract = 0u,
                },
                {
                    .firstM = {8.05e-2, -1.0e-3, 5.0e-3},
                    .secondM = {8.05e-2, 1.0e-3, 5.0e-3},
                    .radiusM = 2.0e-4,
                    .tract = 1u,
                },
            };
        const metalrobo::SurgicalThreadContactSelectionSpec
            twoTractSpec{
                .threadRadiusM = 1.0e-4,
                .maximumSurfaceSeparationM = 0.0,
                .tractCount = 2u,
                .proxyCount = 2u,
            };
        const auto twoTractSelection =
            metalrobo::selectSurgicalThreadContactEdges(
                threadNodes,
                twoTractChannels,
                twoTractSpec
            );
        const auto twoTractReplay =
            metalrobo::selectSurgicalThreadContactEdges(
                threadNodes,
                twoTractChannels,
                twoTractSpec
            );
        require(
            twoTractSelection.succeeded() &&
                twoTractReplay.succeeded() &&
                twoTractSelection.edges ==
                    std::vector<std::uint32_t>{20u, 80u} &&
                twoTractSelection.tractEdges ==
                    std::vector<std::uint32_t>{20u, 80u} &&
                twoTractSelection.edges == twoTractReplay.edges &&
                twoTractSelection.tractEdges ==
                    twoTractReplay.tractEdges &&
                twoTractSelection.tractSurfaceSeparationsM ==
                    twoTractReplay.tractSurfaceSeparationsM,
            "two-tract material-edge selection was not exact and repeatable"
        );
        const std::array<std::uint32_t, 2u> initialProxyEdges{0u, 1u};
        const auto rebindPlan =
            metalrobo::planSurgicalThreadProxyRebind(
                initialProxyEdges,
                twoTractSelection.edges,
                static_cast<std::uint32_t>(threadNodes.size() - 1u)
            );
        require(
            rebindPlan.succeeded() &&
                rebindPlan.transitions ==
                    std::vector<std::vector<std::uint32_t>>{
                        {20u, 1u},
                        {20u, 80u},
                    } &&
                rebindPlan.finalEdges ==
                    std::vector<std::uint32_t>{20u, 80u},
            "two-tract rebind plan did not preserve one-slot transactions"
        );

        const std::span<const metalrobo::SurgicalThreadChannelCapsule>
            firstTract(twoTractChannels.data(), 1u);
        auto oneTractSpec = twoTractSpec;
        oneTractSpec.tractCount = 1u;
        oneTractSpec.maximumSurfaceSeparationM = 1.0e-3;
        const auto oneTractSelection =
            metalrobo::selectSurgicalThreadContactEdges(
                threadNodes,
                firstTract,
                oneTractSpec
            );
        require(
            oneTractSelection.succeeded() &&
                oneTractSelection.edges ==
                    std::vector<std::uint32_t>{19u, 20u} &&
                oneTractSelection.tractEdges ==
                    std::vector<std::uint32_t>{20u},
            "single-tract selection did not retain its nearest overlap edge"
        );
        auto distantChannels = twoTractChannels;
        for (auto& channel : distantChannels) {
            channel.firstM[2] += 1.0e-2;
            channel.secondM[2] += 1.0e-2;
        }
        const auto distantSelection =
            metalrobo::selectSurgicalThreadContactEdges(
                threadNodes,
                distantChannels,
                twoTractSpec
            );
        require(
            distantSelection.status ==
                metalrobo::SurgicalThreadContactSelectionStatus::
                    noContactEdgeSet,
            "distant strand was assigned false puncture-tract ownership"
        );

        std::cout << std::setprecision(10)
            << "surgical_thread_targeting=ok"
            << " center_edge=" << target.centerEdge
            << " center_arc_m=" << target.arcLengthFromSwageM
            << " free_tail_m=" << target.freeTailLengthM
            << " window_length_m=" << target.centerlineWindowLengthM
            << " maximum_deviation_m="
            << target.maximumCenterlineDeviationM
            << " maximum_turning_angle_rad="
            << target.maximumTurningAngleRad
            << " tissue_clearance_m="
            << target.minimumTissueClearanceM
            << " obstacle_clearance_m="
            << target.minimumObstacleClearanceM
            << " frame_error=" << frameError
            << " evaluated_candidates="
            << diagnostics.evaluatedCandidates
            << " straight_candidates="
            << diagnostics.geometricallyStraightCandidates
            << " tissue_clear_candidates="
            << diagnostics.tissueClearCandidates
            << " obstacle_clear_candidates="
            << diagnostics.obstacleClearCandidates
            << " deterministic_replay=exact"
            << " negative_tissue_clearance=reject"
            << " negative_jaw_envelope_penetration=reject"
            << " negative_surface_topology=reject"
            << " contact_edges="
            << twoTractSelection.edges[0] << ','
            << twoTractSelection.edges[1]
            << " contact_maximum_surface_separation_m="
            << twoTractSelection.maximumSurfaceSeparationM
            << " proxy_rebind_commands="
            << rebindPlan.transitions.size()
            << " sparse_two_tract_contact=exact"
            << " negative_distant_tract=reject"
            << " boundary=geometry_not_ik_swept_collision_contact_or_retention"
            << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "surgical_thread_targeting=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
