#include "numi/matter/source_cylindrical_joint_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace numi_matter_joint;
namespace {
constexpr const char* kSourceDeckSHA256 =
    "00b6efb53ad7e7330296cbb9569d358d48ed60819e22732e6149db6fb98a158a";
constexpr const char* kSourceGeometrySHA256 =
    "4155db1d0d7b87ffb2c668102d2495870e4461a539b18e6708f1f4817b5601bf";
constexpr std::array<unsigned, 9> kSourceBodyIds{1, 2, 3, 4, 17, 18, 19, 20, 21};
constexpr std::array<V<double>, 9> kReferenceCOM{
    V<double>{3.8855, 46.3435, 3.6415},
    V<double>{0, 0, 0},
    V<double>{-4.4315, -7.2205, -24.9345},
    V<double>{-1.036, -6.717, .171},
    V<double>{-4.4315, -7.2205, -24.9345},
    V<double>{-1.036, -6.717, .171},
    V<double>{3.8855, 46.3435, 3.6415},
    V<double>{-1.036, -6.717, .171},
    V<double>{1.0835532632546936, 38.06618835977137, 90.2025146484375},
};
struct JointRecord {
    const char* name;
    unsigned idA, idB;
    V<double> origin, axis;
    bool prescribedRotation;
};
constexpr std::array<JointRecord, 6> kSourceJoints{{
    {"Extension_Flexion", 4, 18, {-1.036, -6.717, .171},
     {.9889108468842679, -.1485103932882822, 0}, true},
    {"External_Internal", 17, 3, {-4.4315, -7.2205, -24.9345},
     {0, 0, 1}, false},
    {"Abduction_Adduction", 18, 17, {-4.282665311721631, -6.229429716148487, .171},
     {.1485103932882822, .9889108468842679, 0}, false},
    {"Patellar_Extension_Flexion", 4, 20, {-1.036, -6.717, .171},
     {.9889108468842679, -.1485103932882822, 0}, false},
    {"Patellar_Lateral_Tilt", 19, 1, {3.8855, 46.3435, 3.6415},
     {.007345507674492785, 0, .9999730213945794}, false},
    {"Patellar_Lateral_Rotation", 20, 19, {-4.041046061001632, -6.265715059830884, .171},
     {.1485103049177992, .9889102584370404, -.0010909130158267598}, false},
}};
using Graph = GraphInput<double, 9, 6, 54>;
using Result = GraphOutput<double, 9, 6, 54>;
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
std::size_t indexOf(unsigned sourceId) {
    const auto it = std::find(kSourceBodyIds.begin(), kSourceBodyIds.end(), sourceId);
    require(it != kSourceBodyIds.end(), "source joint refers to an unknown body ID");
    return static_cast<std::size_t>(it - kSourceBodyIds.begin());
}
Graph sourceGraph() {
    Graph graph{};
    for (std::size_t body = 0; body < kReferenceCOM.size(); ++body) {
        graph.bodies[body] = {kReferenceCOM[body], {0, 0, 0, 1}};
    }
    for (std::size_t jointIndex = 0; jointIndex < kSourceJoints.size(); ++jointIndex) {
        const JointRecord& record = kSourceJoints[jointIndex];
        CylindricalJoint<double>& joint = graph.joints[jointIndex];
        joint.bodyA = indexOf(record.idA);
        joint.bodyB = indexOf(record.idB);
        joint.source.referenceA = kReferenceCOM[joint.bodyA];
        joint.source.referenceB = kReferenceCOM[joint.bodyB];
        joint.source.origin = record.origin;
        joint.source.axis = record.axis;
        joint.source.forcePenalty = 10000.0;
        joint.source.momentPenalty = 3000000.0;
        joint.source.prescribedRotation = record.prescribedRotation ? 1u : 0u;
    }
    // Source boundaries prescribe all six coordinates of FBB (2) and TBB (3)
    // to zero. They remain in the source body graph but have no free columns.
    for (std::size_t body = 0; body < kSourceBodyIds.size(); ++body) {
        for (unsigned coordinate = 0; coordinate < 6; ++coordinate) {
            const std::size_t dof = body * 6 + coordinate;
            if (kSourceBodyIds[body] == 2 || kSourceBodyIds[body] == 3) continue;
            if (coordinate < 3) {
                if (coordinate == 0) graph.motion[dof][body].linear.x = 1.0;
                if (coordinate == 1) graph.motion[dof][body].linear.y = 1.0;
                if (coordinate == 2) graph.motion[dof][body].linear.z = 1.0;
            } else {
                if (coordinate == 3) graph.motion[dof][body].angular.x = 1.0;
                if (coordinate == 4) graph.motion[dof][body].angular.y = 1.0;
                if (coordinate == 5) graph.motion[dof][body].angular.z = 1.0;
            }
        }
    }
    return graph;
}
void setBodyB(Graph& graph, std::size_t jointIndex, double rotation, double translation = 0.0) {
    const auto& joint = graph.joints[jointIndex];
    const auto& p = joint.source;
    const auto& a = graph.bodies[joint.bodyA];
    auto& b = graph.bodies[joint.bodyB];
    b.rotation = multiply(a.rotation, exponential(p.axis * rotation));
    const V<double> pointA = a.position + rotate(a.rotation, p.origin - p.referenceA);
    const V<double> axisA = rotate(a.rotation, p.axis);
    b.position = pointA + axisA * translation - rotate(b.rotation, p.origin - p.referenceB);
}
void setBodyA(Graph& graph, std::size_t jointIndex, double rotation, double translation = 0.0) {
    const auto& joint = graph.joints[jointIndex];
    const auto& p = joint.source;
    const auto& b = graph.bodies[joint.bodyB];
    auto& a = graph.bodies[joint.bodyA];
    a.rotation = multiply(b.rotation, exponential(p.axis * -rotation));
    const V<double> pointB = b.position + rotate(b.rotation, p.origin - p.referenceB);
    const V<double> axisA = rotate(a.rotation, p.axis);
    a.position = pointB - axisA * translation - rotate(a.rotation, p.origin - p.referenceA);
}
void makeFlexionPose(Graph& graph) {
    // The source flexion curve reaches -1.57 rad. The 30 degree checkpoint is
    // a kinematic operator test only; the source deck's separate rigid spring,
    // tissue, contact, and prestrain equations are intentionally not implied.
    graph.joints[0].source.rotation = -0.523598775598298873;
    setBodyA(graph, 1, 0.0);                   // TBB (3) anchors TFTO (17)
    setBodyA(graph, 2, 0.0);                   // TFTO (17) anchors TFFO (18)
    setBodyA(graph, 0, graph.joints[0].source.rotation); // source flexion
    setBodyB(graph, 3, .10);                   // FMB (4) -> PFFO (20)
    setBodyB(graph, 5, .05);                   // PFFO (20) -> PFPO (19)
    setBodyB(graph, 4, .02);                   // PFPO (19) -> PTB (1)
}
void perturb(Graph& graph, std::size_t dof, double amount) {
    for (std::size_t body = 0; body < graph.bodies.size(); ++body) {
        const BodyMotion<double>& motion = graph.motion[dof][body];
        graph.bodies[body].position = graph.bodies[body].position + motion.linear * amount;
        graph.bodies[body].rotation = multiply(
            exponential(motion.angular * amount), graph.bodies[body].rotation);
    }
}
double checkTangent(const Graph& graph) {
    Result base{};
    require(evaluateGraph(graph, base), "source rigid graph rejected its state");
    constexpr double h = 3.0e-6;
    double worst = 0.0;
    for (std::size_t column = 0; column < 54; ++column) {
        Graph plus = graph, minus = graph;
        perturb(plus, column, h);
        perturb(minus, column, -h);
        Result a{}, b{};
        require(evaluateGraph(plus, a) && evaluateGraph(minus, b),
                "source graph tangent difference rejected a perturbed state");
        for (std::size_t row = 0; row < 54; ++row) {
            const double fd = (a.residual[row] - b.residual[row]) / (2.0 * h);
            const double expected = base.tangent[row * 54 + column];
            const double error = std::abs(fd - expected) /
                std::max({1.0, std::abs(fd), std::abs(expected)});
            worst = std::max(worst, error);
        }
    }
    return worst;
}
double checkVirtualWork(const Graph& graph, const Result& result) {
    std::array<double, 54> rate{};
    double coordinatePower = 0.0;
    for (std::size_t dof = 0; dof < rate.size(); ++dof) {
        rate[dof] = std::sin(.37 * static_cast<double>(dof + 1));
        coordinatePower += (result.residual[dof] + graph.appliedGeneralizedForce[dof]) * rate[dof];
    }
    double bodyPower = 0.0, scale = 0.0;
    for (std::size_t body = 0; body < graph.bodies.size(); ++body) {
        BodyMotion<double> velocity{};
        for (std::size_t dof = 0; dof < rate.size(); ++dof) {
            velocity.linear = velocity.linear + graph.motion[dof][body].linear * rate[dof];
            velocity.angular = velocity.angular + graph.motion[dof][body].angular * rate[dof];
        }
        const double term = dot(result.bodyWrenches[body].force, velocity.linear) +
                            dot(result.bodyWrenches[body].moment, velocity.angular);
        bodyPower += term;
        scale += std::abs(term);
    }
    return std::abs(bodyPower - coordinatePower) / std::max(1.0, scale);
}
std::array<double, 6> jointGap(const Graph& graph, std::size_t index) {
    const auto& joint = graph.joints[index];
    Input<double> input = joint.source;
    input.positionA = graph.bodies[joint.bodyA].position;
    input.positionB = graph.bodies[joint.bodyB].position;
    input.rotationA = graph.bodies[joint.bodyA].rotation;
    input.rotationB = graph.bodies[joint.bodyB].rotation;
    Output<double> value{}, derivative{};
    require(evaluate(input, Direction<double>{}, value, derivative), "source gap rejected");
    return {value.gap.x, value.gap.y, value.gap.z,
            value.angularGap.x, value.angularGap.y, value.angularGap.z};
}
unsigned patellaConstraintRank(const Graph& graph) {
    // Three source cylinders link bodies 4,20,19,1. Each contributes two
    // translational and two rotational constraints. Rank 12 leaves six
    // relative freedoms after removing the six global rigid-body modes.
    constexpr std::array<std::size_t, 3> links{3, 5, 4};
    constexpr std::array<unsigned, 4> bodyIds{4, 20, 19, 1};
    std::array<std::array<double, 24>, 18> rows{};
    constexpr double h = 1.0e-6;
    std::size_t row = 0;
    for (const std::size_t link : links) {
        for (unsigned component = 0; component < 6; ++component, ++row) {
            for (std::size_t localBody = 0; localBody < bodyIds.size(); ++localBody) {
                const std::size_t body = indexOf(bodyIds[localBody]);
                for (unsigned axis = 0; axis < 6; ++axis) {
                    Graph plus = graph, minus = graph;
                    const std::size_t dof = body * 6 + axis;
                    perturb(plus, dof, h);
                    perturb(minus, dof, -h);
                    const auto a = jointGap(plus, link);
                    const auto b = jointGap(minus, link);
                    rows[row][localBody * 6 + axis] =
                        (a[component] - b[component]) / (2.0 * h);
                }
            }
        }
    }
    double scale = 0.0;
    for (const auto& values : rows) for (double value : values) scale = std::max(scale, std::abs(value));
    unsigned rank = 0;
    for (std::size_t column = 0; column < 24 && rank < rows.size(); ++column) {
        std::size_t pivot = rank;
        for (std::size_t candidate = rank + 1; candidate < rows.size(); ++candidate) {
            if (std::abs(rows[candidate][column]) > std::abs(rows[pivot][column])) pivot = candidate;
        }
        if (std::abs(rows[pivot][column]) <= scale * 1.0e-8) continue;
        std::swap(rows[rank], rows[pivot]);
        const double divisor = rows[rank][column];
        for (std::size_t k = column; k < 24; ++k) rows[rank][k] /= divisor;
        for (std::size_t other = 0; other < rows.size(); ++other) {
            if (other == rank) continue;
            const double factor = rows[other][column];
            for (std::size_t k = column; k < 24; ++k) rows[other][k] -= factor * rows[rank][k];
        }
        ++rank;
    }
    return rank;
}
}

int main() {
    try {
        Graph neutral = sourceGraph();
        Result neutralResult{};
        require(evaluateGraph(neutral, neutralResult), "neutral source graph rejected");
        double neutralResidual = 0.0;
        for (double value : neutralResult.residual) neutralResidual = std::max(neutralResidual, std::abs(value));
        require(neutralResidual < 1.0e-8, "source-neutral cylindrical graph is not in equilibrium");
        const double neutralPowerError = checkVirtualWork(neutral, neutralResult);
        require(neutralPowerError < 5.0e-15, "neutral graph wrench pullback violates virtual work");
        const double neutralTangentError = checkTangent(neutral);
        require(neutralTangentError < 5.0e-5, "neutral graph tangent failed finite differences");

        Graph flexed = sourceGraph();
        makeFlexionPose(flexed);
        Result flexedResult{};
        require(evaluateGraph(flexed, flexedResult), "30-degree source graph rejected");
        double flexedResidual = 0.0;
        for (double value : flexedResult.residual) flexedResidual = std::max(flexedResidual, std::abs(value));
        require(flexedResidual < 2.0e-7, "source cylinders do not admit the 30-degree flexion pose");
        const double flexedPowerError = checkVirtualWork(flexed, flexedResult);
        require(flexedPowerError < 5.0e-15, "flexed graph wrench pullback violates virtual work");
        const double flexedTangentError = checkTangent(flexed);
        require(flexedTangentError < 5.0e-5, "flexed graph tangent failed finite differences");
        const unsigned neutralRank = patellaConstraintRank(neutral);
        const unsigned flexedRank = patellaConstraintRank(flexed);
        require(neutralRank == 12 && flexedRank == 12,
                "patella source graph does not preserve its six relative freedoms");
        Graph invalid = neutral;
        invalid.joints[0].bodyA = invalid.bodies.size();
        Result preserved{};
        preserved.residual[0] = 123.0;
        require(!evaluateGraph(invalid, preserved), "invalid source body reference was admitted");
        require(preserved.residual[0] == 123.0,
                "rejected source graph mutated the caller's output state");
        std::cout << "source_deck_sha256=" << kSourceDeckSHA256
                  << " source_geometry_sha256=" << kSourceGeometrySHA256
                  << " source_bodies=9 source_cylindrical_joints=6 "
                  << "neutral_residual=" << neutralResidual
                  << " flexed_residual=" << flexedResidual
                  << " neutral_tangent_error=" << neutralTangentError
                  << " flexed_tangent_error=" << flexedTangentError
                  << " neutral_virtual_work_error=" << neutralPowerError
                  << " flexed_virtual_work_error=" << flexedPowerError
                  << " invalid_graph_rejected=1"
                  << " patella_constraint_rank_neutral=" << neutralRank
                  << " patella_constraint_rank_30deg=" << flexedRank << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
