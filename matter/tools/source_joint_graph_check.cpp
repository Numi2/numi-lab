#include "numi/matter/source_cylindrical_joint_graph.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace numi_matter_joint;
namespace {
constexpr const char* kExpectedSourceDeckSHA256 =
    "00b6efb53ad7e7330296cbb9569d358d48ed60819e22732e6149db6fb98a158a";
constexpr const char* kExpectedSourceGeometrySHA256 =
    "4155db1d0d7b87ffb2c668102d2495870e4461a539b18e6708f1f4817b5601bf";
constexpr const char* kExpectedSourceGraphProgramSHA256 =
    "e2285453812182f9a525bd3b6f1b32aaae77a52bf7f865563d483ed0b4e7b792";
struct JointRecord {
    unsigned idA, idB;
    V<double> origin, axis;
    double forcePenalty, momentPenalty, translation, rotation;
    bool prescribedTranslation, prescribedRotation;
    std::int32_t translationCurve, rotationCurve;
};
struct BoundaryRecord {
    unsigned bodyId = 0;
    std::uint8_t coordinateMask = 0;
    std::array<double, 6> value{};
    std::array<std::int32_t, 6> curve{};
};
struct LoadCurveRecord {
    std::uint32_t id = 0;
    std::uint32_t interpolation = 0;
    std::vector<double> times;
    std::vector<double> values;
};
struct SourceProgram {
    std::array<unsigned, 9> bodyIds{};
    std::array<V<double>, 9> referenceCOM{};
    std::array<JointRecord, 6> joints{};
    std::array<BoundaryRecord, 2> boundaries{};
    std::vector<LoadCurveRecord> curves;
    std::string deckSHA256;
    std::string geometrySHA256;
};
SourceProgram kSourceProgram{};
constexpr std::array<unsigned, 9> kExpectedBodyIds{1, 2, 3, 4, 17, 18, 19, 20, 21};
using Graph = GraphInput<double, 9, 6, 54>;
using Result = GraphOutput<double, 9, 6, 54>;
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

class ProgramReader {
public:
    explicit ProgramReader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}
    std::uint32_t u32() {
        const auto a = byte(), b = byte(), c = byte(), d = byte();
        return static_cast<std::uint32_t>(a) |
            (static_cast<std::uint32_t>(b) << 8) |
            (static_cast<std::uint32_t>(c) << 16) |
            (static_cast<std::uint32_t>(d) << 24);
    }
    std::int32_t i32() { return static_cast<std::int32_t>(u32()); }
    std::uint8_t u8() { return byte(); }
    double f64() {
        std::uint64_t bits = 0;
        for (unsigned shift = 0; shift < 64; shift += 8)
            bits |= static_cast<std::uint64_t>(byte()) << shift;
        double value = 0.0;
        std::memcpy(&value, &bits, sizeof(value));
        require(std::isfinite(value), "source rigid graph contains a non-finite number");
        return value;
    }
    void skip(std::size_t count) { for (std::size_t i = 0; i < count; ++i) (void)byte(); }
    std::string string(std::size_t count) {
        std::string result;
        result.reserve(count);
        for (std::size_t i = 0; i < count; ++i) result.push_back(static_cast<char>(byte()));
        return result;
    }
    bool atEnd() const { return offset_ == bytes_.size(); }
private:
    std::uint8_t byte() {
        require(offset_ < bytes_.size(), "source rigid graph program is truncated");
        return bytes_[offset_++];
    }
    const std::vector<std::uint8_t>& bytes_;
    std::size_t offset_ = 0;
};

std::string hexDigest(const std::string& raw) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(raw.size() * 2);
    for (unsigned char value : raw) {
        result.push_back(digits[value >> 4]);
        result.push_back(digits[value & 0x0f]);
    }
    return result;
}

void loadSourceProgram(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot open Human-compiled source rigid graph program: " + path);
    std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    require(bytes.size() <= std::numeric_limits<CC_LONG>::max(),
            "Human source rigid graph program is too large to authenticate");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    require(CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest) != nullptr,
            "SHA-256 failed for Human source rigid graph program");
    require(hexDigest(std::string(reinterpret_cast<const char*>(digest), sizeof(digest))) ==
                kExpectedSourceGraphProgramSHA256,
            "Human source rigid graph program artifact hash changed");
    ProgramReader reader(bytes);
    require(reader.string(8) == std::string("NHRGPH2\0", 8),
            "unsupported Human source rigid graph magic");
    const std::uint32_t version = reader.u32();
    const std::uint32_t bodyCount = reader.u32();
    const std::uint32_t jointCount = reader.u32();
    const std::uint32_t boundaryCount = reader.u32();
    const std::uint32_t curveCount = reader.u32();
    require(version == 2 && bodyCount == 9 && jointCount == 6 && boundaryCount == 2 &&
                curveCount == 1,
            "Human source rigid graph program has unsupported counts or version");
    kSourceProgram.deckSHA256 = hexDigest(reader.string(32));
    kSourceProgram.geometrySHA256 = hexDigest(reader.string(32));
    require(kSourceProgram.deckSHA256 == kExpectedSourceDeckSHA256 &&
                kSourceProgram.geometrySHA256 == kExpectedSourceGeometrySHA256,
            "Human rigid graph program is bound to different source files");
    for (std::size_t index = 0; index < kSourceProgram.bodyIds.size(); ++index) {
        kSourceProgram.bodyIds[index] = reader.u32();
        kSourceProgram.referenceCOM[index] = {reader.f64(), reader.f64(), reader.f64()};
        reader.skip(32); // source rigid-body XML SHA-256
        require(kSourceProgram.bodyIds[index] == kExpectedBodyIds[index],
                "Human source rigid body ordering or identity changed");
    }
    for (JointRecord& joint : kSourceProgram.joints) {
        joint.idA = reader.u32(); joint.idB = reader.u32();
        joint.origin = {reader.f64(), reader.f64(), reader.f64()};
        joint.axis = {reader.f64(), reader.f64(), reader.f64()};
        joint.forcePenalty = reader.f64(); joint.momentPenalty = reader.f64();
        joint.translation = reader.f64(); joint.rotation = reader.f64();
        const std::uint32_t prescribedTranslation = reader.u32();
        const std::uint32_t prescribedRotation = reader.u32();
        joint.translationCurve = reader.i32(); joint.rotationCurve = reader.i32();
        reader.skip(32); // source cylindrical-joint XML SHA-256
        require(prescribedTranslation <= 1 && prescribedRotation <= 1,
                "Human source cylindrical joint has an invalid prescribed flag");
        joint.prescribedTranslation = prescribedTranslation != 0;
        joint.prescribedRotation = prescribedRotation != 0;
    }
    for (BoundaryRecord& boundary : kSourceProgram.boundaries) {
        boundary.bodyId = reader.u32();
        boundary.coordinateMask = reader.u8();
        reader.skip(3);
        for (double& value : boundary.value) value = reader.f64();
        for (std::int32_t& curve : boundary.curve) curve = reader.i32();
        reader.skip(32); // source rigid-body-boundary XML SHA-256
    }
    for (std::uint32_t index = 0; index < curveCount; ++index) {
        LoadCurveRecord curve;
        curve.id = reader.u32();
        curve.interpolation = reader.u32();
        const std::uint32_t pointCount = reader.u32();
        require(pointCount >= 2 && pointCount <= 1024,
                "Human source load curve has unsupported point count");
        curve.times.reserve(pointCount);
        curve.values.reserve(pointCount);
        for (std::uint32_t point = 0; point < pointCount; ++point) {
            curve.times.push_back(reader.f64());
            curve.values.push_back(reader.f64());
            if (point != 0)
                require(curve.times[point] > curve.times[point - 1],
                        "Human source load curve times are not strictly increasing");
        }
        reader.skip(32); // source load-curve XML SHA-256
        kSourceProgram.curves.push_back(std::move(curve));
    }
    require(reader.atEnd(), "Human source rigid graph program has trailing data");
    require(kSourceProgram.joints[0].idA == 4 && kSourceProgram.joints[0].idB == 18 &&
                kSourceProgram.joints[1].idA == 17 && kSourceProgram.joints[1].idB == 3 &&
                kSourceProgram.joints[2].idA == 18 && kSourceProgram.joints[2].idB == 17 &&
                kSourceProgram.joints[3].idA == 4 && kSourceProgram.joints[3].idB == 20 &&
                kSourceProgram.joints[4].idA == 19 && kSourceProgram.joints[4].idB == 1 &&
                kSourceProgram.joints[5].idA == 20 && kSourceProgram.joints[5].idB == 19,
            "Human source cylindrical joint order or body references changed");
    require(kSourceProgram.joints[0].prescribedRotation &&
                !kSourceProgram.joints[0].prescribedTranslation &&
                kSourceProgram.joints[0].rotationCurve == 9 &&
                std::abs(kSourceProgram.joints[0].rotation + 1.57) < 1.0e-12,
            "source flexion coordinate is not the compiled prescribed joint");
    require(kSourceProgram.curves[0].id == 9 && kSourceProgram.curves[0].interpolation == 1 &&
                kSourceProgram.curves[0].times == std::vector<double>{0.0, 1.0, 2.0} &&
                kSourceProgram.curves[0].values == std::vector<double>{0.0, 0.0, 1.0},
            "source flexion load curve is not the compiled piecewise-linear program");
    for (std::size_t index = 0; index < kSourceProgram.boundaries.size(); ++index) {
        const BoundaryRecord& boundary = kSourceProgram.boundaries[index];
        const unsigned expectedBody = index == 0 ? 3u : 2u;
        require(boundary.bodyId == expectedBody && boundary.coordinateMask == 0x3fu,
                "source rigid-body boundary is missing or changed");
        for (std::size_t coordinate = 0; coordinate < 6; ++coordinate)
            require(boundary.value[coordinate] == 0.0 && boundary.curve[coordinate] == 9,
                    "source rigid-body boundary value or load curve changed");
    }
}

const LoadCurveRecord& sourceCurve(std::uint32_t id) {
    const auto found = std::find_if(
        kSourceProgram.curves.begin(), kSourceProgram.curves.end(),
        [id](const LoadCurveRecord& curve) { return curve.id == id; });
    require(found != kSourceProgram.curves.end(), "source graph refers to an unloaded curve");
    return *found;
}

double sampleSourceCurve(const LoadCurveRecord& curve, double time) {
    require(std::isfinite(time) && time >= curve.times.front() && time <= curve.times.back(),
            "source graph probe time is outside the compiled load curve");
    const auto upper = std::upper_bound(curve.times.begin(), curve.times.end(), time);
    if (upper == curve.times.begin()) return curve.values.front();
    if (upper == curve.times.end()) return curve.values.back();
    const std::size_t right = static_cast<std::size_t>(upper - curve.times.begin());
    const std::size_t left = right - 1;
    const double fraction = (time - curve.times[left]) /
        (curve.times[right] - curve.times[left]);
    return curve.values[left] + fraction * (curve.values[right] - curve.values[left]);
}

double sourceCurveTimeForValue(const LoadCurveRecord& curve, double value) {
    for (std::size_t right = 1; right < curve.values.size(); ++right) {
        const double a = curve.values[right - 1], b = curve.values[right];
        if (value >= std::min(a, b) && value <= std::max(a, b) && a != b) {
            const double fraction = (value - a) / (b - a);
            return curve.times[right - 1] + fraction *
                (curve.times[right] - curve.times[right - 1]);
        }
    }
    require(std::abs(value - curve.values.front()) < 1.0e-14,
            "requested source flexion value is not on the compiled load curve");
    return curve.times.front();
}

std::size_t indexOf(unsigned sourceId) {
    const auto it = std::find(kSourceProgram.bodyIds.begin(), kSourceProgram.bodyIds.end(), sourceId);
    require(it != kSourceProgram.bodyIds.end(), "source joint refers to an unknown body ID");
    return static_cast<std::size_t>(it - kSourceProgram.bodyIds.begin());
}
Graph sourceGraph() {
    Graph graph{};
    for (std::size_t body = 0; body < kSourceProgram.referenceCOM.size(); ++body) {
        graph.bodies[body] = {kSourceProgram.referenceCOM[body], {0, 0, 0, 1}};
    }
    for (std::size_t jointIndex = 0; jointIndex < kSourceProgram.joints.size(); ++jointIndex) {
        const JointRecord& record = kSourceProgram.joints[jointIndex];
        CylindricalJoint<double>& joint = graph.joints[jointIndex];
        joint.bodyA = indexOf(record.idA);
        joint.bodyB = indexOf(record.idB);
        joint.source.referenceA = kSourceProgram.referenceCOM[joint.bodyA];
        joint.source.referenceB = kSourceProgram.referenceCOM[joint.bodyB];
        joint.source.origin = record.origin;
        joint.source.axis = record.axis;
        joint.source.forcePenalty = record.forcePenalty;
        joint.source.momentPenalty = record.momentPenalty;
        joint.source.translation = record.prescribedTranslation ? record.translation : 0.0;
        // The load-curve pose is sampled explicitly by the flexion operator checks.
        joint.source.rotation = record.prescribedRotation && record.rotationCurve >= 0
            ? 0.0 : record.rotation;
        joint.source.prescribedTranslation = record.prescribedTranslation ? 1u : 0u;
        joint.source.prescribedRotation = record.prescribedRotation ? 1u : 0u;
    }
    // Source boundary coordinates are imported from Human's compiled program.
    for (std::size_t body = 0; body < kSourceProgram.bodyIds.size(); ++body) {
        for (unsigned coordinate = 0; coordinate < 6; ++coordinate) {
            const std::size_t dof = body * 6 + coordinate;
            bool prescribed = false;
            for (const BoundaryRecord& boundary : kSourceProgram.boundaries) {
                if (boundary.bodyId == kSourceProgram.bodyIds[body] &&
                    (boundary.coordinateMask & (1u << coordinate)) != 0u) prescribed = true;
            }
            if (prescribed) continue;
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
void makeFlexionPose(Graph& graph, double flexionRadians) {
    // These checkpoints sample the pinned linear source curve. In particular,
    // its endpoint is -1.57 rad (89.954 degrees), not pi/2. The body pose is a
    // kinematic operator check; no tissue, spring, contact, or prestrain solve
    // is implied.
    const JointRecord& sourceFlexion = kSourceProgram.joints[0];
    const double desiredFraction = std::abs(flexionRadians / sourceFlexion.rotation);
    const LoadCurveRecord& curve = sourceCurve(
        static_cast<std::uint32_t>(sourceFlexion.rotationCurve));
    const double sourceTime = sourceCurveTimeForValue(curve, desiredFraction);
    graph.joints[0].source.rotation = sourceFlexion.rotation *
        sampleSourceCurve(curve, sourceTime);
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

int main(int argc, char** argv) {
    try {
        require(argc <= 3 && (argc == 1 || (argc == 3 &&
                    std::string(argv[1]) == "--source-program")),
                "usage: source_joint_graph_check [--source-program PATH]");
#ifndef NUMI_OPEN_KNEE_SOURCE_GRAPH_PROGRAM
#error "CMake must provide the Human-compiled source rigid graph program"
#endif
        const std::string programPath = argc == 3
            ? argv[2] : NUMI_OPEN_KNEE_SOURCE_GRAPH_PROGRAM;
        loadSourceProgram(programPath);
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

        struct FlexionCheck {
            double degrees;
            double residual = 0.0;
            double powerError = 0.0;
            double tangentError = 0.0;
            unsigned rank = 0u;
        };
        const double sourceEndpointDegrees =
            std::abs(kSourceProgram.joints[0].rotation) * 180.0 / std::acos(-1.0);
        std::array<FlexionCheck, 3> flexionChecks{{
            {30.0}, {60.0}, {sourceEndpointDegrees},
        }};
        for (FlexionCheck& check : flexionChecks) {
            Graph flexed = sourceGraph();
            makeFlexionPose(flexed, check.degrees * std::acos(-1.0) / 180.0);
            Result flexedResult{};
            require(evaluateGraph(flexed, flexedResult),
                    std::to_string(static_cast<unsigned>(check.degrees)) +
                        "-degree source graph rejected");
            for (double value : flexedResult.residual)
                check.residual = std::max(check.residual, std::abs(value));
            require(check.residual < 2.0e-7,
                    "source cylinders do not admit the requested flexion pose");
            check.powerError = checkVirtualWork(flexed, flexedResult);
            require(check.powerError < 5.0e-15,
                    "flexed graph wrench pullback violates virtual work");
            check.tangentError = checkTangent(flexed);
            require(check.tangentError < 5.0e-5,
                    "flexed graph tangent failed finite differences");
            check.rank = patellaConstraintRank(flexed);
            require(check.rank == 12u,
                    "patella source graph changed its six relative freedoms");
        }
        const unsigned neutralRank = patellaConstraintRank(neutral);
        require(neutralRank == 12u,
                "patella source graph does not preserve its six relative freedoms");
        Graph invalid = neutral;
        invalid.joints[0].bodyA = invalid.bodies.size();
        Result preserved{};
        preserved.residual[0] = 123.0;
        require(!evaluateGraph(invalid, preserved), "invalid source body reference was admitted");
        require(preserved.residual[0] == 123.0,
                "rejected source graph mutated the caller's output state");
        std::cout << "source_deck_sha256=" << kSourceProgram.deckSHA256
                  << " source_geometry_sha256=" << kSourceProgram.geometrySHA256
                  << " source_graph_program=" << programPath
                  << " source_bodies=9 source_cylindrical_joints=6 "
                  << "neutral_residual=" << neutralResidual
                  << " neutral_tangent_error=" << neutralTangentError
                  << " neutral_virtual_work_error=" << neutralPowerError
                  << " invalid_graph_rejected=1"
                  << " patella_constraint_rank_neutral=" << neutralRank;
        for (const FlexionCheck& check : flexionChecks) {
            std::cout << " flexion_" << check.degrees
                      << "deg_source_curve_residual=" << check.residual
                      << "_tangent_error=" << check.tangentError
                      << "_virtual_work_error=" << check.powerError
                      << "_patella_rank=" << check.rank;
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
