#include "metalrobo/NumiHumanTissueMass.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>

namespace metalrobo {
namespace {
using Matrix = std::array<double, 9>;
using Vector = std::array<double, 3>;

Matrix secondMoment(const Matrix& inertia) {
    const double halfTrace = 0.5 * (inertia[0] + inertia[4] + inertia[8]);
    Matrix result{};
    for (std::size_t i = 0; i < 9; ++i)
        result[i] = (i % 4 == 0 ? halfTrace : 0.0) - inertia[i];
    return result;
}

Matrix inertiaFromSecond(const Matrix& moment) {
    const double trace = moment[0] + moment[4] + moment[8];
    Matrix result{};
    for (std::size_t i = 0; i < 9; ++i)
        result[i] = (i % 4 == 0 ? trace : 0.0) - moment[i];
    return result;
}

Matrix inertia(const MRBodyPropertiesGPU& body) {
    return {body.inertiaRow0.x, body.inertiaRow0.y, body.inertiaRow0.z,
            body.inertiaRow1.x, body.inertiaRow1.y, body.inertiaRow1.z,
            body.inertiaRow2.x, body.inertiaRow2.y, body.inertiaRow2.z};
}

// Cholesky tests the central SECOND moment as well as inertia. Positive
// inertia alone admits impossible mass distributions (triangle violations).
bool positiveDefinite(const Matrix& value) {
    if (!std::all_of(value.begin(), value.end(), [](double v) { return std::isfinite(v); }))
        return false;
    const double scale = std::max({value[0], value[4], value[8]});
    if (!(scale > 0.0)) return false;
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    Matrix lower{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            if (std::abs(value[3 * row + column] - value[3 * column + row]) > tolerance)
                return false;
            double residual = value[3 * row + column];
            for (std::size_t k = 0; k < column; ++k)
                residual -= lower[3 * row + k] * lower[3 * column + k];
            if (row == column) {
                if (!(residual > tolerance)) return false;
                lower[3 * row + column] = std::sqrt(residual);
            } else lower[3 * row + column] = residual / lower[4 * column];
        }
    }
    return true;
}

Matrix inverse(const Matrix& a) {
    Matrix b{a[4]*a[8]-a[5]*a[7], a[2]*a[7]-a[1]*a[8], a[1]*a[5]-a[2]*a[4],
             a[5]*a[6]-a[3]*a[8], a[0]*a[8]-a[2]*a[6], a[2]*a[3]-a[0]*a[5],
             a[3]*a[7]-a[4]*a[6], a[1]*a[6]-a[0]*a[7], a[0]*a[4]-a[1]*a[3]};
    const double determinant = a[0]*b[0] + a[1]*b[3] + a[2]*b[6];
    for (double& v : b) v /= determinant;
    return b;
}

mr_float4 row(const Matrix& a, std::size_t i) {
    return {static_cast<float>(a[3*i]), static_cast<float>(a[3*i+1]),
            static_cast<float>(a[3*i+2]), 0.0f};
}

NumiHumanTissueMassResult fail(const std::string& reason) {
    return {{}, reason};
}
} // namespace

NumiHumanTissueMassResult compileNumiHumanTissueMassPartition(
    const std::span<const MRBodyPropertiesGPU> bodies,
    const std::span<const NumiHumanTissueMassNode> nodes) {
    if (nodes.empty()) return fail("tissue mass partition has no cooked nodes");
    std::map<std::uint32_t, NumiHumanTissueMassPartition> partitions;
    std::set<std::uint32_t> seen;
    for (const auto& node : nodes) {
        if (node.nodeIndex == MR_INVALID_INDEX || !seen.insert(node.nodeIndex).second)
            return fail("duplicate or invalid cooked tissue node ownership");
        if (node.donorBody >= bodies.size() || !std::isfinite(node.massKg) || node.massKg <= 0.0 ||
            !std::all_of(node.localPosition.begin(), node.localPosition.end(),
                         [](double v) { return std::isfinite(v); }))
            return fail("invalid cooked tissue mass, position or donor");
        auto& p = partitions[node.donorBody];
        p.donorBody = node.donorBody;
        ++p.nodeCount;
        p.tissueMassKg += node.massKg;
        for (std::size_t i = 0; i < 3; ++i) {
            p.tissueFirstMomentKgM[i] += node.massKg * node.localPosition[i];
            for (std::size_t j = 0; j < 3; ++j)
                p.tissueSecondMomentKgM2[3*i+j] +=
                    node.massKg * node.localPosition[i] * node.localPosition[j];
        }
    }
    NumiHumanTissueMassResult result;
    for (auto& [bodyIndex, p] : partitions) {
        const auto& original = bodies[bodyIndex];
        const double mass = original.massAndInverseMass.x;
        const Matrix originalInertia = inertia(original);
        if (original.motionType != MR_MOTION_DYNAMIC || !std::isfinite(mass) || mass <= 0.0 ||
            !positiveDefinite(originalInertia) || !positiveDefinite(secondMoment(originalInertia)))
            return fail("donor is not a positive three-dimensional dynamic mass distribution");
        const double remaining = mass - p.tissueMassKg;
        if (!std::isfinite(remaining) || remaining <= 0.0)
            return fail("tissue consumes or exceeds donor mass");
        Matrix central = secondMoment(originalInertia);
        for (std::size_t i = 0; i < 3; ++i)
            p.remainingCOMOffsetM[i] = -p.tissueFirstMomentKgM[i] / remaining;
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                central[3*i+j] -= p.tissueSecondMomentKgM2[3*i+j] +
                    remaining * p.remainingCOMOffsetM[i] * p.remainingCOMOffsetM[j];
        if (!positiveDefinite(central))
            return fail("tissue partition leaves an impossible residual mass distribution");
        const Matrix residualInertia = inertiaFromSecond(central);
        p.sourceBody = original;
        p.remainingBody = original;
        auto& body = p.remainingBody;
        body.massAndInverseMass = {static_cast<float>(remaining),
                                  static_cast<float>(1.0 / remaining), 0.0f, 0.0f};
        // Metadata records source-link to new COM. Runtime coordinates remain
        // COM-centred and must consume the explicit rebase offset separately.
        body.centerOfMass.x += static_cast<float>(p.remainingCOMOffsetM[0]);
        body.centerOfMass.y += static_cast<float>(p.remainingCOMOffsetM[1]);
        body.centerOfMass.z += static_cast<float>(p.remainingCOMOffsetM[2]);
        body.inertiaRow0 = row(residualInertia, 0);
        body.inertiaRow1 = row(residualInertia, 1);
        body.inertiaRow2 = row(residualInertia, 2);
        const Matrix packedInertia = inertia(body);
        if (!positiveDefinite(packedInertia) || !positiveDefinite(secondMoment(packedInertia)) ||
            !std::isfinite(body.massAndInverseMass.y) || body.massAndInverseMass.x <= 0.0f ||
            !std::isfinite(body.centerOfMass.x) || !std::isfinite(body.centerOfMass.y) ||
            !std::isfinite(body.centerOfMass.z))
            return fail("residual mass distribution is not representable in the engine ABI");
        const Matrix inverseInertia = inverse(packedInertia);
        if (!std::all_of(inverseInertia.begin(), inverseInertia.end(),
            [](double v) { return std::isfinite(static_cast<float>(v)); }))
            return fail("residual inverse inertia overflows the engine ABI");
        body.inverseInertiaRow0 = row(inverseInertia, 0);
        body.inverseInertiaRow1 = row(inverseInertia, 1);
        body.inverseInertiaRow2 = row(inverseInertia, 2);
        // Verify recombination of packed rigid inertia and exact cooked nodes.
        const Matrix packedSecond = secondMoment(packedInertia);
        const Matrix originalSecond = secondMoment(originalInertia);
        const double momentScale = std::max({originalSecond[0], originalSecond[4], originalSecond[8]});
        p.packedMomentRelativeError = std::abs(body.massAndInverseMass.x + p.tissueMassKg - mass) / mass;
        for (std::size_t i = 0; i < 3; ++i) {
            p.packedMomentRelativeError = std::max(p.packedMomentRelativeError,
                std::abs(body.massAndInverseMass.x * p.remainingCOMOffsetM[i] +
                         p.tissueFirstMomentKgM[i]) / std::sqrt(mass * momentScale));
            for (std::size_t j = 0; j < 3; ++j) {
                const double recombined = packedSecond[3*i+j] +
                    body.massAndInverseMass.x * p.remainingCOMOffsetM[i] * p.remainingCOMOffsetM[j] +
                    p.tissueSecondMomentKgM2[3*i+j];
                p.packedMomentRelativeError = std::max(p.packedMomentRelativeError,
                    std::abs(recombined - originalSecond[3*i+j]) / momentScale);
            }
        }
        if (p.packedMomentRelativeError > 16.0 * std::numeric_limits<float>::epsilon())
            return fail("packed mass moments do not conserve the donor mass distribution");
        result.partitions.push_back(p);
    }
    return result;
}

bool rebaseNumiHumanTissueMassPartition(
    const EngineModel& source, const std::span<const NumiHumanTissueMassPartition> partitions,
    EngineModel& result, std::string& error) {
    error.clear();
    const auto reject=[&](const std::string& why){error=why;return false;};
    if(!source.valid(&error)) return false;
    if(partitions.empty()) return reject("missing tissue mass partitions");
    if(!source.constraintProgram.empty()) return reject("spatial ConstraintIR must be recooked after COM rebase");
    std::vector<Vector> offsets(source.bodies.size());
    std::set<std::uint32_t> seen;
    for(const auto& p:partitions) {
        if(p.donorBody>=source.bodies.size() || !seen.insert(p.donorBody).second ||
           std::memcmp(&source.bodies[p.donorBody],&p.sourceBody,sizeof(p.sourceBody))!=0)
            return reject("stale, repeated or duplicate donor mass partition");
        if(!std::all_of(p.remainingCOMOffsetM.begin(),p.remainingCOMOffsetM.end(),[](double v){return std::isfinite(v);}))
            return reject("nonfinite donor frame offset");
        offsets[p.donorBody]=p.remainingCOMOffsetM;
    }
    EngineModel candidate=source;
    for(const auto& p:partitions) candidate.bodies[p.donorBody]=p.remainingBody;
    const auto shift=[&](mr_float4& point,std::uint32_t body){
        if(body==MR_INVALID_INDEX) return;
        point.x=static_cast<float>(point.x-offsets[body][0]);
        point.y=static_cast<float>(point.y-offsets[body][1]);
        point.z=static_cast<float>(point.z-offsets[body][2]);
    };
    for(auto& joint:candidate.joints) {
        shift(joint.parentAnchor,joint.parentBody);
        shift(joint.childAnchor,joint.childBody);
    }
    for(auto& shape:candidate.shapes) shift(shape.localPosition,shape.bodyIndex);
    for(const auto& articulation:source.articulations) {
        if(!seen.contains(articulation.rootBody)) continue;
        if(articulation.rootType!=MR_ROOT_FLOATING)
            return reject("a fixed-root donor requires an authored root-frame recompile");
        const auto& c=offsets[articulation.rootBody];
        const auto qi=articulation.qOffset,vi=articulation.vOffset;
        const Vector u{source.defaultQ[qi+3],source.defaultQ[qi+4],source.defaultQ[qi+5]};
        const double scalar=source.defaultQ[qi+6];
        const double qnorm=std::sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]+scalar*scalar);
        const Vector unit{u[0]/qnorm,u[1]/qnorm,u[2]/qnorm};
        const double s=scalar/qnorm;
        const Vector t{2*(unit[1]*c[2]-unit[2]*c[1]),2*(unit[2]*c[0]-unit[0]*c[2]),2*(unit[0]*c[1]-unit[1]*c[0])};
        const Vector world{c[0]+s*t[0]+unit[1]*t[2]-unit[2]*t[1],
                           c[1]+s*t[1]+unit[2]*t[0]-unit[0]*t[2],
                           c[2]+s*t[2]+unit[0]*t[1]-unit[1]*t[0]};
        const Vector w{source.defaultV[vi+3],source.defaultV[vi+4],source.defaultV[vi+5]};
        const Vector velocity{w[1]*world[2]-w[2]*world[1],w[2]*world[0]-w[0]*world[2],w[0]*world[1]-w[1]*world[0]};
        for(unsigned i=0;i<3;++i) {
            candidate.defaultQ[qi+i]=static_cast<float>(source.defaultQ[qi+i]+world[i]);
            candidate.defaultV[vi+i]=static_cast<float>(source.defaultV[vi+i]+velocity[i]);
        }
    }
    if(!candidate.valid(&error)) return false;
    result=std::move(candidate);return true;
}
} // namespace metalrobo
