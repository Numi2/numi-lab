// Opt-in CPU continuation for the measured production Human shape. It reads
// GPU-produced shared arenas and preserves all 64 ordered coupled sweeps;
// the Metal finish validates the payload and owns physical publication.
#pragma once

#include "metalrobo/engine_types.h"
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_joint_equality_gpu.h"
#include "metalrobo/numi_human_bilateral.h"
#include "metalrobo/numi_human_constraint_projection.h"
#include "metalrobo/numi_human_friction.h"
#include "metalrobo/numi_human_stand_cpu_finish_gpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace metalrobo::detail::stand_cpu_pilot {

constexpr unsigned kDofs = 128u;
constexpr unsigned kContacts = 10u;
constexpr unsigned kEqualities = 51u;
constexpr unsigned kSweeps = 64u;
constexpr float kRegularization = 1.0e-7f;
using Velocity = std::array<float, kDofs>;
using MassFactor = std::array<float, kDofs * kDofs>;

// Shadow the dependent 128-column Cholesky before changing its GPU owner.
// Every row retains the Metal kernel's original increasing inner order.
inline bool factorMass(const float* source, MassFactor& factor,
                       unsigned* failingColumn = nullptr) {
    if (source == nullptr) return false;
    std::array<float, kDofs> rowScale{};
    std::copy_n(source, factor.size(), factor.data());
    for (unsigned row = 0u; row < kDofs; ++row)
        for (unsigned column = 0u; column < kDofs; ++column)
            rowScale[row] = std::max(rowScale[row],
                std::abs(factor[row * kDofs + column]));
    for (unsigned column = 0u; column < kDofs; ++column) {
        float pivot = factor[column * kDofs + column];
        for (unsigned inner = 0u; inner < column; ++inner)
            pivot -= factor[column * kDofs + inner] *
                factor[column * kDofs + inner];
        if (!(pivot > std::max(1.0e-10f,
                              rowScale[column] * 8.0f *
                                  1.1920928955078125e-7f)) ||
            !std::isfinite(pivot)) {
            if (failingColumn) *failingColumn = column;
            return false;
        }
        factor[column * kDofs + column] = std::sqrt(pivot);
        for (unsigned row = column + 1u; row < kDofs; ++row) {
            float value = factor[row * kDofs + column];
            for (unsigned inner = 0u; inner < column; ++inner)
                value -= factor[row * kDofs + inner] *
                    factor[column * kDofs + inner];
            factor[row * kDofs + column] =
                value / factor[column * kDofs + column];
            if (!std::isfinite(factor[row * kDofs + column])) {
                if (failingColumn) *failingColumn = column;
                return false;
            }
        }
    }
    return true;
}

struct Contact {
    float gap{}, mu{}, slop{}, stabilization{}, seedForce{};
    bool active{};
    std::array<Velocity, 3u> jacobian{};
    std::array<float, 9u> matrix{};
    std::array<float, 3u> equalityWork{};
};
struct Limit {
    unsigned dof{};
    float position{}, lowerVelocity{}, upperVelocity{}, mass{};
};
struct Input {
    Velocity freeVelocity{};
    const float* responses = nullptr;
    const float* spatial = nullptr;
    const MRNumiHumanJointEqualityGPU* equalities = nullptr;
    std::array<Contact, kContacts> contacts{};
    std::array<Limit, kDofs> limits{};
    unsigned limitCount{};
    unsigned activeContacts{};
    float minimumGap = INFINITY;
    float maximumPenetration{};
    float timestep{};
};
struct Output {
    Velocity velocity{};
    std::array<float, kContacts * 3u> contactLambda{};
    std::array<float, kEqualities> equalityLambda{};
    std::array<float, kEqualities> finalEqualityRhs{};
    std::array<float, kDofs> limitLambda{};
    unsigned activeLimitUpdates{}, frictionCalls{};
    float contactNormalWork{}, contactTangentialWork{}, equalityWork{},
        limitWork{};
    float contactNormalAbsoluteWork{}, contactTangentialAbsoluteWork{},
        equalityAbsoluteWork{}, limitAbsoluteWork{};
};

// Call only after validating every backing MTLBuffer extent. The pilot is
// deliberately shape-bound so it cannot silently approximate another Human.
inline bool prepare(
    const MRNumiHumanStandDispatchGPU& dispatch,
    const MRArticulationGPU& articulation,
    const MRDofPropertiesGPU* dofs,
    const MRNumiHumanStandContactGPU* contacts,
    const float* q,
    const float* vector,
    const MRArticulatedPointWorldGPU* pointWorld,
    const mr_float4* pointLow,
    const float* pointJacobian,
    const MRNumiHumanJointEqualityGPU* equalities,
    const float* responses,
    const float* spatial,
    Input& input
) {
    if (articulation.nv != kDofs ||
        dispatch.supportContactCount != kContacts ||
        dispatch.jointEqualityCount != kEqualities ||
        dispatch.contactIterationCount != kSweeps ||
        dispatch.groundNormal.x != 0.0f ||
        dispatch.groundNormal.y != 0.0f ||
        dispatch.groundNormal.z != 1.0f ||
        dispatch.groundPointAndTimestep.w <= 0.0f ||
        dofs == nullptr || contacts == nullptr || q == nullptr ||
        vector == nullptr || pointWorld == nullptr || pointLow == nullptr ||
        pointJacobian == nullptr || equalities == nullptr ||
        responses == nullptr || spatial == nullptr)
        return false;
    input.responses = responses;
    input.spatial = spatial;
    input.equalities = equalities;
    input.timestep = dispatch.groundPointAndTimestep.w;
    for (unsigned dof = 0u; dof < kDofs; ++dof) {
        input.freeVelocity[dof] = vector[kDofs + dof];
        if (!std::isfinite(input.freeVelocity[dof])) return false;
        const auto& meta = dofs[articulation.vOffset + dof];
        if ((meta.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u) continue;
        if (meta.qIndex < articulation.qOffset ||
            meta.qIndex >= articulation.qOffset + articulation.nq ||
            input.limitCount >= kDofs ||
            !(meta.limits.x < meta.limits.y) ||
            !std::isfinite(meta.limits.x) ||
            !std::isfinite(meta.limits.y)) return false;
        const unsigned qIndex = meta.qIndex - articulation.qOffset;
        const float position = q[qIndex];
        if (!std::isfinite(position)) return false;
        const float lower = mrNumiHumanLowerLimitVelocityTarget(
            position, meta.limits.x, input.timestep);
        const float upper = mrNumiHumanUpperLimitVelocityTarget(
            position, meta.limits.y, input.timestep);
        const unsigned row = 3u * kContacts + kEqualities + dof;
        const float mass = responses[row * kDofs + dof] +
            kRegularization;
        if (!std::isfinite(lower) || !std::isfinite(upper) ||
            lower > upper || !std::isfinite(mass) ||
            !(mass > kRegularization)) return false;
        input.limits[input.limitCount++] = {dof, position, lower, upper, mass};
    }
    for (unsigned contact = 0u; contact < kContacts; ++contact) {
        const auto& meta = contacts[contact];
        auto& out = input.contacts[contact];
        const unsigned point = meta.pointQueryIndex;
        if (meta.bodyIndex < articulation.firstBody ||
            meta.bodyIndex >= articulation.firstBody +
                articulation.bodyCount ||
            point >= dispatch.pointWorldStride ||
            meta.reserved0 != 0u ||
            !std::isfinite(meta.frictionSlopAndStabilization.x) ||
            !std::isfinite(meta.frictionSlopAndStabilization.y) ||
            !std::isfinite(meta.frictionSlopAndStabilization.z) ||
            !std::isfinite(meta.frictionSlopAndStabilization.w) ||
            meta.frictionSlopAndStabilization.x < 0.0f ||
            meta.frictionSlopAndStabilization.y < 0.0f ||
            meta.frictionSlopAndStabilization.z < 0.0f ||
            meta.frictionSlopAndStabilization.z > 1.0f ||
            meta.frictionSlopAndStabilization.w < 0.0f) return false;
        out.mu = meta.frictionSlopAndStabilization.x;
        out.slop = meta.frictionSlopAndStabilization.y;
        out.stabilization = meta.frictionSlopAndStabilization.z;
        out.seedForce = meta.frictionSlopAndStabilization.w;
        out.gap = pointWorld[point].position.z + pointLow[point].z -
            dispatch.groundPointAndTimestep.z;
        out.active = out.gap <= out.slop;
        if (!std::isfinite(out.gap)) return false;
        input.minimumGap = std::min(input.minimumGap, out.gap);
        input.maximumPenetration = std::max(input.maximumPenetration,
                                            std::max(-out.gap, 0.0f));
        if (out.active) ++input.activeContacts;
        for (unsigned axis = 0u; axis < 3u; ++axis) {
            const unsigned coordinate = (axis + 2u) % 3u;
            for (unsigned dof = 0u; dof < kDofs; ++dof)
                out.jacobian[axis][dof] = pointJacobian[
                    point * 3u * kDofs + coordinate * kDofs + dof];
        }
        if (!out.active) continue;
        const unsigned projectedBase = (2u + kEqualities) * kDofs;
        for (unsigned axis = 0u; axis < 3u; ++axis)
            for (unsigned row = 0u; row < kEqualities; ++row)
                out.equalityWork[axis] = std::fma(
                    spatial[projectedBase +
                        (3u * contact + axis) * kEqualities + row],
                    spatial[kDofs + row], out.equalityWork[axis]);
        for (unsigned row = 0u; row < 3u; ++row) {
            for (unsigned column = 0u; column < 3u; ++column) {
                float value = row == column ? kRegularization : 0.0f;
                const float* response = responses +
                    (3u * contact + column) * kDofs;
                for (unsigned dof = 0u; dof < kDofs; ++dof)
                    value += out.jacobian[row][dof] * response[dof];
                out.matrix[3u * row + column] = value;
            }
        }
    }
    return true;
}

inline bool solve(const Input& input, Output& out) {
    out.velocity = input.freeVelocity;
    const unsigned responseColumns =
        (3u * kContacts + kEqualities + kDofs) * kDofs;
    const float* factor = input.responses + responseColumns;
    const float* scale = factor + kEqualities * kEqualities;
    const float* pivot = scale + kEqualities;
    for (unsigned refinement = 0u; refinement < 2u; ++refinement) {
        std::array<float, kEqualities> rhs{};
        std::array<float, kEqualities> before{};
        for (unsigned row = 0u; row < kEqualities; ++row) {
            const auto& equality = input.equalities[row];
            float velocity = out.velocity[equality.indices.y];
            if (equality.indices.w != MR_INVALID_INDEX)
                velocity = std::fma(-input.spatial[row],
                    out.velocity[equality.indices.w], velocity);
            before[row] = velocity;
            rhs[row] = input.spatial[kDofs + row] - velocity;
        }
        if (!mrNumiHumanBilateralSolve(factor, scale, pivot,
                                       rhs.data(), kEqualities)) return false;
        out.finalEqualityRhs = rhs;
        float refinementWork = 0.0f;
        for (unsigned row = 0u; row < kEqualities; ++row) {
            refinementWork = std::fma(0.5f * rhs[row], before[row],
                                      refinementWork);
            out.equalityLambda[row] += rhs[row];
        }
        for (unsigned dof = 0u; dof < kDofs; ++dof) {
            float correction = 0.0f;
            for (unsigned row = 0u; row < kEqualities; ++row)
                correction = std::fma(rhs[row],
                    input.spatial[2u * kDofs + dof * kEqualities + row],
                    correction);
            out.velocity[dof] += correction;
        }
        for (unsigned row = 0u; row < kEqualities; ++row) {
            const auto& equality = input.equalities[row];
            float velocity = out.velocity[equality.indices.y];
            if (equality.indices.w != MR_INVALID_INDEX)
                velocity = std::fma(-input.spatial[row],
                    out.velocity[equality.indices.w], velocity);
            refinementWork = std::fma(0.5f * rhs[row], velocity,
                                      refinementWork);
        }
        if (!std::isfinite(refinementWork)) return false;
        out.equalityWork += refinementWork;
        out.equalityAbsoluteWork += std::abs(refinementWork);
    }
    for (unsigned contact = 0u; contact < kContacts; ++contact) {
        const auto& c = input.contacts[contact];
        if (!c.active) continue;
        const float seed = mrNumiHumanSupportSeedImpulse(
            c.seedForce, input.timestep, c.gap, c.slop);
        if (!std::isfinite(seed)) return false;
        out.contactLambda[3u * contact] = seed;
        const float* response = input.responses + 3u * contact * kDofs;
        float normalVelocityBeforeSeed = 0.0f;
        for (unsigned dof = 0u; dof < kDofs; ++dof)
            normalVelocityBeforeSeed += c.jacobian[0u][dof] *
                out.velocity[dof];
        for (unsigned dof = 0u; dof < kDofs; ++dof)
            out.velocity[dof] += seed * response[dof];
        const float inducedWork = seed * c.equalityWork[0u];
        const float normalVelocityAfterSeed = std::fma(seed,
            c.matrix[0u] - kRegularization, normalVelocityBeforeSeed);
        const float seedWork = 0.5f * seed *
            (normalVelocityBeforeSeed + normalVelocityAfterSeed);
        if (!std::isfinite(inducedWork) || !std::isfinite(seedWork))
            return false;
        out.equalityWork += inducedWork;
        out.equalityAbsoluteWork += std::abs(inducedWork);
        out.contactNormalWork += seedWork;
        out.contactNormalAbsoluteWork += std::abs(seedWork);
    }
    for (unsigned sweep = 0u; sweep < kSweeps; ++sweep) {
        for (unsigned contact = 0u; contact < kContacts; ++contact) {
            const auto& c = input.contacts[contact];
            if (!c.active) continue;
            float velocity[3u]{};
            for (unsigned axis = 0u; axis < 3u; ++axis)
                for (unsigned dof = 0u; dof < kDofs; ++dof)
                    velocity[axis] += c.jacobian[axis][dof] *
                        out.velocity[dof];
            const float target = mrNumiHumanContactVelocityTarget(
                c.gap, input.timestep, c.stabilization);
            const auto& w = c.matrix;
            const float tangentDeterminant =
                w[4u] * w[8u] - w[5u] * w[7u];
            if (!(w[0u] > kRegularization) ||
                !std::isfinite(w[0u]) ||
                (c.mu > 0.0f &&
                 (!(tangentDeterminant > kRegularization) ||
                  !std::isfinite(tangentDeterminant))))
                return false;
            float next[3u], prior[3u];
            for (unsigned axis = 0u; axis < 3u; ++axis)
                next[axis] = prior[axis] =
                    out.contactLambda[3u * contact + axis];
            next[0u] = std::max(prior[0u] +
                (target - velocity[0u]) / w[0u], 0.0f);
            const float normalDelta = next[0u] - prior[0u];
            const float tangentY = velocity[1u] + w[3u] * normalDelta;
            const float tangentZ = velocity[2u] + w[6u] * normalDelta;
            if (c.mu > 0.0f && next[0u] > 0.0f) {
                const float rhsY = w[4u] * prior[1u] + w[5u] * prior[2u]
                    - tangentY;
                const float rhsZ = w[7u] * prior[1u] + w[8u] * prior[2u]
                    - tangentZ;
                const auto tangent = mrNumiHumanSolveFrictionDisk(
                    w[4u], 0.5f * w[5u] + 0.5f * w[7u], w[8u],
                    rhsY, rhsZ, c.mu * next[0u]);
                if (!tangent.valid) return false;
                next[1u] = tangent.x;
                next[2u] = tangent.y;
                ++out.frictionCalls;
            } else next[1u] = next[2u] = 0.0f;
            const float applied[3u]{next[0u] - prior[0u],
                next[1u] - prior[1u], next[2u] - prior[2u]};
            float inducedWork = 0.0f;
            for (unsigned axis = 0u; axis < 3u; ++axis)
                inducedWork = std::fma(applied[axis],
                    c.equalityWork[axis], inducedWork);
            const float normalVelocityAfter = std::fma(applied[0u],
                w[0u] - kRegularization, velocity[0u]);
            const float tangentBeforeY = std::fma(applied[0u],
                w[3u], velocity[1u]);
            const float tangentBeforeZ = std::fma(applied[0u],
                w[6u], velocity[2u]);
            float tangentAfterY = std::fma(applied[1u],
                w[4u] - kRegularization, tangentBeforeY);
            tangentAfterY = std::fma(applied[2u], w[5u], tangentAfterY);
            float tangentAfterZ = std::fma(applied[1u],
                w[7u], tangentBeforeZ);
            tangentAfterZ = std::fma(applied[2u],
                w[8u] - kRegularization, tangentAfterZ);
            const float normalWork = 0.5f * applied[0u] *
                (velocity[0u] + normalVelocityAfter);
            const float tangentWork = 0.5f *
                (applied[1u] * (tangentBeforeY + tangentAfterY) +
                 applied[2u] * (tangentBeforeZ + tangentAfterZ));
            if (!std::isfinite(inducedWork) ||
                !std::isfinite(normalWork) ||
                !std::isfinite(tangentWork)) return false;
            out.equalityWork += inducedWork;
            out.equalityAbsoluteWork += std::abs(inducedWork);
            out.contactNormalWork += normalWork;
            out.contactTangentialWork += tangentWork;
            out.contactNormalAbsoluteWork += std::abs(normalWork);
            out.contactTangentialAbsoluteWork += std::abs(tangentWork);
            for (unsigned axis = 0u; axis < 3u; ++axis) {
                const float impulse = applied[axis];
                out.contactLambda[3u * contact + axis] = next[axis];
                const float* response = input.responses +
                    (3u * contact + axis) * kDofs;
                for (unsigned dof = 0u; dof < kDofs; ++dof)
                    out.velocity[dof] += impulse * response[dof];
            }
        }
        for (unsigned limit = 0u; limit < input.limitCount; ++limit) {
            const auto& item = input.limits[limit];
            const unsigned dof = item.dof;
            const float previous = out.limitLambda[dof];
            const float next = mrNumiHumanProjectIntervalImpulse(
                previous, out.velocity[dof], item.lowerVelocity,
                item.upperVelocity, item.mass);
            const float impulse = next - previous;
            if (!std::isfinite(impulse)) return false;
            if (impulse == 0.0f) continue;
            out.limitLambda[dof] = next;
            ++out.activeLimitUpdates;
            const float* response = input.responses +
                (3u * kContacts + kEqualities + dof) * kDofs;
            const float* reaction = factor +
                kEqualities * (kEqualities + 3u) + dof * kEqualities;
            float limitWork = 0.5f * impulse * out.velocity[dof];
            float equalityWork = 0.0f;
            for (unsigned row = 0u; row < kEqualities; ++row) {
                const auto& equality = input.equalities[row];
                float velocity = out.velocity[equality.indices.y];
                if (equality.indices.w != MR_INVALID_INDEX)
                    velocity = std::fma(-input.spatial[row],
                        out.velocity[equality.indices.w], velocity);
                const float equalityImpulse = impulse * reaction[row];
                equalityWork = std::fma(0.5f * equalityImpulse,
                    velocity, equalityWork);
                out.equalityLambda[row] += equalityImpulse;
            }
            for (unsigned index = 0u; index < kDofs; ++index)
                out.velocity[index] = std::fma(impulse, response[index],
                                               out.velocity[index]);
            limitWork = std::fma(0.5f * impulse, out.velocity[dof],
                                 limitWork);
            for (unsigned row = 0u; row < kEqualities; ++row) {
                const auto& equality = input.equalities[row];
                float velocity = out.velocity[equality.indices.y];
                if (equality.indices.w != MR_INVALID_INDEX)
                    velocity = std::fma(-input.spatial[row],
                        out.velocity[equality.indices.w], velocity);
                equalityWork = std::fma(0.5f * impulse * reaction[row],
                    velocity, equalityWork);
            }
            if (!std::isfinite(limitWork) ||
                !std::isfinite(equalityWork)) return false;
            out.limitWork += limitWork;
            out.limitAbsoluteWork += std::abs(limitWork);
            out.equalityWork += equalityWork;
            out.equalityAbsoluteWork += std::abs(equalityWork);
        }
    }
    const unsigned projectedBase = (2u + kEqualities) * kDofs;
    for (unsigned row = 0u; row < kEqualities; ++row) {
        float contactImpulse = 0.0f;
        for (unsigned contact = 0u; contact < kContacts; ++contact)
            for (unsigned axis = 0u; axis < 3u; ++axis)
                contactImpulse = std::fma(
                    out.contactLambda[3u * contact + axis],
                    input.spatial[projectedBase +
                        (3u * contact + axis) * kEqualities + row],
                    contactImpulse);
        out.equalityLambda[row] += contactImpulse;
    }
    for (float velocity : out.velocity)
        if (!std::isfinite(velocity)) return false;
    return true;
}

inline bool publish(const Input& input, const Output& out,
                    unsigned stepIndex,
                    MRNumiHumanStandCpuFinishGPU& payload) {
    MRNumiHumanStandCpuFinishGPU next{};
    next.stepIndex = stepIndex;
    next.dofCount = kDofs;
    next.contactCount = kContacts;
    next.equalityCount = kEqualities;
    next.limitCount = input.limitCount;
    next.activeContacts = input.activeContacts;
    next.minimumGap = input.minimumGap;
    next.maximumPenetration = input.maximumPenetration;
    next.contactNormalImpulseWork = out.contactNormalWork;
    next.contactTangentialImpulseWork = out.contactTangentialWork;
    next.equalityImpulseWork = out.equalityWork;
    next.sourceLimitImpulseWork = out.limitWork;
    next.contactNormalAbsoluteImpulseWork =
        out.contactNormalAbsoluteWork;
    next.contactTangentialAbsoluteImpulseWork =
        out.contactTangentialAbsoluteWork;
    next.equalityAbsoluteImpulseWork = out.equalityAbsoluteWork;
    next.sourceLimitAbsoluteImpulseWork = out.limitAbsoluteWork;
    const std::array<float, 10u> scalars{
        next.minimumGap, next.maximumPenetration,
        next.contactNormalImpulseWork, next.contactTangentialImpulseWork,
        next.equalityImpulseWork, next.sourceLimitImpulseWork,
        next.contactNormalAbsoluteImpulseWork,
        next.contactTangentialAbsoluteImpulseWork,
        next.equalityAbsoluteImpulseWork,
        next.sourceLimitAbsoluteImpulseWork,
    };
    for (float value : scalars)
        if (!std::isfinite(value)) return false;
    for (unsigned dof = 0u; dof < kDofs; ++dof) {
        if (!std::isfinite(out.velocity[dof])) return false;
        next.velocity[dof] = out.velocity[dof];
    }
    for (unsigned contact = 0u; contact < kContacts; ++contact) {
        const auto& source = input.contacts[contact];
        next.contactActiveForPostProjection[contact] =
            source.active ? 1u : 0u;
        next.contactTargetNormalVelocityForPostProjection[contact] =
            source.active ? mrNumiHumanContactVelocityTarget(
                source.gap, input.timestep, source.stabilization) : 0.0f;
        if (!std::isfinite(
                next.contactTargetNormalVelocityForPostProjection[contact]))
            return false;
        for (unsigned axis = 0u; axis < 3u; ++axis)
            next.contactLambdas[3u * contact + axis] =
                out.contactLambda[3u * contact + axis];
        for (unsigned entry = 0u; entry < 9u; ++entry)
            next.contactMatrices[9u * contact + entry] =
                source.matrix[entry];
        for (unsigned axis = 0u; axis < 3u; ++axis)
            if (!std::isfinite(next.contactLambdas[3u * contact + axis]))
                return false;
        if (source.active)
            for (unsigned entry = 0u; entry < 9u; ++entry)
                if (!std::isfinite(next.contactMatrices[9u * contact + entry]))
                    return false;
    }
    for (unsigned row = 0u; row < kEqualities; ++row) {
        next.equalityLambdas[row] = out.equalityLambda[row];
        next.equalityRhs[row] = out.finalEqualityRhs[row];
        if (!std::isfinite(next.equalityLambdas[row]) ||
            !std::isfinite(next.equalityRhs[row])) return false;
    }
    for (unsigned index = 0u; index < input.limitCount; ++index) {
        const auto& limit = input.limits[index];
        if (limit.dof >= kDofs || !std::isfinite(limit.position))
            return false;
        next.limitDofs[index] = limit.dof;
        next.limitPreStepPositions[index] = limit.position;
        next.limitAccumulatedImpulses[index] =
            out.limitLambda[limit.dof];
        if (!std::isfinite(next.limitAccumulatedImpulses[index]))
            return false;
    }
    next.abiVersion = MR_NUMI_HUMAN_STAND_CPU_FINISH_ABI_VERSION;
    payload = next;
    return true;
}

} // namespace metalrobo::detail::stand_cpu_pilot
