#pragma once

// Assemble FEBio-style cylindrical-joint residuals for a fixed rigid-body
// graph. This is an equation assembler for a caller's existing nonlinear
// solve, not a standalone time integrator or a complete FEBio model. The
// generalized motion map is frozen for one linearization; callers using an
// articulated map must add its geometric stiffness contribution.
#include "numi/matter/source_cylindrical_joint.h"
#include "numi/matter/source_rigid_spring.h"
#include <array>
#include <cstddef>

namespace numi_matter_joint {

template<class T> struct BodyMotion {
    V<T> linear;   // world spatial velocity at the center of mass
    V<T> angular;  // world spatial angular velocity
};

template<class T> struct BodyWrench {
    V<T> force;
    V<T> moment; // world moment about the center of mass
};

template<class T> struct CylindricalJoint {
    std::size_t bodyA = 0;
    std::size_t bodyB = 0;
    // Source reference centers and connector parameters. Current body states
    // supply positionA/B and rotationA/B during evaluation.
    Input<T> source;
};

template<class T, std::size_t BodyCount, std::size_t JointCount,
         std::size_t DofCount, std::size_t SpringCount = 0>
struct GraphInput {
    std::array<BodyState<T>, BodyCount> bodies{};
    std::array<CylindricalJoint<T>, JointCount> joints{};
    std::array<RigidSpring<T>, SpringCount> springs{};
    // Column-major motion map: motion[dof][body]. A prescribed/fixed body
    // coordinate is represented by zero entries in every motion column.
    std::array<std::array<BodyMotion<T>, BodyCount>, DofCount> motion{};
    std::array<T, DofCount> appliedGeneralizedForce{};
};

template<class T, std::size_t BodyCount, std::size_t JointCount,
         std::size_t DofCount, std::size_t SpringCount = 0>
struct GraphOutput {
    std::array<BodyWrench<T>, BodyCount> bodyWrenches{};
    std::array<T, DofCount> residual{};
    // Row-major d(residual)/d(generalized increment). This is the exact
    // connector directional tangent for a locally constant motion map.
    std::array<T, DofCount * DofCount> tangent{};
    T rigidSpringStoredEnergy{};
};

template<class T, std::size_t BodyCount, std::size_t JointCount,
         std::size_t DofCount, std::size_t SpringCount>
inline bool evaluateGraph(
    const GraphInput<T, BodyCount, JointCount, DofCount, SpringCount>& graph,
    GraphOutput<T, BodyCount, JointCount, DofCount, SpringCount>& result
) {
    if constexpr (BodyCount == 0 || DofCount == 0) return false;
    GraphOutput<T, BodyCount, JointCount, DofCount, SpringCount> candidate{};
    for (const BodyState<T>& body : graph.bodies) {
        const T n = norm2(body.rotation);
        if (!finite(body.position) || !isfinite(n) ||
            n < T(.99999) || n > T(1.00001)) return false;
    }
    for (const CylindricalJoint<T>& joint : graph.joints) {
        if (joint.bodyA >= BodyCount || joint.bodyB >= BodyCount ||
            joint.bodyA == joint.bodyB) return false;
        Input<T> input = joint.source;
        input.positionA = graph.bodies[joint.bodyA].position;
        input.positionB = graph.bodies[joint.bodyB].position;
        input.rotationA = graph.bodies[joint.bodyA].rotation;
        input.rotationB = graph.bodies[joint.bodyB].rotation;
        Output<T> wrench{}, unused{};
        if (!evaluate(input, Direction<T>{}, wrench, unused)) return false;
        BodyWrench<T>& a = candidate.bodyWrenches[joint.bodyA];
        BodyWrench<T>& b = candidate.bodyWrenches[joint.bodyB];
        a.force = a.force + wrench.forceA;
        a.moment = a.moment + wrench.momentA;
        b.force = b.force + wrench.forceB;
        b.moment = b.moment + wrench.momentB;
    }
    for (const RigidSpring<T>& spring : graph.springs) {
        if (spring.bodyA >= BodyCount || spring.bodyB >= BodyCount ||
            spring.bodyA == spring.bodyB) return false;
        RigidSpringOutput<T> wrench{}, unused{};
        if (!evaluateRigidSpring(spring, graph.bodies[spring.bodyA],
                                 graph.bodies[spring.bodyB], {},
                                 wrench, unused)) return false;
        BodyWrench<T>& a = candidate.bodyWrenches[spring.bodyA];
        BodyWrench<T>& b = candidate.bodyWrenches[spring.bodyB];
        a.force = a.force + wrench.forceA;
        a.moment = a.moment + wrench.momentA;
        b.force = b.force + wrench.forceB;
        b.moment = b.moment + wrench.momentB;
        candidate.rigidSpringStoredEnergy += wrench.storedEnergy;
    }
    for (std::size_t row = 0; row < DofCount; ++row) {
        T value = -graph.appliedGeneralizedForce[row];
        for (std::size_t body = 0; body < BodyCount; ++body) {
            value += dot(candidate.bodyWrenches[body].force,
                         graph.motion[row][body].linear) +
                     dot(candidate.bodyWrenches[body].moment,
                         graph.motion[row][body].angular);
        }
        if (!isfinite(value)) return false;
        candidate.residual[row] = value;
    }
    for (std::size_t column = 0; column < DofCount; ++column) {
        for (const CylindricalJoint<T>& joint : graph.joints) {
            Input<T> input = joint.source;
            input.positionA = graph.bodies[joint.bodyA].position;
            input.positionB = graph.bodies[joint.bodyB].position;
            input.rotationA = graph.bodies[joint.bodyA].rotation;
            input.rotationB = graph.bodies[joint.bodyB].rotation;
            const BodyMotion<T>& ma = graph.motion[column][joint.bodyA];
            const BodyMotion<T>& mb = graph.motion[column][joint.bodyB];
            Direction<T> direction{ma.linear, ma.angular, mb.linear, mb.angular};
            Output<T> unused{}, derivative{};
            if (!evaluate(input, direction, unused, derivative)) return false;
            for (std::size_t row = 0; row < DofCount; ++row) {
                const BodyMotion<T>& ra = graph.motion[row][joint.bodyA];
                const BodyMotion<T>& rb = graph.motion[row][joint.bodyB];
                candidate.tangent[row * DofCount + column] +=
                    dot(derivative.forceA, ra.linear) +
                    dot(derivative.momentA, ra.angular) +
                    dot(derivative.forceB, rb.linear) +
                    dot(derivative.momentB, rb.angular);
            }
        }
        for (const RigidSpring<T>& spring : graph.springs) {
            const BodyMotion<T>& ma = graph.motion[column][spring.bodyA];
            const BodyMotion<T>& mb = graph.motion[column][spring.bodyB];
            RigidSpringDirection<T> direction{
                ma.linear, ma.angular, mb.linear, mb.angular};
            RigidSpringOutput<T> unused{}, derivative{};
            if (!evaluateRigidSpring(spring, graph.bodies[spring.bodyA],
                                     graph.bodies[spring.bodyB], direction,
                                     unused, derivative)) return false;
            for (std::size_t row = 0; row < DofCount; ++row) {
                const BodyMotion<T>& ra = graph.motion[row][spring.bodyA];
                const BodyMotion<T>& rb = graph.motion[row][spring.bodyB];
                candidate.tangent[row * DofCount + column] +=
                    dot(derivative.forceA, ra.linear) +
                    dot(derivative.momentA, ra.angular) +
                    dot(derivative.forceB, rb.linear) +
                    dot(derivative.momentB, rb.angular);
            }
        }
    }
    if (!isfinite(candidate.rigidSpringStoredEnergy)) return false;
    for (const T& value : candidate.tangent) {
        if (!isfinite(value)) return false;
    }
    result = candidate;
    return true;
}

} // namespace numi_matter_joint
