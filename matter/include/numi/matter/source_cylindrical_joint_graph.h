#pragma once

// Assemble FEBio-style cylindrical-joint residuals for a fixed rigid-body
// graph. This is an equation assembler for a caller's existing nonlinear
// solve, not a standalone time integrator or a complete FEBio model. The
// generalized motion map is frozen for one linearization; callers using an
// articulated map must add its geometric stiffness contribution.
#include "numi/matter/source_cylindrical_joint.h"
#include <array>
#include <cstddef>

namespace numi_matter_joint {

template<class T> struct BodyState {
    V<T> position; // center of mass, in the graph's consistent length unit
    Q<T> rotation;
};

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

template<class T> struct RigidSpring {
    std::size_t bodyA = 0;
    std::size_t bodyB = 0;
    // Source reference centers and insertion coordinates are in the same
    // source frame. FEBio's zero free_length sentinel selects the initial
    // insertion-point separation.
    V<T> referenceA, referenceB, insertionA, insertionB;
    T stiffness{};
    T freeLength{};
};

template<class T> struct RigidSpringDirection {
    V<T> linearA, angularA, linearB, angularB;
};

template<class T> struct RigidSpringOutput {
    V<T> forceA, momentA, forceB, momentB;
    T storedEnergy{};
};

template<class T> inline bool evaluateRigidSpring(
    const RigidSpring<T>& spring,
    const BodyState<T>& bodyA,
    const BodyState<T>& bodyB,
    const RigidSpringDirection<T>& direction,
    RigidSpringOutput<T>& result,
    RigidSpringOutput<T>& derivative
) {
    if (!finite(spring.referenceA) || !finite(spring.referenceB) ||
        !finite(spring.insertionA) || !finite(spring.insertionB) ||
        !isfinite(spring.stiffness) || spring.stiffness <= T(0) ||
        !isfinite(spring.freeLength) || spring.freeLength < T(0) ||
        !finite(direction.linearA) || !finite(direction.angularA) ||
        !finite(direction.linearB) || !finite(direction.angularB)) return false;
    const V<T> armA = rotate(bodyA.rotation,
        spring.insertionA - spring.referenceA);
    const V<T> armB = rotate(bodyB.rotation,
        spring.insertionB - spring.referenceB);
    const V<T> anchorA = bodyA.position + armA;
    const V<T> anchorB = bodyB.position + armB;
    const V<T> gap = anchorB - anchorA;
    const T lengthSquared = dot(gap, gap);
    if (!isfinite(lengthSquared) || lengthSquared < T(0)) return false;
    const T length = sqrt(lengthSquared);
    const V<T> initialGap = spring.insertionB - spring.insertionA;
    const T restLength = spring.freeLength == T(0)
        ? sqrt(dot(initialGap, initialGap)) : spring.freeLength;
    if (!isfinite(length) || !isfinite(restLength)) return false;

    V<T> forceA{}, derivativeForceA{};
    if (length > T(1.0e-12)) {
        const V<T> axis = gap * (T(1) / length);
        const T extension = length - restLength;
        forceA = axis * (spring.stiffness * extension);
        const V<T> anchorVelocityA = direction.linearA +
            cross(direction.angularA, armA);
        const V<T> anchorVelocityB = direction.linearB +
            cross(direction.angularB, armB);
        const V<T> gapVelocity = anchorVelocityB - anchorVelocityA;
        const T lengthVelocity = dot(axis, gapVelocity);
        derivativeForceA = gapVelocity *
                (spring.stiffness * extension / length) +
            axis * (spring.stiffness * lengthVelocity * restLength / length);
    } else {
        // At the source zero-free-length rest configuration the spring law is
        // smooth: force=k*gap, with tangent kI. A nonzero free length at a
        // collapsed span has no unique axis and must be rejected.
        if (restLength != T(0)) return false;
        const V<T> anchorVelocityA = direction.linearA +
            cross(direction.angularA, armA);
        const V<T> anchorVelocityB = direction.linearB +
            cross(direction.angularB, armB);
        derivativeForceA = (anchorVelocityB - anchorVelocityA) * spring.stiffness;
    }
    const V<T> derivativeArmA = cross(direction.angularA, armA);
    const V<T> derivativeArmB = cross(direction.angularB, armB);
    const V<T> momentA = cross(armA, forceA);
    const V<T> momentB = cross(armB, forceA * T(-1));
    const V<T> derivativeMomentA = cross(derivativeArmA, forceA) +
        cross(armA, derivativeForceA);
    const V<T> derivativeMomentB = cross(derivativeArmB, forceA * T(-1)) +
        cross(armB, derivativeForceA * T(-1));
    const T extension = length - restLength;
    RigidSpringOutput<T> out{
        forceA, momentA, forceA * T(-1), momentB,
        T(0.5) * spring.stiffness * extension * extension};
    RigidSpringOutput<T> tangent{
        derivativeForceA, derivativeMomentA,
        derivativeForceA * T(-1), derivativeMomentB, T(0)};
    if (!finite(out.forceA) || !finite(out.momentA) ||
        !finite(out.forceB) || !finite(out.momentB) ||
        !finite(tangent.forceA) || !finite(tangent.momentA) ||
        !finite(tangent.forceB) || !finite(tangent.momentB) ||
        !isfinite(out.storedEnergy)) return false;
    result = out;
    derivative = tangent;
    return true;
}

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
