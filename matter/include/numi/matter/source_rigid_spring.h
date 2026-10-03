#pragma once
#include "numi/matter/source_cylindrical_joint.h"
#ifdef __METAL_VERSION__
#define NM_SPRING_REF thread
#else
#define NM_SPRING_REF
#endif

namespace numi_matter_joint {
template<class T> struct BodyState {
    V<T> position; // center of mass in the source frame
    Q<T> rotation;
};

template<class T> struct RigidSpring {
    unsigned bodyA = 0;
    unsigned bodyB = 0;
    // Source reference centers and insertion coordinates are in the same
    // source frame. FEBio's zero free_length sentinel selects the initial
    // insertion-point separation.
    V<T> referenceA, referenceB, insertionA, insertionB;
    T stiffness{};
    T freeLength{};
    // Pinned MPFL/LPFL curve: zero force in compression, linear in tension
    // between its recorded -1, 0, +1 mm knots. FEBio's default load-curve
    // extension holds the endpoint force constant outside that interval.
    bool sourcePiecewise = false;
    T sourceKnotExtent{};
};

template<class T> struct RigidSpringDirection {
    V<T> linearA, angularA, linearB, angularB;
};

template<class T> struct RigidSpringOutput {
    V<T> forceA, momentA, forceB, momentB;
    T storedEnergy{};
};

template<class T> inline bool evaluateRigidSpring(
    RigidSpring<T> spring,
    BodyState<T> bodyA,
    BodyState<T> bodyB,
    RigidSpringDirection<T> direction,
    NM_SPRING_REF RigidSpringOutput<T>& result,
    NM_SPRING_REF RigidSpringOutput<T>& derivative
) {
    if (!finite(spring.referenceA) || !finite(spring.referenceB) ||
        !finite(spring.insertionA) || !finite(spring.insertionB) ||
        !isfinite(spring.stiffness) || spring.stiffness <= T(0) ||
        !isfinite(spring.freeLength) || spring.freeLength < T(0) ||
        (spring.sourcePiecewise &&
            (!isfinite(spring.sourceKnotExtent) ||
             !(spring.sourceKnotExtent > T(0)))) ||
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
    const T extension = length - restLength;
    T forceMagnitude = spring.stiffness * extension;
    T localStiffness = spring.stiffness;
    T storedEnergy = T(0.5) * spring.stiffness * extension * extension;
    if (spring.sourcePiecewise) {
        const auto curveForce = [&](const T x) {
            return spring.stiffness *
                (x <= T(0) ? T(0) :
                 (x >= spring.sourceKnotExtent ? spring.sourceKnotExtent : x));
        };
        forceMagnitude = curveForce(extension);
        // FEDataLoadCurve::Deriv uses a centered difference over 0.1% of
        // the full curve domain. This includes the half-slope at each knot.
        const T halfWidth = T(0.002) * spring.sourceKnotExtent;
        localStiffness =
            (curveForce(extension + halfWidth) -
             curveForce(extension - halfWidth)) / (T(2) * halfWidth);
        const T loaded = extension <= T(0) ? T(0) :
            (extension >= spring.sourceKnotExtent
                ? spring.sourceKnotExtent : extension);
        storedEnergy = T(0.5) * spring.stiffness * loaded * loaded;
        if (extension > spring.sourceKnotExtent)
            storedEnergy += spring.stiffness * spring.sourceKnotExtent *
                (extension - spring.sourceKnotExtent);
    }

    V<T> forceA{}, derivativeForceA{};
    if (length > T(1.0e-12)) {
        const V<T> axis = gap * (T(1) / length);
        forceA = axis * forceMagnitude;
        const V<T> anchorVelocityA = direction.linearA +
            cross(direction.angularA, armA);
        const V<T> anchorVelocityB = direction.linearB +
            cross(direction.angularB, armB);
        const V<T> gapVelocity = anchorVelocityB - anchorVelocityA;
        const T lengthVelocity = dot(axis, gapVelocity);
        derivativeForceA = gapVelocity * (forceMagnitude / length) +
            axis * ((localStiffness - forceMagnitude / length) *
                    lengthVelocity);
    } else {
        // At the source zero-free-length rest configuration the spring law is
        // smooth: force=k*gap, with tangent kI. A nonzero free length at a
        // collapsed span has no unique axis and must be rejected.
        if (restLength != T(0)) return false;
        const V<T> anchorVelocityA = direction.linearA +
            cross(direction.angularA, armA);
        const V<T> anchorVelocityB = direction.linearB +
            cross(direction.angularB, armB);
        derivativeForceA = (anchorVelocityB - anchorVelocityA) * localStiffness;
    }
    const V<T> derivativeArmA = cross(direction.angularA, armA);
    const V<T> derivativeArmB = cross(direction.angularB, armB);
    const V<T> momentA = cross(armA, forceA);
    const V<T> momentB = cross(armB, forceA * T(-1));
    const V<T> derivativeMomentA = cross(derivativeArmA, forceA) +
        cross(armA, derivativeForceA);
    const V<T> derivativeMomentB = cross(derivativeArmB, forceA * T(-1)) +
        cross(armB, derivativeForceA * T(-1));
    RigidSpringOutput<T> out{
        forceA, momentA, forceA * T(-1), momentB,
        storedEnergy};
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

} // namespace numi_matter_joint
#undef NM_SPRING_REF
