#pragma once

// Source cylindrical connector equations for quasi-static alpha=1 states.
// Reference: FEBio 2.9 FERigidCylindricalJoint::Residual/Update. This is an
// equation operator for Matter's coupled solve, not another time integrator.
// Coordinates may use any consistent length unit; the force penalty is F/L
// and the moment penalty is F*L/radian. All rotation increments are world
// spatial increments. Multipliers are explicit state, never hidden history.
// The caller owns augmentation, prescribed load curves and transactionality.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define NM_JOINT_REF thread
#else
#include <cmath>
#define NM_JOINT_REF
#endif

namespace numi_matter_joint {
#ifdef __METAL_VERSION__
using metal::sqrt; using metal::sin; using metal::cos; using metal::atan2;
using metal::isfinite;
#else
using std::sqrt; using std::sin; using std::cos; using std::atan2;
using std::isfinite;
#endif
template<class T> struct V { T x{}, y{}, z{}; };
template<class T> struct Q { T x{}, y{}, z{}, w{1}; };
template<class T> inline V<T> operator+(V<T> a,V<T> b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
template<class T> inline V<T> operator-(V<T> a,V<T> b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
template<class T> inline V<T> operator*(V<T> a,T b) { return {a.x*b,a.y*b,a.z*b}; }
template<class T> inline T dot(V<T> a,V<T> b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
template<class T> inline V<T> cross(V<T> a,V<T> b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
template<class T> inline bool finite(V<T> a) { return isfinite(a.x)&&isfinite(a.y)&&isfinite(a.z); }
template<class T> inline T norm2(Q<T> q) { return q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w; }
template<class T> inline Q<T> conjugate(Q<T> q) { return {-q.x,-q.y,-q.z,q.w}; }
template<class T> inline Q<T> multiply(Q<T> a,Q<T> b) {
    return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
            a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
            a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
            a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};
}
template<class T> inline V<T> rotate(Q<T> q,V<T> a) {
    V<T> v{q.x,q.y,q.z};
    return a+cross(v,a)*(T(2)*q.w)+cross(v,cross(v,a))*T(2);
}
template<class T> inline Q<T> exponential(V<T> v) {
    T t2=dot(v,v), t=sqrt(t2);
    T s=t<T(1e-4) ? T(.5)-t2/T(48)+t2*t2/T(3840) : sin(t/T(2))/t;
    return {v.x*s,v.y*s,v.z*s,cos(t/T(2))};
}
template<class T> struct Input {
    V<T> referenceA,referenceB,positionA,positionB,origin,axis;
    Q<T> rotationA,rotationB;
    V<T> forceMultiplier,momentMultiplier;
    T forcePenalty{},momentPenalty{},translation{},rotation{},axialForce{},axialMoment{};
    unsigned prescribedTranslation{},prescribedRotation{};
};
template<class T> struct Output {
    V<T> forceA,momentA,forceB,momentB,gap,angularGap,connectorMoment;
};
template<class T> struct Direction { V<T> positionA,rotationA,positionB,rotationB; };
// One generalized-coordinate Jacobian column, expressed as world spatial
// linear/angular velocity for each source body. The same mapping must be used
// for candidate kinematics and for mapping connector wrenches into the global
// residual; using a separate muscle route or a point-only Jacobian drops the
// connector couple and violates virtual work.
template<class T> struct MotionColumn {
    V<T> linearA,angularA,linearB,angularB;
};
template<class T> inline T generalizedForce(Output<T> wrench,MotionColumn<T> motion) {
    return dot(wrench.forceA,motion.linearA)+dot(wrench.momentA,motion.angularA)+
           dot(wrench.forceB,motion.linearB)+dot(wrench.momentB,motion.angularB);
}
template<class T> inline T generalizedTangent(Output<T> derivative,MotionColumn<T> motion) {
    return generalizedForce(derivative,motion);
}

template<class T> inline bool valid(Input<T> a) {
    T na=norm2(a.rotationA),nb=norm2(a.rotationB),ns=dot(a.axis,a.axis);
    return finite(a.referenceA)&&finite(a.referenceB)&&finite(a.positionA)&&finite(a.positionB)&&
        finite(a.origin)&&finite(a.axis)&&finite(a.forceMultiplier)&&finite(a.momentMultiplier)&&
        isfinite(na)&&isfinite(nb)&&na>T(.99999)&&na<T(1.00001)&&nb>T(.99999)&&nb<T(1.00001)&&
        ns>T(.99999)&&ns<T(1.00001)&&isfinite(a.forcePenalty)&&a.forcePenalty>T(0)&&
        isfinite(a.momentPenalty)&&a.momentPenalty>T(0)&&isfinite(a.translation)&&isfinite(a.rotation)&&
        isfinite(a.axialForce)&&isfinite(a.axialMoment)&&a.prescribedTranslation<=1&&a.prescribedRotation<=1&&
        !(a.prescribedTranslation&&a.axialForce!=T(0))&&!(a.prescribedRotation&&a.axialMoment!=T(0));
}

// Returns residual AND its exact directional derivative. Derivative sign is
// dR, not the -dR convention of the nonlinear solve's linear system. The
// rotation logarithm keeps the source quaternion branch (no sign flipping).
// Reject its 2*pi singularity; never report that configuration as zero error.
template<class T> inline bool evaluate(Input<T> a,Direction<T> d,
    NM_JOINT_REF Output<T>& result,NM_JOINT_REF Output<T>& derivative) {
    if(!valid(a)||!finite(d.positionA)||!finite(d.positionB)||!finite(d.rotationA)||!finite(d.rotationB)) return false;
    V<T> za=rotate(a.rotationA,a.origin-a.referenceA),zb=rotate(a.rotationB,a.origin-a.referenceB);
    V<T> ea=rotate(a.rotationA,a.axis),eb=rotate(a.rotationB,a.axis);
    V<T> dza=cross(d.rotationA,za),dzb=cross(d.rotationB,zb);
    V<T> dea=cross(d.rotationA,ea),deb=cross(d.rotationB,eb);
    V<T> delta=a.positionB+zb-a.positionA-za,ddelta=d.positionB+dzb-d.positionA-dza;
    V<T> gap,dc;
    if(a.prescribedTranslation) { gap=delta-ea*a.translation;dc=ddelta-dea*a.translation; }
    else { gap=delta-ea*dot(ea,delta);dc=ddelta-dea*dot(ea,delta)-ea*(dot(dea,delta)+dot(ea,ddelta)); }
    V<T> angularGap,dksi;
    if(a.prescribedRotation) {
        Q<T> r=multiply(multiply(a.rotationA,exponential(a.axis*a.rotation)),conjugate(a.rotationB));
        T n=sqrt(norm2(r));r={r.x/n,r.y/n,r.z/n,r.w/n};
        V<T> v{r.x,r.y,r.z};T s=sqrt(dot(v,v));
        if(r.w<T(0)&&s<T(1e-6)) return false;
        T angle=T(2)*atan2(s,r.w);
        angularGap=v*(s<T(1e-6)?T(2)+s*s/T(3):angle/s);
        V<T> omega=d.rotationA-rotate(r,d.rotationB);
        T t2=dot(angularGap,angularGap);
        T b=t2<T(1e-6)?T(1)/T(12)+t2/T(720):
            (T(1)-angle*r.w/(T(2)*s))/t2;
        dksi=omega-cross(angularGap,omega)*T(.5)+cross(angularGap,cross(angularGap,omega))*b;
    } else {
        angularGap=cross(ea,eb)*T(.5);
        dksi=(cross(dea,eb)+cross(ea,deb))*T(.5);
    }
    V<T> force=a.forceMultiplier+gap*a.forcePenalty+ea*a.axialForce;
    V<T> moment=a.momentMultiplier+angularGap*a.momentPenalty+ea*a.axialMoment;
    V<T> df=dc*a.forcePenalty+dea*a.axialForce,dm=dksi*a.momentPenalty+dea*a.axialMoment;
    Output<T> out{force,cross(za,force)+moment,force*T(-1),(cross(zb,force)+moment)*T(-1),gap,angularGap,moment};
    Output<T> tangent{df,cross(dza,force)+cross(za,df)+dm,df*T(-1),(cross(dzb,force)+cross(zb,df)+dm)*T(-1),dc,dksi,dm};
    if(!finite(out.forceA)||!finite(out.momentA)||!finite(out.momentB)||
       !finite(tangent.forceA)||!finite(tangent.momentA)||!finite(tangent.momentB)) return false;
    result=out;derivative=tangent;return true;
}
} // namespace numi_matter_joint
#undef NM_JOINT_REF
