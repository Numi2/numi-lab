#pragma once
#include "metalrobo/compensated_translation_gpu.h"

// Derived geometry only. The source quaternion, scalar joint coordinates and
// trigonometric values remain FP32. Keep arithmetic residuals while composing
// translations and rotating offsets; project only an observable or Jacobian.
inline MRCompensatedScalar mrCompensatedNegate(MRCompensatedScalar a) {
    return {-a.high,-a.low};
}
inline MRCompensatedScalar mrCompensatedMultiply(MRCompensatedScalar a, MRCompensatedScalar b) {
    return mrCompensatedAdd(mrCompensatedProduct(a.high,b.high),
        mrCompensatedAdd(mrCompensatedProduct(a.high,b.low),
            mrCompensatedAdd(mrCompensatedProduct(a.low,b.high),mrCompensatedProduct(a.low,b.low))));
}
inline MRCompensatedScalar mrCompensatedDivide(MRCompensatedScalar a, MRCompensatedScalar b) {
    const float first=a.high/b.high;
    const auto remainder=mrCompensatedAdd(a,mrCompensatedNegate(mrCompensatedMultiply(b,{first,0.0f})));
    return mrCompensatedAdd({first,0.0f},{(remainder.high+remainder.low)/b.high,0.0f});
}
inline MRCompensatedScalar mrCompensatedSqrt(MRCompensatedScalar a) {
#ifdef __METAL_VERSION__
    const float first=sqrt(a.high);
#else
    const float first=std::sqrt(a.high);
#endif
    const auto remainder=mrCompensatedAdd(a,mrCompensatedNegate(mrCompensatedProduct(first,first)));
    return mrCompensatedAdd({first,0.0f},{(remainder.high+remainder.low)/(2.0f*first),0.0f});
}
inline MRCompensatedPositionGPU mrCompensatedVector(mr_float4 value) {
    value.w=0.0f; return {value,{0.0f,0.0f,0.0f,0.0f}};
}
inline MRCompensatedPositionGPU mrCompensatedVectorPack(MRCompensatedScalar x,MRCompensatedScalar y,MRCompensatedScalar z) {
    return {{x.high,y.high,z.high,0.0f},{x.low,y.low,z.low,0.0f}};
}
inline MRCompensatedPositionGPU mrCompensatedVectorAdd(MRCompensatedPositionGPU a,MRCompensatedPositionGPU b) {
    return mrCompensatedVectorPack(mrCompensatedAdd({a.high.x,a.low.x},{b.high.x,b.low.x}),
        mrCompensatedAdd({a.high.y,a.low.y},{b.high.y,b.low.y}),
        mrCompensatedAdd({a.high.z,a.low.z},{b.high.z,b.low.z}));
}
inline MRCompensatedPositionGPU mrCompensatedVectorNegate(MRCompensatedPositionGPU a) {
    return {{-a.high.x,-a.high.y,-a.high.z,0.0f},{-a.low.x,-a.low.y,-a.low.z,0.0f}};
}
inline MRCompensatedPositionGPU mrCompensatedVectorScale(MRCompensatedPositionGPU a,MRCompensatedScalar scale) {
    return mrCompensatedVectorPack(mrCompensatedMultiply({a.high.x,a.low.x},scale),
        mrCompensatedMultiply({a.high.y,a.low.y},scale),mrCompensatedMultiply({a.high.z,a.low.z},scale));
}
inline MRCompensatedPositionGPU mrCompensatedVectorCross(MRCompensatedPositionGPU a,MRCompensatedPositionGPU b) {
    const MRCompensatedScalar ax{a.high.x,a.low.x},ay{a.high.y,a.low.y},az{a.high.z,a.low.z};
    const MRCompensatedScalar bx{b.high.x,b.low.x},by{b.high.y,b.low.y},bz{b.high.z,b.low.z};
    return mrCompensatedVectorPack(mrCompensatedAdd(mrCompensatedMultiply(ay,bz),mrCompensatedNegate(mrCompensatedMultiply(az,by))),
        mrCompensatedAdd(mrCompensatedMultiply(az,bx),mrCompensatedNegate(mrCompensatedMultiply(ax,bz))),
        mrCompensatedAdd(mrCompensatedMultiply(ax,by),mrCompensatedNegate(mrCompensatedMultiply(ay,bx))));
}
inline MRCompensatedScalar mrCompensatedVectorDot(MRCompensatedPositionGPU a,MRCompensatedPositionGPU b) {
    return mrCompensatedAdd(mrCompensatedMultiply({a.high.x,a.low.x},{b.high.x,b.low.x}),
        mrCompensatedAdd(mrCompensatedMultiply({a.high.y,a.low.y},{b.high.y,b.low.y}),
            mrCompensatedMultiply({a.high.z,a.low.z},{b.high.z,b.low.z})));
}
inline MRCompensatedPositionGPU mrCompensatedQuaternionRotatePair(mr_float4 q,MRCompensatedPositionGPU vector) {
    const auto axis=mrCompensatedVector(q);
    const auto twiceCross=mrCompensatedVectorScale(mrCompensatedVectorCross(axis,vector),{2.0f,0.0f});
    return mrCompensatedVectorAdd(vector,mrCompensatedVectorAdd(
        mrCompensatedVectorScale(twiceCross,{q.w,0.0f}),mrCompensatedVectorCross(axis,twiceCross)));
}
inline MRCompensatedPositionGPU mrCompensatedQuaternionRotate(mr_float4 q,mr_float4 vector) {
    return mrCompensatedQuaternionRotatePair(q,mrCompensatedVector(vector));
}
inline mr_float4 mrCompensatedGeometryQuaternionMultiply(mr_float4 a,mr_float4 b) {
    // One shared FP32 quaternion expression in both provider and contact owner.
    return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
        a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
        a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
        a.w*b.w-((a.x*b.x+a.y*b.y)+a.z*b.z)};
}
inline MRCompensatedPositionGPU mrCompensatedSupportOffset(
    mr_float4 bodyOrientation,mr_float4 localPoint,mr_float4 supportOrientation,
    mr_float4 supportRadii,mr_float4 normalAndRadius,mr_u32 kind) {
    auto result=mrCompensatedQuaternionRotate(bodyOrientation,localPoint);
    if (kind != 2u) return mrCompensatedVectorAdd(result,mrCompensatedVectorNegate(
        mrCompensatedVectorScale(mrCompensatedVector(normalAndRadius),{normalAndRadius.w,0.0f})));
    const auto shape=mrCompensatedGeometryQuaternionMultiply(bodyOrientation,supportOrientation);
    const mr_float4 conjugate{-shape.x,-shape.y,-shape.z,shape.w};
    const auto direction=mrCompensatedQuaternionRotate(conjugate,normalAndRadius);
    const auto scaled=mrCompensatedVectorPack(
        mrCompensatedMultiply({direction.high.x,direction.low.x},{supportRadii.x,0.0f}),
        mrCompensatedMultiply({direction.high.y,direction.low.y},{supportRadii.y,0.0f}),
        mrCompensatedMultiply({direction.high.z,direction.low.z},{supportRadii.z,0.0f}));
    const auto magnitude=mrCompensatedSqrt(mrCompensatedVectorDot(scaled,scaled));
    const auto surface=mrCompensatedVectorPack(
        mrCompensatedDivide(mrCompensatedMultiply({scaled.high.x,scaled.low.x},{supportRadii.x,0.0f}),magnitude),
        mrCompensatedDivide(mrCompensatedMultiply({scaled.high.y,scaled.low.y},{supportRadii.y,0.0f}),magnitude),
        mrCompensatedDivide(mrCompensatedMultiply({scaled.high.z,scaled.low.z},{supportRadii.z,0.0f}),magnitude));
    return mrCompensatedVectorAdd(result,mrCompensatedVectorNegate(mrCompensatedQuaternionRotatePair(shape,surface)));
}
inline MRCompensatedPositionGPU mrCompensatedTranslationPositionPair(
    MRCompensatedRootTranslationGPU root,MRCompensatedPositionGPU local) {
    return mrCompensatedVectorAdd(mrCompensatedTranslationPosition(root,local.high),mrCompensatedVector(local.low));
}
