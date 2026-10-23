#pragma once

#include "numi/matter/numi_human_shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace numi::matter::fem_body_triangle_contact {

inline NMNumiHumanFEMBodyContactSampleGPU makeTriangleContactSample(
    std::uint32_t slaveNode,
    std::uint32_t bodyIndex,
    nm_float4 v0,
    nm_float4 v1,
    nm_float4 v2,
    float slaveTributaryAreaM2,
    float pressureOverclosureStiffnessPaPerMeter,
    float normalStrainPerPressurePerPa
) {
    NMNumiHumanFEMBodyContactSampleGPU sample{};
    sample.slaveNode = slaveNode;
    sample.bodyIndex = bodyIndex;
    sample.flags = NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
        NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
    sample.bodyLocalPointAndArea = {
        v0.x,v0.y,v0.z,slaveTributaryAreaM2};
    sample.bodyLocalNormalAndReferenceSeparation = {
        v1.x,v1.y,v1.z,pressureOverclosureStiffnessPaPerMeter};
    sample.stiffnessAndNormalStrainPerPressure = {
        v2.x,v2.y,v2.z,normalStrainPerPressurePerPa};
    return sample;
}

inline std::array<nm_float4,3> decodeTriangleContactSample(
    const NMNumiHumanFEMBodyContactSampleGPU& sample
) {
    return {{
        {sample.bodyLocalPointAndArea.x,sample.bodyLocalPointAndArea.y,
         sample.bodyLocalPointAndArea.z,0.0f},
        {sample.bodyLocalNormalAndReferenceSeparation.x,
         sample.bodyLocalNormalAndReferenceSeparation.y,
         sample.bodyLocalNormalAndReferenceSeparation.z,0.0f},
        {sample.stiffnessAndNormalStrainPerPressure.x,
         sample.stiffnessAndNormalStrainPerPressure.y,
         sample.stiffnessAndNormalStrainPerPressure.z,0.0f},
    }};
}

inline bool validTriangleContactSample(
    const NMNumiHumanFEMBodyContactSampleGPU& sample
) {
    const std::uint32_t requiredFlags =
        NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
        NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
    if (sample.flags != requiredFlags || sample.reserved0 != 0u) return false;
    const auto p0 = sample.bodyLocalPointAndArea;
    const auto p1 = sample.bodyLocalNormalAndReferenceSeparation;
    const auto p2 = sample.stiffnessAndNormalStrainPerPressure;
    const std::array<float,12> values{{
        p0.x,p0.y,p0.z,p0.w,p1.x,p1.y,p1.z,p1.w,p2.x,p2.y,p2.z,p2.w}};
    if (!std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); }) ||
        p0.w <= 0.0f || p1.w <= 0.0f || p2.w <= 0.0f) return false;
    const double e1x=double(p1.x)-p0.x;
    const double e1y=double(p1.y)-p0.y;
    const double e1z=double(p1.z)-p0.z;
    const double e2x=double(p2.x)-p0.x;
    const double e2y=double(p2.y)-p0.y;
    const double e2z=double(p2.z)-p0.z;
    const double cx=e1y*e2z-e1z*e2y;
    const double cy=e1z*e2x-e1x*e2z;
    const double cz=e1x*e2y-e1y*e2x;
    const double areaVectorSquared=cx*cx+cy*cy+cz*cz;
    const float stiffnessArea=p0.w*p1.w;
    return std::isfinite(cx) && std::isfinite(cy) && std::isfinite(cz) &&
        std::isfinite(areaVectorSquared) && areaVectorSquared>1.0e-20 &&
        std::isfinite(stiffnessArea) && stiffnessArea>0.0f;
}

} // namespace numi::matter::fem_body_triangle_contact
