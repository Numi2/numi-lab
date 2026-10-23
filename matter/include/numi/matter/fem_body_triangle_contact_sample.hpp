#pragma once

#include "numi/matter/numi_human_shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace numi::matter::fem_body_triangle_contact {

// Compute a unit normal after scaling edges by their longest length. The
// predicate is relative to triangle scale; absolute cross-product thresholds
// incorrectly discard small but non-collinear registered faces.
inline bool scaledTriangleUnitNormal(
    nm_float4 p0, nm_float4 p1, nm_float4 p2,
    std::array<double,3>& unitNormal
) noexcept {
    const double e10=double(p1.x)-p0.x,e11=double(p1.y)-p0.y,e12=double(p1.z)-p0.z;
    const double e20=double(p2.x)-p0.x,e21=double(p2.y)-p0.y,e22=double(p2.z)-p0.z;
    const double e30=e20-e10,e31=e21-e11,e32=e22-e12;
    const double scale=std::max({std::abs(e10),std::abs(e11),std::abs(e12),
        std::abs(e20),std::abs(e21),std::abs(e22),
        std::abs(e30),std::abs(e31),std::abs(e32)});
    if (!std::isfinite(scale) || !(scale>0.0)) return false;
    const double ax=e10/scale,ay=e11/scale,az=e12/scale;
    const double bx=e20/scale,by=e21/scale,bz=e22/scale;
    const double cx=ay*bz-az*by,cy=az*bx-ax*bz,cz=ax*by-ay*bx;
    const double areaLength=std::sqrt(cx*cx+cy*cy+cz*cz);
    if (!std::isfinite(areaLength) || !(areaLength>0.0)) return false;
    unitNormal={cx/areaLength,cy/areaLength,cz/areaLength};
    return std::all_of(unitNormal.begin(),unitNormal.end(),
        [](double value) { return std::isfinite(value); });
}

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
    std::array<double,3> unitNormal{};
    const bool nondegenerate=scaledTriangleUnitNormal(p0,p1,p2,unitNormal);
    const float stiffnessArea=p0.w*p1.w;
    return nondegenerate && std::isfinite(stiffnessArea) && stiffnessArea>0.0f;
}

} // namespace numi::matter::fem_body_triangle_contact
