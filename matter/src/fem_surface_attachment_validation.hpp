#pragma once
#include "numi/matter/numi_human_shared.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>
namespace numi::matter::detail {
// Shared by the actual Human adapter input gate and its host-side coupon.
// Dynamic face area and body-pose stride are checked again on Metal against
// accepted transaction state before any spring force is emitted.
inline bool validFEMSurfaceAttachmentSamples(
    const std::span<const NMNumiHumanFEMSurfaceAttachmentGPU> samples,
    const std::size_t nodeCount
) {
    if (samples.size() > std::numeric_limits<std::uint32_t>::max()) return false;
    std::vector<std::uint32_t> ids;
    ids.reserve(samples.size());
    for (const auto& s : samples) {
        const auto b=s.barycentric, p=s.bodyLocalPointAndArea;
        const auto m=s.stiffnessAndStrainPerPressure;
        const double sum=double(b.x)+double(b.y)+double(b.z);
        const double ka=double(m.x)*double(p.w);
        if (s.node[0]>=nodeCount || s.node[1]>=nodeCount || s.node[2]>=nodeCount ||
            s.node[0]==s.node[1] || s.node[0]==s.node[2] || s.node[1]==s.node[2] ||
            s.bodyIndex==NM_INVALID_INDEX || s.stableID==NM_INVALID_INDEX ||
            s.flags!=NM_NUMI_HUMAN_FEM_SURFACE_ATTACHMENT_ACTIVE ||
            s.reserved0!=0u || s.reserved1!=0u ||
            !std::isfinite(b.x) || !std::isfinite(b.y) || !std::isfinite(b.z) ||
            !std::isfinite(b.w) || b.x<0.0f || b.y<0.0f || b.z<0.0f ||
            b.w!=0.0f || std::abs(sum-1.0)>1.0e-5 ||
            !std::isfinite(p.x) || !std::isfinite(p.y) ||
            !std::isfinite(p.z) || !std::isfinite(p.w) || p.w<=0.0f ||
            !std::isfinite(m.x) || !std::isfinite(m.y) ||
            !std::isfinite(m.z) || !std::isfinite(m.w) ||
            m.x<=0.0f || m.y<=0.0f || m.z!=0.0f || m.w!=0.0f ||
            !std::isfinite(ka) ||
            ka<std::numeric_limits<float>::denorm_min() ||
            ka>std::numeric_limits<float>::max()) return false;
        ids.push_back(s.stableID);
    }
    std::sort(ids.begin(),ids.end());
    return std::adjacent_find(ids.begin(),ids.end())==ids.end();
}
}  // namespace numi::matter::detail
