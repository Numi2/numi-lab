#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace numi::matter::detail {
struct FEMPureCGLSBestOfEligibility {
    bool selectedNewton=false, accepted=false, healthy=false, originalCertificate=false;
    float baseResidual=0.0f, residualTolerance=0.0f, acceptedResidual=0.0f, alpha=0.0f;
};
[[nodiscard]] inline bool femPureCGLSBestOfEligible(
        const FEMPureCGLSBestOfEligibility& v) noexcept {
    return v.selectedNewton && v.accepted && v.healthy && v.originalCertificate &&
        std::isfinite(v.baseResidual) && std::isfinite(v.residualTolerance) &&
        v.residualTolerance>=0.0f && std::isfinite(v.acceptedResidual) &&
        v.acceptedResidual>=0.0f && v.acceptedResidual<v.baseResidual &&
        v.baseResidual>v.residualTolerance && std::isfinite(v.alpha) && v.alpha>0.0f;
}
struct FEMPureCGLSBestOfChoice {
    bool savedNewtonCandidate=false, cglsAccepted=false, cglsHealthy=false, cglsFinite=false;
    float newtonResidual=0.0f, cglsResidual=0.0f;
};
[[nodiscard]] inline bool femPureCGLSBestOfSelect(
        const FEMPureCGLSBestOfChoice& v) noexcept {
    return v.savedNewtonCandidate && v.cglsAccepted && v.cglsHealthy && v.cglsFinite &&
        std::isfinite(v.newtonResidual) && std::isfinite(v.cglsResidual) &&
        v.newtonResidual>=0.0f && v.cglsResidual>=0.0f &&
        v.cglsResidual<v.newtonResidual;
}
template <std::size_t N>
inline bool femPureCGLSBestOfResolveBytes(
        const FEMPureCGLSBestOfChoice& v,
        const unsigned char (&newtonSnapshot)[N],
        const unsigned char (&currentCGLSState)[N],
        unsigned char (&selectedState)[N]) noexcept {
    const bool chooseCGLS=femPureCGLSBestOfSelect(v);
    std::memcpy(selectedState,chooseCGLS?currentCGLSState:newtonSnapshot,N);
    return chooseCGLS;
}
} // namespace numi::matter::detail
