#pragma once

#include "numi/matter/shared.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace numi::matter::detail {

// Admission only; no physical state update. Metal forms FP32 Ds and multiplies
// the serialized FP32 inverse. det(Ds)/det(Dm) is not that executable J.
// Contraction and reduction order can differ between host and Metal, so the
// nominal host float result is diagnostic, never an equality certificate.
struct FEMReferenceDeterminantInterval {
    double center = 0.0;
    double radius = std::numeric_limits<double>::infinity();
    float nominal = std::numeric_limits<float>::quiet_NaN();
    bool finite = false;
    bool noIntermediateOverflow = false;

    [[nodiscard]] bool strictlyAdmitted(nm_float4 limits) const noexcept {
        return finite && noIntermediateOverflow && center - radius >= limits.x &&
            center + radius <= limits.y;
    }
    // This excludes provably outside states, but does not certify the strict
    // GPU domain for rounding-ambiguous boundary snapshots. It preserves an
    // accepted Metal state when host contraction/reduction order differs.
    [[nodiscard]] bool possiblyAdmissible(nm_float4 limits) const noexcept {
        return finite && noIntermediateOverflow && center + radius >= limits.x && center - radius <= limits.y;
    }
};

[[nodiscard]] inline FEMReferenceDeterminantInterval femReferenceDeterminantInterval(
    const NMTetrahedronGPU& t, const NMFEMNodeStateGPU* nodes, std::size_t base=0u
) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    FEMReferenceDeterminantInterval out;
    constexpr double u = double(std::numeric_limits<float>::epsilon()) * 0.5;
    // Min-normal, rather than min-subnormal, also covers denormal flush-to-zero.
    constexpr double eta = std::numeric_limits<float>::min();
    constexpr double gamma6 = (6.0*u)/(1.0-6.0*u);
    constexpr double gamma12 = (12.0*u)/(1.0-12.0*u);
    constexpr double maximum = std::numeric_limits<float>::max();
    const std::array<nm_u32,4> ids{t.nodes.x,t.nodes.y,t.nodes.z,t.nodes.w};
    const std::array<float,9> inverse{t.inverseRestRow0.x,t.inverseRestRow0.y,t.inverseRestRow0.z,
        t.inverseRestRow1.x,t.inverseRestRow1.y,t.inverseRestRow1.z,t.inverseRestRow2.x,t.inverseRestRow2.y,t.inverseRestRow2.z};
    std::array<double,9> ds{},f{},error{};
    std::array<float,9> ds32{},f32{};
    const auto normalOrZero = [](float v) {
        return std::isfinite(v) && (v == 0.0f || std::abs(v) >= std::numeric_limits<float>::min());
    };
    const auto p0=nodes[base+ids[0]].positionAndMass;
    // Input FTZ can amplify a subnormal inverse or coordinate through a
    // normal-sized product. This bounded admission mode rejects such inputs;
    // generated intermediate underflow is covered by eta below.
    bool safe=normalOrZero(p0.x)&&normalOrZero(p0.y)&&normalOrZero(p0.z);
    for (unsigned c=0;c<3;++c) {
        const auto p=nodes[base+ids[c+1]].positionAndMass;
        safe=safe&&normalOrZero(p.x)&&normalOrZero(p.y)&&normalOrZero(p.z);
        ds[c]=double(p.x)-p0.x;ds[3+c]=double(p.y)-p0.y;ds[6+c]=double(p.z)-p0.z;
        ds32[c]=p.x-p0.x;ds32[3+c]=p.y-p0.y;ds32[6+c]=p.z-p0.z;
    }
    for (double value:ds) safe=safe&&std::isfinite(value)&&std::abs(value)<=maximum;
    for (float value:inverse) safe=safe&&normalOrZero(value);
    for (unsigned r=0;r<3;++r) for (unsigned c=0;c<3;++c) {
        double magnitude=0.0, inverseMagnitude=0.0;
        for (unsigned k=0;k<3;++k) {
            const double product=ds[3*r+k]*double(inverse[3*k+c]);
            f[3*r+c]+=product;magnitude+=std::abs(product);inverseMagnitude+=std::abs(double(inverse[3*k+c]));
        }
        // Each term has one coordinate subtraction and one product, followed
        // by at most two additions. gamma6 bounds all fused/nonfused orders
        // and FP64 center evaluation; the eta terms cover underflow.
        error[3*r+c]=gamma6*magnitude+(1.0+gamma6)*eta*(inverseMagnitude+8.0);
        safe=safe&&magnitude+error[3*r+c]<=maximum;
        const float a=ds32[3*r]*inverse[c],b=ds32[3*r+1]*inverse[3+c],d=ds32[3*r+2]*inverse[6+c];
        f32[3*r+c]=(a+b)+d;
    }
    out.center=f[0]*(f[4]*f[8]-f[5]*f[7])-f[1]*(f[3]*f[8]-f[5]*f[6])+f[2]*(f[3]*f[7]-f[4]*f[6]);
    out.nominal=f32[0]*(f32[4]*f32[8]-f32[5]*f32[7])-f32[1]*(f32[3]*f32[8]-f32[5]*f32[6])+f32[2]*(f32[3]*f32[7]-f32[4]*f32[6]);
    constexpr std::array<std::array<unsigned,3>,6> terms{{{{0,4,8}},{{0,5,7}},{{1,3,8}},{{1,5,6}},{{2,3,7}},{{2,4,6}}}};
    double perturbation=0.0, expandedMagnitude=0.0;
    for (const auto& indices:terms) {
        const double a=std::abs(f[indices[0]]),b=std::abs(f[indices[1]]),c=std::abs(f[indices[2]]);
        const double da=error[indices[0]],db=error[indices[1]],dc=error[indices[2]];
        // Expanded multilinear perturbation avoids cancellation in
        // (a+da)(b+db)(c+dc)-abc when errors are small.
        perturbation+=da*b*c+a*db*c+a*b*dc+da*db*c+da*b*dc+a*db*dc+da*db*dc;
        expandedMagnitude+=(a+da)*(b+db)*(c+dc);
    }
    double largest=0.0;for (unsigned i=0;i<9;++i) largest=std::max(largest,std::abs(f[i])+error[i]);
    // A 3x3 determinant is six signed triple products. gamma12 exceeds the
    // longest product/cofactor/reduction rounding path, including FMA forms.
    // All terms are absolute, so the bound also accounts for cancellation.
    out.radius=perturbation+gamma12*expandedMagnitude+32.0*eta*(1.0+largest+largest*largest);
    out.noIntermediateOverflow=safe&&(1.0+gamma12)*expandedMagnitude<=maximum&&2.0*(1.0+gamma12)*largest*largest<=maximum;
    out.finite=std::isfinite(out.center)&&std::isfinite(out.radius)&&out.radius>=0.0;
    return out;
}
} // namespace numi::matter::detail
