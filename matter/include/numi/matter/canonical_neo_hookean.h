#pragma once
#include "numi/matter/paper_compensated_bytecode.h"
#ifdef __METAL_VERSION__
#define NM_NEO_THREAD thread
#else
#define NM_NEO_THREAD
#endif

// Canonical log-J energy: mu/2*(I1-3-2*log(J))+K/2*log(J)^2.
// Preserve reference-deformation low words until after the stress cancellation.
// The output remains FP32; there is no strain floor, stiffness change or history.
inline bool nmCanonicalNeoHookeanResponse(
    NM_NEO_THREAD const float* high, NM_NEO_THREAD const float* low,
    float shear, float bulk, NM_NEO_THREAD const float* direction,
    NM_NEO_THREAD float* stress, NM_NEO_THREAD float* tangent) {
    if (!mrCompensatedFinite(shear) || !mrCompensatedFinite(bulk)) return false;
    MRCompensatedScalar f[9], cofactor[9], inverseTranspose[9];
    for (unsigned i=0;i<9;++i) {
        f[i]={high[i],low ? low[i] : 0.0f};
        if (!mrCompensatedFinite(f[i].high) || !mrCompensatedFinite(f[i].low)) return false;
    }
    for (unsigned r=0;r<3;++r) for (unsigned c=0;c<3;++c) {
        const unsigned a=(r+1)%3,b=(r+2)%3,d=(c+1)%3,e=(c+2)%3;
        cofactor[3*r+c]=mrCompensatedAdd(
            mrCompensatedMultiply(f[3*a+d],f[3*b+e]),
            mrCompensatedNegate(mrCompensatedMultiply(f[3*a+e],f[3*b+d])));
    }
    MRCompensatedScalar determinant{0,0};
    for (unsigned c=0;c<3;++c)
        determinant=mrCompensatedAdd(determinant,mrCompensatedMultiply(f[c],cofactor[c]));
    if (!(determinant.high>0) || !mrCompensatedFinite(determinant.high) ||
        !mrCompensatedFinite(determinant.low)) return false;
    const auto logJ=nmCompensatedLog(determinant);
    const auto coefficient=mrCompensatedAdd(
        mrCompensatedMultiply({bulk,0},logJ),{-shear,0});
    float inverse[9];
    for (unsigned i=0;i<9;++i) {
        inverseTranspose[i]=mrCompensatedDivide(cofactor[i],determinant);
        inverse[i]=inverseTranspose[i].high+inverseTranspose[i].low;
        if (stress) {
            const auto value=mrCompensatedAdd(mrCompensatedMultiply({shear,0},f[i]),
                mrCompensatedMultiply(coefficient,inverseTranspose[i]));
            stress[i]=value.high+value.low;
            if (!mrCompensatedFinite(stress[i])) return false;
        }
    }
    if (tangent) {
        if (!direction) return false;
        float volumeDirection=0;
        for (unsigned i=0;i<9;++i) volumeDirection+=inverse[i]*direction[i];
        const float scalar=coefficient.high+coefficient.low;
        for (unsigned r=0;r<3;++r) for (unsigned c=0;c<3;++c) {
            float product=0;
            for (unsigned a=0;a<3;++a) for (unsigned b=0;b<3;++b)
                product+=inverse[3*r+a]*direction[3*b+a]*inverse[3*b+c];
            tangent[3*r+c]=shear*direction[3*r+c]+
                bulk*volumeDirection*inverse[3*r+c]-scalar*product;
            if (!mrCompensatedFinite(tangent[3*r+c])) return false;
        }
    }
    return true;
}
#undef NM_NEO_THREAD

