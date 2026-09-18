#pragma once

// Dense bilateral block of the EXISTING effective-mass constraint solve.
// Factor the actual E M_eff^-1 E^T contractions, which need not be exactly
// symmetric after FP32 response solves. No diagonal compliance is introduced.
// Scratch: n*n matrix entries, n inverse row scales, n pivot indices, n RHS.
#if defined(__METAL_VERSION__)
#define MR_NH_BILATERAL_DEVICE device
inline bool mrNHBilateralFinite(float x) { return metal::isfinite(x); }
inline float mrNHBilateralAbs(float x) { return metal::abs(x); }
inline float mrNHBilateralSqrt(float x) { return metal::sqrt(x); }
inline float mrNHBilateralFma(float a, float b, float c) { return metal::fma(a,b,c); }
#else
#include <cmath>
#define MR_NH_BILATERAL_DEVICE
inline bool mrNHBilateralFinite(float x) { return std::isfinite(x); }
inline float mrNHBilateralAbs(float x) { return std::abs(x); }
inline float mrNHBilateralSqrt(float x) { return std::sqrt(x); }
inline float mrNHBilateralFma(float a, float b, float c) { return std::fma(a,b,c); }
#endif

// D S D = L U with partial pivoting, D_ii = 1/sqrt(S_ii).
// Pivots are exactly represented integers in a float scratch buffer (n<=160).
inline bool mrNumiHumanBilateralFactor(
    MR_NH_BILATERAL_DEVICE float* matrix,
    MR_NH_BILATERAL_DEVICE float* inverseScale,
    MR_NH_BILATERAL_DEVICE float* pivots,
    unsigned n
) {
    if (n == 0u || n > 160u) return false;
    for (unsigned i=0u; i<n; ++i) {
        const float diagonal=matrix[i*n+i];
        if (!(diagonal>0.0f) || !mrNHBilateralFinite(diagonal)) return false;
        inverseScale[i]=1.0f/mrNHBilateralSqrt(diagonal);
        if (!mrNHBilateralFinite(inverseScale[i])) return false;
    }
    for (unsigned i=0u; i<n; ++i) {
        for (unsigned j=0u; j<n; ++j) {
            matrix[i*n+j]=(matrix[i*n+j]*inverseScale[i])*inverseScale[j];
            if (!mrNHBilateralFinite(matrix[i*n+j])) return false;
        }
    }
    for (unsigned k=0u; k<n; ++k) {
        unsigned pivot=k;
        float largest=mrNHBilateralAbs(matrix[k*n+k]);
        for (unsigned i=k+1u; i<n; ++i) {
            const float value=mrNHBilateralAbs(matrix[i*n+k]);
            if (value>largest) {largest=value; pivot=i;}
        }
        // Rank loss is rejected, never hidden by a compliance/floor term.
        if (!(largest>0.0f) || !mrNHBilateralFinite(largest)) return false;
        pivots[k]=static_cast<float>(pivot);
        if (pivot!=k) {
            for (unsigned j=0u; j<n; ++j) {
                const float value=matrix[k*n+j];
                matrix[k*n+j]=matrix[pivot*n+j];
                matrix[pivot*n+j]=value;
            }
        }
        for (unsigned i=k+1u; i<n; ++i) {
            matrix[i*n+k]/=matrix[k*n+k];
            if (!mrNHBilateralFinite(matrix[i*n+k])) return false;
            for (unsigned j=k+1u; j<n; ++j) {
                matrix[i*n+j]=mrNHBilateralFma(-matrix[i*n+k],matrix[k*n+j],matrix[i*n+j]);
                if (!mrNHBilateralFinite(matrix[i*n+j])) return false;
            }
        }
    }
    return true;
}

// rhs becomes the correction in ORIGINAL multiplier units: delta = D U^-1 L^-1 P D rhs.
inline bool mrNumiHumanBilateralSolve(
    MR_NH_BILATERAL_DEVICE const float* factor,
    MR_NH_BILATERAL_DEVICE const float* inverseScale,
    MR_NH_BILATERAL_DEVICE const float* pivots,
    MR_NH_BILATERAL_DEVICE float* rhs,
    unsigned n
) {
    if (n==0u || n>160u) return false;
    for (unsigned i=0u; i<n; ++i) {
        if (!(inverseScale[i]>0.0f) || !mrNHBilateralFinite(inverseScale[i])) return false;
        rhs[i]*=inverseScale[i];
        if (!mrNHBilateralFinite(rhs[i])) return false;
    }
    // Apply the complete row permutation before substitution. Swapping and
    // eliminating incrementally is incorrect when L's prior columns pivot.
    for (unsigned k=0u; k<n; ++k) {
        if (!mrNHBilateralFinite(pivots[k]) || pivots[k]<static_cast<float>(k) ||
            pivots[k]>=static_cast<float>(n)) return false;
        const unsigned pivot=static_cast<unsigned>(pivots[k]);
        if (pivots[k]!=static_cast<float>(pivot)) return false;
        const float value=rhs[k];rhs[k]=rhs[pivot];rhs[pivot]=value;
    }
    for (unsigned i=0u; i<n; ++i) {
        for (unsigned j=0u; j<i; ++j) rhs[i]=mrNHBilateralFma(-factor[i*n+j],rhs[j],rhs[i]);
    }
    for (unsigned reverse=0u; reverse<n; ++reverse) {
        const unsigned i=n-1u-reverse;
        for (unsigned j=i+1u; j<n; ++j) rhs[i]=mrNHBilateralFma(-factor[i*n+j],rhs[j],rhs[i]);
        const float pivot=factor[i*n+i];
        if (pivot==0.0f || !mrNHBilateralFinite(pivot)) return false;
        rhs[i]/=pivot;
        if (!mrNHBilateralFinite(rhs[i])) return false;
    }
    for (unsigned i=0u; i<n; ++i) {
        rhs[i]*=inverseScale[i];
        if (!mrNHBilateralFinite(rhs[i])) return false;
    }
    return true;
}
#undef MR_NH_BILATERAL_DEVICE
