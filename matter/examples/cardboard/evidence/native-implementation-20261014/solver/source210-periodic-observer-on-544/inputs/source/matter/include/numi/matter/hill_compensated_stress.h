#pragma once

// Compensated FP32 evaluation of the authored orthotropic Green-strain
// elastic trial stress and its exact first derivative with respect to F.
// The constitutive equations and 20-float material-table order are unchanged.

#include "numi/matter/hill_scalar_initializer.h"

#if defined(__METAL_VERSION__)
#define NM_HILL_COMP_INLINE inline
#define NM_HILL_COMP_THREAD thread
#define NM_HILL_COMP_PARAMETER device const float*
#define NM_HILL_COMP_FMA fma
#define NM_HILL_COMP_ISFINITE isfinite
#else
#include <math.h>
#define NM_HILL_COMP_INLINE inline
#define NM_HILL_COMP_THREAD
#define NM_HILL_COMP_PARAMETER const float*
#define NM_HILL_COMP_FMA fmaf
#define NM_HILL_COMP_ISFINITE isfinite
#endif

struct NMHillCompensatedFloatPair {
    float high;
    float low;
};

NM_HILL_COMP_INLINE bool nmHillCompensatedFinite(float value) {
#if defined(__METAL_VERSION__)
    return NM_HILL_COMP_ISFINITE(value);
#else
    return NM_HILL_COMP_ISFINITE(value) != 0;
#endif
}

NM_HILL_COMP_INLINE NMHillCompensatedFloatPair nmHillCompensatedTwoSum(
    float a,float b) {
    const float sum=a+b;
    const float bVirtual=sum-a;
    const float error=(a-(sum-bVirtual))+(b-bVirtual);
    return {sum,error};
}

NM_HILL_COMP_INLINE NMHillCompensatedFloatPair nmHillCompensatedNormalize(
    float high,float low) {
    const float sum=high+low;
    return {sum,low-(sum-high)};
}

NM_HILL_COMP_INLINE NMHillCompensatedFloatPair nmHillCompensatedAdd(
    NMHillCompensatedFloatPair a,NMHillCompensatedFloatPair b) {
    const NMHillCompensatedFloatPair high=nmHillCompensatedTwoSum(a.high,b.high);
    const NMHillCompensatedFloatPair low=nmHillCompensatedTwoSum(a.low,b.low);
    const NMHillCompensatedFloatPair middle=nmHillCompensatedTwoSum(high.low,low.high);
    const NMHillCompensatedFloatPair leading=nmHillCompensatedTwoSum(high.high,middle.high);
    const float tail=(leading.low+middle.low)+low.low;
    return nmHillCompensatedNormalize(leading.high,tail);
}

NM_HILL_COMP_INLINE NMHillCompensatedFloatPair nmHillCompensatedProduct(
    float a,float b) {
    const float product=a*b;
    return {product,NM_HILL_COMP_FMA(a,b,-product)};
}

NM_HILL_COMP_INLINE NMHillCompensatedFloatPair nmHillCompensatedMultiply(
    NMHillCompensatedFloatPair value,float scale) {
    return nmHillCompensatedAdd(nmHillCompensatedProduct(value.high,scale),
        nmHillCompensatedProduct(value.low,scale));
}

NM_HILL_COMP_INLINE NMHillCompensatedFloatPair nmHillCompensatedDivide(
    NMHillCompensatedFloatPair value,float divisor) {
    const float quotient=value.high/divisor;
    const NMHillCompensatedFloatPair reconstructed=
        nmHillCompensatedProduct(quotient,divisor);
    const NMHillCompensatedFloatPair remainder=nmHillCompensatedAdd(value,
        {-reconstructed.high,-reconstructed.low});
    const float correction=(remainder.high+remainder.low)/divisor;
    return nmHillCompensatedNormalize(quotient,correction);
}

NM_HILL_COMP_INLINE NMHillCompensatedFloatPair nmHillCompensatedDot3(
    float a0,float b0,float a1,float b1,float a2,float b2) {
    return nmHillCompensatedAdd(
        nmHillCompensatedAdd(nmHillCompensatedProduct(a0,b0),
            nmHillCompensatedProduct(a1,b1)),
        nmHillCompensatedProduct(a2,b2));
}

NM_HILL_COMP_INLINE int nmHillCompensatedStressInputsValid(
    NM_HILL_COMP_PARAMETER parameters,
    NM_HILL_COMP_THREAD const float* F,
    NM_HILL_COMP_THREAD const float* accepted) {
    for (unsigned i=0;i<20;++i)
        if (!nmHillCompensatedFinite(parameters[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    for (unsigned i=0;i<9;++i)
        if (!nmHillCompensatedFinite(F[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    for (unsigned i=0;i<13;++i)
        if (!nmHillCompensatedFinite(accepted[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    if (!(parameters[10]>0.0f)) return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    return NM_HILL_SCALAR_INITIALIZER_SUCCESS;
}

// Outputs normalized S11,S22,S33,S23,S13,S12. The calculation uses two-float
// error-free sums/products for the Green strain and C:E numerator, followed by
// one compensated quotient. Output bytes are preserved on any invalid input.
NM_HILL_COMP_INLINE int nmHillCompensatedTrialStress(
    NM_HILL_COMP_PARAMETER parameters,
    NM_HILL_COMP_THREAD const float* F,
    NM_HILL_COMP_THREAD const float* accepted,
    NM_HILL_COMP_THREAD float* stress) {
    const int validation=nmHillCompensatedStressInputsValid(parameters,F,accepted);
    if (validation!=NM_HILL_SCALAR_INITIALIZER_SUCCESS) return validation;

    const NMHillCompensatedFloatPair c11=nmHillCompensatedDot3(
        F[0],F[0],F[3],F[3],F[6],F[6]);
    const NMHillCompensatedFloatPair c22=nmHillCompensatedDot3(
        F[1],F[1],F[4],F[4],F[7],F[7]);
    const NMHillCompensatedFloatPair c33=nmHillCompensatedDot3(
        F[2],F[2],F[5],F[5],F[8],F[8]);
    const NMHillCompensatedFloatPair e11=nmHillCompensatedAdd(
        nmHillCompensatedMultiply(nmHillCompensatedAdd(c11,{-1.0f,0.0f}),0.5f),
        {-accepted[0],0.0f});
    const NMHillCompensatedFloatPair e22=nmHillCompensatedAdd(
        nmHillCompensatedMultiply(nmHillCompensatedAdd(c22,{-1.0f,0.0f}),0.5f),
        {-accepted[1],0.0f});
    const NMHillCompensatedFloatPair e33=nmHillCompensatedAdd(
        nmHillCompensatedMultiply(nmHillCompensatedAdd(c33,{-1.0f,0.0f}),0.5f),
        {-accepted[2],0.0f});
    const NMHillCompensatedFloatPair gamma23=nmHillCompensatedAdd(
        nmHillCompensatedDot3(F[1],F[2],F[4],F[5],F[7],F[8]),
        {-2.0f*accepted[3],0.0f});
    const NMHillCompensatedFloatPair gamma13=nmHillCompensatedAdd(
        nmHillCompensatedDot3(F[0],F[2],F[3],F[5],F[6],F[8]),
        {-2.0f*accepted[4],0.0f});
    const NMHillCompensatedFloatPair gamma12=nmHillCompensatedAdd(
        nmHillCompensatedDot3(F[0],F[1],F[3],F[4],F[6],F[7]),
        {-2.0f*accepted[5],0.0f});
    const NMHillCompensatedFloatPair numerator[6]={
        nmHillCompensatedAdd(nmHillCompensatedAdd(
            nmHillCompensatedMultiply(e11,parameters[1]),
            nmHillCompensatedMultiply(e22,parameters[4])),
            nmHillCompensatedMultiply(e33,parameters[5])),
        nmHillCompensatedAdd(nmHillCompensatedAdd(
            nmHillCompensatedMultiply(e11,parameters[4]),
            nmHillCompensatedMultiply(e22,parameters[2])),
            nmHillCompensatedMultiply(e33,parameters[6])),
        nmHillCompensatedAdd(nmHillCompensatedAdd(
            nmHillCompensatedMultiply(e11,parameters[5]),
            nmHillCompensatedMultiply(e22,parameters[6])),
            nmHillCompensatedMultiply(e33,parameters[3])),
        nmHillCompensatedMultiply(gamma23,parameters[9]),
        nmHillCompensatedMultiply(gamma13,parameters[8]),
        nmHillCompensatedMultiply(gamma12,parameters[7])};
    float result[6];
    for (unsigned i=0;i<6;++i) {
        result[i]=nmHillCompensatedDivide(numerator[i],parameters[10]).high;
        if (!nmHillCompensatedFinite(result[i]))
            return NM_HILL_SCALAR_INITIALIZER_NONFINITE;
    }
    for (unsigned i=0;i<6;++i) stress[i]=result[i];
    return NM_HILL_SCALAR_INITIALIZER_SUCCESS;
}

// Analytical Jacobian layout is component-major: tangent[component*9+F_index]
// equals d(normalized trial stress[component])/d(F[F_index]).
NM_HILL_COMP_INLINE int nmHillCompensatedTrialStressAndTangent(
    NM_HILL_COMP_PARAMETER parameters,
    NM_HILL_COMP_THREAD const float* F,
    NM_HILL_COMP_THREAD const float* accepted,
    NM_HILL_COMP_THREAD float* stress,
    NM_HILL_COMP_THREAD float* tangent) {
    float stressResult[6];
    float tangentResult[54];
    const int status=nmHillCompensatedTrialStress(parameters,F,accepted,stressResult);
    if (status!=NM_HILL_SCALAR_INITIALIZER_SUCCESS) return status;
    for (unsigned i=0;i<54;++i) tangentResult[i]=0.0f;
    for (unsigned r=0;r<3;++r) {
        const unsigned i0=3*r;
        const unsigned i1=i0+1;
        const unsigned i2=i0+2;
        tangentResult[0*9+i0]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[1],F[i0]),parameters[10]).high;
        tangentResult[0*9+i1]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[4],F[i1]),parameters[10]).high;
        tangentResult[0*9+i2]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[5],F[i2]),parameters[10]).high;
        tangentResult[1*9+i0]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[4],F[i0]),parameters[10]).high;
        tangentResult[1*9+i1]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[2],F[i1]),parameters[10]).high;
        tangentResult[1*9+i2]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[6],F[i2]),parameters[10]).high;
        tangentResult[2*9+i0]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[5],F[i0]),parameters[10]).high;
        tangentResult[2*9+i1]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[6],F[i1]),parameters[10]).high;
        tangentResult[2*9+i2]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[3],F[i2]),parameters[10]).high;
        tangentResult[3*9+i1]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[9],F[i2]),parameters[10]).high;
        tangentResult[3*9+i2]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[9],F[i1]),parameters[10]).high;
        tangentResult[4*9+i0]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[8],F[i2]),parameters[10]).high;
        tangentResult[4*9+i2]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[8],F[i0]),parameters[10]).high;
        tangentResult[5*9+i0]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[7],F[i1]),parameters[10]).high;
        tangentResult[5*9+i1]=nmHillCompensatedDivide(
            nmHillCompensatedProduct(parameters[7],F[i0]),parameters[10]).high;
    }
    for (unsigned i=0;i<54;++i)
        if (!nmHillCompensatedFinite(tangentResult[i]))
            return NM_HILL_SCALAR_INITIALIZER_NONFINITE;
    for (unsigned i=0;i<6;++i) stress[i]=stressResult[i];
    for (unsigned i=0;i<54;++i) tangent[i]=tangentResult[i];
    return NM_HILL_SCALAR_INITIALIZER_SUCCESS;
}

// Directional derivative dS/dF[dF], evaluated directly from the same
// analytical Green-strain derivative without multiplying a rounded tangent.
NM_HILL_COMP_INLINE int nmHillCompensatedTrialStressDirectionalDerivative(
    NM_HILL_COMP_PARAMETER parameters,
    NM_HILL_COMP_THREAD const float* F,
    NM_HILL_COMP_THREAD const float* dF,
    NM_HILL_COMP_THREAD float* dStress) {
    for (unsigned i=0;i<20;++i)
        if (!nmHillCompensatedFinite(parameters[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    for (unsigned i=0;i<9;++i)
        if (!nmHillCompensatedFinite(F[i])||!nmHillCompensatedFinite(dF[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    if (!(parameters[10]>0.0f)) return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;

    const NMHillCompensatedFloatPair dE11=nmHillCompensatedDot3(
        F[0],dF[0],F[3],dF[3],F[6],dF[6]);
    const NMHillCompensatedFloatPair dE22=nmHillCompensatedDot3(
        F[1],dF[1],F[4],dF[4],F[7],dF[7]);
    const NMHillCompensatedFloatPair dE33=nmHillCompensatedDot3(
        F[2],dF[2],F[5],dF[5],F[8],dF[8]);
    const NMHillCompensatedFloatPair dGamma23=nmHillCompensatedAdd(
        nmHillCompensatedDot3(dF[1],F[2],dF[4],F[5],dF[7],F[8]),
        nmHillCompensatedDot3(F[1],dF[2],F[4],dF[5],F[7],dF[8]));
    const NMHillCompensatedFloatPair dGamma13=nmHillCompensatedAdd(
        nmHillCompensatedDot3(dF[0],F[2],dF[3],F[5],dF[6],F[8]),
        nmHillCompensatedDot3(F[0],dF[2],F[3],dF[5],F[6],dF[8]));
    const NMHillCompensatedFloatPair dGamma12=nmHillCompensatedAdd(
        nmHillCompensatedDot3(dF[0],F[1],dF[3],F[4],dF[6],F[7]),
        nmHillCompensatedDot3(F[0],dF[1],F[3],dF[4],F[6],dF[7]));
    const NMHillCompensatedFloatPair numerator[6]={
        nmHillCompensatedAdd(nmHillCompensatedAdd(
            nmHillCompensatedMultiply(dE11,parameters[1]),
            nmHillCompensatedMultiply(dE22,parameters[4])),
            nmHillCompensatedMultiply(dE33,parameters[5])),
        nmHillCompensatedAdd(nmHillCompensatedAdd(
            nmHillCompensatedMultiply(dE11,parameters[4]),
            nmHillCompensatedMultiply(dE22,parameters[2])),
            nmHillCompensatedMultiply(dE33,parameters[6])),
        nmHillCompensatedAdd(nmHillCompensatedAdd(
            nmHillCompensatedMultiply(dE11,parameters[5]),
            nmHillCompensatedMultiply(dE22,parameters[6])),
            nmHillCompensatedMultiply(dE33,parameters[3])),
        nmHillCompensatedMultiply(dGamma23,parameters[9]),
        nmHillCompensatedMultiply(dGamma13,parameters[8]),
        nmHillCompensatedMultiply(dGamma12,parameters[7])};
    float result[6];
    for (unsigned i=0;i<6;++i) {
        result[i]=nmHillCompensatedDivide(numerator[i],parameters[10]).high;
        if (!nmHillCompensatedFinite(result[i]))
            return NM_HILL_SCALAR_INITIALIZER_NONFINITE;
    }
    for (unsigned i=0;i<6;++i) dStress[i]=result[i];
    return NM_HILL_SCALAR_INITIALIZER_SUCCESS;
}

#undef NM_HILL_COMP_INLINE
#undef NM_HILL_COMP_THREAD
#undef NM_HILL_COMP_PARAMETER
#undef NM_HILL_COMP_FMA
#undef NM_HILL_COMP_ISFINITE
