#pragma once

// Source-equivalent compensated evaluation of the 13-state ideal-Hill local
// residual, candidate Jacobian, and fixed-candidate deformation action.
// This header is intentionally separate from the stress helper so callers may
// use trial stress alone when they do not need the full local system.
#include "numi/matter/hill_compensated_stress.h"

#if defined(__METAL_VERSION__)
#define NM_HILL_SYSTEM_INLINE inline
#define NM_HILL_SYSTEM_THREAD thread
#define NM_HILL_SYSTEM_PARAMETER device const float*
#define NM_HILL_SYSTEM_SQRT sqrt
#define NM_HILL_SYSTEM_ISFINITE isfinite
#else
#include <math.h>
#define NM_HILL_SYSTEM_INLINE inline
#define NM_HILL_SYSTEM_THREAD
#define NM_HILL_SYSTEM_PARAMETER const float*
#define NM_HILL_SYSTEM_SQRT sqrtf
#define NM_HILL_SYSTEM_ISFINITE isfinite
#endif

NM_HILL_SYSTEM_INLINE bool nmHillSystemFinite(float value) {
#if defined(__METAL_VERSION__)
    return NM_HILL_SYSTEM_ISFINITE(value);
#else
    return NM_HILL_SYSTEM_ISFINITE(value)!=0;
#endif
}

NM_HILL_SYSTEM_INLINE NMHillCompensatedFloatPair nmHillSystemMultiplyPairs(
    NMHillCompensatedFloatPair a,NMHillCompensatedFloatPair b) {
    const NMHillCompensatedFloatPair leading=nmHillCompensatedProduct(a.high,b.high);
    const NMHillCompensatedFloatPair cross=nmHillCompensatedAdd(
        nmHillCompensatedProduct(a.high,b.low),
        nmHillCompensatedProduct(a.low,b.high));
    const NMHillCompensatedFloatPair low=nmHillCompensatedProduct(a.low,b.low);
    return nmHillCompensatedAdd(leading,nmHillCompensatedAdd(cross,low));
}

NM_HILL_SYSTEM_INLINE bool nmHillSystemSquareRoot(
    NMHillCompensatedFloatPair value,
    NM_HILL_SYSTEM_THREAD NMHillCompensatedFloatPair* root) {
    const float estimateSquared=value.high+value.low;
    if (!(estimateSquared>=0.0f)||!nmHillSystemFinite(estimateSquared)) return false;
    const float estimate=NM_HILL_SYSTEM_SQRT(estimateSquared);
    if (!nmHillSystemFinite(estimate)) return false;
    if (estimate==0.0f) {
        if (value.high==0.0f&&value.low==0.0f) {
            *root={0.0f,0.0f};
            return true;
        }
        return false;
    }
    const NMHillCompensatedFloatPair square=nmHillSystemMultiplyPairs(
        {estimate,0.0f},{estimate,0.0f});
    const NMHillCompensatedFloatPair remainder=nmHillCompensatedAdd(value,
        {-square.high,-square.low});
    const NMHillCompensatedFloatPair correction=nmHillCompensatedDivide(
        remainder,2.0f*estimate);
    *root=nmHillCompensatedAdd({estimate,0.0f},correction);
    return nmHillSystemFinite(root->high)&&nmHillSystemFinite(root->low);
}

NM_HILL_SYSTEM_INLINE int nmHillCompensatedHillEquivalent(
    NM_HILL_SYSTEM_PARAMETER parameters,
    NM_HILL_SYSTEM_THREAD const float* stress,
    NM_HILL_SYSTEM_THREAD NMHillCompensatedFloatPair* equivalent) {
    for (unsigned i=0;i<20;++i)
        if (!nmHillSystemFinite(parameters[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    for (unsigned i=0;i<6;++i)
        if (!nmHillSystemFinite(stress[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    if (!(parameters[17]>0.0f)) return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    const NMHillCompensatedFloatPair s12=nmHillCompensatedAdd(
        {stress[0],0.0f},{-stress[1],0.0f});
    const NMHillCompensatedFloatPair s23=nmHillCompensatedAdd(
        {stress[1],0.0f},{-stress[2],0.0f});
    const NMHillCompensatedFloatPair s31=nmHillCompensatedAdd(
        {stress[2],0.0f},{-stress[0],0.0f});
    NMHillCompensatedFloatPair squared={0.0f,0.0f};
    squared=nmHillCompensatedAdd(squared,nmHillCompensatedMultiply(
        nmHillSystemMultiplyPairs(s23,s23),parameters[11]));
    squared=nmHillCompensatedAdd(squared,nmHillCompensatedMultiply(
        nmHillSystemMultiplyPairs(s31,s31),parameters[12]));
    squared=nmHillCompensatedAdd(squared,nmHillCompensatedMultiply(
        nmHillSystemMultiplyPairs(s12,s12),parameters[13]));
    squared=nmHillCompensatedAdd(squared,nmHillCompensatedMultiply(
        nmHillSystemMultiplyPairs({stress[3],0.0f},{stress[3],0.0f}),
        2.0f*parameters[14]));
    squared=nmHillCompensatedAdd(squared,nmHillCompensatedMultiply(
        nmHillSystemMultiplyPairs({stress[4],0.0f},{stress[4],0.0f}),
        2.0f*parameters[15]));
    squared=nmHillCompensatedAdd(squared,nmHillCompensatedMultiply(
        nmHillSystemMultiplyPairs({stress[5],0.0f},{stress[5],0.0f}),
        2.0f*parameters[16]));
    squared=nmHillCompensatedAdd(squared,nmHillCompensatedProduct(
        parameters[17],parameters[17]));
    if (!nmHillSystemSquareRoot(squared,equivalent))
        return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    return NM_HILL_SCALAR_INITIALIZER_SUCCESS;
}

NM_HILL_SYSTEM_INLINE int nmHillCompensatedHillImplicitSystem(
    NM_HILL_SYSTEM_PARAMETER parameters,
    NM_HILL_SYSTEM_THREAD const float* F,
    NM_HILL_SYSTEM_THREAD const float* accepted,
    NM_HILL_SYSTEM_THREAD const float* candidate,
    NM_HILL_SYSTEM_THREAD const float* deformationDirection,
    NM_HILL_SYSTEM_THREAD float* residual,
    NM_HILL_SYSTEM_THREAD float* candidateJacobian,
    NM_HILL_SYSTEM_THREAD float* deformationResidualAction) {
    for (unsigned i=0;i<20;++i)
        if (!nmHillSystemFinite(parameters[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    for (unsigned i=0;i<9;++i)
        if (!nmHillSystemFinite(F[i])||!nmHillSystemFinite(deformationDirection[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    for (unsigned i=0;i<13;++i)
        if (!nmHillSystemFinite(accepted[i])||!nmHillSystemFinite(candidate[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    if (!(parameters[10]>0.0f)||!(parameters[17]>0.0f)||
        !(parameters[18]>0.0f)||!(parameters[19]>0.0f))
        return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;

    float calculatedStress[6],stressDirection[6];
    int status=nmHillCompensatedTrialStress(parameters,F,candidate,calculatedStress);
    if (status!=NM_HILL_SCALAR_INITIALIZER_SUCCESS) return status;
    status=nmHillCompensatedTrialStressDirectionalDerivative(
        parameters,F,deformationDirection,stressDirection);
    if (status!=NM_HILL_SCALAR_INITIALIZER_SUCCESS) return status;

    NMHillCompensatedFloatPair qPair;
    const float stress[6]={candidate[6],candidate[7],candidate[8],candidate[9],
        candidate[10],candidate[11]};
    const NMHillCompensatedFloatPair s12=nmHillCompensatedAdd(
        {stress[0],0.0f},{-stress[1],0.0f});
    const NMHillCompensatedFloatPair s23=nmHillCompensatedAdd(
        {stress[1],0.0f},{-stress[2],0.0f});
    status=nmHillCompensatedHillEquivalent(parameters,stress,&qPair);
    if (status!=NM_HILL_SCALAR_INITIALIZER_SUCCESS) return status;
    const float q=qPair.high+qPair.low;
    if (!(q>0.0f)||!nmHillSystemFinite(q))
        return NM_HILL_SCALAR_INITIALIZER_NONFINITE;

    const NMHillCompensatedFloatPair flowNumerator[6]={
        nmHillCompensatedAdd(
            nmHillCompensatedMultiply(nmHillCompensatedAdd(
                {stress[0],0.0f},{-stress[2],0.0f}),parameters[12]),
            nmHillCompensatedMultiply(s12,parameters[13])),
        nmHillCompensatedAdd(
            nmHillCompensatedMultiply(s23,parameters[11]),
            nmHillCompensatedMultiply(nmHillCompensatedAdd(
                {stress[1],0.0f},{-stress[0],0.0f}),parameters[13])),
        nmHillCompensatedAdd(
            nmHillCompensatedMultiply(nmHillCompensatedAdd(
                {stress[2],0.0f},{-stress[1],0.0f}),parameters[11]),
            nmHillCompensatedMultiply(nmHillCompensatedAdd(
                {stress[2],0.0f},{-stress[0],0.0f}),parameters[12])),
        nmHillCompensatedProduct(parameters[14],stress[3]),
        nmHillCompensatedProduct(parameters[15],stress[4]),
        nmHillCompensatedProduct(parameters[16],stress[5])};
    const NMHillCompensatedFloatPair qGradientNumerator[6]={
        flowNumerator[0],flowNumerator[1],flowNumerator[2],
        nmHillCompensatedMultiply(flowNumerator[3],2.0f),
        nmHillCompensatedMultiply(flowNumerator[4],2.0f),
        nmHillCompensatedMultiply(flowNumerator[5],2.0f)};
    const float normalB[3][3]={
        {parameters[12]+parameters[13],-parameters[13],-parameters[12]},
        {-parameters[13],parameters[11]+parameters[13],-parameters[11]},
        {-parameters[12],-parameters[11],parameters[11]+parameters[12]}};
    const float flowShear[3]={parameters[14],parameters[15],parameters[16]};
    float flow[6],qGradient[6];
    for (unsigned i=0;i<6;++i) {
        flow[i]=nmHillCompensatedDivide(flowNumerator[i],q).high;
        qGradient[i]=nmHillCompensatedDivide(qGradientNumerator[i],q).high;
    }

    const NMHillCompensatedFloatPair deltaPair=nmHillCompensatedAdd(
        {candidate[12],0.0f},{-accepted[12],0.0f});
    const float delta=deltaPair.high+deltaPair.low;
    const NMHillCompensatedFloatPair gapPair=nmHillCompensatedAdd(
        {1.0f,0.0f},{-qPair.high,-qPair.low});
    const float gap=gapPair.high+gapPair.low;
    NMHillCompensatedFloatPair rhoSquared=nmHillSystemMultiplyPairs(deltaPair,deltaPair);
    rhoSquared=nmHillCompensatedAdd(rhoSquared,
        nmHillSystemMultiplyPairs(gapPair,gapPair));
    rhoSquared=nmHillCompensatedAdd(rhoSquared,nmHillCompensatedProduct(
        parameters[18],parameters[18]));
    NMHillCompensatedFloatPair rhoPair;
    if (!nmHillSystemSquareRoot(rhoSquared,&rhoPair))
        return NM_HILL_SCALAR_INITIALIZER_NONFINITE;
    const float rho=rhoPair.high+rhoPair.low;
    if (!(rho>0.0f)||!nmHillSystemFinite(rho))
        return NM_HILL_SCALAR_INITIALIZER_NONFINITE;
    const NMHillCompensatedFloatPair fb=nmHillCompensatedAdd(
        nmHillCompensatedAdd(rhoPair,{-deltaPair.high,-deltaPair.low}),
        {-gapPair.high,-gapPair.low});

    float residualResult[13],jacobianResult[169],deformationResult[13];
    for (unsigned i=0;i<169;++i) jacobianResult[i]=0.0f;
    for (unsigned i=0;i<13;++i) deformationResult[i]=0.0f;
    const float residualScale=parameters[19];
    for (unsigned i=0;i<6;++i) {
        const NMHillCompensatedFloatPair plasticResidual=nmHillCompensatedAdd(
            nmHillCompensatedAdd({candidate[i],0.0f},{-accepted[i],0.0f}),
            nmHillCompensatedMultiply({-delta,0.0f},flow[i]));
        residualResult[i]=nmHillCompensatedMultiply(plasticResidual,residualScale).high;
        jacobianResult[13*i+i]=residualScale;
        for (unsigned k=0;k<6;++k) {
            float dFlow=0.0f;
            if (i<3&&k<3) {
                dFlow=normalB[i][k]/q-
                    (flowNumerator[i].high+flowNumerator[i].low)*qGradient[k]/(q*q);
            } else if (i>=3&&i==k) {
                dFlow=flowShear[i-3]/q-
                    (flowNumerator[i].high+flowNumerator[i].low)*qGradient[k]/(q*q);
            } else {
                dFlow=-(flowNumerator[i].high+flowNumerator[i].low)*
                    qGradient[k]/(q*q);
            }
            jacobianResult[13*i+6+k]=-residualScale*delta*dFlow;
        }
        jacobianResult[13*i+12]=-residualScale*flow[i];
    }
    const unsigned normalC[3][3]={{1,4,5},{4,2,6},{5,6,3}};
    for (unsigned i=0;i<6;++i) {
        const unsigned row=6+i;
        const NMHillCompensatedFloatPair stressResidual=nmHillCompensatedAdd(
            {candidate[row],0.0f},{-calculatedStress[i],0.0f});
        residualResult[row]=nmHillCompensatedMultiply(stressResidual,residualScale).high;
        if (i<3) {
            for (unsigned k=0;k<3;++k)
                jacobianResult[13*row+k]=residualScale*
                    (parameters[normalC[i][k]]/parameters[10]);
        } else {
            const unsigned epIndex=i;
            const unsigned shearParameter=i==3?9u:(i==4?8u:7u);
            jacobianResult[13*row+epIndex]=residualScale*
                (2.0f*parameters[shearParameter]/parameters[10]);
        }
        jacobianResult[13*row+row]=residualScale;
        deformationResult[row]=-residualScale*stressDirection[i];
    }
    residualResult[12]=nmHillCompensatedMultiply(fb,residualScale).high;
    const float inverseRho=1.0f/rho;
    jacobianResult[13*12+12]=residualScale*(delta*inverseRho-1.0f);
    for (unsigned i=0;i<6;++i)
        jacobianResult[13*12+6+i]=residualScale*qGradient[i]*(1.0f-gap*inverseRho);

    for (unsigned i=0;i<13;++i)
        if (!nmHillSystemFinite(residualResult[i])||!nmHillSystemFinite(deformationResult[i]))
            return NM_HILL_SCALAR_INITIALIZER_NONFINITE;
    for (unsigned i=0;i<169;++i)
        if (!nmHillSystemFinite(jacobianResult[i]))
            return NM_HILL_SCALAR_INITIALIZER_NONFINITE;
    for (unsigned i=0;i<13;++i) residual[i]=residualResult[i];
    for (unsigned i=0;i<169;++i) candidateJacobian[i]=jacobianResult[i];
    for (unsigned i=0;i<13;++i) deformationResidualAction[i]=deformationResult[i];
    return NM_HILL_SCALAR_INITIALIZER_SUCCESS;
}

#undef NM_HILL_SYSTEM_INLINE
#undef NM_HILL_SYSTEM_THREAD
#undef NM_HILL_SYSTEM_PARAMETER
#undef NM_HILL_SYSTEM_SQRT
#undef NM_HILL_SYSTEM_ISFINITE
