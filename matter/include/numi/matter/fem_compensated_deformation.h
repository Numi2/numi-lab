#pragma once
#include "metalrobo/compensated_geometry_gpu.h"

#ifdef __METAL_VERSION__
#define NM_FEM_WORD_THREAD thread
#else
#define NM_FEM_WORD_THREAD
#endif

// Exact FP32 source coordinates are widened by ordered two-word arithmetic.
// Keeping I + grad(u) as two words avoids quantizing small strains to zero
// before the authored material program sees them. Matrices are row-major.
inline void nmFEMReferenceDeformationWords(
    NM_FEM_WORD_THREAD const float* displacementHigh,
    NM_FEM_WORD_THREAD const float* displacementLow,
    NM_FEM_WORD_THREAD const float* inverseRest,
    NM_FEM_WORD_THREAD const float* frame,
    NM_FEM_WORD_THREAD float* high,
    NM_FEM_WORD_THREAD float* low) {
    MRCompensatedScalar world[9];
    for (unsigned row=0u;row<3u;++row) {
        for (unsigned column=0u;column<3u;++column) {
            MRCompensatedScalar value{row==column ? 1.0f : 0.0f,0.0f};
            for(unsigned k=0u;k<3u;++k) {
                const auto difference=mrCompensatedAdd(
                    {displacementHigh[3u*(k+1u)+row],displacementLow[3u*(k+1u)+row]},
                    {-displacementHigh[row],-displacementLow[row]});
                value=mrCompensatedAdd(value,mrCompensatedMultiply(
                    difference,{inverseRest[3u*k+column],0.0f}));
            }
            world[3u*row+column]=value;
        }
    }
    for(unsigned row=0u;row<3u;++row) {
        for(unsigned column=0u;column<3u;++column) {
            MRCompensatedScalar value{0.0f,0.0f};
            for(unsigned k=0u;k<3u;++k)
                value=mrCompensatedAdd(value,mrCompensatedMultiply(
                    world[3u*row+k],{frame[3u*k+column],0.0f}));
            high[3u*row+column]=value.high;
            low[3u*row+column]=value.low;
        }
    }
}

// Preserve the original single-word API for standalone arithmetic callers.
inline void nmFEMReferenceDeformationWords(
    NM_FEM_WORD_THREAD const float* displacement,
    NM_FEM_WORD_THREAD const float* inverseRest,
    NM_FEM_WORD_THREAD const float* frame,
    NM_FEM_WORD_THREAD float* high,
    NM_FEM_WORD_THREAD float* low) {
    NM_FEM_WORD_THREAD float displacementLow[12];
    for (unsigned i=0u;i<12u;++i) displacementLow[i]=0.0f;
    nmFEMReferenceDeformationWords(
        displacement,displacementLow,inverseRest,frame,high,low);
}

#undef NM_FEM_WORD_THREAD
