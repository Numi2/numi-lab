#pragma once

#include "metalrobo/numi_human_stand_gpu.h"

// Internal single-owner CPU continuation handoff. The CPU must populate every
// field for the exact current step before the GPU resumes postprocessing.
#define MR_NUMI_HUMAN_STAND_CPU_FINISH_ABI_VERSION 1u
typedef struct MRNumiHumanStandCpuFinishGPU {
    mr_u32 abiVersion;
    mr_u32 stepIndex;
    mr_u32 dofCount;
    mr_u32 contactCount;
    mr_u32 equalityCount;
    mr_u32 limitCount;
    mr_u32 activeContacts;
    mr_u32 reserved0;
    float minimumGap;
    float maximumPenetration;
    float contactNormalImpulseWork;
    float contactTangentialImpulseWork;
    float equalityImpulseWork;
    float sourceLimitImpulseWork;
    float contactNormalAbsoluteImpulseWork;
    float contactTangentialAbsoluteImpulseWork;
    float equalityAbsoluteImpulseWork;
    float sourceLimitAbsoluteImpulseWork;
    float velocity[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    float contactLambdas[3u * MR_NUMI_HUMAN_STAND_MAX_CONTACTS];
    float equalityLambdas[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    float equalityRhs[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    mr_u32 limitDofs[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    float limitPreStepPositions[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    float limitAccumulatedImpulses[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    float contactMatrices[9u * MR_NUMI_HUMAN_STAND_MAX_CONTACTS];
    mr_u32 contactActiveForPostProjection[MR_NUMI_HUMAN_STAND_MAX_CONTACTS];
    float contactTargetNormalVelocityForPostProjection[
        MR_NUMI_HUMAN_STAND_MAX_CONTACTS];
} MRNumiHumanStandCpuFinishGPU;
