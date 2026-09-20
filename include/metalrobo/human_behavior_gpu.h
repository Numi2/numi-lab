#pragma once
#include "metalrobo/engine_types.h"

#define MR_HUMAN_BEHAVIOR_ABI_VERSION 1u
#define MR_HUMAN_BEHAVIOR_COMPLETE_AUDIT_MASK 255u
// This is a measurement ABI, not a simulation or publication authority.
struct MRHumanBehaviorProgramGPU {
    mr_u32 abiVersion, task, bodyCount, dofCount;
    mr_u32 rootBody, trunkBody, velocityBody, forbiddenBodyCount;
    mr_u64 fingerprint, timestepNanoseconds;
    mr_float4 worldOriginHigh, worldOriginLow;
    mr_float4 rootOriginHigh, rootOriginLow;
    mr_float4 worldUp, worldForward, trunkUp;
    mr_float4 limitsHigh, limitsLow; // height, tilt radians, planar speed, target
};

struct MRHumanBehaviorDispatchGPU {
    mr_u32 environmentCount, bodyPoseStride, pointJacobianStride, vStride;
    mr_u32 bodyJacobianPointOffset, reserved0, reserved1, reserved2;
    mr_u64 transactionFingerprint, linearizationEpoch, slotGeneration;
    mr_u64 physicsGeneration, acceptedTimestampNanoseconds;
};

// Populated only by actual owner audits. A missing bit is unknown, not zero.
// The initial implementation may measure posture while reporting coverage=0.
struct MRHumanBehaviorAuditGPU {
    mr_u32 coveredMask, violationMask, forbiddenContactCoverage, forbiddenContactCount;
};

struct MRHumanBehaviorCandidateGPU {
    mr_u32 abiVersion, status, postureValid, settled;
    mr_u64 programFingerprint, transactionFingerprint, linearizationEpoch;
    mr_u64 slotGeneration, physicsGeneration, acceptedTimestampNanoseconds;
    mr_float4 valueHigh, valueLow; // source-origin height, tilt, planar, forward
    MRHumanBehaviorAuditGPU audit;
};

// This record is installed internally only after releasePublishedRoot succeeds.
// A COMMITTED GPU fence by itself does not authorize accepted accumulation.
struct MRHumanBehaviorReleaseGPU {
    mr_u64 programFingerprint, transactionFingerprint, linearizationEpoch;
    mr_u64 slotGeneration, physicsGeneration, acceptedTimestampNanoseconds;
    mr_u64 publicationSerial, jointFenceFingerprint;
    mr_u32 released, reserved0, reserved1, reserved2;
};

// One deterministic reducer lane per environment. No atomic FP reduction.
struct MRHumanBehaviorReductionGPU {
    mr_u32 abiVersion, status, initialPostureValid, initialSettled;
    mr_u64 programFingerprint, publicationSerial, acceptedRootCount, endNanoseconds;
    mr_u64 initialPhysicsGeneration, initialTimestampNanoseconds;
    mr_u64 metricSampleCount, auditCoveredRootCount, postureViolationCount, settledSuffixSteps;
    mr_u64 speedErrorSampleCount, rejectedAttemptCount, auditCoveredAttemptCount;
    mr_u64 auditViolations[8];
    mr_float4 extremaHigh, extremaLow; // minimum height, maximum tilt/planar, SSE
    mr_u64 lastTransactionFingerprint, lastJointFenceFingerprint;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(MRHumanBehaviorProgramGPU) == 192);
static_assert(sizeof(MRHumanBehaviorDispatchGPU) == 72);
static_assert(sizeof(MRHumanBehaviorCandidateGPU) == 112);
static_assert(sizeof(MRHumanBehaviorReleaseGPU) == 80);
#endif
