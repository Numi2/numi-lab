#pragma once
#include "metalrobo/engine_types.h"

#define MR_HUMAN_BEHAVIOR_ABI_VERSION 1u
#define MR_HUMAN_BEHAVIOR_AUDIT_ROOT_ASSISTANCE (1u << 0u)
#define MR_HUMAN_BEHAVIOR_AUDIT_DIRECT_TORQUE (1u << 1u)
#define MR_HUMAN_BEHAVIOR_AUDIT_KINEMATIC_OVERRIDE (1u << 2u)
#define MR_HUMAN_BEHAVIOR_AUDIT_UNREGISTERED_FORCE (1u << 3u)
#define MR_HUMAN_BEHAVIOR_AUDIT_SOURCE_CONSTRAINT_OMISSION (1u << 4u)
#define MR_HUMAN_BEHAVIOR_AUDIT_UNACCEPTED_PUBLICATION (1u << 5u)
#define MR_HUMAN_BEHAVIOR_AUDIT_NONFINITE (1u << 6u)
#define MR_HUMAN_BEHAVIOR_AUDIT_UNEXPECTED_RESET (1u << 7u)
#define MR_HUMAN_BEHAVIOR_COMPLETE_AUDIT_MASK 255u
#define MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION 1u
#define MR_HUMAN_BEHAVIOR_TRACE_STATUS_DISABLED 0u
#define MR_HUMAN_BEHAVIOR_TRACE_STATUS_READY 1u
#define MR_HUMAN_BEHAVIOR_TRACE_STATUS_OVERFLOW 2u
#define MR_HUMAN_BEHAVIOR_TRACE_STATUS_INVALID 3u
#define MR_HUMAN_BEHAVIOR_TRACE_STATUS_FINALIZED 4u
#define MR_HUMAN_BEHAVIOR_TRACE_JOINTLY_PUBLISHED 1u
#define MR_HUMAN_BEHAVIOR_TRACE_REJECTED_RELEASED 2u
#define MR_HUMAN_BEHAVIOR_TRACE_METRIC_UNAVAILABLE 0u
#define MR_HUMAN_BEHAVIOR_TRACE_METRIC_ACCEPTED 1u
#define MR_HUMAN_BEHAVIOR_TRACE_METRIC_REJECTED_CANDIDATE 2u
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

// Runtime-owned context for the optional diagnostic trace. The behavior
// reducer validates this identity against the release/fence before appending;
// malformed or absent context can only make the trace incomplete.
struct MR_ALIGN16 MRHumanBehaviorTraceAttemptContextGPU {
    mr_u32 abiVersion, structSize, present, controlStep;
    mr_u32 runtimeFailureStage, auditCoveredMask, auditViolationMask;
    mr_u32 forbiddenContactCoverage;
    mr_u32 forbiddenContactCount, reserved0, reserved1, reserved2;
    mr_u64 basePublicationEpoch, basePhysicsGeneration;
    mr_u64 baseAcceptedTimestampNanoseconds, baseAcceptedTokenFingerprint;
    mr_u64 basePublicationFingerprint;
    mr_u64 candidateStateProofFingerprint;
    mr_u64 candidateAcceptedTokenFingerprint;
    mr_u64 candidatePublicationFingerprint;
    mr_u64 afterPublicationEpoch, afterPhysicsGeneration;
    mr_u64 afterAcceptedTimestampNanoseconds;
    mr_u64 afterAcceptedTokenFingerprint, afterPublicationFingerprint;
};

// Fixed-capacity page state. Appends are O(1). Draining resets only the page
// window; aggregate counters and the fingerprint chain remain monotonic.
struct MR_ALIGN16 MRHumanBehaviorTracePageGPU {
    mr_u32 abiVersion, structSize, status, recordCapacity;
    mr_u64 traceInstanceFingerprint, expectedAcceptedRoots;
    mr_u64 chunkIndex, recordCount;
    mr_u64 firstAttemptIndex, lastAttemptIndex;
    mr_u64 pagePreviousRecordFingerprint, lastRecordFingerprint;
    mr_u64 observedAttemptCount, totalRecordCount;
    mr_u64 droppedRecordCount, drainedRecordCount;
    mr_u64 acceptedRecordCount, rejectedRecordCount;
    mr_u64 acceptedProofRecordCount, nativeAuditRecordCount;
    mr_u64 forbiddenContactCoveredAcceptedCount;
    mr_u64 lastAfterPublicationEpoch, lastAfterPhysicsGeneration;
    mr_u64 lastAfterAcceptedTimestampNanoseconds;
    mr_u64 lastAfterAcceptedTokenFingerprint;
    mr_u64 lastAfterPublicationFingerprint;
};

// Byte-for-byte mirror of mrnx_behavior_trace_record_v1. It intentionally
// carries only copied identities and source-bound candidate measurements; it
// owns no state and grants no publication authority.
struct MR_ALIGN16 MRHumanBehaviorTraceRecordGPU {
    mr_u32 abiVersion, structSize, disposition, metricKind;
    mr_u32 controlStep, runtimeFailureStage, candidateStatus, postureValid;
    mr_u32 settled, auditCoveredMask, auditViolationMask;
    mr_u32 forbiddenContactCoverage, forbiddenContactCount;
    mr_u32 reserved0, reserved1, reserved2;
    mr_u64 attemptIndex, transactionFingerprint, linearizationEpoch;
    mr_u64 slotGeneration, basePublicationEpoch, basePhysicsGeneration;
    mr_u64 baseAcceptedTimestampNanoseconds;
    mr_u64 baseAcceptedTokenFingerprint, candidatePhysicsGeneration;
    mr_u64 candidateTimestampNanoseconds;
    mr_u64 candidateStateProofFingerprint;
    mr_u64 candidateAcceptedTokenFingerprint;
    mr_u64 candidatePublicationFingerprint;
    mr_u64 afterPublicationEpoch, afterPhysicsGeneration;
    mr_u64 afterAcceptedTimestampNanoseconds;
    mr_u64 afterAcceptedTokenFingerprint, afterPublicationFingerprint;
    mr_u64 jointFenceFingerprint;
    float valueHigh[4], valueLow[4];
    mr_u64 reservedTail, previousRecordFingerprint, recordFingerprint;
};

#ifndef __METAL_VERSION__
#include <cstddef>
static_assert(sizeof(MRHumanBehaviorProgramGPU) == 192);
static_assert(sizeof(MRHumanBehaviorDispatchGPU) == 72);
static_assert(sizeof(MRHumanBehaviorCandidateGPU) == 112);
static_assert(sizeof(MRHumanBehaviorReleaseGPU) == 80);
static_assert(sizeof(MRHumanBehaviorTraceAttemptContextGPU) == 160);
static_assert(sizeof(MRHumanBehaviorTracePageGPU) == 192);
static_assert(sizeof(MRHumanBehaviorTraceRecordGPU) == 272);
static_assert(offsetof(MRHumanBehaviorTraceRecordGPU, recordFingerprint) == 264);
#endif
