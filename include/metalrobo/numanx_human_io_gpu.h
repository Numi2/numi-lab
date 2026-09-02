#pragma once

// Pointer-free ABI shared by C++, Objective-C++, and Metal for the NumanX
// Human motor/proprioception transaction adapter. Keep this header free of
// STL and Objective-C types.

#include "metalrobo/mujoco_muscle_gpu.h"
#include "metalrobo/numi_human_stand_gpu.h"

#define MR_NUMANX_HUMAN_IO_ABI_VERSION 7u
#define MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT 10u
#define MR_NUMANX_HUMAN_PROPRIOCEPTION_VALIDITY_ALL 0x000003ffu
#define MR_NUMANX_HUMAN_INTEROCEPTION_FEATURE_COUNT 6u
#define MR_NUMANX_HUMAN_INTEROCEPTION_VALIDITY_ALL 0x0000003fu
#define MR_NUMANX_HUMAN_KINESTHESIA_RECEPTOR_COUNT 128u
#define MR_NUMANX_HUMAN_KINESTHESIA_FEATURE_COUNT 7u
#define MR_NUMANX_HUMAN_KINESTHESIA_VALIDITY_ALL 0x0000007fu
#define MR_NUMANX_HUMAN_VESTIBULAR_RECEPTOR_COUNT 1u
#define MR_NUMANX_HUMAN_VESTIBULAR_FEATURE_COUNT 22u
#define MR_NUMANX_HUMAN_VESTIBULAR_VALIDITY_ALL 0x003fffffu
#define MR_NUMANX_HUMAN_AUDITION_RECEPTOR_COUNT 24u
#define MR_NUMANX_HUMAN_AUDITION_FEATURE_COUNT 8u
#define MR_NUMANX_HUMAN_AUDITION_FIRST_VALIDITY 0x000000fbu
#define MR_NUMANX_HUMAN_AUDITION_VALIDITY_ALL 0x000000ffu
#define MR_NUMANX_HUMAN_VISION_WIDTH 64u
#define MR_NUMANX_HUMAN_VISION_HEIGHT 48u
#define MR_NUMANX_HUMAN_VISION_RECEPTOR_COUNT \
    (MR_NUMANX_HUMAN_VISION_WIDTH * MR_NUMANX_HUMAN_VISION_HEIGHT)
#define MR_NUMANX_HUMAN_VISION_FEATURE_COUNT 8u
#define MR_NUMANX_HUMAN_VISION_VALIDITY_RAY 0x00000007u
#define MR_NUMANX_HUMAN_VISION_VALIDITY_DEPTH 0x00000008u
#define MR_NUMANX_HUMAN_VISION_VALIDITY_GEOMETRY 0x00000070u
#define MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT 10u
#define MR_NUMANX_HUMAN_TOUCH_FEATURE_COUNT 7u
#define MR_NUMANX_HUMAN_SUPPORT_CONSEQUENCE_VERSION 1u
#define MR_NUMANX_HUMAN_TOUCH_VALIDITY_ALL 0x0000007fu

// Exact version-3 NumiBrain motor-output contract consumed by NumanX. The
// layout is duplicated here deliberately so MetalRobo does not acquire a
// build-time dependency on the Swift package. Cross-repository ABI tests must
// keep its 80-byte size and offsets identical to NBMotorOutputHeader.
#define MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION 3u
#define MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION 1u
#define MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION 7u
#define MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID 1u
#define MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW (1u << 1u)
#define MR_NUMANX_BRAIN_JOINT_TRANSACTION_BYTE_COUNT 96u
#define MR_NUMANX_BRAIN_JOINT_SUBSTEP_BYTE_COUNT 72u
#define MR_NUMANX_BRAIN_MOTOR_CANDIDATE_BYTE_COUNT 152u
#define MR_NUMANX_BRAIN_MOTOR_OUTPUT_HEADER_BYTE_COUNT 80u
#define MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT 16u
#define MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT 16u
#define MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION 1u
#define MR_NUMANX_BRAIN_MOTOR_READY_ABI_VERSION 1u
#define MR_NUMANX_BRAIN_MOTOR_READY_GATE_BYTE_COUNT 160u
#define MR_NUMANX_BRAIN_READY_GATE_SUCCESS 1u
#define MR_NUMANX_BRAIN_READY_GATE_FAILURE 2u

enum MRNumanXBrainMotorOutputFlags : mr_u32 {
    MR_NUMANX_BRAIN_MOTOR_OUTPUT_VALID = 1u << 0u,
    MR_NUMANX_BRAIN_MOTOR_OUTPUT_EMERGENCY_STOP = 1u << 1u,
    MR_NUMANX_BRAIN_MOTOR_OUTPUT_LOCALIZED_SOURCE_INHIBITION = 1u << 2u,
    MR_NUMANX_BRAIN_MOTOR_OUTPUT_LOCALIZED_WITHDRAWAL = 1u << 3u,
};

#define MR_NUMANX_BRAIN_MOTOR_OUTPUT_KNOWN_FLAGS \
    (MR_NUMANX_BRAIN_MOTOR_OUTPUT_VALID | \
     MR_NUMANX_BRAIN_MOTOR_OUTPUT_EMERGENCY_STOP | \
     MR_NUMANX_BRAIN_MOTOR_OUTPUT_LOCALIZED_SOURCE_INHIBITION | \
     MR_NUMANX_BRAIN_MOTOR_OUTPUT_LOCALIZED_WITHDRAWAL)

typedef struct MR_ALIGN16 MRNumanXBrainMotorOutputHeaderGPU {
    mr_u32 formatVersion;
    mr_u32 flags;
    mr_u64 timestampMicroseconds;
    mr_u64 brainGeneration;
    mr_u64 profileFingerprint;
    mr_u64 protectiveCommandFingerprint;
    mr_u32 muscleCount;
    mr_u32 environmentIdentifier;
    float motorInhibition;
    float autonomicArousal;
    mr_u32 actuatorCommandKind;
    mr_u32 reserved;
    float outputMinimum;
    float outputMaximum;
    mr_u64 outputFingerprint;
} MRNumanXBrainMotorOutputHeaderGPU;

// Exact value mirrors of NumiBrain ABI v7. They are host-validated before
// any borrowed Metal resource is offered to the Human transaction. GPU virtual
// addresses are ephemeral lease identity and are deliberately included in the
// candidate fingerprint, exactly as in NumiBrainABI.
typedef struct MRNumanXBrainJointTransactionToken {
    mr_u32 formatVersion;
    mr_u32 environmentIdentifier;
    mr_u64 episodeIdentifier;
    mr_u64 controlStepIdentifier;
    mr_u64 parameterVersionFingerprint;
    mr_u64 baseBrainGeneration;
    mr_u64 basePhysicsGeneration;
    mr_u64 committedTimestampMicroseconds;
    mr_u64 targetTimestampMicroseconds;
    mr_u64 shadowGeneration;
    mr_u64 randomCounterGeneration;
    mr_u32 flags;
    mr_u32 reserved;
    mr_u64 transactionFingerprint;
} MRNumanXBrainJointTransactionToken;

typedef struct MRNumanXBrainJointSubstepToken {
    mr_u64 transactionFingerprint;
    mr_u32 substepIndex;
    mr_u32 attemptIndex;
    mr_u64 startTimestampMicroseconds;
    mr_u64 durationMicroseconds;
    mr_u64 candidateTimestampMicroseconds;
    mr_u64 shadowGeneration;
    mr_u64 randomCounterGeneration;
    mr_u32 flags;
    mr_u32 reserved;
    mr_u64 substepFingerprint;
} MRNumanXBrainJointSubstepToken;

typedef struct MRNumanXBrainMotorCandidate {
    mr_u32 formatVersion;
    mr_u32 flags;
    mr_u64 transactionFingerprint;
    mr_u64 substepFingerprint;
    mr_u64 acceptedBrainTimestampMicroseconds;
    mr_u64 brainGeneration;
    mr_u64 motorProfileFingerprint;
    mr_u64 motorOutputHeaderGPUAddress;
    mr_u64 muscleExcitationGPUAddress;
    mr_u64 randomCounterGeneration;
    mr_u32 motorOutputHeaderByteCount;
    mr_u32 muscleExcitationByteCount;
    mr_u32 muscleCount;
    mr_u32 environmentIdentifier;
    mr_u64 autonomicCommandGPUAddress;
    mr_u32 autonomicCommandByteCount;
    mr_u32 autonomicCommandCount;
    mr_u64 activeSensingCommandGPUAddress;
    mr_u32 activeSensingCommandByteCount;
    mr_u32 activeSensingCommandCount;
    mr_u32 actuatorCommandKind;
    mr_u32 reserved;
    mr_u64 speciesTemplateFingerprint;
    mr_u64 compiledSpeciesTemplateFingerprint;
    mr_u64 candidateFingerprint;
} MRNumanXBrainMotorCandidate;

// Exact value mirror of NumiBrain's 160-byte async motor-ready gate. The
// paired shared event provides ordering/liveness only; this record is the GPU
// authority that permits HumanIO to touch the candidate payload.
typedef struct MR_ALIGN16 MRNumanXBrainMotorReadyGateGPU {
    mr_u32 abiVersion;
    mr_u32 structBytes;
    mr_u32 status;
    mr_u32 environment;
    mr_u32 substepIndex;
    mr_u32 attemptIndex;
    mr_u32 muscleCount;
    mr_u32 actuatorCommandKind;
    mr_u64 controlStep;
    mr_u64 transactionFingerprint;
    mr_u64 substepFingerprint;
    mr_u64 candidateFingerprint;
    mr_u64 motorOutputFingerprint;
    mr_u64 motorProfileFingerprint;
    mr_u64 brainGeneration;
    mr_u64 acceptedBrainTimestampMicroseconds;
    mr_u64 randomCounterGeneration;
    mr_u64 speciesTemplateFingerprint;
    mr_u64 compiledSpeciesTemplateFingerprint;
    mr_u64 brainProgramFingerprint;
    mr_u64 fastProgramFingerprint;
    mr_u64 decisionGateFingerprint;
    mr_u64 reserved64_0;
    mr_u64 gateFingerprint;
} MRNumanXBrainMotorReadyGateGPU;

enum MRNumanXHumanMotorHeaderValidation : mr_u32 {
    MR_NUMANX_HUMAN_MOTOR_HEADER_PENDING = 0u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_VALID = 1u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_FORMAT = 2u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_FLAGS = 3u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_IDENTITY = 4u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_GENERATION = 5u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_NONFINITE = 6u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_RANGE = 7u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_COMMAND_KIND = 8u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_RELATION = 9u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_PAYLOAD = 10u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_FINGERPRINT = 11u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_READY_GATE = 12u,
};

// One receptor row is emitted for each environment-major MyoSim muscle. The
// sample is causal but intentionally mixed-time in exactly the same way as the
// enclosing NumanX phase contract: it is evaluated from pre-dynamics route
// geometry at time t and delivered only after the stand step reaches t + dt.
// Activation/fibre state is the state after the explicit MyoSim update at t.
enum MRNumanXHumanProprioceptionFeature : mr_u32 {
    MR_NUMANX_HUMAN_FEATURE_EXCITATION = 0u,
    MR_NUMANX_HUMAN_FEATURE_ACTIVATION = 1u,
    MR_NUMANX_HUMAN_FEATURE_FIBRE_LENGTH_METRES = 2u,
    MR_NUMANX_HUMAN_FEATURE_FIBRE_VELOCITY_METRES_PER_SECOND = 3u,
    MR_NUMANX_HUMAN_FEATURE_PATH_LENGTH_METRES = 4u,
    MR_NUMANX_HUMAN_FEATURE_PATH_VELOCITY_METRES_PER_SECOND = 5u,
    MR_NUMANX_HUMAN_FEATURE_APPLIED_ACTIVE_FORCE_NEWTONS = 6u,
    MR_NUMANX_HUMAN_FEATURE_TENDON_TENSION_NEWTONS = 7u,
    MR_NUMANX_HUMAN_FEATURE_ACTIVATION_DERIVATIVE_PER_SECOND = 8u,
    MR_NUMANX_HUMAN_FEATURE_NORMALIZED_EQUILIBRIUM_RESIDUAL = 9u,
};

// Six bounded causal physiology proxies are emitted for every muscle from the
// accepted MyoSim state/result. They are not a blood-gas or thermal solver;
// they preserve explicit provenance from excitation, activation, fibre
// velocity, tendon load, and equilibrium residual while giving NumiBrain the
// exact six-feature interoceptive topology it consumes.
enum MRNumanXHumanInteroceptionFeature : mr_u32 {
    MR_NUMANX_HUMAN_INTEROCEPTION_ENERGY_AVAILABILITY = 0u,
    MR_NUMANX_HUMAN_INTEROCEPTION_OXYGEN_AVAILABILITY = 1u,
    MR_NUMANX_HUMAN_INTEROCEPTION_CARBON_DIOXIDE_LOAD = 2u,
    MR_NUMANX_HUMAN_INTEROCEPTION_THERMAL_LOAD = 3u,
    MR_NUMANX_HUMAN_INTEROCEPTION_FATIGUE_LOAD = 4u,
    MR_NUMANX_HUMAN_INTEROCEPTION_TISSUE_STRESS = 5u,
};

enum MRNumanXHumanMotorValidationFlags : mr_u32 {
    MR_NUMANX_HUMAN_MOTOR_FINITE = 1u << 0u,
    MR_NUMANX_HUMAN_MOTOR_IN_UNIT_INTERVAL = 1u << 1u,
    MR_NUMANX_HUMAN_MOTOR_COPIED = 1u << 2u,
    MR_NUMANX_HUMAN_MOTOR_HEADER_AUTHENTICATED = 1u << 3u,
};

// Constants are copied into the command buffer at every beginStep. The caller
// binds the excitation slice with a Metal buffer offset, so all strides below
// are in scalar elements rather than bytes.
typedef struct MR_ALIGN16 MRNumanXHumanMotorDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 muscleCount;
    mr_u32 stateStride;

    mr_u32 excitationEnvironmentStride;
    mr_u32 stepIndex;
    mr_u32 stepCount;
    mr_u32 flags;

    mr_u32 motorOutputFormatVersion;
    mr_u32 actuatorCommandKind;
    mr_u32 environmentIdentifierBase;
    mr_u32 headerEnvironmentStride;

    mr_u64 transactionFingerprint;
    mr_u64 motorCandidateFingerprint;
    mr_u64 acceptedBrainGeneration;
    mr_u64 candidateSensorGeneration;
    mr_u64 expectedExcitationGPUAddress;
    mr_u64 expectedMotorOutputHeaderGPUAddress;
    mr_u64 acceptedBrainTimestampMicroseconds;
    mr_u64 motorProfileFingerprint;
    mr_u64 programFingerprint;
} MRNumanXHumanMotorDispatchGPU;

// Constants used by both post-stand kernels. Proprioception is laid out as
// [environment][step][muscle/receptor][feature]; validity is
// [environment][step][muscle/receptor], one UInt32 bit mask per receptor.
typedef struct MR_ALIGN16 MRNumanXHumanProprioceptionDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 muscleCount;
    mr_u32 featureCount;

    mr_u32 stepIndex;
    mr_u32 stepCount;
    mr_u32 stateStride;
    mr_u32 resultStride;

    mr_u32 proprioceptionEnvironmentStride;
    mr_u32 proprioceptionStepStride;
    mr_u32 validityEnvironmentStride;
    mr_u32 validityStepStride;

    mr_u32 flags;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;

    // x = timestep seconds; yzw are required zero.
    mr_float4 timestepSecondsAndReserved;

    mr_u64 transactionFingerprint;
    mr_u64 motorCandidateFingerprint;
    mr_u64 acceptedBrainGeneration;
    mr_u64 candidateSensorGeneration;
    mr_u64 expectedExcitationGPUAddress;
    mr_u64 programFingerprint;
} MRNumanXHumanProprioceptionDispatchGPU;

// Exact 16-byte mirror of NumiBrain's active-sensing command. command is a
// signed, normalized gaze actuator; confidence is [0,1]. The upper/lower flag
// words remain provenance-only in this first visual producer.
typedef struct MR_ALIGN16 MRNumanXActiveSensingCommandGPU {
    float command;
    float confidence;
    mr_u32 attentionAllocationMask;
    mr_u32 kindAndFlags;
} MRNumanXActiveSensingCommandGPU;

// Body-local authored visual bounds derived from the validated .mrvpack.
// The source pack remains the authority; this compact representation avoids
// duplicating its full indexed mesh in the sensor hot path while preserving a
// source-authored physical visibility envelope for every articulated body.
typedef struct MR_ALIGN16 MRNumanXVisualBodyBoundsGPU {
    mr_float4 minimum;
    mr_float4 maximum;
} MRNumanXVisualBodyBoundsGPU;

// Read-only mirror of Matter's accepted Human support consequence. Matter is
// the sole writer; this type exists only so the HumanIO shader can consume the
// exact bytes without linking its private solver headers.
typedef struct MR_ALIGN16 MRNumanXHumanSupportConsequenceGPU {
    mr_uint4 identity;
    mr_float4 pointAndSeparation;
    mr_float4 impulseAndNormal;
    mr_float4 tangentVelocityAndImpulse;
} MRNumanXHumanSupportConsequenceGPU;

// Constants for the five same-command-buffer supplemental sensor channels.
// All output tensors are one-environment, one-step, receptor-major FP32 with
// one UInt32 validity mask per receptor.
typedef struct MR_ALIGN16 MRNumanXHumanSupplementalDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 qCoordinateCount;
    mr_u32 dofCount;
    mr_u32 bodyCount;

    mr_u32 pointCount;
    mr_u32 supportPointOffset;
    mr_u32 supportPointCount;
    mr_u32 headBodyIndex;

    mr_u32 visionWidth;
    mr_u32 visionHeight;
    mr_u32 bodyBoundsCount;
    mr_u32 reserved0;

    mr_u64 sensorGeneration;
    mr_u64 transactionFingerprint;
    mr_u64 substepFingerprint;
    mr_u64 expectedActiveSensingGPUAddress;
    mr_u64 visualSourceFingerprint;
    mr_u64 programFingerprint;
    mr_u64 expectedSupportConsequencesGPUAddress;
    mr_u64 matterProgramFingerprint;

    mr_float4 groundPoint;
    mr_float4 groundNormal;
    mr_float4 cameraLocalPosition;
    mr_float4 cameraLocalOrientation;
    // fx, fy, cx, cy.
    mr_float4 visionIntrinsics;
    // minimum depth, maximum depth, depth quantum, timestep seconds.
    mr_float4 visionDepthAndTimestep;
} MRNumanXHumanSupplementalDispatchGPU;

#if !defined(__METAL_VERSION__)
#include <cstddef>
static_assert(MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT < 32u);
static_assert(MR_NUMANX_HUMAN_INTEROCEPTION_FEATURE_COUNT < 32u);
static_assert(sizeof(MRNumanXBrainJointTransactionToken) ==
    MR_NUMANX_BRAIN_JOINT_TRANSACTION_BYTE_COUNT);
static_assert(alignof(MRNumanXBrainJointTransactionToken) == 8u);
static_assert(offsetof(
    MRNumanXBrainJointTransactionToken, transactionFingerprint) == 88u);
static_assert(sizeof(MRNumanXBrainJointSubstepToken) ==
    MR_NUMANX_BRAIN_JOINT_SUBSTEP_BYTE_COUNT);
static_assert(alignof(MRNumanXBrainJointSubstepToken) == 8u);
static_assert(offsetof(
    MRNumanXBrainJointSubstepToken, substepFingerprint) == 64u);
static_assert(sizeof(MRNumanXBrainMotorCandidate) ==
    MR_NUMANX_BRAIN_MOTOR_CANDIDATE_BYTE_COUNT);
static_assert(alignof(MRNumanXBrainMotorCandidate) == 8u);
static_assert(offsetof(
    MRNumanXBrainMotorCandidate, transactionFingerprint) == 8u);
static_assert(offsetof(
    MRNumanXBrainMotorCandidate, motorOutputHeaderGPUAddress) == 48u);
static_assert(offsetof(
    MRNumanXBrainMotorCandidate, autonomicCommandGPUAddress) == 88u);
static_assert(offsetof(
    MRNumanXBrainMotorCandidate, activeSensingCommandGPUAddress) == 104u);
static_assert(offsetof(
    MRNumanXBrainMotorCandidate, speciesTemplateFingerprint) == 128u);
static_assert(offsetof(
    MRNumanXBrainMotorCandidate, candidateFingerprint) == 144u);
static_assert(sizeof(MRNumanXBrainMotorReadyGateGPU) ==
    MR_NUMANX_BRAIN_MOTOR_READY_GATE_BYTE_COUNT);
static_assert(alignof(MRNumanXBrainMotorReadyGateGPU) == 16u);
static_assert(offsetof(
    MRNumanXBrainMotorReadyGateGPU, gateFingerprint) == 152u);
static_assert(sizeof(MRNumanXBrainMotorOutputHeaderGPU) == 80u);
static_assert(alignof(MRNumanXBrainMotorOutputHeaderGPU) == 16u);
static_assert(
    offsetof(MRNumanXBrainMotorOutputHeaderGPU, timestampMicroseconds) == 8u
);
static_assert(
    offsetof(MRNumanXBrainMotorOutputHeaderGPU, outputFingerprint) == 72u
);
static_assert(sizeof(MRNumanXHumanMotorDispatchGPU) == 128u);
static_assert(alignof(MRNumanXHumanMotorDispatchGPU) == 16u);
static_assert(
    offsetof(MRNumanXHumanMotorDispatchGPU, transactionFingerprint) == 48u
);
static_assert(
    offsetof(
        MRNumanXHumanMotorDispatchGPU,
        expectedMotorOutputHeaderGPUAddress
    ) == 88u
);
static_assert(sizeof(MRNumanXHumanProprioceptionDispatchGPU) == 128u);
static_assert(alignof(MRNumanXHumanProprioceptionDispatchGPU) == 16u);
static_assert(
    offsetof(
        MRNumanXHumanProprioceptionDispatchGPU,
        timestepSecondsAndReserved
    ) == 64u
);
static_assert(
    offsetof(
        MRNumanXHumanProprioceptionDispatchGPU,
        transactionFingerprint
    ) == 80u
);
static_assert(sizeof(MRNumanXActiveSensingCommandGPU) == 16u);
static_assert(alignof(MRNumanXActiveSensingCommandGPU) == 16u);
static_assert(sizeof(MRNumanXVisualBodyBoundsGPU) == 32u);
static_assert(alignof(MRNumanXVisualBodyBoundsGPU) == 16u);
static_assert(sizeof(MRNumanXHumanSupplementalDispatchGPU) == 208u);
static_assert(sizeof(MRNumanXHumanSupportConsequenceGPU) == 64u);
static_assert(alignof(MRNumanXHumanSupplementalDispatchGPU) == 16u);
static_assert(offsetof(
    MRNumanXHumanSupplementalDispatchGPU, sensorGeneration) == 48u);
static_assert(offsetof(
    MRNumanXHumanSupplementalDispatchGPU, groundPoint) == 112u);
static_assert(offsetof(
    MRNumanXHumanSupplementalDispatchGPU, visionDepthAndTimestep) == 192u);
#endif
