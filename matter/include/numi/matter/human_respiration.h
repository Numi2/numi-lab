#pragma once
#include "numi/matter/shared.h"
#include "metalrobo/mujoco_muscle_gpu.h"

// Reduced respiratory coordinates are swept volumes, not prescribed motion.
// All volumes and flows are SI. Gas amounts are m3 at STPD; blood content is
// m3(STPD)/m3(blood). Parameters belong to the reference adult, not a patient.
typedef struct NM_ALIGN16 NMHumanRespirationParameters {
    // FRC, anatomical dead space, lung compliance, airway resistance.
    nm_float4 lung; // m3, m3, m3/Pa, Pa s/m3
    // Diaphragm and rib swept-volume compliance and damping.
    nm_float4 chest; // m3/Pa, m3/Pa, Pa s/m3, Pa s/m3
    // Swept-volume inertance, diaphragm area, rib effective area.
    nm_float4 geometry; // Pa s2/m3, Pa s2/m3, m2, m2
    // Rest pleural pressure, dry atmospheric pressure, STPD/BTPS, dt.
    nm_float4 environment; // Pa, Pa, dimensionless, s
    // Inspired O2/CO2 fractions, O2 demand and CO2 production (STPD m3/s).
    nm_float4 metabolism;
    // Hb oxygen capacity, dissolved O2 slope per mmHg, P50, Hill exponent.
    nm_float4 oxygen;
    // CO2 content at 40mmHg, content slope /mmHg, capillary transit factor, 0.
    nm_float4 carbonDioxide;
    // Four systemic venous destination rows and metabolic fractions.
    nm_uint4 tissueRows;
    nm_float4 tissueFractions;
    // pulmonary exchange edge, arterial sensing row, compartments, edges.
    nm_uint4 topology;
    MRMujocoMuscleGPU muscles[2]; // diaphragm, inspiratory intercostals
} NMHumanRespirationParameters;

typedef struct NM_ALIGN16 NMHumanBloodBalanceLedger {
    float continuityResidualSumM3;
    float continuityResidualCompensationM3;
    float physicalVolumeDeltaSumM3;
    float physicalVolumeDeltaCompensationM3;
} NMHumanBloodBalanceLedger;

// One source-bound transection event. The anatomy and crossing-proof hashes
// identify source data only; they do not claim a real individual or a verified
// lumen intersection. The caller may set active only after the actual cut-plane
// and source-mesh crossing has been checked by the Human anatomy owner.
typedef struct NM_ALIGN16 NMHumanVascularCutEventGPU {
    nm_uint4 identity; // source anatomy stable ID, donor compartment stable ID, runtime body index, active
    nm_float4 aperture; // effective area m2, outward unit normal xyz in source-body frame
    nm_float4 fluid; // density kg/m3, discharge coefficient, ambient gauge pressure Pa, donor row index
    nm_float4 sourcePointAndGround; // source-body local outlet xyz, existing ground-plane z (world)
    nm_u64 sourceIdentity[4]; // SHA-256 of the exact source anatomy member
    nm_u64 crossingProofIdentity[4]; // SHA-256 of the exact outer-surface section proof
} NMHumanVascularCutEventGPU;

#define NM_HUMAN_EXTERNAL_BLOOD_PARCEL_CAPACITY 256u

typedef struct NM_ALIGN16 NMHumanExternalBloodParcelGPU {
    nm_float4 positionAndVolume; // world position xyz, blood volume m3
    nm_float4 velocityAndMass; // world velocity xyz, mass kg
    nm_float4 gasAndFlags; // O2 amount STPD m3, CO2 amount STPD m3, active flag, reserved
    nm_float4 intrinsicAngularMomentum; // kg m2/s, retained by the batched point parcel
} NMHumanExternalBloodParcelGPU;

// Candidate-only reduced ballistic packets and a grounded terminal sink. This
// is not a free-surface fluid solver and has no body-contact formulation.
typedef struct NM_ALIGN16 NMHumanExternalBloodInventoryGPU {
    nm_float4 total; // external volume m3, mass kg, O2 amount STPD m3, CO2 amount STPD m3
    nm_float4 compensation; // Kahan compensation for volume, mass, O2, CO2
    nm_float4 grounded; // grounded volume m3, mass kg, O2 amount STPD m3, CO2 amount STPD m3
    nm_float4 groundedCompensation;
    nm_float4 jetImpulse; // accumulated requested body reaction impulse N s (xyz), reserved
    nm_float4 jetTorqueImpulse; // cumulative relative-jet reaction angular impulse N m s
    nm_float4 pendingParcel; // volume m3, mass kg, O2 and CO2 STPD m3 before 8 ms packetization
    nm_float4 pendingMomentum; // pending external linear momentum kg m/s xyz
    nm_float4 pendingPositionMass; // mass-weighted outlet position xyz, pending mass kg
    nm_float4 pendingAngularMomentum; // pending external angular momentum about world origin kg m2/s
    nm_float4 energyLedger; // pending kinetic J, cumulative merge dissipation J, compensation, reserved
    nm_float4 groundReactionImpulse; // cumulative floor-sink impulse N s
    nm_float4 groundReactionAngularImpulse; // cumulative floor-sink angular impulse N m s
    nm_float4 groundImpactDissipation; // cumulative floor-sink kinetic energy J
    nm_float4 bodyMassBalance; // reference positive body mass kg, candidate positive body mass kg, external blood mass kg, residual kg
    nm_float4 bodyMassBalanceMax; // maximum absolute mass residual kg, reserved
    nm_float4 transferMomentumResidual; // algebraic removed body momentum minus parcel and requested recoil (kg m/s)
    nm_float4 transferAngularResidual; // angular momentum balance residual about world origin (kg m2/s)
    NMHumanExternalBloodParcelGPU parcels[NM_HUMAN_EXTERNAL_BLOOD_PARCEL_CAPACITY];
} NMHumanExternalBloodInventoryGPU;

typedef struct NM_ALIGN16 NMHumanRespirationState {
    // Diaphragm/rib swept volume and velocities, SI.
    nm_float4 motion;
    // Total lung volume, alveolar pressure, pleural pressure, mouth airflow.
    nm_float4 mechanics;
    MRMujocoMuscleStateGPU muscles[2];
    // O2 and CO2 amounts followed by Kahan residuals.
    nm_float4 bloodGas[21];
    nm_float4 alveolarGas;
    nm_float4 deadSpaceGas;
    // Cumulative inspired minus expired O2/CO2, with Kahan residuals.
    nm_float4 environmentGas;
    // Cumulative consumed O2 and produced CO2, with Kahan residuals.
    nm_float4 metabolicGas;
    // Initial total O2/CO2, maximum absolute conservation defect of each (STPD m3).
    nm_float4 gasBudget;
    // PaO2 mmHg, PaCO2 mmHg, SaO2, pulmonary O2 uptake STPD m3/s.
    nm_float4 observation;
    // Accepted muscle excitations xy, CPG frequency bpm z, VE setpoint L/min w.
    nm_float4 control;
    // LV, RV, aortic and pulmonary arterial pressures, Pa, from the circuit.
    nm_float4 cardiacPressure;
    // Cumulative aortic/pulmonary ejection, current/last LV stroke volume, m3.
    nm_float4 cardiacFlow;
    // Total blood, LV/RV chamber volume, maximum absolute blood error, m3.
    nm_float4 circulation;
    // Right atrium, right ventricle, left atrium, left ventricle, m3.
    // These are the actual four source chamber volumes used for geometry.
    nm_float4 chamberVolumes;
    // Completed LV filling/ejection cycles, filling-seen, outlet-open, reserved.
    nm_uint4 cardiacStatus;
    // Max/min lung volume this cycle, last tidal volume, integrated inspiration.
    nm_float4 breath;
    // accepted steps, complete breaths, previous inspiration flag, failure code.
    nm_uint4 status;
    // Compensated accepted-step ledgers separate global continuity residual,
    // rounded physical volume delta, and endpoint summation rounding.
    NMHumanBloodBalanceLedger bloodBalance;
    // Accepted inspiration event step, prior event step, interval steps,
    // reserved. Events use accepted airflow with 0.001 ml/s hysteresis;
    // near-zero round-off sign chatter is not an additional breath.
    nm_uint4 breathTiming;
    // Last event cumulative inspiration, cumulative Kahan compensation,
    // last complete event-interval inspiration, last event compensation (m3).
    // Observer bookkeeping only; these fields do not drive physiology.
    nm_float4 breathAccounting;
    // Net swept air volume in this accepted physical step, its volume-balance
    // residual (m3), maximum blood donor fraction, and gas substep count.
    // Endpoint airflow/pressures remain in mechanics; this records its integral.
    nm_float4 transportStep;
    // Same accepted candidate/resolve transaction as CVSim21; rejected cut
    // events and their outlet transfers remain absent from accepted state.
    NMHumanVascularCutEventGPU vascularCutOutlet;
    NMHumanExternalBloodInventoryGPU externalBlood;
} NMHumanRespirationState;

typedef struct NM_ALIGN16 NMHumanRespirationDispatch {
    nm_u32 environmentCount;
    nm_u32 vascularStride;
    nm_u32 reject; // test-only explicit rejected candidate; never changes state
    nm_u32 outletReleaseStep; // same control step as NUMI_HUMAN_SEPARATION_STEP; INVALID disables
    nm_u32 controlStep;
    nm_u32 reserved[3];
    nm_u32 articulationFirstBody;
    nm_u32 bodyPoseStride;
    nm_u32 bodyJacobianPointOffset;
    nm_u32 pointJacobianStride;
    nm_u32 dofCount;
    nm_u32 vStride;
    nm_u32 generalizedForceOffset;
    nm_u32 generalizedForceStride;
    nm_u32 bodyPropertyCount;
    nm_u32 outletBatchSteps; // minimum candidate-step interval between reduced ballistic packets
} NMHumanRespirationDispatch;

typedef struct NM_ALIGN16 NMHumanRespirationBrainDispatch {
    nm_u64 sourceTimeMicroseconds;
    nm_u64 targetTimeMicroseconds;
    nm_u64 sourceRootIdentity;
    nm_u64 targetRootIdentity;
    nm_u32 sourceStep;
    nm_u32 environmentCount;
    float driveScale;
    nm_u32 reserved;
} NMHumanRespirationBrainDispatch;
