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
    // Completed LV filling/ejection cycles, filling-seen, outlet-open, reserved.
    nm_uint4 cardiacStatus;
    // Max/min lung volume this cycle, last tidal volume, integrated inspiration.
    nm_float4 breath;
    // accepted steps, complete breaths, previous inspiration flag, failure code.
    nm_uint4 status;
} NMHumanRespirationState;

typedef struct NM_ALIGN16 NMHumanRespirationDispatch {
    nm_u32 environmentCount;
    nm_u32 vascularStride;
    nm_u32 reject; // test-only explicit rejected candidate; never changes state
    nm_u32 reserved;
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
