#pragma once
#define ROD_N 9
#define ROD_E 8
struct RodInput {
 unsigned n;
 float edgeHigh[ROD_E*3], edgeLow[ROD_E*3];
 float vel[ROD_N*3], twist[ROD_E], omega[ROD_E], targetDelta[6], targetDeltaLow[6];
 float mass[ROD_N], inertia[ROD_E], ellHigh, ellLow, ea, ei, gj, h;
};
struct RodOutput {
 unsigned status, iterations, backtracks;
 float displacement[ROD_N*3], displacementLow[ROD_N*3], twistDelta[ROD_E], reaction[6];
 float maxLinear, maxAngular;
};
struct ResidentState {
 RodInput input;
 float sumHigh[ROD_N*3],sumLow[ROD_N*3],lastReaction[6];
 unsigned acceptedSteps;
};
struct StepControl { float targetHigh[6],targetLow[6]; unsigned inject; };
struct ResidentRecord { RodOutput output; ResidentState before,after; };
