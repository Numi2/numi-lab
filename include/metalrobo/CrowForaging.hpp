#pragma once
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/FiniteResource.hpp"
#include "metalrobo/VisualPlatform.hpp"
#include <array>

namespace metalrobo {
// Engineering development fixture, not measured crow eyes or biological food.
// Authoring only: execution remains in the common world/resource/sensor owners.
inline constexpr unsigned crowForagingCameraWidth=160, crowForagingCameraHeight=120;
inline constexpr unsigned crowForagingCueEndTick=32, crowForagingIntakeStartTick=192;

struct CrowForagingVisual {
    WorldFamily family;
    std::vector<SensorSpec> sensors;
    VisualSceneManifestV3 manifest;
    unsigned head=MR_INVALID_INDEX,target=MR_INVALID_INDEX;
    std::vector<MRFiniteResourceCueGPU> resourceCues;
};

[[nodiscard]] RunManifest makeAmericanCrowForagingFixtureManifest(bool occluded);
// Writes collision-proxy packs plus optional authored visual-only port
// indicators. Paths are immutable per run; existing files are rejected.
[[nodiscard]] CrowForagingVisual authorAmericanCrowForagingVisual(
    const CompiledRun& run,const std::filesystem::path& output,bool foodPortIndicators=false,
    mr_float4 cameraOrientation={-.5f,.5f,-.5f,.5f});

struct CrowCuedForagingProgram {
    FiniteResourceConfig resources;
    VisualSensorProgram sensor;
};
[[nodiscard]] FiniteResourceConfig authorAmericanCrowCuedForagingResources(
    const CompiledRun& physicalRun,const std::array<mr_float4,2>& calibratedPorts,
    unsigned environments,std::uint64_t allocationSeed);
void applyAmericanCrowForagingFixtureLimits(RunManifest& manifest);
// Staged experiment: neutral motor/history before decision; afterwards only
// head.yaw and beak.open act. This is not a free-locomotion task.
void applyAmericanCrowMemoryChoiceCurriculum(TaskPack& task,unsigned decisionStep,unsigned headHold=0,unsigned beakHold=0);
// Bind once, before compiling the retinal run. Port centers must come from an
// independently admitted physical calibration in the actual ground frame.
// The returned resource program fingerprint must be bound into TaskPack;
// allocationSeed is runtime life identity and never an actor feature.
[[nodiscard]] CrowCuedForagingProgram authorAmericanCrowCuedForagingProgram(
    const CompiledRun& physicalRun,const CrowForagingVisual& visual,
    const std::array<mr_float4,2>& calibratedPorts,unsigned environments,
    std::uint64_t allocationSeed);
// The existing retinal task ABI with its exact authored resource fingerprint.
// The task ID alone never identifies the environment: all native fingerprints
// remain mandatory for policy/checkpoint admission.
[[nodiscard]] TaskPack makeAmericanCrowCuedForagingTaskPack(
    TaskObservationProgram& observations,TaskResetProgram& reset,
    const CrowCuedForagingProgram& authored);
} // namespace metalrobo
