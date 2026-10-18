#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <numi/matter/human_physiology.hpp>
#include <numi/matter/matter.hpp>
#include <metalrobo/engine_types.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace nm = numi::matter;

static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

static void printStatus(const NMMatterStatusGPU& s) {
    std::cout << "status_code=" << s.code
              << " environment=" << s.environment
              << " object_index=" << s.objectIndex
              << " failing_index=" << s.failingIndex
              << " completed_microsteps=" << s.completedMicrosteps
              << " fgmres_iterations=" << s.fgmresIterations
              << " contact_count=" << s.contactCount
              << " diagnostic_x=" << s.diagnostics.x
              << " diagnostic_y=" << s.diagnostics.y
              << " diagnostic_z=" << s.diagnostics.z
              << " diagnostic_w=" << s.diagnostics.w;
}

static std::vector<nm_float4> pureGenericState;
static std::vector<nm_float4> pureDenseState;
static std::vector<nm_float4> dormantMPMState;

static double maxStateDifference(const std::vector<nm_float4>& a,
                                const std::vector<nm_float4>& b) {
    if (a.size() != b.size()) return INFINITY;
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result = std::max({result, std::abs(double(a[i].x) - double(b[i].x)),
            std::abs(double(a[i].y) - double(b[i].y)),
            std::abs(double(a[i].z) - double(b[i].z)),
            std::abs(double(a[i].w) - double(b[i].w))});
    }
    return result;
}


template<class T> static bool same(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
static void physicalEqual(const nm::RuntimeStateSnapshot& a,const nm::RuntimeStateSnapshot& b,const char* label) {
#define CHECK(field) require(same(a.field,b.field), std::string(label)+" changed " #field)
    CHECK(particles); CHECK(particleMaterialState); CHECK(vascularState); CHECK(vascularClock);
    CHECK(contactHistories); CHECK(deformableContactHistories); CHECK(topologyStates); CHECK(schedulers); CHECK(adaptive); CHECK(environmentParameters); CHECK(identification);
#undef CHECK
}
struct Callback { unsigned count[4]{}; };
static bool extension(void* ctx,nm::AcceptedStepExtensionPhase phase,const nm::AcceptedStepExtensionView&) {
    auto& c=*static_cast<Callback*>(ctx); const auto i=static_cast<unsigned>(phase);
    if(i>=4) return false; ++c.count[i]; return true;
}
static void runMode(const nm::CompiledWorld& world,bool,const char*) {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice(); require(device!=nil,"no GPU");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLBuffer> status=[device newBufferWithLength:sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
    *static_cast<MRMetalWorldStatusGPU*>(status.contents)={};
    static_cast<MRMetalWorldStatusGPU*>(status.contents)->code=MR_STEP_SUCCESS;
    struct Gate { uint32_t code,environment,completed; };
    id<MTLBuffer> gate=[device newBufferWithLength:sizeof(Gate) options:MTLResourceStorageModeShared];
    nm::Runtime runtime; nm::RuntimeConfiguration cfg; cfg.metallib=NUMI_MATTER_METALLIB;
    cfg.environmentCount=1; cfg.captureEvents=true; cfg.captureDiagnostics=true; cfg.adaptiveTransfer=false;
    const auto& d=world.dispatch;
    std::cout<<"scope gridNodes="<<d.gridNodeCount<<" mpmBlocks="<<d.mpmBlockCount
    <<" params="<<d.parameterCount<<" ID="<<d.identificationCandidateCount<<" topologyCapacity="<<d.topologyNodeCapacity
    <<" deformableCapacity="<<d.deformableContactCapacity<<" contacts="<<d.contactPairCount<<" flags="<<d.flags
    <<" eventStride="<<d.eventStride<<" learned="<<d.learnedMaterialCount<<" mutations="<<d.mutationCommandCount<<"\n";
    if (std::getenv("NUMI_INSPECT_ONLY")) return;
    auto init=runtime.initialize(world,cfg); require(init.encoded,init.message);
    std::cout<<"device="<<init.device<<" fingerprint="<<runtime.fingerprint()<<"\n";
    const auto initial=runtime.snapshot(); require(initial.available,initial.message);
    auto step=[&](bool accept) {
        *static_cast<Gate*>(gate.contents)={accept?0u:99u,0u,1u};
        Callback context; id<MTLCommandBuffer> cb=[queue commandBuffer];
        nm::EncodeRequest r{}; r.commandBuffer=(__bridge void*)cb; r.environmentStatuses=(__bridge void*)status;
        r.physicsSubsteps=1; r.timestepSeconds=runtime.timestepSeconds(); r.runAdaptiveTransfer=false;
        r.acceptedStepExtensionContext=&context; r.encodeAcceptedStepExtension=&extension;
        r.deferAcceptedControlCursorPublication=true; r.deferAcceptedStepExtensionFinalization=true;
        r.phase=nm::EncodePhase::preDynamics; auto d=runtime.encode(r); require(d.encoded,d.message);
        r.phase=nm::EncodePhase::postCommit; d=runtime.encode(r); require(d.encoded,d.message);
        nm::AcceptedStepFinalizationGate g{}; g.statusRecords=(__bridge void*)gate; g.recordCount=1;
        g.recordStrideBytes=sizeof(Gate); g.completedCountOffsetBytes=8; g.expectedCompletedCount=1;
        g.environmentIndexOffsetBytes=4;
        d=runtime.finalizeAcceptedStepExtension((__bridge void*)cb,0,0,g); require(d.encoded,d.message);
        nm::AcceptedControlCursorGate cg{}; cg.statusCodes=(__bridge void*)gate; cg.recordCount=1; cg.recordStrideBytes=sizeof(Gate);
        d=runtime.encodeAcceptedControlCursorPublication((__bridge void*)cb,0,0,cg); require(d.encoded,d.message);
        [cb commit]; [cb waitUntilCompleted]; require(cb.status==MTLCommandBufferStatusCompleted,"GPU failure");
        auto snap=runtime.snapshot(); require(snap.available,snap.message); require(!snap.statuses.empty(),"missing status");
        std::cout<<"accept="<<accept<<" code="<<snap.statuses[0].code<<" microsteps="<<snap.statuses[0].completedMicrosteps
                 <<" GPU_seconds="<<(cb.GPUEndTime-cb.GPUStartTime)<<" callbacks="<<context.count[0]<<","<<context.count[1]<<","<<context.count[2]<<","<<context.count[3]<<"\n";
        require(context.count[0]==1 && context.count[3]==1,"frame callback count");
        require(accept ? snap.statuses[0].code==NM_STATUS_SUCCESS : snap.statuses[0].code!=NM_STATUS_SUCCESS,"wrong acceptance");
        return snap;
    };
    const auto rejected=step(false); physicalEqual(initial,rejected,"external rejection");
    require(initial.controlStep==rejected.controlStep && initial.physicsSubstep==rejected.physicsSubstep,"reject advanced cursor");
    const auto retried=step(true); require(!same(initial.particles,retried.particles),"accepted particle did not move");
    require(!same(initial.vascularClock,retried.vascularClock),"accepted circulation clock did not advance");
    auto restored=runtime.restore(initial); require(restored.encoded,restored.message);
    const auto clean=step(true); physicalEqual(retried,clean,"retry vs restored clean acceptance");
    std::cout<<"PASS external_gate_full_rollback_retry_restore\n";
}

int main() {
    @autoreleasepool {
        try {
            const std::filesystem::path networkPath =
                "/Users/numi/numi-human-resting-evidence-20261005/reference-circulation-001/resting_reference_lv15.native.v3.json";
            nm::VascularNetworkSource vascular;
            std::string error;
            require(nm::readHumanPhysiologyNetwork(networkPath, vascular, &error), error);

            nm::WorldSource source;
            source.environmentCount = 1u;
            source.frameTimestep = 0.00025;
            source.gravity = {0.0, 0.0, 0.0};
            source.vascular = vascular;
            source.mixedSolver.relativeResidual = 1.0e-7;
            source.mixedSolver.newtonIterations = 12u;
            source.mixedSolver.fgmresIterations = 64u;

            const auto compiled = nm::compileWorld(source, {.maximumRateExponent = NM_MAX_RATE_EXPONENT});
            for (const auto& d : compiled.diagnostics) {
                std::cout << "compile_diag_severity=" << static_cast<unsigned>(d.severity)
                          << " compile_diag=" << d.message << "\n";
            }
            require(compiled.succeeded(), "objectless pure-vascular compile failed");
            const auto& world = compiled.world;
            std::cout << "source_identity=";
            for (const auto x : source.vascular.sourceIdentity) std::cout << std::hex << x << ":";
            std::cout << std::dec << " world_fingerprint=" << world.fingerprint
                      << " env=" << world.dispatch.environmentCount
                      << " objects=" << world.dispatch.objectCount
                      << " materials=" << world.dispatch.materialCount
                      << " particles=" << world.dispatch.particleCount
                      << " fem_nodes=" << world.dispatch.femNodeCount
                      << " tets=" << world.dispatch.tetrahedronCount
                      << " mpm_nodes=" << world.dispatch.mpmActiveNodeCapacity
                      << " mpm_grids=" << world.dispatch.mpmGridCount
                      << " max_rate_exponent=" << world.dispatch.maximumRateExponent
                      << " compartments=" << world.vascular.compartments.size()
                      << " connections=" << world.vascular.connections.size()
                      << " unknowns=" << world.vascular.unknowns.size()
                      << " dt_s=" << source.frameTimestep
                      << "\n";
            require(world.dispatch.environmentCount == 1u &&
                    world.dispatch.objectCount == 0u &&
                    world.dispatch.materialCount == 0u &&
                    world.dispatch.particleCount == 0u &&
                    world.dispatch.femNodeCount == 0u &&
                    world.dispatch.tetrahedronCount == 0u &&
                    world.dispatch.mpmActiveNodeCapacity == 0u &&
                    world.dispatch.mpmGridCount == 0u &&
                    world.vascular.compartments.size() == 21u &&
                    world.vascular.connections.size() == 24u &&
                    world.vascular.unknowns.size() == 45u &&
                    world.dispatch.maximumRateExponent == NM_MAX_RATE_EXPONENT &&
                    std::abs(source.frameTimestep - 0.00025f) < 1.0e-9f,
                    "compiled world is not the intended pure 21/24/45 graph at dt=0.25ms");



            auto parsed = nm::parseMatterFile(NUMI_MATTER_MATERIAL);
            require(parsed.succeeded(), "dormant MPM material failed to parse");
            nm::WorldSource mpmSource = source;
            mpmSource.materials.push_back(parsed.material);
            nm::ObjectSource dormant;
            dormant.name = "dormant_mpm_capacity_control";
            dormant.materialIndex = 0u;
            dormant.representation = nm::Representation::mpm;
            dormant.automaticRepresentation = false;
            dormant.twoWayCoupling = false;
            dormant.adaptive = false;
            dormant.deformableContact = false;
            dormant.deformableSelfContact = false;
            dormant.characteristicLength = 0.005;
            dormant.mpmGridMinimum = {-0.01, -0.01, -0.01};
            dormant.mpmGridMaximum = {0.01, 0.01, 0.01};
            nm::ParticleSource reserve;
            reserve.position = {0.0, 0.0, 0.0};
            reserve.velocity = {0.1, 0.0, 0.0};
            reserve.mass = 1.0506e-3;
            reserve.referenceVolume = 1.0e-6;
            reserve.active = true;
            dormant.particles.push_back(reserve);
            mpmSource.objects.push_back(std::move(dormant));
            const auto mpmCompiled = nm::compileWorld(
                mpmSource, {.maximumRateExponent = NM_MAX_RATE_EXPONENT});
            for (const auto& d : mpmCompiled.diagnostics) {
                std::cout << "mpm_compile_diag_severity=" << static_cast<unsigned>(d.severity)
                          << " mpm_compile_diag=" << d.message << "\n";
            }
            require(mpmCompiled.succeeded(), "dormant MPM plus vascular compile failed");
            const auto& mpmWorld = mpmCompiled.world;
            std::uint32_t activeParticleCount = 0u;
            for (const auto& particle : mpmWorld.mpm.particles) {
                if ((particle.identity.w & NM_OBJECT_ACTIVE) != 0u) ++activeParticleCount;
            }
            std::cout << "dormant_world_fingerprint=" << mpmWorld.fingerprint
                      << " env=" << mpmWorld.dispatch.environmentCount
                      << " objects=" << mpmWorld.dispatch.objectCount
                      << " materials=" << mpmWorld.dispatch.materialCount
                      << " particles=" << mpmWorld.dispatch.particleCount
                      << " mpm_cooked_particles=" << mpmWorld.mpm.particles.size()
                      << " active_particles=" << activeParticleCount
                      << " mpm_nodes=" << mpmWorld.dispatch.mpmActiveNodeCapacity
                      << " mpm_grid_nodes=" << mpmWorld.dispatch.gridNodeCount
                      << " mpm_grids=" << mpmWorld.dispatch.mpmGridCount
                      << " fem_nodes=" << mpmWorld.dispatch.femNodeCount
                      << " tets=" << mpmWorld.dispatch.tetrahedronCount
                      << " compartments=" << mpmWorld.vascular.compartments.size()
                      << " connections=" << mpmWorld.vascular.connections.size()
                      << " unknowns=" << mpmWorld.vascular.unknowns.size()
                      << " max_rate_exponent=" << mpmWorld.dispatch.maximumRateExponent
                      << " dt_s=" << mpmSource.frameTimestep << "\n";
            require(mpmWorld.dispatch.environmentCount == 1u &&
                    mpmWorld.dispatch.objectCount == 1u &&
                    mpmWorld.dispatch.materialCount == 1u &&
                    mpmWorld.dispatch.particleCount == 1u &&
                    mpmWorld.mpm.particles.size() == 1u &&
                    activeParticleCount == 1u &&
                    mpmWorld.dispatch.femNodeCount == 0u &&
                    mpmWorld.dispatch.tetrahedronCount == 0u &&
                    mpmWorld.dispatch.maximumRateExponent == NM_MAX_RATE_EXPONENT &&
                    mpmWorld.vascular.compartments.size() == 21u &&
                    mpmWorld.vascular.connections.size() == 24u &&
                    mpmWorld.vascular.unknowns.size() == 45u &&
                    std::abs(mpmSource.frameTimestep - 0.00025) < 1.0e-12,
                    "compiled world is not a dormant-MPM object plus the same vascular graph");
            runMode(mpmWorld, false, "dormant_mpm_generic_fullcap");
            std::cout << "comparison pure_generic_vs_dense45_max_abs_normalized_state="
                      << maxStateDifference(pureGenericState, pureDenseState) << "\n";
            std::cout << "comparison pure_generic_vs_dormant_mpm_max_abs_normalized_state="
                      << maxStateDifference(pureGenericState, dormantMPMState) << "\n";
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            return 2;
        }
    }
}
