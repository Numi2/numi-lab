#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/HeterogeneousWorld.hpp"
#include <limits>
#include "metalrobo/MetalWorld.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
mr_float4 f4(float x, float y, float z, float w=0.f) { return {x,y,z,w}; }
MRBodyStateGPU staticBody() {
    MRBodyStateGPU body{}; body.position.w=1.f; body.orientation.w=1.f;
    body.flagsAndIndices[0]=MR_MOTION_STATIC; body.flagsAndIndices[1]=MR_INVALID_INDEX;
    return body;
}
template <class T> void save(const std::filesystem::path& path, const std::vector<T>& data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    check(f.good(), "output open failed");
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()*sizeof(T)));
    f.close(); check(f.good(), "output write failed");
}
template <class T> bool equal(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
metalrobo::MetalWorldStepConfig fixtureConfig() {
    metalrobo::MetalWorldStepConfig config;
    config.timestepSeconds=1.f/120.f;
    config.physicsSubsteps=1u;
    config.solverMode=metalrobo::MetalWorldSolverMode::temporalCone;
    config.ccdMode=metalrobo::MetalWorldCCDMode::speculative;
    config.velocityIterations=8u; config.finalVelocityIterations=4u;
    config.deterministic=true; config.warmStart=true;
    config.captureContactEvidence=true; config.captureConvexCacheEvidence=true; config.captureRodWitnessEvidence=true;
    return config;
}
struct Trial { metalrobo::MetalWorldResult first, second; };
Trial runTrial(const metalrobo::CompiledWorld& world, const std::vector<float>& initialQ,
    const std::vector<float>& initialV, const std::filesystem::path& out, int trial) {
    metalrobo::MetalWorldContext context;
    metalrobo::MetalWorldResidentState resident;
    const std::vector<float> efforts(world.nv(),0.f);
    auto floor=staticBody(); floor.position.y=-1.f; const std::vector<MRBodyStateGPU> scene{floor};
    auto config=fixtureConfig();
    metalrobo::MetalWorldBatch firstBatch{.environmentCount=1u,.controlStepCount=1u,
        .initialQ=initialQ,.initialV=initialV,.efforts=efforts,.initialSceneBodies=scene};
    metalrobo::MetalWorldSubmission firstSubmission;
    const auto init=context.initializeResidentState(world,firstBatch,config,resident,firstSubmission);
    std::cout<<"trial="<<trial<<" phase=init_submit status="<<static_cast<int>(init.status)<<" message="<<init.message<<std::endl;
    check(init.succeeded() && firstSubmission.valid(), "initial submit failed");
    Trial result;
    const auto first=firstSubmission.wait(result.first);
    std::cout<<"trial="<<trial<<" phase=init_wait status="<<static_cast<int>(first.status)<<" failed="<<first.failedStepCount<<" message="<<first.message<<std::endl;
    save(out/("trial"+std::to_string(trial)+"-first-cache.bin"), result.first.contactEvidence.convexCaches);
    save(out/("trial"+std::to_string(trial)+"-first-q.bin"), result.first.finalQ);
    save(out/("trial"+std::to_string(trial)+"-first-scene.bin"), result.first.finalSceneBodies);
    save(out/("trial"+std::to_string(trial)+"-first-rod-nodes.bin"), result.first.finalRodNodes);
    save(out/("trial"+std::to_string(trial)+"-first-rod-edges.bin"), result.first.finalRodEdges);
    save(out/("trial"+std::to_string(trial)+"-first-rod-witnesses.bin"), result.first.contactEvidence.rodToolWitnesses);
    save(out/("trial"+std::to_string(trial)+"-first-rod-status.bin"), result.first.rodStatuses);
    save(out/("trial"+std::to_string(trial)+"-first-world-status.bin"), result.first.statuses);
    save(out/("trial"+std::to_string(trial)+"-first-contact-status.bin"), result.first.contactStatuses);
    save(out/("trial"+std::to_string(trial)+"-first-v.bin"), result.first.finalV);
    save(out/("trial"+std::to_string(trial)+"-first-manifold-headers.bin"), result.first.contactEvidence.manifoldHeaders);
    save(out/("trial"+std::to_string(trial)+"-first-manifold-points.bin"), result.first.contactEvidence.manifoldPoints);
    save(out/("trial"+std::to_string(trial)+"-first-manifold-counts.bin"), result.first.contactEvidence.manifoldCounts);
    check(first.succeeded() && first.failedStepCount==0 && resident.valid(), "initial state failed");
    metalrobo::MetalWorldBatch next{.environmentCount=1u,.controlStepCount=1u,.efforts=efforts};
    metalrobo::MetalWorldSubmission secondSubmission;
    const auto submitted=context.submitResident(world,next,config,resident,secondSubmission);
    std::cout<<"trial="<<trial<<" phase=continue_submit status="<<static_cast<int>(submitted.status)<<" message="<<submitted.message<<std::endl;
    check(submitted.succeeded() && secondSubmission.valid(), "continue submit failed");
    const auto second=secondSubmission.wait(result.second);
    std::cout<<"trial="<<trial<<" phase=continue_wait status="<<static_cast<int>(second.status)<<" failed="<<second.failedStepCount<<" message="<<second.message<<std::endl;
    save(out/("trial"+std::to_string(trial)+"-second-cache.bin"), result.second.contactEvidence.convexCaches);
    save(out/("trial"+std::to_string(trial)+"-second-q.bin"), result.second.finalQ);
    save(out/("trial"+std::to_string(trial)+"-second-scene.bin"), result.second.finalSceneBodies);
    save(out/("trial"+std::to_string(trial)+"-second-rod-nodes.bin"), result.second.finalRodNodes);
    save(out/("trial"+std::to_string(trial)+"-second-rod-edges.bin"), result.second.finalRodEdges);
    save(out/("trial"+std::to_string(trial)+"-second-rod-witnesses.bin"), result.second.contactEvidence.rodToolWitnesses);
    save(out/("trial"+std::to_string(trial)+"-second-rod-status.bin"), result.second.rodStatuses);
    save(out/("trial"+std::to_string(trial)+"-second-world-status.bin"), result.second.statuses);
    save(out/("trial"+std::to_string(trial)+"-second-contact-status.bin"), result.second.contactStatuses);
    save(out/("trial"+std::to_string(trial)+"-second-v.bin"), result.second.finalV);
    save(out/("trial"+std::to_string(trial)+"-second-manifold-headers.bin"), result.second.contactEvidence.manifoldHeaders);
    save(out/("trial"+std::to_string(trial)+"-second-manifold-points.bin"), result.second.contactEvidence.manifoldPoints);
    save(out/("trial"+std::to_string(trial)+"-second-manifold-counts.bin"), result.second.contactEvidence.manifoldCounts);
    check(second.succeeded() && second.failedStepCount==0 && resident.valid(), "continued state failed");
    return result;
}
}
int main(int argc, char** argv) {
    try {
        check(argc==2, "expected output directory");
        const std::filesystem::path out=argv[1];
        check(std::filesystem::create_directory(out), "output already exists or cannot be created");
        auto model=metalrobo::makeFreeSphereEngineModel();
        model.shapes[1].dimensions.x=0.002f;
        model.shapes[1].contactRestAndBoundingRadius={2.0e-5f,0.f,0.002f,0.f};
        model.world.gravityAndTimestep={0.f,0.f,0.f,1.f/120.f};
        model.defaultQ[0]=0.06f; model.defaultQ[1]=-0.0019f; model.defaultQ[2]=0.f;
        auto floor=staticBody(); floor.position.y=-1.f;
        const std::array<MRBodyStateGPU,1> scene{floor};
        const std::array<metalrobo::HeterogeneousWorldComponent,1> components{{{.model=&model,.instanceId="rod_seed_tool",.defaultSceneBodies=scene}}};
        metalrobo::HeterogeneousRodProgram rod;
        rod.instanceId="rod_seed_strand";
        rod.model=metalrobo::makeStraightSutureRod(5u,0.12);
        rod.defaultState=metalrobo::makeDiscreteElasticRodDefaultState(rod.model);
        rod.collision.enableToolCollision=true;
        const std::array<metalrobo::HeterogeneousRodProgram,1> rods{rod};
        metalrobo::HeterogeneousWorld source;
        const auto composed=metalrobo::composeHeterogeneousWorld(components,rods,source);
        check(composed.succeeded(),"compose failed");
        metalrobo::CompiledWorld world;
        const auto compiled=metalrobo::compileMetalWorld(source,world);
        check(compiled.succeeded() && world.fingerprint()==2573094028256544495ull,"fixture identity mismatch");
        auto q=source.model.defaultQ; auto v=source.model.defaultV;
        std::cout<<"world_fingerprint="<<world.fingerprint()<<" rod_pairs="<<world.rodToolPairs().size()<<std::endl;
        const auto a=runTrial(world,q,v,out,0);
        const auto b=runTrial(world,q,v,out,1);
        const auto& witnesses=a.first.contactEvidence.rodToolWitnesses;
        std::size_t valid=0, nonzero=0, firstValid=witnesses.size();
        for (std::size_t i=0;i<witnesses.size();++i) {
            const auto& w=witnesses[i];
            if ((w.featuresAndFlags.w & MR_ROD_TOOL_WITNESS_VALID)==0u) continue;
            if (firstValid==witnesses.size()) firstValid=i;
            ++valid;
            if (w.impulses.x!=0.f || w.impulses.y!=0.f || w.impulses.z!=0.f || w.impulses.w!=0.f) ++nonzero;
        }
        const auto sameState=[](const metalrobo::MetalWorldResult& x,const metalrobo::MetalWorldResult& y) {
            return equal(x.finalQ,y.finalQ) && equal(x.finalV,y.finalV) && equal(x.finalSceneBodies,y.finalSceneBodies) &&
                equal(x.finalRodNodes,y.finalRodNodes) && equal(x.finalRodEdges,y.finalRodEdges) &&
                equal(x.contactEvidence.rodToolWitnesses,y.contactEvidence.rodToolWitnesses) &&
                equal(x.contactEvidence.convexCaches,y.contactEvidence.convexCaches) &&
                equal(x.contactEvidence.manifoldHeaders,y.contactEvidence.manifoldHeaders) &&
                equal(x.contactEvidence.manifoldPoints,y.contactEvidence.manifoldPoints) &&
                equal(x.contactEvidence.manifoldCounts,y.contactEvidence.manifoldCounts) &&
                equal(x.rodStatuses,y.rodStatuses) && equal(x.statuses,y.statuses) && equal(x.contactStatuses,y.contactStatuses);
        };
        const bool repeat=sameState(a.first,b.first) && sameState(a.second,b.second);
        std::cout<<"witness_capacity="<<witnesses.size()<<" valid="<<valid<<" nonzero_impulse="<<nonzero<<" independent_repeat="<<repeat<<std::endl;
        check(valid>0 && nonzero>0 && repeat,"rod witness population/repeat gate failed");
        metalrobo::MetalWorldContext seedContext;
        metalrobo::MetalWorldResidentState seeded;
        auto config=fixtureConfig();
        const auto& state=a.first;
        metalrobo::MetalWorldNoStepSeed seed{
            .worldFingerprint=state.residentBoundary.worldFingerprint,
            .configurationFingerprint=state.residentBoundary.configurationFingerprint,
            .q=state.finalQ,.v=state.finalV,.sceneBodies=state.finalSceneBodies,
            .manifoldHeaders=state.contactEvidence.manifoldHeaders,
            .manifoldPoints=state.contactEvidence.manifoldPoints,
            .manifoldCounts=state.contactEvidence.manifoldCounts,
            .convexCaches=state.contactEvidence.convexCaches, .rodNodes=state.finalRodNodes,
            .rodEdges=state.finalRodEdges, .rodWitnesses=state.contactEvidence.rodToolWitnesses};
        seed.payloadFingerprint=metalrobo::metalWorldNoStepSeedPayloadFingerprint(seed);
        auto wrongWorld=seed; wrongWorld.worldFingerprint^=1u;
        wrongWorld.payloadFingerprint=metalrobo::metalWorldNoStepSeedPayloadFingerprint(wrongWorld);
        auto bad=seedContext.seedResidentNoStep(world,config,wrongWorld,seeded);
        const bool worldReject=!bad.succeeded() && !seeded.valid();
        auto wrongConfig=config; wrongConfig.velocityIterations++;
        bad=seedContext.seedResidentNoStep(world,wrongConfig,seed,seeded);
        const bool configReject=!bad.succeeded() && !seeded.valid();
        auto truncated=seed; truncated.rodWitnesses=seed.rodWitnesses.subspan(0,seed.rodWitnesses.size()-1);
        truncated.payloadFingerprint=metalrobo::metalWorldNoStepSeedPayloadFingerprint(truncated);
        bad=seedContext.seedResidentNoStep(world,config,truncated,seeded);
        const bool lengthReject=!bad.succeeded() && !seeded.valid();
        auto badNodes=state.finalRodNodes;
        badNodes[0].position.x=std::numeric_limits<float>::quiet_NaN();
        auto invalidNode=seed; invalidNode.rodNodes=badNodes;
        invalidNode.payloadFingerprint=metalrobo::metalWorldNoStepSeedPayloadFingerprint(invalidNode);
        bad=seedContext.seedResidentNoStep(world,config,invalidNode,seeded);
        const bool nodeReject=!bad.succeeded() && !seeded.valid();
        auto changedWitnesses=state.contactEvidence.rodToolWitnesses;
        changedWitnesses[firstValid].identity.y ^= 1u;
        auto wrongPair=seed; wrongPair.rodWitnesses=changedWitnesses;
        wrongPair.payloadFingerprint=metalrobo::metalWorldNoStepSeedPayloadFingerprint(wrongPair);
        bad=seedContext.seedResidentNoStep(world,config,wrongPair,seeded);
        const bool pairReject=!bad.succeeded() && !seeded.valid();
        changedWitnesses=state.contactEvidence.rodToolWitnesses;
        changedWitnesses[firstValid].identity.w ^= 1u;
        auto wrongCollider=seed; wrongCollider.rodWitnesses=changedWitnesses;
        wrongCollider.payloadFingerprint=metalrobo::metalWorldNoStepSeedPayloadFingerprint(wrongCollider);
        bad=seedContext.seedResidentNoStep(world,config,wrongCollider,seeded);
        const bool colliderReject=!bad.succeeded() && !seeded.valid();
        changedWitnesses=state.contactEvidence.rodToolWitnesses;
        changedWitnesses[firstValid].materialAndGeneration.z ^= 1u;
        auto wrongTopology=seed; wrongTopology.rodWitnesses=changedWitnesses;
        wrongTopology.payloadFingerprint=metalrobo::metalWorldNoStepSeedPayloadFingerprint(wrongTopology);
        bad=seedContext.seedResidentNoStep(world,config,wrongTopology,seeded);
        const bool topologyReject=!bad.succeeded() && !seeded.valid();
        changedWitnesses=state.contactEvidence.rodToolWitnesses;
        changedWitnesses[firstValid].impulses.x=std::numeric_limits<float>::quiet_NaN();
        auto nonfinite=seed; nonfinite.rodWitnesses=changedWitnesses;
        nonfinite.payloadFingerprint=metalrobo::metalWorldNoStepSeedPayloadFingerprint(nonfinite);
        bad=seedContext.seedResidentNoStep(world,config,nonfinite,seeded);
        const bool finiteReject=!bad.succeeded() && !seeded.valid();
        changedWitnesses=state.contactEvidence.rodToolWitnesses;
        changedWitnesses[firstValid].rodPointAndWeight.x=std::numeric_limits<float>::quiet_NaN();
        auto nonfiniteGeometry=seed; nonfiniteGeometry.rodWitnesses=changedWitnesses;
        nonfiniteGeometry.payloadFingerprint=metalrobo::metalWorldNoStepSeedPayloadFingerprint(nonfiniteGeometry);
        bad=seedContext.seedResidentNoStep(world,config,nonfiniteGeometry,seeded);
        const bool geometryReject=!bad.succeeded() && !seeded.valid();
        changedWitnesses=state.contactEvidence.rodToolWitnesses;
        changedWitnesses[firstValid].featuresAndFlags.w |= 0x80000000u;
        auto badFlags=seed; badFlags.rodWitnesses=changedWitnesses;
        badFlags.payloadFingerprint=metalrobo::metalWorldNoStepSeedPayloadFingerprint(badFlags);
        bad=seedContext.seedResidentNoStep(world,config,badFlags,seeded);
        const bool flagsReject=!bad.succeeded() && !seeded.valid();
        changedWitnesses=state.contactEvidence.rodToolWitnesses;
        reinterpret_cast<unsigned char*>(changedWitnesses.data())[0]^=1u;
        auto corrupt=seed; corrupt.rodWitnesses=changedWitnesses;
        bad=seedContext.seedResidentNoStep(world,config,corrupt,seeded);
        const bool corruptReject=!bad.succeeded() && !seeded.valid();
        std::cout<<"seed_negative world="<<worldReject<<" config="<<configReject<<" length="<<lengthReject<<" node="<<nodeReject<<" pair="<<pairReject<<" collider="<<colliderReject<<" topology="<<topologyReject<<" finite="<<finiteReject<<" geometry="<<geometryReject<<" flags="<<flagsReject<<" corrupt="<<corruptReject<<std::endl;
        check(worldReject && configReject && lengthReject && nodeReject && pairReject && colliderReject && topologyReject && finiteReject && geometryReject && flagsReject && corruptReject,"negative admission failed");
        auto fault=seed; fault.testOnlyInjectReadbackMismatch=true;
        const auto faultResult=seedContext.seedResidentNoStep(world,config,fault,seeded);
        const bool faultRejected=!faultResult.succeeded() && !seeded.valid();
        std::cout<<"post_submit_readback_fault_rejected="<<faultRejected<<" status="<<static_cast<int>(faultResult.status)<<" message="<<faultResult.message<<std::endl;
        check(faultRejected, "post-submit mismatch published a destination");
        const auto admitted=seedContext.seedResidentNoStep(world,config,seed,seeded);
        std::cout<<"no_step_seed status="<<static_cast<int>(admitted.status)<<" message="<<admitted.message<<" valid="<<seeded.valid()<<std::endl;
        check(admitted.succeeded() && seeded.valid(), "no-step seed failed");
        const auto occupied=seedContext.seedResidentNoStep(world,config,seed,seeded);
        const bool occupiedRejected=!occupied.succeeded() && seeded.valid();
        std::cout<<"seed_occupied_rejected="<<occupiedRejected<<std::endl;
        check(occupiedRejected, "occupied destination was not preserved");
        const std::vector<float> efforts(world.nv(),0.f);
        metalrobo::MetalWorldBatch next{.environmentCount=1u,.controlStepCount=1u,.efforts=efforts};
        metalrobo::MetalWorldSubmission badContinuation;
        const auto changed=seedContext.submitResident(world,next,wrongConfig,seeded,badContinuation);
        const bool changedRejected=!changed.succeeded() && !badContinuation.valid() && seeded.valid();
        std::cout<<"seeded_changed_config_rejected="<<changedRejected<<" message="<<changed.message<<std::endl;
        check(changedRejected, "seeded changed-configuration continuation was not rejected");
        metalrobo::MetalWorldSubmission continuation;
        const auto submitted=seedContext.submitResident(world,next,config,seeded,continuation);
        check(submitted.succeeded() && continuation.valid(), "seeded continuation submit failed");
        metalrobo::MetalWorldResult restored;
        const auto completed=continuation.wait(restored);
        save(out/"seeded-second-cache.bin",restored.contactEvidence.convexCaches);
        save(out/"seeded-second-q.bin",restored.finalQ);
        save(out/"seeded-second-scene.bin",restored.finalSceneBodies);
        save(out/"seeded-second-rod-nodes.bin",restored.finalRodNodes);
        save(out/"seeded-second-rod-edges.bin",restored.finalRodEdges);
        save(out/"seeded-second-rod-witnesses.bin",restored.contactEvidence.rodToolWitnesses);
        save(out/"seeded-second-rod-status.bin",restored.rodStatuses);
        save(out/"seeded-second-world-status.bin",restored.statuses);
        save(out/"seeded-second-contact-status.bin",restored.contactStatuses);
        save(out/"seeded-second-v.bin",restored.finalV);
        save(out/"seeded-second-manifold-headers.bin",restored.contactEvidence.manifoldHeaders);
        save(out/"seeded-second-manifold-points.bin",restored.contactEvidence.manifoldPoints);
        save(out/"seeded-second-manifold-counts.bin",restored.contactEvidence.manifoldCounts);
        const bool matched=completed.succeeded() && completed.failedStepCount==0 &&
            equal(restored.contactEvidence.convexCaches,a.second.contactEvidence.convexCaches) &&
            equal(restored.finalQ,a.second.finalQ) && equal(restored.finalV,a.second.finalV) &&
            equal(restored.finalSceneBodies,a.second.finalSceneBodies) &&
            equal(restored.finalRodNodes,a.second.finalRodNodes) && equal(restored.finalRodEdges,a.second.finalRodEdges) &&
            equal(restored.contactEvidence.rodToolWitnesses,a.second.contactEvidence.rodToolWitnesses) &&
            equal(restored.rodStatuses,a.second.rodStatuses) && equal(restored.statuses,a.second.statuses) &&
            equal(restored.contactStatuses,a.second.contactStatuses) &&
            equal(restored.contactEvidence.manifoldHeaders,a.second.contactEvidence.manifoldHeaders) &&
            equal(restored.contactEvidence.manifoldPoints,a.second.contactEvidence.manifoldPoints) &&
            equal(restored.contactEvidence.manifoldCounts,a.second.contactEvidence.manifoldCounts);
        std::cout<<"seeded_continuation_match="<<matched<<" status="<<static_cast<int>(completed.status)<<" message="<<completed.message<<std::endl;
        check(matched, "seeded continuation did not match uninterrupted owner");
        std::cout<<"resident_rod_witness_no_step_seed=PASS coupled_restoration_claim=0"<<std::endl;
        return 0;
    } catch (const std::exception& e) { std::cerr<<"resident_rod_witness_fixture=FAIL message="<<e.what()<<std::endl; return 1; }
}
