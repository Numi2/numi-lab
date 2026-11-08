#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/GeometryCooker.hpp"
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
struct Trial { metalrobo::MetalWorldResult first, second; };
Trial runTrial(const metalrobo::CompiledWorld& world, const std::vector<float>& initialQ,
    const std::vector<float>& initialV, const std::filesystem::path& out, int trial) {
    metalrobo::MetalWorldContext context;
    metalrobo::MetalWorldResidentState resident;
    const std::vector<float> efforts(world.nv(),0.f);
    const std::vector<MRBodyStateGPU> scene{staticBody()};
    metalrobo::MetalWorldStepConfig config;
    config.timestepSeconds=1.f/120.f;
    config.physicsSubsteps=1u;
    config.solverMode=metalrobo::MetalWorldSolverMode::temporalCone;
    config.ccdMode=metalrobo::MetalWorldCCDMode::speculative;
    config.velocityIterations=8u; config.finalVelocityIterations=4u;
    config.deterministic=true; config.warmStart=true;
    config.captureContactEvidence=true; config.captureConvexCacheEvidence=true;
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
    save(out/("trial"+std::to_string(trial)+"-first-v.bin"), result.first.finalV);
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
    save(out/("trial"+std::to_string(trial)+"-second-v.bin"), result.second.finalV);
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
        const std::array<mr_float4,8> vertices{{
            f4(-.05f,-.08f,-.08f,1),f4(.05f,-.08f,-.08f,1),f4(.05f,.08f,-.08f,1),f4(-.05f,.08f,-.08f,1),
            f4(-.05f,-.08f,.08f,1),f4(.05f,-.08f,.08f,1),f4(.05f,.08f,.08f,1),f4(-.05f,.08f,.08f,1)}};
        const std::array<std::uint32_t,36> indices{{0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7}};
        const auto cooked=metalrobo::cookConvexGeometry(model,vertices,indices);
        check(cooked.succeeded(), "convex geometry cook failed");
        auto& convex=model.shapes.front(); convex.shapeType=MR_SHAPE_CONVEX;
        convex.geometryOffset=cooked.geometryIndex; convex.geometryCount=1;
        convex.dimensions=f4(1,1,1); convex.contactRestAndBoundingRadius=f4(.002f,0,.13f);
        auto& cylinder=model.shapes.back(); cylinder.shapeType=MR_SHAPE_CYLINDER;
        cylinder.dimensions=f4(.025f,.06f,0); cylinder.contactRestAndBoundingRadius=f4(.002f,0,.066f);
        model.world.gravityAndTimestep=f4(0,0,0,1.f/120.f);
        metalrobo::CompiledWorld world;
        const auto compiled=metalrobo::compileMetalWorld(model,0u,world);
        check(compiled.succeeded() && world.eligiblePairs().size()==1 && world.eligiblePairs()[0].pairClass==MR_COLLISION_PAIR_CONVEX,
            "fixture did not compile as one convex pair");
        auto q=model.defaultQ; auto v=model.defaultV;
        q[0]=.0755f; q[1]=0.f; v[0]=-.05f;
        std::cout<<"world_fingerprint="<<world.fingerprint()<<" initial_x="<<q[0]<<" velocity_x="<<v[0]<<std::endl;
        const auto a=runTrial(world,q,v,out,0);
        const auto b=runTrial(world,q,v,out,1);
        const auto& cache=a.first.contactEvidence.convexCaches;
        const auto* bytes=reinterpret_cast<const unsigned char*>(cache.data());
        const bool populated=!cache.empty() && std::any_of(bytes,bytes+cache.size()*sizeof(MRConvexQueryCacheGPU),[](unsigned char c){return c!=0;});
        const bool hasContact=!a.first.contactStatuses.empty() && a.first.contactStatuses[0].activeContacts>0;
        const bool repeat=equal(a.first.contactEvidence.convexCaches,b.first.contactEvidence.convexCaches) &&
            equal(a.first.finalQ,b.first.finalQ) && equal(a.first.finalV,b.first.finalV) &&
            equal(a.second.contactEvidence.convexCaches,b.second.contactEvidence.convexCaches) &&
            equal(a.second.finalQ,b.second.finalQ) && equal(a.second.finalV,b.second.finalV);
        std::cout<<"cache_count="<<cache.size()<<" populated="<<populated<<" contact="<<hasContact<<" independent_initial_and_continuation_byte_equal="<<repeat<<std::endl;
        check(populated && hasContact && repeat, "population/contact/repeat gate failed");
        std::cout<<"resident_convex_fixture=PASS restoration_claim=0"<<std::endl;
        return 0;
    } catch (const std::exception& e) { std::cerr<<"resident_convex_fixture=FAIL message="<<e.what()<<std::endl; return 1; }
}
