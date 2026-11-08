#include "metalrobo/EngineModel.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
int main(int argc,char**argv) {
    if(argc!=2) return 2;
    auto model=metalrobo::makeFreeSphereEngineModel();
    metalrobo::CompiledWorld world;
    auto compiled=metalrobo::compileMetalWorld(model,0,world);
    if (!compiled.succeeded()) { std::cerr<<"compile "<<compiled.message<<"\n"; return 2; }
    auto q=model.defaultQ; q[1]=10;
    auto v=model.defaultV;
    std::vector<float> efforts(1000u*world.nv(),0);
    MRBodyStateGPU ground{};
    ground.position.w=1; ground.orientation.w=1;
    ground.flagsAndIndices[0]=MR_MOTION_STATIC;
    ground.flagsAndIndices[1]=MR_INVALID_INDEX;
    ground.flagsAndIndices[2]=0;
    std::array<MRBodyStateGPU,1> bodies{ground};
    metalrobo::MetalWorldBatch batch;
    batch.environmentCount=1; batch.controlStepCount=1000;
    batch.initialQ=q; batch.initialV=v; batch.efforts=efforts; batch.initialSceneBodies=bodies;
    metalrobo::MetalWorldStepConfig cfg;
    cfg.timestepSeconds=.001f; cfg.physicsSubsteps=1;
    cfg.solverMode=metalrobo::MetalWorldSolverMode::freeMotionABA;
    cfg.deterministic=true; cfg.metallibPath=argv[1];
    metalrobo::MetalWorldContext ctx;
    metalrobo::MetalWorldResult result;
    auto d=ctx.run(world,batch,cfg,result);
    std::cout<<std::setprecision(17)<<"success="<<d.succeeded()<<" published="<<d.published
        <<" accepted="<<d.successfulStepCount<<" failed="<<d.failedStepCount
        <<" host="<<static_cast<unsigned>(d.status)<<" gpu="<<d.firstGPUStatusCode
        <<" message="<<d.message<<"\n";
    if(!d.succeeded()||!d.published||d.failedStepCount!=0||result.finalQ.size()<2||result.finalV.size()<2) return 1;
    double y=result.finalQ[1], vy=result.finalV[1];
    constexpr double expectedY=10-9.81*.001*.001*1000*1001/2;
    constexpr double expectedV=-9.81;
    std::cout<<"y_m="<<y<<" vy_m_s="<<vy<<" y_error_m="<<std::abs(y-expectedY)
             <<" velocity_error_m_s="<<std::abs(vy-expectedV)<<"\n";
    return std::abs(y-expectedY)<=.001 && std::abs(vy-expectedV)<=.001 ? 0:1;
}
