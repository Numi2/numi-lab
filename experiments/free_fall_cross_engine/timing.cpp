#include "metalrobo/EngineModel.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <mujoco/mujoco.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
using Clock=std::chrono::steady_clock;
struct Trial {double seconds,ey,ev; bool pass;};
int main(int argc,char**argv) {
    if(argc!=3) return 2;
    char error[1024]{};
    mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
    if(!m) { std::cerr<<error<<"\n";return 2; }
    m->opt.disableflags |= mjDSBL_CONTACT;
    auto model=metalrobo::makeFreeSphereEngineModel();
    model.shapes.clear(); model.world.shapeCount=0;
    metalrobo::CompiledWorld world;
    if(!metalrobo::compileMetalWorld(model,0,world).succeeded()) return 2;
    metalrobo::MetalWorldConfig wc;wc.metallibPath=argv[2];
    metalrobo::MetalWorldContext ctx(wc);
    constexpr double Y=10-9.81*.001*.001*1000*1001/2, V=-9.81;
    std::cout<<std::setprecision(17)<<"environments,trial,engine,wall_seconds,max_y_error_m,max_v_error_m_s,passed\n";
    bool all=true;
    for(unsigned count:{1u,32u}) {
        std::vector<float> q(count*world.nq()),v(count*world.nv(),0),effort(count*1000u*world.nv(),0);
        for(unsigned i=0;i<count;++i) { for(unsigned j=0;j<world.nq();++j) q[i*world.nq()+j]=model.defaultQ[j]; q[i*world.nq()+1]=10; }
        MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;
        ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;
        std::vector<MRBodyStateGPU> bodies(count,ground);
        metalrobo::MetalWorldBatch batch;
        batch.environmentCount=count;batch.controlStepCount=1000;batch.initialQ=q;batch.initialV=v;
        batch.efforts=effort;batch.initialSceneBodies=bodies;
        metalrobo::MetalWorldStepConfig cfg;
        cfg.timestepSeconds=.001f;cfg.physicsSubsteps=1;
        cfg.solverMode=metalrobo::MetalWorldSolverMode::freeMotionABA;
        cfg.deterministic=true;cfg.publishStateTrajectory=false;
        std::vector<mjData*> data;
        for(unsigned i=0;i<count;++i) {auto d=mj_makeData(m); if(!d) return 2; data.push_back(d);}
        auto numi=[&]() {
            metalrobo::MetalWorldResult result;
            auto t=Clock::now(); auto d=ctx.run(world,batch,cfg,result);
            double sec=std::chrono::duration<double>(Clock::now()-t).count();
            double ey=0,ev=0;
            bool pass=d.succeeded()&&d.published&&d.failedStepCount==0&&
                      result.finalQ.size()==count*world.nq()&&result.finalV.size()==count*world.nv();
            if(pass) for(unsigned i=0;i<count;++i) {
                ey=std::max(ey,std::abs(static_cast<double>(result.finalQ[i*world.nq()+1])-Y));
                ev=std::max(ev,std::abs(static_cast<double>(result.finalV[i*world.nv()+1])-V));
            }
            return Trial{sec,ey,ev,pass&&ey<=.001&&ev<=.001};
        };
        auto mujoco=[&]() {
            auto t=Clock::now(); double ey=0,ev=0;
            for(auto d:data) { mj_resetData(m,d); for(int j=0;j<1000;++j) mj_step(m,d);
                ey=std::max(ey,std::abs(d->qpos[1]-Y));ev=std::max(ev,std::abs(d->qvel[1]-V));}
            double sec=std::chrono::duration<double>(Clock::now()-t).count();
            return Trial{sec,ey,ev,std::isfinite(ey)&&std::isfinite(ev)&&ey<=.001&&ev<=.001};
        };
        auto wn=numi(),wm=mujoco();
        if(!wn.pass || !wm.pass) {std::cerr<<"warmup failed, environments="<<count<<" numi="<<wn.pass<<" mujoco="<<wm.pass<<"\n"; return 1;}
        for(int i=0;i<5;++i) {
            Trial a,b; if(i%2==0) {a=numi();b=mujoco();} else {b=mujoco();a=numi();}
            for(auto pair:{std::pair{"numi",a},std::pair{"mujoco_serial",b}}) {
                const auto& t=pair.second;
                std::cout<<count<<','<<i<<','<<pair.first<<','<<t.seconds<<','<<t.ey<<','<<t.ev<<','<<t.pass<<"\n";
                all=all&&t.pass;
            }
        }
        for(auto d:data) mj_deleteData(d);
    }
    mj_deleteModel(m);
    return all?0:1;
}
