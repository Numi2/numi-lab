#include "metalrobo/EngineModel.hpp"
#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <array>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        auto model=metalrobo::makeFreeSphereEngineModel();
        model.shapes[1].dimensions.x=0.002f;
        model.shapes[1].contactRestAndBoundingRadius={2.0e-5f,0.f,0.002f,0.f};
        model.world.gravityAndTimestep={0.f,0.f,0.f,1.f/120.f};
        model.defaultQ[0]=0.06f; model.defaultQ[1]=-0.0019f; model.defaultQ[2]=0.f;
        MRBodyStateGPU floor{};
        floor.position={0.f,-1.f,0.f,1.f}; floor.orientation={0.f,0.f,0.f,1.f};
        floor.flagsAndIndices[0]=MR_MOTION_STATIC; floor.flagsAndIndices[1]=MR_INVALID_INDEX;
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
        if (!composed.succeeded()) throw std::runtime_error("compose: "+composed.message);
        metalrobo::CompiledWorld world;
        const auto compiled=metalrobo::compileMetalWorld(source,world);
        if (!compiled.succeeded()) throw std::runtime_error("compile: "+compiled.message);
        std::cout<<"world_fingerprint="<<world.fingerprint()<<" rods="<<world.rodCount()
            <<" scene_bodies="<<world.sceneBodyCount()<<" rod_colliders="<<world.rodColliders().size()
            <<" rod_tool_pairs="<<world.rodToolPairs().size()<<" rod_nodes="<<world.rodNodeCount()
            <<" rod_edges="<<world.rodEdgeCount()<<"\n";
        for (std::size_t i=0;i<world.rodToolPairs().size();++i) {
            const auto& p=world.rodToolPairs()[i];
            std::cout<<"pair="<<i<<" rod_collider="<<p.rodCollider<<" rigid_collider="<<p.rigidCollider
                <<" pair_class="<<p.pairClass<<" flags="<<p.flags<<"\n";
        }
        if (world.rodCount()!=1 || world.rodToolPairs().empty() || world.rodColliders().size()!=4)
            throw std::runtime_error("fixture missing rod/tool topology");
        std::cout<<"host_fixture=PASS gpu_executed=0 restoration_claim=0\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"host_fixture=FAIL reason="<<e.what()<<"\n";
        return 1;
    }
}
