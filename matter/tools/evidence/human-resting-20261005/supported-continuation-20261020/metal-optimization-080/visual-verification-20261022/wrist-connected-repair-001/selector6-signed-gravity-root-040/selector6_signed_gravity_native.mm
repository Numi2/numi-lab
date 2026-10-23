#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace numi::matter;
static void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
struct Result { std::array<double,3> v{}; bool accepted=false,replay=false; double mass=0; };
static Result run(const MaterialProgram& material,const char* metallib,unsigned axis,double sign,bool selector6){
    WorldSource source; source.environmentCount=1; source.frameTimestep=0.000125;
    source.gravity={0.,0.,0.}; source.gravity[axis]=sign*9.81;
    source.materials={material};
    source.mixedSolver.femPreconditioner=selector6?FEMPreconditionerMode::coupledFEMTangentBlockJacobi:FEMPreconditionerMode::scalarDiagonal;
    source.mixedSolver.relativeResidual=1.e-7;
    source.mixedSolver.newtonIterations=12;
    source.mixedSolver.fgmresRestart=16;
    source.mixedSolver.fgmresIterations=64;
    ObjectSource o; o.name="signed_gravity_free_tetrahedron"; o.materialIndex=0;
    o.representation=Representation::fem; o.mixedFEM=false; o.twoWayCoupling=false;
    o.deformableContact=false; o.deformableSelfContact=false; o.characteristicLength=.01;
    o.femNodes={{{0.,0.,0.}},{{.01,0.,0.}},{{0.,.01,0.}},{{0.,0.,.01}}};
    o.tetrahedra={{{0,1,2,3}}}; source.objects={o};
    CompileOptions options; options.maximumRateExponent=0;
    auto c=compileWorld(source,options); need(c.succeeded(),"fixture compile failed");
    Runtime runtime; RuntimeConfiguration config; config.metallib=metallib; config.environmentCount=1;
    config.captureEvents=false;config.captureDiagnostics=true;config.adaptiveTransfer=false;
    auto init=runtime.initialize(c.world,config); need(init.encoded,"Runtime initialize: "+init.message);
    auto initial=runtime.snapshot();need(initial.available,"initial snapshot unavailable");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"no Metal device");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    auto statuses=[device newBufferWithLength:sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
    auto step=[&](){
        std::memset(statuses.contents,0,statuses.length);
        id<MTLCommandBuffer> cmd=[queue commandBuffer]; EncodeRequest request;
        request.commandBuffer=(__bridge void*)cmd;request.environmentStatuses=(__bridge void*)statuses;
        request.physicsSubsteps=1;request.controlStep=0;request.timestepSeconds=runtime.timestepSeconds();request.runAdaptiveTransfer=false;
        request.phase=EncodePhase::preDynamics;auto d=runtime.encode(request);need(d.encoded,"preDynamics: "+d.message);
        request.phase=EncodePhase::postCommit;d=runtime.encode(request);need(d.encoded,"postCommit: "+d.message);
        [cmd commit];[cmd waitUntilCompleted];need(cmd.status==MTLCommandBufferStatusCompleted,"GPU command failed");
        return static_cast<const MRMetalWorldStatusGPU*>(statuses.contents)->code;
    };
    const auto code=step();auto first=runtime.snapshot();need(first.available,"snapshot unavailable");
    Result r; r.accepted=code==0;
    for(const auto& n:first.femNodes){double m=n.positionAndMass.w;r.mass+=m;r.v[0]+=m*n.velocityAndInverseMass.x;r.v[1]+=m*n.velocityAndInverseMass.y;r.v[2]+=m*n.velocityAndInverseMass.z;}
    for(auto& v:r.v)v/=r.mass;
    auto restore=runtime.restore(initial);need(restore.encoded,"restore: "+restore.message);
    const auto replayCode=step();auto replay=runtime.snapshot();
    r.replay=replayCode==code&&first.femNodes.size()==replay.femNodes.size()&&std::memcmp(first.femNodes.data(),replay.femNodes.data(),first.femNodes.size()*sizeof(first.femNodes[0]))==0;
    std::cout<<std::setprecision(12)<<"{\"axis\":"<<axis<<",\"sign\":"<<sign<<",\"selector\":"<<(selector6?6:0)<<",\"code\":"<<code<<",\"mass\":"<<r.mass<<",\"velocity\":["<<r.v[0]<<","<<r.v[1]<<","<<r.v[2]<<"],\"bitwise_replay\":"<<(r.replay?"true":"false")<<"}\n"<<std::flush;
    return r;
}
int main(int argc,const char** argv){@autoreleasepool{try{
    need(argc==3,"usage: test material metallib");auto m=parseMatterFile(argv[1]);need(m.succeeded(),"material parse failed");
    bool ok=true;constexpr double expected=.000125*9.81;
    for(unsigned axis=0;axis<3;axis++){
        auto plus=run(m.material,argv[2],axis,1,true);auto minus=run(m.material,argv[2],axis,-1,true);
        for(unsigned k=0;k<3;k++){
            double target=k==axis?expected:0;
            ok=ok&&std::abs(plus.v[k]-target)<expected*1.e-3&&std::abs(minus.v[k]+target)<expected*1.e-3;
            ok=ok&&std::abs(plus.v[k]+minus.v[k])<expected*1.e-3;
        }
        ok=ok&&plus.accepted&&minus.accepted&&plus.replay&&minus.replay;
    }
    need(ok,"selector6 signed gravity / momentum / replay gate failed");
    std::cout<<"PASS native selector6 +/- gravity on all three axes, analytic total momentum, bitwise replay\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}}
