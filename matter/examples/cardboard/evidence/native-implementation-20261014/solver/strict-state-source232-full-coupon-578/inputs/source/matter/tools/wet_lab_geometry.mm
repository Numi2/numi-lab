#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace numi::matter;
static void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
static double number(id value){need([value isKindOfClass:[NSNumber class]],"expected numeric field");double x=[value doubleValue];need(std::isfinite(x),"nonfinite numeric field");return x;}
static unsigned integer(id value,unsigned maximum){double x=number(value);need(x>=0&&x<=maximum&&std::floor(x)==x,"invalid bounded integer");return unsigned(x);}
int main(int argc,const char* argv[]){@autoreleasepool{try{
    need(argc==4,"usage: numi-matter-wet-lab mesh.json material.nmatter new-geometry.json");
    NSString* input=[NSString stringWithUTF8String:argv[1]],*output=[NSString stringWithUTF8String:argv[3]];
    need(![[NSFileManager defaultManager] fileExistsAtPath:output],"geometry output already exists");
    NSData* bytes=[NSData dataWithContentsOfFile:input];need(bytes&&bytes.length<=4*1024*1024,"missing/oversized mesh input");
    NSError* error=nil;id obj=[NSJSONSerialization JSONObjectWithData:bytes options:0 error:&error];
    need([obj isKindOfClass:[NSDictionary class]],"mesh must be a JSON object");NSDictionary* plan=obj;
    need([plan[@"format"] isEqual:@"numilab-wet-lab-mesh/v1"],"unsupported mesh format");
    NSSet* keys=[NSSet setWithArray:@[@"format",@"nodesMetres",@"tetrahedra",@"fixedNodes",@"initialStretchX",@"steps",@"stepSeconds"]];
    for(id key in plan)need([keys containsObject:key],"unknown mesh field");
    NSArray* nodes=plan[@"nodesMetres"],*tets=plan[@"tetrahedra"],*fixed=plan[@"fixedNodes"];
    need([nodes isKindOfClass:[NSArray class]]&&nodes.count>=4&&nodes.count<=4096,"invalid mesh nodes");
    need([tets isKindOfClass:[NSArray class]]&&tets.count>0&&tets.count<=8192,"invalid mesh tetrahedra");
    need([fixed isKindOfClass:[NSArray class]],"missing fixed nodes");
    unsigned steps=integer(plan[@"steps"],512);need(steps>0,"need at least one accepted step");
    double dt=number(plan[@"stepSeconds"]),stretch=number(plan[@"initialStretchX"]);
    need(dt>=1e-7&&dt<=1e-3&&stretch>=.9&&stretch<=1.1,"unsupported step or preload");
    auto parsed=parseMatterFile(argv[2]);need(parsed.succeeded(),"material did not parse");
    for(auto& p:parsed.material.parameters)p.identifiable=false;
    WorldSource source;source.gravity={0,0,0};source.frameTimestep=dt;source.materials={parsed.material};
    source.mixedSolver.newtonIterations=12;source.mixedSolver.fgmresIterations=64;
    ObjectSource tissue;tissue.name="wet-lab-authored-scaffold";tissue.representation=Representation::fem;tissue.mixedFEM=false;
    tissue.deformableContact=false;tissue.deformableSelfContact=false;tissue.characteristicLength=.001;
    for(id row in nodes){need([row isKindOfClass:[NSArray class]]&&[row count]==3,"node arity");
        std::array<double,3> p{number(row[0]),number(row[1]),number(row[2])};
        for(double v:p)need(std::abs(v)<=1,"geometry coordinate outside bounded metre domain");
        tissue.femReferenceNodes.push_back(p);p[0]*=stretch;tissue.femNodes.push_back(p);}
    for(id row in tets){need([row isKindOfClass:[NSArray class]]&&[row count]==4,"tetrahedron arity");
        TetrahedronSource t;for(unsigned i=0;i<4;++i)t.nodes[i]=integer(row[i],unsigned(nodes.count-1));tissue.tetrahedra.push_back(t);}
    for(id n in fixed)tissue.femFixedNodes.push_back(integer(n,unsigned(nodes.count-1)));
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes.bytes,CC_LONG(bytes.length),digest);
    std::memcpy(tissue.femReferenceSourceIdentity.data(),digest,sizeof(digest));source.objects={tissue};
    auto cooked=compileWorld(source,{.maximumRateExponent=0,.emitSpecializedMetal=false});
    if(!cooked.succeeded()){std::string text="mesh compile failed";for(const auto& d:cooked.diagnostics)text+="; "+d.message;throw std::runtime_error(text);}
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal unavailable");id<MTLCommandQueue> queue=[device newCommandQueue];need(queue!=nil,"Metal queue unavailable");
    Runtime runtime;RuntimeConfiguration config;config.metallib=NUMI_MATTER_METALLIB;config.captureEvents=false;config.captureDiagnostics=true;config.adaptiveTransfer=false;
    auto init=runtime.initialize(cooked.world,config);need(init.encoded,init.message);
    auto statuses=[device newBufferWithLength:sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];need(statuses!=nil,"status allocation failed");
    auto initial=runtime.snapshot();need(initial.available,initial.message);
    for(unsigned step=0;step<steps;++step){std::memset(statuses.contents,0,sizeof(MRMetalWorldStatusGPU));auto cb=[queue commandBuffer];need(cb!=nil,"command allocation failed");
        EncodeRequest r;r.commandBuffer=(__bridge void*)cb;r.environmentStatuses=(__bridge void*)statuses;r.controlStep=step;r.physicsSubsteps=1;r.timestepSeconds=runtime.timestepSeconds();r.runAdaptiveTransfer=false;
        r.phase=EncodePhase::preDynamics;auto a=runtime.encode(r);need(a.encoded,a.message);
        r.phase=EncodePhase::postCommit;auto b=runtime.encode(r);need(b.encoded,b.message);
        [cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Matter Metal execution failed");
        auto s=runtime.snapshot();need(s.available&&s.statuses.size()==1&&s.statuses[0].code==NM_STATUS_SUCCESS,"Matter step rejected; no geometry published");}
    auto accepted=runtime.snapshot();NSMutableArray* positions=[NSMutableArray array];double maximumDisplacement=0;
    for(unsigned i=0;i<accepted.femNodes.size();++i){const auto& p=accepted.femNodes[i].positionAndMass;const auto& q=initial.femNodes[i].positionAndMass;
        maximumDisplacement=std::max(maximumDisplacement,std::sqrt(double((p.x-q.x)*(p.x-q.x)+(p.y-q.y)*(p.y-q.y)+(p.z-q.z)*(p.z-q.z))));
        [positions addObject:@[@(p.x),@(p.y),@(p.z)]];}
    NSMutableString* hash=[NSMutableString string];for(auto v:digest)[hash appendFormat:@"%02x",v];
    NSDictionary* result=@{@"format":@"numilab-wet-lab-geometry/v1",@"status":@"accepted",@"owner":@"NumiLab Matter",
        @"nodesMetres":positions,@"tetrahedra":tets,@"referenceNodesMetres":nodes,@"sourceSHA256":hash,
        @"sourcePhysicsFingerprint":[NSString stringWithFormat:@"%llu",(unsigned long long)accepted.sourcePhysicsFingerprint],
        @"deviceProgramFingerprint":[NSString stringWithFormat:@"%llu",(unsigned long long)accepted.deviceProgramFingerprint],
        @"device":device.name,@"acceptedSteps":@(steps),@"timeSeconds":@(steps*runtime.timestepSeconds()),
        @"maximumDisplacementMetres":@(maximumDisplacement),@"evidenceStatus":@"numerical-mechanics-only",
        @"limits":@"Authored scaffold and constitutive parameters; not calibrated biological tissue. One-way accepted geometry export; no biological force feedback."};
    NSData* encoded=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];need(encoded!=nil,"JSON encoding failed");
    need([encoded writeToFile:output options:NSDataWritingWithoutOverwriting error:&error],"geometry publication failed");
    std::cout<<"accepted Matter geometry: "<<steps<<" steps, displacement "<<maximumDisplacement<<" m\n";return 0;
}catch(const std::exception& e){std::cerr<<"wet-lab geometry: "<<e.what()<<"\n";return 1;}}}
