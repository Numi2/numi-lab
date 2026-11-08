#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace metalrobo;
template<class T> void require(const T& d,const char* what){if(!d.succeeded())throw std::runtime_error(std::string(what)+": "+d.message);}
void check(bool b,const char* m){if(!b)throw std::runtime_error(m);}
struct Captured { std::uint64_t generation; HybridObservationBatch frame; };
struct Producer {
 MetalHybridRenderer& renderer; const MetalWorldFamilyContext& worlds;
 std::shared_ptr<const Captured> accepted; std::uint64_t generation=0;
 explicit Producer(MetalHybridRenderer& r,const MetalWorldFamilyContext& w):renderer(r),worlds(w){}
 bool capture(std::uint32_t count) {
  auto done=renderer.render(worlds,count,0);
  if(!done.succeeded()){std::cout<<"capture_render_failure status="<<(unsigned)done.status<<" message="<<done.message<<"\n";return false;}
  HybridObservationBatch candidate; auto read=renderer.readback(candidate);
  if(!read.succeeded()){std::cout<<"capture_readback_failure status="<<(unsigned)read.status<<" message="<<read.message<<"\n";return false;}
  auto pixels=static_cast<std::size_t>(candidate.width)*candidate.height*count;
  if(candidate.environmentCount!=count || pixels==0 || candidate.depth.size()!=pixels || candidate.validity.size()!=pixels || candidate.identities.size()!=pixels) return false;
  accepted=std::make_shared<const Captured>(Captured{generation+1,std::move(candidate)});++generation;return true;
 }
};
VisualRenderSceneV3 scene(const WorldTemplate& t,float x) {
 VisualRenderSceneV3 s;s.id="bounded_capture_probe";s.assetCount=static_cast<std::uint32_t>(t.assets.size());
 s.gaussians.push_back({{x,0.f,.08f,.98f},{.06f,.02f,.04f,1.f},{0.f,0.f,0.f,1.f},{1.f,.1f,.05f,0.f},{0u,MR_INVALID_INDEX,42u,MR_HYBRID_GAUSSIAN_ASSET_LOCAL}});
 s.environment=makeNeutralStudioEnvironmentV2();s.lightRig=makeStudioKeyLightRigV1();s.fingerprint=computeVisualRenderSceneV3Fingerprint(s);return s;
}
void save(const std::string& base,const Captured& c) {
 std::ofstream f(base,std::ios::binary);check(bool(f),"open output");
 auto v=[&](const auto& a){std::uint64_t n=a.size();f.write((const char*)&n,sizeof(n));if(n)f.write((const char*)a.data(),n*sizeof(a[0]));};
 f.write((const char*)&c.generation,sizeof(c.generation));f.write((const char*)&c.frame.metadata,sizeof(c.frame.metadata));
 v(c.frame.depth);v(c.frame.validity);v(c.frame.identities);f.flush();check(bool(f),"write output");
}
std::vector<unsigned> counts(const HybridObservationBatch& f,std::uint32_t bit){
 auto pixels=std::size_t(f.width)*f.height;std::vector<unsigned> out(f.environmentCount);
 for(unsigned e=0;e<f.environmentCount;++e)for(std::size_t p=0;p<pixels;++p)out[e]+=(f.validity[e*pixels+p]&bit)!=0;return out;
}
int main(int argc,char**argv){@autoreleasepool{try{
 check(argc==3,"usage: probe METALLIB OUTPUT_DIR"); std::string dir=argv[2];
 WorldTemplate t;require(compileEpisodeTwin(makeFrankaPickPlaceEpisodeTwin(),makeFrankaPickPlaceEngineModel(),t),"episode");
 WorldFamily f;require(compileWorldFamily(t,makeFrankaPickPlaceWorldProgram(),f),"family");
 MetalWorldFamilyContext worlds({.metallibPath=argv[1]});require(worlds.compile(f,4u),"compile worlds");require(worlds.sample(4u,29u),"sample worlds");
 MetalHybridRenderer visible({.metallibPath=argv[1]});require(visible.compile(scene(t,.45f),VisualRendererProfileV1::sensorFast(),4u),"visible compile");
 Producer p(visible,worlds);check(p.capture(4),"visible capture");auto first=p.accepted;auto geom=counts(first->frame,MR_VISUAL_VALIDITY_GEOMETRY);auto frames=counts(first->frame,MR_VISUAL_VALIDITY_FRAME);
 for(unsigned e=0;e<4;++e){std::cout<<"visible env="<<e<<" geometry="<<geom[e]<<" frame="<<frames[e]<<"\n";check(geom[e]>0&&frames[e]==first->frame.width*first->frame.height,"visible prerequisite");}
 save(dir+"/visible.bin",*first);
 check(!p.capture(5),"out-of-capacity capture should fail");std::cout<<"failure_published="<<(p.accepted!=first)<<" generation="<<p.generation<<"\n";check(p.accepted==first&&p.generation==1,"failed capture published");
 check(p.capture(4),"visible retry");std::cout<<"retry_present="<<(bool)p.accepted<<" generation="<<p.generation<<"\n";check(p.generation==2,"retry generation");save(dir+"/retry.bin",*p.accepted);
 MetalHybridRenderer empty({.metallibPath=argv[1]});require(empty.compile(scene(t,100.f),VisualRendererProfileV1::sensorFast(),4u),"empty compile");
 Producer e(empty,worlds);check(e.capture(4),"empty capture");auto eg=counts(e.accepted->frame,MR_VISUAL_VALIDITY_GEOMETRY);auto ef=counts(e.accepted->frame,MR_VISUAL_VALIDITY_FRAME);
 for(unsigned i=0;i<4;++i){std::cout<<"empty env="<<i<<" geometry="<<eg[i]<<" frame="<<ef[i]<<"\n";check(eg[i]==0&&ef[i]==e.accepted->frame.width*e.accepted->frame.height,"empty prerequisite");}
 std::cout<<"empty_present="<<(bool)e.accepted<<" generation="<<e.generation<<"\n";save(dir+"/empty.bin",*e.accepted);return 0;
}catch(const std::exception& e){std::cerr<<"error="<<e.what()<<"\n";return 2;}}}
