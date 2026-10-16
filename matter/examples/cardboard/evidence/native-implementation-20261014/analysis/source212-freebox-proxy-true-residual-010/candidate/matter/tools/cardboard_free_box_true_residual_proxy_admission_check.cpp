#include "numi/matter/detail/free_box_true_residual_proxy_admission.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
using namespace numi::matter::detail;
unsigned checks=0;
void require(bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);}
NMRigidProxyGPU capsule(unsigned body){NMRigidProxyGPU p{};p.shapeKind=NM_RIGID_CAPSULE;p.bodyIndex=body;p.sceneBodyIndex=NM_INVALID_INDEX;p.flags=NM_RIGID_PRESCRIBED_TRANSLATION;p.adaptiveObjectIndex=NM_INVALID_INDEX;p.generalizedFreeBodyIndex=NM_INVALID_INDEX;return p;}
NMRigidProxyGPU box(unsigned body,unsigned scene){NMRigidProxyGPU p{};p.shapeKind=NM_RIGID_BOX;p.bodyIndex=body;p.sceneBodyIndex=scene;p.flags=NM_RIGID_DYNAMIC;p.adaptiveObjectIndex=NM_INVALID_INDEX;p.generalizedFreeBodyIndex=0;return p;}
NMContactPairGPU pair(unsigned node,unsigned proxy){NMContactPairGPU p{};p.continuumNode=node;p.rigidProxy=proxy;p.objectIndex=0;return p;}
FreeBoxCapsuleProxyScopeInput make(const std::vector<NMRigidProxyGPU>& ps,const std::vector<NMContactPairGPU>& cs){
 FreeBoxCapsuleProxyScopeInput x;x.proxies=ps.data();x.proxyCount=ps.size();x.pairs=cs.data();x.pairCount=cs.size();x.femNodeCount=8;x.currentBodyCount=3;x.currentBodyStride=3;x.sceneBodyCount=1;x.sceneBodyStride=1;x.requiresCurrentBodies=true;x.hasPrescribedTranslation=true;x.hasPrescribedPrevious=true;x.hasPrescribedEndpoints=true;x.requiresSceneBodies=true;x.hasPrimalContactArguments=true;x.hasCoupledGeneralizedCandidate=true;x.hasCoupledPointJacobians=true;return x;
}
void expect(FreeBoxCapsuleProxyScopeInput x,FreeBoxCapsuleProxyScopeFailure f,const char* m){require(validateFreeBoxCapsuleProxyScope(x)==f,m);}
}
int main(){try{
 std::vector<NMRigidProxyGPU> ps{capsule(0),capsule(1),box(2,0)};
 std::vector<NMContactPairGPU> cs{pair(0,0),pair(3,1),pair(7,2)};
 expect(make(ps,cs),FreeBoxCapsuleProxyScopeFailure::none,"valid capsule plus one free box must admit");
 expect(make(ps,{pair(7,2)}),FreeBoxCapsuleProxyScopeFailure::contact_pair_coverage,"box-only pairs must reject");
 expect(make(ps,{pair(0,0),pair(3,1)}),FreeBoxCapsuleProxyScopeFailure::contact_pair_coverage,"capsule-only pairs must reject");
 auto bad=cs;bad[0].objectIndex=1;expect(make(ps,bad),FreeBoxCapsuleProxyScopeFailure::contact_pair_map,"foreign object pair must reject");
 bad=cs;bad[0].continuumNode=8;expect(make(ps,bad),FreeBoxCapsuleProxyScopeFailure::contact_pair_map,"out-of-range FEM node must reject");
 bad=cs;bad[0].rigidProxy=3;expect(make(ps,bad),FreeBoxCapsuleProxyScopeFailure::contact_pair_map,"out-of-range proxy must reject");
 auto x=make(ps,cs);x.requiresSceneBodies=false;expect(x,FreeBoxCapsuleProxyScopeFailure::scene_body_state,"missing free body scene state must reject");
 x=make(ps,cs);x.hasCoupledPointJacobians=false;expect(x,FreeBoxCapsuleProxyScopeFailure::contact_state,"missing point Jacobians must reject");
 x=make(ps,cs);x.hasPrescribedEndpoints=false;expect(x,FreeBoxCapsuleProxyScopeFailure::current_body_state,"missing capsule endpoints must reject");
 auto pbad=ps;pbad[0].flags=NM_RIGID_DYNAMIC;expect(make(pbad,cs),FreeBoxCapsuleProxyScopeFailure::unsupported_proxy,"dynamic capsule must reject");
 pbad=ps;pbad[2].flags=NM_RIGID_DYNAMIC|NM_RIGID_ARTICULATED;expect(make(pbad,cs),FreeBoxCapsuleProxyScopeFailure::unsupported_proxy,"articulated BOX must reject");
 pbad=ps;pbad[1].adaptiveObjectIndex=0;expect(make(pbad,cs),FreeBoxCapsuleProxyScopeFailure::unsupported_proxy,"adaptive capsule must reject");
 pbad=ps;pbad[1]=box(1,0);bad=cs;bad[1]=pair(3,1);expect(make(pbad,bad),FreeBoxCapsuleProxyScopeFailure::free_box_count,"second BOX must reject");
 pbad=ps;pbad[0].shapeKind=NM_RIGID_SPHERE;expect(make(pbad,cs),FreeBoxCapsuleProxyScopeFailure::unsupported_proxy,"sphere proxy must reject");
 std::vector<NMRigidProxyGPU> maxSupported(16,capsule(0));maxSupported.back()=box(2,0);
 expect(make(maxSupported,{pair(0,0),pair(7,15)}),FreeBoxCapsuleProxyScopeFailure::none,"existing 16-proxy bound must admit");
 std::vector<NMRigidProxyGPU> many(17,capsule(0));many.back()=box(2,0);
 expect(make(many,{pair(0,0),pair(7,16)}),FreeBoxCapsuleProxyScopeFailure::proxy_count,"proxy count above full-contact bound must reject");
 std::cout<<"free_box_capsule_true_residual_proxy_checks="<<checks<<" physical_transactions=0\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
