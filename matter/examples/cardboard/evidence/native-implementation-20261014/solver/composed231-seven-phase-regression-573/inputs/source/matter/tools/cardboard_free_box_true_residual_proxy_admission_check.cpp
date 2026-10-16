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
NMRigidProxyGPU prescribedBox(unsigned body){NMRigidProxyGPU p{};p.shapeKind=NM_RIGID_BOX;p.bodyIndex=body;p.sceneBodyIndex=NM_INVALID_INDEX;p.flags=NM_RIGID_PRESCRIBED_TRANSLATION;p.adaptiveObjectIndex=NM_INVALID_INDEX;p.generalizedFreeBodyIndex=NM_INVALID_INDEX;return p;}
NMContactPairGPU pair(unsigned node,unsigned proxy){NMContactPairGPU p{};p.continuumNode=node;p.rigidProxy=proxy;p.objectIndex=0;return p;}
FreeBoxCapsuleProxyScopeInput make(const std::vector<NMRigidProxyGPU>& ps,const std::vector<NMContactPairGPU>& cs){
 FreeBoxCapsuleProxyScopeInput x;x.proxies=ps.data();x.proxyCount=ps.size();x.pairs=cs.data();x.pairCount=cs.size();x.femNodeCount=8;x.currentBodyCount=3;x.currentBodyStride=3;x.sceneBodyCount=1;x.sceneBodyStride=1;x.requiresCurrentBodies=true;x.hasPrescribedTranslation=true;x.hasPrescribedPrevious=true;x.hasPrescribedEndpoints=true;x.requiresSceneBodies=true;x.hasPrimalContactArguments=true;x.hasCoupledGeneralizedCandidate=true;x.hasCoupledPointJacobians=true;return x;
}
void expect(FreeBoxCapsuleProxyScopeInput x,FreeBoxCapsuleProxyScopeFailure f,const char* m){require(validateFreeBoxCapsuleProxyScope(x)==f,m);}
}
int main(){try{
 std::vector<NMRigidProxyGPU> ps{capsule(0),capsule(1),box(2,0)};
 std::vector<NMContactPairGPU> cs{pair(0,0),pair(3,1),pair(7,2)};
 std::vector<NMRigidProxyGPU> singleBox{box(2,0)};
 std::vector<NMContactPairGPU> singleBoxPairs{pair(7,0)};
 auto oneBox=make(singleBox,singleBoxPairs); oneBox.hasPrescribedTranslation=false; oneBox.hasPrescribedPrevious=false; oneBox.hasPrescribedEndpoints=false;
 expect(oneBox,FreeBoxCapsuleProxyScopeFailure::none,"legacy single dynamic BOX must admit without capsule endpoint state");
 auto noCurrent=oneBox; noCurrent.requiresCurrentBodies=false; expect(noCurrent,FreeBoxCapsuleProxyScopeFailure::current_body_state,"single BOX still requires current body state");
 std::vector<NMContactPairGPU> singleBoxMultiPairs{pair(7,0),pair(6,0)};
 expect(make(singleBox,singleBoxMultiPairs),FreeBoxCapsuleProxyScopeFailure::none,"single BOX pairs alone must admit");
 std::vector<NMRigidProxyGPU> capsuleOnly{capsule(0)};
 std::vector<NMContactPairGPU> capsuleOnlyPairs{pair(0,0)};
 expect(make(capsuleOnly,capsuleOnlyPairs),FreeBoxCapsuleProxyScopeFailure::free_box_count,"capsule without BOX must reject");
 expect(make(ps,cs),FreeBoxCapsuleProxyScopeFailure::none,"valid capsule plus one free box must admit");
 // Exact authored shipping role shape: 3 capsule punches + 6 rails sharing
 // the three support-body identities + platen/floor + dynamic impactor.
 std::vector<NMRigidProxyGPU> shipping{
  capsule(0),prescribedBox(1),prescribedBox(1),
  capsule(2),prescribedBox(3),prescribedBox(3),
  capsule(4),prescribedBox(5),prescribedBox(5),
  prescribedBox(6),prescribedBox(7),box(8,8)};
 std::vector<NMContactPairGPU> shippingPairs;
 for(unsigned i=0;i<shipping.size();++i)shippingPairs.push_back(pair(i%8u,i));
 auto shippingScope=make(shipping,shippingPairs);
 shippingScope.currentBodyCount=9;shippingScope.currentBodyStride=9;
 shippingScope.sceneBodyCount=9;shippingScope.sceneBodyStride=9;
 expect(shippingScope,FreeBoxCapsuleProxyScopeFailure::none,
        "authored 12-proxy eight-prescribed-body shipping role map must admit");
 std::vector<NMContactPairGPU> shippingMissingBoxPairs{
     pair(0,0),pair(3,3),pair(7,11)};
 auto shippingMissingBox=make(shipping,shippingMissingBoxPairs);
 shippingMissingBox.currentBodyCount=9;shippingMissingBox.currentBodyStride=9;
 shippingMissingBox.sceneBodyCount=9;shippingMissingBox.sceneBodyStride=9;
 expect(shippingMissingBox,FreeBoxCapsuleProxyScopeFailure::contact_pair_coverage,
        "missing prescribed BOX pair class must reject");
 auto shippingMissingEndpoints=shippingScope;
 shippingMissingEndpoints.hasPrescribedEndpoints=false;
 expect(shippingMissingEndpoints,
        FreeBoxCapsuleProxyScopeFailure::current_body_state,
        "prescribed shipping boxes require endpoint state");
 auto shippingBadMap=shippingScope;
 shippingPairs.back().rigidProxy=12;
 shippingBadMap.pairs=shippingPairs.data();
 expect(shippingBadMap,FreeBoxCapsuleProxyScopeFailure::contact_pair_map,
        "unmapped shipping proxy index must reject");
 std::vector<NMRigidProxyGPU> ninePrescribed=shipping;
 ninePrescribed.insert(ninePrescribed.end()-1,prescribedBox(9));
 std::vector<NMContactPairGPU> ninePairs;
 for(unsigned i=0;i<ninePrescribed.size();++i)ninePairs.push_back(pair(i%8u,i));
 auto nineScope=make(ninePrescribed,ninePairs);
 nineScope.currentBodyCount=10;nineScope.currentBodyStride=10;
 nineScope.sceneBodyCount=9;nineScope.sceneBodyStride=9;
 expect(nineScope,FreeBoxCapsuleProxyScopeFailure::prescribed_body_count,
        "ninth distinct prescribed tool body must reject");
 std::vector<NMRigidProxyGPU> twoFree=shipping;
 twoFree.insert(twoFree.end(),box(9,9));
 std::vector<NMContactPairGPU> twoFreePairs;
 for(unsigned i=0;i<twoFree.size();++i)twoFreePairs.push_back(pair(i%8u,i));
 auto twoFreeScope=make(twoFree,twoFreePairs);
 twoFreeScope.currentBodyCount=10;twoFreeScope.currentBodyStride=10;
 twoFreeScope.sceneBodyCount=10;twoFreeScope.sceneBodyStride=10;
 expect(twoFreeScope,FreeBoxCapsuleProxyScopeFailure::free_box_count,
        "more than one dynamic free BOX must reject");
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
