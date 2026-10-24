#include "NumiHumanMaterialColumnSelfContact.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace numi::matter::human;
std::vector<NMFEMTopologyNodeGPU> nodes(unsigned n) {
    std::vector<NMFEMTopologyNodeGPU> a(n);
    for(unsigned i=0;i<n;++i){a[i].identity.x=i;a[i].identity.y=7;a[i].identity.z=1;a[i].identity.w=NM_TOPOLOGY_ACTIVE;}
    return a;
}
std::uint64_t oldFingerprint(std::uint64_t source,unsigned object,unsigned count) {
    std::uint64_t h=1469598103934665603ull;
    auto add=[&](std::uint8_t b){h^=b;h*=1099511628211ull;};
    for(unsigned s=0;s<64;s+=8)add(static_cast<std::uint8_t>(source>>s));
    constexpr char m[]="numi-human-three-layer-material-column-self-contact-v1";
    for(char b:m)add(static_cast<std::uint8_t>(b));
    for(unsigned s=0;s<32;s+=8){add(static_cast<std::uint8_t>(object>>s));add(static_cast<std::uint8_t>(count>>s));}
    return h?h:1u;
}
int main(){
    auto a=nodes(12);tagMaterialColumnNodes(a,7,4);
    for(unsigned i=0;i<12;++i)assert(materialColumnTag(a[i].identity.w)==i%4+1u);
    assert(materialColumnPhysicsFingerprint(900,7,4)==oldFingerprint(900,7,4));
    std::vector<std::uint32_t> cols{1,2,3,4,1,2,3,4,1,2,3,4,0};
    auto b=nodes(13);tagMaterialColumnNodes(b,7,4,cols);
    assert(materialColumnTag(b[12].identity.w)==0u);
    assert(!sameMaterialColumnAdjacency(7,b[12].identity.w,7,b[0].identity.w));
    assert(sameMaterialColumnAdjacency(7,b[0].identity.w,7,b[4].identity.w));
    assert(materialColumnPhysicsFingerprint(900,7,4,cols)!=oldFingerprint(900,7,4));
    std::swap(cols[0],cols[12]);auto c=nodes(13);tagMaterialColumnNodes(c,7,4,cols);
    assert(materialColumnTag(c[0].identity.w)==0 && materialColumnTag(c[12].identity.w)==1);
    assert(materialColumnPhysicsFingerprint(900,7,4,cols)!=materialColumnPhysicsFingerprint(900,7,4,std::vector<std::uint32_t>{1,2,3,4,1,2,3,4,1,2,3,4,0}));
    auto bad=nodes(13),before=bad;cols[12]=5;
    bool rejected=false;try{tagMaterialColumnNodes(bad,7,4,cols);}catch(const std::invalid_argument&){rejected=true;}
    assert(rejected && std::memcmp(bad.data(),before.data(),bad.size()*sizeof(bad[0]))==0);
    cols[12]=0;rejected=false;try{tagMaterialColumnNodes(bad,7,4,cols);}catch(const std::invalid_argument&){rejected=true;}
    assert(rejected && std::memcmp(bad.data(),before.data(),bad.size()*sizeof(bad[0]))==0);
    std::cout<<"legacy tag/fingerprint parity; explicit rows; no interior exclusions; invalid preflight rollback: PASS\n";
}
