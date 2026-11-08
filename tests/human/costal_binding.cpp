#include "metalrobo/NumiHumanTissueBinding.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

template<class T> void append(std::vector<std::byte>& out,const T& value) {
    const auto* p=reinterpret_cast<const std::byte*>(&value);
    out.insert(out.end(),p,p+sizeof value);
}
int main() {
    using namespace metalrobo;
    const std::array<std::uint8_t,32> cart{1},rigid{2},registration{3};
    auto make=[&](bool split) {
        std::vector<std::byte> bytes;
        const std::array<char,8> magic=split?std::array<char,8>{'N','H','T','B','I','N','D','2'}:
            std::array<char,8>{'N','H','T','B','I','N','D','1'};
        append(bytes,magic);
        append(bytes,std::array<std::uint32_t,6>{split?2u:1u,14,3,4,1,0});
        append(bytes,cart);append(bytes,rigid);append(bytes,registration);
        append(bytes,std::array<double,16>{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1});
        for(int i=0;i<14;++i)append(bytes,split?std::array<std::uint32_t,3>{MR_INVALID_INDEX,1,2}:
            std::array<std::uint32_t,3>{1,1,1});
        if(split)append(bytes,std::array<std::uint32_t,4>{1,1,2,2});
        return bytes;
    };
    auto legacy=make(false), split=make(true);
    NumiHumanCostalBinding b;std::string error;
    auto require=[&](bool ok,const char* why){if(!ok){std::cerr<<why<<": "<<error<<"\n";std::exit(1);}};
    require(decodeNumiHumanCostalBinding(legacy,cart,rigid,b,error),"legacy decode");
    require(b.regions.size()==14&&b.nodeDonorBodies.empty()&&b.regions[0].sternalBody==b.regions[0].ribBody,"legacy ownership");
    require(decodeNumiHumanCostalBinding(split,cart,rigid,b,error),"split decode");
    require(b.regions.size()==14&&b.regions[0].sternalBody==1&&b.regions[0].ribBody==2&&
        b.nodeDonorBodies==std::vector<std::uint32_t>({1,1,2,2}),"split ownership");
    auto reject=[&](std::vector<std::byte> bytes){
        require(!decodeNumiHumanCostalBinding(bytes,cart,rigid,b,error)&&b.regions.empty()&&b.nodeDonorBodies.empty(),"atomic invalid rejection");
    };
    auto bad=split;bad.pop_back();reject(bad);
    bad=split;bad.push_back(std::byte{0});reject(bad);
    bad=split;std::uint32_t value=3;std::memcpy(bad.data()+424,&value,4);reject(bad);
    bad=split;value=1;std::memcpy(bad.data()+256,&value,4);reject(bad);
    bad=split;value=3;std::memcpy(bad.data()+260,&value,4);reject(bad);
    bad=split;bad[32]^=std::byte{1};reject(bad);
    bad=legacy;value=2;std::memcpy(bad.data()+264,&value,4);reject(bad);
    std::cout<<"legacy compatibility, explicit donor/anchor separation, and seven atomic rejection cases passed\n";
}
