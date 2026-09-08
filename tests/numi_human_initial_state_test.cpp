#include "metalrobo/NumiHumanInitialState.hpp"
#include "metalrobo/mrnx_bridge_v1.h"
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool ok) { if (!ok) throw std::runtime_error("NHINIT1 regression failed"); }
}
int main() {
    static_assert(sizeof(mrnx_runtime_config_v7) == 264u);
    static_assert(offsetof(mrnx_runtime_config_v7, initial_state_payload_path) == 248u);
    static_assert(offsetof(mrnx_runtime_config_v7, expected_initial_state_fingerprint) == 256u);
    try {
        metalrobo::NumiHumanInitialState input;
        input.humanSourceFingerprint=0x0123456789abcdefull;
        input.worldFingerprint=7; input.timestepMicroseconds=100;
        input.sourceArchiveSHA256[0]=17;
        input.q={0,0,1,0,0,0,1,0.2f}; input.v={0,0,0,0,0,0,0.3f};
        MRMujocoMuscleStateGPU m{};m.excitationAndActivation={0.7f,0.4f,0.12f,-0.01f};
        input.muscles={m};
        std::vector<std::byte> bytes;std::string error;
        require(metalrobo::encodeNumiHumanInitialState(input,bytes,error));
        require(bytes.size()==96u+4u*(8u+7u+4u));
        require(bytes[40]==std::byte{0xef} && bytes[47]==std::byte{0x01});
        metalrobo::NumiHumanInitialState output;
        require(metalrobo::decodeNumiHumanInitialState(bytes,8,7,1,input.sourceArchiveSHA256,output,error));
        require(output.q==input.q && output.v==input.v && output.humanSourceFingerprint==input.humanSourceFingerprint);
        require(std::memcmp(output.muscles.data(),input.muscles.data(),16)==0);
        std::vector<std::byte> replay;require(metalrobo::encodeNumiHumanInitialState(output,replay,error));require(replay==bytes);
        const auto rejected=[&](const std::vector<std::byte>& invalid){
            require(!metalrobo::decodeNumiHumanInitialState(invalid,8,7,1,input.sourceArchiveSHA256,output,error));
            require(!error.empty() && output.q==input.q && output.v==input.v);
            require(std::memcmp(output.muscles.data(),input.muscles.data(),16)==0);
        };
        for (std::size_t at : {0u,8u,12u,16u,20u,24u,28u,32u,36u,64u}) {
            auto invalid=bytes;invalid[at]^=std::byte{1};rejected(invalid);
        }
        auto truncated=bytes;truncated.pop_back();rejected(truncated);
        auto extra=bytes;extra.push_back(std::byte{0});rejected(extra);
        for (std::size_t at : {40u,48u,56u}) {
            auto invalid=bytes;for (unsigned i=0;i<8;++i) invalid[at+i]=std::byte{0};rejected(invalid);
        }
        for (unsigned type=0;type<6;++type) {
            auto invalid=input;
            switch(type) {
            case 0:invalid.q[0]=std::numeric_limits<float>::quiet_NaN();break;
            case 1:invalid.v[0]=std::numeric_limits<float>::infinity();break;
            case 2:invalid.muscles[0].excitationAndActivation.x=-0.1f;break;
            case 3:invalid.muscles[0].excitationAndActivation.y=1.1f;break;
            case 4:invalid.muscles[0].excitationAndActivation.z=0;break;
            default:invalid.muscles[0].excitationAndActivation.w=std::numeric_limits<float>::infinity();break;
            }
            replay=bytes;require(!metalrobo::encodeNumiHumanInitialState(invalid,replay,error));require(replay==bytes);
        }
        // ABI admission rejects malformed v7 before native allocation.
        mrnx_runtime_info_v1 info{};
        require(mrnx_bridge_v1_runtime_create_v7(nullptr,&info)==nullptr);
        require(info.status==MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
        mrnx_runtime_config_v7 config{};
        config.abi_version=MRNX_RUNTIME_CONFIG_ABI_V7; config.struct_size=sizeof(config);
        require(mrnx_bridge_v1_runtime_create_v7(&config,&info)==nullptr);
        std::cout << "NHINIT1 round-trip, replay, malformed state, output preservation and v7 ABI passed\n";
        return 0;
    } catch(const std::exception& error) { std::cerr << error.what() << '\n';return 1; }
}
