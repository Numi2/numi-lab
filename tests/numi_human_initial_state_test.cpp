#include "metalrobo/NumiHumanInitialState.hpp"
#include "metalrobo/mrnx_bridge_v1.h"
#include "numi_human_static_dynamic_handoff_cases.hpp"
#include <cstring>
#include <bit>
#include <cmath>
#include <string_view>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
std::size_t checks = 0;
void require(bool ok, const char* label="NHINIT regression") { ++checks; if (!ok) throw std::runtime_error(label); }
void put(std::vector<std::byte>& bytes, std::size_t at, std::uint64_t value, unsigned count) {
    for (unsigned i=0;i<count;++i) bytes[at+i]=std::byte((value>>(8*i))&255u);
}
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
        const std::string_view legacyHex="4e48494e495431000100000060000000080000000700000001000000040000000000000000000000efcdab896745230107000000000000006400000000000000110000000000000000000000000000000000000000000000000000000000000000000000000000000000803f0000000000000000000000000000803fcdcc4c3e0000000000000000000000000000000000000000000000009a99993e3333333fcdcccc3e8fc2f53d0ad723bc";
        const std::string_view digits="0123456789abcdef";
        for (std::size_t i=0;i<bytes.size();++i) {
            const auto byte=std::to_integer<unsigned>(bytes[i]);
            require(legacyHex[2*i]==digits[byte>>4] && legacyHex[2*i+1]==digits[byte&15], "NHINIT1 golden byte identity");
        }
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
        require(!output.rootTranslation && output.timestepNanoseconds==0u, "v1 preserves absent extension");
        auto tiny=input; tiny.q={0}; tiny.v={0};
        require(metalrobo::encodeNumiHumanInitialState(tiny,replay,error), "v1 generic small q remains supported");
        metalrobo::NumiHumanInitialState tinyOutput;
        require(metalrobo::decodeNumiHumanInitialState(replay,1,1,1,input.sourceArchiveSHA256,tinyOutput,error));
        tiny.timestepNanoseconds=100000u;
        const auto tinyBytes=replay;
        require(!metalrobo::encodeNumiHumanInitialState(tiny,replay,error) && replay==tinyBytes, "v2 rejects non-root q");

        auto v2input=input;
        v2input.timestepMicroseconds=0u; v2input.timestepNanoseconds=12500u;
        MRCompensatedRootTranslationGPU root{};
        root.reference={1.0f,-0.5f,1.0f,0.0f};
        root.displacement={0x1p-26f,0.25f,-0.125f,0.0f};
        root.correction={0x1p-55f,0.0f,0.0f,0.0f};
        v2input.rootTranslation=root;
        const auto projection=mrCompensatedTranslationProjection(root);
        v2input.q[0]=projection.x; v2input.q[1]=projection.y; v2input.q[2]=projection.z;
        std::vector<std::byte> v2bytes;
        require(metalrobo::encodeNumiHumanInitialState(v2input,v2bytes,error), "12500 ns compensated encode");
        require(v2bytes.size()==160u+4u*(8u+7u+4u), "NHINIT2 exact header size");
        require(v2bytes[6]==std::byte{'2'} && v2bytes[8]==std::byte{2} && v2bytes[32]==std::byte{1});
        require(v2bytes[96]==std::byte{0xd4} && v2bytes[97]==std::byte{0x30}, "12500 ns is exact LE integer");
        metalrobo::NumiHumanInitialState v2output;
        require(metalrobo::decodeNumiHumanInitialState(v2bytes,8,7,1,input.sourceArchiveSHA256,v2output,error));
        require(v2output.timestepNanoseconds==12500u && v2output.timestepMicroseconds==0u && v2output.rootTranslation.has_value());
        require(std::memcmp(&*v2output.rootTranslation,&root,sizeof(root))==0, "all low components preserved");
        require(metalrobo::encodeNumiHumanInitialState(v2output,replay,error) && replay==v2bytes, "NHINIT2 byte replay");
        const auto rejectedV2=[&](const std::vector<std::byte>& invalid,const char* label) {
            require(!metalrobo::decodeNumiHumanInitialState(invalid,8,7,1,input.sourceArchiveSHA256,v2output,error),label);
            require(!error.empty(), "rejection has a diagnostic");
            require(metalrobo::encodeNumiHumanInitialState(v2output,replay,error) && replay==v2bytes, "decode failure preserves entire v2 output");
        };
        for(std::size_t at : {0u,6u,8u,12u,16u,20u,24u,28u,36u,64u,152u,159u,160u}) {
            auto invalid=v2bytes; invalid[at]^=std::byte{1}; rejectedV2(invalid,"v2 header/source/reserved/projection mutation");
        }
        for(std::uint32_t flags : {2u,3u,0xffffffffu}) {
            auto invalid=v2bytes; put(invalid,32,flags,4); rejectedV2(invalid,"unknown v2 flags");
        }
        for(std::size_t at : {116u,132u,148u}) {
            auto invalid=v2bytes; put(invalid,at,0x80000000u,4); rejectedV2(invalid,"negative zero w padding");
            put(invalid,at,0x3f800000u,4); rejectedV2(invalid,"nonzero w padding");
        }
        for(std::size_t at : {104u,120u,136u}) {
            auto invalid=v2bytes; put(invalid,at,0x7fc00000u,4); rejectedV2(invalid,"nonfinite root field");
        }
        for(auto ns : {0ull,1'000'000'001ull,std::numeric_limits<unsigned long long>::max()}) {
            auto invalid=v2bytes; put(invalid,96,ns,8); rejectedV2(invalid,"invalid ns clock");
        }
        for(auto us : {12ull,13ull,std::numeric_limits<unsigned long long>::max()}) {
            auto invalid=v2bytes; put(invalid,56,us,8); rejectedV2(invalid,"inconsistent or overflowing dual clock");
        }
        auto invalid=v2bytes; put(invalid,136,0x3f800000u,4); rejectedV2(invalid,"noncanonical compensation");
        invalid=v2bytes; put(invalid,104,0x7f7fffffu,4); put(invalid,120,0x7f7fffffu,4);
        rejectedV2(invalid,"finite root projection overflow");
        invalid=v2bytes; put(invalid,32,0u,4); rejectedV2(invalid,"absent pair cannot carry state");
        invalid=v2bytes; invalid.resize(96); rejectedV2(invalid,"truncated v2 extension");
        invalid=v2bytes; invalid.pop_back(); rejectedV2(invalid,"truncated v2 state");
        invalid=v2bytes; invalid.push_back(std::byte{0}); rejectedV2(invalid,"v2 extension bytes rejected");
        require(!metalrobo::decodeNumiHumanInitialState(v2bytes,0xffffffffu,0xffffffffu,0xffffffffu,
                input.sourceArchiveSHA256,v2output,error), "oversized dimensions rejected before allocation");

        require(metalrobo::numiHumanInitialStateTimestepNanoseconds(input)==100000u, "legacy effective ns");
        require(metalrobo::numiHumanInitialStateTimestepNanoseconds(v2input)==12500u, "exact submicrosecond clock");
        auto invalidClock=input; invalidClock.timestepNanoseconds=12500u;
        require(metalrobo::numiHumanInitialStateTimestepNanoseconds(invalidClock)==0u, "conflicting clock accessor");
        invalidClock.timestepNanoseconds=0u; invalidClock.timestepMicroseconds=std::numeric_limits<std::uint64_t>::max();
        require(metalrobo::numiHumanInitialStateTimestepNanoseconds(invalidClock)==0u, "clock multiplication overflow admission");
        auto clockOnly=input; clockOnly.timestepMicroseconds=0; clockOnly.timestepNanoseconds=12500;
        require(metalrobo::encodeNumiHumanInitialState(clockOnly,replay,error), "clock-only v2 migration");
        const auto clockBytes=replay;
        require(clockBytes[32]==std::byte{0});
        for(std::size_t i=104;i<160;++i) require(clockBytes[i]==std::byte{0}, "absent pair and reserved bytes canonical zero");
        metalrobo::NumiHumanInitialState clockOutput;
        require(metalrobo::decodeNumiHumanInitialState(clockBytes,8,7,1,input.sourceArchiveSHA256,clockOutput,error));
        require(!clockOutput.rootTranslation && clockOutput.timestepNanoseconds==12500);
        require(metalrobo::encodeNumiHumanInitialState(clockOutput,replay,error) && replay==clockBytes);
        auto badAbsent=clockBytes; badAbsent[104]=std::byte{1};
        require(!metalrobo::decodeNumiHumanInitialState(badAbsent,8,7,1,input.sourceArchiveSHA256,clockOutput,error));
        require(metalrobo::encodeNumiHumanInitialState(clockOutput,replay,error) && replay==clockBytes, "absent pair reject atomicity");

        auto pairOnly=v2input; pairOnly.timestepNanoseconds=0; pairOnly.timestepMicroseconds=100;
        require(metalrobo::encodeNumiHumanInitialState(pairOnly,replay,error), "pair-only selects v2 exact migrated ns");
        const auto pairBytes=replay;
        require(metalrobo::decodeNumiHumanInitialState(pairBytes,8,7,1,input.sourceArchiveSHA256,clockOutput,error));
        require(clockOutput.timestepMicroseconds==100 && clockOutput.timestepNanoseconds==100000);
        require(metalrobo::encodeNumiHumanInitialState(clockOutput,replay,error) && replay==pairBytes);
        for(unsigned mode=0;mode<6;++mode) {
            auto bad=v2input;
            switch(mode) {
            case 0:bad.timestepMicroseconds=12;break;
            case 1:bad.timestepNanoseconds=std::numeric_limits<std::uint64_t>::max();break;
            case 2:bad.rootTranslation->correction.w=-0.0f;break;
            case 3:bad.rootTranslation->correction.x=1.0f;break;
            case 4:bad.q[0]=std::nextafter(bad.q[0],2.0f);break;
            default:bad.rootTranslation->reference.x=std::numeric_limits<float>::infinity();break;
            }
            replay=v2bytes;
            require(!metalrobo::encodeNumiHumanInitialState(bad,replay,error) && replay==v2bytes, "encode failure preserves entire output");
        }
        // ABI admission rejects malformed v7 before native allocation.
        mrnx_runtime_info_v1 info{};
        require(mrnx_bridge_v1_runtime_create_v7(nullptr,&info)==nullptr);
        require(info.status==MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
        mrnx_runtime_config_v7 config{};
        config.abi_version=MRNX_RUNTIME_CONFIG_ABI_V7; config.struct_size=sizeof(config);
        require(mrnx_bridge_v1_runtime_create_v7(&config,&info)==nullptr);

        const std::size_t handoffChecks =
            metalrobo::test::runNumiHumanStaticDynamicHandoffCases();
        checks += handoffChecks;
        std::cout << "NHINIT1 golden migration, NHINIT2 compensated/ns serialization, and static/dynamic handoff contract: "
                  << checks << " controls passed; no GPU execution\n";
        return 0;
    } catch(const std::exception& error) { std::cerr << error.what() << '\n';return 1; }
}
