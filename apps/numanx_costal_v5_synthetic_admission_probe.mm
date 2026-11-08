#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/mrnx_bridge_v1.h"
#include "metalrobo/NumiHumanRuntimeIdentity.hpp"
#include "numi/matter/matter.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

static std::vector<std::byte> read(const char* path) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()<=0) throw std::runtime_error(std::string("cannot read ")+path);
    std::vector<std::byte> bytes(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);file.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
    if(!file) throw std::runtime_error(std::string("short read ")+path);
    return bytes;
}
static void mix(std::uint64_t& h,std::uint8_t v){h=(h^v)*1099511628211ull;}
static void mix64(std::uint64_t& h,std::uint64_t v){for(unsigned i=0;i<8;i++)mix(h,static_cast<std::uint8_t>(v>>(8*i)));}
int main(int argc,const char** argv) {
  @autoreleasepool { try {
    if(argc!=11) throw std::runtime_error("usage: probe rigid muscle support equality visual vision world binding cartilage metalrobo_metallib (Matter metallib via MRNX_MATTER_METALLIB)");
    const char* matterLib=std::getenv("MRNX_MATTER_METALLIB");
    if(!matterLib) throw std::runtime_error("missing Matter metallib");
    const auto rigid=read(argv[1]),muscle=read(argv[2]),support=read(argv[3]);
    const auto eq=read(argv[4]),binding=read(argv[8]);
    const auto base=metalrobo::numiHumanRuntimeBaseSourceFingerprint(rigid,muscle,support);
    const auto eqfp=metalrobo::numiHumanRuntimePayloadFingerprint(eq);
    const auto bindfp=metalrobo::numiHumanRuntimePayloadFingerprint(binding);
    numi::matter::CompiledWorld world;std::string error;
    if(!numi::matter::readPackage(argv[7],world,nullptr,&error)) throw std::runtime_error("world: "+error);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device) throw std::runtime_error("Metal device unavailable");
    mrnx_runtime_config_v5 config{};config.abi_version=MRNX_RUNTIME_CONFIG_ABI_V5;config.struct_size=sizeof(config);
    config.costal_cartilage_payload_path=argv[9];config.costal_binding_payload_path=argv[8];config.expected_costal_binding_fingerprint=bindfp;
    auto& v4=config.runtime;v4.abi_version=MRNX_RUNTIME_CONFIG_ABI_V4;v4.struct_size=sizeof(v4);
    v4.joint_equality_payload_path=argv[4];v4.expected_joint_equality_fingerprint=eqfp;
    auto& v3=v4.runtime;v3.abi_version=MRNX_RUNTIME_CONFIG_ABI_V3;v3.struct_size=sizeof(v3);
    v3.matter_world_package_path=argv[7];v3.expected_matter_world_fingerprint=world.fingerprint;v3.expected_model_source_fingerprint=base;
    auto& v2=v3.runtime;v2.abi_version=MRNX_RUNTIME_CONFIG_ABI_V2;v2.struct_size=sizeof(v2);
    v2.metal_device=(__bridge void*)device;v2.rigid_payload_path=argv[1];v2.muscle_payload_path=argv[2];
    v2.support_contact_payload_path=argv[3];v2.visual_pack_path=argv[5];v2.vision_profile_path=argv[6];
    v2.metalrobo_metallib_path=argv[10];v2.matter_metallib_path=matterLib;
    v2.timestep_microseconds=10;v2.maximum_retained_bytes=2ull*1024*1024*1024;v2.transaction_slot_count=2;
    std::cout<<"synthetic=1 nodes="<<world.fem.nodes.size()<<" tets="<<world.fem.tetrahedra.size()<<" anchors="<<world.fem.humanAttachments.size()<<" world="<<world.fingerprint<<"\n";
    auto bad=config;bad.expected_costal_binding_fingerprint^=1u;mrnx_runtime_info_v1 failed{};
    auto* rejected=mrnx_bridge_v1_runtime_create_v5(&bad,&failed);
    if(rejected){mrnx_bridge_v1_runtime_drop(rejected);throw std::runtime_error("bad binding fingerprint admitted");}
    if(failed.status!=MRNX_RUNTIME_ASSET_FAILURE_V1) throw std::runtime_error("bad fingerprint wrong status "+std::to_string(failed.status));
    std::cout<<"PASS v5 rejects mismatched binding identity status="<<failed.status<<"\n";
    mrnx_runtime_info_v1 info{};auto* runtime=mrnx_bridge_v1_runtime_create_v5(&config,&info);
    if(!runtime || info.status!=MRNX_RUNTIME_READY_V1) throw std::runtime_error("v5 creation failed status="+std::to_string(info.status));
    auto expected=metalrobo::numiHumanRuntimeAppendPayloadOwner(base,"NHEQ2",eqfp);
    for(auto b:std::string("NHTMASS1"))mix(expected,static_cast<std::uint8_t>(b));
    mix64(expected,bindfp);mix64(expected,world.fingerprint);
    if(info.model_source_fingerprint!=expected){mrnx_bridge_v1_runtime_drop(runtime);throw std::runtime_error("v5 source identity mismatch");}
    std::cout<<"PASS native v5 admission source_fingerprint="<<info.model_source_fingerprint<<" synthetic_two_owner=1\n";
    mrnx_bridge_v1_runtime_drop(runtime);return 0;
  } catch(const std::exception& e){std::cerr<<"v5 synthetic admission FAILED: "<<e.what()<<"\n";return 1;} }
}
