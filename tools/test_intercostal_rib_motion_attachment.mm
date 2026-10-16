#import <Foundation/Foundation.h>
#include "NumiHumanRestingIntercostalRibAttachment.hpp"
#include <cstdlib>
#include <iostream>

static NSString* idsSHA(const std::vector<unsigned>& ids) {
    std::vector<std::uint8_t> bytes;
    for(unsigned local:ids) {
        bytes.push_back(std::uint8_t(local));
        bytes.push_back(std::uint8_t(local>>8));
        bytes.push_back(std::uint8_t(local>>16));
        bytes.push_back(std::uint8_t(local>>24));
    }
    return [NSString stringWithUTF8String:numiHumanIntercostalAttachmentIDSHA(bytes).c_str()];
}

static NSDictionary* attachment(unsigned rib,const std::vector<unsigned>& ids,
    NSString* anatomySHA,NSString* bonesSHA,id channelStable=nil,id channelIndex=nil,
    id channelBody=nil) {
    const bool is48=rib==48;
    const unsigned actualChannel=rib-36u;
    NSMutableArray* idArray=[NSMutableArray array];
    for(unsigned local:ids)[idArray addObject:@(local)];
    NSDictionary* channel=@{
        @"semantic":@51004,
        @"stable_id":(channelStable ? channelStable : @(rib)),
        @"body_index":(channelBody ? channelBody : @20),
        @"channel_kind":@5,
        @"channel_index":(channelIndex ? channelIndex : @(actualChannel)),
        @"source_member_id":is48?@"FJ3228":(rib==49?@"FJ3229":@"FJ3230"),
        @"source_member_sha256":is48?
            @"803a7134c1e69f640b87f9c6c4a846c869379446e771a1a7c9d18914c331a603":
            (rib==49?@"631b4f63625fa6cc89ad83961b2156abe5878d1baf4a8cb2b2acbaa06bc97f58":
            @"8c097e93d79c610bed49d6c027d25c8439447c20bfb94fa64009b9c4db4d8f5b")
    };
    NSDictionary* surface=@{
        @"semantic":@51010,@"stable_id":@313,@"layer":@1,@"body_index":@20,
        @"local_vertex_count":@100,@"interface_local_vertex_ids":idArray,
        @"interface_local_vertex_ids_sha256":idsSHA(ids)
    };
    NSDictionary* partition=@{
        @"interface_vertex_count":@(ids.size()),
        @"partition_ancestry_sha256":[@"" stringByPaddingToLength:64 withString:@"a" startingAtIndex:0],
        @"strict_float32_audit_sha256":[@"" stringByPaddingToLength:64 withString:@"b" startingAtIndex:0],
        @"source_preparation_report_sha256":[@"" stringByPaddingToLength:64 withString:@"c" startingAtIndex:0],
        @"actual_capture_pack_sha256":[@"" stringByPaddingToLength:64 withString:@"d" startingAtIndex:0],
        @"actual_capture_receipt_sha256":[@"" stringByPaddingToLength:64 withString:@"e" startingAtIndex:0],
        @"native_admission":@NO,@"retained_motion_admission":@NO
    };
    return @{
        @"schema":[NSString stringWithFormat:@"numi.human.intercostal313-rib%u-interface-motion-attachment.v1",rib],
        @"status":@"inferred_passive_visual_attachment_source_preparation_only",
        @"anatomy_payload_sha256":anatomySHA,@"bones_payload_sha256":bonesSHA,
        @"existing_rib_channel":channel,@"surface":surface,@"source_partition":partition
    };
}

static NSDictionary* provenance() {
    return @{
        @"source_id_map":@{@"313":@{
            @"body_index":@20,@"layer":@1,@"source_member":@"FJ1451M",
            @"source_sha256":@"d4c021ba00dbe2bd3c7e722cb5982df7659d7d68fe6f4ec980588bfd65d31214"
        }},
        @"rib_source_member_to_stable_id":@{@"FJ3228":@48,@"FJ3229":@49,@"FJ3230":@50}
    };
}

static NumiHumanIntercostalRibAttachmentContext context(
    const std::map<unsigned,unsigned>& channels,const std::map<unsigned,unsigned>& bones,
    bool verified=true,unsigned row=313,unsigned body=20,unsigned layer=1) {
    return {row,body,layer,100,verified,&channels,&bones};
}

static void expectReject(const char* label,NSDictionary* bindings,NSString* anatomySHA,
    NSString* bonesSHA,const NumiHumanIntercostalRibAttachmentContext& ctx) {
    @autoreleasepool {
        bool rejected=false;
        try {
            (void)numiHumanParseIntercostalRibAttachments(bindings,provenance(),anatomySHA,bonesSHA,ctx);
        } catch(const std::exception&) { rejected=true; }
        if(!rejected) {
            std::cerr<<"FAIL expected rejection: "<<label<<"\n";
            std::exit(1);
        }
        std::cout<<"PASS rejects "<<label<<"\n";
    }
}

int main() {
    @autoreleasepool {
        NSString* anatomySHA=[@"" stringByPaddingToLength:64 withString:@"1" startingAtIndex:0];
        NSString* bonesSHA=[@"" stringByPaddingToLength:64 withString:@"2" startingAtIndex:0];
        const std::map<unsigned,unsigned> channels{{48,12},{49,13},{50,14}};
        const std::map<unsigned,unsigned> boneBodies{{48,20},{49,20},{50,20}};
        const auto ctx=context(channels,boneBodies);
        NSDictionary* good=@{
            @"intercostal_313_rib48_motion_attachment":attachment(48,{3,9},anatomySHA,bonesSHA),
            @"intercostal_313_rib49_motion_attachment":attachment(49,{17,23},anatomySHA,bonesSHA)
        };
        const auto parsed=numiHumanParseIntercostalRibAttachments(good,provenance(),anatomySHA,bonesSHA,ctx);
        if(parsed.size()!=4||parsed.at(3)!=12||parsed.at(9)!=12||
            parsed.at(17)!=13||parsed.at(23)!=13) {
            std::cerr<<"FAIL positive 48/49 channel binding\n";return 1;
        }
        std::cout<<"PASS positive 48/49 channels\n";

        NSDictionary* with50=@{
            @"intercostal_313_rib48_motion_attachment":attachment(48,{3,9},anatomySHA,bonesSHA),
            @"intercostal_313_rib49_motion_attachment":attachment(49,{17,23},anatomySHA,bonesSHA),
            @"intercostal_313_rib50_motion_attachment":attachment(50,{31,37},anatomySHA,bonesSHA)
        };
        const auto p50=numiHumanParseIntercostalRibAttachments(with50,provenance(),anatomySHA,bonesSHA,ctx);
        if(p50.size()!=6||p50.at(31)!=14||p50.at(37)!=14)return 1;
        std::cout<<"PASS opt-in rib50 existing channel14 and source identity\n";
        expectReject("duplicate slot across rib channels",
            @{@"intercostal_313_rib49_motion_attachment":attachment(49,{23},anatomySHA,bonesSHA),
              @"intercostal_313_rib50_motion_attachment":attachment(50,{23},anatomySHA,bonesSHA)},
            anatomySHA,bonesSHA,ctx);
        expectReject("rib50 wrong motion channel",
            @{@"intercostal_313_rib50_motion_attachment":attachment(50,{31},anatomySHA,bonesSHA,nil,@13,nil)},
            anatomySHA,bonesSHA,ctx);

        NSDictionary* legacy=@{};
        if(!numiHumanParseIntercostalRibAttachments(legacy,provenance(),anatomySHA,bonesSHA,ctx).empty()) {
            std::cerr<<"FAIL legacy receipt compatibility\n";return 1;
        }
        std::cout<<"PASS legacy receipt without opt-in\n";

        expectReject("stale anatomy payload SHA",
            @{@"intercostal_313_rib48_motion_attachment":attachment(48,{3},bonesSHA,bonesSHA)},
            anatomySHA,bonesSHA,ctx);
        expectReject("duplicate local vertex",
            @{@"intercostal_313_rib48_motion_attachment":attachment(48,{3,3},anatomySHA,bonesSHA)},
            anatomySHA,bonesSHA,ctx);
        expectReject("wrong typed bone stable ID",
            @{@"intercostal_313_rib48_motion_attachment":attachment(48,{3},anatomySHA,bonesSHA,@YES,nil,nil)},
            anatomySHA,bonesSHA,ctx);
        expectReject("wrong rib channel",
            @{@"intercostal_313_rib49_motion_attachment":attachment(49,{23},anatomySHA,bonesSHA,nil,@12,nil)},
            anatomySHA,bonesSHA,ctx);
        const std::map<unsigned,unsigned> wrongBoneBodies{{48,19},{49,20}};
        expectReject("wrong bone owner",
            @{@"intercostal_313_rib48_motion_attachment":attachment(48,{3},anatomySHA,bonesSHA)},
            anatomySHA,bonesSHA,context(channels,wrongBoneBodies));
        expectReject("unverified bone owner map",
            @{@"intercostal_313_rib48_motion_attachment":attachment(48,{3},anatomySHA,bonesSHA)},
            anatomySHA,bonesSHA,context(channels,boneBodies,false));
        expectReject("wrong typed NHA row body",
            @{@"intercostal_313_rib48_motion_attachment":attachment(48,{3},anatomySHA,bonesSHA)},
            anatomySHA,bonesSHA,context(channels,boneBodies,true,313,23,1));
        std::cout<<"All intercostal rib attachment loader tests passed\n";
    }
    return 0;
}
