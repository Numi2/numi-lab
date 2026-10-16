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
    id channelBody=nil,unsigned rowId=313) {
    const std::array<const char*,12> members{{"FJ3228","FJ3229","FJ3230","FJ3231","FJ3232","FJ3233","FJ3234","FJ3235","FJ3236","FJ3225","FJ3226","FJ3227"}};
    const std::array<const char*,12> hashes{{
        "803a7134c1e69f640b87f9c6c4a846c869379446e771a1a7c9d18914c331a603",
        "631b4f63625fa6cc89ad83961b2156abe5878d1baf4a8cb2b2acbaa06bc97f58",
        "8c097e93d79c610bed49d6c027d25c8439447c20bfb94fa64009b9c4db4d8f5b",
        "c1259fcdaf463bdecfc0868a3acc964edd4901ab03eb539b28e40462671be1fb",
        "c40bbcd5c2b98bb94161819a2cf824296289ce68eb62b37f3b175af4aadfd351",
        "8160b28fd649aadd705a05cc3e6e9cfd7035c44d8258b5a50977ec89fe8e7414",
        "372c650b2b7bc3f6767aa50a0f5b7e5dac6a15a17aeea5f5017166ba5d328757",
        "7d9525200958a9545391a2e135649580282108d77f7ae88a34003a56f48f4c8b",
        "ec09e201beaaa70c24a41eedcef4671118b453361f827be02fb73fc361208db8",
        "40680b5a3556b5984a5006e01649ff20cb4c88d9d3a7fef79cbad0648a972d54",
        "24a27d42732af55ad1e7142179f2dd4f906de78b701fbe5c459d8884a026f250",
        "3b7e5d6673a1be5bc14d72a4ea659d361b143f3fda5bca3ec758c176daba16c5"}};
    NSString* member=[NSString stringWithUTF8String:members.at(rib-48u)];
    NSString* memberSHA=[NSString stringWithUTF8String:hashes.at(rib-48u)];

    const unsigned actualChannel=rib-36u;
    NSMutableArray* idArray=[NSMutableArray array];
    for(unsigned local:ids)[idArray addObject:@(local)];
    NSDictionary* channel=@{
        @"semantic":@51004,
        @"stable_id":(channelStable ? channelStable : @(rib)),
        @"body_index":(channelBody ? channelBody : @20),
        @"channel_kind":@5,
        @"channel_index":(channelIndex ? channelIndex : @(actualChannel)),
        @"source_member_id":member,
        @"source_member_sha256":memberSHA
    };
    NSDictionary* surface=@{
        @"semantic":@51010,@"stable_id":@(rowId),@"layer":@1,@"body_index":@20,
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
        @"schema":[NSString stringWithFormat:@"numi.human.intercostal%u-rib%u-interface-motion-attachment.v1",rowId,rib],
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
        },@"315":@{
            @"body_index":@20,@"layer":@1,@"source_member":@"FJ1455M",
            @"source_sha256":@"d92760297b5c8ceacf1013460e372abcb05f97192ffc514758d74ac0b73af373"
        },@"317":@{
            @"body_index":@20,@"layer":@1,@"source_member":@"FJ1454M",
            @"source_sha256":@"7458ea18bdf836da68b0220da5421cddb1bb653da909d998dea14db3d5ef00c5"
        }},
        @"rib_source_member_to_stable_id":@{@"FJ3228":@48,@"FJ3229":@49,@"FJ3230":@50,@"FJ3231":@51,@"FJ3232":@52,@"FJ3233":@53,@"FJ3234":@54,@"FJ3235":@55,@"FJ3236":@56,@"FJ3225":@57,@"FJ3226":@58,@"FJ3227":@59}
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
        std::map<unsigned,unsigned> channels;
        for(unsigned rib=48;rib<=59;++rib)channels.emplace(rib,rib-36u);
        std::map<unsigned,unsigned> boneBodies;
        for(unsigned rib=48;rib<=59;++rib)boneBodies.emplace(rib,20);
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
        for(unsigned muscle:std::array<unsigned,3>{{313,315,317}})for(unsigned rib=48;rib<=59;++rib) {
            NSString* key=[NSString stringWithFormat:@"intercostal_%u_rib%u_motion_attachment",muscle,rib];
            NSDictionary* candidate=@{key:attachment(rib,{31,37},anatomySHA,bonesSHA,nil,nil,nil,muscle)};
            const auto result=numiHumanParseIntercostalRibAttachments(candidate,provenance(),anatomySHA,bonesSHA,
                context(channels,boneBodies,true,muscle));
            if(result.size()!=2||result.at(31)!=rib-36u||result.at(37)!=rib-36u)return 1;
            expectReject("extended intercostal wrong channel",
                @{key:attachment(rib,{31},anatomySHA,bonesSHA,nil,@((rib-36u+1u)%24u),nil,muscle)},
                anatomySHA,bonesSHA,context(channels,boneBodies,true,muscle));
            std::cout<<"PASS intercostal "<<muscle<<" existing rib channel "<<rib-36u<<"\n";
        }
        expectReject("unsupported rib60",
            @{@"intercostal_315_rib60_motion_attachment":attachment(59,{31},anatomySHA,bonesSHA,nil,nil,nil,315)},
            anatomySHA,bonesSHA,context(channels,boneBodies,true,315));
        std::cout<<"All intercostal rib attachment loader tests passed\n";
    }
    return 0;
}
