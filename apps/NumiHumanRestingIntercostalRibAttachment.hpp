#pragma once

#include <CommonCrypto/CommonDigest.h>
#include <Foundation/Foundation.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct NumiHumanIntercostalRibAttachmentContext {
    unsigned rowStableId=0;
    unsigned rowBodyIndex=0;
    unsigned rowLayer=0;
    unsigned rowVertexCount=0;
    bool bonesSourceOwnerBindingsVerified=false;
    const std::map<unsigned,unsigned>* ribChannelByStableId=nullptr;
    const std::map<unsigned,unsigned>* boneBodyByStableId=nullptr;
};

[[noreturn]] static inline void numiHumanIntercostalAttachmentFail(const char* message) {
    throw std::runtime_error(message);
}

static inline void numiHumanIntercostalAttachmentRequire(bool condition,const char* message) {
    if(!condition)numiHumanIntercostalAttachmentFail(message);
}

static inline unsigned numiHumanIntercostalAttachmentUnsigned(id value,const char* label) {
    numiHumanIntercostalAttachmentRequire([value isKindOfClass:NSNumber.class],
        label);
    numiHumanIntercostalAttachmentRequire(
        CFGetTypeID((__bridge CFTypeRef)value)!=CFBooleanGetTypeID(),label);
    const double number=[value doubleValue];
    numiHumanIntercostalAttachmentRequire(std::isfinite(number)&&number>=0&&
        number<=std::numeric_limits<unsigned>::max()&&std::floor(number)==number,label);
    return static_cast<unsigned>(number);
}

static inline NSString* numiHumanIntercostalAttachmentString(id value,const char* label) {
    numiHumanIntercostalAttachmentRequire([value isKindOfClass:NSString.class],label);
    return (NSString*)value;
}

static inline bool numiHumanIntercostalAttachmentValidSHA(NSString* value) {
    if(![value isKindOfClass:NSString.class]||value.length!=64)return false;
    for(NSUInteger i=0;i<value.length;++i) {
        const unichar c=[value characterAtIndex:i];
        if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F')))return false;
    }
    return true;
}

static inline std::string numiHumanIntercostalAttachmentIDSHA(
    const std::vector<std::uint8_t>& bytes) {
    numiHumanIntercostalAttachmentRequire(!bytes.empty()&&bytes.size()<=UINT32_MAX,
        "intercostal attachment ID bytes are empty or too large");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    numiHumanIntercostalAttachmentRequire(
        CC_SHA256(bytes.data(),static_cast<CC_LONG>(bytes.size()),digest)!=nullptr,
        "intercostal attachment ID SHA-256 failed");
    static constexpr char hex[]="0123456789abcdef";
    std::string result;result.reserve(CC_SHA256_DIGEST_LENGTH*2u);
    for(const unsigned char byte:digest) {
        result.push_back(hex[byte>>4]);result.push_back(hex[byte&0x0fu]);
    }
    return result;
}

static inline std::map<unsigned,unsigned> numiHumanParseIntercostalRibAttachments(
    NSDictionary* bindings,NSDictionary* provenance,NSString* anatomyPayloadSHA256,
    NSString* bonesPayloadSHA256,const NumiHumanIntercostalRibAttachmentContext& context) {
    numiHumanIntercostalAttachmentRequire([bindings isKindOfClass:NSDictionary.class]&&
        [provenance isKindOfClass:NSDictionary.class],
        "intercostal attachment receipt bindings or provenance are missing");
    const unsigned rowId=context.rowStableId;
    const bool supportedRow=rowId==313||rowId==315||rowId==317;
    NSString* prefix=[NSString stringWithFormat:@"intercostal_%u_rib",rowId];
    bool hasAny=false;
    for(id rawKey in bindings)if([rawKey isKindOfClass:NSString.class]&&
        [(NSString*)rawKey hasPrefix:prefix]) {
        hasAny=true;
        bool supportedKey=false;
        for(unsigned rib=48;rib<=59;++rib)
            supportedKey|=[(NSString*)rawKey isEqualToString:
                [NSString stringWithFormat:@"intercostal_%u_rib%u_motion_attachment",rowId,rib]];
        numiHumanIntercostalAttachmentRequire(supportedKey,
            "unsupported intercostal rib-channel attachment key");
    }
    if(!supportedRow) {
        for(id rawKey in bindings)if([rawKey isKindOfClass:NSString.class]&&
            [(NSString*)rawKey hasPrefix:@"intercostal_313_rib"])
            numiHumanIntercostalAttachmentFail("unsupported intercostal source row");
    }
    if(!hasAny)return {};
    numiHumanIntercostalAttachmentRequire(supportedRow&&
        context.rowBodyIndex==20&&context.rowLayer==1&&context.rowVertexCount>0,
        "per-vertex rib attachment requires NHA intercostal row, layer 1, body 20");
    numiHumanIntercostalAttachmentRequire(context.bonesSourceOwnerBindingsVerified&&
        context.ribChannelByStableId!=nullptr&&context.boneBodyByStableId!=nullptr,
        "intercostal rib attachment lacks verified native bone-owner mappings");
    numiHumanIntercostalAttachmentRequire(
        numiHumanIntercostalAttachmentValidSHA(anatomyPayloadSHA256)&&
        numiHumanIntercostalAttachmentValidSHA(bonesPayloadSHA256),
        "intercostal attachment loaded payload SHA is malformed");

    NSDictionary* sourceIDMap=[provenance[@"source_id_map"] isKindOfClass:NSDictionary.class]?
        provenance[@"source_id_map"]:nil;
    NSDictionary* sourceRow=[sourceIDMap[[NSString stringWithFormat:@"%u",rowId]] isKindOfClass:NSDictionary.class]?
        sourceIDMap[[NSString stringWithFormat:@"%u",rowId]]:nil;
    numiHumanIntercostalAttachmentRequire(sourceRow!=nil&&
        numiHumanIntercostalAttachmentUnsigned(sourceRow[@"body_index"],"intercostal row body_index") ==20&&
        numiHumanIntercostalAttachmentUnsigned(sourceRow[@"layer"],"intercostal row layer")==1&&
        [numiHumanIntercostalAttachmentString(sourceRow[@"source_member"],"intercostal row source member")
            isEqualToString:(rowId==313?@"FJ1451M":(rowId==315?@"FJ1455M":@"FJ1454M"))]&&
        [numiHumanIntercostalAttachmentString(sourceRow[@"source_sha256"],"intercostal row source SHA")
            isEqualToString:(rowId==313?@"d4c021ba00dbe2bd3c7e722cb5982df7659d7d68fe6f4ec980588bfd65d31214":(rowId==315?@"d92760297b5c8ceacf1013460e372abcb05f97192ffc514758d74ac0b73af373":@"7458ea18bdf836da68b0220da5421cddb1bb653da909d998dea14db3d5ef00c5"))],
        "per-vertex rib attachment intercostal row source identity differs");
    NSDictionary* ribSourceMap=[provenance[@"rib_source_member_to_stable_id"] isKindOfClass:NSDictionary.class]?
        provenance[@"rib_source_member_to_stable_id"]:nil;
    numiHumanIntercostalAttachmentRequire(ribSourceMap!=nil,
        "per-vertex rib attachment lacks the receipt rib-source identity map");

    const std::array<unsigned,12> attachmentRibs{{48,49,50,51,52,53,54,55,56,57,58,59}};
    const std::array<const char*,12> attachmentMembers{{"FJ3228","FJ3229","FJ3230","FJ3231","FJ3232","FJ3233","FJ3234","FJ3235","FJ3236","FJ3225","FJ3226","FJ3227"}};
    const std::array<const char*,12> attachmentMemberSHAs{{
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
    std::map<unsigned,unsigned> result;
    for(unsigned i=0;i<attachmentRibs.size();++i) {
        const unsigned rib=attachmentRibs[i];
        NSString* key=[NSString stringWithFormat:@"intercostal_%u_rib%u_motion_attachment",rowId,rib];
        id raw=bindings[key];if(!raw)continue;
        numiHumanIntercostalAttachmentRequire([raw isKindOfClass:NSDictionary.class],
            "intercostal per-vertex rib attachment must be an object");
        NSDictionary* attachment=(NSDictionary*)raw;
        NSString* expectedSchema=[NSString stringWithFormat:
            @"numi.human.intercostal%u-rib%u-interface-motion-attachment.v1",rowId,rib];
        numiHumanIntercostalAttachmentRequire(
            [numiHumanIntercostalAttachmentString(attachment[@"schema"],"attachment schema")
                isEqualToString:expectedSchema]&&
            [numiHumanIntercostalAttachmentString(attachment[@"status"],"attachment status")
                isEqualToString:@"inferred_passive_visual_attachment_source_preparation_only"],
            "unsupported intercostal per-vertex rib attachment schema or status");
        numiHumanIntercostalAttachmentRequire(
            [numiHumanIntercostalAttachmentString(attachment[@"anatomy_payload_sha256"],
                "anatomy payload SHA") isEqualToString:anatomyPayloadSHA256]&&
            [numiHumanIntercostalAttachmentString(attachment[@"bones_payload_sha256"],
                "bones payload SHA") isEqualToString:bonesPayloadSHA256],
            "intercostal per-vertex rib attachment payload identity differs");
        NSDictionary* channel=[attachment[@"existing_rib_channel"] isKindOfClass:NSDictionary.class]?
            attachment[@"existing_rib_channel"]:nil;
        const auto channelIt=context.ribChannelByStableId->find(rib);
        numiHumanIntercostalAttachmentRequire(channel!=nil&&
            channelIt!=context.ribChannelByStableId->end()&&
            numiHumanIntercostalAttachmentUnsigned(channel[@"semantic"],"rib semantic")==51004&&
            numiHumanIntercostalAttachmentUnsigned(channel[@"stable_id"],"rib stable ID")==rib&&
            numiHumanIntercostalAttachmentUnsigned(channel[@"body_index"],"rib body index")==20&&
            numiHumanIntercostalAttachmentUnsigned(channel[@"channel_kind"],"rib channel kind")==5&&
            numiHumanIntercostalAttachmentUnsigned(channel[@"channel_index"],"rib channel index")==channelIt->second&&
            [numiHumanIntercostalAttachmentString(channel[@"source_member_id"],"rib source member")
                isEqualToString:[NSString stringWithUTF8String:attachmentMembers[i]]]&&
            [numiHumanIntercostalAttachmentString(channel[@"source_member_sha256"],"rib source member SHA")
                isEqualToString:[NSString stringWithUTF8String:attachmentMemberSHAs[i]]]&&
            numiHumanIntercostalAttachmentUnsigned(
                ribSourceMap[[NSString stringWithUTF8String:attachmentMembers[i]]],
                "rib source stable ID")==rib,
            "intercostal attachment does not identify the existing typed rib channel");
        const auto bone=context.boneBodyByStableId->find(rib);
        numiHumanIntercostalAttachmentRequire(bone!=context.boneBodyByStableId->end()&&
            bone->second==20,
            "intercostal attachment NHBONES row identity or owner is unverified");

        NSDictionary* surface=[attachment[@"surface"] isKindOfClass:NSDictionary.class]?
            attachment[@"surface"]:nil;
        numiHumanIntercostalAttachmentRequire(surface!=nil&&
            numiHumanIntercostalAttachmentUnsigned(surface[@"semantic"],"surface semantic")==51010&&
            numiHumanIntercostalAttachmentUnsigned(surface[@"stable_id"],"surface stable ID")==rowId&&
            numiHumanIntercostalAttachmentUnsigned(surface[@"layer"],"surface layer")==1&&
            numiHumanIntercostalAttachmentUnsigned(surface[@"body_index"],"surface body index")==20&&
            numiHumanIntercostalAttachmentUnsigned(surface[@"local_vertex_count"],"surface vertex count")==
                context.rowVertexCount,
            "intercostal attachment surface differs from the typed intercostal row row allocation");
        NSArray* rawIDs=surface[@"interface_local_vertex_ids"];
        NSString* expectedIDsSHA=numiHumanIntercostalAttachmentString(
            surface[@"interface_local_vertex_ids_sha256"],"local vertex ID SHA");
        numiHumanIntercostalAttachmentRequire([rawIDs isKindOfClass:NSArray.class]&&
            rawIDs.count>0&&numiHumanIntercostalAttachmentValidSHA(expectedIDsSHA),
            "intercostal attachment local vertex inventory is absent or malformed");
        std::vector<std::uint8_t> canonicalIDBytes;canonicalIDBytes.reserve(rawIDs.count*4u);
        unsigned previous=0;bool first=true;
        for(id value in rawIDs) {
            const unsigned local=numiHumanIntercostalAttachmentUnsigned(value,"local vertex ID");
            numiHumanIntercostalAttachmentRequire(local<context.rowVertexCount&&
                (first||local>previous),
                "intercostal attachment local vertex IDs are out of range, duplicated, or unsorted");
            numiHumanIntercostalAttachmentRequire(result.emplace(local,channelIt->second).second,
                "one intercostal row vertex is assigned to multiple rib motion channels");
            previous=local;first=false;
            canonicalIDBytes.push_back(std::uint8_t(local));
            canonicalIDBytes.push_back(std::uint8_t(local>>8));
            canonicalIDBytes.push_back(std::uint8_t(local>>16));
            canonicalIDBytes.push_back(std::uint8_t(local>>24));
        }
        numiHumanIntercostalAttachmentRequire(
            numiHumanIntercostalAttachmentIDSHA(canonicalIDBytes)==expectedIDsSHA.UTF8String,
            "intercostal attachment local vertex inventory SHA-256 differs");
        NSDictionary* partition=[attachment[@"source_partition"] isKindOfClass:NSDictionary.class]?
            attachment[@"source_partition"]:nil;
        numiHumanIntercostalAttachmentRequire(partition!=nil,
            "intercostal attachment lacks its source partition proof pins");
        numiHumanIntercostalAttachmentRequire(
            numiHumanIntercostalAttachmentUnsigned(partition[@"interface_vertex_count"],
                "interface vertex count")==rawIDs.count,
            "intercostal attachment interface vertex count differs from its local-ID inventory");
        for(NSString* field in @[@"partition_ancestry_sha256",@"strict_float32_audit_sha256",
                @"source_preparation_report_sha256",@"actual_capture_pack_sha256",
                @"actual_capture_receipt_sha256"])
            numiHumanIntercostalAttachmentRequire(
                numiHumanIntercostalAttachmentValidSHA(partition[field]),
                "intercostal attachment has an invalid or missing source proof hash");
        NSNumber* nativeAdmission=partition[@"native_admission"];
        NSNumber* motionAdmission=partition[@"retained_motion_admission"];
        numiHumanIntercostalAttachmentRequire([nativeAdmission isKindOfClass:NSNumber.class]&&
            CFGetTypeID((__bridge CFTypeRef)nativeAdmission)==CFBooleanGetTypeID()&&
            !nativeAdmission.boolValue&&[motionAdmission isKindOfClass:NSNumber.class]&&
            CFGetTypeID((__bridge CFTypeRef)motionAdmission)==CFBooleanGetTypeID()&&
            !motionAdmission.boolValue,
            "source-only intercostal attachment must not claim native or retained-motion admission");
    }
    return result;
}
