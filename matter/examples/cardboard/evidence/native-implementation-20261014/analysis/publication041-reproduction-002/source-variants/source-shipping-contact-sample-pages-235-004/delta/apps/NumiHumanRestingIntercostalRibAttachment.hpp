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
    const bool has48=bindings[@"intercostal_313_rib48_motion_attachment"]!=nil;
    const bool has49=bindings[@"intercostal_313_rib49_motion_attachment"]!=nil;
    bool hasAny=false;
    for(id rawKey in bindings)if([rawKey isKindOfClass:NSString.class]&&
        [(NSString*)rawKey hasPrefix:@"intercostal_313_rib"]) {
        hasAny=true;
        numiHumanIntercostalAttachmentRequire(
            [(NSString*)rawKey isEqualToString:@"intercostal_313_rib48_motion_attachment"]||
            [(NSString*)rawKey isEqualToString:@"intercostal_313_rib49_motion_attachment"],
            "unsupported NHA313 rib-channel attachment key");
    }
    if(!hasAny)return {};
    numiHumanIntercostalAttachmentRequire(has48||has49,
        "unsupported NHA313 rib-channel attachment declaration");
    numiHumanIntercostalAttachmentRequire(context.rowStableId==313&&
        context.rowBodyIndex==20&&context.rowLayer==1&&context.rowVertexCount>0,
        "per-vertex rib attachment requires NHA (51010,313), layer 1, body 20");
    numiHumanIntercostalAttachmentRequire(context.bonesSourceOwnerBindingsVerified&&
        context.ribChannelByStableId!=nullptr&&context.boneBodyByStableId!=nullptr,
        "intercostal rib attachment lacks verified native bone-owner mappings");
    numiHumanIntercostalAttachmentRequire(
        numiHumanIntercostalAttachmentValidSHA(anatomyPayloadSHA256)&&
        numiHumanIntercostalAttachmentValidSHA(bonesPayloadSHA256),
        "intercostal attachment loaded payload SHA is malformed");

    NSDictionary* sourceIDMap=[provenance[@"source_id_map"] isKindOfClass:NSDictionary.class]?
        provenance[@"source_id_map"]:nil;
    NSDictionary* source313=[sourceIDMap[@"313"] isKindOfClass:NSDictionary.class]?
        sourceIDMap[@"313"]:nil;
    numiHumanIntercostalAttachmentRequire(source313!=nil&&
        numiHumanIntercostalAttachmentUnsigned(source313[@"body_index"],"NHA313 body_index") ==20&&
        numiHumanIntercostalAttachmentUnsigned(source313[@"layer"],"NHA313 layer")==1&&
        [numiHumanIntercostalAttachmentString(source313[@"source_member"],"NHA313 source member")
            isEqualToString:@"FJ1451M"]&&
        [numiHumanIntercostalAttachmentString(source313[@"source_sha256"],"NHA313 source SHA")
            isEqualToString:@"d4c021ba00dbe2bd3c7e722cb5982df7659d7d68fe6f4ec980588bfd65d31214"],
        "per-vertex rib attachment NHA313 source identity differs");
    NSDictionary* ribSourceMap=[provenance[@"rib_source_member_to_stable_id"] isKindOfClass:NSDictionary.class]?
        provenance[@"rib_source_member_to_stable_id"]:nil;
    numiHumanIntercostalAttachmentRequire(ribSourceMap!=nil,
        "per-vertex rib attachment lacks the receipt rib-source identity map");

    const std::array<unsigned,2> attachmentRibs{{48,49}};
    const std::array<const char*,2> attachmentMembers{{"FJ3228","FJ3229"}};
    const std::array<const char*,2> attachmentMemberSHAs{{
        "803a7134c1e69f640b87f9c6c4a846c869379446e771a1a7c9d18914c331a603",
        "631b4f63625fa6cc89ad83961b2156abe5878d1baf4a8cb2b2acbaa06bc97f58"}};
    std::map<unsigned,unsigned> result;
    for(unsigned i=0;i<attachmentRibs.size();++i) {
        const unsigned rib=attachmentRibs[i];
        NSString* key=[NSString stringWithFormat:@"intercostal_313_rib%u_motion_attachment",rib];
        id raw=bindings[key];if(!raw)continue;
        numiHumanIntercostalAttachmentRequire([raw isKindOfClass:NSDictionary.class],
            "intercostal per-vertex rib attachment must be an object");
        NSDictionary* attachment=(NSDictionary*)raw;
        NSString* expectedSchema=[NSString stringWithFormat:
            @"numi.human.intercostal313-rib%u-interface-motion-attachment.v1",rib];
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
            numiHumanIntercostalAttachmentUnsigned(surface[@"stable_id"],"surface stable ID")==313&&
            numiHumanIntercostalAttachmentUnsigned(surface[@"layer"],"surface layer")==1&&
            numiHumanIntercostalAttachmentUnsigned(surface[@"body_index"],"surface body index")==20&&
            numiHumanIntercostalAttachmentUnsigned(surface[@"local_vertex_count"],"surface vertex count")==
                context.rowVertexCount,
            "intercostal attachment surface differs from the typed NHA313 row allocation");
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
                "one NHA313 vertex is assigned to multiple rib motion channels");
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
