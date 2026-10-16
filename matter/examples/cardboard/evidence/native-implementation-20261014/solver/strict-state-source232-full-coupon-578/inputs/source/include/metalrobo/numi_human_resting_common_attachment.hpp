#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>

namespace numi_human_resting_common_field {
inline float quinticSmoothstep(float low,float high,float value) {
    const float t=std::clamp((value-low)/(high-low),0.0f,1.0f);
    return t*t*t*(10.0f+t*(-15.0f+6.0f*t));
}

struct PassiveAttachmentRangeContract {
    std::uint32_t stableId=0,semantic=0,bodyIndex=0,sourceLayer=0;
    std::uint32_t firstVertex=0,vertexCount=0;
    bool hasOriginalBodyIndex=false,hasAttachment=false,quinticSmoothstep=false;
    std::uint32_t originalBodyIndex=0,anchorBodyIndex=0,torsoBodyIndex=0,sourceSuperiorAxis=0;
    float transitionLower=0,transitionUpper=0;
};

inline bool validPassiveAttachmentRanges(
    std::span<const PassiveAttachmentRangeContract> ranges,
    std::uint32_t firstVertex,std::uint32_t expectedEnd) {
    constexpr std::array<std::uint32_t,7> volumeOwners{{318,319,320,321,1,23,24}};
    std::uint64_t cursor=firstVertex;
    for(std::size_t i=0;i<ranges.size();++i) {
        const auto& r=ranges[i];
        const std::uint32_t expectedLayer=r.semantic==51011?2u:r.semantic==51021?5u:r.semantic==51022?6u:0u;
        if(!r.stableId||!expectedLayer||r.bodyIndex!=20||r.sourceLayer!=expectedLayer||
           r.firstVertex!=cursor||!r.vertexCount||
           cursor+r.vertexCount>std::numeric_limits<std::uint32_t>::max()) return false;
        for(auto id:volumeOwners)if(r.stableId==id)return false;
        for(std::size_t j=0;j<i;++j)if(r.stableId==ranges[j].stableId)return false;
        if(r.stableId==11) {
            if(r.semantic!=51011||!r.hasOriginalBodyIndex||r.originalBodyIndex!=7||!r.hasAttachment||
               r.anchorBodyIndex!=7||r.torsoBodyIndex!=20||r.sourceSuperiorAxis!=1||!r.quinticSmoothstep||
               r.transitionLower!=-0.10f||r.transitionUpper!=-0.055f) return false;
        } else if(r.hasOriginalBodyIndex||r.hasAttachment) return false;
        cursor+=r.vertexCount;
    }
    return cursor==expectedEnd;
}
}
