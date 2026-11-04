#pragma once
#include "numi/matter/numi_human_shared.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <initializer_list>
#include <span>
#include <unordered_map>
#include <vector>

namespace numi::matter::detail {
// Private immutable GPU storage, not an asset ABI. Logical row order is exact.
inline constexpr std::uint32_t compactHumanContactMagic = 0x4e484331u;
static_assert(sizeof(NMNumiHumanFEMBodyContactSampleGPU) == 64u);
static_assert(sizeof(NMNumiHumanFEMBodyContactNeighborhoodRangeGPU) == 32u);
template <std::size_t N> struct ContactWordHash {
    std::size_t operator()(const std::array<std::uint32_t,N>& words) const noexcept {
        std::uint64_t hash=1469598103934665603ull;
        for (const auto word:words) { hash^=word; hash*=1099511628211ull; }
        return static_cast<std::size_t>(hash);
    }
};
struct CompactHumanContactRows {
    std::vector<std::byte> bytes;
    std::uint32_t rows=0u, templates=0u, seeds=0u;
};
inline bool decodeCompactHumanContactRow(std::span<const std::byte> bytes,
    std::uint32_t row, NMNumiHumanFEMBodyContactSampleGPU& sample,
    NMNumiHumanFEMBodyContactNeighborhoodRangeGPU& range) {
    if(bytes.size()<32u)return false;
    std::array<std::uint32_t,8> h{};
    std::memcpy(h.data(),bytes.data(),32u);
    const std::uint64_t mapEnd=32ull+8ull*h[0];
    const std::uint64_t templateEnd=std::uint64_t(h[5])+64ull*h[1];
    const std::uint64_t seedEnd=std::uint64_t(h[6])+96ull*h[2];
    if(h[3]!=compactHumanContactMagic||!h[0]||!h[1]||!h[2]||
       row>=h[0]||h[4]!=32u||(h[5]&15u)||(h[6]&15u)||
       mapEnd>h[5]||templateEnd>h[6]||seedEnd!=h[7]||h[7]!=bytes.size())
        return false;
    std::array<std::uint32_t,2> ids{};
    std::memcpy(ids.data(),bytes.data()+32u+8ull*row,8u);
    if(ids[0]>=h[1]||ids[1]>=h[2])return false;
    std::array<std::uint32_t,16> values{},seed{};
    std::memcpy(values.data(),bytes.data()+h[5]+64ull*ids[0],64u);
    const auto* seedBase=bytes.data()+h[6]+96ull*ids[1];
    std::memcpy(seed.data(),seedBase,64u);
    values[1]=seed[1];
    for(const auto first:{4u,8u,12u})
        for(std::uint32_t k=0;k<3u;++k)values[first+k]=seed[first+k];
    std::memcpy(&sample,values.data(),64u);
    std::memcpy(&range,seedBase+64u,32u);
    return true;
}
inline bool packCompactHumanContactRows(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> ranges,
    CompactHumanContactRows& result) {
    result={};
    if(samples.empty()||samples.size()!=ranges.size()||
       samples.size()>(std::numeric_limits<std::uint32_t>::max()-32u)/8u)return false;
    using Template=std::array<std::uint32_t,16>;
    using Seed=std::array<std::uint32_t,24>;
    std::unordered_map<Template,std::uint32_t,ContactWordHash<16>> templateIds;
    std::unordered_map<Seed,std::uint32_t,ContactWordHash<24>> seedIds;
    std::vector<Template> templates;
    std::vector<Seed> seeds;
    std::vector<std::array<std::uint32_t,2>> rows;
    rows.reserve(samples.size());
    for(std::size_t i=0;i<samples.size();++i) {
        Template value{};
        std::memcpy(value.data(),&samples[i],64u);
        // Neither raw public layout admits this private storage marker.
        if(value[3]==compactHumanContactMagic||ranges[i].reserved0==compactHumanContactMagic)
            return false;
        Seed seed{};
        seed[1]=value[1];value[1]=0u;
        for(const auto first:{4u,8u,12u})
            for(std::uint32_t k=0;k<3u;++k) {
                seed[first+k]=value[first+k];value[first+k]=0u;
            }
        std::memcpy(seed.data()+16u,&ranges[i],32u);
        auto [ti,nt]=templateIds.emplace(value,static_cast<std::uint32_t>(templates.size()));
        if(nt)templates.push_back(value);
        auto [si,ns]=seedIds.emplace(seed,static_cast<std::uint32_t>(seeds.size()));
        if(ns)seeds.push_back(seed);
        rows.push_back({ti->second,si->second});
    }
    const auto aligned=[](std::uint64_t n){return (n+15ull)&~15ull;};
    const std::uint64_t templateBase=aligned(32ull+rows.size()*8ull);
    const std::uint64_t seedBase=aligned(templateBase+templates.size()*64ull);
    const std::uint64_t total=seedBase+seeds.size()*96ull;
    if(total>std::numeric_limits<std::uint32_t>::max())return false;
    result.rows=static_cast<std::uint32_t>(rows.size());
    result.templates=static_cast<std::uint32_t>(templates.size());
    result.seeds=static_cast<std::uint32_t>(seeds.size());
    const std::array<std::uint32_t,8> header{result.rows,result.templates,result.seeds,
        compactHumanContactMagic,32u,static_cast<std::uint32_t>(templateBase),
        static_cast<std::uint32_t>(seedBase),static_cast<std::uint32_t>(total)};
    result.bytes.resize(static_cast<std::size_t>(total));
    std::memcpy(result.bytes.data(),header.data(),32u);
    std::memcpy(result.bytes.data()+32u,rows.data(),rows.size()*8u);
    std::memcpy(result.bytes.data()+templateBase,templates.data(),templates.size()*64u);
    std::memcpy(result.bytes.data()+seedBase,seeds.data(),seeds.size()*96u);
    // Check every logical row before installing immutable shared GPU storage.
    for(std::uint32_t i=0;i<result.rows;++i) {
        NMNumiHumanFEMBodyContactSampleGPU sample{};
        NMNumiHumanFEMBodyContactNeighborhoodRangeGPU range{};
        if(!decodeCompactHumanContactRow(result.bytes,i,sample,range)||
           std::memcmp(&sample,&samples[i],64u)!=0||
           std::memcmp(&range,&ranges[i],32u)!=0) {result={};return false;}
    }
    return true;
}
} // namespace numi::matter::detail
