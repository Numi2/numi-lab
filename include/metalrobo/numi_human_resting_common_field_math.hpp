#pragma once
#include <array>
#include <cstdint>
#include "metalrobo/numi_human_resting_visual_gpu.h"

namespace numi_human_resting_common_field {
using Exponent = std::array<std::uint8_t, MR_HUMAN_RESTING_COMMON_COORDINATE_COUNT>;
consteval void appendLexicographic(
    std::array<Exponent, MR_HUMAN_RESTING_COMMON_CUBIC_TERM_COUNT>& output,
    Exponent& current, unsigned coordinate, unsigned remainingDegree, unsigned& count) {
    if (coordinate == MR_HUMAN_RESTING_COMMON_COORDINATE_COUNT) {
        output[count++] = current;
        return;
    }
    for (unsigned exponent=0; exponent<=remainingDegree; ++exponent) {
        current[coordinate]=static_cast<std::uint8_t>(exponent);
        appendLexicographic(output,current,coordinate+1,remainingDegree-exponent,count);
    }
}
consteval auto monomialExponents() {
    std::array<Exponent,MR_HUMAN_RESTING_COMMON_CUBIC_TERM_COUNT> result{};
    Exponent current{};
    unsigned count=0;
    appendLexicographic(result,current,0,3,count);
    if(count!=MR_HUMAN_RESTING_COMMON_CUBIC_TERM_COUNT) throw "wrong cubic term count";
    return result;
}
inline constexpr auto kMonomialExponents=monomialExponents();
consteval bool validOrder() {
    for(std::size_t i=0;i<kMonomialExponents.size();++i) {
        unsigned degree=0;
        for(auto e:kMonomialExponents[i]) degree+=e;
        if(degree>3) return false;
        if(i) {
            bool less=false,greater=false;
            for(std::size_t j=0;j<7&&!less&&!greater;++j) {
                less=kMonomialExponents[i-1][j]<kMonomialExponents[i][j];
                greater=kMonomialExponents[i-1][j]>kMonomialExponents[i][j];
            }
            if(!less||greater) return false;
        }
    }
    return true;
}
static_assert(kMonomialExponents.size()==120&&validOrder());
inline unsigned termIndex(const Exponent& exponent) {
    for(unsigned i=0;i<kMonomialExponents.size();++i)
        if(kMonomialExponents[i]==exponent)return i;
    return MR_HUMAN_RESTING_COMMON_CUBIC_TERM_COUNT;
}
}
