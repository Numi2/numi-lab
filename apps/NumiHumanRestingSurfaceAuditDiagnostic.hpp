#pragma once
#include "../include/metalrobo/numi_human_resting_visual_gpu.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

namespace numiHumanRestingSurfaceAudit {
struct ExactBinary32Product {
    std::uint64_t significand;
    int power;
    bool negative;
};

inline ExactBinary32Product exactBinary32Product(float x,float y) {
    const std::uint32_t xBits=std::bit_cast<std::uint32_t>(x);
    const std::uint32_t yBits=std::bit_cast<std::uint32_t>(y);
    const std::uint32_t xExponent=(xBits>>23u)&0xffu;
    const std::uint32_t yExponent=(yBits>>23u)&0xffu;
    std::uint64_t xSignificand=xBits&0x7fffffu,ySignificand=yBits&0x7fffffu;
    int xPower=-149,yPower=-149;
    if(xExponent){xSignificand|=0x800000u;xPower=int(xExponent)-150;}
    if(yExponent){ySignificand|=0x800000u;yPower=int(yExponent)-150;}
    std::uint64_t significand=xSignificand*ySignificand;
    int power=xPower+yPower;
    if(significand)while((significand&1u)==0u){significand>>=1u;++power;}
    return {significand,power,((xBits^yBits)&0x80000000u)!=0u};
}

// The compensated predicate is exact for binary32 edge values. The full
// point-coordinate fallback below handles origin subtraction cancellation.
inline bool exactBinary32ProductsEqual(float a,float b,float c,float d) {
    if(!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(c)||!std::isfinite(d))return false;
    const float ab=a*b,cd=c*d;
    if(ab!=cd)return false;
    const float abError=std::fma(a,b,-ab),cdError=std::fma(c,d,-cd);
    const float minimumNormal=std::numeric_limits<float>::min();
    if(std::isfinite(abError)&&std::isfinite(cdError)&&abError!=0.0f&&cdError!=0.0f&&
       std::abs(abError)>=minimumNormal&&std::abs(cdError)>=minimumNormal)return abError==cdError;
    const auto lhs=exactBinary32Product(a,b),rhs=exactBinary32Product(c,d);
    if(lhs.significand==0||rhs.significand==0)return lhs.significand==rhs.significand;
    return lhs.negative==rhs.negative&&lhs.significand==rhs.significand&&lhs.power==rhs.power;
}

inline bool exactBinary32RequiresPointFallback(float value) {
    const std::uint32_t bits=std::bit_cast<std::uint32_t>(value);
    const std::uint32_t exponent=(bits>>23u)&0xffu;
    const std::uint32_t fraction=bits&0x7fffffu;
    return (exponent==0u&&fraction!=0u)||(exponent>0u&&exponent<25u);
}

inline float exactBinary32DifferenceTail(float a,float b,float difference) {
    const float bVirtual=a-difference;
    const float aVirtual=difference+bVirtual;
    const float bRound=bVirtual-b;
    const float aRound=a-aVirtual;
    return aRound+bRound;
}

inline void addExactBinary32Word(std::array<std::uint64_t,9>& accumulator,std::size_t limb,std::uint64_t word) {
    while(word!=0u&&limb<accumulator.size()) {
        const std::uint64_t previous=accumulator[limb];
        accumulator[limb]=previous+word;
        word=accumulator[limb]<previous?1u:0u;
        ++limb;
    }
}

inline void accumulateExactBinary32Product(std::array<std::uint64_t,9>& positive,
    std::array<std::uint64_t,9>& negative,float a,float b,bool negate) {
    const auto product=exactBinary32Product(a,b);
    if(product.significand==0)return;
    const int shift=product.power+298;
    const auto bitShift=unsigned(shift&63);
    const auto limb=std::size_t(shift>>6);
    const std::uint64_t low=product.significand<<bitShift;
    const std::uint64_t high=bitShift?product.significand>>(64u-bitShift):0u;
    auto& accumulator=(product.negative!=negate)?negative:positive;
    addExactBinary32Word(accumulator,limb,low);
    if(high)addExactBinary32Word(accumulator,limb+1u,high);
}

inline bool exactBinary32PointAreaComponentIsZero(
    float a0,float b0,float c0,float a1,float b1,float c1) {
    std::array<std::uint64_t,9> positive{},negative{};
    accumulateExactBinary32Product(positive,negative,b0,c1,false);
    accumulateExactBinary32Product(positive,negative,b0,a1,true);
    accumulateExactBinary32Product(positive,negative,a0,c1,true);
    accumulateExactBinary32Product(positive,negative,b1,c0,true);
    accumulateExactBinary32Product(positive,negative,b1,a0,false);
    accumulateExactBinary32Product(positive,negative,a1,c0,false);
    return positive==negative;
}

inline bool exactBinary32TriangleIsZero(
    float ax,float ay,float az,float bx,float by,float bz,float cx,float cy,float cz) {
    const float ux=bx-ax,uy=by-ay,uz=bz-az;
    const float vx=cx-ax,vy=cy-ay,vz=cz-az;
    const bool safeCompensatedRange=
        !exactBinary32RequiresPointFallback(ax)&&!exactBinary32RequiresPointFallback(ay)&&
        !exactBinary32RequiresPointFallback(az)&&!exactBinary32RequiresPointFallback(bx)&&
        !exactBinary32RequiresPointFallback(by)&&!exactBinary32RequiresPointFallback(bz)&&
        !exactBinary32RequiresPointFallback(cx)&&!exactBinary32RequiresPointFallback(cy)&&
        !exactBinary32RequiresPointFallback(cz)&&std::isfinite(ux)&&std::isfinite(uy)&&std::isfinite(uz)&&
        std::isfinite(vx)&&std::isfinite(vy)&&std::isfinite(vz);
    const bool exactEdges=safeCompensatedRange&&
        exactBinary32DifferenceTail(bx,ax,ux)==0.0f&&
        exactBinary32DifferenceTail(by,ay,uy)==0.0f&&
        exactBinary32DifferenceTail(bz,az,uz)==0.0f&&
        exactBinary32DifferenceTail(cx,ax,vx)==0.0f&&
        exactBinary32DifferenceTail(cy,ay,vy)==0.0f&&
        exactBinary32DifferenceTail(cz,az,vz)==0.0f;
    if(!exactEdges) {
        return exactBinary32PointAreaComponentIsZero(ay,by,cy,az,bz,cz)&&
            exactBinary32PointAreaComponentIsZero(az,bz,cz,ax,bx,cx)&&
            exactBinary32PointAreaComponentIsZero(ax,bx,cx,ay,by,cy);
    }
    return exactBinary32ProductsEqual(uy,vz,uz,vy)&&
        exactBinary32ProductsEqual(uz,vx,ux,vz)&&
        exactBinary32ProductsEqual(ux,vy,uy,vx);
}

inline std::uint32_t classifyRenderedTriangleAreaFailure(
    const mr_float4& a,const mr_float4& b,const mr_float4& c) {
    const float ux=b.x-a.x,uy=b.y-a.y,uz=b.z-a.z;
    const float vx=c.x-a.x,vy=c.y-a.y,vz=c.z-a.z;
    const float nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
    if(!std::isfinite(nx)||!std::isfinite(ny)||!std::isfinite(nz))
        return MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA;
    if(exactBinary32TriangleIsZero(a.x,a.y,a.z,b.x,b.y,b.z,c.x,c.y,c.z))
        return MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA;
    return MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE;
}

inline const char* triangleFailureName(std::uint32_t kind) {
    switch(kind) {
        case MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA:return "nonfinite_area";
        case MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA:return "exact_zero_area";
        default:return "none";
    }
}

inline std::string float32BitsHex(float value) {
    std::ostringstream out;out<<std::hex<<std::setw(8)<<std::setfill('0')<<std::bit_cast<std::uint32_t>(value);
    return out.str();
}

inline std::string describeSurfaceFailure(
    unsigned stableId,unsigned auditIndex,unsigned indexBufferStart,unsigned status,float relativeError,
    const MRHumanRestingSurfaceFailureGPU& firstFailure,std::string_view sourcePackContentHash) {
    std::ostringstream out;out<<std::setprecision(std::numeric_limits<float>::max_digits10)
        <<"physical_endpoint=accepted surface_endpoint=rejected stable_id="<<stableId
        <<" audit_index="<<auditIndex<<" status="<<status
        <<" relative_volume_error="<<relativeError
        <<" source_pack_content_hash="<<sourcePackContentHash;
    if(status&1u)out<<" failure=volume_owner_mismatch";
    if(!(status&2u))out<<" first_invalid_triangle=not_applicable";
    if(status&2u) {
        if(firstFailure.surfaceTriangleKind.x==auditIndex&&
           firstFailure.surfaceTriangleKind.z!=MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE) {
            const auto triangle=firstFailure.surfaceTriangleKind.y;
            out<<" first_invalid_triangle_local="<<triangle
               <<" index_buffer_offset="<<(indexBufferStart+3u*triangle)
               <<" area_failure="<<triangleFailureName(firstFailure.surfaceTriangleKind.z)
               <<" mesh_vertex_indices=["<<firstFailure.vertexIndices.x<<','
               <<firstFailure.vertexIndices.y<<','<<firstFailure.vertexIndices.z<<']';
            out<<" rendered_positions_m_hex=[";
            for(unsigned k=0;k<3;++k) {
                if(k)out<<';';
                const auto& p=firstFailure.renderedPositions[k];
                out<<std::hexfloat<<p.x<<','<<p.y<<','<<p.z<<std::defaultfloat;
            }
            out<<"] rendered_positions_f32_bits=[";
            for(unsigned k=0;k<3;++k) {
                if(k)out<<';';
                const auto& p=firstFailure.renderedPositions[k];
                out<<float32BitsHex(p.x)<<','<<float32BitsHex(p.y)<<','<<float32BitsHex(p.z);
            }
            out<<']';
        } else {
            out<<" first_invalid_triangle=unavailable diagnostic_record=missing_or_mismatched";
        }
    }
    return out.str();
}
} // namespace numiHumanRestingSurfaceAudit
