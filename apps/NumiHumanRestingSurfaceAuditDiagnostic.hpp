#pragma once
#include "../include/metalrobo/numi_human_resting_visual_gpu.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

namespace numiHumanRestingSurfaceAudit {
inline std::uint32_t classifyRenderedTriangleAreaFailure(
    const mr_float4& a,const mr_float4& b,const mr_float4& c) {
    const float ux=b.x-a.x,uy=b.y-a.y,uz=b.z-a.z;
    const float vx=c.x-a.x,vy=c.y-a.y,vz=c.z-a.z;
    const float nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
    if(!std::isfinite(nx)||!std::isfinite(ny)||!std::isfinite(nz))
        return MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA;
    if(nx==0.0f&&ny==0.0f&&nz==0.0f)
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
