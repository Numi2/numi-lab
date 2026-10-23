#pragma once
// ObjC++ presentation pass for AcceptedFEMSkinTransfer. Call updateDeltas only
// from the accepted-snapshot presentation callback, then encode after ordinary
// skinning and before mesh draw. It writes render vertices only.
#import <Metal/Metal.h>
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/visual_platform_types.h"
#include "NumiHumanAcceptedFEMSkinTransfer.hpp"

namespace numi::human {
class AcceptedFEMSkinRenderPass {
public:
    using Transfer = AcceptedFEMSkinTransfer;
    using Point = Transfer::Point;
    using Row = Transfer::GPUTransferRow;

    bool initialize(id<MTLDevice> device, const std::vector<Row>& rows,
                    std::uint32_t coarseCount, std::string* error) {
        reset();
        if (!device || rows.empty() || coarseCount==0) {
            if (error) *error = "render transfer needs device, rows, and coarse rest points";
            return false;
        }
        if (rows.size() > UINT32_MAX || coarseCount > UINT32_MAX) {
            if (error) *error = "render transfer exceeds uint32 index range";
            return false;
        }
        static_assert(sizeof(Row) == 32, "transfer row ABI");
        static_assert(sizeof(MRVisualVertexGPUV2) == 80, "renderer vertex ABI size");
        static_assert(offsetof(MRVisualVertexGPUV2, position) == 0, "renderer position offset");
        static_assert(offsetof(MRVisualVertexGPUV2, normalAndTangentSign) == 16, "renderer normal offset");
        static_assert(offsetof(MRVisualVertexGPUV2, tangent) == 32, "renderer tangent offset");
        static_assert(offsetof(MRVisualVertexGPUV2, texcoord01) == 48, "renderer texcoord offset");
        static_assert(offsetof(MRVisualVertexGPUV2, color) == 64, "renderer color offset");
        static_assert(sizeof(MRVisualVertexGPUV2) == 80, "renderer vertex ABI size");
        static_assert(offsetof(MRVisualVertexGPUV2, position) == 0, "renderer position offset");
        static_assert(offsetof(MRVisualVertexGPUV2, normalAndTangentSign) == 16, "renderer normal offset");
        static_assert(offsetof(MRVisualVertexGPUV2, tangent) == 32, "renderer tangent offset");
        static_assert(offsetof(MRVisualVertexGPUV2, texcoord01) == 48, "renderer texcoord offset");
        static_assert(offsetof(MRVisualVertexGPUV2, color) == 64, "renderer color offset");
        NSError* nsError = nil;
        NSString* source = [NSString stringWithUTF8String:metalSource()];
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&nsError];
        if (!library) {
            if (error) *error = nsError ? std::string(nsError.localizedDescription.UTF8String)
                                        : "Metal transfer library compilation failed";
            return false;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"nm_human_accepted_fem_skin_transfer"];
        if (!function) {
            if (error) *error = "Metal transfer function is absent";
            return false;
        }
        pipeline_ = [device newComputePipelineStateWithFunction:function error:&nsError];
        if (!pipeline_) {
            if (error) *error = nsError ? std::string(nsError.localizedDescription.UTF8String)
                                        : "Metal transfer pipeline creation failed";
            return false;
        }
        rows_ = [device newBufferWithBytes:rows.data()
                                    length:rows.size()*sizeof(Row)
                                   options:MTLResourceStorageModeShared];
        const std::vector<std::array<float,4>> zero(coarseCount, {0,0,0,0});
        baseline_ = [device newBufferWithBytes:zero.data()
                                        length:zero.size()*sizeof(zero[0])
                                       options:MTLResourceStorageModeShared];
        delta_ = [device newBufferWithBytes:zero.data()
                                     length:zero.size()*sizeof(zero[0])
                                    options:MTLResourceStorageModeShared];
        if (!rows_ || !baseline_ || !delta_) {
            if (error) *error = "Metal transfer buffer allocation failed";
            reset();
            return false;
        }
        rowCount_ = static_cast<std::uint32_t>(rows.size());
        coarseCount_ = coarseCount;
        if (error) error->clear();
        return true;
    }

    // The renderer's presentation callback is serialized: present() waits for
    // its command buffer before returning, so this may reuse shared storage
    // between presentations. Do not call while an encode using these buffers
    // is in flight.
    bool updateAcceptedState(const std::vector<std::array<float,4>>& deltas,
                             const std::vector<Point>& currentBaseline,
                             std::string* error) {
        if (!delta_ || deltas.size()!=coarseCount_ || currentBaseline.size()!=coarseCount_) {
            if (error) *error = "accepted FEM displacement/baseline count mismatch";
            return false;
        }
        for (std::size_t i=0;i<deltas.size();++i) {
            const auto& d=deltas[i]; const auto& p=currentBaseline[i];
            const float x=float(p[0]), y=float(p[1]), z=float(p[2]);
            if (!std::isfinite(d[0]) || !std::isfinite(d[1]) || !std::isfinite(d[2]) ||
                !std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) ||
                !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                if (error) *error = "nonfinite or out-of-range accepted transfer field";
                return false;
            }
        }
        auto* baselineBytes=static_cast<std::array<float,4>*>([baseline_ contents]);
        void* deltaBytes=[delta_ contents];
        if (!baselineBytes || !deltaBytes) {
            if (error) *error = "accepted transfer shared buffers are not CPU accessible";
            return false;
        }
        std::memcpy(deltaBytes, deltas.data(), deltas.size()*sizeof(deltas[0]));
        for (std::size_t i=0;i<currentBaseline.size();++i) {
            const auto& p=currentBaseline[i];
            baselineBytes[i]={float(p[0]),float(p[1]),float(p[2]),1.0f};
        }
        if (error) error->clear();
        return true;
    }

    template<class EncoderCallbacks>
    bool encode(const EncoderCallbacks& encoder, void* meshVertices,
                std::uint32_t skinFirstVertex) const {
        if (!pipeline_ || !rows_ || !baseline_ || !delta_ || !meshVertices ||
            !encoder.valid()) return false;
        const std::uint32_t parameters[2] = {rowCount_, skinFirstVertex};
        encoder.setPipeline(encoder.context, (__bridge void*)pipeline_);
        encoder.setBytes(encoder.context, parameters, sizeof(parameters), 0);
        encoder.setBuffer(encoder.context, (__bridge void*)rows_, 0, 1);
        encoder.setBuffer(encoder.context, (__bridge void*)delta_, 0, 2);
        encoder.setBuffer(encoder.context, meshVertices, 0, 3);
        encoder.setBuffer(encoder.context, (__bridge void*)baseline_, 0, 4);
        encoder.dispatchThreads(encoder.context, rowCount_, 64);
        return true;
    }

    std::uint32_t rowCount() const { return rowCount_; }
    std::uint32_t coarseCount() const { return coarseCount_; }

    void reset() {
        rows_ = nil;
        baseline_ = nil;
        delta_ = nil;
        pipeline_ = nil;
        rowCount_ = coarseCount_ = 0;
    }

private:
    static const char* metalSource() {
        return R"MSL(
#include <metal_stdlib>
using namespace metal;
struct TransferRow {
    uint coarse0; uint coarse1; uint coarse2; uint donor;
    float weight0; float weight1; float weight2; float restDistance;
};
struct MRVisualVertexGPUV2 {
    float4 position;
    float4 normalAndTangentSign;
    float4 tangent;
    float4 texcoord01;
    float4 color;
};
inline float3 rotateNormal(float3 v, float3 from, float3 to) {
    float c = clamp(dot(from,to), -1.0f, 1.0f);
    float3 axis = cross(from,to);
    float q2 = dot(axis,axis) + (1.0f+c)*(1.0f+c);
    if (q2 <= 1.0e-14f) {
        if (c > 0.0f) return v;
        float3 t = normalize(cross(from, fabs(from.x)<0.8f ? float3(1,0,0) : float3(0,1,0)));
        return 2.0f*dot(v,t)*t-v;
    }
    float inv = rsqrt(q2);
    float3 q = axis*inv;
    float w = (1.0f+c)*inv;
    return v + 2.0f*cross(q, cross(q,v)+w*v);
}
kernel void nm_human_accepted_fem_skin_transfer(
    constant uint2& parameters [[buffer(0)]],
    device const TransferRow* rows [[buffer(1)]],
    device const float4* coarseDelta [[buffer(2)]],
    device MRVisualVertexGPUV2* vertices [[buffer(3)]],
    device const float4* currentBaseline [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= parameters.x) return;
    const TransferRow row = rows[i];
    const float sum = row.weight0+row.weight1+row.weight2;
    if (!(sum > 0.0f)) return;
    const float3 d0=coarseDelta[row.coarse0].xyz;
    const float3 d1=coarseDelta[row.coarse1].xyz;
    const float3 d2=coarseDelta[row.coarse2].xyz;
    const float3 displacement=row.weight0*d0+row.weight1*d1+row.weight2*d2;
    device MRVisualVertexGPUV2& rv=vertices[parameters.y+i];
    rv.position.xyz+=displacement;

    const float3 e10=currentBaseline[row.coarse1].xyz-currentBaseline[row.coarse0].xyz;
    const float3 e20=currentBaseline[row.coarse2].xyz-currentBaseline[row.coarse0].xyz;
    const float3 oldCross=cross(e10,e20);
    const float3 newCross=cross(e10+d1-d0,e20+d2-d0);
    const float oldLen=length(oldCross),newLen=length(newCross);
    if (oldLen>1.0e-12f && newLen>1.0e-12f) {
        const float3 oldN=oldCross/oldLen,newN=newCross/newLen;
        float3 n=rotateNormal(rv.normalAndTangentSign.xyz,oldN,newN);
        float3 t=rotateNormal(rv.tangent.xyz,oldN,newN);
        const float n2=dot(n,n),t2=dot(t,t);
        if (n2>1.0e-12f) rv.normalAndTangentSign.xyz=n*rsqrt(n2);
        if (t2>1.0e-12f) rv.tangent.xyz=t*rsqrt(t2);
    }
}
)MSL";
    }

    __strong id<MTLBuffer> rows_ = nil;
    __strong id<MTLBuffer> baseline_ = nil;
    __strong id<MTLBuffer> delta_ = nil;
    __strong id<MTLComputePipelineState> pipeline_ = nil;
    std::uint32_t rowCount_ = 0, coarseCount_ = 0;
};
} // namespace numi::human

