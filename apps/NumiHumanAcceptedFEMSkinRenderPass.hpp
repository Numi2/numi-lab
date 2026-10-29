#pragma once
// ObjC++ presentation pass for AcceptedFEMSkinTransfer. Call updateDeltas only
// from the accepted-snapshot presentation callback, then encode after ordinary
// skinning and before mesh draw. It writes render vertices only.
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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
    using TetrahedralRow = Transfer::GPUTetrahedralTransferRow;

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
        directPositionRows_=true;
        for(const auto& row:rows) {
            unsigned unit=0;float sum=0.0f;
            for(unsigned c=0;c<3;++c) {
                if(row.coarseVertex[c]>=coarseCount||!std::isfinite(row.weight[c])||row.weight[c]<0) {
                    if(error)*error="render transfer row has invalid physical node or weight";
                    reset();return false;
                }
                unit+=row.weight[c]==1.0f;sum+=row.weight[c];
            }
            if(sum!=0.0f&&(sum!=1.0f||unit!=1||row.restDistance!=0.0f))directPositionRows_=false;
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
        return initializeStorage(device,rows.data(),rows.size(),coarseCount,
            "nm_human_accepted_fem_skin_transfer",error);
    }

    // Full-volume barycentric embedding. Preparation certifies source identity,
    // region, extrapolation and reference reconstruction; presentation gathers
    // the same accepted GPU coordinates used by the physical volume.
    bool initializeTetrahedral(id<MTLDevice> device,
                              const std::vector<TetrahedralRow>& rows,
                              const std::vector<Point>& reference,
                              std::string* error) {
        reset();
        if(!device||rows.empty()||reference.size()<4u||
           rows.size()>UINT32_MAX||reference.size()>UINT32_MAX) {
            if(error)*error="tetrahedral skin transfer has invalid dimensions";
            return false;
        }
        for(const auto& p:reference)for(const double value:p)
            if(!std::isfinite(value)||!std::isfinite(float(value))) {
                if(error)*error="tetrahedral skin reference cannot be represented in Float32";
                return false;
            }
        for(const auto& row:rows) {
            double sum=0.0,magnitude=0.0;
            for(unsigned c=0;c<4;++c) {
                if(row.node[c]>=reference.size()||!std::isfinite(row.weight[c])) {
                    if(error)*error="tetrahedral skin row has invalid physical node or weight";
                    return false;
                }
                for(unsigned j=0;j<c;++j)if(row.node[j]==row.node[c]) {
                    if(error)*error="tetrahedral skin row repeats a physical node";
                    return false;
                }
                sum+=row.weight[c];magnitude+=std::abs(double(row.weight[c]));
            }
            const auto p32=[&](unsigned c) {
                const auto& p=reference[row.node[c]];
                return Point{double(float(p[0])),double(float(p[1])),double(float(p[2]))};
            };
            const auto a=p32(0),b=p32(1),c=p32(2),d=p32(3);
            const Point u{b[0]-a[0],b[1]-a[1],b[2]-a[2]};
            const Point v{c[0]-a[0],c[1]-a[1],c[2]-a[2]};
            const Point w{d[0]-a[0],d[1]-a[1],d[2]-a[2]};
            const double determinant=u[0]*(v[1]*w[2]-v[2]*w[1])
                -u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0]);
            if(!std::isfinite(determinant)||!(determinant>0.0)) {
                if(error)*error="tetrahedral skin reference has nonpositive Float32 cell";
                return false;
            }
            const double bound=8.0*std::numeric_limits<float>::epsilon()*std::max(1.0,magnitude);
            if(std::abs(sum-1.0)>bound) {
                if(error)*error="tetrahedral skin row does not preserve translation";
                return false;
            }
        }
        if(!initializeStorage(device,rows.data(),rows.size(),std::uint32_t(reference.size()),
                "nm_human_accepted_fem_skin_tetrahedral",error))return false;
        auto* base=static_cast<std::array<float,4>*>([baseline_ contents]);
        auto* positions=static_cast<std::array<float,4>*>([delta_ contents]);
        if(!base||!positions) {
            if(error)*error="tetrahedral skin transfer buffers are inaccessible";
            reset();return false;
        }
        for(std::size_t i=0;i<reference.size();++i) {
            const auto& p=reference[i];
            base[i]={float(p[0]),float(p[1]),float(p[2]),1.0f};
            positions[i]=base[i];
        }
        tetrahedralRows_=true;absolutePositions_=true;
        if(error)error->clear();
        return true;
    }

    bool updateAcceptedTetrahedralPositions(
            const std::vector<std::array<float,4>>& positions,std::string* error) {
        if(!tetrahedralRows_||!delta_||positions.size()!=coarseCount_) {
            if(error)*error="tetrahedral accepted skin position count or mode mismatch";
            return false;
        }
        for(const auto& p:positions)for(unsigned a=0;a<3;++a)
            if(!std::isfinite(p[a])) {
                if(error)*error="nonfinite accepted tetrahedral skin position";
                return false;
            }
        void* bytes=[delta_ contents];
        if(!bytes) {
            if(error)*error="tetrahedral accepted skin position buffer is inaccessible";
            return false;
        }
        std::memcpy(bytes,positions.data(),positions.size()*sizeof(positions[0]));
        if(error)error->clear();
        return true;
    }

    // The renderer's presentation callback is serialized: present() waits for
    // its command buffer before returning, so this may reuse shared storage
    // between presentations. Do not call while an encode using these buffers
    // is in flight.
    bool updateAcceptedState(const std::vector<std::array<float,4>>& deltas,
                             const std::vector<Point>& currentBaseline,
                             std::string* error) {
        if (tetrahedralRows_ || !delta_ || deltas.size()!=coarseCount_ || currentBaseline.size()!=coarseCount_) {
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
        absolutePositions_=false;
        if (error) error->clear();
        return true;
    }

    // For a conforming detail vertex that is exactly one physical FEM node,
    // copy the accepted position without LBS + displacement cancellation.
    // No geometry is integrated here; all positions come from one accepted snapshot.
    bool updateAcceptedPositions(const std::vector<std::array<float,4>>& positions,
                                 const std::vector<Point>& currentBaseline,
                                 std::string* error) {
        if(!directPositionRows_) {
            if(error)*error="absolute skin positions require exact one-hot physical-node ancestry";
            return false;
        }
        if(!updateAcceptedState(positions,currentBaseline,error))return false;
        absolutePositions_=true;return true;
    }

    template<class EncoderCallbacks>
    bool encode(const EncoderCallbacks& encoder, void* meshVertices,
                std::uint32_t skinFirstVertex) const {
        if (!pipeline_ || !rows_ || !baseline_ || !delta_ || !meshVertices ||
            !encoder.valid()) return false;
        const std::uint32_t parameters[4] = {rowCount_, skinFirstVertex, absolutePositions_?1u:0u, 0u};
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
        absolutePositions_=false;directPositionRows_=false;tetrahedralRows_=false;
    }

private:
    bool initializeStorage(id<MTLDevice> device,const void* rowData,std::size_t rowCount,
                           std::uint32_t coarseCount,const char* functionName,
                           std::string* error) {
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
        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:functionName]];
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
        rows_ = [device newBufferWithBytes:rowData
                                    length:rowCount*sizeof(TetrahedralRow)
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
        rowCount_ = static_cast<std::uint32_t>(rowCount);
        coarseCount_ = coarseCount;
        if (error) error->clear();
        return true;
    }

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
    constant uint4& parameters [[buffer(0)]],
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
    if(parameters.z!=0u) {
        // One-hot admission guarantees a direct load, preserving every F32 bit.
        const uint selected=row.weight0==1.0f?row.coarse0:
            (row.weight1==1.0f?row.coarse1:row.coarse2);
        rv.position.xyz=coarseDelta[selected].xyz;
    } else rv.position.xyz+=displacement;

    const float3 e10=currentBaseline[row.coarse1].xyz-currentBaseline[row.coarse0].xyz;
    const float3 e20=currentBaseline[row.coarse2].xyz-currentBaseline[row.coarse0].xyz;
    const float3 oldCross=cross(e10,e20);
    const float3 newCross=parameters.z!=0u?cross(d1-d0,d2-d0):cross(e10+d1-d0,e20+d2-d0);
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

struct TetrahedralTransferRow {
    uint4 node;
    float4 weight;
};
kernel void nm_human_accepted_fem_skin_tetrahedral(
    constant uint4& parameters [[buffer(0)]],
    device const TetrahedralTransferRow* rows [[buffer(1)]],
    device const float4* accepted [[buffer(2)]],
    device MRVisualVertexGPUV2* vertices [[buffer(3)]],
    device const float4* reference [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=parameters.x)return;
    const TetrahedralTransferRow row=rows[i];
    const float3 x0=accepted[row.node.x].xyz,x1=accepted[row.node.y].xyz;
    const float3 x2=accepted[row.node.z].xyz,x3=accepted[row.node.w].xyz;
    device MRVisualVertexGPUV2& rv=vertices[parameters.y+i];
    rv.position.xyz=row.weight.x*x0+row.weight.y*x1+row.weight.z*x2+row.weight.w*x3;
    const float3 r0=reference[row.node.x].xyz;
    const float3 r1=reference[row.node.y].xyz-r0,r2=reference[row.node.z].xyz-r0,r3=reference[row.node.w].xyz-r0;
    const float det=dot(r1,cross(r2,r3));
    if(!(det>0.0f)||!isfinite(det))return;
    const float3x3 inverseReference=transpose(float3x3(cross(r2,r3),cross(r3,r1),cross(r1,r2)))/det;
    const float3x3 F=float3x3(x1-x0,x2-x0,x3-x0)*inverseReference;
    const float3x3 cofactorF=float3x3(cross(F[1],F[2]),cross(F[2],F[0]),cross(F[0],F[1]));
    const float3 n=cofactorF*rv.normalAndTangentSign.xyz;
    const float3 t=F*rv.tangent.xyz;
    const float n2=dot(n,n),t2=dot(t,t);
    if(n2>1.0e-20f&&isfinite(n2))rv.normalAndTangentSign.xyz=n*rsqrt(n2);
    if(t2>1.0e-20f&&isfinite(t2))rv.tangent.xyz=t*rsqrt(t2);
}

)MSL";
    }

    __strong id<MTLBuffer> rows_ = nil;
    __strong id<MTLBuffer> baseline_ = nil;
    __strong id<MTLBuffer> delta_ = nil;
    __strong id<MTLComputePipelineState> pipeline_ = nil;
    std::uint32_t rowCount_ = 0, coarseCount_ = 0;
    bool absolutePositions_=false,directPositionRows_=false,tetrahedralRows_=false;
};
} // namespace numi::human

