#include <metal_stdlib>

#include "metalrobo/visual_platform_types.h"

using namespace metal;

struct MRVisualCookTransformGPU {
    float4 positionRow0;
    float4 positionRow1;
    float4 positionRow2;
    float4 normalRow0;
    float4 normalRow1;
    float4 normalRow2;
    uint vertexCount;
    uint mirrored;
    uint2 reserved;
};

kernel void mr_visual_cook_bake_residual_v3(
    device MRVisualVertexGPUV2* vertices [[buffer(0)]],
    constant MRVisualCookTransformGPU& transform [[buffer(1)]],
    const uint index [[thread_position_in_grid]]
) {
    if (index >= transform.vertexCount) {
        return;
    }
    MRVisualVertexGPUV2 cookedVertex = vertices[index];
    const float4 position = float4(cookedVertex.position.xyz, 1.0f);
    cookedVertex.position = float4(
        dot(transform.positionRow0, position),
        dot(transform.positionRow1, position),
        dot(transform.positionRow2, position),
        1.0f
    );

    const float3 normal = normalize(float3(
        dot(
            transform.normalRow0.xyz,
            cookedVertex.normalAndTangentSign.xyz
        ),
        dot(
            transform.normalRow1.xyz,
            cookedVertex.normalAndTangentSign.xyz
        ),
        dot(
            transform.normalRow2.xyz,
            cookedVertex.normalAndTangentSign.xyz
        )
    ));
    float3 tangent = float3(
        dot(transform.positionRow0.xyz, cookedVertex.tangent.xyz),
        dot(transform.positionRow1.xyz, cookedVertex.tangent.xyz),
        dot(transform.positionRow2.xyz, cookedVertex.tangent.xyz)
    );
    tangent -= normal * dot(normal, tangent);
    if (dot(tangent, tangent) <= 1.0e-12f) {
        const float3 reference =
            abs(normal.z) < 0.999f
            ? float3(0.0f, 0.0f, 1.0f)
            : float3(0.0f, 1.0f, 0.0f);
        tangent = cross(reference, normal);
    }
    tangent = normalize(tangent);
    cookedVertex.normalAndTangentSign = float4(
        normal,
        transform.mirrored != 0u
            ? -cookedVertex.normalAndTangentSign.w
            : cookedVertex.normalAndTangentSign.w
    );
    cookedVertex.tangent = float4(tangent, 0.0f);
    vertices[index] = cookedVertex;
}
