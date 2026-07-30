#include <metal_stdlib>

#include "VisualPBR.metal"

using namespace metal;
using namespace metalrobo_pbr;

namespace {

float3 cubeDirection(
    const uint face,
    const float2 coordinate
) {
    const float2 uv = 2.0f * coordinate - 1.0f;
    switch (face) {
    case 0u:
        return normalize(float3(1.0f, -uv.y, -uv.x));
    case 1u:
        return normalize(float3(-1.0f, -uv.y, uv.x));
    case 2u:
        return normalize(float3(uv.x, 1.0f, uv.y));
    case 3u:
        return normalize(float3(uv.x, -1.0f, -uv.y));
    case 4u:
        return normalize(float3(uv.x, -uv.y, 1.0f));
    default:
        return normalize(float3(-uv.x, -uv.y, -1.0f));
    }
}

float2 equirectangularCoordinate(const float3 direction) {
    return float2(
        atan2(direction.y, direction.x) /
                (2.0f * M_PI_F) +
            0.5f,
        asin(clamp(direction.z, -1.0f, 1.0f)) /
                M_PI_F +
            0.5f
    );
}

float3 cosineDirection(
    const float2 sample,
    const float3 normal
) {
    const float radius = sqrt(sample.x);
    const float phi = 2.0f * M_PI_F * sample.y;
    const float3 local = float3(
        radius * cos(phi),
        radius * sin(phi),
        sqrt(max(0.0f, 1.0f - sample.x))
    );
    float3 tangent;
    float3 bitangent;
    basis(normal, tangent, bitangent);
    return normalize(
        tangent * local.x +
        bitangent * local.y +
        normal * local.z
    );
}

} // namespace

kernel void mr_environment_equirect_to_cube(
    texture2d<float, access::sample> source [[texture(0)]],
    texturecube<float, access::write> destination [[texture(1)]],
    const uint3 coordinate [[thread_position_in_grid]]
) {
    const uint size = destination.get_width();
    if (coordinate.x >= size || coordinate.y >= size ||
        coordinate.z >= 6u) {
        return;
    }
    constexpr sampler sourceSampler(
        coord::normalized,
        address::repeat,
        filter::linear
    );
    const float2 pixel =
        (float2(coordinate.xy) + 0.5f) / float(size);
    const float3 direction =
        cubeDirection(coordinate.z, pixel);
    destination.write(
        max(
            source.sample(
                sourceSampler,
                equirectangularCoordinate(direction)
            ),
            0.0f
        ),
        coordinate.xy,
        coordinate.z
    );
}

kernel void mr_environment_diffuse_irradiance(
    texturecube<float, access::sample> source [[texture(0)]],
    texturecube<float, access::write> destination [[texture(1)]],
    constant uint& sampleCount [[buffer(0)]],
    const uint3 coordinate [[thread_position_in_grid]]
) {
    const uint size = destination.get_width();
    if (coordinate.x >= size || coordinate.y >= size ||
        coordinate.z >= 6u) {
        return;
    }
    constexpr sampler cubeSampler(
        coord::normalized,
        address::clamp_to_edge,
        filter::linear,
        mip_filter::linear
    );
    const float2 pixel =
        (float2(coordinate.xy) + 0.5f) / float(size);
    const float3 normal =
        cubeDirection(coordinate.z, pixel);
    float3 irradiance = 0.0f;
    for (uint sample = 0u;
         sample < sampleCount;
         ++sample) {
        const float3 direction = cosineDirection(
            hammersley(sample, sampleCount),
            normal
        );
        irradiance += source.sample(
            cubeSampler,
            direction,
            level(0.0f)
        ).xyz;
    }
    destination.write(
        float4(
            M_PI_F * irradiance /
                float(max(sampleCount, 1u)),
            1.0f
        ),
        coordinate.xy,
        coordinate.z
    );
}

kernel void mr_environment_prefilter_specular(
    texturecube<float, access::sample> source [[texture(0)]],
    texturecube<float, access::write> destination [[texture(1)]],
    constant float4& parameters [[buffer(0)]],
    const uint3 coordinate [[thread_position_in_grid]]
) {
    const uint size = uint(parameters.x);
    const uint sampleCount = uint(parameters.y);
    const float roughness = parameters.z;
    const float sourceFaceSize = parameters.w;
    if (coordinate.x >= size || coordinate.y >= size ||
        coordinate.z >= 6u) {
        return;
    }
    constexpr sampler cubeSampler(
        coord::normalized,
        address::clamp_to_edge,
        filter::linear,
        mip_filter::linear
    );
    const float2 pixel =
        (float2(coordinate.xy) + 0.5f) / float(size);
    const float3 normal =
        cubeDirection(coordinate.z, pixel);
    if (roughness <= 1.0e-6f) {
        destination.write(
            source.sample(
                cubeSampler,
                normal,
                level(0.0f)
            ),
            coordinate.xy,
            coordinate.z
        );
        return;
    }
    const float3 view = normal;
    const float alpha =
        max(roughness * roughness, 0.0025f);
    const float sourceTexelSolidAngle =
        4.0f * M_PI_F /
        max(6.0f * sourceFaceSize * sourceFaceSize, 1.0f);
    float3 accumulated = 0.0f;
    float weight = 0.0f;
    for (uint sample = 0u;
         sample < sampleCount;
         ++sample) {
        const float3 halfVector = sampleGGXVNDF(
            view,
            alpha,
            hammersley(sample, sampleCount)
        );
        const float3 light =
            normalize(reflect(-view, halfVector));
        const float noL = max(dot(normal, light), 0.0f);
        if (noL <= 0.0f) {
            continue;
        }
        const float noH = max(dot(normal, halfVector), 0.0f);
        const float voH = max(dot(view, halfVector), 1.0e-5f);
        const float pdf =
            distributionGGX(noH, roughness) * noH /
            max(4.0f * voH, 1.0e-5f);
        const float sampleSolidAngle =
            1.0f /
            max(float(sampleCount) * pdf, 1.0e-5f);
        const float sourceLod = roughness <= 1.0e-4f
            ? 0.0f
            : max(
                  0.5f * log2(
                      sampleSolidAngle /
                      sourceTexelSolidAngle
                  ),
                  0.0f
              );
        accumulated += source.sample(
            cubeSampler,
            light,
            level(sourceLod)
        ).xyz * noL;
        weight += noL;
    }
    destination.write(
        float4(
            accumulated / max(weight, 1.0e-5f),
            1.0f
        ),
        coordinate.xy,
        coordinate.z
    );
}

kernel void mr_environment_integrate_brdf(
    texture2d<half, access::write> destination [[texture(0)]],
    constant uint& sampleCount [[buffer(0)]],
    const uint2 coordinate [[thread_position_in_grid]]
) {
    const uint width = destination.get_width();
    const uint height = destination.get_height();
    if (coordinate.x >= width || coordinate.y >= height) {
        return;
    }
    const float noV =
        (float(coordinate.x) + 0.5f) / float(width);
    const float roughness =
        (float(coordinate.y) + 0.5f) / float(height);
    const float3 view = float3(
        sqrt(max(0.0f, 1.0f - noV * noV)),
        0.0f,
        noV
    );
    const float alpha =
        max(roughness * roughness, 0.0025f);
    float scale = 0.0f;
    float bias = 0.0f;
    for (uint sample = 0u;
         sample < sampleCount;
         ++sample) {
        const float3 halfVector = sampleGGXVNDF(
            view,
            alpha,
            hammersley(sample, sampleCount)
        );
        const float3 light =
            normalize(reflect(-view, halfVector));
        const float noL = max(light.z, 0.0f);
        const float voH = max(dot(view, halfVector), 0.0f);
        if (noL <= 0.0f) {
            continue;
        }
        const float smithG2 =
            4.0f * noL * noV *
            visibilitySmithGGX(noV, noL, roughness);
        const float visibilityWeight =
            smithG2 /
            max(smithG1GGX(noV, roughness), 1.0e-5f);
        const float fresnel =
            pow(1.0f - voH, 5.0f);
        scale += (1.0f - fresnel) * visibilityWeight;
        bias += fresnel * visibilityWeight;
    }
    destination.write(
        half4(
            half(scale / float(max(sampleCount, 1u))),
            half(bias / float(max(sampleCount, 1u))),
            half(0.0f),
            half(1.0f)
        ),
        coordinate
    );
}
