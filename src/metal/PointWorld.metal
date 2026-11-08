#include <metal_stdlib>

using namespace metal;

// PointWorld's release input is ImageNet-normalized 320x180 sRGB.  These
// kernels are deliberately independent of simulator state: the visual
// provider owns the source textures and the model session only borrows them.
kernel void mr_pointworld_normalize_rgb(
    device const uchar4* source [[buffer(0)]],
    device float4* normalized [[buffer(1)]],
    constant uint& pixelCount [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= pixelCount) {
        return;
    }
    constexpr float3 mean = float3(0.485f, 0.456f, 0.406f);
    constexpr float3 standardDeviation = float3(0.229f, 0.224f, 0.225f);
    const float3 srgb = float3(source[index].xyz) / 255.0f;
    normalized[index] = float4((srgb - mean) / standardDeviation, 1.0f);
}

// Metric depth backprojection uses the immutable camera calibration carried by
// PointWorldObservationV1.  A zero w marks invalid depth; no synthetic scene
// point is introduced for an invalid pixel.
kernel void mr_pointworld_backproject_depth(
    device const float* depth [[buffer(0)]],
    device const uchar* validity [[buffer(1)]],
    device float4* cameraPoints [[buffer(2)]],
    constant float4& intrinsics [[buffer(3)]], // fx, fy, cx, cy
    constant uint2& imageSize [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    const uint pixelCount = imageSize.x * imageSize.y;
    if (index >= pixelCount || validity[index] == 0u || !isfinite(depth[index]) || depth[index] <= 0.0f) {
        if (index < pixelCount) {
            cameraPoints[index] = float4(0.0f);
        }
        return;
    }
    const uint x = index % imageSize.x;
    const uint y = index / imageSize.x;
    const float z = depth[index];
    cameraPoints[index] = float4(
        (float(x) - intrinsics.z) * z / intrinsics.x,
        (float(y) - intrinsics.w) * z / intrinsics.y,
        z,
        1.0f
    );
}
