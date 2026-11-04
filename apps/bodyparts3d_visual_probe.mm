#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/VisualPlatform.hpp"
#include "NumiHumanRestingInspectionLayers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Result>
void require(const Result& result, const char* operation) {
    if (!result.succeeded()) {
        throw std::runtime_error(
            std::string{operation} + ": " + result.message
        );
    }
}

metalrobo::WorldPose cameraToward(
    const mr_float4 position,
    const mr_float4 target
) {
    const float x = target.x - position.x;
    const float y = target.y - position.y;
    const float z = target.z - position.z;
    const float inverseLength = 1.0f / std::sqrt(x * x + y * y + z * z);
    const mr_float4 forward{x * inverseLength, y * inverseLength, z * inverseLength, 0.0f};
    // Renderer camera +z is forward and screen-up is local -y.  Preserve a
    // world-z horizon rather than accepting the shortest-arc camera rotation,
    // whose roll changes when an inspection camera crosses an axis.
    const mr_float4 preferredUp{0.0f, 0.0f, 1.0f, 0.0f};
    mr_float4 localY{-preferredUp.x, -preferredUp.y, -preferredUp.z, 0.0f};
    mr_float4 localX{
        localY.y * forward.z - localY.z * forward.y,
        localY.z * forward.x - localY.x * forward.z,
        localY.x * forward.y - localY.y * forward.x,
        0.0f,
    };
    const float localXLength = std::sqrt(
        localX.x * localX.x + localX.y * localX.y + localX.z * localX.z
    );
    if (localXLength < 1.0e-5f) {
        throw std::runtime_error("BodyParts3D inspection camera is parallel to world up");
    }
    localX.x /= localXLength;
    localX.y /= localXLength;
    localX.z /= localXLength;
    localY = {
        forward.y * localX.z - forward.z * localX.y,
        forward.z * localX.x - forward.x * localX.z,
        forward.x * localX.y - forward.y * localX.x,
        0.0f,
    };
    const float m00 = localX.x;
    const float m01 = localY.x;
    const float m02 = forward.x;
    const float m10 = localX.y;
    const float m11 = localY.y;
    const float m12 = forward.y;
    const float m20 = localX.z;
    const float m21 = localY.z;
    const float m22 = forward.z;
    mr_float4 orientation{};
    const float trace = m00 + m11 + m22;
    if (trace > 0.0f) {
        const float scale = 2.0f * std::sqrt(trace + 1.0f);
        orientation = {(m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25f * scale};
    } else if (m00 > m11 && m00 > m22) {
        const float scale = 2.0f * std::sqrt(1.0f + m00 - m11 - m22);
        orientation = {0.25f * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale};
    } else if (m11 > m22) {
        const float scale = 2.0f * std::sqrt(1.0f + m11 - m00 - m22);
        orientation = {(m01 + m10) / scale, 0.25f * scale, (m12 + m21) / scale, (m02 - m20) / scale};
    } else {
        const float scale = 2.0f * std::sqrt(1.0f + m22 - m00 - m11);
        orientation = {(m02 + m20) / scale, (m12 + m21) / scale, 0.25f * scale, (m10 - m01) / scale};
    }
    return {position, orientation};
}

mr_float4 normalizedDirection(const mr_float4 value, const char* context) {
    const float lengthSquared = value.x * value.x + value.y * value.y + value.z * value.z;
    if (!(std::isfinite(lengthSquared) && lengthSquared > 1.0e-10f)) {
        throw std::runtime_error(std::string(context) + " is degenerate");
    }
    const float inverseLength = 1.0f / std::sqrt(lengthSquared);
    return {value.x * inverseLength, value.y * inverseLength, value.z * inverseLength, 0.0f};
}

struct PackFraming {
    mr_float4 centre{};
    mr_float4 minimum{};
    mr_float4 maximum{};
    float distance = 0.0f;
};

PackFraming packFraming(
    const metalrobo::VisualAssetPackV2& pack,
    const bool tightFrame,
    const bool focusLowerThird,
    const bool fillFrame,
    const bool supineView
) {
    if (pack.vertices.empty()) {
        throw std::runtime_error("BodyParts3D visual pack has no vertices");
    }
    mr_float4 minimum{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        0.0f,
    };
    mr_float4 maximum{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        0.0f,
    };
    const auto extend = [&](const mr_float4 position) {
        if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z))
            throw std::runtime_error("visual preview bounds contain nonfinite coordinates");
        minimum.x = std::min(minimum.x, position.x);
        minimum.y = std::min(minimum.y, position.y);
        minimum.z = std::min(minimum.z, position.z);
        maximum.x = std::max(maximum.x, position.x);
        maximum.y = std::max(maximum.y, position.y);
        maximum.z = std::max(maximum.z, position.z);
    };
    if (supineView) {
        // Accepted Human exports freeze articulated instance transforms. Bounds
        // must use those transforms, rather than mixing owner-local vertices
        // with the world-space skin and contact bed.
        for (const auto& instance : pack.instances) {
            if (instance.binding.z != MR_VISUAL_BINDING_WORLD)
                throw std::runtime_error("--supine-view unresolved binding kind="+std::to_string(instance.binding.z)+" owner="+std::to_string(instance.binding.y)+" semantic="+std::to_string(instance.identity.x)+" stable="+std::to_string(instance.identity.w));
            for (std::uint32_t j = 0; j < instance.geometry.y; ++j) {
                const auto& primitive = pack.primitives.at(instance.geometry.x + j);
                for (std::uint32_t k = 0; k < primitive.geometry.y; ++k) {
                    const auto p = pack.vertices.at(pack.indices.at(primitive.geometry.x + k)).position;
                    const auto q = instance.orientation;
                    const mr_float4 t{2.0f*(q.y*p.z-q.z*p.y),
                        2.0f*(q.z*p.x-q.x*p.z), 2.0f*(q.x*p.y-q.y*p.x), 0.0f};
                    const auto placement = instance.translationAndScale;
                    extend({placement.x + placement.w*(p.x+q.w*t.x+q.y*t.z-q.z*t.y),
                        placement.y + placement.w*(p.y+q.w*t.y+q.z*t.x-q.x*t.z),
                        placement.z + placement.w*(p.z+q.w*t.z+q.x*t.y-q.y*t.x), 0.0f});
                }
            }
        }
    } else {
        for (const auto& vertex : pack.vertices) extend(vertex.position);
    }
    const mr_float4 centre{
        0.5f * (minimum.x + maximum.x),
        0.5f * (minimum.y + maximum.y),
        0.5f * (minimum.z + maximum.z),
        0.0f,
    };
    const float extent = std::max({
        maximum.x - minimum.x,
        maximum.y - minimum.y,
        maximum.z - minimum.z,
    });
    const float verticalExtent = maximum.z - minimum.z;
    const float distance = focusLowerThird
        ? std::max(0.65f * verticalExtent, 0.30f)
        : (fillFrame
            ? std::max(0.95f * verticalExtent, 0.30f)
            : std::max(1.35f * extent, tightFrame ? 0.0f : 2.25f));
    return {centre, minimum, maximum, distance};
}

metalrobo::SensorSpec makeCamera(
    const std::string& id,
    const mr_float4 position,
    const mr_float4 target,
    const std::uint32_t dimension
) {
    metalrobo::SensorSpec camera;
    camera.id = id;
    camera.parentAssetId = "workspace";
    camera.parentKind = MR_WORLD_SENSOR_PARENT_ASSET;
    camera.kind = MR_WORLD_SENSOR_RGBD;
    camera.localPose = cameraToward(position, target);
    camera.width = dimension;
    camera.height = dimension;
    const float halfDimension = 0.5f * static_cast<float>(dimension);
    const float focalLength = 0.703125f * static_cast<float>(dimension);
    camera.intrinsics = {focalLength, focalLength, halfDimension, halfDimension};
    camera.maximumDepthMeters = 12.0f;
    return camera;
}

metalrobo::VisualLightRigV1 makeAnatomyStudioLightRig(
    const mr_float4 centre,
    const mr_float4 cameraPosition,
    const float cameraDistance
) {
    // This source layer is reviewed from three fixed cameras.  A world-fixed
    // key made the front skin overexposed and the posterior silhouette flat.
    // Reframe only the light rig: the exact source vertices remain untouched.
    const float distance = std::max(cameraDistance, 0.35f);
    const float intensityScale = distance * distance;
    const mr_float4 view = normalizedDirection(
        {cameraPosition.x - centre.x, cameraPosition.y - centre.y,
         cameraPosition.z - centre.z, 0.0f},
        "BodyParts3D camera direction"
    );
    const mr_float4 target{centre.x, centre.y, centre.z + 0.04f * distance, 0.0f};
    const auto makeArea = [&] (
        const mr_float4 position,
        const mr_float4 color,
        const float intensity,
        const float width,
        const float height,
        const std::uint32_t id
    ) {
        const mr_float4 direction = normalizedDirection(
            {target.x - position.x, target.y - position.y, target.z - position.z, 0.0f},
            "BodyParts3D softbox direction"
        );
        MRVisualLightGPUV1 light{};
        light.positionAndRange = {position.x, position.y, position.z, 20.0f};
        light.directionAndSpot = {direction.x, direction.y, direction.z, -1.0f};
        light.colorAndIntensity = {color.x, color.y, color.z, intensity * intensityScale};
        light.shape = {width * distance, height * distance, -1.0f, 0.08f};
        light.shadow = {1u, 0u, 0u, 0u};
        light.identity = {MR_VISUAL_LIGHT_RECTANGLE, MR_VISUAL_LIGHT_UNIT_NIT, 0u, id};
        return light;
    };
    const mr_float4 key{cameraPosition.x + 0.24f * distance,
                        cameraPosition.y - 0.13f * distance,
                        cameraPosition.z + 0.31f * distance, 0.0f};
    const mr_float4 fill{cameraPosition.x - 0.32f * distance,
                         cameraPosition.y + 0.18f * distance,
                         cameraPosition.z + 0.10f * distance, 0.0f};
    const mr_float4 rim{centre.x - 0.82f * distance * view.x,
                        centre.y - 0.82f * distance * view.y,
                        centre.z - 0.82f * distance * view.z + 0.42f * distance, 0.0f};
    metalrobo::VisualLightRigV1 rig;
    rig.id = "anatomy_studio_camera_relative_three_point";
    rig.contentHash = "builtin:anatomy-studio-camera-relative-three-point-v2";
    rig.lights = {
        makeArea(key, {1.0f, 0.98f, 0.95f, 0.0f}, 145.0f, 0.88f, 0.70f, 1u),
        makeArea(fill, {0.80f, 0.87f, 1.0f, 0.0f}, 48.0f, 0.96f, 0.78f, 2u),
        makeArea(rim, {1.0f, 0.94f, 0.86f, 0.0f}, 82.0f, 0.72f, 0.58f, 3u),
    };
    return rig;
}

metalrobo::VisualMotionSampleBatchV1 staticMotion(
    const std::uint32_t bodyCount,
    const std::uint64_t frameIndex
) {
    metalrobo::VisualMotionSampleBatchV1 motion;
    motion.environmentCount = 1u;
    motion.bodyCount = bodyCount;
    motion.sampleCount = 2u;
    motion.exposureOpenSeconds = 0.0;
    motion.exposureCloseSeconds = 1.0 / 120.0;
    motion.timestampsSeconds = {motion.exposureOpenSeconds, motion.exposureCloseSeconds};
    motion.bodyStates.resize(2u * bodyCount);
    for (MRBodyStateGPU& body : motion.bodyStates) {
        body.orientation.w = 1.0f;
    }
    motion.scenarioIdentity = 8601u;
    motion.sensorIdentity = 8601u;
    motion.frameIndex = frameIndex;
    motion.sensorSequence = static_cast<std::uint32_t>(frameIndex);
    return motion;
}

std::uint8_t toSrgb(const float value) {
    const float linear = std::clamp(value, 0.0f, 1.0f);
    const float encoded = linear <= 0.0031308f
        ? 12.92f * linear
        : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
    return static_cast<std::uint8_t>(std::lround(255.0f * std::clamp(encoded, 0.0f, 1.0f)));
}

void writePpm(
    const std::filesystem::path& path,
    const metalrobo::HybridObservationBatch& observation
) {
    const std::size_t pixels = static_cast<std::size_t>(observation.width) * observation.height;
    if (observation.environmentCount != 1u || observation.rgb.size() != pixels) {
        throw std::runtime_error("BodyParts3D preview RGB readback has an unexpected layout");
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("could not create visual preview frame: " + path.string());
    }
    stream << "P6\n" << observation.width << " " << observation.height << "\n255\n";
    for (const mr_float4 pixel : observation.rgb) {
        const std::array<std::uint8_t, 3u> encoded{
            toSrgb(pixel.x),
            toSrgb(pixel.y),
            toSrgb(pixel.z),
        };
        stream.write(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    }
    if (!stream) {
        throw std::runtime_error("could not write visual preview frame: " + path.string());
    }
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc < 3) {
                std::cerr << "usage: metalrobo_bodyparts3d_visual_probe PACK.mrvpack OUTPUT_DIRECTORY [--dimension <128..4096>] [--tight-frame] [--fill-frame] [--focus-lower-third] [--source-semantic-coverage] [--supine-view] [--accepted-receipt PATH] [--human-resting-skin]\n";
                return 2;
            }
            std::uint32_t dimension = 512u;
            bool tightFrame = false;
            bool focusLowerThird = false;
            bool fillFrame = false;
            bool sourceSemanticCoverage = false;
            bool supineView = false;
            bool humanRestingSkin = false;
            std::filesystem::path acceptedReceipt;
            for (int argument = 3; argument < argc; ++argument) {
                const std::string option{argv[argument]};
                if (option == "--human-resting-skin") {
                    if (humanRestingSkin)
                        throw std::runtime_error("--human-resting-skin may only be provided once");
                    humanRestingSkin = true;
                    continue;
                }
                if (option == "--accepted-receipt") {
                    if (!acceptedReceipt.empty() || ++argument >= argc)
                        throw std::runtime_error("--accepted-receipt requires exactly one receipt path");
                    acceptedReceipt = argv[argument];
                    continue;
                }
                if (option == "--supine-view") {
                    if (supineView) throw std::runtime_error("--supine-view may only be provided once");
                    supineView = true;
                    continue;
                }
                if (option == "--source-semantic-coverage") {
                    if (sourceSemanticCoverage)
                        throw std::runtime_error("--source-semantic-coverage may only be provided once");
                    sourceSemanticCoverage = true;
                    continue;
                }
                if (option == "--tight-frame") {
                    if (tightFrame) {
                        throw std::runtime_error("--tight-frame may only be provided once");
                    }
                    tightFrame = true;
                    continue;
                }
                if (option == "--focus-lower-third") {
                    if (focusLowerThird || fillFrame) {
                        throw std::runtime_error("--focus-lower-third may only be provided once and cannot be combined with --fill-frame");
                    }
                    focusLowerThird = true;
                    continue;
                }
                if (option == "--fill-frame") {
                    if (fillFrame || focusLowerThird) {
                        throw std::runtime_error("--fill-frame may only be provided once and cannot be combined with --focus-lower-third");
                    }
                    fillFrame = true;
                    continue;
                }
                if (option != "--dimension" || ++argument >= argc) {
                    throw std::runtime_error("expected --dimension <128..4096>, --tight-frame, --fill-frame, --focus-lower-third, --source-semantic-coverage, --supine-view, --accepted-receipt PATH, or --human-resting-skin");
                }
                const std::string dimensionText{argv[argument]};
                std::size_t parsed = 0u;
                const unsigned long candidate = std::stoul(dimensionText, &parsed);
                if (parsed != dimensionText.size() || candidate < 128u || candidate > 4096u) {
                    throw std::runtime_error("visual preview dimension must be an integer in [128, 4096]");
                }
                dimension = static_cast<std::uint32_t>(candidate);
            }
            std::filesystem::path packPath{argv[1]};
            const std::filesystem::path outputDirectory{argv[2]};
            metalrobo::VisualAssetPackV2 pack;
            std::string reason;
            if (!metalrobo::readVisualAssetPack(packPath, pack, &reason)) {
                throw std::runtime_error("could not read visual pack: " + reason);
            }
            if (humanRestingSkin && acceptedReceipt.empty())
                throw std::runtime_error("--human-resting-skin requires --accepted-receipt");
            if (!acceptedReceipt.empty()) {
                NSError* error = nil;
                NSData* data = [NSData dataWithContentsOfFile:
                    [NSString stringWithUTF8String:acceptedReceipt.c_str()]];
                id decoded = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:&error] : nil;
                if (![decoded isKindOfClass:[NSDictionary class]])
                    throw std::runtime_error("accepted receipt is not a JSON object");
                NSDictionary* receipt = decoded;
                if (metalrobo::computeVisualAssetPackContentHash(pack) != pack.contentHash ||
                    ![receipt[@"physical_endpoint"] isEqual:@"accepted"] ||
                    ![receipt[@"surface_audit_endpoint"] isEqual:@"passed"] ||
                    ![receipt[@"pack_content_hash"] isEqual:
                        [NSString stringWithUTF8String:pack.contentHash.c_str()]])
                    throw std::runtime_error("accepted receipt does not certify this visual pack");
                NSArray* poses = receipt[@"accepted_body_poses"];
                if (![poses isKindOfClass:[NSArray class]] ||
                    poses.count != [receipt[@"accepted_body_pose_count"] unsignedIntegerValue])
                    throw std::runtime_error("accepted receipt pose count differs");
                std::size_t frozen = 0;
                for (auto& instance : pack.instances) {
                    if (instance.binding.z == MR_VISUAL_BINDING_WORLD) continue;
                    if (instance.binding.z != MR_VISUAL_BINDING_ARTICULATED_LINK ||
                        instance.binding.y >= poses.count)
                        throw std::runtime_error("accepted receipt cannot resolve this instance binding");
                    id bodyValue = poses[instance.binding.y];
                    if (![bodyValue isKindOfClass:[NSDictionary class]])
                        throw std::runtime_error("accepted receipt body pose is not an object");
                    NSDictionary* body = bodyValue;
                    NSArray* position = body[@"position_m"];
                    NSArray* orientation = body[@"quaternion_xyzw"];
                    if (![position isKindOfClass:[NSArray class]] ||
                        ![orientation isKindOfClass:[NSArray class]] ||
                        [body[@"body_index"] unsignedIntValue] != instance.binding.y ||
                        position.count != 3 || orientation.count != 4)
                        throw std::runtime_error("accepted receipt body pose is malformed");
                    const mr_float4 b{[position[0] floatValue], [position[1] floatValue],
                        [position[2] floatValue], 0.0f};
                    const mr_float4 q{[orientation[0] floatValue], [orientation[1] floatValue],
                        [orientation[2] floatValue], [orientation[3] floatValue]};
                    const auto p = instance.translationAndScale;
                    const mr_float4 t{2*(q.y*p.z-q.z*p.y), 2*(q.z*p.x-q.x*p.z),
                        2*(q.x*p.y-q.y*p.x), 0};
                    instance.translationAndScale = {b.x+p.x+q.w*t.x+q.y*t.z-q.z*t.y,
                        b.y+p.y+q.w*t.y+q.z*t.x-q.x*t.z,
                        b.z+p.z+q.w*t.z+q.x*t.y-q.y*t.x, p.w};
                    const auto a = instance.orientation;
                    instance.orientation = {q.w*a.x+q.x*a.w+q.y*a.z-q.z*a.y,
                        q.w*a.y-q.x*a.z+q.y*a.w+q.z*a.x,
                        q.w*a.z+q.x*a.y-q.y*a.x+q.z*a.w,
                        q.w*a.w-q.x*a.x-q.y*a.y-q.z*a.z};
                    instance.binding.z = MR_VISUAL_BINDING_WORLD;
                    ++frozen;
                }
                if (humanRestingSkin) {
                    // Resting Skin inspection retains outer O, bed, source iris,
                    // and the owner's sclera/lens presentation clones. Inner I/D
                    // boundaries and other anatomy stay in the pack, unmodified.
                    const std::set<unsigned> noRetiredIris;
                    for (auto& instance : pack.instances) {
                        const auto semantic = instance.identity.x;
                        const auto stable = instance.identity.w;
                        unsigned mask = (semantic == 51007u && stable != 0xFFFFFD11u &&
                            stable != 0xFFFFFD12u) || semantic == 51999u ? 1u : 0u;
                        mask = numiHumanRestingInspectionLayerMask(
                            semantic, stable, mask, 51010u, noRetiredIris);
                        if (stable == kNumiHumanRightScleraWindowStableID ||
                            stable == kNumiHumanLeftScleraWindowStableID ||
                            stable == kNumiHumanRightPosteriorLensBackingStableID ||
                            stable == kNumiHumanLeftPosteriorLensBackingStableID) mask |= 1u;
                        if (!(mask & kNumiHumanSkinInspectionLayerBit)) instance.binding.w = 0u;
                    }
                    pack.preprocessingProvenance += "/resting_skin_inspection_visibility_only";
                }
                std::filesystem::create_directories(outputDirectory);
                pack.preprocessingProvenance += "/accepted_receipt_instance_pose_replay";
                pack.contentHash = metalrobo::computeVisualAssetPackContentHash(pack);
                packPath = outputDirectory / (humanRestingSkin ? "accepted-skin-frozen.mrvpack" : "accepted-frozen.mrvpack");
                if (!metalrobo::writeVisualAssetPack(pack, packPath, &reason))
                    throw std::runtime_error("could not retain accepted replay pack: " + reason);
                std::cout << "accepted_receipt=" << acceptedReceipt << " frozen_instances=" << frozen
                          << " accepted_time_s=" << [receipt[@"accepted_time_s"] doubleValue] << '\n';
            }
            std::set<std::uint32_t> sourceSemantics{8601u};
            if (sourceSemanticCoverage) {
                for (const auto& primitive : pack.primitives)
                    if (primitive.identity.x != 0u && primitive.identity.x != MR_INVALID_INDEX)
                        sourceSemantics.insert(primitive.identity.x);
                for (const auto& instance : pack.instances)
                    if (instance.identity.x != 0u && instance.identity.x != MR_INVALID_INDEX)
                        sourceSemantics.insert(instance.identity.x);
            }
            const PackFraming framing = packFraming(pack, tightFrame, focusLowerThird, fillFrame, supineView);
            mr_float4 target = framing.centre;
            if (focusLowerThird) {
                target.z = framing.minimum.z + 0.32f * (framing.maximum.z - framing.minimum.z);
            }
            if (supineView && (fillFrame || focusLowerThird))
                throw std::runtime_error("--supine-view cannot use vertical crop options");
            const auto distance = framing.distance;
            const std::array cameraDefinitions{
                std::pair{supineView ? "supine_anterior" : "axis_negative_y", supineView
                    ? mr_float4{target.x, target.y - 0.12f*distance, target.z + distance, 0.0f}
                    : mr_float4{target.x, target.y - distance, target.z + 0.08f, 0.0f}},
                std::pair{supineView ? "supine_oblique" : "oblique_positive_x_negative_y", supineView
                    ? mr_float4{target.x + 0.55f*distance, target.y - 0.5f*distance, target.z + distance, 0.0f}
                    : mr_float4{target.x + 0.82f*distance, target.y - 0.82f*distance, target.z + 0.24f*distance, 0.0f}},
                std::pair{supineView ? "supine_sagittal" : "axis_positive_y", supineView
                    ? mr_float4{target.x + distance, target.y, target.z + 0.2f*distance, 0.0f}
                    : mr_float4{target.x, target.y + distance, target.z + 0.08f, 0.0f}},
            };

            const metalrobo::EngineModel model = metalrobo::makeFrankaPickPlaceEngineModel();
            metalrobo::EpisodeTwin episode = metalrobo::makeFrankaPickPlaceEpisodeTwin();
            episode.id = "bodyparts3d_source_static_preview_v1";
            episode.sensors.clear();
            for (const auto& [id, position] : cameraDefinitions) {
                episode.sensors.push_back(makeCamera(id, position, target, dimension));
            }
            metalrobo::WorldTemplate world;
            require(metalrobo::compileEpisodeTwin(episode, model, world), "preview episode compile");
            metalrobo::WorldProgram program;
            program.id = "bodyparts3d_source_static_preview_world_v1";
            metalrobo::WorldFamily family;
            require(metalrobo::compileWorldFamily(world, program, family), "preview world compile");
            metalrobo::MetalWorldFamilyContext worlds;
            require(worlds.compile(family, 1u), "preview world device compile");
            require(worlds.sample(1u, 8601u), "preview world sample");

            const std::array assets{
                metalrobo::VisualAssetReferenceV3{packPath, pack.contentHash, 0u, 8601u, 1u},
            };
            metalrobo::MetalHybridRendererConfig configuration;
            configuration.width = dimension;
            configuration.height = dimension;
            // A static source-inspection frame owns its reference workspace
            // for the entire encode/readback pair.  A fresh renderer is made
            // below for each fixed camera so high-resolution frames never
            // inherit another camera's retained observation.
            configuration.maximumReferenceFramesInFlight = 1u;
            configuration.clearColorAndDepth = {0.012f, 0.018f, 0.028f, 1.0e30f};
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (device == nil || queue == nil) {
                throw std::runtime_error("could not create an Apple Metal command queue");
            }
            std::filesystem::create_directories(outputDirectory);
            for (std::uint32_t camera = 0u; camera < cameraDefinitions.size(); ++camera) {
                metalrobo::MetalHybridRenderer renderer(configuration);
                metalrobo::VisualSceneManifestV3 manifest;
                if (!metalrobo::compileVisualSceneManifestV3(
                        world,
                        assets,
                        metalrobo::makeNeutralStudioEnvironmentV2(),
                        makeAnatomyStudioLightRig(
                            framing.centre, cameraDefinitions[camera].second, framing.distance
                        ),
                        manifest,
                        &reason
                    )) {
                    throw std::runtime_error("preview scene compile: " + reason);
                }
                require(
                    renderer.compile(
                        std::move(manifest.renderScene),
                        metalrobo::VisualRendererProfileV1::sensorReference(),
                        1u
                    ),
                    "preview renderer compile"
                );
                id<MTLCommandBuffer> command = [queue commandBuffer];
                metalrobo::MetalHybridFrameCommandContext context;
                context.commandBuffer = (__bridge void*)command;
                require(
                    renderer.encodeFrame(
                        worlds,
                        staticMotion(static_cast<std::uint32_t>(model.bodies.size()), camera + 1u),
                        camera,
                        context
                    ),
                    "preview frame encode"
                );
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted) {
                    throw std::runtime_error(
                        "preview command failed: " +
                        std::string{command.error.localizedDescription.UTF8String}
                    );
                }
                metalrobo::HybridObservationBatch observation;
                require(renderer.readback(observation), "preview frame readback");
                const std::filesystem::path frame = outputDirectory /
                    (std::string{cameraDefinitions[camera].first} + ".ppm");
                writePpm(frame, observation);
                const std::size_t covered = static_cast<std::size_t>(std::count_if(
                    observation.segmentation.begin(), observation.segmentation.end(),
                    [&](std::uint32_t semantic) {
                        return sourceSemantics.contains(semantic);
                    }
                ));
                if (covered == 0u) {
                    throw std::runtime_error("preview frame contains no BodyParts3D source pixels");
                }
                std::cout << "source_semantic_coverage=" << sourceSemanticCoverage
                          << " supine_view=" << supineView
                          << " human_resting_skin=" << humanRestingSkin
                          << " eligible_source_ids=" << sourceSemantics.size() << ' ';
                std::cout << "view=" << cameraDefinitions[camera].first
                          << " covered_pixels=" << covered
                          << " total_pixels=" << observation.segmentation.size()
                          << " frame=" << frame.string() << '\n';
            }
            std::cout << "device=\"" << device.name.UTF8String << "\""
                      << " source_pack=" << pack.contentHash
                      << " vertices=" << pack.vertices.size()
                      << " triangles=" << pack.indices.size() / 3u
                      << " static_source_preview=ok\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "metalrobo_bodyparts3d_visual_probe: " << error.what() << '\n';
            return 1;
        }
    }
}
