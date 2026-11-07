metalrobo::VisualAssetPackV2 makeAuthoredObjectPack(
    const std::uint32_t bodyIndex
) {
    metalrobo::VisualAssetPackV2 pack;
    pack.id = "pick_object_authored";
    pack.sourceUri = "probe://pick_object_authored";
    pack.sourceContentHash = "sha256:pick-object-probe";
    pack.license = "CC0-1.0";
    pack.preprocessingProvenance =
        "metalrobo_visual_platform_probe/v2";
    constexpr std::array positions{
        std::array{-0.07f, -0.07f, -0.07f},
        std::array{ 0.07f, -0.07f, -0.07f},
        std::array{ 0.00f,  0.08f, -0.07f},
        std::array{ 0.00f,  0.00f,  0.09f},
    };
    for (const auto& position : positions) {
        const float length = std::sqrt(
            position[0] * position[0] +
            position[1] * position[1] +
            position[2] * position[2]
        );
        const std::array<float, 3u> normal{
            position[0] / length,
            position[1] / length,
            position[2] / length,
        };
        const std::array<float, 3u> reference =
            std::abs(normal[2]) < 0.999f
            ? std::array<float, 3u>{0.0f, 0.0f, 1.0f}
            : std::array<float, 3u>{0.0f, 1.0f, 0.0f};
        std::array<float, 3u> tangent{
            reference[1] * normal[2] -
                reference[2] * normal[1],
            reference[2] * normal[0] -
                reference[0] * normal[2],
            reference[0] * normal[1] -
                reference[1] * normal[0],
        };
        const float tangentLength = std::sqrt(
            tangent[0] * tangent[0] +
            tangent[1] * tangent[1] +
            tangent[2] * tangent[2]
        );
        for (float& component : tangent) {
            component /= tangentLength;
        }
        pack.vertices.push_back({
            {position[0], position[1], position[2], 1.0f},
            {
                normal[0],
                normal[1],
                normal[2],
                1.0f,
            },
            {tangent[0], tangent[1], tangent[2], 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f},
            {1.0f, 1.0f, 1.0f, 1.0f},
        });
    }
    pack.indices = {
        0u, 2u, 1u,
        0u, 1u, 3u,
        1u, 2u, 3u,
        2u, 0u, 3u,
    };
    MRVisualMaterialGPUV2 material{};
    material.baseColorAndOpacity = {
        0.82f,
        0.12f,
        0.03f,
        1.0f,
    };
    material.surface = {0.3f, 0.1f, 1.0f, 1.0f};
    material.coatingAndAlphaCutoff =
        {0.2f, 0.15f, 1.0f, 0.5f};
    material.textureIndices0 = {
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
    };
    material.textureIndices1 = material.textureIndices0;
    material.reserved = material.textureIndices0;
    material.flags = {
        MR_VISUAL_ALPHA_OPAQUE,
        MR_VISUAL_MATERIAL_DOUBLE_SIDED,
        0u,
        1u,
    };
    pack.materials = {material};
    MRVisualPrimitiveGPUV2 primitive{};
    primitive.geometry = {
        0u,
        static_cast<std::uint32_t>(pack.indices.size()),
        0u,
        0u,
    };
    primitive.identity = {1u, 1u, bodyIndex, 1u};
    primitive.boundsMinimum = {-0.07f, -0.07f, -0.07f, 1.0f};
    primitive.boundsMaximum = {0.07f, 0.08f, 0.09f, 1.0f};
    pack.primitives = {primitive};
    MRVisualInstanceGPUV2 instance{};
    instance.translationAndScale = {0.0f, 0.0f, 0.0f, 1.0f};
    instance.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    instance.binding = {
        0u,
        bodyIndex,
        MR_VISUAL_BINDING_RIGID_BODY,
        MR_VISUAL_INSTANCE_CASTS_SHADOW |
            MR_VISUAL_INSTANCE_RECEIVES_SHADOW |
            MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR,
    };
    instance.identity = {1u, 1u, bodyIndex, 1u};
    instance.geometry = {0u, 1u, 0u, 0u};
    pack.instances = {instance};
    pack.symbolicBindings = {{
        "pick_object",
        "pick_object",
        0u,
        bodyIndex,
        MR_VISUAL_BINDING_RIGID_BODY,
    }};
    pack.contentHash =
        metalrobo::computeVisualAssetPackContentHash(pack);
    return pack;
}
