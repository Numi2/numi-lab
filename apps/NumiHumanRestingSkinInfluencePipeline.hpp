#pragma once
#include <Metal/Metal.h>
#include <cstdint>

enum class NumiHumanSkinInfluenceLayout : std::uint32_t {
    vertexMajor = 0u,
    tile32 = 1u,
};

struct NumiHumanSkinInfluencePipelineConfig {
    bool validatedStaticSkinInfluences = false;
    bool specializeSkinInfluenceLayout = false;
    NumiHumanSkinInfluenceLayout influenceLayout =
        NumiHumanSkinInfluenceLayout::vertexMajor;
    bool skipNonAncestorSupportBodyDofs = false;
    bool contouredBed = false;
};

inline id<MTLFunction> numiHumanRestingMakeSkinInfluenceFunction(
    id<MTLLibrary> library, NSString* name,
    const NumiHumanSkinInfluencePipelineConfig config,
    NSError** error
) {
    MTLFunctionConstantValues* constants =
        [[MTLFunctionConstantValues alloc] init];
    // Optional Metal arguments require a specialized function in both modes.
    // Explicit false removes the bed buffers from the legacy pipeline.
    const bool contouredBed = config.contouredBed;
    [constants setConstantValue:&contouredBed type:MTLDataTypeBool atIndex:47u];
    if (config.validatedStaticSkinInfluences) {
        bool validated = true;
        [constants setConstantValue:&validated
                              type:MTLDataTypeBool atIndex:42u];
    }
    if (config.specializeSkinInfluenceLayout) {
        bool tiled = config.influenceLayout ==
            NumiHumanSkinInfluenceLayout::tile32;
        [constants setConstantValue:&tiled
                              type:MTLDataTypeBool atIndex:15u];
    }
    if (config.skipNonAncestorSupportBodyDofs) {
        bool skipNonAncestorSupportBodyDofs = true;
        [constants setConstantValue:&skipNonAncestorSupportBodyDofs
                              type:MTLDataTypeBool atIndex:43u];
    }
    return [library newFunctionWithName:name
                         constantValues:constants
                                  error:error];
}

