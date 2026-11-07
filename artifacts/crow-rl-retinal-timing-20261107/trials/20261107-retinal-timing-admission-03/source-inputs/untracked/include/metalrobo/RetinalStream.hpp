#pragma once
#include "metalrobo/retinal_stream_types.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>
namespace metalrobo {
struct RetinalStreamConfig {
    std::uint32_t environments=0,sourceWidth=0,sourceHeight=0,width=0,height=0,eyes=2;
    std::uint32_t sampleNumerator=3,sampleDenominator=10,latencyTicks=2;
    float tickSeconds=.02f;
    std::size_t maximumRetainedBytes=256ull*1024*1024;
    std::filesystem::path metallibPath;
};
struct RetinalStreamSnapshot {
    std::vector<MRRetinalStateGPU> states;
    std::vector<std::uint32_t> errors;
};
namespace detail {struct RetinalStreamRuntime;}
// Native sensor-only area reduction, timing, delay and publication. Inputs
// are borrowed id<MTLBuffer> objects: immutable TaskState and linear float4
// RGB per eye. No identities/depth/world poses are accepted. encode() only
// encodes in the caller's command buffer; it never waits or reads back.
// Source RGB may come from the native renderer or explicit replay fixtures.
// This context does not itself admit a biological sensor, a TaskPack input
// contract, world restore, renderer cadence or learner integration.
class RetinalStreamContext {
public:
    explicit RetinalStreamContext(RetinalStreamConfig config);
    ~RetinalStreamContext();
    RetinalStreamContext(const RetinalStreamContext&)=delete;
    RetinalStreamContext& operator=(const RetinalStreamContext&)=delete;
    [[nodiscard]] bool encode(void* commandBuffer,void* taskStates,std::span<void* const> eyeRGB,
        void* actorOutput,std::uint32_t actorStride,std::uint32_t featureOffset,
        std::size_t outputBaseElements=0);
    [[nodiscard]] std::uint32_t featureCount() const noexcept;
    [[nodiscard]] RetinalStreamSnapshot snapshot() const;
private:
    std::shared_ptr<detail::RetinalStreamRuntime> runtime_;
};
}
