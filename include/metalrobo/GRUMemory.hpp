#pragma once
#include "metalrobo/gru_memory_types.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>
namespace metalrobo {
struct GRUCellParameters {
    std::uint32_t inputs=0, hidden=0;
    float maximumAbsoluteInput=100.0f;
    // Row-major [r,z,n][hidden][input or hidden], separate reset-after biases.
    std::vector<float> inputWeights, recurrentWeights, inputBias, recurrentBias;
};
// Validates dimensions, finite values and finite FP32 accumulator bounds.
// No output artifact is admitted if any authored tensor is invalid.
[[nodiscard]] std::uint64_t gruCellFingerprint(const GRUCellParameters& parameters);
struct GRUMemoryConfig {
    std::uint32_t environments=0;
    std::size_t maximumRetainedBytes=256ull*1024*1024;
    std::filesystem::path metallibPath;
};
struct GRUMemorySnapshot {
    // {proposal step, episode, initialized, proposal was episode-reset}.
    std::vector<mr_uint4> metadata;
    std::vector<float> before, proposed;
    // Global sticky error followed by one lane error per environment.
    std::vector<std::uint32_t> errors;
};
namespace detail {struct GRUMemoryRuntime;}
// Native recurrent operator, not a second policy or simulator. TaskState is
// borrowed read-only. PolicyPack/MetalWorld/sequence-learner integration is a
// separate required admission, not implied by this component's existence.
class GRUMemoryContext {
public:
    GRUMemoryContext(GRUMemoryConfig config, GRUCellParameters parameters);
    ~GRUMemoryContext();
    GRUMemoryContext(const GRUMemoryContext&)=delete;
    GRUMemoryContext& operator=(const GRUMemoryContext&)=delete;
    // Same-shaped next weights may be installed only at an idle publication
    // boundary. This retains history and performs no neural/physical step.
    void install(GRUCellParameters parameters);
    void reset();
    // Only encodes in the borrowed uncommitted command; no host wait/readback.
    // Distinct buffers are required. Trace records the exact pre-cell hidden
    // state and episode-reset mask for future recurrent policy learning.
    [[nodiscard]] bool encode(void* commandBuffer, void* taskStates, void* input,
        std::uint32_t inputStride, void* output, void* beforeTrace, void* resetTrace,
        bool preview=false, std::size_t inputBase=0, std::size_t outputBase=0,
        std::size_t traceBase=0, std::size_t resetBase=0);
    [[nodiscard]] bool healthy() const;
    [[nodiscard]] std::uint64_t parameterFingerprint() const;
    [[nodiscard]] GRUMemorySnapshot snapshot() const;
    [[nodiscard]] std::vector<std::byte> checkpointState() const;
    void restoreState(std::span<const std::byte> bytes);
private:
    std::shared_ptr<detail::GRUMemoryRuntime> runtime_;
};
}
