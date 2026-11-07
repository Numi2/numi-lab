#pragma once
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/finite_resource_types.h"
#include <filesystem>
#include <memory>

namespace metalrobo {
struct FiniteResourceConfig {
    std::uint64_t worldFingerprint=0u;
    std::uint32_t environmentCount=0u, bodyCount=0u, nq=0u;
    MRFiniteResourceProgramGPU program{};
    std::vector<MRFiniteResourcePatchGPU> patches;
};
namespace detail { struct FiniteResourceRuntime; }

// Nutritional tracer ledger. Uses MetalWorld's accepted postCommit stream;
// ordinary task/scene resets do not own or reinitialize this state. Keep the
// context alive until its borrowed world submissions have completed.
class FiniteResourceContext {
public:
    explicit FiniteResourceContext(FiniteResourceConfig config);
    ~FiniteResourceContext();
    FiniteResourceContext(const FiniteResourceContext&)=delete;
    FiniteResourceContext& operator=(const FiniteResourceContext&)=delete;
    [[nodiscard]] MetalWorldDevicePhysicsProgram deviceProgram();
    [[nodiscard]] std::vector<MRFiniteResourceStateGPU> snapshot() const;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    // A resource-only checkpoint. It cannot restore the surrounding world,
    // contact caches, policies, sensory history or agent memory.
    void saveCheckpoint(const std::filesystem::path& path) const;
    void restoreCheckpoint(const std::filesystem::path& path);
private:
    std::shared_ptr<detail::FiniteResourceRuntime> runtime_;
};
} // namespace metalrobo
