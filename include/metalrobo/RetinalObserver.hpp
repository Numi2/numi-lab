#pragma once
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/RetinalStream.hpp"
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/VisualPlatform.hpp"
#include <memory>
#include <array>
namespace metalrobo {
namespace detail {struct RetinalObserverRuntime;}
class FiniteResourceContext;
// Explicit paired causal evaluation of a fixed policy. Only cue mesh visibility
// changes; food allocation, camera, rasterization and physical state stay native.
enum class RetinalCueEvaluationIntervention { none, absent, swapBinary };
// Two instantaneous head-camera renders in the caller's native command buffer,
// then GPU area reduction/cadence/delay before actor inference. Truth planes
// remain private to the renderer and are never accepted by RetinalStream.
// The owner must outlive use of deviceProgram(); command completion retains
// its resources. This is a development instrument, not biological optics.
class RetinalObserverContext {
public:
    RetinalObserverContext(const CompiledRun&,const WorldFamily&,
        const VisualSceneManifestV3&,std::uint32_t environments,std::uint64_t seed);
    RetinalObserverContext(const CompiledRun&,const WorldFamily&,
        const VisualSceneManifestV3&,std::uint32_t environments,std::uint64_t seed,
        const FiniteResourceContext* cueResources);
    RetinalObserverContext(const CompiledRun&,const WorldFamily&,
        const VisualSceneManifestV3&,std::uint32_t environments,std::uint64_t seed,
        const FiniteResourceContext* cueResources,RetinalCueEvaluationIntervention);
    ~RetinalObserverContext();
    RetinalObserverContext(const RetinalObserverContext&)=delete;
    RetinalObserverContext& operator=(const RetinalObserverContext&)=delete;
    [[nodiscard]] MetalWorldDeviceObservationProgram deviceProgram() noexcept;
    void reset(std::uint64_t seed);
    [[nodiscard]] RetinalStreamSnapshot snapshot() const;
    [[nodiscard]] std::vector<std::byte> checkpointState() const;
    void restoreState(std::span<const std::byte> state);
    // Explicit quiescent instrument boundary; never called by encode().
    [[nodiscard]] std::array<HybridObservationBatch,2> inspectImages() const;
private:
    std::shared_ptr<detail::RetinalObserverRuntime> runtime_;
};
}
