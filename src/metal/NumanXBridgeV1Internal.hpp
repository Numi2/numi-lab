#pragma once

#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/MetalNumanXHumanIO.hpp"
#include "metalrobo/mrnx_bridge_v1.h"

#include <memory>

namespace metalrobo::numanx_bridge_v1 {

struct Domain;
using DomainPtr = std::shared_ptr<Domain>;
using PreparedPhysicalCompletion = void (*)(
    void* context,
    bool ready,
    std::uint64_t slotGeneration
) noexcept;
enum class PreparedTerminalDisposition : std::uint32_t {
    published = 1u,
    rejected = 2u,
    terminalNoTouch = 3u,
};
using PreparedTerminalCompletion = void (*)(
    void* context,
    PreparedTerminalDisposition disposition,
    const mrnx_root_v1& root,
    const mrnx_candidate_view_v1* candidate,
    const mrnx_candidate_channel_v1* channels,
    std::uint32_t channelCount
) noexcept;

// Internal-only constructors used by the future provenance-valid full-body
// runtime owner and by the native bridge qualification probe. There is no C
// cast/adoption entrypoint: Swift can receive only handles created after these
// validators consume the exact move-only MetalRobo capabilities.
// Both adoptions are failure-atomic: they move only after all fallible
// validation and allocation succeeds. A nullptr return leaves caller
// ownership intact, and that caller must explicitly reject or quarantine the
// unresolved capability rather than letting it fall out of scope.
[[nodiscard]] DomainPtr makeDomain(void* metalDevice) noexcept;

[[nodiscard]] mrnx_candidate_v1* adoptCandidate(
    const DomainPtr& domain,
    MetalNumanXHumanIOCandidatePublicationLease&& lease
) noexcept;

// Extends a just-adopted HumanIO candidate with same-command-buffer sensor
// channels whose ranges are retained by the opaque candidate and later bound
// by Brain's pending-sensor publication fingerprint. The operation is
// one-shot, pre-bind, and validates exact same-device non-overlap.
[[nodiscard]] bool attachCandidateChannels(
    mrnx_candidate_v1* candidate,
    const mrnx_candidate_channel_v1* channels,
    std::uint32_t channelCount
) noexcept;

[[nodiscard]] mrnx_prepared_v1* adoptPrepared(
    const DomainPtr& domain,
    MetalNumanXHumanMatterPrepared&& prepared,
    std::shared_ptr<void> runtimeOwner = {},
    void* terminalContext = nullptr,
    PreparedTerminalCompletion terminalCompletion = nullptr
) noexcept;

[[nodiscard]] bool registerPreparedPhysicalCompletion(
    mrnx_prepared_v1* prepared,
    void* completionContext,
    PreparedPhysicalCompletion completion
) noexcept;

// Marks the C capability terminal after the native owner has already reported
// physical terminal-no-touch. It deliberately retains the lifecycle hold and
// therefore cannot make the slot reusable.
void markPreparedPhysicalTerminal(mrnx_prepared_v1* prepared) noexcept;

[[nodiscard]] bool installPreparedCultureView(
    mrnx_prepared_v1* prepared,
    const mrnx_culture_prepared_view_v1& view
) noexcept;

} // namespace metalrobo::numanx_bridge_v1
