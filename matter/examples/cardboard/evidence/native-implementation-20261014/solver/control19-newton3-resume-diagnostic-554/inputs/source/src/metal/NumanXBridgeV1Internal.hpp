#pragma once

#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/MetalNumanXHumanIO.hpp"
#include "metalrobo/mrnx_bridge_v1.h"
#include "metalrobo/numanx_human_matter_adapter_gpu.h"

#include <memory>

namespace metalrobo::numanx_bridge_v1 {

struct Domain;
using DomainPtr = std::shared_ptr<Domain>;
using DomainPublicRead = bool (*)(void* context) noexcept;
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
    std::uint32_t channelCount,
    const MRNumanXHumanMatterJointPublicationFenceGPU* committedFence
) noexcept;

// Internal-only constructors used by the provenance-valid full-body runtime
// owner and by native bridge qualification probes. There is no C
// cast/adoption entrypoint: Swift can receive only handles created after these
// validators consume the exact move-only MetalRobo capabilities.
// Both adoptions are failure-atomic: they move only after all fallible
// validation and allocation succeeds. A nullptr return leaves caller
// ownership intact, and that caller must explicitly reject or quarantine the
// unresolved capability rather than letting it fall out of scope.
[[nodiscard]] DomainPtr makeDomain(void* metalDevice) noexcept;

// Runs one aggregate-reader operation while the sole bridge publication gate
// is held shared. A poisoned domain fails before invoking the callback. Both
// legacy and exact accepted-release paths, plus winning timeout quarantine,
// invoke their terminal completion while holding this gate exclusively;
// terminal completions must therefore update the aggregate directly and must
// not reacquire this reader gate.
[[nodiscard]] bool withDomainPublicReadGate(
    const DomainPtr& domain,
    void* context,
    DomainPublicRead read
) noexcept;

// Atomic epoch published by accepted release. Returns zero for a null or
// poisoned domain. Aggregate readers should call this from their read-gate
// callback and require equality with the copied tuple's epoch.
[[nodiscard]] std::uint64_t domainPublicationEpoch(
    const DomainPtr& domain
) noexcept;

[[nodiscard]] mrnx_candidate_v1* adoptCandidate(
    const DomainPtr& domain,
    MetalNumanXHumanIOCandidatePublicationLease&& lease
) noexcept;

// Exact-clock adoption retains the same opaque handle shape while binding the
// completed HumanIO lease to the canonical inbound authority and GPU-derived
// accepted proof/token pair. The proof is validated and discarded; the opaque
// candidate retains only the compact accepted token. Adoption derives timing,
// the two HumanIO-owned sensor channels, and the exact sensor packet rather
// than accepting caller-authored copies.
[[nodiscard]] mrnx_candidate_v1* adoptCandidateV2(
    const DomainPtr& domain,
    MetalNumanXHumanIOCandidatePublicationLease&& lease,
    const mrnx_exact_inbound_authority_v2& inboundAuthority,
    const MRNumanXAcceptedStateProofGPUV2& acceptedStateProof,
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& acceptedToken
) noexcept;

// Extends a just-adopted HumanIO candidate with same-command-buffer sensor
// channels whose ranges are retained by the opaque candidate and later bound
// by Brain's pending-sensor publication fingerprint. Exact attachment also
// adapts the native publication program: Matter sees the complete canonical
// packet fingerprint, while reserve/publish/reject is translated to HumanIO's
// distinct private two-channel capability. The operation is one-shot,
// pre-bind, and validates exact same-device non-overlap.
[[nodiscard]] bool attachCandidateChannels(
    mrnx_candidate_v1* candidate,
    const mrnx_candidate_channel_v1* channels,
    std::uint32_t channelCount
) noexcept;

// Supplemental descriptors may arrive in any order. The bridge merges them
// with HumanIO's base proprioception/interoception channels, rejects duplicate
// modalities, and stores the complete exact set in increasing modality order.
[[nodiscard]] bool attachCandidateChannelsV2(
    mrnx_candidate_v1* candidate,
    const mrnx_candidate_channel_v2* channels,
    std::uint32_t channelCount
) noexcept;

[[nodiscard]] mrnx_prepared_v1* adoptPrepared(
    const DomainPtr& domain,
    MetalNumanXHumanMatterPrepared&& prepared,
    std::shared_ptr<void> runtimeOwner = {},
    void* terminalContext = nullptr,
    PreparedTerminalCompletion terminalCompletion = nullptr
) noexcept;

[[nodiscard]] mrnx_prepared_v1* adoptPreparedV2(
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
