#pragma once

#include "metalrobo/MetalNumanXHumanMatter.hpp"
#include "metalrobo/human_behavior_gpu.h"

#include <cstdint>

namespace metalrobo {

// Exact identity expected at one authoritative terminal callback. This keeps
// the nonfinite audit tied to the same candidate measured by the behavior
// program instead of accepting a merely successful or stale sample.
struct HumanBehaviorNativeAttemptIdentity {
    std::uint64_t behaviorProgramFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
    std::uint64_t physicsGeneration = 0u;
    std::uint64_t acceptedTimestampNanoseconds = 0u;
};

struct HumanBehaviorNativeAuditResult {
    std::uint32_t coveredMask = 0u;
    std::uint32_t violationMask = 0u;
};

// Produces only the native nonfinite audit bit. physicalOwnerReady must come
// from the exact physical owner after its receipt and proof-program checks;
// callers cannot use a diagnostic payload alone to establish coverage.
//
// A successful Human/Matter/world tuple proves the physical owners traversed
// their fail-closed finite guards. The independently source-bound behavior
// sample must also be the exact terminal candidate. Missing or unsuccessful
// owners leave coverage unknown. Finite coverage with a nonfinite diagnostic
// or metric scalar is reported as a violation rather than silently discarded.
[[nodiscard]] HumanBehaviorNativeAuditResult
humanBehaviorNativeNonfiniteAudit(
    bool physicalOwnerReady,
    const MetalNumanXHumanMatterPhysicalOutcome* physicalOutcome,
    const MRHumanBehaviorCandidateGPU* behaviorCandidate,
    const HumanBehaviorNativeAttemptIdentity& expected
) noexcept;

} // namespace metalrobo
