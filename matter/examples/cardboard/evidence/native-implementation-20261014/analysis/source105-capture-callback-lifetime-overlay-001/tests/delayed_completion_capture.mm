#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include "../source/matter/src/coupled_capture_manifest_context.hpp"

#include <atomic>
#include <cstdio>
#include <string>
#include <vector>

using numi::matter::detail::CoupledCaptureManifestContext;
using Completion = void (^)(void);

static Completion makeCompletion(
    bool rhsPreserved,
    bool cachedOperatorUsed,
    std::uint64_t maxPayloadBytes,
    std::vector<std::string> omittedOptionalBuffers,
    std::atomic<int>* result) {
    const CoupledCaptureManifestContext capturedManifestContext{
        rhsPreserved,
        cachedOperatorUsed,
        maxPayloadBytes,
        std::move(omittedOptionalBuffers)
    };
    return ^{
        const bool matches =
            capturedManifestContext.rhsPreservedBeforeFGMRES &&
            capturedManifestContext.cachedOperatorUsed &&
            capturedManifestContext.maxPayloadBytes == 33554432u &&
            capturedManifestContext.omittedOptionalBuffers.size() == 2u &&
            capturedManifestContext.omittedOptionalBuffers[0] == "optional-a" &&
            capturedManifestContext.omittedOptionalBuffers[1] == "optional-b";
        result->store(matches ? 1 : -1, std::memory_order_release);
    };
}

int main() {
    std::atomic<int> result{0};
    bool rhs = true;
    bool cache = true;
    std::uint64_t cap = 33554432u;
    std::vector<std::string> omitted{"optional-a", "optional-b"};
    Completion completion = makeCompletion(rhs, cache, cap, omitted, &result);

    // Mutate every source after completion construction and before invocation.
    rhs = false;
    cache = false;
    cap = 1u;
    omitted.assign(1u, "mutated");

    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
        usleep(20000);
        completion();
        dispatch_semaphore_signal(done);
    });
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    const int observed = result.load(std::memory_order_acquire);
    if (observed != 1) {
        std::fprintf(stderr, "delayed by-value completion failed: %d\n", observed);
        return 1;
    }
    std::puts("delayed by-value completion passed");
    return 0;
}
