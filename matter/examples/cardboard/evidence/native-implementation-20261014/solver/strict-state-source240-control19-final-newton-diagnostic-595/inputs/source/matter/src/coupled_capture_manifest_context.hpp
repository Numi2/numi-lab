#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace numi::matter::detail {

// Values needed by the asynchronous coupled-capture completion handler.
// Keeping them in an owned value object prevents the handler from reaching
// through the enclosing encode lambda's reference captures after it returns.
struct CoupledCaptureManifestContext {
    bool rhsPreservedBeforeFGMRES = false;
    bool cachedOperatorUsed = false;
    std::uint64_t maxPayloadBytes = 0u;
    std::vector<std::string> omittedOptionalBuffers;
};

} // namespace numi::matter::detail
