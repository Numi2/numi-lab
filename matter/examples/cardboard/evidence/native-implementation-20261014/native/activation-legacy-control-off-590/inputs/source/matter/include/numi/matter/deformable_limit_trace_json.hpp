#pragma once

#include <sstream>
#include <utility>

namespace numi::matter::detail {

// Shared by the production first-failure trace emitter and its JSON roundtrip test.
// Closing the array here ensures following optional fields remain siblings of
// "barrier", regardless of whether per_alpha_trace is attached.
template <class AppendFloat>
inline void appendDeformableLimitTraceBarrier(
    std::ostringstream& output, float x, float y, float z, float w,
    AppendFloat&& appendFloat) {
    output << ",\"barrier\":[";
    appendFloat(output, x); output << ',';
    appendFloat(output, y); output << ',';
    appendFloat(output, z); output << ',';
    appendFloat(output, w);
    output << ']';
}

} // namespace numi::matter::detail
