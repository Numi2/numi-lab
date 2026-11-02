#pragma once

#import <Metal/Metal.h>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

namespace numi::matter::detail {
// These tables are immutable after admission. A shared Metal buffer owns the
// same bytes for the entire adapter lifetime; retaining a second CPU allocation
// is unnecessary. Validate byte parity before relinquishing staging ownership.
template<class T>
inline bool identicalImmutableRows(std::span<const T> a, std::span<const T> b) {
    static_assert(std::is_trivially_copyable_v<T>);
    return a.size() == b.size() &&
        (a.empty() || a.data() == b.data() || std::memcmp(a.data(), b.data(), a.size_bytes()) == 0);
}
template<class T>
inline bool retainSharedImmutableRows(id<MTLBuffer> buffer,
    std::vector<T>& staging, std::span<const T>& rows,
    std::size_t& releasedBytes) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (!rows.empty() && (buffer == nil ||
        buffer.storageMode != MTLStorageModeShared ||
        buffer.length < rows.size_bytes() || buffer.contents == nullptr ||
        std::memcmp(buffer.contents, rows.data(), rows.size_bytes()) != 0))
        return false;
    const auto count = rows.size();
    const T* uploaded = count == 0 ? nullptr : static_cast<const T*>(buffer.contents);
    rows = std::span<const T>(uploaded, count);
    releasedBytes += staging.capacity() * sizeof(T);
    std::vector<T>().swap(staging);
    return true;
}
} // namespace numi::matter::detail
