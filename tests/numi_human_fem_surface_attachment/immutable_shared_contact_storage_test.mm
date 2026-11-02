#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "immutable_shared_contact_storage.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace numi::matter::detail;
static void require(bool v, const char* why) { if (!v) throw std::runtime_error(why); }
int main() { @autoreleasepool { try {
    auto device = MTLCreateSystemDefaultDevice(); require(device != nil, "Metal device absent");
    std::vector<std::uint32_t> staging{7, 11, 13, 17};
    std::span<const std::uint32_t> rows(staging);
    const std::array<std::uint32_t,4> same{7,11,13,17}, different{7,11,13,19};
    require(identicalImmutableRows(rows, std::span<const std::uint32_t>(same)), "equal rows differ");
    require(!identicalImmutableRows(rows, std::span<const std::uint32_t>(different)), "different rows alias");
    require(!identicalImmutableRows(rows, rows.first(3)), "different lengths alias");
    auto buffer = [device newBufferWithBytes:rows.data() length:rows.size_bytes() options:MTLResourceStorageModeShared];
    auto bad = [device newBufferWithBytes:different.data() length:sizeof(different) options:MTLResourceStorageModeShared];
    std::size_t freed = 0;
    require(!retainSharedImmutableRows(bad, staging, rows, freed), "changed upload accepted");
    require(freed == 0 && staging.size() == 4 && rows.data() == staging.data(), "failed parity changed ownership");
    auto shortBuffer = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    require(!retainSharedImmutableRows(shortBuffer, staging, rows, freed), "short upload accepted");
    auto privateBuffer = [device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
    require(!retainSharedImmutableRows(privateBuffer, staging, rows, freed), "private memory exposed to CPU");
    const auto capacityBytes = staging.capacity() * sizeof(std::uint32_t);
    require(retainSharedImmutableRows(buffer, staging, rows, freed), "shared upload rejected");
    require(staging.capacity() == 0 && staging.empty() && freed == capacityBytes, "CPU allocation retained");
    require(rows.data() == buffer.contents && identicalImmutableRows(rows, std::span<const std::uint32_t>(same)), "uploaded rows lost");
    std::vector<std::uint32_t> empty; std::span<const std::uint32_t> noRows;
    require(retainSharedImmutableRows(nil, empty, noRows, freed) && noRows.empty(), "empty table rejected");
    require(identicalImmutableRows(noRows, noRows), "empty alias rejected");
    std::cout << "PASS immutable shared contact storage: exact bytes, alias parity, lifetime, empty and rejection controls\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } } }
