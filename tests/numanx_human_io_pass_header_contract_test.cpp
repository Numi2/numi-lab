#include "../src/metal/NumanXHumanIOPassContract.hpp"
#include <cassert>
#include <cstdint>

int main() {
    using namespace metalrobo;
    constexpr auto prefix = metalNumanXTransactionPassStructSizeForABI(
        kMetalNumanXTransactionABIVersion);
    static_assert(prefix == offsetof(MetalNumanXTransactionPass,
                                     exactCandidateKinematicsContext));
    static_assert(prefix < sizeof(MetalNumanXTransactionPass));
    static_assert(detail::validHumanIOTransactionPassHeader(6u, prefix, 0u));
    static_assert(!detail::validHumanIOTransactionPassHeader(6u, prefix - 1u, 0u));
    static_assert(!detail::validHumanIOTransactionPassHeader(6u, prefix + 1u, 0u));
    static_assert(!detail::validHumanIOTransactionPassHeader(
        6u, sizeof(MetalNumanXTransactionPass), 0u));
    static_assert(!detail::validHumanIOTransactionPassHeader(5u, prefix, 0u));
    static_assert(!detail::validHumanIOTransactionPassHeader(7u, prefix, 0u));
    static_assert(!detail::validHumanIOTransactionPassHeader(6u, prefix, 1u));
    return 0;
}
