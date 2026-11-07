#include "../src/metal/NumanXTransactionPassContract.hpp"
#include <cassert>
#include <cstdint>

int main() {
    using namespace metalrobo;
    constexpr auto prefix = metalNumanXTransactionPassStructSizeForABI(
        kMetalNumanXTransactionABIVersion);
    static_assert(prefix == offsetof(MetalNumanXTransactionPass,
                                     exactCandidateKinematicsContext));
    static_assert(prefix < sizeof(MetalNumanXTransactionPass));
    static_assert(detail::validNumanXABI6TransactionPassHeader(6u, prefix, 0u));
    static_assert(!detail::validNumanXABI6TransactionPassHeader(6u, prefix - 1u, 0u));
    static_assert(!detail::validNumanXABI6TransactionPassHeader(6u, prefix + 1u, 0u));
    static_assert(!detail::validNumanXABI6TransactionPassHeader(
        6u, sizeof(MetalNumanXTransactionPass), 0u));
    static_assert(!detail::validNumanXABI6TransactionPassHeader(5u, prefix, 0u));
    static_assert(!detail::validNumanXABI6TransactionPassHeader(7u, prefix, 0u));
    static_assert(!detail::validNumanXABI6TransactionPassHeader(6u, prefix, 1u));
    return 0;
}
