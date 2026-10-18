#include "metalrobo/MetalArticulatedOperator.hpp"

#include <cstddef>
#include <cstdint>

using namespace metalrobo;

static bool encode(void*, const MetalNumanXTransactionPass&) noexcept {
    return true;
}
static void abort(void*, void*) noexcept {}

int main() {
    static_assert(sizeof(MetalNumanXTransactionProgram) == 48u);
    static_assert(offsetof(MetalNumanXTransactionProgram,
                          deferAcceptedPublication) == 41u);

    MetalNumanXTransactionProgram legacyDefault{};
    legacyDefault.abiVersion = kMetalNumanXTransactionLegacyABIVersion;
    legacyDefault.structSize = sizeof(legacyDefault);
    auto* defaultRaw = reinterpret_cast<unsigned char*>(&legacyDefault);
    defaultRaw[offsetof(MetalNumanXTransactionProgram,
                        deferAcceptedPublication)] = 0xa5u;
    if (legacyDefault.configured() || legacyDefault.valid() ||
        legacyDefault.usesDeferredAcceptedPublication()) {
        return 1;
    }

    MetalNumanXTransactionProgram legacy{};
    legacy.abiVersion = kMetalNumanXTransactionLegacyABIVersion;
    legacy.structSize = sizeof(legacy);
    legacy.context = &legacy;
    legacy.encode = &encode;
    legacy.abort = &abort;
    legacy.fingerprint = 1u;

    // ABI 4 used this byte as padding. Poison it and verify every legacy
    // program query avoids evaluating the new bool member.
    auto* raw = reinterpret_cast<unsigned char*>(&legacy);
    raw[offsetof(MetalNumanXTransactionProgram, deferAcceptedPublication)] =
        0xa5u;
    if (!legacy.valid() || !legacy.configured() ||
        legacy.usesDeferredAcceptedPublication()) {
        return 2;
    }

    MetalNumanXTransactionProgram current{};
    current.context = &current;
    current.encode = &encode;
    current.abort = &abort;
    current.fingerprint = 2u;
    current.deferAcceptedPublication = true;
    if (current.abiVersion != kMetalNumanXTransactionABIVersion ||
        !current.valid() || !current.configured() ||
        !current.usesDeferredAcceptedPublication()) {
        return 3;
    }

    MetalNumanXTransactionProgram unsupported{};
    unsupported.abiVersion = 3u;
    if (unsupported.valid() || !unsupported.configured()) return 4;
    unsupported.abiVersion = 6u;
    if (unsupported.valid() || !unsupported.configured()) return 5;
    return 0;
}
