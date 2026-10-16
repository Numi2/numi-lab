#include "numi/matter/regional_consumer_capture.hpp"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

int main(int argc, char** argv) {
    using namespace numi::matter::regional_consumer_capture;
    assert(argc == 2);

    const auto parsed = parseSelector("12:3:7:2:5:1");
    assert(parsed);
    assert(parsed.value.control == 12u && parsed.value.physics == 3u &&
        parsed.value.newton == 7u && parsed.value.restart == 2u &&
        parsed.value.column == 5u && parsed.value.environment == 1u);
    assert(matches(parsed.value, 12u, 3u, 7u, 2u, 5u, 1u));
    assert(!matches(parsed.value, 12u, 3u, 7u, 1u, 5u, 1u));
    assert((expectedGeneratorTag(parsed.value) ==
        std::array<std::uint32_t, 4>{12u, 3u, 7u, 1u}));

    assert(singleTickScope(0u, 0u, 1u));
    assert(!singleTickScope(1u, 0u, 2u));
    assert(!singleTickScope(0u, 1u, 1u));
    assert(!singleTickScope(0u, 0u, 2u));
    bool attempted = false;
    assert(claimOnce(attempted, true));
    assert(attempted);
    assert(!claimOnce(attempted, true));
    assert(!claimOnce(attempted, false));

    assert(parseSelector("").error == SelectorError::empty);
    assert(parseSelector("1:2:3:4:5").error == SelectorError::wrong_field_count);
    assert(parseSelector("1:2:3:4:5:6:7").error == SelectorError::wrong_field_count);
    assert(parseSelector("1:2:3:+4:5:6").error == SelectorError::non_decimal);
    assert(parseSelector("1:2:3:4x:5:6").error == SelectorError::non_decimal);
    assert(parseSelector("1:2:3:0:5:6").error == SelectorError::restart_zero);
    assert(parseSelector("4294967296:2:3:4:5:6").error == SelectorError::overflow);
    assert(parseSelector("1::3:4:5:6").error == SelectorError::non_decimal);

    std::uint64_t bytes = 0u;
    assert(checkedArrayBytes({2u, 3u, 16u}, 4u, 384u, bytes) && bytes == 384u);
    assert(!checkedArrayBytes({2u, 3u, 16u}, 4u, 383u, bytes));
    assert(!checkedArrayBytes({0u, 3u}, 4u, 1024u, bytes));
    assert(!checkedArrayBytes(
        {std::numeric_limits<std::uint64_t>::max(), 2u}, 4u,
        std::numeric_limits<std::uint64_t>::max(), bytes));
    std::uint64_t aggregate = 16u;
    assert(checkedAppendBytes(aggregate, 16u, 32u, bytes) && bytes == 32u);
    assert(!checkedAppendBytes(aggregate, 17u, 32u, bytes));

    // Source464's exact selector-5, one-environment shape from the external
    // size audit: both graph matrix copies dominate, while the full append
    // inventory remains below the new bounded 64 MiB payload cap.
    constexpr std::uint64_t N = 882u, T = 1920u, S = 15u, M = 27u;
    constexpr std::uint64_t Gp = 27u, Gpn = 2089u, Gd = 6267u;
    constexpr std::uint64_t Gf = 4721337u, Gpivot = 6267u;
    constexpr std::uint64_t source464FullPayloadBytes = 39825484u;
    constexpr std::uint64_t float4Bytes = 16u, u32Bytes = 4u;
    constexpr std::uint64_t floatBytes = 4u, nodeBytes = 80u;
    constexpr std::uint64_t fgmresBytes = 32u, statusBytes = 48u;
    const std::array<std::uint64_t, 36> source464ArrayBytes{
        N*float4Bytes, N*float4Bytes,
        M*N*float4Bytes, M*N*float4Bytes,
        M*M*floatBytes, M*u32Bytes, u32Bytes,
        M*floatBytes, M*floatBytes,
        Gp*4u*u32Bytes, Gpn*u32Bytes, Gd*4u*u32Bytes,
        (N+1u)*u32Bytes, Gpn*u32Bytes, N*u32Bytes,
        Gf*floatBytes, Gf*floatBytes, Gpivot*u32Bytes,
        Gp*u32Bytes, Gpn*float4Bytes,
        N*nodeBytes, N*nodeBytes, T*S*floatBytes, T*u32Bytes,
        T*3u*float4Bytes, T*3u*float4Bytes,
        T*81u*floatBytes, T*u32Bytes,
        fgmresBytes, statusBytes, 4u*u32Bytes,
        floatBytes, floatBytes, floatBytes, u32Bytes, u32Bytes};
    std::uint64_t source464ComputedBytes = 0u;
    for (const auto arrayBytes : source464ArrayBytes) {
        std::uint64_t nextBytes = 0u;
        assert(checkedAppendBytes(source464ComputedBytes, arrayBytes,
            kAggregatePayloadCapBytes, nextBytes));
        source464ComputedBytes = nextBytes;
    }
    assert(source464ComputedBytes == source464FullPayloadBytes);
    assert(!checkedAppendBytes(0u, source464FullPayloadBytes,
        32ull * 1024ull * 1024ull, bytes));
    assert(checkedAppendBytes(0u, source464FullPayloadBytes,
        kAggregatePayloadCapBytes, bytes) && bytes == source464FullPayloadBytes);
    constexpr std::uint64_t source464GraphFloats = Gf;
    std::uint64_t graphMatrixBytes = 0u;
    assert(checkedArrayBytes({source464GraphFloats}, sizeof(float),
        kAggregatePayloadCapBytes, graphMatrixBytes));
    std::uint64_t graphPairBytes = 0u;
    assert(checkedAppendBytes(graphMatrixBytes, graphMatrixBytes,
        kAggregatePayloadCapBytes, graphPairBytes));
    assert(graphPairBytes == 37770696u);
    assert(graphPairBytes < source464FullPayloadBytes);
    assert(source464FullPayloadBytes > 32ull * 1024ull * 1024ull);
    assert(source464FullPayloadBytes <= kAggregatePayloadCapBytes);
    assert(kAggregatePayloadCapBytes - source464FullPayloadBytes == 27283380u);

    std::ifstream source(argv[1], std::ios::binary);
    assert(source.good());
    const std::string runtime((std::istreambuf_iterator<char>(source)),
                              std::istreambuf_iterator<char>());
    const auto apply = runtime.find(
        "dispatchThreads(\"nm_fgmres_graph_apply_patches\"");
    assert(apply != std::string::npos);
    const auto barrier = runtime.find(
        "[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];", apply);
    const auto capture = runtime.find(
        "writeRegionalConsumerCapture(", barrier);
    const auto femOperator = runtime.find(
        "dispatchThreads(\"nm_fem_apply_operator_elements\"", capture);
    assert(barrier != std::string::npos && capture != std::string::npos &&
        femOperator != std::string::npos && apply < barrier && barrier < capture &&
        capture < femOperator);
    assert(runtime.find("regionalConsumerCaptureRequest.enabled &&") !=
        std::string::npos);
    assert(runtime.find("aggregate_64MiB_payload_cap_exceeded") != std::string::npos);
    assert(runtime.find("singleTickScope(") != std::string::npos);
    assert(runtime.find("claimOnce(") != std::string::npos);
    assert(runtime.find("NM_FEM_NEWTON_TRACE_ROOT") != std::string::npos);
    assert(runtime.find("selector_attempt_count") != std::string::npos);
    assert(runtime.find("micro_generation") != std::string::npos);
    assert(runtime.find("augmented_coarse_lu") != std::string::npos);
    assert(runtime.find("augmented_effective_P") != std::string::npos);
    assert(runtime.find("augmented_AP") != std::string::npos);
    assert(runtime.find("graph_patch_original_matrices") != std::string::npos);
    assert(runtime.find("restart_generation") != std::string::npos);
    assert(runtime.find("restart_mode_valid") != std::string::npos);
    assert(runtime.find("restart_tail_preserved") != std::string::npos);
    assert(runtime.find("consumer restart r uses generator tag") !=
        std::string::npos);
    assert(runtime.find("not raw RAP; reconstruct raw RAP") !=
        std::string::npos);
    return 0;
}
