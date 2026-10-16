#include "numi/matter/shared.h"
#include <cstdint>
#include <iostream>

int main() {
    std::cout << "sizeof_NMContactSampleGPU=" << sizeof(NMContactSampleGPU) << '\n';
    std::cout << "sizeof_NMMatterDispatchGPU=" << sizeof(NMMatterDispatchGPU) << '\n';
    std::cout << "sizeof_NMMatterStatusGPU=" << sizeof(NMMatterStatusGPU) << '\n';

    constexpr std::uint64_t pairs = 6335568ull;
    constexpr std::uint64_t capacity = 65536ull;
    constexpr std::uint64_t blocks = (pairs + 255ull) / 256ull;
    constexpr std::uint64_t pairSlots = pairs * sizeof(std::uint32_t);
    constexpr std::uint64_t candidateIDs = capacity * sizeof(std::uint32_t);
    constexpr std::uint64_t offsets = blocks * sizeof(std::uint32_t);
    constexpr std::uint64_t countBytes = sizeof(std::uint32_t);
    constexpr std::uint64_t dispatchBytes = sizeof(NMMatterDispatchGPU);
    constexpr std::uint64_t statusBytes = sizeof(NMMatterStatusGPU);
    constexpr std::uint64_t scalarInputsBytes = 2u * sizeof(std::uint32_t);
    constexpr std::uint64_t outputVectorCopies = 3u *
        (pairSlots + candidateIDs + countBytes + statusBytes);

    std::cout << "shipping_pair_count=" << pairs << '\n';
    std::cout << "shipping_candidate_capacity=" << capacity << '\n';
    std::cout << "shipping_blocks=" << blocks << '\n';
    std::cout << "checker_pair_slot_buffer_bytes=" << pairSlots << '\n';
    std::cout << "checker_candidate_id_buffer_bytes=" << candidateIDs << '\n';
    std::cout << "checker_block_offset_buffer_bytes=" << offsets << '\n';
    std::cout << "checker_count_buffer_bytes=" << countBytes << '\n';
    std::cout << "checker_dispatch_buffer_bytes=" << dispatchBytes << '\n';
    std::cout << "checker_status_buffer_bytes=" << statusBytes << '\n';
    std::cout << "checker_capacity_and_block_scalar_bytes=" << scalarInputsBytes << '\n';
    std::cout << "checker_one_gpu_run_buffer_bytes="
              << pairSlots + candidateIDs + offsets + countBytes + dispatchBytes +
                     statusBytes + scalarInputsBytes << '\n';
    std::cout << "three_cpu_output_vector_copies_estimate_bytes="
              << outputVectorCopies << '\n';
    std::cout << "runtime_candidate_sample_buffer_bytes_not_allocated_by_checker="
              << capacity * sizeof(NMContactSampleGPU) << '\n';
    return 0;
}
