#include "numi/matter/sparse_private_contact_admission.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
unsigned checks = 0u;

void require(const bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    using namespace numi::matter::detail;
    try {
        const auto tiny = admitSparsePrivateContact(1u, 1u, 1u, false, false);
        require(tiny.admitted() && tiny.blocksPerEnvironment == 1u &&
                tiny.pairSlotBytes == 16u && tiny.candidateIdBytes == 16u &&
                tiny.candidateCountBytes == 16u &&
                tiny.blockOffsetsBytes == 16u && tiny.totalBytes == 224u,
            "one-pair arena must include each 16-byte sentinel-padded typed buffer");

        const auto aboveDense =
            admitSparsePrivateContact(1u, 65537u, 64u, true, false);
        require(aboveDense.admitted() &&
                aboveDense.blocksPerEnvironment == 257u &&
                aboveDense.blockOffsetsBytes == 257u * 4u &&
                aboveDense.totalBytes == 64u * 160u + 65537u * 4u +
                    64u * 4u + 65537u * 160u + 16u + 257u * 4u,
            "parity-enabled block offset bytes are missing from admission");

        constexpr std::uint32_t shippingPairs = 6335568u;
        const auto shipping =
            admitSparsePrivateContact(1u, shippingPairs, 65536u, false, false);
        require(shipping.admitted() &&
                shipping.blocksPerEnvironment == 24749u &&
                shipping.blockOffsetsBytes == 98996u &&
                shipping.totalBytes == 36189188u &&
                shipping.totalBytes <= kSparsePrivateContactScratchBudgetBytes,
            "shipping-sized pair scan must fit the unchanged admitted scratch budget");

        const auto twoEnvironment =
            admitSparsePrivateContact(2u, shippingPairs, 65536u, false, false);
        require(!twoEnvironment.admitted() &&
                twoEnvironment.error ==
                    SparsePrivateContactAdmissionError::scratch_budget_exceeded &&
                twoEnvironment.blockOffsetsBytes == 197992u,
            "multi-environment admission must include block offsets and keep the old cap");

        const auto flattened = admitSparsePrivateContact(2u,
            std::numeric_limits<std::uint32_t>::max(), 1u, false, false);
        require(!flattened.admitted() && flattened.error ==
                SparsePrivateContactAdmissionError::flattened_pair_domain_exceeds_uint32,
            "flattened pair arithmetic must fail before block-size admission");

        std::cout << "sparse_private_admission_checks=" << checks
                  << " shipping_pair_count=" << shippingPairs
                  << " shipping_total_bytes=" << shipping.totalBytes
                  << " offset_bytes=" << shipping.blockOffsetsBytes
                  << " physical_transactions=0\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "sparse admission checker error: " << exception.what() << '\n';
        return 1;
    }
}
