#include <iostream>
#include "numi/matter/deformable_contact_trial_policy.h"

int main() {
    using numi_matter_contact_policy::addSupport;
    using numi_matter_contact_policy::useFEMFeasibilityOnly;
    constexpr unsigned int fem = 1u;
    constexpr unsigned int mpm = 2u;
    const unsigned int noSupport = 0u;
    const unsigned int femOnly = addSupport(noSupport, fem);
    const unsigned int mpmOnly = addSupport(noSupport, mpm);
    const unsigned int mixed = addSupport(addSupport(noSupport, fem), mpm);
    if (useFEMFeasibilityOnly(0u, femOnly)) return 1;
    if (!useFEMFeasibilityOnly(1u, femOnly)) return 2;
    if (useFEMFeasibilityOnly(1u, mpmOnly)) return 3;
    if (useFEMFeasibilityOnly(1u, mixed)) return 4;
    if (useFEMFeasibilityOnly(1u, noSupport)) return 5;
    if (useFEMFeasibilityOnly(0u, mixed)) return 6;
    if (useFEMFeasibilityOnly(1u, fem | 4u)) return 7;
    std::cout << "shared production policy helper: 7 cases passed\n";
    return 0;
}
