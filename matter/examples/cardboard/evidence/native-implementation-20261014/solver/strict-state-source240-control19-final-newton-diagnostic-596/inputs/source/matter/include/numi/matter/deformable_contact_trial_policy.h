#ifndef NUMI_MATTER_DEFORMABLE_CONTACT_TRIAL_POLICY_H
#define NUMI_MATTER_DEFORMABLE_CONTACT_TRIAL_POLICY_H

// Shared host/Metal policy classification for one deformable-contact support.
// The global flag is set only when exact full-contact residual Armijo is active.
namespace numi_matter_contact_policy {
static inline unsigned int addSupport(unsigned int mask,
                                      unsigned int representationBit) {
    return mask | representationBit;
}
static inline bool useFEMFeasibilityOnly(unsigned int fullResidualArmijo,
                                         unsigned int supportMask) {
    const unsigned int femBit = 1u;
    return fullResidualArmijo != 0u && supportMask == femBit;
}
} // namespace numi_matter_contact_policy

#endif
