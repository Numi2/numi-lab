#include "numi/matter/shared.h"
#include "numi/matter/sparse_private_contact_admission.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <limits>

using namespace numi::matter;
using numi::matter::detail::admitSparsePrivateContact;

static_assert(sizeof(NMContactSampleGPU) == 160);
static_assert(sizeof(NMContactPairGPU) == 16);
static_assert(sizeof(NMDeformableContactCandidateGPU) == 16);
static_assert(sizeof(NMDeformableContactGPU) == 432);
static_assert(sizeof(NMDeformableContactHistoryGPU) == 48);
static_assert(sizeof(NMContinuumSurfacePrimitiveGPU) == 64);
static_assert(sizeof(NMFEMNodeStateGPU) == 80);
static_assert(sizeof(NMFEMElementVectorGPU) == 64);
static_assert(sizeof(NMIncidenceRangeGPU) == 16);
static_assert(sizeof(NMCohesiveFaceGPU) == 384);
static_assert(sizeof(NMCohesiveFaceVectorGPU) == 96);
static_assert(sizeof(NMDeformableContactFailureGPU) == 384);
static_assert(sizeof(NMIndirectDispatchGPU) == 16);

int main() {
    constexpr std::uint64_t N = 528444;
    constexpr std::uint64_t T = 1477728;
    constexpr std::uint64_t C = 260835;
    constexpr std::uint64_t F = 1054876;
    constexpr std::uint64_t contactEligibleNodes = 527964;
    constexpr std::uint64_t proxies = 12;
    constexpr std::uint64_t pairCount = contactEligibleNodes * proxies;
    constexpr std::uint64_t pairUpperBound = N * proxies;
    constexpr std::uint64_t K = 65536;
    constexpr std::uint64_t Kself = 131072;
    constexpr std::uint64_t leaf = std::bit_ceil(F);

    static_assert(pairCount == 6335568);
    static_assert(pairUpperBound == 6341328);
    static_assert(4 * T == 5910912);
    static_assert(leaf == 2097152);
    static_assert(4 * leaf < std::numeric_limits<std::uint32_t>::max());
    static_assert(pairCount < std::numeric_limits<std::uint32_t>::max());
    static_assert(6 * Kself < std::numeric_limits<std::uint32_t>::max());

    const auto sparse = admitSparsePrivateContact(
        1u, static_cast<std::uint32_t>(pairCount), static_cast<std::uint32_t>(K),
        false, true, true);
    if (!sparse.admitted() || sparse.totalBytes != 36090192u) return 1;
    const auto sparseUpper = admitSparsePrivateContact(
        1u, static_cast<std::uint32_t>(pairUpperBound), static_cast<std::uint32_t>(K),
        false, true, true);
    if (!sparseUpper.admitted() || sparseUpper.totalBytes != 36113232u) return 2;

    const std::uint64_t surfaceBytes =
        F * sizeof(NMContinuumSurfacePrimitiveGPU) +
        4 * leaf * sizeof(nm_float4) +
        4 * F * sizeof(std::uint32_t);
    if (surfaceBytes != 218607808u) return 3;

    // Mirrors runtime.mm's private full-contact Armijo arena formula. Production
    // surface/BVH/radix arrays are shared only when the explicit alias flag is
    // admitted; the self-contact capacity arena is always private.
    const std::uint64_t privateSelfSlots = Kself * (
        sizeof(NMDeformableContactCandidateGPU) +
        sizeof(NMDeformableContactGPU) + 2 * sizeof(std::uint32_t) +
        6 * sizeof(std::uint32_t));
    const std::uint64_t nodeRanges = N * sizeof(NMIncidenceRangeGPU);
    const std::uint64_t smallPrivate =
        4 * sizeof(std::uint32_t) + sizeof(NMIndirectDispatchGPU);
    const std::uint64_t privateSelfAliasesSurface =
        privateSelfSlots + nodeRanges + smallPrivate;
    const std::uint64_t privateSelfOwnSurface =
        privateSelfAliasesSurface + surfaceBytes;
    if (privateSelfSlots != 62914560u || nodeRanges != 8455104u ||
        privateSelfAliasesSurface != 71369696u ||
        privateSelfOwnSurface != 289977504u) return 4;

    const std::uint64_t cohesiveTrialBytes = C * (
        sizeof(NMCohesiveFaceGPU) + sizeof(NMCohesiveFaceVectorGPU));
    if (cohesiveTrialBytes != 125200800u) return 5;

    const std::uint64_t densePrivateRigidSampleBytes = pairCount *
        sizeof(NMContactSampleGPU);
    const std::uint64_t sparseSlotMapBytes = pairCount * sizeof(std::uint32_t);
    if (densePrivateRigidSampleBytes != 1013690880u ||
        sparseSlotMapBytes != 25342272u) return 6;

    std::printf(
        "{\"schema\":\"numi.cardboard.shipping_runtime_layout_proof.v1\","
        "\"nodes\":%llu,\"tetrahedra\":%llu,\"cohesive_faces\":%llu,"
        "\"surface_faces\":%llu,\"proxies\":%llu,\"eligible_pairs\":%llu,"
        "\"all_node_pair_upper_bound\":%llu,\"bvh_leaf_count\":%llu,"
        "\"surface_persistent_bytes\":%llu,"
        "\"production_rigid_sample_bytes_lower_bound\":%llu,"
        "\"sparse_private_K65536_bytes_exact_pairs\":%llu,"
        "\"sparse_private_K65536_bytes_upper_pairs\":%llu,"
        "\"private_self_K131072_bytes_alias_surface\":%llu,"
        "\"private_self_K131072_bytes_own_surface\":%llu,"
        "\"cohesive_private_one_face_plus_force_per_face_bytes\":%llu,"
        "\"u32_domains_safe_at_one_environment\":true,"
        "\"qualification\":false}\n",
        static_cast<unsigned long long>(N),
        static_cast<unsigned long long>(T),
        static_cast<unsigned long long>(C),
        static_cast<unsigned long long>(F),
        static_cast<unsigned long long>(proxies),
        static_cast<unsigned long long>(pairCount),
        static_cast<unsigned long long>(pairUpperBound),
        static_cast<unsigned long long>(leaf),
        static_cast<unsigned long long>(surfaceBytes),
        static_cast<unsigned long long>(densePrivateRigidSampleBytes),
        static_cast<unsigned long long>(sparse.totalBytes),
        static_cast<unsigned long long>(sparseUpper.totalBytes),
        static_cast<unsigned long long>(privateSelfAliasesSurface),
        static_cast<unsigned long long>(privateSelfOwnSurface),
        static_cast<unsigned long long>(cohesiveTrialBytes));
    return 0;
}
