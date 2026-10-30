#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

// Internal, bounded importer profiles. The larger reference body is an explicit
// build choice; the original profile remains the default. Neither profile
// changes material, determinant, provenance, mass, contact or runtime gates.
// Actual Metal allocation and accepted-scene qualification remain mandatory.
namespace metalrobo::human::coarse_tissue::physical_skin_child_capacity_detail {
#if defined(NUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY)
#if NUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY != 1
#error NUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY must be exactly 1 when defined
#endif
inline constexpr bool fullReference = true;
inline constexpr std::size_t maxNodes = 500000u;
inline constexpr std::size_t maxCells = 2000000u;
inline constexpr std::size_t maxRoleNodes = 500000u;
inline constexpr std::size_t maxMechanicalNodes = 500000u;
inline constexpr std::size_t maxBoundaryFaces = 2000000u;
inline constexpr std::size_t maxCandidateEdges = 12000000u;
inline constexpr std::size_t maxAugmentations = 4000000u;
#else
inline constexpr bool fullReference = false;
inline constexpr std::size_t maxNodes = 300000u;
inline constexpr std::size_t maxCells = 200000u;
inline constexpr std::size_t maxRoleNodes = 200000u;
inline constexpr std::size_t maxMechanicalNodes = 200000u;
inline constexpr std::size_t maxBoundaryFaces = 200000u;
inline constexpr std::size_t maxCandidateEdges = 3000000u;
inline constexpr std::size_t maxAugmentations = 300000u;
#endif
static_assert(maxNodes <= std::numeric_limits<std::uint32_t>::max());
static_assert(maxCandidateEdges <= std::numeric_limits<std::size_t>::max());
static_assert(maxAugmentations <= std::numeric_limits<std::size_t>::max());
static_assert(maxCells <= std::numeric_limits<std::uint32_t>::max()/4u);
static_assert(maxCells <= std::numeric_limits<std::size_t>::max()/(81u*sizeof(float)));
}
