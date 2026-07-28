#include "metalrobo/ParallelABASchedule.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <queue>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

ParallelABAScheduleDiagnostics fail(
    ParallelABAScheduleDiagnostics diagnostics,
    const ParallelABAScheduleStatus status,
    std::string message,
    const std::uint32_t articulation = MR_INVALID_INDEX
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    diagnostics.firstFailingArticulation = articulation;
    return diagnostics;
}

bool checkedU32(
    const std::size_t value,
    std::uint32_t& output
) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    output = static_cast<std::uint32_t>(value);
    return true;
}

bool checkedSpan(
    const std::uint32_t offset,
    const std::uint32_t count,
    const std::size_t capacity
) {
    return
        offset <= capacity &&
        count <= capacity - offset;
}

bool scalarOrFixedJoint(const std::uint32_t type) {
    return
        type == MR_JOINT_FIXED ||
        type == MR_JOINT_REVOLUTE ||
        type == MR_JOINT_CONTINUOUS ||
        type == MR_JOINT_PRISMATIC;
}

std::uint64_t hashBytes(
    std::uint64_t hash,
    const void* data,
    const std::size_t size
) {
    const auto* bytes =
        static_cast<const unsigned char*>(data);
    for (std::size_t index = 0u; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash;
}

template <typename T>
std::uint64_t hashVector(
    std::uint64_t hash,
    const std::vector<T>& values
) {
    const std::uint64_t size = values.size();
    hash = hashBytes(hash, &size, sizeof(size));
    if (!values.empty()) {
        hash = hashBytes(
            hash,
            values.data(),
            values.size() * sizeof(T)
        );
    }
    return hash;
}

std::uint64_t scheduleFingerprint(
    const ParallelABASchedule& schedule
) {
    const std::uint32_t version =
        MR_PARALLEL_ABA_SCHEDULE_ABI_VERSION;
    std::uint64_t hash =
        hashBytes(kFnvOffset, &version, sizeof(version));
    hash = hashVector(hash, schedule.articulations);
    hash = hashVector(hash, schedule.levels);
    hash = hashVector(hash, schedule.parentReductions);
    hash = hashVector(hash, schedule.levelBodies);
    hash = hashVector(hash, schedule.bodyOrder);
    hash = hashVector(hash, schedule.parentLocal);
    hash = hashVector(hash, schedule.inboundJoint);
    hash = hashVector(hash, schedule.childOffsets);
    hash = hashVector(hash, schedule.childIndices);
    return hash == 0u ? 1u : hash;
}

} // namespace

bool ParallelABASchedule::valid(std::string* reason) const {
    const auto reject = [&](std::string message) {
        if (reason != nullptr) {
            *reason = std::move(message);
        }
        return false;
    };
    if (articulations.empty()) {
        return reject("parallel ABA schedule has no articulations");
    }

    for (std::size_t articulationIndex = 0u;
         articulationIndex < articulations.size();
         ++articulationIndex) {
        const MRParallelABAArticulationGPU& descriptor =
            articulations[articulationIndex];
        if (descriptor.abiVersion !=
                MR_PARALLEL_ABA_SCHEDULE_ABI_VERSION ||
            descriptor.articulationIndex != articulationIndex ||
            descriptor.bodyCount == 0u ||
            descriptor.rootLocalBody >= descriptor.bodyCount ||
            descriptor.jointCount + 1u != descriptor.bodyCount ||
            descriptor.reserved0 != 0u ||
            descriptor.reserved1 != 0u) {
            return reject(
                "parallel ABA articulation descriptor is malformed"
            );
        }
        constexpr std::uint32_t knownFlags =
            MR_PARALLEL_ABA_FIXED_ROOT |
            MR_PARALLEL_ABA_FLOATING_ROOT |
            MR_PARALLEL_ABA_SERIAL_CHAIN |
            MR_PARALLEL_ABA_BRANCHING;
        const std::uint32_t rootFlags =
            descriptor.flags &
            (
                MR_PARALLEL_ABA_FIXED_ROOT |
                MR_PARALLEL_ABA_FLOATING_ROOT
            );
        if ((descriptor.flags & ~knownFlags) != 0u ||
            (rootFlags != MR_PARALLEL_ABA_FIXED_ROOT &&
             rootFlags != MR_PARALLEL_ABA_FLOATING_ROOT) ||
            ((descriptor.flags & MR_PARALLEL_ABA_SERIAL_CHAIN) !=
                 0u &&
             (descriptor.flags & MR_PARALLEL_ABA_BRANCHING) != 0u)) {
            return reject(
                "parallel ABA articulation flags are inconsistent"
            );
        }
        if (!checkedSpan(
                descriptor.forwardLevelOffset,
                descriptor.forwardLevelCount,
                levels.size()
            ) ||
            !checkedSpan(
                descriptor.reverseLevelOffset,
                descriptor.reverseLevelCount,
                levels.size()
            ) ||
            !checkedSpan(
                descriptor.bodyOrderOffset,
                descriptor.bodyCount,
                bodyOrder.size()
            ) ||
            !checkedSpan(
                descriptor.parentLocalOffset,
                descriptor.bodyCount,
                parentLocal.size()
            ) ||
            !checkedSpan(
                descriptor.inboundJointOffset,
                descriptor.bodyCount,
                inboundJoint.size()
            ) ||
            !checkedSpan(
                descriptor.childOffsetOffset,
                descriptor.bodyCount + 1u,
                childOffsets.size()
            ) ||
            !checkedSpan(
                descriptor.childIndexOffset,
                descriptor.childIndexCount,
                childIndices.size()
            )) {
            return reject(
                "parallel ABA articulation stream span is out of range"
            );
        }
        if (descriptor.forwardLevelCount !=
                descriptor.maximumDepth + 1u ||
            descriptor.reverseLevelCount !=
                descriptor.maximumDepth) {
            return reject(
                "parallel ABA level count does not match tree depth"
            );
        }

        std::vector<std::uint8_t> seen(
            descriptor.bodyCount,
            0u
        );
        for (std::uint32_t bodyOrdinal = 0u;
             bodyOrdinal < descriptor.bodyCount;
             ++bodyOrdinal) {
            const std::uint32_t localBody = bodyOrder[
                descriptor.bodyOrderOffset + bodyOrdinal
            ];
            if (localBody >= descriptor.bodyCount ||
                seen[localBody] != 0u) {
                return reject(
                    "parallel ABA body order is not a permutation"
                );
            }
            seen[localBody] = 1u;
        }
        if (bodyOrder[descriptor.bodyOrderOffset] !=
            descriptor.rootLocalBody) {
            return reject(
                "parallel ABA body order does not begin at the root"
            );
        }
        for (std::uint32_t localBody = 0u;
             localBody < descriptor.bodyCount;
             ++localBody) {
            const std::uint32_t parent = parentLocal[
                descriptor.parentLocalOffset + localBody
            ];
            const std::uint32_t joint = inboundJoint[
                descriptor.inboundJointOffset + localBody
            ];
            if (localBody == descriptor.rootLocalBody) {
                if (parent != MR_INVALID_INDEX ||
                    joint != MR_INVALID_INDEX) {
                    return reject(
                        "parallel ABA root owns an inbound edge"
                    );
                }
            } else if (parent >= descriptor.bodyCount ||
                       parent == localBody ||
                       joint == MR_INVALID_INDEX) {
                return reject(
                    "parallel ABA non-root edge is malformed"
                );
            }
        }

        const std::uint32_t childBegin = childOffsets[
            descriptor.childOffsetOffset
        ];
        const std::uint32_t childEnd = childOffsets[
            descriptor.childOffsetOffset + descriptor.bodyCount
        ];
        if (childBegin != descriptor.childIndexOffset ||
            childEnd != descriptor.childIndexOffset +
                descriptor.childIndexCount ||
            descriptor.childIndexCount != descriptor.jointCount) {
            return reject(
                "parallel ABA child CSR does not cover tree edges"
            );
        }
        for (std::uint32_t localBody = 0u;
             localBody < descriptor.bodyCount;
             ++localBody) {
            const std::uint32_t first = childOffsets[
                descriptor.childOffsetOffset + localBody
            ];
            const std::uint32_t end = childOffsets[
                descriptor.childOffsetOffset + localBody + 1u
            ];
            if (first > end ||
                first < descriptor.childIndexOffset ||
                end > descriptor.childIndexOffset +
                    descriptor.childIndexCount) {
                return reject(
                    "parallel ABA child CSR offset is malformed"
                );
            }
            std::uint32_t previous = 0u;
            for (std::uint32_t child = first;
                 child < end;
                 ++child) {
                const std::uint32_t localChild =
                    childIndices[child];
                if (localChild >= descriptor.bodyCount ||
                    parentLocal[
                        descriptor.parentLocalOffset + localChild
                    ] != localBody ||
                    (child != first && localChild <= previous)) {
                    return reject(
                        "parallel ABA child list is not stable"
                    );
                }
                previous = localChild;
            }
        }

        std::uint32_t forwardBodies = 0u;
        for (std::uint32_t levelIndex = 0u;
             levelIndex < descriptor.forwardLevelCount;
             ++levelIndex) {
            const MRParallelABALevelGPU& level = levels[
                descriptor.forwardLevelOffset + levelIndex
            ];
            if (!checkedSpan(
                    level.bodyOffset,
                    level.bodyCount,
                    levelBodies.size()
                ) ||
                level.parentReductionOffset != 0u ||
                level.parentReductionCount != 0u ||
                level.bodyCount == 0u) {
                return reject(
                    "parallel ABA forward frontier is malformed"
                );
            }
            forwardBodies += level.bodyCount;
        }
        if (forwardBodies != descriptor.bodyCount) {
            return reject(
                "parallel ABA forward frontiers do not cover bodies"
            );
        }
        std::uint32_t reverseBodies = 0u;
        for (std::uint32_t levelIndex = 0u;
             levelIndex < descriptor.reverseLevelCount;
             ++levelIndex) {
            const MRParallelABALevelGPU& level = levels[
                descriptor.reverseLevelOffset + levelIndex
            ];
            if (!checkedSpan(
                    level.bodyOffset,
                    level.bodyCount,
                    levelBodies.size()
                ) ||
                !checkedSpan(
                    level.parentReductionOffset,
                    level.parentReductionCount,
                    parentReductions.size()
                ) ||
                level.bodyCount == 0u ||
                level.parentReductionCount == 0u) {
                return reject(
                    "parallel ABA reverse frontier is malformed"
                );
            }
            reverseBodies += level.bodyCount;
        }
        if (reverseBodies + 1u != descriptor.bodyCount) {
            return reject(
                "parallel ABA reverse frontiers do not cover non-root bodies"
            );
        }
    }

    if (fingerprint == 0u ||
        fingerprint != scheduleFingerprint(*this)) {
        return reject("parallel ABA schedule fingerprint mismatch");
    }
    if (reason != nullptr) {
        reason->clear();
    }
    return true;
}

ParallelABAScheduleDiagnostics compileParallelABASchedule(
    const EngineModel& model,
    ParallelABASchedule& output
) {
    ParallelABAScheduleDiagnostics diagnostics{};
    diagnostics.articulationCount = static_cast<std::uint32_t>(
        model.articulations.size()
    );
    std::string modelReason;
    if (!model.valid(&modelReason)) {
        return fail(
            std::move(diagnostics),
            ParallelABAScheduleStatus::invalidModel,
            "invalid EngineModel: " + modelReason
        );
    }
    if (model.articulations.empty()) {
        return fail(
            std::move(diagnostics),
            ParallelABAScheduleStatus::unsupportedTopology,
            "parallel ABA requires at least one articulation"
        );
    }

    try {
        ParallelABASchedule staged;
        staged.articulations.reserve(model.articulations.size());

        for (std::size_t articulationIndex = 0u;
             articulationIndex < model.articulations.size();
             ++articulationIndex) {
            const MRArticulationGPU& articulation =
                model.articulations[articulationIndex];
            if ((articulation.rootType != MR_ROOT_FIXED &&
                 articulation.rootType != MR_ROOT_FLOATING) ||
                articulation.bodyCount == 0u ||
                articulation.jointCount + 1u !=
                    articulation.bodyCount ||
                articulation.rootBody < articulation.firstBody ||
                articulation.rootBody >=
                    articulation.firstBody +
                        articulation.bodyCount) {
                return fail(
                    std::move(diagnostics),
                    ParallelABAScheduleStatus::unsupportedTopology,
                    "articulation is not a supported rooted tree",
                    static_cast<std::uint32_t>(articulationIndex)
                );
            }
            const std::uint32_t rootLocal =
                articulation.rootBody - articulation.firstBody;
            std::vector<std::uint32_t> parent(
                articulation.bodyCount,
                MR_INVALID_INDEX
            );
            std::vector<std::uint32_t> inbound(
                articulation.bodyCount,
                MR_INVALID_INDEX
            );
            std::vector<std::vector<std::uint32_t>> children(
                articulation.bodyCount
            );
            for (std::uint32_t localJoint = 0u;
                 localJoint < articulation.jointCount;
                 ++localJoint) {
                const std::uint32_t globalJoint =
                    articulation.firstJoint + localJoint;
                const MRJointDescriptorGPU& joint =
                    model.joints[globalJoint];
                if (!scalarOrFixedJoint(joint.jointType) ||
                    joint.parentBody < articulation.firstBody ||
                    joint.parentBody >= articulation.firstBody +
                        articulation.bodyCount ||
                    joint.childBody < articulation.firstBody ||
                    joint.childBody >= articulation.firstBody +
                        articulation.bodyCount) {
                    return fail(
                        std::move(diagnostics),
                        ParallelABAScheduleStatus::unsupportedTopology,
                        "articulation contains an unsupported tree edge",
                        static_cast<std::uint32_t>(articulationIndex)
                    );
                }
                const std::uint32_t localParent =
                    joint.parentBody - articulation.firstBody;
                const std::uint32_t localChild =
                    joint.childBody - articulation.firstBody;
                if (localChild == rootLocal ||
                    localChild == localParent ||
                    inbound[localChild] != MR_INVALID_INDEX) {
                    return fail(
                        std::move(diagnostics),
                        ParallelABAScheduleStatus::unsupportedTopology,
                        "articulation is cyclic or has multiple parents",
                        static_cast<std::uint32_t>(articulationIndex)
                    );
                }
                parent[localChild] = localParent;
                inbound[localChild] = globalJoint;
                children[localParent].push_back(localChild);
            }
            if (parent[rootLocal] != MR_INVALID_INDEX ||
                inbound[rootLocal] != MR_INVALID_INDEX) {
                return fail(
                    std::move(diagnostics),
                    ParallelABAScheduleStatus::unsupportedTopology,
                    "articulation root has an inbound edge",
                    static_cast<std::uint32_t>(articulationIndex)
                );
            }
            bool branching = false;
            for (std::vector<std::uint32_t>& childList : children) {
                std::ranges::sort(childList);
                branching = branching || childList.size() > 1u;
            }

            std::vector<std::uint32_t> depth(
                articulation.bodyCount,
                MR_INVALID_INDEX
            );
            std::vector<std::vector<std::uint32_t>> frontiers(1u);
            frontiers[0].push_back(rootLocal);
            depth[rootLocal] = 0u;
            std::vector<std::uint32_t> order;
            order.reserve(articulation.bodyCount);
            for (std::size_t levelIndex = 0u;
                 levelIndex < frontiers.size();
                 ++levelIndex) {
                std::vector<std::uint32_t>& level =
                    frontiers[levelIndex];
                std::ranges::sort(level);
                for (const std::uint32_t localBody : level) {
                    order.push_back(localBody);
                    for (const std::uint32_t child :
                         children[localBody]) {
                        if (depth[child] != MR_INVALID_INDEX) {
                            return fail(
                                std::move(diagnostics),
                                ParallelABAScheduleStatus::
                                    unsupportedTopology,
                                "articulation tree contains a cycle",
                                static_cast<std::uint32_t>(
                                    articulationIndex
                                )
                            );
                        }
                        depth[child] = static_cast<std::uint32_t>(
                            levelIndex + 1u
                        );
                        if (frontiers.size() == levelIndex + 1u) {
                            frontiers.emplace_back();
                        }
                        frontiers[levelIndex + 1u].push_back(child);
                    }
                }
            }
            if (order.size() != articulation.bodyCount) {
                return fail(
                    std::move(diagnostics),
                    ParallelABAScheduleStatus::unsupportedTopology,
                    "articulation tree is disconnected",
                    static_cast<std::uint32_t>(articulationIndex)
                );
            }

            MRParallelABAArticulationGPU descriptor{};
            descriptor.abiVersion =
                MR_PARALLEL_ABA_SCHEDULE_ABI_VERSION;
            descriptor.articulationIndex =
                static_cast<std::uint32_t>(articulationIndex);
            descriptor.rootLocalBody = rootLocal;
            descriptor.bodyCount = articulation.bodyCount;
            descriptor.jointCount = articulation.jointCount;
            descriptor.flags =
                articulation.rootType == MR_ROOT_FLOATING
                ? MR_PARALLEL_ABA_FLOATING_ROOT
                : MR_PARALLEL_ABA_FIXED_ROOT;
            descriptor.flags |= branching
                ? MR_PARALLEL_ABA_BRANCHING
                : MR_PARALLEL_ABA_SERIAL_CHAIN;
            descriptor.maximumDepth = static_cast<std::uint32_t>(
                frontiers.size() - 1u
            );
            descriptor.maximumLevelWidth = 0u;
            for (const auto& level : frontiers) {
                descriptor.maximumLevelWidth = std::max(
                    descriptor.maximumLevelWidth,
                    static_cast<std::uint32_t>(level.size())
                );
            }
            diagnostics.maximumDepth = std::max(
                diagnostics.maximumDepth,
                descriptor.maximumDepth
            );
            diagnostics.maximumLevelWidth = std::max(
                diagnostics.maximumLevelWidth,
                descriptor.maximumLevelWidth
            );

            if (!checkedU32(
                    staged.levels.size(),
                    descriptor.forwardLevelOffset
                )) {
                return fail(
                    std::move(diagnostics),
                    ParallelABAScheduleStatus::capacityOverflow,
                    "parallel ABA forward-level offset overflow",
                    static_cast<std::uint32_t>(articulationIndex)
                );
            }
            descriptor.forwardLevelCount =
                static_cast<std::uint32_t>(frontiers.size());
            for (const std::vector<std::uint32_t>& frontier :
                 frontiers) {
                MRParallelABALevelGPU level{};
                if (!checkedU32(
                        staged.levelBodies.size(),
                        level.bodyOffset
                    )) {
                    return fail(
                        std::move(diagnostics),
                        ParallelABAScheduleStatus::capacityOverflow,
                        "parallel ABA level-body offset overflow",
                        static_cast<std::uint32_t>(
                            articulationIndex
                        )
                    );
                }
                level.bodyCount =
                    static_cast<std::uint32_t>(frontier.size());
                staged.levelBodies.insert(
                    staged.levelBodies.end(),
                    frontier.begin(),
                    frontier.end()
                );
                staged.levels.push_back(level);
            }

            if (!checkedU32(
                    staged.levels.size(),
                    descriptor.reverseLevelOffset
                )) {
                return fail(
                    std::move(diagnostics),
                    ParallelABAScheduleStatus::capacityOverflow,
                    "parallel ABA reverse-level offset overflow",
                    static_cast<std::uint32_t>(articulationIndex)
                );
            }
            descriptor.reverseLevelCount =
                descriptor.maximumDepth;
            std::uint32_t stableReductionOrdinal = 0u;
            for (std::size_t reverse = 0u;
                 reverse < descriptor.maximumDepth;
                 ++reverse) {
                const std::size_t depthIndex =
                    frontiers.size() - 1u - reverse;
                const std::vector<std::uint32_t>& frontier =
                    frontiers[depthIndex];
                MRParallelABALevelGPU level{};
                if (!checkedU32(
                        staged.levelBodies.size(),
                        level.bodyOffset
                    ) ||
                    !checkedU32(
                        staged.parentReductions.size(),
                        level.parentReductionOffset
                    )) {
                    return fail(
                        std::move(diagnostics),
                        ParallelABAScheduleStatus::capacityOverflow,
                        "parallel ABA reverse frontier overflow",
                        static_cast<std::uint32_t>(
                            articulationIndex
                        )
                    );
                }
                level.bodyCount =
                    static_cast<std::uint32_t>(frontier.size());
                staged.levelBodies.insert(
                    staged.levelBodies.end(),
                    frontier.begin(),
                    frontier.end()
                );
                std::vector<std::uint32_t> parents;
                parents.reserve(frontier.size());
                for (const std::uint32_t localBody : frontier) {
                    parents.push_back(parent[localBody]);
                }
                std::ranges::sort(parents);
                parents.erase(
                    std::unique(parents.begin(), parents.end()),
                    parents.end()
                );
                level.parentReductionCount =
                    static_cast<std::uint32_t>(parents.size());
                for (const std::uint32_t localParent : parents) {
                    MRParallelABAParentReductionGPU reduction{};
                    reduction.parentLocalBody = localParent;
                    reduction.childCount = static_cast<std::uint32_t>(
                        children[localParent].size()
                    );
                    reduction.stableOrdinal =
                        stableReductionOrdinal++;
                    staged.parentReductions.push_back(reduction);
                }
                staged.levels.push_back(level);
            }

            if (!checkedU32(
                    staged.bodyOrder.size(),
                    descriptor.bodyOrderOffset
                ) ||
                !checkedU32(
                    staged.parentLocal.size(),
                    descriptor.parentLocalOffset
                ) ||
                !checkedU32(
                    staged.inboundJoint.size(),
                    descriptor.inboundJointOffset
                ) ||
                !checkedU32(
                    staged.childOffsets.size(),
                    descriptor.childOffsetOffset
                ) ||
                !checkedU32(
                    staged.childIndices.size(),
                    descriptor.childIndexOffset
                )) {
                return fail(
                    std::move(diagnostics),
                    ParallelABAScheduleStatus::capacityOverflow,
                    "parallel ABA topology stream offset overflow",
                    static_cast<std::uint32_t>(articulationIndex)
                );
            }
            staged.bodyOrder.insert(
                staged.bodyOrder.end(),
                order.begin(),
                order.end()
            );
            staged.parentLocal.insert(
                staged.parentLocal.end(),
                parent.begin(),
                parent.end()
            );
            staged.inboundJoint.insert(
                staged.inboundJoint.end(),
                inbound.begin(),
                inbound.end()
            );
            for (std::uint32_t localBody = 0u;
                 localBody < articulation.bodyCount;
                 ++localBody) {
                staged.childOffsets.push_back(
                    static_cast<std::uint32_t>(
                        staged.childIndices.size()
                    )
                );
                staged.childIndices.insert(
                    staged.childIndices.end(),
                    children[localBody].begin(),
                    children[localBody].end()
                );
            }
            staged.childOffsets.push_back(
                static_cast<std::uint32_t>(
                    staged.childIndices.size()
                )
            );
            descriptor.childIndexCount = articulation.jointCount;

            // Parent reductions are emitted before the CSR offsets are known.
            // Patch their absolute stable child spans now.
            for (std::uint32_t reverseLevel = 0u;
                 reverseLevel < descriptor.reverseLevelCount;
                 ++reverseLevel) {
                MRParallelABALevelGPU& level = staged.levels[
                    descriptor.reverseLevelOffset + reverseLevel
                ];
                for (std::uint32_t reductionIndex = 0u;
                     reductionIndex < level.parentReductionCount;
                     ++reductionIndex) {
                    MRParallelABAParentReductionGPU& reduction =
                        staged.parentReductions[
                            level.parentReductionOffset +
                            reductionIndex
                        ];
                    reduction.firstChildIndex = staged.childOffsets[
                        descriptor.childOffsetOffset +
                        reduction.parentLocalBody
                    ];
                }
            }
            staged.articulations.push_back(descriptor);
        }

        staged.fingerprint = scheduleFingerprint(staged);
        std::string scheduleReason;
        if (!staged.valid(&scheduleReason)) {
            return fail(
                std::move(diagnostics),
                ParallelABAScheduleStatus::invalidCompiledSchedule,
                "compiled parallel ABA schedule is invalid: " +
                    scheduleReason
            );
        }
        output = std::move(staged);
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return fail(
            std::move(diagnostics),
            ParallelABAScheduleStatus::allocationFailure,
            "parallel ABA schedule allocation failed"
        );
    }
}

const char* parallelABAScheduleStatusName(
    const ParallelABAScheduleStatus status
) noexcept {
    switch (status) {
    case ParallelABAScheduleStatus::success:
        return "success";
    case ParallelABAScheduleStatus::invalidModel:
        return "invalid_model";
    case ParallelABAScheduleStatus::unsupportedTopology:
        return "unsupported_topology";
    case ParallelABAScheduleStatus::capacityOverflow:
        return "capacity_overflow";
    case ParallelABAScheduleStatus::invalidCompiledSchedule:
        return "invalid_compiled_schedule";
    case ParallelABAScheduleStatus::allocationFailure:
        return "allocation_failure";
    }
    return "unknown";
}

} // namespace metalrobo
