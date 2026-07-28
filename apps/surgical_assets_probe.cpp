#include "metalrobo/Collision.hpp"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/SurgicalAssets.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

mr_float4 f4(
    const double x,
    const double y,
    const double z,
    const double w = 0.0
) {
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z),
        static_cast<float>(w),
    };
}

MRBodyStateGPU bodyState(
    const double x,
    const double y,
    const double z,
    const std::uint32_t motion
) {
    MRBodyStateGPU result{};
    result.position = f4(x, y, z, 1.0);
    result.orientation = f4(0.0, 0.0, 0.0, 1.0);
    result.flagsAndIndices[0] = motion;
    return result;
}

MRShapeGPU planeShape(const std::uint32_t body) {
    MRShapeGPU result{};
    result.bodyIndex = body;
    result.shapeType = MR_SHAPE_PLANE;
    result.collisionGroup = 1u;
    result.collisionMask = 1u;
    result.slotGeneration = 900000u;
    result.localPosition = f4(0.0, 0.0, 0.0, 1.0);
    result.localRotation = f4(0.0, 0.0, 0.0, 1.0);
    return result;
}

MRShapeGPU jawShape(
    const std::uint32_t body,
    const std::uint32_t generation,
    const mr_float4 rotation,
    const double radius,
    const double halfLength
) {
    MRShapeGPU result{};
    result.bodyIndex = body;
    result.shapeType = MR_SHAPE_CAPSULE;
    result.collisionGroup = 1u;
    result.collisionMask = 1u;
    result.slotGeneration = generation;
    result.localPosition = f4(0.0, 0.0, 0.0, 1.0);
    result.localRotation = rotation;
    result.dimensions = f4(radius, halfLength, 0.0, 0.0);
    result.contactRestAndBoundingRadius =
        f4(0.00002, 0.0, radius + halfLength, 0.0);
    return result;
}

metalrobo::CollisionConfig collisionConfig(
    const std::uint32_t pairCapacity = 2048u,
    const std::uint32_t contactCapacity = 4096u
) {
    metalrobo::CollisionConfig result;
    result.environment = 19u;
    result.capacities = {
        .pairCapacity = pairCapacity,
        .rawContactCapacity = contactCapacity,
        .manifoldCapacity = pairCapacity,
    };
    result.manifoldBreakingSeparation = 0.003;
    result.manifoldBreakingTangential = 0.003;
    result.manifoldMergeDistance = 0.0001;
    result.manifoldNormalCosine = 0.95;
    return result;
}

double inertiaInverseResidual(const MRBodyPropertiesGPU& body) {
    const double inertia[3][3] = {
        {
            body.inertiaRow0.x,
            body.inertiaRow0.y,
            body.inertiaRow0.z,
        },
        {
            body.inertiaRow1.x,
            body.inertiaRow1.y,
            body.inertiaRow1.z,
        },
        {
            body.inertiaRow2.x,
            body.inertiaRow2.y,
            body.inertiaRow2.z,
        },
    };
    const double inverse[3][3] = {
        {
            body.inverseInertiaRow0.x,
            body.inverseInertiaRow0.y,
            body.inverseInertiaRow0.z,
        },
        {
            body.inverseInertiaRow1.x,
            body.inverseInertiaRow1.y,
            body.inverseInertiaRow1.z,
        },
        {
            body.inverseInertiaRow2.x,
            body.inverseInertiaRow2.y,
            body.inverseInertiaRow2.z,
        },
    };
    double maximum = 0.0;
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0u; inner < 3u; ++inner) {
                value += inertia[row][inner] * inverse[inner][column];
            }
            maximum = std::max(
                maximum,
                std::abs(
                    value -
                    (row == column ? 1.0 : 0.0)
                )
            );
        }
    }
    return maximum;
}

void requirePositiveInertia(
    const metalrobo::SurgicalRigidAsset& asset,
    const std::string& label
) {
    const MRBodyPropertiesGPU& body = asset.body;
    require(asset.volumeM3 > 0.0, label + " volume is not positive");
    require(asset.massKg > 0.0, label + " mass is not positive");
    require(
        std::abs(
            static_cast<double>(body.massAndInverseMass.x) -
            asset.massKg
        ) <= asset.massKg * 2.0e-7,
        label + " body mass does not match geometry mass"
    );
    require(
        std::abs(
            static_cast<double>(body.massAndInverseMass.x) *
                body.massAndInverseMass.y -
            1.0
        ) < 2.0e-7,
        label + " inverse mass is inconsistent"
    );
    require(
        body.inertiaRow0.x > 0.0f &&
        body.inertiaRow1.y > 0.0f &&
        body.inertiaRow2.z > 0.0f,
        label + " inertia diagonal is not positive"
    );
    require(
        inertiaInverseResidual(body) < 2.0e-6,
        label + " inverse inertia residual is too large"
    );
}

bool sameRigidAsset(
    const metalrobo::SurgicalRigidAsset& a,
    const metalrobo::SurgicalRigidAsset& b
) {
    return
        std::memcmp(&a.body, &b.body, sizeof(a.body)) == 0 &&
        std::memcmp(&a.material, &b.material, sizeof(a.material)) == 0 &&
        a.shapes.size() == b.shapes.size() &&
        std::memcmp(
            a.shapes.data(),
            b.shapes.data(),
            a.shapes.size() * sizeof(MRShapeGPU)
        ) == 0 &&
        a.volumeM3 == b.volumeM3 &&
        a.massKg == b.massKg &&
        a.geometryCenterOfMassM == b.geometryCenterOfMassM &&
        a.localAabbLowerM == b.localAabbLowerM &&
        a.localAabbUpperM == b.localAabbUpperM;
}

std::array<double, 6> collisionAabb(
    const metalrobo::SurgicalRigidAsset& asset
) {
    std::vector<MRBodyStateGPU> bodies(
        static_cast<std::size_t>(asset.shapes.front().bodyIndex) + 1u
    );
    for (MRBodyStateGPU& body : bodies) {
        body = bodyState(0.0, 0.0, 0.0, MR_MOTION_STATIC);
    }
    bodies.back() = bodyState(0.0, 0.0, 0.0, asset.body.motionType);
    metalrobo::PersistentManifoldCache cache;
    const metalrobo::CollisionFrame frame =
        metalrobo::collideCpuReference(
            asset.shapes,
            bodies,
            collisionConfig(),
            cache
        );
    require(frame.succeeded(), "asset AABB collision compilation failed");
    require(
        frame.worldAabbs.size() == asset.shapes.size(),
        "asset AABB count mismatch"
    );

    std::array<double, 6> result{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (const MRAabbGPU& aabb : frame.worldAabbs) {
        result[0] = std::min(result[0], static_cast<double>(aabb.lower.x));
        result[1] = std::min(result[1], static_cast<double>(aabb.lower.y));
        result[2] = std::min(result[2], static_cast<double>(aabb.lower.z));
        result[3] = std::max(result[3], static_cast<double>(aabb.upper.x));
        result[4] = std::max(result[4], static_cast<double>(aabb.upper.y));
        result[5] = std::max(result[5], static_cast<double>(aabb.upper.z));
    }
    return result;
}

void requireAabbAgreement(
    const metalrobo::SurgicalRigidAsset& asset,
    const std::string& label
) {
    const std::array<double, 6> actual = collisionAabb(asset);
    const std::array<double, 6> authored{
        asset.localAabbLowerM[0],
        asset.localAabbLowerM[1],
        asset.localAabbLowerM[2],
        asset.localAabbUpperM[0],
        asset.localAabbUpperM[1],
        asset.localAabbUpperM[2],
    };
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        require(
            actual[axis] <= authored[axis] &&
            authored[axis] - actual[axis] < 8.0e-6,
            label + " lower AABB is not conservative"
        );
        require(
            actual[axis + 3u] >= authored[axis + 3u] &&
            actual[axis + 3u] - authored[axis + 3u] < 8.0e-6,
            label + " upper AABB is not conservative"
        );
    }
}

std::size_t needlePlaneContacts(
    const metalrobo::CurvedSutureNeedleAsset& needle
) {
    std::vector<MRShapeGPU> shapes;
    shapes.reserve(needle.rigid.shapes.size() + 1u);
    shapes.push_back(planeShape(0u));
    shapes.insert(
        shapes.end(),
        needle.rigid.shapes.begin(),
        needle.rigid.shapes.end()
    );
    std::vector<MRBodyStateGPU> bodies{
        bodyState(0.0, 0.0, 0.0, MR_MOTION_STATIC),
        bodyState(
            0.0,
            -needle.rigid.localAabbLowerM[1] - 0.00001,
            0.0,
            MR_MOTION_DYNAMIC
        ),
    };
    metalrobo::PersistentManifoldCache cache;
    const metalrobo::CollisionFrame frame =
        metalrobo::collideCpuReference(
            shapes,
            bodies,
            collisionConfig(),
            cache
        );
    require(frame.succeeded(), "needle/plane collision failed");
    require(!frame.rawContacts.empty(), "needle did not contact plane");
    return frame.rawContacts.size();
}

std::size_t needleJawContacts(
    const metalrobo::CurvedSutureNeedleAsset& needle
) {
    const std::uint32_t graspIndex =
        (needle.metadata.graspShapeBegin +
         needle.metadata.graspShapeEnd) / 2u;
    require(
        graspIndex < needle.rigid.shapes.size(),
        "needle grasp metadata is out of range"
    );
    const MRShapeGPU& grasp = needle.rigid.shapes[graspIndex];
    const double jawRadius = 0.00030;
    const double penetration = 0.00005;
    const double offset =
        static_cast<double>(grasp.dimensions.x) +
        jawRadius - penetration;

    std::vector<MRShapeGPU> shapes = needle.rigid.shapes;
    const std::uint32_t firstJaw =
        static_cast<std::uint32_t>(shapes.size());
    shapes.push_back(jawShape(
        1u,
        910001u,
        grasp.localRotation,
        jawRadius,
        std::max(0.0015, static_cast<double>(grasp.dimensions.y))
    ));
    shapes.push_back(jawShape(
        2u,
        910002u,
        grasp.localRotation,
        jawRadius,
        std::max(0.0015, static_cast<double>(grasp.dimensions.y))
    ));

    std::vector<MRBodyStateGPU> bodies{
        bodyState(0.0, 0.0, 0.0, MR_MOTION_DYNAMIC),
        bodyState(
            grasp.localPosition.x,
            grasp.localPosition.y,
            static_cast<double>(grasp.localPosition.z) + offset,
            MR_MOTION_KINEMATIC
        ),
        bodyState(
            grasp.localPosition.x,
            grasp.localPosition.y,
            static_cast<double>(grasp.localPosition.z) - offset,
            MR_MOTION_KINEMATIC
        ),
    };
    metalrobo::PersistentManifoldCache cache;
    const metalrobo::CollisionFrame frame =
        metalrobo::collideCpuReference(
            shapes,
            bodies,
            collisionConfig(),
            cache
        );
    require(frame.succeeded(), "needle/jaw collision failed");

    std::size_t jawContacts = 0u;
    for (const std::uint32_t pairIndex : frame.rawContactPairIndices) {
        require(pairIndex < frame.pairs.size(), "jaw pair index is invalid");
        const MRCandidatePairGPU& pair = frame.pairs[pairIndex];
        if (pair.colliderA >= firstJaw || pair.colliderB >= firstJaw) {
            ++jawContacts;
        }
    }
    require(jawContacts >= 2u, "both jaws did not engage needle");
    return jawContacts;
}

std::size_t ringPegContacts(
    const metalrobo::SurgicalTrainingRingAsset& ring,
    const metalrobo::SurgicalPegBlockAsset& pegBlock
) {
    std::vector<MRShapeGPU> shapes = ring.rigid.shapes;
    shapes.insert(
        shapes.end(),
        pegBlock.rigid.shapes.begin(),
        pegBlock.rigid.shapes.end()
    );
    require(
        !pegBlock.metadata.pegCenters.empty(),
        "peg metadata is empty"
    );
    const std::array<double, 3>& peg =
        pegBlock.metadata.pegCenters.front();
    const double desiredCenterlineDistance =
        pegBlock.spec.pegRadiusM.value +
        ring.spec.tubeRadiusM.value -
        0.00005;
    const double ringOffset =
        ring.spec.majorRadiusM.value - desiredCenterlineDistance;
    std::vector<MRBodyStateGPU> bodies{
        bodyState(
            peg[0] + ringOffset,
            peg[1],
            peg[2],
            MR_MOTION_DYNAMIC
        ),
        bodyState(0.0, 0.0, 0.0, MR_MOTION_STATIC),
    };

    metalrobo::PersistentManifoldCache cache;
    const metalrobo::CollisionFrame frame =
        metalrobo::collideCpuReference(
            shapes,
            bodies,
            collisionConfig(),
            cache
        );
    require(frame.succeeded(), "ring/peg collision failed");
    require(!frame.rawContacts.empty(), "ring did not contact training peg");
    return frame.rawContacts.size();
}

void requireStandaloneEngineModel(
    const metalrobo::SurgicalRigidAsset& asset
) {
    metalrobo::EngineModel model;
    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount = 1u;
    model.world.shapeCount =
        static_cast<std::uint32_t>(asset.shapes.size());
    model.world.materialCount = 1u;
    model.world.pairCapacity = 2048u;
    model.world.contactCapacity = 4096u;
    model.world.constraintCapacity = 4096u;
    model.world.islandCapacity = 1u;
    model.world.solverType = MR_SOLVER_QUALITY_NEWTON;
    model.world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    model.world.gravityAndTimestep = f4(0.0, -9.81, 0.0, 0.001);
    model.world.solverScales = f4(1.0e-7, 1.0e-12, 1.0, 1.0e-6);
    model.bodies.push_back(asset.body);
    model.shapes = asset.shapes;
    model.materials.push_back(asset.material);
    model.name = "surgical-asset-probe";
    std::string reason;
    require(
        model.valid(&reason),
        "standalone EngineModel rejected surgical asset: " + reason
    );
}

} // namespace

int main() {
    try {
        const metalrobo::SurgicalAssetIds needleIds{
            .bodyIndex = 1u,
            .materialIndex = 0u,
            .slotGenerationBase = 10000u,
            .collisionGroup = 1u,
            .collisionMask = 1u,
            .motionType = MR_MOTION_DYNAMIC,
        };
        const metalrobo::CurvedSutureNeedleAsset needle =
            metalrobo::makeCurvedSutureNeedleAsset(needleIds);
        const metalrobo::CurvedSutureNeedleAsset needleReplay =
            metalrobo::makeCurvedSutureNeedleAsset(needleIds);
        require(
            sameRigidAsset(needle.rigid, needleReplay.rigid),
            "needle factory replay changed bytes"
        );
        require(
            needle.metadata.centerlineRadiusM > 0.011 &&
            needle.metadata.centerlineRadiusM < 0.012,
            "GS-21 centerline radius is inconsistent"
        );
        require(
            needle.metadata.maximumCenterlineErrorM < 0.000015,
            "needle arc discretization is too coarse"
        );
        require(
            std::abs(
                needle.metadata.representedArcLengthM -
                needle.spec.arcLengthM.value
            ) < 0.00002,
            "needle represented arc length drifted"
        );
        require(
            needle.metadata.swageShapeEnd <=
                needle.metadata.graspShapeBegin &&
            needle.metadata.graspShapeBegin <
                needle.metadata.graspShapeEnd &&
            needle.metadata.graspShapeEnd <=
                needle.metadata.tipShapeBegin &&
            needle.metadata.tipShapeBegin <
                needle.metadata.tipShapeEnd,
            "needle semantic zones overlap or are empty"
        );
        require(
            needle.rigid.massKg > 0.00010 &&
            needle.rigid.massKg < 0.00040,
            "needle geometry-derived mass is implausible"
        );
        require(
            needle.rigid.shapes.front().dimensions.x >
                needle.spec.crossSectionRadiusM.value &&
            needle.rigid.shapes.back().dimensions.x <
                0.25 * needle.spec.crossSectionRadiusM.value,
            "needle swage/tip radii do not encode a taper"
        );
        for (std::size_t index = 0u;
             index < needle.rigid.shapes.size();
             ++index) {
            require(
                needle.rigid.shapes[index].slotGeneration ==
                    needleIds.slotGenerationBase + index,
                "needle collision IDs are not stable and contiguous"
            );
        }
        requirePositiveInertia(needle.rigid, "needle");
        requireAabbAgreement(needle.rigid, "needle");

        const metalrobo::SurgicalAssetIds ringIds{
            .bodyIndex = 0u,
            .materialIndex = 0u,
            .slotGenerationBase = 20000u,
            .collisionGroup = 1u,
            .collisionMask = 1u,
            .motionType = MR_MOTION_DYNAMIC,
        };
        const metalrobo::SurgicalTrainingRingAsset ring =
            metalrobo::makeSurgicalTrainingRingAsset(ringIds);
        const metalrobo::SurgicalTrainingRingAsset ringReplay =
            metalrobo::makeSurgicalTrainingRingAsset(ringIds);
        require(
            sameRigidAsset(ring.rigid, ringReplay.rigid),
            "ring factory replay changed bytes"
        );
        require(
            ring.metadata.maximumCenterlineErrorM < 0.000025,
            "training ring discretization is too coarse"
        );
        requirePositiveInertia(ring.rigid, "ring");
        requireAabbAgreement(ring.rigid, "ring");

        const metalrobo::SurgicalAssetIds pegIds{
            .bodyIndex = 1u,
            .materialIndex = 1u,
            .slotGenerationBase = 30000u,
            .collisionGroup = 1u,
            .collisionMask = 1u,
            .motionType = MR_MOTION_STATIC,
        };
        const metalrobo::SurgicalPegBlockAsset pegBlock =
            metalrobo::makeSurgicalPegBlockAsset(pegIds);
        const metalrobo::SurgicalPegBlockAsset pegReplay =
            metalrobo::makeSurgicalPegBlockAsset(pegIds);
        require(
            sameRigidAsset(pegBlock.rigid, pegReplay.rigid),
            "peg-block factory replay changed bytes"
        );
        require(
            pegBlock.metadata.pegCount == 6u &&
            pegBlock.rigid.shapes.size() == 7u,
            "peg-block topology is incorrect"
        );
        require(
            pegBlock.rigid.body.massAndInverseMass.y == 0.0f,
            "static peg block has nonzero inverse mass"
        );
        require(
            pegBlock.rigid.massKg > 0.0 &&
            pegBlock.rigid.body.inertiaRow0.x > 0.0f &&
            pegBlock.rigid.body.inertiaRow1.y > 0.0f &&
            pegBlock.rigid.body.inertiaRow2.z > 0.0f,
            "static peg-block geometry lacks mass properties"
        );
        requireAabbAgreement(pegBlock.rigid, "peg block");

        metalrobo::SurgicalAssetIds dynamicPegIds = pegIds;
        dynamicPegIds.motionType = MR_MOTION_DYNAMIC;
        const metalrobo::SurgicalPegBlockAsset dynamicPegBlock =
            metalrobo::makeSurgicalPegBlockAsset(dynamicPegIds);
        requirePositiveInertia(dynamicPegBlock.rigid, "dynamic peg block");

        const std::size_t planeContacts =
            needlePlaneContacts(needle);

        const metalrobo::SurgicalAssetIds localNeedleIds{
            .bodyIndex = 0u,
            .materialIndex = 0u,
            .slotGenerationBase = 40000u,
            .collisionGroup = 1u,
            .collisionMask = 1u,
            .motionType = MR_MOTION_DYNAMIC,
        };
        const metalrobo::CurvedSutureNeedleAsset localNeedle =
            metalrobo::makeCurvedSutureNeedleAsset(localNeedleIds);
        requireStandaloneEngineModel(localNeedle.rigid);
        const std::size_t jawContacts =
            needleJawContacts(localNeedle);
        const std::size_t pegContacts =
            ringPegContacts(ring, pegBlock);

        bool rejectedInvalid = false;
        try {
            metalrobo::CurvedSutureNeedleSpec invalid;
            invalid.tipTaperLengthM.value = invalid.arcLengthM.value;
            static_cast<void>(
                metalrobo::makeCurvedSutureNeedleAsset({}, invalid)
            );
        } catch (const std::invalid_argument&) {
            rejectedInvalid = true;
        }
        require(rejectedInvalid, "invalid needle spec was accepted");

        std::cout
            << std::setprecision(10)
            << "surgical_assets=rigid_research"
            << " needle_shapes=" << needle.rigid.shapes.size()
            << " needle_mass_g=" << needle.rigid.massKg * 1000.0
            << " needle_arc_mm="
            << needle.spec.arcLengthM.value * 1000.0
            << " needle_arc_error_um="
            << needle.metadata.maximumCenterlineErrorM * 1.0e6
            << " needle_plane_contacts=" << planeContacts
            << " needle_jaw_contacts=" << jawContacts
            << " ring_shapes=" << ring.rigid.shapes.size()
            << " peg_shapes=" << pegBlock.rigid.shapes.size()
            << " ring_peg_contacts=" << pegContacts
            << " source=\""
            << metalrobo::surgicalValueBasisName(
                needle.spec.arcLengthM.basis
            )
            << "\" defaults=\""
            << metalrobo::surgicalValueBasisName(
                needle.spec.crossSectionRadiusM.basis
            )
            << "\" replay=bitwise"
            << " tissue_model=none"
            << " clinical_validation=no"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "surgical_assets status=failed error=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
