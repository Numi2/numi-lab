#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/Collision.hpp"

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
#include <utility>
#include <vector>

#ifndef METALROBO_COLLISION_METALLIB
#define METALROBO_COLLISION_METALLIB ""
#endif

namespace {

constexpr std::uint8_t kOutputSentinel = 0xa5u;
constexpr double kWitnessTolerance = 8.0e-6;

struct Scene {
    std::vector<MRBodyStateGPU> bodies;
    std::vector<MRShapeGPU> shapes;
    std::vector<metalrobo::CollisionPairExclusion> exclusions;
    std::vector<MRCandidatePairGPU> gpuExclusions;
};

struct MetalRun {
    MRSolverStatusGPU status{};
    std::vector<MRCandidatePairGPU> pairs;
    std::vector<MRRawContactGPU> contacts;
    std::vector<std::uint32_t> contactPairIndices;
    bool outputBuffersUntouched = false;
    bool unusedOutputSlotsUntouched = false;
    std::string deviceName;
};

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

MRBodyStateGPU makeBody(
    const float x,
    const float y,
    const float z,
    const std::uint32_t motion
) {
    MRBodyStateGPU result{};
    result.position = f4(x, y, z, 1.0f);
    result.orientation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    result.flagsAndIndices[0] = motion;
    return result;
}

MRShapeGPU makeShape(
    const std::uint32_t body,
    const std::uint32_t type,
    const float radius,
    const std::uint32_t group = 1u,
    const std::uint32_t mask = 1u,
    const mr_float4 localPosition =
        f4(0.0f, 0.0f, 0.0f, 1.0f)
) {
    MRShapeGPU result{};
    result.bodyIndex = body;
    result.shapeType = type;
    result.collisionGroup = group;
    result.collisionMask = mask;
    result.slotGeneration = 1000u + body;
    result.localPosition = localPosition;
    result.localRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    result.dimensions = f4(radius, 0.0f, 0.0f, 0.0f);
    result.contactRestAndBoundingRadius =
        f4(0.02f, 0.0f, radius, 0.0f);
    return result;
}

MRShapeGPU makeCapsuleShape(
    const std::uint32_t body,
    const float radius,
    const float halfLength,
    const std::uint32_t group,
    const std::uint32_t mask,
    const mr_float4 localRotation
) {
    MRShapeGPU result = makeShape(
        body,
        MR_SHAPE_CAPSULE,
        radius,
        group,
        mask
    );
    result.localRotation = localRotation;
    result.dimensions =
        f4(radius, halfLength, 0.0f, 0.0f);
    result.contactRestAndBoundingRadius =
        f4(0.02f, 0.0f, radius + halfLength, 0.0f);
    return result;
}

MRShapeGPU makeBoxShape(
    const std::uint32_t body,
    const float halfX,
    const float halfY,
    const float halfZ,
    const std::uint32_t group,
    const std::uint32_t mask,
    const mr_float4 localRotation
) {
    MRShapeGPU result = makeShape(
        body,
        MR_SHAPE_BOX,
        1.0f,
        group,
        mask
    );
    result.localRotation = localRotation;
    result.dimensions = f4(halfX, halfY, halfZ, 0.0f);
    result.contactRestAndBoundingRadius = f4(
        0.02f,
        0.0f,
        std::sqrt(
            halfX * halfX +
            halfY * halfY +
            halfZ * halfZ
        ),
        0.0f
    );
    return result;
}

MRShapeGPU makeCylinderShape(
    const std::uint32_t body,
    const float radius,
    const float halfLength,
    const mr_float4 localRotation,
    const float contactOffset = 0.02f
) {
    MRShapeGPU result = makeShape(
        body,
        MR_SHAPE_CYLINDER,
        radius
    );
    result.localRotation = localRotation;
    result.dimensions =
        f4(radius, halfLength, 0.0f, 0.0f);
    result.contactRestAndBoundingRadius = f4(
        contactOffset,
        0.0f,
        std::sqrt(
            radius * radius +
            halfLength * halfLength
        ),
        0.0f
    );
    return result;
}

Scene makeScene() {
    Scene scene;
    scene.bodies = {
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_STATIC),
        makeBody(-1.20f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(-0.35f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(0.50f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(2.00f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(2.85f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(4.00f, 0.45f, 0.0f, MR_MOTION_STATIC),
        makeBody(6.00f, 2.00f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(6.90f, 2.90f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(9.00f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(9.50f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(11.00f, 2.00f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(13.00f, 0.20f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(15.00f, 0.00f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(0.00f, 0.00f, 0.0f, MR_MOTION_STATIC),
        makeBody(17.00f, 0.00f, 0.0f, MR_MOTION_DYNAMIC),
    };
    scene.shapes = {
        makeShape(0u, MR_SHAPE_PLANE, 0.0f),
        makeShape(1u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(2u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(3u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(4u, MR_SHAPE_SPHERE, 0.5f, 2u, 2u),
        makeShape(5u, MR_SHAPE_SPHERE, 0.5f, 2u, 2u),
        makeShape(6u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(7u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(8u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(9u, MR_SHAPE_SPHERE, 0.5f, 1u, 0u),
        makeShape(10u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(
            11u,
            MR_SHAPE_SPHERE,
            0.5f,
            1u,
            1u,
            f4(-0.2f, 0.0f, 0.0f, 1.0f)
        ),
        makeShape(
            11u,
            MR_SHAPE_SPHERE,
            0.5f,
            1u,
            1u,
            f4(0.2f, 0.0f, 0.0f, 1.0f)
        ),
        makeCapsuleShape(
            12u,
            0.25f,
            0.50f,
            4u,
            4u,
            f4(
                0.0f,
                0.0f,
                0.7071067811865476f,
                0.7071067811865476f
            )
        ),
        makeBoxShape(
            13u,
            0.35f,
            0.01f,
            0.25f,
            4u,
            4u,
            f4(
                0.0f,
                0.25881904510252074f,
                0.0f,
                0.9659258262890683f
            )
        ),
        makeShape(
            14u,
            MR_SHAPE_PLANE,
            0.0f,
            4u,
            4u
        ),
        makeShape(
            15u,
            MR_SHAPE_CONVEX,
            0.25f,
            4u,
            4u
        ),
    };
    scene.shapes.back().flags =
        MR_SHAPE_FLAG_SIMULATION_DISABLED;
    scene.exclusions = {
        {2u, 1u},
        {1u, 2u},
        {12u, 12u},
    };
    scene.gpuExclusions.reserve(scene.exclusions.size());
    for (const auto exclusion : scene.exclusions) {
        MRCandidatePairGPU gpu{};
        gpu.colliderA = exclusion.colliderA;
        gpu.colliderB = exclusion.colliderB;
        scene.gpuExclusions.push_back(gpu);
    }
    return scene;
}

Scene makeCapsulePairScene() {
    constexpr float quarterTurn = 0.7071067811865475f;
    constexpr float boundaryOffset =
        0.3676884472372877f;
    Scene scene;

    auto sphere = [](const float contactOffset = 0.0f) {
        MRShapeGPU result = makeShape(
            0u,
            MR_SHAPE_SPHERE,
            0.40f
        );
        result.contactRestAndBoundingRadius.x = contactOffset;
        return result;
    };
    auto capsule = [](const float contactOffset = 0.0f) {
        MRShapeGPU result = makeCapsuleShape(
            0u,
            0.25f,
            1.0f,
            1u,
            1u,
            f4(0.0f, 0.0f, 0.0f, 1.0f)
        );
        result.contactRestAndBoundingRadius.x = contactOffset;
        return result;
    };
    auto appendPair = [&](
        const MRBodyStateGPU bodyA,
        MRShapeGPU shapeA,
        const MRBodyStateGPU bodyB,
        MRShapeGPU shapeB,
        const std::uint32_t group
    ) {
        const std::uint32_t bodyAIndex =
            static_cast<std::uint32_t>(scene.bodies.size());
        shapeA.bodyIndex = bodyAIndex;
        shapeB.bodyIndex = bodyAIndex + 1u;
        shapeA.collisionGroup = group;
        shapeA.collisionMask = group;
        shapeB.collisionGroup = group;
        shapeB.collisionMask = group;
        shapeA.slotGeneration = 500u + bodyAIndex;
        shapeB.slotGeneration = 501u + bodyAIndex;
        scene.bodies.push_back(bodyA);
        scene.bodies.push_back(bodyB);
        scene.shapes.push_back(shapeA);
        scene.shapes.push_back(shapeB);
    };

    appendPair(
        makeBody(0.649f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        1u << 0u
    );
    appendPair(
        makeBody(5.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        makeBody(5.649f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        1u << 1u
    );
    appendPair(
        makeBody(10.649f, 1.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        makeBody(10.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        1u << 2u
    );
    appendPair(
        makeBody(15.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        makeBody(15.649f, 1.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        1u << 3u
    );

    MRBodyStateGPU skewBodyB =
        makeBody(20.0f, 0.0f, 0.5f, MR_MOTION_DYNAMIC);
    skewBodyB.orientation =
        f4(0.0f, 0.0f, quarterTurn, quarterTurn);
    appendPair(
        makeBody(20.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        skewBodyB,
        capsule(),
        1u << 4u
    );
    appendPair(
        makeBody(25.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        makeBody(25.4f, 0.5f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        1u << 5u
    );
    appendPair(
        makeBody(30.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        makeBody(30.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        1u << 6u
    );
    appendPair(
        makeBody(35.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(0.01f),
        makeBody(
            35.0f + boundaryOffset,
            0.0f,
            boundaryOffset,
            MR_MOTION_DYNAMIC
        ),
        capsule(0.01f),
        1u << 7u
    );
    appendPair(
        makeBody(40.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(0.01f),
        makeBody(40.371f, 0.0f, 0.371f, MR_MOTION_DYNAMIC),
        capsule(0.01f),
        1u << 8u
    );
    return scene;
}

Scene makeSphereBoxScene() {
    constexpr float sineEighthTurn = 0.3826834323650898f;
    constexpr float cosineEighthTurn = 0.9238795325112867f;
    Scene scene;

    auto sphere = [] {
        MRShapeGPU result = makeShape(
            0u,
            MR_SHAPE_SPHERE,
            0.30f
        );
        result.contactRestAndBoundingRadius.x = 0.0f;
        return result;
    };
    auto box = [](
        const float halfX = 0.50f,
        const float halfY = 0.50f,
        const float halfZ = 0.50f
    ) {
        MRShapeGPU result = makeBoxShape(
            0u,
            halfX,
            halfY,
            halfZ,
            1u,
            1u,
            f4(0.0f, 0.0f, 0.0f, 1.0f)
        );
        result.contactRestAndBoundingRadius.x = 0.0f;
        return result;
    };
    auto appendPair = [&](
        const MRBodyStateGPU bodyA,
        MRShapeGPU shapeA,
        const MRBodyStateGPU bodyB,
        MRShapeGPU shapeB,
        const std::uint32_t group
    ) {
        const std::uint32_t bodyAIndex =
            static_cast<std::uint32_t>(scene.bodies.size());
        shapeA.bodyIndex = bodyAIndex;
        shapeB.bodyIndex = bodyAIndex + 1u;
        shapeA.collisionGroup = group;
        shapeA.collisionMask = group;
        shapeB.collisionGroup = group;
        shapeB.collisionMask = group;
        shapeA.slotGeneration = 700u + bodyAIndex;
        shapeB.slotGeneration = 701u + bodyAIndex;
        scene.bodies.push_back(bodyA);
        scene.bodies.push_back(bodyB);
        scene.shapes.push_back(shapeA);
        scene.shapes.push_back(shapeB);
    };

    appendPair(
        makeBody(0.799f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        1u << 0u
    );
    appendPair(
        makeBody(5.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        makeBody(5.799f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        1u << 1u
    );
    appendPair(
        makeBody(10.70f, 0.70f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        makeBody(10.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        1u << 2u
    );
    appendPair(
        makeBody(15.67f, 0.67f, 0.67f, MR_MOTION_DYNAMIC),
        sphere(),
        makeBody(15.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        1u << 3u
    );
    MRBodyStateGPU rotatedBox =
        makeBody(20.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC);
    rotatedBox.orientation = f4(
        0.0f,
        0.0f,
        sineEighthTurn,
        cosineEighthTurn
    );
    appendPair(
        makeBody(
            20.5649783103f,
            0.5649783103f,
            0.0f,
            MR_MOTION_DYNAMIC
        ),
        sphere(),
        rotatedBox,
        box(),
        1u << 4u
    );
    appendPair(
        makeBody(25.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        makeBody(25.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(0.50f, 0.60f, 0.70f),
        1u << 5u
    );
    return scene;
}

Scene makeCapsuleBoxBoxScene() {
    constexpr float sineEighthTurn = 0.3826834323650898f;
    constexpr float cosineEighthTurn = 0.9238795325112867f;
    Scene scene;

    auto capsule = [] {
        MRShapeGPU result = makeCapsuleShape(
            0u,
            0.20f,
            0.60f,
            1u,
            1u,
            f4(0.0f, 0.0f, 0.0f, 1.0f)
        );
        result.contactRestAndBoundingRadius.x = 0.0f;
        return result;
    };
    auto box = [] {
        MRShapeGPU result = makeBoxShape(
            0u,
            0.50f,
            0.50f,
            0.50f,
            1u,
            1u,
            f4(0.0f, 0.0f, 0.0f, 1.0f)
        );
        result.contactRestAndBoundingRadius.x = 0.0f;
        return result;
    };
    auto appendPair = [&](
        const MRBodyStateGPU bodyA,
        MRShapeGPU shapeA,
        const MRBodyStateGPU bodyB,
        MRShapeGPU shapeB,
        const std::uint32_t group
    ) {
        const std::uint32_t bodyAIndex =
            static_cast<std::uint32_t>(scene.bodies.size());
        shapeA.bodyIndex = bodyAIndex;
        shapeB.bodyIndex = bodyAIndex + 1u;
        shapeA.collisionGroup = group;
        shapeA.collisionMask = group;
        shapeB.collisionGroup = group;
        shapeB.collisionMask = group;
        shapeA.slotGeneration = 800u + bodyAIndex;
        shapeB.slotGeneration = 801u + bodyAIndex;
        scene.bodies.push_back(bodyA);
        scene.bodies.push_back(bodyB);
        scene.shapes.push_back(shapeA);
        scene.shapes.push_back(shapeB);
    };

    appendPair(
        makeBody(0.68f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        1u << 0u
    );
    appendPair(
        makeBody(5.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        makeBody(5.68f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        1u << 1u
    );
    appendPair(
        makeBody(10.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        makeBody(10.8f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        1u << 2u
    );
    MRShapeGPU rotatedBox = box();
    rotatedBox.localRotation = f4(
        0.0f,
        sineEighthTurn,
        0.0f,
        cosineEighthTurn
    );
    appendPair(
        makeBody(15.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        makeBody(15.65f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        rotatedBox,
        1u << 3u
    );
    return scene;
}

Scene makeCylinderScene() {
    constexpr float sineQuarterTurn =
        0.7071067811865475244f;
    constexpr float sineEighthTurn =
        0.3826834323650897717f;
    constexpr float cosineEighthTurn =
        0.9238795325112867561f;
    constexpr float radius = 0.30f;
    constexpr float halfLength = 0.60f;
    constexpr float tiltedVerticalExtent =
        (radius + halfLength) * sineQuarterTurn;

    Scene scene;
    scene.bodies = {
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_STATIC),
        makeBody(-3.0f, 0.55f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(-1.0f, 0.55f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(1.0f, 0.25f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(
            3.0f,
            tiltedVerticalExtent - 0.05f,
            0.0f,
            MR_MOTION_DYNAMIC
        ),
    };
    // Exercise transform composition, not only shape-local rotation.
    scene.bodies[4].orientation = f4(
        0.0f,
        0.0f,
        -sineEighthTurn,
        cosineEighthTurn
    );
    scene.shapes = {
        makeShape(0u, MR_SHAPE_PLANE, 0.0f),
        makeCylinderShape(
            1u,
            radius,
            halfLength,
            f4(0.0f, 0.0f, 0.0f, 1.0f)
        ),
        makeCylinderShape(
            2u,
            radius,
            halfLength,
            f4(1.0f, 0.0f, 0.0f, 0.0f)
        ),
        makeCylinderShape(
            3u,
            radius,
            halfLength,
            f4(
                0.0f,
                0.0f,
                -sineQuarterTurn,
                sineQuarterTurn
            )
        ),
        makeCylinderShape(
            4u,
            radius,
            halfLength,
            f4(0.0f, 0.0f, 0.0f, 1.0f)
        ),
    };
    return scene;
}

Scene makeCylinderFirstScene() {
    Scene scene;
    scene.bodies = {
        makeBody(-3.0f, 0.55f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_STATIC),
    };
    scene.shapes = {
        makeCylinderShape(
            0u,
            0.30f,
            0.60f,
            f4(0.0f, 0.0f, 0.0f, 1.0f)
        ),
        makeShape(1u, MR_SHAPE_PLANE, 0.0f),
    };
    return scene;
}

Scene makeCylinderOrientationSweepScene() {
    constexpr std::uint32_t cylinderCount = 32u;
    constexpr double pi = 3.14159265358979323846;
    Scene scene;
    scene.bodies.reserve(cylinderCount + 1u);
    scene.shapes.reserve(cylinderCount + 1u);
    scene.bodies.push_back(
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_STATIC)
    );
    scene.shapes.push_back(
        makeShape(0u, MR_SHAPE_PLANE, 0.0f)
    );
    for (std::uint32_t index = 0u;
         index < cylinderCount;
         ++index) {
        const double angle =
            pi * static_cast<double>(index) /
            static_cast<double>(cylinderCount - 1u);
        const double bodyAngle =
            index % 3u == 0u ? 0.4 * angle : angle;
        const double localAngle =
            index % 3u == 0u ? 0.6 * angle : 0.0;
        const std::uint32_t body =
            static_cast<std::uint32_t>(scene.bodies.size());
        scene.bodies.push_back(makeBody(
            2.0f * static_cast<float>(index % 8u),
            0.0f,
            2.0f * static_cast<float>(index / 8u),
            MR_MOTION_DYNAMIC
        ));
        scene.bodies.back().orientation = f4(
            0.0f,
            0.0f,
            -static_cast<float>(std::sin(0.5 * bodyAngle)),
            static_cast<float>(std::cos(0.5 * bodyAngle))
        );
        const float radius =
            0.10f + 0.01f * static_cast<float>(index % 7u);
        const float halfLength =
            0.15f + 0.02f * static_cast<float>(index % 5u);
        scene.shapes.push_back(makeCylinderShape(
            body,
            radius,
            halfLength,
            f4(
                0.0f,
                0.0f,
                -static_cast<float>(
                    std::sin(0.5 * localAngle)
                ),
                static_cast<float>(
                    std::cos(0.5 * localAngle)
                )
            ),
            0.005f +
                0.001f * static_cast<float>(index % 4u)
        ));
    }
    return scene;
}

Scene makeCylinderNearCapRegressionScene() {
    // A tiny arbitrary-azimuth tilt stays inside the cap-manifold branch, but
    // its exact support direction lies between all four fixed ring axes.
    // The cap center is separated while the true rim support penetrates. All
    // four fixed ring samples remain separated, so this is a direct regression
    // for conservative support rather than a generous contact-distance test.
    constexpr double angle = 9.0e-7;
    constexpr double inverseRootTwo =
        0.7071067811865475244;
    constexpr double radius = 1.0;
    constexpr double halfLength = 0.5;
    constexpr double capCenterSeparation = 7.5e-7;
    const double halfAngle = 0.5 * angle;

    Scene scene;
    scene.bodies = {
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_STATIC),
        makeBody(
            0.0f,
            static_cast<float>(
                halfLength * std::cos(angle) +
                capCenterSeparation
            ),
            0.0f,
            MR_MOTION_DYNAMIC
        ),
    };
    scene.shapes = {
        makeShape(0u, MR_SHAPE_PLANE, 0.0f),
        makeCylinderShape(
            1u,
            static_cast<float>(radius),
            static_cast<float>(halfLength),
            f4(
                static_cast<float>(
                    -inverseRootTwo * std::sin(halfAngle)
                ),
                0.0f,
                static_cast<float>(
                    inverseRootTwo * std::sin(halfAngle)
                ),
                static_cast<float>(std::cos(halfAngle))
            ),
            0.0f
        ),
    };
    scene.shapes[0].contactRestAndBoundingRadius.x = 0.0f;
    return scene;
}

Scene makeCylinderNearCapSubEpsilonRegressionScene() {
    // This tilt has radialSquared below the engine's general geometry epsilon.
    // Radius amplification still makes its exact support physically material:
    // the true rim penetrates by 1e-5 while every fixed cardinal ring sample
    // is separated. Direction normalization must therefore use a zero test,
    // not a geometry-length threshold.
    constexpr double angle = 8.0e-8;
    constexpr double inverseRootTwo =
        0.7071067811865475244;
    constexpr double radius = 1000.0;
    constexpr double halfLength = 0.5;
    constexpr double penetration = 1.0e-5;
    const double halfAngle = 0.5 * angle;

    Scene scene;
    scene.bodies = {
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_STATIC),
        makeBody(
            0.0f,
            static_cast<float>(
                halfLength * std::cos(angle) +
                radius * std::sin(angle) -
                penetration
            ),
            0.0f,
            MR_MOTION_DYNAMIC
        ),
    };
    scene.shapes = {
        makeShape(0u, MR_SHAPE_PLANE, 0.0f),
        makeCylinderShape(
            1u,
            static_cast<float>(radius),
            static_cast<float>(halfLength),
            f4(
                static_cast<float>(
                    -inverseRootTwo * std::sin(halfAngle)
                ),
                0.0f,
                static_cast<float>(
                    inverseRootTwo * std::sin(halfAngle)
                ),
                static_cast<float>(std::cos(halfAngle))
            ),
            0.0f
        ),
    };
    scene.shapes[0].contactRestAndBoundingRadius.x = 0.0f;
    return scene;
}

std::string nsString(NSString* value) {
    if (value == nil || value.UTF8String == nullptr) {
        return {};
    }
    return std::string{value.UTF8String};
}

std::string describeError(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    const std::string localized =
        nsString(error.localizedDescription);
    return localized.empty()
        ? nsString(error.description)
        : localized;
}

template <typename T>
id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const T* data,
    const std::size_t count,
    NSString* label
) {
    require(
        count <= std::numeric_limits<NSUInteger>::max() / sizeof(T),
        "Metal buffer byte count overflow"
    );
    require(
        count == 0u || data != nullptr,
        "non-empty Metal buffer has null source data"
    );
    id<MTLBuffer> buffer = nil;
    if (count == 0u) {
        buffer = [device
            newBufferWithLength:sizeof(T)
                        options:MTLResourceStorageModeShared];
        if (buffer != nil) {
            std::memset(buffer.contents, 0, sizeof(T));
        }
    } else {
        const NSUInteger byteCount =
            static_cast<NSUInteger>(count * sizeof(T));
        buffer = [device
            newBufferWithBytes:data
                        length:byteCount
                       options:MTLResourceStorageModeShared];
    }
    require(
        buffer != nil,
        "failed to allocate Metal buffer '" + nsString(label) + "'"
    );
    buffer.label = label;
    return buffer;
}

template <typename T>
std::vector<T> sentinelVector(const std::size_t count) {
    std::vector<T> result(std::max<std::size_t>(count, 1u));
    std::memset(
        result.data(),
        kOutputSentinel,
        result.size() * sizeof(T)
    );
    return result;
}

template <typename T>
bool sameBytes(
    const std::span<const T> left,
    const std::span<const T> right
) {
    return left.size() == right.size() &&
        std::memcmp(
            left.data(),
            right.data(),
            left.size_bytes()
        ) == 0;
}

template <typename T>
std::vector<T> copyBuffer(
    id<MTLBuffer> buffer,
    const std::size_t count
) {
    std::vector<T> result(count);
    if (count > 0u) {
        std::memcpy(
            result.data(),
            buffer.contents,
            count * sizeof(T)
        );
    }
    return result;
}

MetalRun collideOnMetal(
    const Scene& scene,
    const std::uint32_t environment,
    const std::uint32_t pairCapacity,
    const std::uint32_t contactCapacity
) {
    @autoreleasepool {
        const std::string metallibPath =
            METALROBO_COLLISION_METALLIB;
        require(
            !metallibPath.empty(),
            "METALROBO_COLLISION_METALLIB is empty"
        );

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal-capable device is available");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "failed to create Metal command queue");

        NSString* path =
            [NSString stringWithUTF8String:metallibPath.c_str()];
        require(path != nil, "metallib path is not valid UTF-8");
        NSError* error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:[NSURL fileURLWithPath:path]
                        error:&error];
        require(
            library != nil,
            "failed to load collision metallib: " +
                describeError(error)
        );
        id<MTLFunction> function =
            [library newFunctionWithName:@"mr_collide_baseline"];
        require(
            function != nil,
            "metallib does not contain mr_collide_baseline"
        );
        error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function
                                                   error:&error];
        require(
            pipeline != nil,
            "failed to create collision pipeline: " +
                describeError(error)
        );

        const auto pairSeed =
            sentinelVector<MRCandidatePairGPU>(pairCapacity);
        const auto contactSeed =
            sentinelVector<MRRawContactGPU>(contactCapacity);
        const auto contactPairSeed =
            sentinelVector<std::uint32_t>(contactCapacity);
        MRSolverStatusGPU statusSeed{};
        std::memset(
            &statusSeed,
            kOutputSentinel,
            sizeof(statusSeed)
        );

        id<MTLBuffer> shapes = makeBuffer(
            device,
            scene.shapes.data(),
            scene.shapes.size(),
            @"collision shapes"
        );
        id<MTLBuffer> bodies = makeBuffer(
            device,
            scene.bodies.data(),
            scene.bodies.size(),
            @"collision bodies"
        );
        id<MTLBuffer> exclusions = makeBuffer(
            device,
            scene.gpuExclusions.data(),
            scene.gpuExclusions.size(),
            @"collision exclusions"
        );
        id<MTLBuffer> pairs = makeBuffer(
            device,
            pairSeed.data(),
            pairSeed.size(),
            @"candidate pairs"
        );
        id<MTLBuffer> contacts = makeBuffer(
            device,
            contactSeed.data(),
            contactSeed.size(),
            @"raw contacts"
        );
        id<MTLBuffer> contactPairIndices = makeBuffer(
            device,
            contactPairSeed.data(),
            contactPairSeed.size(),
            @"contact pair indices"
        );
        id<MTLBuffer> status = makeBuffer(
            device,
            &statusSeed,
            1u,
            @"collision status"
        );

        const std::uint32_t bodyCount =
            static_cast<std::uint32_t>(scene.bodies.size());
        const std::uint32_t shapeCount =
            static_cast<std::uint32_t>(scene.shapes.size());
        const std::uint32_t exclusionCount =
            static_cast<std::uint32_t>(
                scene.gpuExclusions.size()
            );
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        require(
            commandBuffer != nil,
            "failed to create Metal command buffer"
        );
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        require(
            encoder != nil,
            "failed to create Metal compute encoder"
        );
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:shapes offset:0 atIndex:0];
        [encoder setBuffer:bodies offset:0 atIndex:1];
        [encoder setBuffer:exclusions offset:0 atIndex:2];
        [encoder setBuffer:pairs offset:0 atIndex:3];
        [encoder setBuffer:contacts offset:0 atIndex:4];
        [encoder setBuffer:contactPairIndices offset:0 atIndex:5];
        [encoder setBuffer:status offset:0 atIndex:6];
        [encoder setBytes:&bodyCount
                   length:sizeof(bodyCount)
                  atIndex:7];
        [encoder setBytes:&shapeCount
                   length:sizeof(shapeCount)
                  atIndex:8];
        [encoder setBytes:&environment
                   length:sizeof(environment)
                  atIndex:9];
        [encoder setBytes:&exclusionCount
                   length:sizeof(exclusionCount)
                  atIndex:10];
        [encoder setBytes:&pairCapacity
                   length:sizeof(pairCapacity)
                  atIndex:11];
        [encoder setBytes:&contactCapacity
                   length:sizeof(contactCapacity)
                  atIndex:12];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        require(
            commandBuffer.status ==
                MTLCommandBufferStatusCompleted,
            "collision dispatch failed: " +
                describeError(commandBuffer.error)
        );

        MetalRun result;
        std::memcpy(
            &result.status,
            status.contents,
            sizeof(result.status)
        );
        const auto allPairs =
            copyBuffer<MRCandidatePairGPU>(
                pairs,
                pairSeed.size()
            );
        const auto allContacts =
            copyBuffer<MRRawContactGPU>(
                contacts,
                contactSeed.size()
            );
        const auto allContactPairIndices =
            copyBuffer<std::uint32_t>(
                contactPairIndices,
                contactPairSeed.size()
            );
        result.outputBuffersUntouched =
            sameBytes<MRCandidatePairGPU>(allPairs, pairSeed) &&
            sameBytes<MRRawContactGPU>(
                allContacts,
                contactSeed
            ) &&
            sameBytes<std::uint32_t>(
                allContactPairIndices,
                contactPairSeed
            );

        const bool succeeded =
            result.status.code == MR_STEP_SUCCESS;
        if (succeeded) {
            require(
                result.status.requiredPairs <= pairCapacity &&
                    result.status.requiredContacts <=
                        contactCapacity,
                "successful kernel exceeded declared capacity"
            );
            result.pairs.assign(
                allPairs.begin(),
                allPairs.begin() +
                    result.status.requiredPairs
            );
            result.contacts.assign(
                allContacts.begin(),
                allContacts.begin() +
                    result.status.requiredContacts
            );
            result.contactPairIndices.assign(
                allContactPairIndices.begin(),
                allContactPairIndices.begin() +
                    result.status.requiredContacts
            );

            const auto pairTail = std::span{
                allPairs
            }.subspan(result.status.requiredPairs);
            const auto pairSeedTail = std::span{
                pairSeed
            }.subspan(result.status.requiredPairs);
            const auto contactTail = std::span{
                allContacts
            }.subspan(result.status.requiredContacts);
            const auto contactSeedTail = std::span{
                contactSeed
            }.subspan(result.status.requiredContacts);
            const auto indexTail = std::span{
                allContactPairIndices
            }.subspan(result.status.requiredContacts);
            const auto indexSeedTail = std::span{
                contactPairSeed
            }.subspan(result.status.requiredContacts);
            result.unusedOutputSlotsUntouched =
                sameBytes<MRCandidatePairGPU>(
                    pairTail,
                    pairSeedTail
                ) &&
                sameBytes<MRRawContactGPU>(
                    contactTail,
                    contactSeedTail
                ) &&
                sameBytes<std::uint32_t>(
                    indexTail,
                    indexSeedTail
                );
        }
        result.deviceName = nsString(device.name);
        return result;
    }
}

bool samePair(
    const MRCandidatePairGPU& left,
    const MRCandidatePairGPU& right
) {
    return left.environment == right.environment &&
        left.colliderA == right.colliderA &&
        left.colliderB == right.colliderB &&
        left.flags == right.flags;
}

double componentError(
    const mr_float4 left,
    const mr_float4 right
) {
    return std::max({
        std::abs(static_cast<double>(left.x) - right.x),
        std::abs(static_cast<double>(left.y) - right.y),
        std::abs(static_cast<double>(left.z) - right.z),
        std::abs(static_cast<double>(left.w) - right.w),
    });
}

double compareWithCpu(
    const metalrobo::CollisionFrame& cpu,
    const MetalRun& gpu
) {
    require(
        cpu.pairs.size() == gpu.pairs.size(),
        "CPU/Metal candidate pair counts differ"
    );
    require(
        cpu.rawContacts.size() == gpu.contacts.size(),
        "CPU/Metal raw contact counts differ"
    );
    require(
        cpu.rawContactPairIndices ==
            gpu.contactPairIndices,
        "CPU/Metal raw-contact pair indices differ"
    );

    for (std::size_t index = 0u;
         index < cpu.pairs.size();
         ++index) {
        require(
            samePair(cpu.pairs[index], gpu.pairs[index]),
            "CPU/Metal sorted candidate pairs differ"
        );
        if (index > 0u) {
            const auto& previous = gpu.pairs[index - 1u];
            const auto& current = gpu.pairs[index];
            require(
                std::pair{
                    previous.colliderA,
                    previous.colliderB
                } <
                    std::pair{
                        current.colliderA,
                        current.colliderB
                    },
                "Metal candidate pairs are not strictly sorted"
            );
        }
    }

    double maximumError = 0.0;
    for (std::size_t index = 0u;
         index < cpu.rawContacts.size();
         ++index) {
        const MRRawContactGPU& cpuContact =
            cpu.rawContacts[index];
        const MRRawContactGPU& gpuContact =
            gpu.contacts[index];
        const double nx = gpuContact.normalAndSeparation.x;
        const double ny = gpuContact.normalAndSeparation.y;
        const double nz = gpuContact.normalAndSeparation.z;
        const double normalLength =
            std::sqrt(nx * nx + ny * ny + nz * nz);
        const double dx =
            gpuContact.pointBWorld.x -
            gpuContact.pointAWorld.x;
        const double dy =
            gpuContact.pointBWorld.y -
            gpuContact.pointAWorld.y;
        const double dz =
            gpuContact.pointBWorld.z -
            gpuContact.pointAWorld.z;
        const double projected =
            dx * nx + dy * ny + dz * nz;
        require(
            std::isfinite(normalLength) &&
                std::abs(normalLength - 1.0) <=
                    kWitnessTolerance &&
                std::abs(
                    projected -
                    gpuContact.normalAndSeparation.w
                ) <= kWitnessTolerance,
            "Metal contact violates unit-normal or signed-witness convention"
        );
        for (std::size_t field = 0u; field < 4u; ++field) {
            require(
                cpuContact.featureAndFlags[field] ==
                    gpuContact.featureAndFlags[field],
                "CPU/Metal stable feature keys differ"
            );
        }
        maximumError = std::max({
            maximumError,
            componentError(
                cpuContact.normalAndSeparation,
                gpuContact.normalAndSeparation
            ),
            componentError(
                cpuContact.pointAWorld,
                gpuContact.pointAWorld
            ),
            componentError(
                cpuContact.pointBWorld,
                gpuContact.pointBWorld
            ),
        });
    }
    require(
        maximumError <= kWitnessTolerance,
        "CPU/Metal contact witness parity exceeded tolerance"
    );
    return maximumError;
}

bool containsPair(
    const std::span<const MRCandidatePairGPU> pairs,
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    return std::ranges::any_of(
        pairs,
        [=](const MRCandidatePairGPU& pair) {
            return pair.colliderA == colliderA &&
                pair.colliderB == colliderB;
        }
    );
}

bool pairHasContact(
    const MetalRun& run,
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    for (const std::uint32_t pairIndex :
         run.contactPairIndices) {
        require(
            pairIndex < run.pairs.size(),
            "Metal contact references an invalid pair"
        );
        if (run.pairs[pairIndex].colliderA == colliderA &&
            run.pairs[pairIndex].colliderB == colliderB) {
            return true;
        }
    }
    return false;
}

std::vector<MRRawContactGPU> contactsForPair(
    const MetalRun& run,
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    std::vector<MRRawContactGPU> result;
    for (std::size_t contactIndex = 0u;
         contactIndex < run.contactPairIndices.size();
         ++contactIndex) {
        const std::uint32_t pairIndex =
            run.contactPairIndices[contactIndex];
        require(
            pairIndex < run.pairs.size(),
            "Metal contact references an invalid pair"
        );
        const MRCandidatePairGPU& pair = run.pairs[pairIndex];
        if (pair.colliderA == colliderA &&
            pair.colliderB == colliderB) {
            result.push_back(run.contacts[contactIndex]);
        }
    }
    return result;
}

std::uint32_t manifoldPointCount(
    const metalrobo::CollisionFrame& frame,
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    for (const MRManifoldHeaderGPU& header :
         frame.manifoldHeaders) {
        if (header.pairAndCount[1] == colliderA &&
            header.pairAndCount[2] == colliderB) {
            return header.pairAndCount[3];
        }
    }
    return 0u;
}

bool shapeAppearsInPair(
    const std::span<const MRCandidatePairGPU> pairs,
    const std::uint32_t collider
) {
    return std::ranges::any_of(
        pairs,
        [=](const MRCandidatePairGPU& pair) {
            return pair.colliderA == collider ||
                pair.colliderB == collider;
        }
    );
}

std::uint32_t expectedFeature(
    const std::uint32_t shapeType,
    const std::uint32_t localFeature
) {
    return ((shapeType & 0x0fu) << 28u) |
        (localFeature & 0x0fffffffu);
}

bool sameSuccessfulRun(
    const MetalRun& left,
    const MetalRun& right
) {
    return
        left.status.code == right.status.code &&
        left.status.requiredPairs ==
            right.status.requiredPairs &&
        left.status.requiredContacts ==
            right.status.requiredContacts &&
        sameBytes<MRCandidatePairGPU>(
            left.pairs,
            right.pairs
        ) &&
        sameBytes<MRRawContactGPU>(
            left.contacts,
            right.contacts
        ) &&
        left.contactPairIndices ==
            right.contactPairIndices;
}

double verifyCapsulePairPrimitives(
    const std::uint32_t environment
) {
    const Scene scene = makeCapsulePairScene();
    metalrobo::CollisionConfig config;
    config.environment = environment;
    config.capacities = {
        .pairCapacity = 16u,
        .rawContactCapacity = 16u,
        .manifoldCapacity = 16u,
    };
    metalrobo::PersistentManifoldCache cache;
    const auto cpu = metalrobo::collideCpuReference(
        scene.shapes,
        scene.bodies,
        config,
        cache
    );
    const MetalRun gpu = collideOnMetal(
        scene,
        environment,
        16u,
        16u
    );
    require(
        cpu.succeeded() &&
            cpu.pairs.size() == 9u &&
            cpu.rawContacts.size() == 8u &&
            gpu.status.code == MR_STEP_SUCCESS &&
            gpu.status.requiredPairs == 9u &&
            gpu.status.requiredContacts == 8u &&
            gpu.unusedOutputSlotsUntouched,
        "capsule pair CPU/Metal topology is wrong"
    );
    const double witnessError = compareWithCpu(cpu, gpu);

    for (std::uint32_t localPair = 0u;
         localPair < 4u;
         ++localPair) {
        const std::uint32_t colliderA = 2u * localPair;
        const std::uint32_t colliderB = colliderA + 1u;
        const auto contacts =
            contactsForPair(gpu, colliderA, colliderB);
        const bool sphereFirst = localPair % 2u == 0u;
        const std::uint32_t capsuleLocalFeature =
            localPair < 2u ? 2u : 1u;
        require(
            containsPair(gpu.pairs, colliderA, colliderB) &&
                gpu.pairs[localPair].flags ==
                    metalrobo::collisionPairSphereCapsule &&
                contacts.size() == 1u &&
                contacts[0].featureAndFlags[0] ==
                    expectedFeature(
                        sphereFirst
                            ? MR_SHAPE_SPHERE
                            : MR_SHAPE_CAPSULE,
                        sphereFirst
                            ? 0u
                            : capsuleLocalFeature
                    ) &&
                contacts[0].featureAndFlags[1] ==
                    expectedFeature(
                        sphereFirst
                            ? MR_SHAPE_CAPSULE
                            : MR_SHAPE_SPHERE,
                        sphereFirst
                            ? capsuleLocalFeature
                            : 0u
                    ),
            "Metal sphere/capsule input order or feature ID is wrong"
        );
    }
    const auto sphereSegment =
        contactsForPair(gpu, 0u, 1u).front();
    const auto capsuleSegment =
        contactsForPair(gpu, 2u, 3u).front();
    const auto sphereEndpoint =
        contactsForPair(gpu, 4u, 5u).front();
    const auto capsuleEndpoint =
        contactsForPair(gpu, 6u, 7u).front();
    require(
        std::abs(
            sphereSegment.normalAndSeparation.x +
                capsuleSegment.normalAndSeparation.x
        ) <= 3.0e-6f &&
            std::abs(
                sphereSegment.normalAndSeparation.w -
                    capsuleSegment.normalAndSeparation.w
            ) <= 3.0e-6f &&
            std::abs(
                sphereEndpoint.normalAndSeparation.x +
                    capsuleEndpoint.normalAndSeparation.x
            ) <= 3.0e-6f &&
            std::abs(
                sphereEndpoint.normalAndSeparation.w -
                    capsuleEndpoint.normalAndSeparation.w
            ) <= 3.0e-6f,
        "Metal sphere/capsule swapped-order reciprocity failed"
    );

    constexpr std::array<std::uint32_t, 5> featureA{
        2u, 2u, 0u, 0u, 0u,
    };
    constexpr std::array<std::uint32_t, 5> featureB{
        2u, 0u, 0u, 0u, 0u,
    };
    for (std::size_t localPair = 0u;
         localPair < featureA.size();
         ++localPair) {
        const std::uint32_t colliderA =
            static_cast<std::uint32_t>(8u + 2u * localPair);
        const std::uint32_t colliderB = colliderA + 1u;
        const auto contacts =
            contactsForPair(gpu, colliderA, colliderB);
        const bool outsideOffset = localPair == 4u;
        require(
            containsPair(gpu.pairs, colliderA, colliderB) &&
                gpu.pairs[4u + localPair].flags ==
                    metalrobo::collisionPairCapsuleCapsule &&
                contacts.size() == (outsideOffset ? 0u : 1u),
            "Metal capsule/capsule adversarial topology is wrong"
        );
        if (!outsideOffset) {
            require(
                contacts[0].featureAndFlags[0] ==
                        expectedFeature(
                            MR_SHAPE_CAPSULE,
                            featureA[localPair]
                        ) &&
                    contacts[0].featureAndFlags[1] ==
                        expectedFeature(
                            MR_SHAPE_CAPSULE,
                            featureB[localPair]
                        ),
                "Metal capsule/capsule closest-feature ID is wrong"
            );
        }
    }
    const auto skew = contactsForPair(gpu, 8u, 9u).front();
    const auto parallel =
        contactsForPair(gpu, 10u, 11u).front();
    const auto coincident =
        contactsForPair(gpu, 12u, 13u).front();
    const auto boundary =
        contactsForPair(gpu, 14u, 15u).front();
    require(
        std::abs(skew.normalAndSeparation.w) <= 3.0e-6f &&
            std::abs(
                parallel.normalAndSeparation.w + 0.10f
            ) <= 3.0e-6f &&
            std::abs(
                coincident.normalAndSeparation.w + 0.50f
            ) <= 3.0e-6f &&
            std::abs(
                boundary.normalAndSeparation.w - 0.01999f
            ) <= 2.0e-5f,
        "Metal capsule/capsule signed separation is wrong"
    );

    const MetalRun replay = collideOnMetal(
        scene,
        environment,
        16u,
        16u
    );
    const MetalRun exact = collideOnMetal(
        scene,
        environment,
        9u,
        8u
    );
    require(
        sameSuccessfulRun(gpu, replay) &&
            sameSuccessfulRun(gpu, exact) &&
            exact.unusedOutputSlotsUntouched,
        "Metal capsule pair replay or exact capacity changed output"
    );
    const MetalRun pairOverflow = collideOnMetal(
        scene,
        environment,
        8u,
        16u
    );
    const MetalRun contactOverflow = collideOnMetal(
        scene,
        environment,
        16u,
        7u
    );
    require(
        pairOverflow.status.code ==
                MR_STEP_PAIR_CAPACITY_OVERFLOW &&
            pairOverflow.status.requiredPairs == 9u &&
            pairOverflow.status.requiredContacts == 8u &&
            pairOverflow.outputBuffersUntouched &&
            contactOverflow.status.code ==
                MR_STEP_CONTACT_CAPACITY_OVERFLOW &&
            contactOverflow.status.requiredPairs == 9u &&
            contactOverflow.status.requiredContacts == 8u &&
            contactOverflow.outputBuffersUntouched,
        "Metal capsule pair capacity failure was not exact and transactional"
    );
    return witnessError;
}

double verifySphereBoxPrimitive(
    const std::uint32_t environment
) {
    const Scene scene = makeSphereBoxScene();
    metalrobo::CollisionConfig config;
    config.environment = environment;
    config.capacities = {
        .pairCapacity = 8u,
        .rawContactCapacity = 8u,
        .manifoldCapacity = 8u,
    };
    metalrobo::PersistentManifoldCache cache;
    const auto cpu = metalrobo::collideCpuReference(
        scene.shapes,
        scene.bodies,
        config,
        cache
    );
    const MetalRun gpu = collideOnMetal(
        scene,
        environment,
        8u,
        8u
    );
    require(
        cpu.succeeded() &&
            cpu.pairs.size() == 6u &&
            cpu.rawContacts.size() == 6u &&
            gpu.status.code == MR_STEP_SUCCESS &&
            gpu.status.requiredPairs == 6u &&
            gpu.status.requiredContacts == 6u &&
            gpu.unusedOutputSlotsUntouched,
        "sphere/box CPU/Metal topology is wrong"
    );
    const double witnessError = compareWithCpu(cpu, gpu);

    constexpr std::array<std::uint32_t, 6> boxFeatures{
        14u, 14u, 17u, 26u, 14u, 28u,
    };
    for (std::size_t localPair = 0u;
         localPair < boxFeatures.size();
         ++localPair) {
        const std::uint32_t colliderA =
            static_cast<std::uint32_t>(2u * localPair);
        const std::uint32_t colliderB = colliderA + 1u;
        const auto contacts =
            contactsForPair(gpu, colliderA, colliderB);
        const bool sphereFirst = localPair != 1u;
        require(
            gpu.pairs[localPair].flags ==
                    metalrobo::collisionPairSphereBox &&
                contacts.size() == 1u &&
                contacts[0].featureAndFlags[0] ==
                    expectedFeature(
                        sphereFirst
                            ? MR_SHAPE_SPHERE
                            : MR_SHAPE_BOX,
                        sphereFirst
                            ? 0u
                            : boxFeatures[localPair]
                    ) &&
                contacts[0].featureAndFlags[1] ==
                    expectedFeature(
                        sphereFirst
                            ? MR_SHAPE_BOX
                            : MR_SHAPE_SPHERE,
                        sphereFirst
                            ? boxFeatures[localPair]
                            : 0u
                    ),
            "Metal sphere/box face, edge, corner, or interior feature is wrong"
        );
    }
    const auto sphereFirst =
        contactsForPair(gpu, 0u, 1u).front();
    const auto boxFirst =
        contactsForPair(gpu, 2u, 3u).front();
    const auto rotated =
        contactsForPair(gpu, 8u, 9u).front();
    const auto inside =
        contactsForPair(gpu, 10u, 11u).front();
    require(
        std::abs(
            sphereFirst.normalAndSeparation.x +
                boxFirst.normalAndSeparation.x
        ) <= 3.0e-6f &&
            std::abs(
                sphereFirst.normalAndSeparation.w -
                    boxFirst.normalAndSeparation.w
            ) <= 3.0e-6f,
        "Metal sphere/box swapped-order reciprocity failed"
    );
    require(
        std::abs(
            rotated.normalAndSeparation.x +
                0.7071067811865475f
        ) <= 4.0e-6f &&
            std::abs(
                rotated.normalAndSeparation.y +
                    0.7071067811865475f
            ) <= 4.0e-6f &&
            std::abs(
                rotated.normalAndSeparation.w + 0.001f
            ) <= 4.0e-6f,
        "Metal rotated sphere/box witness is wrong"
    );
    require(
        std::abs(inside.normalAndSeparation.x + 1.0f) <=
                3.0e-6f &&
            std::abs(
                inside.normalAndSeparation.w + 0.80f
            ) <= 3.0e-6f,
        "Metal sphere-inside-box signed witness is wrong"
    );

    const MetalRun replay = collideOnMetal(
        scene,
        environment,
        8u,
        8u
    );
    const MetalRun exact = collideOnMetal(
        scene,
        environment,
        6u,
        6u
    );
    const MetalRun pairOverflow = collideOnMetal(
        scene,
        environment,
        5u,
        8u
    );
    const MetalRun contactOverflow = collideOnMetal(
        scene,
        environment,
        8u,
        5u
    );
    require(
        sameSuccessfulRun(gpu, replay) &&
            sameSuccessfulRun(gpu, exact) &&
            exact.unusedOutputSlotsUntouched &&
            pairOverflow.status.code ==
                MR_STEP_PAIR_CAPACITY_OVERFLOW &&
            pairOverflow.status.requiredPairs == 6u &&
            pairOverflow.status.requiredContacts == 6u &&
            pairOverflow.outputBuffersUntouched &&
            contactOverflow.status.code ==
                MR_STEP_CONTACT_CAPACITY_OVERFLOW &&
            contactOverflow.status.requiredPairs == 6u &&
            contactOverflow.status.requiredContacts == 6u &&
            contactOverflow.outputBuffersUntouched,
        "Metal sphere/box replay or capacity transaction failed"
    );
    return witnessError;
}

double verifyCapsuleBoxAndBoxBoxPrimitives(
    const std::uint32_t environment
) {
    const Scene scene = makeCapsuleBoxBoxScene();
    metalrobo::CollisionConfig config;
    config.environment = environment;
    config.capacities = {
        .pairCapacity = 8u,
        .rawContactCapacity = 32u,
        .manifoldCapacity = 8u,
    };
    metalrobo::PersistentManifoldCache cache;
    const auto cpu = metalrobo::collideCpuReference(
        scene.shapes,
        scene.bodies,
        config,
        cache
    );
    const MetalRun gpu = collideOnMetal(
        scene,
        environment,
        8u,
        32u
    );
    require(
        cpu.succeeded() &&
            cpu.pairs.size() == 4u &&
            cpu.rawContacts.size() >= 4u &&
            gpu.status.code == MR_STEP_SUCCESS &&
            gpu.status.requiredPairs == 4u &&
            gpu.status.requiredContacts ==
                cpu.rawContacts.size() &&
            gpu.unusedOutputSlotsUntouched,
        "capsule/box and box/box CPU/Metal topology is wrong"
    );
    const double witnessError = compareWithCpu(cpu, gpu);
    for (std::uint32_t pairIndex = 0u;
         pairIndex < gpu.pairs.size();
         ++pairIndex) {
        const auto& pair = gpu.pairs[pairIndex];
        require(
            !contactsForPair(
                gpu,
                pair.colliderA,
                pair.colliderB
            ).empty() &&
                pair.flags ==
                    (pairIndex < 2u
                         ? metalrobo::collisionPairCapsuleBox
                         : metalrobo::collisionPairBoxBox),
            "Metal capsule/box or box/box pair class/contact is wrong"
        );
    }
    const MetalRun replay = collideOnMetal(
        scene,
        environment,
        8u,
        32u
    );
    require(
        sameSuccessfulRun(gpu, replay),
        "Metal capsule/box or box/box replay changed output"
    );
    return witnessError;
}

double verifyCylinderPrimitive(
    const std::uint32_t environment
) {
    const Scene scene = makeCylinderScene();
    metalrobo::CollisionConfig config;
    config.environment = environment;
    config.capacities = {
        .pairCapacity = 8u,
        .rawContactCapacity = 16u,
        .manifoldCapacity = 8u,
    };
    metalrobo::PersistentManifoldCache cache;
    const metalrobo::CollisionFrame cpu =
        metalrobo::collideCpuReference(
            scene.shapes,
            scene.bodies,
            config,
            cache,
            scene.exclusions
        );
    require(
        cpu.succeeded() &&
            cpu.pairs.size() == 4u &&
            cpu.rawContacts.size() == 11u &&
            cpu.manifoldHeaders.size() == 4u,
        "CPU cylinder orientation corpus changed topology"
    );
    const MetalRun gpu = collideOnMetal(
        scene,
        environment,
        8u,
        16u
    );
    require(
        gpu.status.code == MR_STEP_SUCCESS &&
            gpu.status.requiredPairs == 4u &&
            gpu.status.requiredContacts == 11u &&
            gpu.unusedOutputSlotsUntouched,
        "Metal cylinder orientation corpus changed topology"
    );
    const double witnessError = compareWithCpu(cpu, gpu);

    for (const MRCandidatePairGPU& pair : gpu.pairs) {
        require(
            pair.colliderA == 0u &&
                pair.colliderB >= 1u &&
                pair.colliderB <= 4u &&
                pair.flags ==
                    metalrobo::collisionPairCylinderPlane,
            "cylinder/plane pair classification is invalid"
        );
    }

    const std::array<std::vector<std::uint32_t>, 4>
        expectedLocalFeatures{{
            {0u, 1u, 2u, 3u},
            {4u, 5u, 6u, 7u},
            {8u, 9u},
            {10u},
        }};
    for (std::uint32_t cylinder = 1u;
         cylinder <= 4u;
         ++cylinder) {
        const std::vector<MRRawContactGPU> contacts =
            contactsForPair(gpu, 0u, cylinder);
        const auto& expected =
            expectedLocalFeatures[cylinder - 1u];
        require(
            contacts.size() == expected.size() &&
                manifoldPointCount(cpu, 0u, cylinder) ==
                    expected.size(),
            "cylinder support feature cardinality is wrong"
        );
        for (std::size_t point = 0u;
             point < contacts.size();
             ++point) {
            require(
                contacts[point].featureAndFlags[0] ==
                    expectedFeature(MR_SHAPE_PLANE, 0u) &&
                    contacts[point].featureAndFlags[1] ==
                        expectedFeature(
                            MR_SHAPE_CYLINDER,
                            expected[point]
                        ) &&
                    std::abs(
                        contacts[point]
                                .normalAndSeparation.w +
                            0.05f
                    ) <= 4.0e-6f,
                "cylinder feature ID or exact separation is wrong"
            );
        }
    }

    const std::vector<MRRawContactGPU> sideContacts =
        contactsForPair(gpu, 0u, 3u);
    require(
        sideContacts[0].pointBWorld.x <
                sideContacts[1].pointBWorld.x &&
            std::abs(
                sideContacts[1].pointBWorld.x -
                    sideContacts[0].pointBWorld.x -
                    1.2f
            ) <= 5.0e-6f,
        "cylinder side rim endpoints are not deterministically ordered"
    );

    const auto verifyAabb =
        [&](const std::uint32_t shapeIndex,
            const std::array<double, 3>& center,
            const std::array<double, 3>& extent) {
            const MRAabbGPU& aabb = cpu.worldAabbs[shapeIndex];
            const std::array<double, 3> lower{
                aabb.lower.x,
                aabb.lower.y,
                aabb.lower.z,
            };
            const std::array<double, 3> upper{
                aabb.upper.x,
                aabb.upper.y,
                aabb.upper.z,
            };
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                const double exactLower =
                    center[axis] - extent[axis];
                const double exactUpper =
                    center[axis] + extent[axis];
                require(
                    lower[axis] <= exactLower &&
                        upper[axis] >= exactUpper &&
                        exactLower - lower[axis] <= 3.0e-4 &&
                        upper[axis] - exactUpper <= 3.0e-4,
                    "oriented cylinder AABB is inward or needlessly loose"
                    " shape=" + std::to_string(shapeIndex) +
                    " axis=" + std::to_string(axis) +
                    " lower=" + std::to_string(lower[axis]) +
                    " exact_lower=" +
                        std::to_string(exactLower) +
                    " upper=" + std::to_string(upper[axis]) +
                    " exact_upper=" +
                        std::to_string(exactUpper)
                );
            }
        };
    constexpr double radius = 0.30;
    constexpr double halfLength = 0.60;
    constexpr double offset = 0.02;
    constexpr double inverseRootTwo =
        0.7071067811865475244;
    verifyAabb(
        1u,
        {-3.0, 0.55, 0.0},
        {radius + offset, halfLength + offset, radius + offset}
    );
    verifyAabb(
        2u,
        {-1.0, 0.55, 0.0},
        {radius + offset, halfLength + offset, radius + offset}
    );
    verifyAabb(
        3u,
        {1.0, 0.25, 0.0},
        {halfLength + offset, radius + offset, radius + offset}
    );
    const double tiltedExtent =
        (radius + halfLength) * inverseRootTwo + offset;
    verifyAabb(
        4u,
        {3.0, tiltedExtent - offset - 0.05, 0.0},
        {tiltedExtent, tiltedExtent, radius + offset}
    );

    const MetalRun replay = collideOnMetal(
        scene,
        environment,
        8u,
        16u
    );
    require(
        sameSuccessfulRun(gpu, replay),
        "cylinder Metal replay is not bit deterministic"
    );
    const MetalRun exact = collideOnMetal(
        scene,
        environment,
        gpu.status.requiredPairs,
        gpu.status.requiredContacts
    );
    require(
        sameSuccessfulRun(gpu, exact) &&
            exact.unusedOutputSlotsUntouched,
        "exact cylinder capacities are not deterministic"
    );
    const MetalRun pairOverflow = collideOnMetal(
        scene,
        environment,
        gpu.status.requiredPairs - 1u,
        16u
    );
    const MetalRun contactOverflow = collideOnMetal(
        scene,
        environment,
        8u,
        gpu.status.requiredContacts - 1u
    );
    require(
        pairOverflow.status.code ==
                MR_STEP_PAIR_CAPACITY_OVERFLOW &&
            pairOverflow.outputBuffersUntouched &&
            contactOverflow.status.code ==
                MR_STEP_CONTACT_CAPACITY_OVERFLOW &&
            contactOverflow.outputBuffersUntouched,
        "cylinder capacity failure was not transactional"
    );

    const std::vector<metalrobo::PersistentManifold>
        cacheSnapshot(cache.entries().begin(), cache.entries().end());
    metalrobo::CollisionConfig cpuOverflowConfig = config;
    cpuOverflowConfig.capacities.rawContactCapacity = 10u;
    const metalrobo::CollisionFrame cpuOverflow =
        metalrobo::collideCpuReference(
            scene.shapes,
            scene.bodies,
            cpuOverflowConfig,
            cache,
            scene.exclusions
        );
    require(
        cpuOverflow.diagnostics.code ==
                MR_STEP_CONTACT_CAPACITY_OVERFLOW &&
            cpuOverflow.worldAabbs.empty() &&
            sameBytes<metalrobo::PersistentManifold>(
                cache.entries(),
                cacheSnapshot
            ),
        "CPU cylinder overflow mutated frame payload or manifold cache"
    );

    const Scene reverseScene = makeCylinderFirstScene();
    metalrobo::PersistentManifoldCache reverseCache;
    const metalrobo::CollisionFrame reverseCpu =
        metalrobo::collideCpuReference(
            reverseScene.shapes,
            reverseScene.bodies,
            config,
            reverseCache,
            reverseScene.exclusions
        );
    const MetalRun reverseGpu = collideOnMetal(
        reverseScene,
        environment,
        2u,
        5u
    );
    require(
        reverseCpu.succeeded() &&
            reverseGpu.status.code == MR_STEP_SUCCESS,
        "reversed cylinder/plane endpoint order failed"
    );
    (void) compareWithCpu(reverseCpu, reverseGpu);
    const std::vector<MRRawContactGPU> reverseContacts =
        contactsForPair(reverseGpu, 0u, 1u);
    const std::vector<MRRawContactGPU> forwardContacts =
        contactsForPair(gpu, 0u, 1u);
    require(
        reverseContacts.size() == 4u &&
            forwardContacts.size() == 4u,
        "reversed cylinder cap lost witnesses"
    );
    for (std::size_t point = 0u;
         point < reverseContacts.size();
         ++point) {
        require(
            reverseContacts[point].featureAndFlags[0] ==
                    expectedFeature(
                        MR_SHAPE_CYLINDER,
                        static_cast<std::uint32_t>(point)
                    ) &&
                reverseContacts[point].featureAndFlags[1] ==
                    expectedFeature(MR_SHAPE_PLANE, 0u) &&
                reverseContacts[point].normalAndSeparation.y <
                    -0.99999f &&
                componentError(
                    reverseContacts[point].pointAWorld,
                    forwardContacts[point].pointBWorld
                ) <= kWitnessTolerance &&
                componentError(
                    reverseContacts[point].pointBWorld,
                    forwardContacts[point].pointAWorld
                ) <= kWitnessTolerance,
            "cylinder/plane endpoint swap changed geometry or features"
        );
    }

    const auto requireCylinderRejected =
        [&](Scene invalid, const std::string& label) {
            metalrobo::PersistentManifoldCache invalidCache;
            const metalrobo::CollisionFrame invalidCpu =
                metalrobo::collideCpuReference(
                    invalid.shapes,
                    invalid.bodies,
                    config,
                    invalidCache,
                    invalid.exclusions
                );
            const MetalRun invalidGpu = collideOnMetal(
                invalid,
                environment,
                8u,
                16u
            );
            require(
                invalidCpu.diagnostics.code ==
                        MR_STEP_NONFINITE_INPUT &&
                    invalidCpu.worldAabbs.empty() &&
                    invalidCpu.pairs.empty() &&
                    invalidCache.size() == 0u &&
                    invalidGpu.status.code ==
                        MR_STEP_NONFINITE_INPUT &&
                    invalidGpu.outputBuffersUntouched,
                label
            );
        };

    Scene zeroRadius = scene;
    zeroRadius.shapes[1].dimensions.x = 0.0f;
    requireCylinderRejected(
        std::move(zeroRadius),
        "zero cylinder radius was not rejected transactionally"
    );
    Scene zeroHalfLength = scene;
    zeroHalfLength.shapes[1].dimensions.y = 0.0f;
    requireCylinderRejected(
        std::move(zeroHalfLength),
        "zero cylinder half length was not rejected transactionally"
    );
    Scene subnormalHalfLength = scene;
    subnormalHalfLength.shapes[1].dimensions.y =
        std::numeric_limits<float>::denorm_min();
    requireCylinderRejected(
        std::move(subnormalHalfLength),
        "subnormal cylinder half length was not rejected"
    );
    Scene nonfiniteHalfLength = scene;
    nonfiniteHalfLength.shapes[1].dimensions.y =
        std::numeric_limits<float>::quiet_NaN();
    requireCylinderRejected(
        std::move(nonfiniteHalfLength),
        "non-finite cylinder half length was not rejected"
    );
    Scene outOfDomainRadius = scene;
    outOfDomainRadius.shapes[1].dimensions.x =
        std::nextafter(
            MR_MAX_COLLISION_INPUT_COORDINATE,
            std::numeric_limits<float>::infinity()
        );
    requireCylinderRejected(
        std::move(outOfDomainRadius),
        "out-of-domain cylinder radius was not rejected"
    );
    Scene invalidQuaternion = scene;
    invalidQuaternion.shapes[1].localRotation =
        f4(0.0f, 0.0f, 0.0f, 0.0f);
    requireCylinderRejected(
        std::move(invalidQuaternion),
        "invalid cylinder quaternion was not rejected"
    );

    Scene disabledMalformed = scene;
    disabledMalformed.shapes[1].flags =
        MR_SHAPE_FLAG_SIMULATION_DISABLED;
    disabledMalformed.shapes[1].dimensions =
        f4(0.0f, 0.0f, 0.0f, 0.0f);
    metalrobo::PersistentManifoldCache disabledCache;
    const metalrobo::CollisionFrame disabledCpu =
        metalrobo::collideCpuReference(
            disabledMalformed.shapes,
            disabledMalformed.bodies,
            config,
            disabledCache,
            disabledMalformed.exclusions
        );
    const MetalRun disabledGpu = collideOnMetal(
        disabledMalformed,
        environment,
        8u,
        16u
    );
    require(
        disabledCpu.succeeded() &&
            disabledGpu.status.code == MR_STEP_SUCCESS &&
            !shapeAppearsInPair(disabledCpu.pairs, 1u) &&
            !shapeAppearsInPair(disabledGpu.pairs, 1u),
        "disabled cylinder incorrectly required active dimensions"
    );

    const Scene sweepScene =
        makeCylinderOrientationSweepScene();
    metalrobo::CollisionConfig sweepConfig = config;
    sweepConfig.capacities = {
        .pairCapacity = 64u,
        .rawContactCapacity = 128u,
        .manifoldCapacity = 64u,
    };
    metalrobo::PersistentManifoldCache sweepCache;
    const metalrobo::CollisionFrame sweepCpu =
        metalrobo::collideCpuReference(
            sweepScene.shapes,
            sweepScene.bodies,
            sweepConfig,
            sweepCache,
            sweepScene.exclusions
        );
    const MetalRun sweepGpu = collideOnMetal(
        sweepScene,
        environment,
        64u,
        128u
    );
    require(
        sweepCpu.succeeded() &&
            sweepCpu.pairs.size() == 32u &&
            sweepGpu.status.code == MR_STEP_SUCCESS &&
            sweepGpu.status.requiredPairs == 32u,
        "cylinder orientation sweep changed broadphase topology"
    );
    const double sweepWitnessError =
        compareWithCpu(sweepCpu, sweepGpu);
    const MetalRun sweepReplay = collideOnMetal(
        sweepScene,
        environment,
        64u,
        128u
    );
    require(
        sameSuccessfulRun(sweepGpu, sweepReplay),
        "cylinder orientation sweep is not bit deterministic"
    );

    const Scene nearCapScene =
        makeCylinderNearCapRegressionScene();
    metalrobo::CollisionConfig nearCapConfig = config;
    nearCapConfig.capacities = {
        .pairCapacity = 2u,
        .rawContactCapacity = 8u,
        .manifoldCapacity = 2u,
    };
    metalrobo::PersistentManifoldCache nearCapCache;
    const metalrobo::CollisionFrame nearCapCpu =
        metalrobo::collideCpuReference(
            nearCapScene.shapes,
            nearCapScene.bodies,
            nearCapConfig,
            nearCapCache,
            nearCapScene.exclusions
        );
    const MetalRun nearCapGpu = collideOnMetal(
        nearCapScene,
        environment,
        2u,
        8u
    );
    constexpr std::uint32_t exactNearCapFeature = 10u;
    const std::uint32_t packedNearCapFeature =
        expectedFeature(
            MR_SHAPE_CYLINDER,
            exactNearCapFeature
        );
    const auto cpuExact = std::ranges::find_if(
        nearCapCpu.rawContacts,
        [packedNearCapFeature](const MRRawContactGPU& contact) {
            return contact.featureAndFlags[1] ==
                packedNearCapFeature;
        }
    );
    const std::vector<MRRawContactGPU> nearCapGpuContacts =
        contactsForPair(nearCapGpu, 0u, 1u);
    const auto gpuExact = std::ranges::find_if(
        nearCapGpuContacts,
        [packedNearCapFeature](const MRRawContactGPU& contact) {
            return contact.featureAndFlags[1] ==
                packedNearCapFeature;
        }
    );
    require(
        nearCapCpu.succeeded() &&
            nearCapCpu.pairs.size() == 1u &&
            nearCapCpu.rawContacts.size() == 1u &&
            cpuExact != nearCapCpu.rawContacts.end() &&
            nearCapGpu.status.code == MR_STEP_SUCCESS &&
            nearCapGpu.status.requiredPairs == 1u &&
            gpuExact != nearCapGpuContacts.end() &&
            cpuExact->normalAndSeparation.w < 0.0f,
        "near-cap arbitrary-azimuth cylinder support was missed"
    );
    const double nearCapWitnessError = std::max(
        componentError(
            cpuExact->pointAWorld,
            gpuExact->pointAWorld
        ),
        std::max(
            componentError(
                cpuExact->pointBWorld,
                gpuExact->pointBWorld
            ),
            componentError(
                cpuExact->normalAndSeparation,
                gpuExact->normalAndSeparation
            )
        )
    );
    require(
        nearCapWitnessError <= kWitnessTolerance,
        "near-cap exact cylinder witness diverged on Metal"
    );

    const Scene subEpsilonScene =
        makeCylinderNearCapSubEpsilonRegressionScene();
    metalrobo::PersistentManifoldCache subEpsilonCache;
    const metalrobo::CollisionFrame subEpsilonCpu =
        metalrobo::collideCpuReference(
            subEpsilonScene.shapes,
            subEpsilonScene.bodies,
            nearCapConfig,
            subEpsilonCache,
            subEpsilonScene.exclusions
        );
    const MetalRun subEpsilonGpu = collideOnMetal(
        subEpsilonScene,
        environment,
        2u,
        8u
    );
    const auto subEpsilonCpuExact = std::ranges::find_if(
        subEpsilonCpu.rawContacts,
        [packedNearCapFeature](const MRRawContactGPU& contact) {
            return contact.featureAndFlags[1] ==
                packedNearCapFeature;
        }
    );
    const std::vector<MRRawContactGPU> subEpsilonGpuContacts =
        contactsForPair(subEpsilonGpu, 0u, 1u);
    const auto subEpsilonGpuExact = std::ranges::find_if(
        subEpsilonGpuContacts,
        [packedNearCapFeature](const MRRawContactGPU& contact) {
            return contact.featureAndFlags[1] ==
                packedNearCapFeature;
        }
    );
    const float subEpsilonCpuSeparation =
        subEpsilonCpuExact != subEpsilonCpu.rawContacts.end()
        ? subEpsilonCpuExact->normalAndSeparation.w
        : std::numeric_limits<float>::infinity();
    const float subEpsilonGpuSeparation =
        subEpsilonGpuExact != subEpsilonGpuContacts.end()
        ? subEpsilonGpuExact->normalAndSeparation.w
        : std::numeric_limits<float>::infinity();
    require(
        subEpsilonCpu.succeeded() &&
            subEpsilonCpu.pairs.size() == 1u &&
            subEpsilonCpuExact !=
                subEpsilonCpu.rawContacts.end() &&
            subEpsilonCpuSeparation < 0.0f &&
            subEpsilonGpu.status.code == MR_STEP_SUCCESS &&
            subEpsilonGpu.status.requiredPairs == 1u &&
            subEpsilonGpuExact !=
                subEpsilonGpuContacts.end() &&
            subEpsilonGpuSeparation < 0.0f,
        "sub-epsilon cylinder tilt lost its amplified exact support"
        " cpu_status=" +
            std::to_string(subEpsilonCpu.diagnostics.code) +
            " cpu_pairs=" +
            std::to_string(subEpsilonCpu.pairs.size()) +
            " cpu_contacts=" +
            std::to_string(subEpsilonCpu.rawContacts.size()) +
            " cpu_separation=" +
            std::to_string(subEpsilonCpuSeparation) +
            " gpu_status=" +
            std::to_string(subEpsilonGpu.status.code) +
            " gpu_pairs=" +
            std::to_string(subEpsilonGpu.status.requiredPairs) +
            " gpu_contacts=" +
            std::to_string(subEpsilonGpuContacts.size()) +
            " gpu_separation=" +
            std::to_string(subEpsilonGpuSeparation)
    );

    return std::max({
        witnessError,
        sweepWitnessError,
        nearCapWitnessError,
    });
}

} // namespace

int main() {
    try {
        constexpr std::uint32_t environment = 23u;
        const Scene emptyScene;
        metalrobo::CollisionConfig emptyConfig;
        emptyConfig.environment = environment;
        metalrobo::PersistentManifoldCache emptyCache;
        const metalrobo::CollisionFrame emptyCpu =
            metalrobo::collideCpuReference(
                emptyScene.shapes,
                emptyScene.bodies,
                emptyConfig,
                emptyCache,
                emptyScene.exclusions
            );
        const MetalRun emptyMetal = collideOnMetal(
            emptyScene,
            environment,
            0u,
            0u
        );
        require(
            emptyCpu.succeeded() &&
                emptyCpu.pairs.empty() &&
                emptyCpu.rawContacts.empty() &&
                emptyMetal.status.code == MR_STEP_SUCCESS &&
                emptyMetal.status.requiredPairs == 0u &&
                emptyMetal.status.requiredContacts == 0u &&
                emptyMetal.outputBuffersUntouched,
            "zero-shape CPU/Metal collision world diverged"
        );

        const Scene scene = makeScene();

        metalrobo::CollisionConfig cpuConfig;
        cpuConfig.environment = environment;
        cpuConfig.capacities = {
            .pairCapacity = 128u,
            .rawContactCapacity = 128u,
            .manifoldCapacity = 128u,
        };
        metalrobo::PersistentManifoldCache cpuCache;
        const metalrobo::CollisionFrame cpu =
            metalrobo::collideCpuReference(
                scene.shapes,
                scene.bodies,
                cpuConfig,
                cpuCache,
                scene.exclusions
            );
        require(cpu.succeeded(), "CPU collision reference failed");
        require(
            !cpu.pairs.empty() && !cpu.rawContacts.empty(),
            "collision scene did not exercise the pipeline"
        );

        const std::uint32_t pairCapacity =
            static_cast<std::uint32_t>(cpu.pairs.size() + 3u);
        const std::uint32_t contactCapacity =
            static_cast<std::uint32_t>(
                cpu.rawContacts.size() + 3u
            );
        const MetalRun first = collideOnMetal(
            scene,
            environment,
            pairCapacity,
            contactCapacity
        );
        if (first.status.code != MR_STEP_SUCCESS) {
            std::cerr
                << "collision_baseline_code="
                << first.status.code
                << " required_pairs="
                << first.status.requiredPairs
                << " required_contacts="
                << first.status.requiredContacts
                << '\n';
        }
        require(
            first.status.code == MR_STEP_SUCCESS,
            "Metal collision baseline did not succeed"
        );
        require(
            first.status.requiredPairs == cpu.pairs.size() &&
                first.status.requiredContacts ==
                    cpu.rawContacts.size(),
            "Metal preflight counts differ from CPU reference"
        );
        require(
            first.status.activeContacts ==
                first.status.requiredContacts,
            "Metal active-contact diagnostic is incorrect"
        );
        require(
            first.unusedOutputSlotsUntouched,
            "Metal wrote beyond the required output prefix"
        );
        const double maximumWitnessError =
            compareWithCpu(cpu, first);
        const double cylinderWitnessError =
            verifyCylinderPrimitive(environment + 1u);
        const double capsulePairWitnessError =
            verifyCapsulePairPrimitives(environment + 2u);
        const double sphereBoxWitnessError =
            verifySphereBoxPrimitive(environment + 3u);
        const double capsuleBoxBoxWitnessError =
            verifyCapsuleBoxAndBoxBoxPrimitives(
                environment + 4u
            );

        require(
            containsPair(first.pairs, 2u, 3u) &&
                pairHasContact(first, 2u, 3u),
            "sphere/sphere analytic narrowphase was not exercised"
        );
        require(
            containsPair(first.pairs, 0u, 1u) &&
                pairHasContact(first, 0u, 1u),
            "sphere/plane analytic narrowphase was not exercised"
        );
        require(
            containsPair(first.pairs, 7u, 8u) &&
                !pairHasContact(first, 7u, 8u),
            "broadphase-only diagonal sphere pair was not preserved"
        );
        require(
            !containsPair(first.pairs, 1u, 2u),
            "explicit pair exclusion was ignored"
        );
        require(
            !containsPair(first.pairs, 9u, 10u),
            "collision group/mask filtering was ignored"
        );
        require(
            !containsPair(first.pairs, 11u, 12u),
            "same-body filtering was ignored"
        );
        require(
            !containsPair(first.pairs, 0u, 6u),
            "static/static filtering was ignored"
        );
        require(
            containsPair(first.pairs, 13u, 15u),
            "capsule/plane analytic pair was not emitted"
        );
        require(
            containsPair(first.pairs, 14u, 15u),
            "box/plane analytic pair was not emitted"
        );

        const std::vector<MRRawContactGPU> capsuleContacts =
            contactsForPair(first, 13u, 15u);
        require(
            capsuleContacts.size() == 2u,
            "horizontal capsule did not emit both endpoint witnesses"
        );
        for (std::uint32_t endpoint = 0u;
             endpoint < capsuleContacts.size();
             ++endpoint) {
            require(
                capsuleContacts[endpoint].featureAndFlags[0] ==
                    expectedFeature(
                        MR_SHAPE_CAPSULE,
                        endpoint
                    ) &&
                    capsuleContacts[endpoint]
                            .featureAndFlags[1] ==
                        expectedFeature(MR_SHAPE_PLANE, 0u),
                "capsule endpoint features were not stable and ordered"
            );
        }
        require(
            std::abs(
                capsuleContacts[0].normalAndSeparation.w -
                capsuleContacts[1].normalAndSeparation.w
            ) <= 2.0e-6f &&
                std::abs(
                    capsuleContacts[0].pointAWorld.x -
                    capsuleContacts[1].pointAWorld.x
                ) >= 0.99f,
            "equal-depth capsule endpoint degeneracy was mishandled"
        );

        const std::vector<MRRawContactGPU> boxContacts =
            contactsForPair(first, 14u, 15u);
        require(
            boxContacts.size() == 8u,
            "box/plane did not preserve all eight raw corner witnesses"
        );
        for (std::uint32_t boxVertex = 0u;
             boxVertex < boxContacts.size();
             ++boxVertex) {
            require(
                boxContacts[boxVertex].featureAndFlags[0] ==
                    expectedFeature(
                        MR_SHAPE_BOX,
                        boxVertex
                    ) &&
                    boxContacts[boxVertex]
                            .featureAndFlags[1] ==
                        expectedFeature(MR_SHAPE_PLANE, 0u),
                "box corner features were not stable and ordered"
            );
        }
        require(
            manifoldPointCount(cpu, 14u, 15u) == 4u,
            "CPU manifold did not reduce eight box witnesses to four"
        );
        require(
            !shapeAppearsInPair(first.pairs, 16u),
            "simulation-disabled unsupported geometry entered pairs"
        );

        const MetalRun replay = collideOnMetal(
            scene,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            sameSuccessfulRun(first, replay),
            "Metal collision replay was not bit deterministic"
        );
        const MetalRun exactCapacity = collideOnMetal(
            scene,
            environment,
            first.status.requiredPairs,
            first.status.requiredContacts
        );
        require(
            sameSuccessfulRun(first, exactCapacity) &&
                exactCapacity.unusedOutputSlotsUntouched,
            "exact declared capacities did not succeed deterministically"
        );

        require(
            first.status.requiredPairs > 0u &&
                first.status.requiredContacts > 0u,
            "overflow probes need nonzero requirements"
        );
        const MetalRun pairOverflow = collideOnMetal(
            scene,
            environment,
            first.status.requiredPairs - 1u,
            contactCapacity
        );
        require(
            pairOverflow.status.code ==
                MR_STEP_PAIR_CAPACITY_OVERFLOW &&
                pairOverflow.status.requiredPairs ==
                    first.status.requiredPairs &&
                pairOverflow.status.requiredContacts ==
                    first.status.requiredContacts &&
                pairOverflow.outputBuffersUntouched,
            "pair-capacity preflight was not exact and transactional"
        );

        const MetalRun contactOverflow = collideOnMetal(
            scene,
            environment,
            pairCapacity,
            first.status.requiredContacts - 1u
        );
        require(
            contactOverflow.status.code ==
                MR_STEP_CONTACT_CAPACITY_OVERFLOW &&
                contactOverflow.status.requiredPairs ==
                    first.status.requiredPairs &&
                contactOverflow.status.requiredContacts ==
                    first.status.requiredContacts &&
                contactOverflow.outputBuffersUntouched,
            "contact-capacity preflight was not exact and transactional"
        );

        Scene invalid = scene;
        invalid.shapes[3].localPosition.x =
            std::numeric_limits<float>::quiet_NaN();
        const MetalRun invalidResult = collideOnMetal(
            invalid,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            invalidResult.status.code ==
                MR_STEP_NONFINITE_INPUT &&
                invalidResult.outputBuffersUntouched,
            "non-finite input did not fail transactionally"
        );

        Scene invalidCapsule = scene;
        invalidCapsule.shapes[13].dimensions.y = 0.0f;
        const MetalRun invalidCapsuleResult = collideOnMetal(
            invalidCapsule,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            invalidCapsuleResult.status.code ==
                MR_STEP_NONFINITE_INPUT &&
                invalidCapsuleResult.outputBuffersUntouched,
            "degenerate capsule dimensions did not fail transactionally"
        );

        Scene invalidFlags = scene;
        invalidFlags.shapes[1].flags = 1u << 31u;
        metalrobo::PersistentManifoldCache invalidFlagsCache;
        const metalrobo::CollisionFrame invalidFlagsCpu =
            metalrobo::collideCpuReference(
                invalidFlags.shapes,
                invalidFlags.bodies,
                cpuConfig,
                invalidFlagsCache,
                invalidFlags.exclusions
            );
        const MetalRun invalidFlagsMetal = collideOnMetal(
            invalidFlags,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            invalidFlagsCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                invalidFlagsCpu.pairs.empty() &&
                invalidFlagsMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                invalidFlagsMetal.outputBuffersUntouched,
            "unknown shape flags did not fail consistently"
        );

        Scene invalidExclusion = scene;
        const std::uint32_t invalidCollider =
            static_cast<std::uint32_t>(
                invalidExclusion.shapes.size()
            );
        invalidExclusion.exclusions.push_back(
            {0u, invalidCollider}
        );
        invalidExclusion.gpuExclusions.push_back(
            {
                environment,
                0u,
                invalidCollider,
                0u
            }
        );
        metalrobo::PersistentManifoldCache
            invalidExclusionCache;
        const metalrobo::CollisionFrame invalidExclusionCpu =
            metalrobo::collideCpuReference(
                invalidExclusion.shapes,
                invalidExclusion.bodies,
                cpuConfig,
                invalidExclusionCache,
                invalidExclusion.exclusions
            );
        const MetalRun invalidExclusionMetal = collideOnMetal(
            invalidExclusion,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            invalidExclusionCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                invalidExclusionCpu.pairs.empty() &&
                invalidExclusionMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                invalidExclusionMetal.outputBuffersUntouched,
            "out-of-range exclusions did not fail consistently"
        );

        Scene invalidUnusedBody = scene;
        MRBodyStateGPU unusedBody =
            makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC);
        unusedBody.orientation =
            f4(0.0f, 0.0f, 0.0f, 0.0f);
        invalidUnusedBody.bodies.push_back(unusedBody);
        metalrobo::PersistentManifoldCache
            invalidUnusedBodyCache;
        const metalrobo::CollisionFrame invalidUnusedBodyCpu =
            metalrobo::collideCpuReference(
                invalidUnusedBody.shapes,
                invalidUnusedBody.bodies,
                cpuConfig,
                invalidUnusedBodyCache,
                invalidUnusedBody.exclusions
            );
        const MetalRun invalidUnusedBodyMetal = collideOnMetal(
            invalidUnusedBody,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            invalidUnusedBodyCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                invalidUnusedBodyCpu.pairs.empty() &&
                invalidUnusedBodyMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                invalidUnusedBodyMetal.outputBuffersUntouched,
            "malformed unused body did not fail consistently"
        );

        Scene malformedUnsupported = scene;
        malformedUnsupported.shapes[16].flags = 0u;
        malformedUnsupported.shapes[16].dimensions.x =
            std::numeric_limits<float>::quiet_NaN();
        metalrobo::PersistentManifoldCache
            malformedUnsupportedCache;
        const metalrobo::CollisionFrame malformedUnsupportedCpu =
            metalrobo::collideCpuReference(
                malformedUnsupported.shapes,
                malformedUnsupported.bodies,
                cpuConfig,
                malformedUnsupportedCache,
                malformedUnsupported.exclusions
            );
        const MetalRun malformedUnsupportedMetal = collideOnMetal(
            malformedUnsupported,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            malformedUnsupportedCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                malformedUnsupportedCpu.pairs.empty() &&
                malformedUnsupportedMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                malformedUnsupportedMetal.outputBuffersUntouched,
            "malformed unsupported shape error precedence diverged"
        );

        Scene globalPrecedence = scene;
        globalPrecedence.shapes[0].shapeType =
            MR_SHAPE_CYLINDER;
        globalPrecedence.shapes[1].dimensions.x =
            std::numeric_limits<float>::quiet_NaN();
        metalrobo::PersistentManifoldCache
            globalPrecedenceCache;
        const metalrobo::CollisionFrame globalPrecedenceCpu =
            metalrobo::collideCpuReference(
                globalPrecedence.shapes,
                globalPrecedence.bodies,
                cpuConfig,
                globalPrecedenceCache,
                globalPrecedence.exclusions
            );
        const MetalRun globalPrecedenceMetal = collideOnMetal(
            globalPrecedence,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            globalPrecedenceCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                globalPrecedenceCpu.pairs.empty() &&
                globalPrecedenceMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                globalPrecedenceMetal.outputBuffersUntouched,
            "global common-record error precedence diverged"
        );

        Scene derivedOverflow = scene;
        const std::uint32_t disabledBody =
            derivedOverflow.shapes[16].bodyIndex;
        derivedOverflow.bodies[disabledBody].position.x =
            std::numeric_limits<float>::max();
        derivedOverflow.shapes[16].localPosition.x =
            std::numeric_limits<float>::max();
        metalrobo::PersistentManifoldCache
            derivedOverflowCache;
        const metalrobo::CollisionFrame derivedOverflowCpu =
            metalrobo::collideCpuReference(
                derivedOverflow.shapes,
                derivedOverflow.bodies,
                cpuConfig,
                derivedOverflowCache,
                derivedOverflow.exclusions
            );
        const MetalRun derivedOverflowMetal = collideOnMetal(
            derivedOverflow,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            derivedOverflowCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                derivedOverflowCpu.pairs.empty() &&
                derivedOverflowMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                derivedOverflowMetal.outputBuffersUntouched,
            "derived FP32 transform overflow was not rejected"
        );

        Scene extremeExtent = scene;
        extremeExtent.shapes[1].dimensions.x =
            std::numeric_limits<float>::max();
        extremeExtent.shapes[1]
            .contactRestAndBoundingRadius.x = 1.0e30f;
        metalrobo::PersistentManifoldCache extremeExtentCache;
        const metalrobo::CollisionFrame extremeExtentCpu =
            metalrobo::collideCpuReference(
                extremeExtent.shapes,
                extremeExtent.bodies,
                cpuConfig,
                extremeExtentCache,
                extremeExtent.exclusions
            );
        const MetalRun extremeExtentMetal = collideOnMetal(
            extremeExtent,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            extremeExtentCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                extremeExtentCpu.pairs.empty() &&
                extremeExtentMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                extremeExtentMetal.outputBuffersUntouched,
            "out-of-domain finite extent was accepted"
        );

        Scene subnormalDimension = scene;
        subnormalDimension.shapes[1].dimensions.x =
            std::numeric_limits<float>::denorm_min();
        metalrobo::PersistentManifoldCache
            subnormalDimensionCache;
        const metalrobo::CollisionFrame subnormalDimensionCpu =
            metalrobo::collideCpuReference(
                subnormalDimension.shapes,
                subnormalDimension.bodies,
                cpuConfig,
                subnormalDimensionCache,
                subnormalDimension.exclusions
            );
        const MetalRun subnormalDimensionMetal = collideOnMetal(
            subnormalDimension,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            subnormalDimensionCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                subnormalDimensionCpu.pairs.empty() &&
                subnormalDimensionMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                subnormalDimensionMetal.outputBuffersUntouched,
            "subnormal active dimension was accepted"
        );

        Scene subnormalOffset = scene;
        subnormalOffset.shapes[1]
            .contactRestAndBoundingRadius.x =
                -std::numeric_limits<float>::denorm_min();
        metalrobo::PersistentManifoldCache
            subnormalOffsetCache;
        const metalrobo::CollisionFrame subnormalOffsetCpu =
            metalrobo::collideCpuReference(
                subnormalOffset.shapes,
                subnormalOffset.bodies,
                cpuConfig,
                subnormalOffsetCache,
                subnormalOffset.exclusions
            );
        const MetalRun subnormalOffsetMetal = collideOnMetal(
            subnormalOffset,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            subnormalOffsetCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                subnormalOffsetCpu.pairs.empty() &&
                subnormalOffsetMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                subnormalOffsetMetal.outputBuffersUntouched,
            "signed subnormal contact offset was accepted"
        );

        Scene quaternionBoundary = scene;
        const std::uint32_t boundaryBody =
            quaternionBoundary.shapes[16].bodyIndex;
        quaternionBoundary.bodies[boundaryBody].orientation =
            f4(
                0.72941094636917114,
                0.61207860708236694,
                -0.0090453298762440681,
                -0.30533197522163391
            );
        metalrobo::PersistentManifoldCache
            quaternionBoundaryCache;
        const metalrobo::CollisionFrame quaternionBoundaryCpu =
            metalrobo::collideCpuReference(
                quaternionBoundary.shapes,
                quaternionBoundary.bodies,
                cpuConfig,
                quaternionBoundaryCache,
                quaternionBoundary.exclusions
            );
        const MetalRun quaternionBoundaryMetal = collideOnMetal(
            quaternionBoundary,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            quaternionBoundaryCpu.succeeded() &&
                quaternionBoundaryCpu.pairs.size() ==
                    cpu.pairs.size() &&
                quaternionBoundaryMetal.status.code ==
                    MR_STEP_SUCCESS &&
                quaternionBoundaryMetal.status.requiredPairs ==
                    first.status.requiredPairs &&
                quaternionBoundaryMetal.status.requiredContacts ==
                    first.status.requiredContacts,
            "FP32 quaternion acceptance boundary diverged"
        );

        Scene activeUnsupported = scene;
        activeUnsupported.shapes[16].flags = 0u;
        const MetalRun unsupportedResult = collideOnMetal(
            activeUnsupported,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            unsupportedResult.status.code == MR_STEP_UNSUPPORTED &&
                unsupportedResult.outputBuffersUntouched,
            "active unsupported geometry did not fail transactionally"
        );

        std::cout << std::scientific << std::setprecision(6)
                  << "device=\"" << first.deviceName << "\""
                  << " broadphase=metal_o_n2_baseline"
                  << " shapes=" << scene.shapes.size()
                  << " pairs=" << first.status.requiredPairs
                  << " raw_contacts="
                  << first.status.requiredContacts
                  << " capsule_endpoint_contacts="
                  << capsuleContacts.size()
                  << " box_raw_contacts="
                  << boxContacts.size()
                  << " box_manifold_contacts="
                  << manifoldPointCount(cpu, 14u, 15u)
                  << " max_witness_error="
                  << maximumWitnessError
                  << " cylinder_max_witness_error="
                  << cylinderWitnessError
                  << " capsule_pair_max_witness_error="
                  << capsulePairWitnessError
                  << " sphere_box_max_witness_error="
                  << sphereBoxWitnessError
                  << " capsule_box_box_max_witness_error="
                  << capsuleBoxBoxWitnessError
                  << " pair_classes=10"
                  << " sphere_capsule_order=yes"
                  << " capsule_capsule_adversarial=yes"
                  << " sphere_box_adversarial=yes"
                  << " capsule_box_adversarial=yes"
                  << " box_box_sat=yes"
                  << " cylinder_cap_side_rim=yes"
                  << " cylinder_endpoint_order=yes"
                  << " cylinder_aabb_tight=yes"
                  << " cylinder_adversarial=yes"
                  << " canonical_filters=yes"
                  << " stable_features=yes"
                  << " deterministic_replay=yes"
                  << " overflow_transactional=yes"
                  << " finite_validation=yes"
                  << " strict_shape_flags=yes"
                  << " strict_exclusions=yes"
                  << " strict_body_stream=yes"
                  << " error_precedence=yes"
                  << " derived_transform_validation=yes"
                  << " bounded_collision_domain=yes"
                  << " subnormal_policy=yes"
                  << " quaternion_boundary_parity=yes"
                  << " zero_shape_world=yes"
                  << " disabled_unsupported_skipped=yes"
                  << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_collision_gpu_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
