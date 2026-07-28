#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/ConstraintSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace {

constexpr std::size_t kBodyCount = 3;
constexpr std::size_t kContactCount = 2;
constexpr double kAbsoluteTolerance = 3.0e-4;
constexpr double kRelativeTolerance = 3.0e-4;
constexpr double kInvariantTolerance = 5.0e-4;

using Bodies = std::array<MRBodyStateGPU, kBodyCount>;
using Contacts = std::array<MRContactConstraintGPU, kContactCount>;

struct Vec3 {
    double x;
    double y;
    double z;
};

struct MetalResult {
    Bodies bodies{};
    Contacts contacts{};
    MRSolverStatusGPU status{};
    std::string deviceName;
};

mr_float4 makeFloat4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

Vec3 xyz(const mr_float4 value) {
    return {
        static_cast<double>(value.x),
        static_cast<double>(value.y),
        static_cast<double>(value.z),
    };
}

Vec3 add(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 subtract(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 cross(const Vec3 a, const Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double dot(const Vec3 a, const Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 pointVelocity(
    const MRBodyStateGPU& body,
    const mr_float4 point
) {
    const Vec3 arm = subtract(xyz(point), xyz(body.position));
    return add(
        xyz(body.linearVelocityAndInverseMass),
        cross(xyz(body.angularVelocity), arm)
    );
}

float preSolveNormalVelocity(
    const Bodies& bodies,
    const MRContactConstraintGPU& contact
) {
    const Vec3 relative = subtract(
        pointVelocity(bodies[contact.bodyB], contact.pointAndSeparation),
        pointVelocity(bodies[contact.bodyA], contact.pointAndSeparation)
    );
    return static_cast<float>(dot(relative, xyz(contact.normal)));
}

MRBodyStateGPU makeBody(
    const Vec3 position,
    const Vec3 linearVelocity,
    const Vec3 angularVelocity,
    const double inverseMass,
    const Vec3 inverseInertiaDiagonal,
    const MRMotionType motionType,
    const std::uint32_t linkIndex
) {
    MRBodyStateGPU body{};
    body.position = makeFloat4(
        static_cast<float>(position.x),
        static_cast<float>(position.y),
        static_cast<float>(position.z)
    );
    body.orientation = makeFloat4(0.0f, 0.0f, 0.0f, 1.0f);
    body.linearVelocityAndInverseMass = makeFloat4(
        static_cast<float>(linearVelocity.x),
        static_cast<float>(linearVelocity.y),
        static_cast<float>(linearVelocity.z),
        static_cast<float>(inverseMass)
    );
    body.angularVelocity = makeFloat4(
        static_cast<float>(angularVelocity.x),
        static_cast<float>(angularVelocity.y),
        static_cast<float>(angularVelocity.z)
    );
    body.inverseInertiaWorldRow0 = makeFloat4(
        static_cast<float>(inverseInertiaDiagonal.x),
        0.0f,
        0.0f
    );
    body.inverseInertiaWorldRow1 = makeFloat4(
        0.0f,
        static_cast<float>(inverseInertiaDiagonal.y),
        0.0f
    );
    body.inverseInertiaWorldRow2 = makeFloat4(
        0.0f,
        0.0f,
        static_cast<float>(inverseInertiaDiagonal.z)
    );
    body.flagsAndIndices[0] = static_cast<mr_u32>(motionType);
    body.flagsAndIndices[1] = 0u;
    body.flagsAndIndices[2] = linkIndex;
    body.flagsAndIndices[3] = 0u;
    return body;
}

MRContactConstraintGPU makeContact(
    const std::uint32_t bodyA,
    const std::uint32_t bodyB,
    const Vec3 point,
    const double separation,
    const double staticFriction,
    const double dynamicFriction,
    const std::uint64_t pairKey,
    const std::uint64_t featureKey
) {
    MRContactConstraintGPU contact{};
    contact.bodyA = bodyA;
    contact.bodyB = bodyB;
    contact.flags = 0u;
    contact.islandIndex = 0u;
    contact.pairKey = pairKey;
    contact.featureKey = featureKey;
    contact.pointAndSeparation = makeFloat4(
        static_cast<float>(point.x),
        static_cast<float>(point.y),
        static_cast<float>(point.z),
        static_cast<float>(separation)
    );
    contact.normal = makeFloat4(0.0f, 0.0f, 1.0f);
    contact.friction = makeFloat4(
        static_cast<float>(staticFriction),
        static_cast<float>(dynamicFriction),
        0.0f,
        0.0f
    );
    // No restitution, compliance, or impulse cap in this oracle case.
    contact.response = makeFloat4(0.0f, 1.0f, 0.0f, 0.0f);
    contact.targetVelocityAndPreSolveNormal =
        makeFloat4(0.0f, 0.0f, 0.0f, 0.0f);
    contact.impulses = makeFloat4(0.0f, 0.0f, 0.0f, 0.0f);
    return contact;
}

std::pair<Bodies, Contacts> makeStackCase() {
    Bodies bodies{
        makeBody(
            {0.0, 0.0, -0.5},
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0},
            0.0,
            {0.0, 0.0, 0.0},
            MR_MOTION_STATIC,
            0u
        ),
        makeBody(
            {0.0, 0.0, 0.5},
            {0.80, -0.35, -1.20},
            {0.10, -0.20, 0.30},
            0.5,
            {1.40, 1.10, 0.90},
            MR_MOTION_DYNAMIC,
            1u
        ),
        makeBody(
            {0.0, 0.0, 1.5},
            {-0.55, 0.45, -1.80},
            {-0.25, 0.15, -0.10},
            1.0,
            {1.80, 1.60, 1.30},
            MR_MOTION_DYNAMIC,
            2u
        ),
    };

    Contacts contacts{
        // Frictionless ground isolates horizontal momentum from the static
        // reaction while still exercising stack propagation.
        makeContact(
            0u,
            1u,
            {0.12, -0.08, 0.0},
            -0.012,
            0.0,
            0.0,
            0x0000000000000001ull,
            0x47524f554e440001ull
        ),
        // The upper/lower contact has slip in both tangent directions and an
        // off-centre point, exercising coupled friction and angular response.
        makeContact(
            1u,
            2u,
            {-0.10, 0.06, 1.0},
            -0.008,
            0.75,
            0.55,
            0x0000000100000002ull,
            0x535441434b000001ull
        ),
    };

    for (MRContactConstraintGPU& contact : contacts) {
        contact.targetVelocityAndPreSolveNormal.w =
            preSolveNormalVelocity(bodies, contact);
    }
    return {bodies, contacts};
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
    const std::string description = nsString(error.localizedDescription);
    return description.empty() ? nsString(error.description) : description;
}

template <typename T>
id<MTLBuffer> makeSharedBuffer(
    id<MTLDevice> device,
    const T* data,
    const std::size_t count,
    NSString* label
) {
    if (data == nullptr || count == 0u) {
        throw std::invalid_argument("cannot allocate an empty Metal buffer");
    }
    if (count > std::numeric_limits<NSUInteger>::max() / sizeof(T)) {
        throw std::overflow_error("Metal buffer byte count overflow");
    }
    const NSUInteger bytes = static_cast<NSUInteger>(count * sizeof(T));
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:data
                           length:bytes
                          options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        throw std::runtime_error(
            "failed to allocate shared Metal buffer '" + nsString(label) + "'"
        );
    }
    buffer.label = label;
    return buffer;
}

MetalResult solveOnMetal(
    const Bodies& initialBodies,
    const Contacts& initialContacts,
    const MRSolverBatchGPU& batch
) {
    @autoreleasepool {
        const std::string metallibPath = METALROBO_DEFAULT_METALLIB;
        if (metallibPath.empty()) {
            throw std::runtime_error(
                "METALROBO_DEFAULT_METALLIB is empty"
            );
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            throw std::runtime_error("no Metal-capable device is available");
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            throw std::runtime_error("failed to create Metal command queue");
        }

        NSString* path =
            [NSString stringWithUTF8String:metallibPath.c_str()];
        if (path == nil) {
            throw std::runtime_error(
                "METALROBO_DEFAULT_METALLIB is not valid UTF-8"
            );
        }
        NSError* error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                error:&error];
        if (library == nil) {
            throw std::runtime_error(
                "failed to load metallib '" + metallibPath +
                "': " + describeError(error)
            );
        }

        id<MTLFunction> function =
            [library newFunctionWithName:@"mr_solve_contact_constraints"];
        if (function == nil) {
            throw std::runtime_error(
                "metallib does not contain mr_solve_contact_constraints"
            );
        }
        error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            throw std::runtime_error(
                "failed to create contact solver pipeline: " +
                describeError(error)
            );
        }

        MRSolverStatusGPU initialStatus{};
        initialStatus.code = MR_STEP_UNSUPPORTED;
        initialStatus.iterations = 0xffffffffu;
        id<MTLBuffer> batchBuffer =
            makeSharedBuffer(device, &batch, 1u, @"contact batches");
        id<MTLBuffer> bodyBuffer = makeSharedBuffer(
            device,
            initialBodies.data(),
            initialBodies.size(),
            @"contact bodies"
        );
        id<MTLBuffer> contactBuffer = makeSharedBuffer(
            device,
            initialContacts.data(),
            initialContacts.size(),
            @"contact constraints"
        );
        id<MTLBuffer> statusBuffer =
            makeSharedBuffer(device, &initialStatus, 1u, @"solver statuses");

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (commandBuffer == nil || encoder == nil) {
            throw std::runtime_error(
                "failed to create Metal command buffer or encoder"
            );
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:batchBuffer offset:0 atIndex:0];
        [encoder setBuffer:bodyBuffer offset:0 atIndex:1];
        [encoder setBuffer:contactBuffer offset:0 atIndex:2];
        [encoder setBuffer:statusBuffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error(
                "contact solver command failed: " +
                describeError(commandBuffer.error)
            );
        }

        MetalResult result{};
        std::memcpy(
            result.bodies.data(),
            bodyBuffer.contents,
            sizeof(result.bodies)
        );
        std::memcpy(
            result.contacts.data(),
            contactBuffer.contents,
            sizeof(result.contacts)
        );
        std::memcpy(
            &result.status,
            statusBuffer.contents,
            sizeof(result.status)
        );
        result.deviceName = nsString(device.name);
        return result;
    }
}

bool finiteFloat4(const mr_float4 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z) && std::isfinite(value.w);
}

bool finiteBodyOutput(const MRBodyStateGPU& body) {
    return finiteFloat4(body.linearVelocityAndInverseMass) &&
        finiteFloat4(body.angularVelocity);
}

bool finiteStatus(const MRSolverStatusGPU& status) {
    return finiteFloat4(status.residuals);
}

bool statusSucceeded(const std::uint32_t code) {
    return code == MR_STEP_SUCCESS ||
        code == MR_STEP_FIXED_BUDGET_COMPLETE;
}

bool nearlyEqual(
    const double left,
    const double right,
    const double absoluteTolerance = kAbsoluteTolerance,
    const double relativeTolerance = kRelativeTolerance
) {
    const double scale = std::max(std::abs(left), std::abs(right));
    return std::abs(left - right) <=
        absoluteTolerance + relativeTolerance * scale;
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Vec3 dynamicLinearMomentum(const Bodies& bodies) {
    Vec3 momentum{};
    for (const MRBodyStateGPU& body : bodies) {
        const double inverseMass = body.linearVelocityAndInverseMass.w;
        if (inverseMass <= 0.0) {
            continue;
        }
        const double mass = 1.0 / inverseMass;
        momentum.x += mass * body.linearVelocityAndInverseMass.x;
        momentum.y += mass * body.linearVelocityAndInverseMass.y;
        momentum.z += mass * body.linearVelocityAndInverseMass.z;
    }
    return momentum;
}

double maximumConeViolation(const Contacts& contacts) {
    double maximum = 0.0;
    for (const MRContactConstraintGPU& contact : contacts) {
        const double normal = contact.impulses.x;
        const double tangent =
            std::hypot(contact.impulses.y, contact.impulses.z);
        const double friction =
            std::max(contact.friction.x, contact.friction.y);
        maximum = std::max(maximum, std::max(-normal, 0.0));
        maximum = std::max(maximum, tangent - friction * normal);
        maximum = std::max(
            maximum,
            std::abs(static_cast<double>(contact.impulses.w)) -
                static_cast<double>(contact.friction.w) * normal
        );
        if (contact.response.w > 0.0f) {
            maximum = std::max(
                maximum,
                normal - static_cast<double>(contact.response.w)
            );
        }
    }
    return std::max(maximum, 0.0);
}

void validateStaticBody(
    const MRBodyStateGPU& initial,
    const MRBodyStateGPU& solved,
    const char* backend
) {
    require(
        std::memcmp(&initial, &solved, sizeof(initial)) == 0,
        std::string{backend} + " solver modified the static body"
    );
}

void verifyTransactionalArithmeticFailure(
    metalrobo::ContactSolverConfig config
) {
    auto [bodies, contacts] = makeStackCase();
    contacts[0] = makeContact(
        0u,
        1u,
        {0.0, 0.0, 0.0},
        -0.01,
        0.0,
        0.0,
        1u,
        1u
    );
    contacts[1] = makeContact(
        0u,
        2u,
        {0.0, 0.0, 1.0},
        -0.01,
        0.0,
        0.0,
        2u,
        1u
    );
    contacts[0].impulses.x = 1.0f;
    contacts[1].impulses.x = 1.0e30f;
    bodies[2].linearVelocityAndInverseMass.w = 1.0e20f;
    for (MRContactConstraintGPU& contact : contacts) {
        contact.targetVelocityAndPreSolveNormal.w =
            preSolveNormalVelocity(bodies, contact);
    }
    const Bodies originalBodies = bodies;
    const Contacts originalContacts = contacts;

    config.enableWarmStart = true;
    config.warmStartScale = 1.0;
    config.velocityIterations = 1u;
    Bodies cpuBodies = bodies;
    Contacts cpuContacts = contacts;
    const auto cpu = metalrobo::solveContactConstraints(
        cpuBodies,
        cpuContacts,
        config
    );
    require(
        cpu.code == MR_STEP_NONFINITE_RESULT &&
            std::memcmp(
                cpuBodies.data(),
                originalBodies.data(),
                sizeof(cpuBodies)
            ) == 0 &&
            std::memcmp(
                cpuContacts.data(),
                originalContacts.data(),
                sizeof(cpuContacts)
            ) == 0,
        "CPU arithmetic-failure rollback regressed"
    );

    const MRSolverBatchGPU batch = metalrobo::makeSolverBatch(
        0u,
        static_cast<std::uint32_t>(bodies.size()),
        0u,
        static_cast<std::uint32_t>(contacts.size()),
        config
    );
    const MetalResult gpu = solveOnMetal(bodies, contacts, batch);
    require(
        gpu.status.code == MR_STEP_NONFINITE_RESULT &&
            std::memcmp(
                gpu.bodies.data(),
                originalBodies.data(),
                sizeof(gpu.bodies)
            ) == 0 &&
            std::memcmp(
                gpu.contacts.data(),
                originalContacts.data(),
                sizeof(gpu.contacts)
            ) == 0,
        "Metal arithmetic failure published partial impulses or velocities"
    );
}

} // namespace

int main() {
    try {
        const auto [initialBodies, initialContacts] = makeStackCase();

        metalrobo::ContactSolverConfig config;
        config.timestep = 1.0 / 240.0;
        config.errorReduction = 0.18;
        config.penetrationSlop = 1.0e-4;
        config.maxDepenetrationVelocity = 1.5;
        config.impulseTolerance = 1.0e-7;
        config.warmStartScale = 1.0;
        config.minimumInverseLinearEffectiveMass = 1.0e-8;
        config.minimumInverseAngularEffectiveMass = 1.0e-8;
        config.velocityIterations = 48u;
        config.enableWarmStart = false;
        config.enableEarlyExit = false;
        config.deterministic = true;

        Bodies cpuBodies = initialBodies;
        Contacts cpuContacts = initialContacts;
        const metalrobo::ContactSolverDiagnostics cpuDiagnostics =
            metalrobo::solveContactConstraints(
                cpuBodies,
                cpuContacts,
                config
            );
        const MRSolverBatchGPU batch = metalrobo::makeSolverBatch(
            0u,
            static_cast<std::uint32_t>(initialBodies.size()),
            0u,
            static_cast<std::uint32_t>(initialContacts.size()),
            config
        );
        const MetalResult gpu =
            solveOnMetal(initialBodies, initialContacts, batch);

        require(cpuDiagnostics.succeeded(), "CPU solver status is not success");
        require(
            statusSucceeded(gpu.status.code),
            "Metal solver status is not success"
        );
        require(
            static_cast<std::uint32_t>(cpuDiagnostics.code) == gpu.status.code,
            "CPU and Metal solver status codes differ"
        );
        require(
            cpuDiagnostics.iterations == config.velocityIterations &&
                gpu.status.iterations == config.velocityIterations,
            "fixed-budget iteration count is incorrect"
        );
        require(
            cpuDiagnostics.activeContacts == kContactCount &&
                gpu.status.activeContacts == kContactCount,
            "active-contact count is incorrect"
        );
        require(
            cpuDiagnostics.islandCount == 1u &&
                gpu.status.islandCount == 1u,
            "vertical stack should form one constraint island"
        );
        require(finiteStatus(gpu.status), "Metal status contains non-finite data");
        require(
            std::isfinite(cpuDiagnostics.maximumImpulseDelta) &&
                std::isfinite(cpuDiagnostics.maximumNormalResidual) &&
                std::isfinite(cpuDiagnostics.maximumConeViolation) &&
                std::isfinite(
                    cpuDiagnostics.inverseLinearEffectiveMassSpread
                ),
            "CPU diagnostics contain non-finite data"
        );

        double maximumLinearError = 0.0;
        double maximumAngularError = 0.0;
        for (std::size_t body = 0; body < kBodyCount; ++body) {
            require(
                finiteBodyOutput(cpuBodies[body]) &&
                    finiteBodyOutput(gpu.bodies[body]),
                "body velocity contains non-finite data"
            );
            const std::array<double, 3> cpuLinear{
                cpuBodies[body].linearVelocityAndInverseMass.x,
                cpuBodies[body].linearVelocityAndInverseMass.y,
                cpuBodies[body].linearVelocityAndInverseMass.z,
            };
            const std::array<double, 3> gpuLinear{
                gpu.bodies[body].linearVelocityAndInverseMass.x,
                gpu.bodies[body].linearVelocityAndInverseMass.y,
                gpu.bodies[body].linearVelocityAndInverseMass.z,
            };
            const std::array<double, 3> cpuAngular{
                cpuBodies[body].angularVelocity.x,
                cpuBodies[body].angularVelocity.y,
                cpuBodies[body].angularVelocity.z,
            };
            const std::array<double, 3> gpuAngular{
                gpu.bodies[body].angularVelocity.x,
                gpu.bodies[body].angularVelocity.y,
                gpu.bodies[body].angularVelocity.z,
            };
            for (std::size_t axis = 0; axis < 3u; ++axis) {
                maximumLinearError = std::max(
                    maximumLinearError,
                    std::abs(cpuLinear[axis] - gpuLinear[axis])
                );
                maximumAngularError = std::max(
                    maximumAngularError,
                    std::abs(cpuAngular[axis] - gpuAngular[axis])
                );
                require(
                    nearlyEqual(cpuLinear[axis], gpuLinear[axis]),
                    "CPU/Metal linear velocity parity exceeded FP32 tolerance"
                );
                require(
                    nearlyEqual(cpuAngular[axis], gpuAngular[axis]),
                    "CPU/Metal angular velocity parity exceeded FP32 tolerance"
                );
            }
        }

        double maximumImpulseError = 0.0;
        for (std::size_t contact = 0; contact < kContactCount; ++contact) {
            require(
                finiteFloat4(cpuContacts[contact].impulses) &&
                    finiteFloat4(gpu.contacts[contact].impulses),
                "contact impulse contains non-finite data"
            );
            const std::array<double, 4> cpuImpulse{
                cpuContacts[contact].impulses.x,
                cpuContacts[contact].impulses.y,
                cpuContacts[contact].impulses.z,
                cpuContacts[contact].impulses.w,
            };
            const std::array<double, 4> gpuImpulse{
                gpu.contacts[contact].impulses.x,
                gpu.contacts[contact].impulses.y,
                gpu.contacts[contact].impulses.z,
                gpu.contacts[contact].impulses.w,
            };
            for (std::size_t row = 0; row < 4u; ++row) {
                maximumImpulseError = std::max(
                    maximumImpulseError,
                    std::abs(cpuImpulse[row] - gpuImpulse[row])
                );
                require(
                    nearlyEqual(cpuImpulse[row], gpuImpulse[row]),
                    "CPU/Metal impulse parity exceeded FP32 tolerance"
                );
            }
        }

        validateStaticBody(initialBodies[0], cpuBodies[0], "CPU");
        validateStaticBody(initialBodies[0], gpu.bodies[0], "Metal");

        const double cpuConeViolation = maximumConeViolation(cpuContacts);
        const double gpuConeViolation = maximumConeViolation(gpu.contacts);
        require(
            cpuConeViolation <= kInvariantTolerance &&
                gpuConeViolation <= kInvariantTolerance,
            "a solved contact impulse lies outside its friction cone"
        );
        const double cpuFrictionImpulse = std::hypot(
            cpuContacts[1].impulses.y,
            cpuContacts[1].impulses.z
        );
        const double gpuFrictionImpulse = std::hypot(
            gpu.contacts[1].impulses.y,
            gpu.contacts[1].impulses.z
        );
        require(
            cpuContacts[0].impulses.x > 0.0f &&
                cpuContacts[1].impulses.x > 0.0f &&
                gpu.contacts[0].impulses.x > 0.0f &&
                gpu.contacts[1].impulses.x > 0.0f,
            "vertical stack did not activate both normal constraints"
        );
        require(
            cpuFrictionImpulse > 1.0e-5 &&
                gpuFrictionImpulse > 1.0e-5,
            "frictional stack contact produced no tangent impulse"
        );

        const Vec3 initialMomentum = dynamicLinearMomentum(initialBodies);
        const Vec3 cpuMomentum = dynamicLinearMomentum(cpuBodies);
        const Vec3 gpuMomentum = dynamicLinearMomentum(gpu.bodies);
        const double cpuMomentumXYError = std::hypot(
            cpuMomentum.x - initialMomentum.x,
            cpuMomentum.y - initialMomentum.y
        );
        const double gpuMomentumXYError = std::hypot(
            gpuMomentum.x - initialMomentum.x,
            gpuMomentum.y - initialMomentum.y
        );
        require(
            cpuMomentumXYError <= kInvariantTolerance &&
                gpuMomentumXYError <= kInvariantTolerance,
            "horizontal momentum changed despite zero external tangent impulse"
        );

        const std::array<double, 4> cpuResiduals{
            cpuDiagnostics.maximumImpulseDelta,
            cpuDiagnostics.maximumNormalResidual,
            cpuDiagnostics.maximumConeViolation,
            cpuDiagnostics.inverseLinearEffectiveMassSpread,
        };
        const std::array<double, 4> gpuResiduals{
            gpu.status.residuals.x,
            gpu.status.residuals.y,
            gpu.status.residuals.z,
            gpu.status.residuals.w,
        };
        for (std::size_t index = 0; index < cpuResiduals.size(); ++index) {
            require(
                nearlyEqual(
                    cpuResiduals[index],
                    gpuResiduals[index],
                    1.0e-3,
                    1.0e-3
                ),
                "CPU/Metal solver diagnostic parity exceeded tolerance"
            );
        }
        require(
            gpu.status.residuals.z <= kInvariantTolerance,
            "Metal solver reported a friction-cone violation"
        );
        verifyTransactionalArithmeticFailure(config);

        std::cout << std::scientific << std::setprecision(6)
                  << "device=\"" << gpu.deviceName << "\""
                  << " bodies=" << kBodyCount
                  << " contacts=" << kContactCount
                  << " iterations=" << gpu.status.iterations
                  << " max_linear_error=" << maximumLinearError
                  << " max_angular_error=" << maximumAngularError
                  << " max_impulse_error=" << maximumImpulseError
                  << " max_cone_violation="
                  << std::max(cpuConeViolation, gpuConeViolation)
                  << " momentum_xy_error="
                  << std::max(cpuMomentumXYError, gpuMomentumXYError)
                  << " friction_impulse=" << gpuFrictionImpulse
                  << " arithmetic_rollback=yes"
                  << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_contact_gpu_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
