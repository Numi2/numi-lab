#include "metalrobo/RigidBodyWorld.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <tuple>
#include <utility>

namespace metalrobo {
namespace {

constexpr double kTiny = 1.0e-15;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Quaternion {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
};

struct Mat3 {
    double m[3][3]{};
};

Vec3 operator+(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3 value, const double scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

Vec3 operator/(const Vec3 value, const double scale) {
    return {value.x / scale, value.y / scale, value.z / scale};
}

double dot(const Vec3 a, const Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 a, const Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

Vec3 xyz(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

mr_float4 f4(const Vec3 value, const float w = 0.0f) {
    return {
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
        w,
    };
}

bool finite4(const mr_float4 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z) && std::isfinite(value.w);
}

bool finiteMaterial(const MRMaterialGPU& material) {
    return finite4(material.friction) &&
        finite4(material.response) &&
        finite4(material.geometry) &&
        material.friction.x >= 0.0f &&
        material.friction.y >= 0.0f &&
        material.friction.x >= material.friction.y &&
        material.friction.z >= 0.0f &&
        material.friction.w >= 0.0f &&
        material.response.x >= 0.0f &&
        material.response.x <= 1.0f &&
        material.response.y >= 0.0f &&
        material.response.z >= 0.0f &&
        material.response.w >= 0.0f &&
        material.geometry.x >= 0.0f &&
        material.geometry.y >= 0.0f;
}

Quaternion normalized(const Quaternion value) {
    const double length = std::sqrt(
        value.x * value.x + value.y * value.y +
        value.z * value.z + value.w * value.w
    );
    if (!(length > kTiny) || !std::isfinite(length)) {
        return {};
    }
    return {
        value.x / length,
        value.y / length,
        value.z / length,
        value.w / length,
    };
}

Mat3 rotationMatrix(const Quaternion input) {
    const Quaternion q = normalized(input);
    const double xx = q.x * q.x;
    const double yy = q.y * q.y;
    const double zz = q.z * q.z;
    const double xy = q.x * q.y;
    const double xz = q.x * q.z;
    const double yz = q.y * q.z;
    const double xw = q.x * q.w;
    const double yw = q.y * q.w;
    const double zw = q.z * q.w;
    return {{
        {1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)},
        {2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)},
        {2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)},
    }};
}

Vec3 operator*(const Mat3& matrix, const Vec3 value) {
    return {
        matrix.m[0][0] * value.x +
            matrix.m[0][1] * value.y +
            matrix.m[0][2] * value.z,
        matrix.m[1][0] * value.x +
            matrix.m[1][1] * value.y +
            matrix.m[1][2] * value.z,
        matrix.m[2][0] * value.x +
            matrix.m[2][1] * value.y +
            matrix.m[2][2] * value.z,
    };
}

Mat3 matrix(
    const mr_float4 row0,
    const mr_float4 row1,
    const mr_float4 row2
) {
    return {{
        {row0.x, row0.y, row0.z},
        {row1.x, row1.y, row1.z},
        {row2.x, row2.y, row2.z},
    }};
}

Quaternion quaternion(const mr_float4 value) {
    return normalized({value.x, value.y, value.z, value.w});
}

Vec3 pointVelocity(
    const MRBodyStateGPU& body,
    const Vec3 point
) {
    if (body.flagsAndIndices[0] == MR_MOTION_STATIC) {
        return {};
    }
    return xyz(body.linearVelocityAndInverseMass) +
        cross(
            xyz(body.angularVelocity),
            point - xyz(body.position)
        );
}

double geometricMean(const float left, const float right) {
    return std::sqrt(
        static_cast<double>(left) * static_cast<double>(right)
    );
}

std::uint64_t pairKey(
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    return
        (static_cast<std::uint64_t>(colliderA) << 32u) |
        colliderB;
}

std::uint64_t featureKey(
    const std::uint32_t featureA,
    const std::uint32_t featureB
) {
    return
        (static_cast<std::uint64_t>(featureA) << 32u) |
        featureB;
}

bool constraintsSorted(
    const std::vector<MRContactConstraintGPU>& constraints
) {
    return std::ranges::is_sorted(
        constraints,
        [](const MRContactConstraintGPU& left,
           const MRContactConstraintGPU& right) {
            return std::tie(left.pairKey, left.featureKey) <
                std::tie(right.pairKey, right.featureKey);
        }
    );
}

std::pair<Vec3, Vec3> contactBasis(const Vec3 unitNormal) {
    const Vec3 absolute{
        std::abs(unitNormal.x),
        std::abs(unitNormal.y),
        std::abs(unitNormal.z),
    };
    Vec3 reference{};
    if (absolute.x <= absolute.y && absolute.x <= absolute.z) {
        reference = {1.0, 0.0, 0.0};
    } else if (absolute.y <= absolute.z) {
        reference = {0.0, 1.0, 0.0};
    } else {
        reference = {0.0, 0.0, 1.0};
    }
    const Vec3 tangent = cross(reference, unitNormal);
    const double tangentLength = norm(tangent);
    const Vec3 tangentU = tangent / tangentLength;
    return {tangentU, cross(unitNormal, tangentU)};
}

std::pair<Vec3, Vec3> contactBasis(
    const Vec3 unitNormal,
    const Vec3 authoredTangent
) {
    const Vec3 tangent =
        authoredTangent -
        unitNormal * dot(unitNormal, authoredTangent);
    const Vec3 tangentU = tangent / norm(tangent);
    return {tangentU, cross(unitNormal, tangentU)};
}

double normalTargetVelocity(
    const MRContactConstraintGPU& contact,
    const ContactSolverConfig& config
) {
    const double penetration = std::min(
        static_cast<double>(contact.pointAndSeparation.w) +
            config.penetrationSlop,
        0.0
    );
    const double positional = std::min(
        config.maxDepenetrationVelocity,
        -config.errorReduction * penetration / config.timestep
    );
    double restitution = 0.0;
    if ((contact.flags & MR_CONSTRAINT_FLAG_NEW_IMPACT) != 0u &&
        contact.targetVelocityAndPreSolveNormal.w <
            -contact.response.y) {
        restitution =
            -static_cast<double>(contact.response.x) *
            contact.targetVelocityAndPreSolveNormal.w;
    }
    return std::max(positional, restitution) +
        dot(
            xyz(contact.targetVelocityAndPreSolveNormal),
            xyz(contact.normal)
        );
}

struct QualityMaximalProblem {
    DenseConicProblem problem;
    std::vector<std::uint32_t> dynamicBodies;
    std::vector<std::uint32_t> bodyToDynamic;
};

QualityMaximalProblem buildQualityMaximalProblem(
    const std::span<const MRBodyStateGPU> bodies,
    const std::span<const MRContactConstraintGPU> contacts,
    const RigidBodyWorldConfig& config
) {
    QualityMaximalProblem result;
    result.bodyToDynamic.assign(bodies.size(), MR_INVALID_INDEX);
    for (std::uint32_t body = 0u; body < bodies.size(); ++body) {
        if (bodies[body].flagsAndIndices[0] == MR_MOTION_DYNAMIC) {
            result.bodyToDynamic[body] =
                static_cast<std::uint32_t>(
                    result.dynamicBodies.size()
                );
            result.dynamicBodies.push_back(body);
        }
    }
    result.problem.nv = static_cast<std::uint32_t>(
        6u * result.dynamicBodies.size()
    );
    result.problem.inverseMass.assign(
        static_cast<std::size_t>(result.problem.nv) *
            result.problem.nv,
        0.0
    );
    result.problem.freeVelocity.assign(result.problem.nv, 0.0);

    for (std::size_t dynamic = 0u;
         dynamic < result.dynamicBodies.size();
         ++dynamic) {
        const MRBodyStateGPU& body =
            bodies[result.dynamicBodies[dynamic]];
        const std::size_t offset = 6u * dynamic;
        const double inverseMass =
            body.linearVelocityAndInverseMass.w;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            result.problem.inverseMass[
                (offset + axis) * result.problem.nv +
                offset + axis
            ] = inverseMass;
        }
        const Mat3 inverseInertia = matrix(
            body.inverseInertiaWorldRow0,
            body.inverseInertiaWorldRow1,
            body.inverseInertiaWorldRow2
        );
        for (std::size_t row = 0u; row < 3u; ++row) {
            for (std::size_t column = 0u; column < 3u; ++column) {
                result.problem.inverseMass[
                    (offset + 3u + row) * result.problem.nv +
                    offset + 3u + column
                ] = inverseInertia.m[row][column];
            }
        }
        const Vec3 linear = xyz(body.linearVelocityAndInverseMass);
        const Vec3 angular = xyz(body.angularVelocity);
        result.problem.freeVelocity[offset] = linear.x;
        result.problem.freeVelocity[offset + 1u] = linear.y;
        result.problem.freeVelocity[offset + 2u] = linear.z;
        result.problem.freeVelocity[offset + 3u] = angular.x;
        result.problem.freeVelocity[offset + 4u] = angular.y;
        result.problem.freeVelocity[offset + 5u] = angular.z;
    }

    result.problem.contacts.reserve(contacts.size());
    for (const MRContactConstraintGPU& contact : contacts) {
        const Vec3 normal =
            xyz(contact.normal) / norm(xyz(contact.normal));
        const auto [tangentU, tangentV] =
            contactBasis(normal, xyz(contact.tangent));
        const std::array<Vec3, 3u> directions{
            normal,
            tangentU,
            tangentV,
        };
        DenseContactBlock block;
        block.normalJacobian.assign(result.problem.nv, 0.0);
        block.tangentUJacobian.assign(result.problem.nv, 0.0);
        block.tangentVJacobian.assign(result.problem.nv, 0.0);
        const std::array<std::vector<double>*, 3u> rows{
            &block.normalJacobian,
            &block.tangentUJacobian,
            &block.tangentVJacobian,
        };
        const Vec3 point = xyz(contact.pointAndSeparation);
        const MRBodyStateGPU& bodyA = bodies[contact.bodyA];
        const MRBodyStateGPU& bodyB = bodies[contact.bodyB];

        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            const Vec3 direction = directions[axis];
            const auto addBody = [&](const std::uint32_t bodyIndex,
                                     const double sign) {
                const std::uint32_t dynamic =
                    result.bodyToDynamic[bodyIndex];
                if (dynamic == MR_INVALID_INDEX) {
                    return;
                }
                const std::size_t offset = 6u * dynamic;
                const Vec3 angularRow =
                    cross(
                        point - xyz(bodies[bodyIndex].position),
                        direction
                    ) * sign;
                (*rows[axis])[offset] += sign * direction.x;
                (*rows[axis])[offset + 1u] += sign * direction.y;
                (*rows[axis])[offset + 2u] += sign * direction.z;
                (*rows[axis])[offset + 3u] += angularRow.x;
                (*rows[axis])[offset + 4u] += angularRow.y;
                (*rows[axis])[offset + 5u] += angularRow.z;
            };
            addBody(contact.bodyA, -1.0);
            addBody(contact.bodyB, 1.0);

            Vec3 nonDynamicRelative{};
            if (bodyA.flagsAndIndices[0] != MR_MOTION_DYNAMIC) {
                nonDynamicRelative =
                    nonDynamicRelative - pointVelocity(bodyA, point);
            }
            if (bodyB.flagsAndIndices[0] != MR_MOTION_DYNAMIC) {
                nonDynamicRelative =
                    nonDynamicRelative + pointVelocity(bodyB, point);
            }
            const double surfaceTarget = dot(
                xyz(contact.targetVelocityAndPreSolveNormal),
                direction
            );
            const double target =
                axis == 0u
                ? normalTargetVelocity(contact, config.contact)
                : surfaceTarget;
            block.targetVelocity[axis] =
                target - dot(nonDynamicRelative, direction);
        }
        const double normalRegularization =
            static_cast<double>(contact.response.z) /
                (config.contact.timestep * config.contact.timestep) +
            config.qualityTangentialRegularization;
        block.regularization = {
            normalRegularization,
            config.qualityTangentialRegularization,
            config.qualityTangentialRegularization,
        };
        // Quality mode only accepts one Coulomb coefficient. Its caller
        // rejects distinct static/dynamic coefficients instead of silently
        // changing material semantics when the solver mode changes.
        block.friction = contact.friction.x;
        block.warmImpulse = {
            contact.impulses.x,
            contact.impulses.y,
            contact.impulses.z,
        };
        result.problem.contacts.push_back(std::move(block));
    }
    return result;
}

bool publishQualitySolution(
    const QualityMaximalProblem& problem,
    const QualityContactSolution& solution,
    const std::span<MRBodyStateGPU> bodies,
    const std::span<MRContactConstraintGPU> contacts
) {
    if (solution.velocity.size() != problem.problem.nv ||
        solution.impulses.size() != 3u * contacts.size()) {
        return false;
    }
    for (std::size_t dynamic = 0u;
         dynamic < problem.dynamicBodies.size();
         ++dynamic) {
        MRBodyStateGPU& body = bodies[problem.dynamicBodies[dynamic]];
        const std::size_t offset = 6u * dynamic;
        body.linearVelocityAndInverseMass = {
            static_cast<float>(solution.velocity[offset]),
            static_cast<float>(solution.velocity[offset + 1u]),
            static_cast<float>(solution.velocity[offset + 2u]),
            body.linearVelocityAndInverseMass.w,
        };
        body.angularVelocity = {
            static_cast<float>(solution.velocity[offset + 3u]),
            static_cast<float>(solution.velocity[offset + 4u]),
            static_cast<float>(solution.velocity[offset + 5u]),
            0.0f,
        };
        if (!finite4(body.linearVelocityAndInverseMass) ||
            !finite4(body.angularVelocity)) {
            return false;
        }
    }
    for (std::size_t contact = 0u;
         contact < contacts.size();
         ++contact) {
        contacts[contact].impulses = {
            static_cast<float>(solution.impulses[3u * contact]),
            static_cast<float>(solution.impulses[3u * contact + 1u]),
            static_cast<float>(solution.impulses[3u * contact + 2u]),
            0.0f,
        };
        if (!finite4(contacts[contact].impulses)) {
            return false;
        }
    }
    return true;
}

ContactSolverDiagnostics solveThroughputIslands(
    const std::span<MRBodyStateGPU> bodies,
    const std::span<MRContactConstraintGPU> contacts,
    const ContactSolverConfig& config
) {
    if (contacts.size() <= MR_MAX_CONTACTS_PER_SOLVER_BATCH) {
        return solveContactConstraints(bodies, contacts, config);
    }

    ContactSolverDiagnostics aggregate;
    const std::vector<ConstraintIsland> islands =
        buildConstraintIslands(bodies, contacts);
    aggregate.islandCount =
        static_cast<std::uint32_t>(islands.size());
    aggregate.requiredIslands = aggregate.islandCount;
    for (std::size_t islandIndex = 0u;
         islandIndex < islands.size();
         ++islandIndex) {
        const ConstraintIsland& island = islands[islandIndex];
        if (island.contacts.size() >
            MR_MAX_CONTACTS_PER_SOLVER_BATCH) {
            aggregate.code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
            aggregate.requiredContacts =
                static_cast<std::uint32_t>(
                    std::min<std::size_t>(
                        island.contacts.size(),
                        std::numeric_limits<std::uint32_t>::max()
                    )
                );
            return aggregate;
        }
        std::vector<MRContactConstraintGPU> batch;
        batch.reserve(island.contacts.size());
        for (const std::uint32_t contactIndex : island.contacts) {
            MRContactConstraintGPU contact = contacts[contactIndex];
            contact.islandIndex =
                static_cast<std::uint32_t>(islandIndex);
            batch.push_back(contact);
        }
        ContactSolverDiagnostics solved = solveContactConstraints(
            bodies,
            batch,
            config
        );
        if (!solved.succeeded()) {
            return solved;
        }
        for (std::size_t local = 0u; local < batch.size(); ++local) {
            contacts[island.contacts[local]] = batch[local];
        }
        if (solved.code == MR_STEP_FIXED_BUDGET_COMPLETE) {
            aggregate.code = MR_STEP_FIXED_BUDGET_COMPLETE;
        }
        aggregate.iterations =
            std::max(aggregate.iterations, solved.iterations);
        aggregate.activeContacts += solved.activeContacts;
        aggregate.maximumImpulseDelta = std::max(
            aggregate.maximumImpulseDelta,
            solved.maximumImpulseDelta
        );
        aggregate.maximumNormalResidual = std::max(
            aggregate.maximumNormalResidual,
            solved.maximumNormalResidual
        );
        aggregate.maximumConeViolation = std::max(
            aggregate.maximumConeViolation,
            solved.maximumConeViolation
        );
        aggregate.inverseLinearEffectiveMassSpread = std::max(
            aggregate.inverseLinearEffectiveMassSpread,
            solved.inverseLinearEffectiveMassSpread
        );
    }
    return aggregate;
}

} // namespace

ContactAssemblyResult assembleContactConstraints(
    const CollisionFrame& collision,
    const std::span<const MRShapeGPU> shapes,
    const std::span<const MRMaterialGPU> materials,
    const std::span<const MRBodyStateGPU> bodies,
    const std::uint32_t constraintCapacity
) {
    ContactAssemblyResult result;
    if (!collision.succeeded() ||
        collision.rawContacts.size() !=
            collision.rawContactPairIndices.size() ||
        collision.manifoldPoints.size() !=
            4u * collision.manifoldHeaders.size() ||
        !std::ranges::all_of(materials, finiteMaterial)) {
        result.diagnostics.code = MR_STEP_NONFINITE_INPUT;
        return result;
    }
    if (std::ranges::any_of(
            materials,
            [](const MRMaterialGPU& material) {
                return material.response.w != 0.0f ||
                    material.geometry.x != 0.0f ||
                    material.geometry.y != 0.0f ||
                    material.geometry.z != 0.0f ||
                    material.geometry.w != 0.0f;
            }
        )) {
        // Dissipation, material-level skin, and adhesion have not yet been
        // mapped into the common contact block. Reject them rather than
        // silently interpreting dissipation as the block's impulse cap.
        result.diagnostics.code = MR_STEP_UNSUPPORTED;
        return result;
    }
    std::uint64_t requiredConstraints = 0u;
    for (const MRManifoldHeaderGPU& header :
         collision.manifoldHeaders) {
        if (header.pairAndCount[3] == 0u ||
            header.pairAndCount[3] > 4u) {
            result.diagnostics.code = MR_STEP_NONFINITE_INPUT;
            return result;
        }
        requiredConstraints += header.pairAndCount[3];
    }
    result.diagnostics.requiredConstraints =
        static_cast<std::uint32_t>(
            std::min<std::uint64_t>(
                requiredConstraints,
                std::numeric_limits<std::uint32_t>::max()
            )
        );
    if (requiredConstraints > constraintCapacity) {
        result.diagnostics.code =
            MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW;
        return result;
    }

    result.constraints.reserve(
        static_cast<std::size_t>(requiredConstraints)
    );
    for (std::size_t manifold = 0u;
         manifold < collision.manifoldHeaders.size();
         ++manifold) {
        const MRManifoldHeaderGPU& header =
            collision.manifoldHeaders[manifold];
        const std::uint32_t colliderA = header.pairAndCount[1];
        const std::uint32_t colliderB = header.pairAndCount[2];
        if (colliderA >= shapes.size() ||
            colliderB >= shapes.size()) {
            result.constraints.clear();
            result.diagnostics.code = MR_STEP_NONFINITE_INPUT;
            return result;
        }
        const MRShapeGPU& shapeA = shapes[colliderA];
        const MRShapeGPU& shapeB = shapes[colliderB];
        if (shapeA.bodyIndex >= bodies.size() ||
            shapeB.bodyIndex >= bodies.size() ||
            shapeA.materialIndex >= materials.size() ||
            shapeB.materialIndex >= materials.size() ||
            (
                bodies[shapeA.bodyIndex].flagsAndIndices[0] !=
                    MR_MOTION_DYNAMIC &&
                bodies[shapeB.bodyIndex].flagsAndIndices[0] !=
                    MR_MOTION_DYNAMIC
            )) {
            result.constraints.clear();
            result.diagnostics.code = MR_STEP_UNSUPPORTED;
            return result;
        }
        const MRMaterialGPU& materialA =
            materials[shapeA.materialIndex];
        const MRMaterialGPU& materialB =
            materials[shapeB.materialIndex];
        const MRBodyStateGPU& bodyA = bodies[shapeA.bodyIndex];
        const MRBodyStateGPU& bodyB = bodies[shapeB.bodyIndex];
        const Mat3 rotationA =
            rotationMatrix(quaternion(bodyA.orientation));
        const Mat3 rotationB =
            rotationMatrix(quaternion(bodyB.orientation));
        const Vec3 normal = rotationA * xyz(header.normalAndAge);
        const double normalLength = norm(normal);
        if (!(normalLength > kTiny) ||
            !std::isfinite(normalLength)) {
            result.constraints.clear();
            result.diagnostics.code = MR_STEP_NONFINITE_INPUT;
            return result;
        }
        const Vec3 unitNormal = normal / normalLength;
        Vec3 tangent =
            rotationA * xyz(header.tangentAndMetric);
        tangent = tangent -
            unitNormal * dot(unitNormal, tangent);
        const double tangentLength = norm(tangent);
        tangent =
            tangentLength > kTiny &&
                    std::isfinite(tangentLength)
            ? tangent / tangentLength
            : contactBasis(unitNormal).first;
        for (std::uint32_t pointIndex = 0u;
             pointIndex < header.pairAndCount[3];
             ++pointIndex) {
            const MRManifoldPointGPU& manifoldPoint =
                collision.manifoldPoints[
                    manifold * 4u + pointIndex
                ];
            const Vec3 pointA =
                xyz(bodyA.position) +
                rotationA * xyz(manifoldPoint.localAnchorA);
            const Vec3 pointB =
                xyz(bodyB.position) +
                rotationB * xyz(manifoldPoint.localAnchorB);
            const Vec3 point = (pointA + pointB) * 0.5;
            const double geometricSeparation =
                dot(pointB - pointA, unitNormal);
            const double restSeparation =
                static_cast<double>(
                    shapeA.contactRestAndBoundingRadius.y
                ) +
                shapeB.contactRestAndBoundingRadius.y;
            const double effectiveSeparation =
                geometricSeparation - restSeparation;
            const Vec3 relative =
                pointVelocity(bodyB, point) -
                pointVelocity(bodyA, point);

            MRContactConstraintGPU constraint{};
            constraint.bodyA = shapeA.bodyIndex;
            constraint.bodyB = shapeB.bodyIndex;
            constraint.pairKey = pairKey(colliderA, colliderB);
            constraint.featureKey = featureKey(
                manifoldPoint.featureAndLife[0],
                manifoldPoint.featureAndLife[1]
            );
            constraint.pointAndSeparation =
                f4(point, static_cast<float>(effectiveSeparation));
            constraint.normal = f4(unitNormal);
            constraint.tangent = f4(tangent);
            constraint.friction = {
                static_cast<float>(geometricMean(
                    materialA.friction.x,
                    materialB.friction.x
                )),
                static_cast<float>(geometricMean(
                    materialA.friction.y,
                    materialB.friction.y
                )),
                static_cast<float>(geometricMean(
                    materialA.friction.z,
                    materialB.friction.z
                )),
                static_cast<float>(geometricMean(
                    materialA.friction.w,
                    materialB.friction.w
                )),
            };
            constraint.response = {
                std::max(materialA.response.x, materialB.response.x),
                std::max(materialA.response.y, materialB.response.y),
                materialA.response.z + materialB.response.z,
                0.0f,
            };
            constraint.targetVelocityAndPreSolveNormal = {
                0.0f,
                0.0f,
                0.0f,
                static_cast<float>(dot(relative, unitNormal)),
            };
            if (manifoldPoint.featureAndLife[2] == 0u) {
                constraint.flags |= MR_CONSTRAINT_FLAG_NEW_IMPACT;
                ++result.diagnostics.newImpactCount;
            }
            result.diagnostics.maximumPenetration = std::max(
                result.diagnostics.maximumPenetration,
                std::max(-effectiveSeparation, 0.0)
            );
            result.constraints.push_back(constraint);
        }
    }

    std::ranges::sort(
        result.constraints,
        [](const MRContactConstraintGPU& left,
           const MRContactConstraintGPU& right) {
            return std::tie(left.pairKey, left.featureKey) <
                std::tie(right.pairKey, right.featureKey);
        }
    );
    if (!constraintsSorted(result.constraints)) {
        result.constraints.clear();
        result.diagnostics.code = MR_STEP_NONFINITE_RESULT;
    }
    return result;
}

RigidBodyStepDiagnostics stepRigidBodyWorldCpu(
    const std::span<const MRBodyPropertiesGPU> properties,
    const std::span<MRBodyStateGPU> states,
    const std::span<const MRShapeGPU> shapes,
    const std::span<const MRMaterialGPU> materials,
    const std::span<const BodyWrench> wrenches,
    const RigidBodyWorldConfig& config,
    RigidBodyWorldCache& cache
) {
    RigidBodyStepDiagnostics diagnostics;
    if (properties.size() != states.size() ||
        config.freeMotion.timestep != config.contact.timestep ||
        !(config.qualityTangentialRegularization > 0.0) ||
        !std::isfinite(config.qualityTangentialRegularization) ||
        (
            config.solverType != MR_SOLVER_LEGACY_PROJECTED &&
            config.solverType != MR_SOLVER_QUALITY_NEWTON
        )) {
        diagnostics.code = MR_STEP_NONFINITE_INPUT;
        return diagnostics;
    }
    if (config.freeMotion.integrator !=
        FreeBodyIntegrator::symplecticEuler) {
        // A post-contact endpoint velocity is insufficient to reconstruct
        // the full implicit-midpoint position/orientation increment.
        diagnostics.code = MR_STEP_UNSUPPORTED;
        return diagnostics;
    }
    if (std::ranges::any_of(
            properties,
            [](const MRBodyPropertiesGPU& body) {
                return body.articulationIndex != MR_INVALID_INDEX;
            }
        )) {
        // This pipeline is maximal-coordinate. Articulated q/v must use the
        // articulated dynamics/contact path; integrating links independently
        // would silently destroy joint constraints.
        diagnostics.code = MR_STEP_UNSUPPORTED;
        return diagnostics;
    }

    std::vector<MRBodyStateGPU> working(states.begin(), states.end());
    RigidBodyWorldCache workingCache = cache;
    diagnostics.freeMotion = predictFreeBodyVelocities(
        properties,
        working,
        wrenches,
        config.freeMotion
    );
    if (!diagnostics.freeMotion.succeeded()) {
        diagnostics.code = diagnostics.freeMotion.code;
        return diagnostics;
    }
    diagnostics.collision = {};
    CollisionFrame collision = collideCpuReference(
        shapes,
        working,
        config.collision,
        workingCache.manifolds
    );
    diagnostics.collision = collision.diagnostics;
    if (!collision.succeeded()) {
        diagnostics.code = collision.diagnostics.code;
        return diagnostics;
    }

    ContactAssemblyResult assembly = assembleContactConstraints(
        collision,
        shapes,
        materials,
        working,
        config.constraintCapacity
    );
    diagnostics.assembly = assembly.diagnostics;
    diagnostics.contactCount =
        static_cast<std::uint32_t>(assembly.constraints.size());
    diagnostics.maximumPenetration =
        assembly.diagnostics.maximumPenetration;
    if (!assembly.diagnostics.succeeded()) {
        diagnostics.code = assembly.diagnostics.code;
        return diagnostics;
    }

    ++workingCache.step;
    workingCache.impulses.beginStep(workingCache.step);
    workingCache.impulses.seed(assembly.constraints);
    if (assembly.constraints.empty()) {
        diagnostics.solver.code = MR_STEP_SUCCESS;
    } else if (config.solverType == MR_SOLVER_QUALITY_NEWTON) {
        const bool unsupportedFriction =
            std::ranges::any_of(
                assembly.constraints,
                [](const MRContactConstraintGPU& contact) {
                    return contact.friction.x != contact.friction.y ||
                        contact.friction.z > 0.0f ||
                        contact.friction.w > 0.0f ||
                        contact.response.w > 0.0f;
                }
            );
        if (unsupportedFriction) {
            diagnostics.code = MR_STEP_UNSUPPORTED;
            return diagnostics;
        }
        const QualityMaximalProblem qualityProblem =
            buildQualityMaximalProblem(
                working,
                assembly.constraints,
                config
            );
        diagnostics.qualitySolver = solveQualityContactProblem(
            qualityProblem.problem,
            config.quality
        );
        if (!diagnostics.qualitySolver.converged() ||
            !publishQualitySolution(
                qualityProblem,
                diagnostics.qualitySolver,
                working,
                assembly.constraints
            )) {
            diagnostics.code =
                diagnostics.qualitySolver.converged()
                ? MR_STEP_NONFINITE_RESULT
                : diagnostics.qualitySolver.code;
            return diagnostics;
        }
    } else {
        diagnostics.solver = solveThroughputIslands(
            working,
            assembly.constraints,
            config.contact
        );
        if (!diagnostics.solver.succeeded()) {
            diagnostics.code = diagnostics.solver.code;
            return diagnostics;
        }
    }
    workingCache.impulses.commit(assembly.constraints);
    workingCache.impulses.prune(8u);

    const FreeBodyIntegratorDiagnostics configurationIntegration =
        integrateFreeBodyConfigurations(
            properties,
            working,
            config.freeMotion.timestep
        );
    if (!configurationIntegration.succeeded()) {
        diagnostics.code = configurationIntegration.code;
        return diagnostics;
    }

    std::ranges::copy(working, states.begin());
    cache = std::move(workingCache);
    diagnostics.code =
        config.solverType == MR_SOLVER_QUALITY_NEWTON
        ? diagnostics.qualitySolver.code
        : diagnostics.solver.code;
    return diagnostics;
}

} // namespace metalrobo
