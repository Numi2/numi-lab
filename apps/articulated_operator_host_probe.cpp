#include "metalrobo/G1.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
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
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

MRArticulatedPointImpulseGPU point(
    const std::uint32_t body,
    const mr_float4 local,
    const mr_float4 impulse
) {
    MRArticulatedPointImpulseGPU result{};
    result.bodyIndex = body;
    result.localPoint = local;
    result.worldImpulse = impulse;
    return result;
}

metalrobo::EngineModel duplicateModel(
    const metalrobo::EngineModel& source
) {
    metalrobo::EngineModel combined = source;
    const std::uint32_t articulationBase =
        static_cast<std::uint32_t>(
            combined.articulations.size()
        );
    const std::uint32_t bodyBase =
        static_cast<std::uint32_t>(combined.bodies.size());
    const std::uint32_t jointBase =
        static_cast<std::uint32_t>(combined.joints.size());
    const std::uint32_t qBase =
        static_cast<std::uint32_t>(combined.defaultQ.size());
    const std::uint32_t vBase =
        static_cast<std::uint32_t>(combined.defaultV.size());
    const std::uint32_t materialBase =
        static_cast<std::uint32_t>(combined.materials.size());

    for (MRArticulationGPU articulation :
         source.articulations) {
        articulation.rootBody += bodyBase;
        articulation.firstBody += bodyBase;
        articulation.firstJoint += jointBase;
        articulation.qOffset += qBase;
        articulation.vOffset += vBase;
        combined.articulations.push_back(articulation);
    }
    for (MRJointDescriptorGPU joint : source.joints) {
        joint.parentBody += bodyBase;
        joint.childBody += bodyBase;
        joint.qOffset += qBase;
        joint.vOffset += vBase;
        combined.joints.push_back(joint);
    }
    for (MRDofPropertiesGPU dof : source.dofs) {
        dof.articulationIndex += articulationBase;
        if (dof.jointIndex != MR_INVALID_INDEX) {
            dof.jointIndex += jointBase;
        }
        if (dof.qIndex != MR_INVALID_INDEX) {
            dof.qIndex += qBase;
        }
        dof.vIndex += vBase;
        combined.dofs.push_back(dof);
    }
    for (MRBodyPropertiesGPU body : source.bodies) {
        if (body.articulationIndex != MR_INVALID_INDEX) {
            body.articulationIndex += articulationBase;
        }
        if (body.parentBody != MR_INVALID_INDEX) {
            body.parentBody += bodyBase;
        }
        if (body.inboundJoint != MR_INVALID_INDEX) {
            body.inboundJoint += jointBase;
        }
        combined.bodies.push_back(body);
    }
    for (const MRMaterialGPU material : source.materials) {
        combined.materials.push_back(material);
    }
    for (MRShapeGPU shape : source.shapes) {
        shape.bodyIndex += bodyBase;
        shape.materialIndex += materialBase;
        combined.shapes.push_back(shape);
    }
    combined.defaultQ.insert(
        combined.defaultQ.end(),
        source.defaultQ.begin(),
        source.defaultQ.end()
    );
    combined.defaultV.insert(
        combined.defaultV.end(),
        source.defaultV.begin(),
        source.defaultV.end()
    );
    combined.world.bodyCount =
        static_cast<mr_u32>(combined.bodies.size());
    combined.world.articulationCount =
        static_cast<mr_u32>(combined.articulations.size());
    combined.world.jointCount =
        static_cast<mr_u32>(combined.joints.size());
    combined.world.shapeCount =
        static_cast<mr_u32>(combined.shapes.size());
    combined.world.materialCount =
        static_cast<mr_u32>(combined.materials.size());
    combined.world.nq =
        static_cast<mr_u32>(combined.defaultQ.size());
    combined.world.nv =
        static_cast<mr_u32>(combined.defaultV.size());
    combined.name = source.name + "_duplicated";
    return combined;
}

template <typename T>
bool byteEqual(
    const std::vector<T>& left,
    const std::vector<T>& right
) {
    return left.size() == right.size() &&
        (left.empty() ||
         std::memcmp(
             left.data(),
             right.data(),
             left.size() * sizeof(T)
        ) == 0);
}

bool sameDispatch(
    const MRArticulatedOperatorDispatchGPU& left,
    const MRArticulatedOperatorDispatchGPU& right
) {
    return left.articulationIndex == right.articulationIndex &&
        left.environmentCount == right.environmentCount &&
        left.pointCount == right.pointCount &&
        left.flags == right.flags &&
        left.qStride == right.qStride &&
        left.pointStride == right.pointStride &&
        left.bodyPoseStride == right.bodyPoseStride &&
        left.pointWorldStride == right.pointWorldStride &&
        left.massMatrixStride == right.massMatrixStride &&
        left.pointJacobianStride ==
            right.pointJacobianStride &&
        left.generalizedStride == right.generalizedStride &&
        left.reserved0 == right.reserved0;
}

bool sameLayout(
    const metalrobo::MetalArticulatedOperatorLayout& left,
    const metalrobo::MetalArticulatedOperatorLayout& right
) {
    return sameDispatch(left.dispatch, right.dispatch) &&
        left.qElements == right.qElements &&
        left.qBytes == right.qBytes &&
        left.pointElements == right.pointElements &&
        left.pointBytes == right.pointBytes &&
        left.bodyPoseElements == right.bodyPoseElements &&
        left.bodyPoseBytes == right.bodyPoseBytes &&
        left.pointWorldElements == right.pointWorldElements &&
        left.pointWorldBytes == right.pointWorldBytes &&
        left.massMatrixElements == right.massMatrixElements &&
        left.massMatrixBytes == right.massMatrixBytes &&
        left.pointJacobianElements ==
            right.pointJacobianElements &&
        left.pointJacobianBytes == right.pointJacobianBytes &&
        left.generalizedElements == right.generalizedElements &&
        left.generalizedBytes == right.generalizedBytes &&
        left.statusElements == right.statusElements &&
        left.statusBytes == right.statusBytes &&
        left.totalAllocatedBytes == right.totalAllocatedBytes;
}

bool byteEqual(
    const metalrobo::MetalArticulatedOperatorResult& left,
    const metalrobo::MetalArticulatedOperatorResult& right
) {
    return sameLayout(left.layout, right.layout) &&
        byteEqual(left.bodyPoses, right.bodyPoses) &&
        byteEqual(left.pointWorld, right.pointWorld) &&
        byteEqual(
            left.diagnosticMassMatrix,
            right.diagnosticMassMatrix
        ) &&
        byteEqual(
            left.pointJacobians,
            right.pointJacobians
        ) &&
        byteEqual(
            left.generalizedImpulse,
            right.generalizedImpulse
        ) &&
        byteEqual(left.deltaVelocity, right.deltaVelocity) &&
        byteEqual(left.statuses, right.statuses);
}

std::string replayDifference(
    const metalrobo::MetalArticulatedOperatorResult& left,
    const metalrobo::MetalArticulatedOperatorResult& right
) {
    if (!sameLayout(left.layout, right.layout)) {
        return "layout";
    }
    if (!byteEqual(left.bodyPoses, right.bodyPoses)) {
        return "body poses";
    }
    if (!byteEqual(left.pointWorld, right.pointWorld)) {
        return "point world";
    }
    if (!byteEqual(
            left.diagnosticMassMatrix,
            right.diagnosticMassMatrix
        )) {
        return "diagnostic mass";
    }
    if (!byteEqual(left.pointJacobians, right.pointJacobians)) {
        return "point Jacobians";
    }
    if (!byteEqual(
            left.generalizedImpulse,
            right.generalizedImpulse
        )) {
        return "generalized impulse";
    }
    if (!byteEqual(left.deltaVelocity, right.deltaVelocity)) {
        return "delta velocity";
    }
    if (!byteEqual(left.statuses, right.statuses)) {
        return "statuses";
    }
    return {};
}

metalrobo::MetalArticulatedOperatorResult sentinelResult() {
    metalrobo::MetalArticulatedOperatorResult result{};
    std::memset(&result.layout, 0x5a, sizeof(result.layout));
    result.bodyPoses.resize(1u);
    result.pointWorld.resize(2u);
    result.diagnosticMassMatrix = {91.0f, -7.0f};
    result.pointJacobians = {3.0f};
    result.generalizedImpulse = {-11.0f, 4.0f};
    result.deltaVelocity = {8.0f};
    result.statuses.resize(1u);
    std::memset(
        result.bodyPoses.data(),
        0xa5,
        result.bodyPoses.size() *
            sizeof(MRArticulatedBodyPoseGPU)
    );
    std::memset(
        result.pointWorld.data(),
        0x3c,
        result.pointWorld.size() *
            sizeof(MRArticulatedPointWorldGPU)
    );
    std::memset(
        result.statuses.data(),
        0xc3,
        result.statuses.size() *
            sizeof(MRArticulatedOperatorStatusGPU)
    );
    return result;
}

void requirePredispatchRejection(
    const metalrobo::MetalArticulatedOperatorDiagnostics& diagnostics,
    const metalrobo::MetalArticulatedOperatorHostStatus expected,
    const metalrobo::MetalArticulatedOperatorResult& result,
    const metalrobo::MetalArticulatedOperatorResult& sentinel,
    const std::string& label
) {
    require(
        diagnostics.status == expected &&
            !diagnostics.dispatched &&
            !diagnostics.published &&
            byteEqual(result, sentinel),
        label + " was not rejected transactionally before dispatch"
    );
}

} // namespace

int main() {
    try {
        using metalrobo::MetalArticulatedOperatorConfig;
        using metalrobo::MetalArticulatedOperatorHostStatus;
        using metalrobo::MetalArticulatedOperatorInput;
        using metalrobo::MetalArticulatedOperatorResult;

        const metalrobo::EngineModel g1 =
            metalrobo::makeUnitreeG1EngineModel();
        std::string reason;
        require(
            g1.valid(&reason),
            "canonical G1 is invalid: " + reason
        );
        const MRArticulationGPU& articulation =
            g1.articulations[0];
        constexpr std::size_t environmentCount = 2u;
        constexpr std::size_t pointCount = 2u;

        std::vector<float> q(
            environmentCount * articulation.nq
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            std::copy(
                g1.defaultQ.begin() + articulation.qOffset,
                g1.defaultQ.begin() +
                    articulation.qOffset + articulation.nq,
                q.begin() + environment * articulation.nq
            );
        }
        q[articulation.nq + 7u] += 0.015f;

        std::vector<MRArticulatedPointImpulseGPU> points{
            point(6u, f4(0.02f, 0.01f, -0.03f),
                  f4(1.0f, -0.5f, 0.25f)),
            point(12u, f4(-0.01f, 0.03f, 0.02f),
                  f4(-0.2f, 0.6f, 0.9f)),
            point(6u, f4(0.01f, -0.02f, -0.04f),
                  f4(0.4f, 0.7f, -0.3f)),
            point(12u, f4(0.03f, 0.01f, 0.01f),
                  f4(-0.8f, 0.2f, 0.5f)),
        };

        MetalArticulatedOperatorInput input{
            .articulationIndex = 0u,
            .environmentCount = environmentCount,
            .pointCount = pointCount,
            .q = q,
            .points = points,
        };
        MetalArticulatedOperatorConfig config{
            .writeDiagnosticMassMatrix = true,
        };
        MetalArticulatedOperatorResult result;
        const auto diagnostics =
            metalrobo::runMetalArticulatedOperator(
                g1,
                input,
                result,
                config
            );
        require(
            diagnostics.succeeded() &&
                diagnostics.dispatched &&
                diagnostics.published &&
                diagnostics.successfulEnvironmentCount ==
                    environmentCount &&
                diagnostics.failedEnvironmentCount == 0u,
            std::string("valid G1 host dispatch failed: ") +
                metalrobo::metalArticulatedOperatorHostStatusName(
                    diagnostics.status
                ) +
                " " + diagnostics.message
        );
        const auto& layout = result.layout;
        require(
            layout.dispatch.qStride == articulation.nq &&
                layout.dispatch.pointStride == pointCount &&
                layout.dispatch.bodyPoseStride ==
                    articulation.bodyCount &&
                layout.dispatch.massMatrixStride ==
                    articulation.nv * articulation.nv &&
                layout.dispatch.pointJacobianStride ==
                    pointCount * 3u * articulation.nv &&
                layout.dispatch.generalizedStride ==
                    articulation.nv &&
                layout.qElements ==
                    environmentCount * articulation.nq &&
                layout.pointElements ==
                    environmentCount * pointCount &&
                layout.bodyPoseElements ==
                    environmentCount * articulation.bodyCount &&
                layout.massMatrixElements ==
                    environmentCount *
                    articulation.nv * articulation.nv &&
                layout.pointJacobianElements ==
                    environmentCount * pointCount * 3u *
                    articulation.nv &&
                layout.generalizedElements ==
                    environmentCount * articulation.nv &&
                layout.statusElements == environmentCount &&
                layout.totalAllocatedBytes > 0u,
            "public host layout does not match compact G1 dimensions"
        );
        require(
            result.bodyPoses.size() ==
                    layout.bodyPoseElements &&
                result.pointWorld.size() ==
                    layout.pointWorldElements &&
                result.diagnosticMassMatrix.size() ==
                    layout.massMatrixElements &&
                result.pointJacobians.size() ==
                    layout.pointJacobianElements &&
                result.generalizedImpulse.size() ==
                    layout.generalizedElements &&
                result.deltaVelocity.size() ==
                    layout.generalizedElements &&
                result.statuses.size() ==
                    layout.statusElements &&
                std::all_of(
                    result.statuses.begin(),
                    result.statuses.end(),
                    [](const MRArticulatedOperatorStatusGPU& status) {
                        return status.code ==
                            MR_ARTICULATED_OPERATOR_SUCCESS;
                    }
                ),
            "typed output sizes or statuses are inconsistent"
        );

        MetalArticulatedOperatorResult replay;
        const auto replayDiagnostics =
            metalrobo::runMetalArticulatedOperator(
                g1,
                input,
                replay,
                config
            );
        require(
            replayDiagnostics.succeeded(),
            "public host replay dispatch failed"
        );
        const std::string replayMismatch =
            replayDifference(result, replay);
        require(
            replayMismatch.empty(),
            "public host replay is not bitwise deterministic: " +
                replayMismatch
        );

        const metalrobo::EngineModel offsetModel =
            duplicateModel(g1);
        require(
            offsetModel.valid(&reason) &&
                offsetModel.articulations.size() == 2u,
            "duplicated offset model is invalid: " + reason
        );
        const MRArticulationGPU& offsetArticulation =
            offsetModel.articulations[1];
        require(
            offsetArticulation.firstBody != 0u &&
                offsetArticulation.firstJoint != 0u &&
                offsetArticulation.qOffset != 0u &&
                offsetArticulation.vOffset != 0u,
            "offset canary did not shift every global stream"
        );
        std::vector<float> singleQ(
            g1.defaultQ.begin(),
            g1.defaultQ.end()
        );
        std::vector<float> offsetQ(
            offsetModel.defaultQ.begin() +
                offsetArticulation.qOffset,
            offsetModel.defaultQ.begin() +
                offsetArticulation.qOffset +
                offsetArticulation.nq
        );
        std::vector<MRArticulatedPointImpulseGPU> singlePoint{
            point(
                6u,
                f4(0.02f, 0.01f, -0.03f),
                f4(1.0f, -0.5f, 0.25f)
            ),
        };
        std::vector<MRArticulatedPointImpulseGPU> offsetPoint =
            singlePoint;
        offsetPoint[0].bodyIndex +=
            offsetArticulation.firstBody;
        const MetalArticulatedOperatorInput singleInput{
            .articulationIndex = 0u,
            .environmentCount = 1u,
            .pointCount = 1u,
            .q = singleQ,
            .points = singlePoint,
        };
        const MetalArticulatedOperatorInput offsetInput{
            .articulationIndex = 1u,
            .environmentCount = 1u,
            .pointCount = 1u,
            .q = offsetQ,
            .points = offsetPoint,
        };
        MetalArticulatedOperatorResult singleResult;
        MetalArticulatedOperatorResult offsetResult;
        const auto singleDiagnostics =
            metalrobo::runMetalArticulatedOperator(
                g1,
                singleInput,
                singleResult,
                config
            );
        const auto offsetDiagnostics =
            metalrobo::runMetalArticulatedOperator(
                offsetModel,
                offsetInput,
                offsetResult,
                config
            );
        require(
            singleDiagnostics.succeeded() &&
                offsetDiagnostics.succeeded() &&
                byteEqual(
                    singleResult.bodyPoses,
                    offsetResult.bodyPoses
                ) &&
                byteEqual(
                    singleResult.pointWorld,
                    offsetResult.pointWorld
                ) &&
                byteEqual(
                    singleResult.diagnosticMassMatrix,
                    offsetResult.diagnosticMassMatrix
                ) &&
                byteEqual(
                    singleResult.pointJacobians,
                    offsetResult.pointJacobians
                ) &&
                byteEqual(
                    singleResult.generalizedImpulse,
                    offsetResult.generalizedImpulse
                ) &&
                byteEqual(
                    singleResult.deltaVelocity,
                    offsetResult.deltaVelocity
                ),
            "nonzero articulation/body/joint/q/v offsets changed "
            "the Metal operator"
        );

        const MetalArticulatedOperatorResult sentinel =
            sentinelResult();
        MetalArticulatedOperatorResult rejected = sentinel;
        MetalArticulatedOperatorInput undersizedQ = input;
        undersizedQ.q = std::span<const float>(
            q.data(),
            q.size() - 1u
        );
        auto rejectedDiagnostics =
            metalrobo::runMetalArticulatedOperator(
                g1,
                undersizedQ,
                rejected,
                config
            );
        requirePredispatchRejection(
            rejectedDiagnostics,
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            rejected,
            sentinel,
            "undersized q layout"
        );

        rejected = sentinel;
        MetalArticulatedOperatorInput undersizedPoints = input;
        undersizedPoints.points =
            std::span<const MRArticulatedPointImpulseGPU>(
                points.data(),
                points.size() - 1u
            );
        rejectedDiagnostics =
            metalrobo::runMetalArticulatedOperator(
                g1,
                undersizedPoints,
                rejected,
                config
            );
        requirePredispatchRejection(
            rejectedDiagnostics,
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            rejected,
            sentinel,
            "undersized point layout"
        );

        rejected = sentinel;
        MetalArticulatedOperatorInput overflow = input;
        overflow.environmentCount =
            std::numeric_limits<std::size_t>::max();
        rejectedDiagnostics =
            metalrobo::runMetalArticulatedOperator(
                g1,
                overflow,
                rejected,
                config
            );
        requirePredispatchRejection(
            rejectedDiagnostics,
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            rejected,
            sentinel,
            "environment-count overflow"
        );

        rejected = sentinel;
        MetalArticulatedOperatorInput shaderAddressOverflow =
            input;
        shaderAddressOverflow.environmentCount =
            std::numeric_limits<std::uint32_t>::max();
        rejectedDiagnostics =
            metalrobo::runMetalArticulatedOperator(
                g1,
                shaderAddressOverflow,
                rejected,
                config
            );
        requirePredispatchRejection(
            rejectedDiagnostics,
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            rejected,
            sentinel,
            "shader element-address overflow"
        );

        rejected = sentinel;
        MetalArticulatedOperatorInput overCapacity = input;
        overCapacity.pointCount =
            MR_ARTICULATED_OPERATOR_MAX_POINTS + 1u;
        rejectedDiagnostics =
            metalrobo::runMetalArticulatedOperator(
                g1,
                overCapacity,
                rejected,
                config
            );
        requirePredispatchRejection(
            rejectedDiagnostics,
            MetalArticulatedOperatorHostStatus::capacityOverflow,
            rejected,
            sentinel,
            "point capacity overflow"
        );

        rejected = sentinel;
        metalrobo::EngineModel invalidModel = g1;
        --invalidModel.world.nv;
        rejectedDiagnostics =
            metalrobo::runMetalArticulatedOperator(
                invalidModel,
                input,
                rejected,
                config
            );
        requirePredispatchRejection(
            rejectedDiagnostics,
            MetalArticulatedOperatorHostStatus::invalidModel,
            rejected,
            sentinel,
            "invalid canonical model"
        );

        rejected = sentinel;
        std::vector<MRArticulatedPointImpulseGPU> invalidPoints =
            points;
        invalidPoints[0].bodyIndex = MR_INVALID_INDEX;
        MetalArticulatedOperatorInput invalidPointInput = input;
        invalidPointInput.points = invalidPoints;
        rejectedDiagnostics =
            metalrobo::runMetalArticulatedOperator(
                g1,
                invalidPointInput,
                rejected,
                config
            );
        requirePredispatchRejection(
            rejectedDiagnostics,
            MetalArticulatedOperatorHostStatus::invalidPointQuery,
            rejected,
            sentinel,
            "invalid point query"
        );

        std::vector<float> zeroPointQ(
            g1.defaultQ.begin() + articulation.qOffset,
            g1.defaultQ.begin() +
                articulation.qOffset + articulation.nq
        );
        MetalArticulatedOperatorInput zeroPointInput{
            .articulationIndex = 0u,
            .environmentCount = 1u,
            .pointCount = 0u,
            .q = zeroPointQ,
            .points = {},
        };
        MetalArticulatedOperatorResult zeroPointResult;
        const auto zeroPointDiagnostics =
            metalrobo::runMetalArticulatedOperator(
                g1,
                zeroPointInput,
                zeroPointResult
            );
        require(
            zeroPointDiagnostics.succeeded() &&
                zeroPointResult.pointWorld.empty() &&
                zeroPointResult.pointJacobians.empty() &&
                zeroPointResult.diagnosticMassMatrix.empty() &&
                zeroPointResult.generalizedImpulse.size() ==
                    articulation.nv &&
                zeroPointResult.deltaVelocity.size() ==
                    articulation.nv,
            "logically empty Metal buffers were not bound safely"
        );

        const metalrobo::EngineModel freeModel =
            metalrobo::makeFreeSphereEngineModel();
        std::vector<float> derivedOverflowQ =
            freeModel.defaultQ;
        derivedOverflowQ[0] =
            std::numeric_limits<float>::max();
        std::vector<MRArticulatedPointImpulseGPU>
            derivedOverflowPoint{
                point(
                    freeModel.articulations[0].rootBody,
                    f4(
                        std::numeric_limits<float>::max(),
                        0.0f,
                        0.0f
                    ),
                    f4(0.0f, 0.0f, 0.0f)
                ),
            };
        MetalArticulatedOperatorInput derivedOverflowInput{
            .articulationIndex = 0u,
            .environmentCount = 1u,
            .pointCount = 1u,
            .q = derivedOverflowQ,
            .points = derivedOverflowPoint,
        };
        MetalArticulatedOperatorResult gpuRejected;
        const auto gpuRejectedDiagnostics =
            metalrobo::runMetalArticulatedOperator(
                freeModel,
                derivedOverflowInput,
                gpuRejected
            );
        require(
            gpuRejectedDiagnostics.status ==
                MetalArticulatedOperatorHostStatus::
                    gpuEnvironmentFailure &&
                gpuRejectedDiagnostics.dispatched &&
                gpuRejectedDiagnostics.published &&
                gpuRejectedDiagnostics.failedEnvironmentCount ==
                    1u &&
                gpuRejected.statuses.size() == 1u &&
                gpuRejected.statuses[0].code ==
                    MR_ARTICULATED_OPERATOR_NONFINITE_RESULT,
            "completed GPU rejection did not publish typed status"
        );

        std::cout
            << std::setprecision(8)
            << "articulated_operator_host=metal"
            << " device=\"" << diagnostics.deviceName << '"'
            << " environments=" << environmentCount
            << " q_elements=" << layout.qElements
            << " point_elements=" << layout.pointElements
            << " mass_elements=" << layout.massMatrixElements
            << " jacobian_elements="
            << layout.pointJacobianElements
            << " allocated_bytes="
            << layout.totalAllocatedBytes
            << " elapsed_ms="
            << diagnostics.elapsedMilliseconds
            << " replay=bitwise"
            << " offset_articulation=pass"
            << " predispatch_canaries=7"
            << " empty_buffers=pass"
            << " gpu_status_publication=pass"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "articulated_operator_host status=failed reason="
            << exception.what() << '\n';
        return 1;
    }
}
