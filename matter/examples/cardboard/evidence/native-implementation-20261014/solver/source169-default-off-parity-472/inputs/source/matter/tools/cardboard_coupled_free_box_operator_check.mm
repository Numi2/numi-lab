// Native coupled FEM/free-BOX operator fixture; no physical time steps.
#define main preserved_free_box_angular_main
#include "cardboard_free_box_angular_native_check.mm"
#undef main

#include "numi/matter/matter.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace numi::matter;

constexpr double kDt = 0.01;
constexpr unsigned kFEMNodes = 4;
constexpr unsigned kRigidDofs = 6;
constexpr unsigned kUnknowns = 3 * kFEMNodes + kRigidDofs;
constexpr double kCharacteristicLength = 0.05;

struct CoupledDevice {
    Device base;
    id<MTLComputePipelineState> femOperator = nil;
    id<MTLComputePipelineState> gatherNodes = nil;
    explicit CoupledDevice() {
        femOperator = base.makePipeline(
            @"numi_matter_metal::nm_fem_apply_operator_elements");
        gatherNodes = base.makePipeline(
            @"numi_matter_metal::nm_fgmres_gather_nodes");
    }
};

struct CoupledContactResources {
    id<MTLBuffer> argumentBuffer = nil;
    id<MTLBuffer> pair = nil;
    id<MTLBuffer> sample = nil;
    id<MTLBuffer> rigid = nil;
    id<MTLBuffer> nodeIncidence = nil;
    id<MTLBuffer> nodeRanges = nil;
    id<MTLBuffer> activePairs = nil;
    id<MTLBuffer> activeSlots = nil;
    id<MTLBuffer> activeCounts = nil;
    id<MTLBuffer> deformable = nil;
    id<MTLBuffer> deformableIncidence = nil;
    id<MTLBuffer> deformableRanges = nil;
    id<MTLBuffer> mpmToActive = nil;
    id<MTLBuffer> mpmCounts = nil;
    id<MTLBuffer> deformableCounts = nil;
    id<MTLBuffer> proxy = nil;
};

struct CoupledState {
    ProductionContact::State contactSource{};
    std::array<double, kUnknowns> candidate{};
    std::array<double, kUnknowns> direction{};
};

struct CoupledFixture {
    CoupledDevice& device;
    ProductionContact contactOracle;
    CompiledWorld world;
    NMMatterDispatchGPU dispatch{};
    NMBridgeDispatchGPU bridge{};
    NMMicrostepGPU microstep{};
    NMMixedSolverGPU solver{};
    NMRigidProxyGPU proxy{};
    NMContactPairGPU pair{};
    id<MTLArgumentEncoder> contactEncoder = nil;
    V3 localPoint{0.0608, 0.014, 0.003};
    std::array<V3, kFEMNodes> rest{};
    ProductionContact::State baseContact{};

    explicit CoupledFixture(CoupledDevice& d)
        : device(d), contactOracle(d.base) {
        const ContactCase pointCase{"coupled_face_contact", localPoint};
        baseContact = contactOracle.baseline(pointCase);
        // Use a small but nonzero source angular velocity and a small closing
        // velocity. This keeps the smooth free-BOX SO(3) path active while the
        // contact sample remains on the strict inertial-barrier branch.
        baseContact.body.angularVelocity = mf4({0.012, -0.009, 0.007});
        baseContact.body.linearVelocityAndInverseMass =
            mf4({0.00015, -0.0001, 0.0002}, 0.4);
        baseContact.rigid = contactOracle.candidateRigid(baseContact, {});
        const V3 bodyCenter = v3(baseContact.body.position);
        const Q4 qsource = q4(baseContact.body.orientation);
        const Q4 qcandidate = integrate(qsource,
            v3(baseContact.body.angularVelocity), kDt);
        const Q4 proxyQ = qmul(qcandidate,
            q4(contactOracle.proxySource().localOrientation));
        const V3 proxyCenter = add(bodyCenter,
            multiply(rotation(qcandidate),
                     v3(contactOracle.proxySource().localCenterAndRadius)));
        const V3 node0 = add(proxyCenter, multiply(rotation(proxyQ), localPoint));

        rest = {node0, add(node0, V3{0.02, 0.0, 0.0}),
                add(node0, V3{0.0, 0.02, 0.0}),
                add(node0, V3{0.0, 0.0, 0.02})};
        const auto parsed = parseMatter(R"(
            material coupled_operator_neo_hookean {
                parameter density : kg/m^3 = 1000;
                parameter mu : Pa = 30000;
                parameter lambda : Pa = 50000;
                model neo_hookean;
                energy = neo_hookean(mu, lambda);
                valid = J() - 0.2;
                supports fem;
            }
        )");
        need(parsed.succeeded(), "coupled fixture material parse failed");
        MaterialProgram material = parsed.material;
        material.staticFriction = 0.0;
        material.dynamicFriction = 0.0;

        WorldSource source;
        source.environmentCount = 1u;
        source.frameTimestep = kDt;
        source.gravity = {0.0, 0.0, 0.0};
        source.contactSlop = 0.012;
        source.materials = {material};
        ObjectSource object;
        object.name = "one_tet_for_full_fem_free_box_operator";
        object.representation = Representation::fem;
        object.materialIndex = 0u;
        object.mixedFEM = false;
        object.twoWayCoupling = false;
        object.adaptive = false;
        object.deformableContact = false;
        object.deformableSelfContact = false;
        object.characteristicLength = 0.02;
        for (const V3& x : rest) object.femNodes.push_back(x);
        object.tetrahedra = {{{0u, 1u, 2u, 3u}}};
        source.objects = {object};
        auto compiled = compileWorld(source,
            {.maximumRateExponent = 0u, .emitSpecializedMetal = false});
        need(compiled.succeeded(), "coupled fixture compile failed");
        world = std::move(compiled.world);
        need(world.fem.nodes.size() == kFEMNodes &&
                world.fem.tetrahedra.size() == 1u,
                "fixture did not compile to one complete tetrahedron");

        dispatch = world.dispatch;
        dispatch.environmentCount = 1u;
        dispatch.rigidProxyCount = 1u;
        dispatch.contactPairCount = 1u;
        dispatch.rigidGeneralizedCapacity = kRigidDofs;
        dispatch.reservedMixed1 = 1u;
        dispatch.gridNodeCount = 0u;
        dispatch.mpmActiveNodeCapacity = 0u;
        dispatch.deformableContactCapacity = 0u;
        dispatch.maximumRateExponent = 0u;
        dispatch.gravityAndTimestep = {0.0f, 0.0f, 0.0f, float(kDt)};
        dispatch.numericalLimits.x = float(contactOracle.thickness());
        bridge.environmentCount = 1u;
        bridge.rigidProxyCount = 1u;
        bridge.currentBodyCount = 1u;
        bridge.currentBodyStride = 1u;
        bridge.time = {float(1.0 / kDt), float(kDt), 0.0f, 0.0f};
        microstep.microtickCount = 1u;
        microstep.microtick = 0u;
        microstep.flags = NM_MICROSTEP_FGMRES_OPERATOR;
        microstep.time = {float(kDt), float(1.0 / kDt), 0.0f, float(kDt)};
        solver.contactAcceptance.x = 0.001f;
        proxy = contactOracle.proxySource();
        pair.continuumNode = 0u; // FEM node zero; gridNodeCount is zero.
        pair.rigidProxy = 0u;
        pair.objectIndex = 0u;
        pair.materialInterface = 0u;
        id<MTLFunction> layout = [device.base.library
            newFunctionWithName:@"numi_matter_metal::nm_primal_contact_argument_layout"];
        need(layout != nil, "contact argument layout function missing");
        contactEncoder = [layout newArgumentEncoderWithBufferIndex:0u];
        need(contactEncoder != nil, "cannot make contact argument encoder");
        baseContact.node = world.fem.nodes[0];
        baseContact.node.velocityAndInverseMass =
            {0.0f, 0.0f, 0.0f, world.fem.nodes[0].velocityAndInverseMass.w};
        baseContact.node.positionAndMass = world.fem.nodes[0].positionAndMass;
        baseContact.boxLocalPoint = localPoint;
    }

    std::vector<NMFEMNodeStateGPU> nodesFor(
        const std::array<double, kUnknowns>& x) const {
        std::vector<NMFEMNodeStateGPU> nodes = world.fem.nodes;
        for (unsigned n = 0; n < kFEMNodes; ++n) {
            const unsigned b = 3u * n;
            const V3 dv{float(x[b]), float(x[b + 1u]), float(x[b + 2u])};
            auto& node = nodes[n];
            const V3 restPoint = v3(world.fem.nodes[n].positionAndMass);
            const V3 oldVelocity = v3(world.fem.nodes[n].velocityAndInverseMass);
            const V3 velocity = add(oldVelocity, dv);
            const V3 position = add(restPoint, mul(velocity, kDt));
            node.positionAndMass.x = float(position[0]);
            node.positionAndMass.y = float(position[1]);
            node.positionAndMass.z = float(position[2]);
            node.velocityAndInverseMass.x = float(velocity[0]);
            node.velocityAndInverseMass.y = float(velocity[1]);
            node.velocityAndInverseMass.z = float(velocity[2]);
            node.deltaVelocity = {float(dv[0]), float(dv[1]), float(dv[2]), 0.0f};
        }
        return nodes;
    }

    std::array<double, 6> rigidCandidate(
        const std::array<double, kUnknowns>& x) const {
        std::array<double, 6> out{};
        for (unsigned i = 0; i < 6u; ++i) out[i] = x[3u * kFEMNodes + i];
        return out;
    }

    CoupledContactResources contactResources(
        id<MTLBuffer> sampleB, id<MTLBuffer> rigidB) const {
        CoupledContactResources r;
        r.pair = makeBuffer(device.base.gpu, std::vector<NMContactPairGPU>{pair});
        r.sample = sampleB;
        r.rigid = rigidB;
        std::vector<uint> incidence{0u};
        std::vector<NMIncidenceRangeGPU> ranges(
            dispatch.gridNodeCount + dispatch.femNodeCount);
        ranges[0] = {0u, 1u, 0u, 0u};
        r.nodeIncidence = makeBuffer(device.base.gpu, incidence);
        r.nodeRanges = makeBuffer(device.base.gpu, ranges);
        r.activePairs = makeBuffer(device.base.gpu, std::vector<uint>{0u});
        r.activeSlots = makeBuffer(device.base.gpu, std::vector<uint>{0u});
        r.activeCounts = makeBuffer(device.base.gpu, std::vector<uint>{1u});
        r.deformable = makeBuffer(device.base.gpu,
            std::vector<NMDeformableContactGPU>(1u));
        r.deformableIncidence = makeBuffer(device.base.gpu, std::vector<uint>{0u});
        r.deformableRanges = makeBuffer(device.base.gpu, ranges);
        r.mpmToActive = makeBuffer(device.base.gpu,
            std::vector<uint>{NM_INVALID_INDEX});
        r.mpmCounts = makeBuffer(device.base.gpu, std::vector<uint>{0u});
        r.deformableCounts = makeBuffer(device.base.gpu, std::vector<uint>{0u});
        r.argumentBuffer = [device.base.gpu
            newBufferWithLength:contactEncoder.encodedLength
                        options:MTLResourceStorageModeShared];
        need(r.argumentBuffer != nil, "contact argument buffer allocation failed");
        [contactEncoder setArgumentBuffer:r.argumentBuffer offset:0u];
        id<MTLBuffer> args[15] = {
            r.pair, r.sample, r.nodeIncidence, r.nodeRanges,
            r.activePairs, r.activeSlots, r.activeCounts,
            r.deformable, r.deformableIncidence, r.deformableRanges,
            r.mpmToActive, r.mpmCounts,
            makeBuffer(device.base.gpu, std::vector<NMRigidProxyGPU>{proxy}),
            rigidB, r.deformableCounts};
        for (NSUInteger i = 0u; i < 15u; ++i)
            [contactEncoder setBuffer:args[i] offset:0u atIndex:i];
        // The proxy buffer is retained through the argument encoder; keep an
        // owned reference on the resource structure's companion lifetime.
        r.proxy = args[12];
        return r;
    }

    struct SampleResult { id<MTLBuffer> sample=nil, rigid=nil, status=nil; };

    SampleResult evaluateSample(
        const std::array<double, kUnknowns>& x,
        const std::vector<NMFEMNodeStateGPU>& nodes) {
        const auto rigid = contactOracle.candidateRigid(
            baseContact, rigidCandidate(x));
        auto candidateRigidB = makeBuffer(device.base.gpu,
            std::vector<NMRigidStateGPU>{rigid});
        auto statusesB = makeBuffer(device.base.gpu,
            std::vector<NMMatterStatusGPU>(1u));
        auto sampleB = makeBuffer(device.base.gpu,
            std::vector<NMContactSampleGPU>(1u));
        const auto dispatchB = makeBuffer(device.base.gpu, dispatch);
        const auto microB = makeBuffer(device.base.gpu, microstep);
        const auto bridgeB = makeBuffer(device.base.gpu, bridge);
        const auto objectB = makeBuffer(device.base.gpu, world.objects);
        const auto materialB = makeBuffer(device.base.gpu, world.materials);
        const auto gridB = makeBuffer(device.base.gpu,
            std::vector<NMGridNodeStateGPU>(1u));
        const auto nodesB = makeBuffer(device.base.gpu, nodes);
        const auto proxyB = makeBuffer(device.base.gpu,
            std::vector<NMRigidProxyGPU>{proxy});
        const auto pairsB = makeBuffer(device.base.gpu,
            std::vector<NMContactPairGPU>{pair});
        const auto schedulerB = makeBuffer(device.base.gpu,
            std::vector<NMSchedulerStateGPU>{world.schedulers[0]});
        const auto adaptiveB = makeBuffer(device.base.gpu,
            std::vector<NMAdaptiveStateGPU>{world.adaptive[0]});
        const auto generationsB = makeBuffer(device.base.gpu, std::vector<uint>{0u});
        const auto historyB = makeBuffer(device.base.gpu, std::vector<nm_float4>(1u));
        const auto bodyB = makeBuffer(device.base.gpu,
            std::vector<MRBodyStateGPU>{baseContact.body});
        const auto puncturesB = makeBuffer(device.base.gpu,
            std::vector<NMPunctureChannelGPU>(1u));
        const auto candidateIdsB = makeBuffer(device.base.gpu, std::vector<uint>{0u});
        const auto candidateCountsB = makeBuffer(device.base.gpu, std::vector<uint>{1u});
        const uint candidateCapacity = 1u;
        const uint compact = 0u;
        auto command = [device.base.queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        need(command != nil && encoder != nil, "contact evaluation encoder failed");
        [encoder setComputePipelineState:device.base.evaluate];
        id<MTLBuffer> args[22] = {dispatchB, microB, objectB, materialB,
            gridB, nodesB, proxyB, candidateRigidB, pairsB, schedulerB,
            adaptiveB, sampleB, generationsB, statusesB, historyB, bridgeB,
            bodyB, puncturesB, candidateIdsB, candidateCountsB,
            makeBuffer(device.base.gpu, candidateCapacity), makeBuffer(device.base.gpu, compact)};
        for (NSUInteger i = 0u; i < 22u; ++i)
            [encoder setBuffer:args[i] offset:0u atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        need(command.status == MTLCommandBufferStatusCompleted,
                "contact evaluation command failed");
        const auto status = *static_cast<const NMMatterStatusGPU*>(statusesB.contents);
        need(status.code == NM_STATUS_SUCCESS,
                "contact sample status=" + std::to_string(status.code));
        return {sampleB,candidateRigidB,statusesB};
    }

    std::array<double, kUnknowns> nativeAction(
        const std::array<double, kUnknowns>& x,
        const std::array<double, kUnknowns>& direction) {
        const auto nodes = nodesFor(x);
        const auto evaluated = evaluateSample(x, nodes);
        const auto sampleB = evaluated.sample;
        const auto rigidB = evaluated.rigid;
        const auto contact = contactResources(sampleB, rigidB);
        const uint femCount = dispatch.femNodeCount;
        const uint rigidBase = 2u * dispatch.environmentCount * femCount +
            dispatch.environmentCount * dispatch.mpmActiveNodeCapacity;
        const uint vectorCount = rigidBase + dispatch.rigidGeneralizedCapacity;
        std::vector<nm_float4> vector(vectorCount), work(vectorCount);
        for (unsigned n = 0u; n < kFEMNodes; ++n) {
            vector[n] = {float(direction[3u*n]), float(direction[3u*n+1u]),
                         float(direction[3u*n+2u]), 0.0f};
        }
        for (unsigned d = 0u; d < 6u; ++d)
            vector[rigidBase+d] = {float(direction[3u*kFEMNodes+d]), 0,0,0};
        const auto vectorB = makeBuffer(device.base.gpu, vector);
        const auto workB = makeBuffer(device.base.gpu, work);
        const auto nodeB = makeBuffer(device.base.gpu, nodes);
        const auto tetraB = makeBuffer(device.base.gpu, world.fem.tetrahedra);
        std::vector<float> params;
        for (const auto& p : world.parameters) params.push_back(p.valueAndBounds.x);
        const auto paramsB = makeBuffer(device.base.gpu, params);
        const auto programsB = makeBuffer(device.base.gpu, world.scalarPrograms);
        const auto instructionsB = makeBuffer(device.base.gpu, world.instructions);
        const auto stateB = makeBuffer(device.base.gpu,
            std::vector<float>{0.0f});
        const auto schedulersB = makeBuffer(device.base.gpu, world.schedulers);
        const auto adaptiveB = makeBuffer(device.base.gpu, world.adaptive);
        const auto fieldsB = makeBuffer(device.base.gpu, world.fem.fields);
        const auto mixedB = makeBuffer(device.base.gpu, world.mixedMaterials);
        const auto learnedB = makeBuffer(device.base.gpu, world.learnedMaterials);
        const auto layersB = makeBuffer(device.base.gpu, world.learnedLayers);
        const auto weightsB = makeBuffer(device.base.gpu, world.learnedWeights);
        const auto fgmresB = makeBuffer(device.base.gpu,
            std::vector<NMFGMRESStateGPU>(1u));
        const auto statusB = makeBuffer(device.base.gpu,
            std::vector<NMMatterStatusGPU>(1u));
        const auto outputB = makeBuffer(device.base.gpu,
            std::vector<NMFEMElementVectorGPU>(1u));
        const auto cacheB = makeBuffer(device.base.gpu, std::vector<float>{0.0f});
        const auto cacheValidB = makeBuffer(device.base.gpu, std::vector<uint>{0u});
        const auto fallbackMetricsB = makeBuffer(device.base.gpu,
            std::vector<nm_float4>(world.dispatch.environmentCount, nm_float4{}));
        const uint useCache = 0u;
        auto cmd = [device.base.queue commandBuffer];
        auto enc = [cmd computeCommandEncoder];
        need(cmd != nil && enc != nil, "FEM operator encoder failed");
        [enc setComputePipelineState:device.femOperator];
        [enc setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [enc setBytes:&microstep length:sizeof(microstep) atIndex:1u];
        id<MTLBuffer> femArgs[27] = {};
        femArgs[2] = makeBuffer(device.base.gpu, world.objects);
        femArgs[3] = makeBuffer(device.base.gpu, world.materials);
        femArgs[4] = programsB; femArgs[5] = instructionsB;
        femArgs[6] = paramsB; femArgs[7] = nodeB; femArgs[8] = tetraB;
        femArgs[9] = stateB; femArgs[10] = vectorB;
        femArgs[11] = schedulersB; femArgs[12] = adaptiveB;
        femArgs[13] = outputB; femArgs[14] = statusB;
        femArgs[16] = mixedB; femArgs[17] = fieldsB;
        femArgs[18] = learnedB; femArgs[19] = layersB;
        femArgs[20] = weightsB;
        femArgs[21] = makeBuffer(device.base.gpu, world.mixedSolver);
        femArgs[22] = fgmresB; femArgs[23] = cacheB;
        femArgs[24] = cacheValidB; femArgs[25] = makeBuffer(device.base.gpu, useCache);
        femArgs[26] = fallbackMetricsB;
        for (NSUInteger i = 2u; i < 27u; ++i)
            if (i != 15u) [enc setBuffer:femArgs[i] offset:0u atIndex:i];
        [enc dispatchThreads:MTLSizeMake(1u,1u,1u)
              threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        need(cmd.status == MTLCommandBufferStatusCompleted,
                "FEM operator command failed");
        const auto status = *static_cast<const NMMatterStatusGPU*>(statusB.contents);
        need(status.code == NM_STATUS_SUCCESS,
                "FEM operator status=" + std::to_string(status.code));

        const auto elementB = outputB;
        cmd = [device.base.queue commandBuffer]; enc = [cmd computeCommandEncoder];
        need(cmd != nil && enc != nil, "FEM gather encoder failed");
        [enc setComputePipelineState:device.gatherNodes];
        [enc setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [enc setBytes:&microstep length:sizeof(microstep) atIndex:1u];
        [enc setBuffer:femArgs[2] offset:0u atIndex:2u];
        [enc setBuffer:nodeB offset:0u atIndex:3u];
        [enc setBuffer:tetraB offset:0u atIndex:4u];
        [enc setBuffer:elementB offset:0u atIndex:5u];
        [enc setBuffer:makeBuffer(device.base.gpu, world.fem.nodeIncidence)
                 offset:0u atIndex:6u];
        [enc setBuffer:makeBuffer(device.base.gpu, world.fem.nodeRanges)
                 offset:0u atIndex:7u];
        [enc setBuffer:schedulersB offset:0u atIndex:8u];
        [enc setBuffer:adaptiveB offset:0u atIndex:9u];
        [enc setBuffer:vectorB offset:0u atIndex:10u];
        [enc setBuffer:workB offset:0u atIndex:11u];
        [enc setBuffer:contact.argumentBuffer offset:0u atIndex:12u];
        [enc setBuffer:fgmresB offset:0u atIndex:13u];
        const uint articulatedNv = 0u;
        [enc setBytes:&articulatedNv length:sizeof(articulatedNv) atIndex:14u];
        [enc setBuffer:makeBuffer(device.base.gpu, std::vector<float>{0.0f})
                 offset:0u atIndex:15u];
        [enc dispatchThreads:MTLSizeMake(femCount,1u,1u)
              threadsPerThreadgroup:MTLSizeMake(4u,1u,1u)];
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        need(cmd.status == MTLCommandBufferStatusCompleted,
                "FEM/contact gather command failed");

        const auto candidateB = makeBuffer(device.base.gpu,
            std::vector<float>{float(x[12]), float(x[13]), float(x[14]),
                               float(x[15]), float(x[16]), float(x[17])});
        cmd = [device.base.queue commandBuffer]; enc = [cmd computeCommandEncoder];
        need(cmd != nil && enc != nil, "free rigid operator encoder failed");
        [enc setComputePipelineState:device.base.applyFreeRigid];
        [enc setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        const uint articulated = 0u;
        const uint allowConvergedState = 0u;
        [enc setBytes:&articulated length:sizeof(articulated) atIndex:1u];
        [enc setBuffer:vectorB offset:0u atIndex:2u];
        [enc setBuffer:workB offset:0u atIndex:3u];
        [enc setBuffer:contact.argumentBuffer offset:0u atIndex:4u];
        [enc setBuffer:fgmresB offset:0u atIndex:5u];
        [enc setBuffer:candidateB offset:0u atIndex:6u];
        [enc setBytes:&allowConvergedState length:sizeof(allowConvergedState)
               atIndex:7u];
        [enc dispatchThreads:MTLSizeMake(6u,1u,1u)
              threadsPerThreadgroup:MTLSizeMake(6u,1u,1u)];
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        need(cmd.status == MTLCommandBufferStatusCompleted,
                "free rigid operator command failed");

        cmd = [device.base.queue commandBuffer]; enc = [cmd computeCommandEncoder];
        need(cmd != nil && enc != nil, "rigid contact operator encoder failed");
        [enc setComputePipelineState:device.base.apply];
        [enc setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [enc setBytes:&articulated length:sizeof(articulated) atIndex:1u];
        [enc setBuffer:vectorB offset:0u atIndex:2u];
        [enc setBuffer:workB offset:0u atIndex:3u];
        [enc setBuffer:contact.argumentBuffer offset:0u atIndex:4u];
        [enc setBuffer:fgmresB offset:0u atIndex:5u];
        [enc setBuffer:makeBuffer(device.base.gpu,
            std::vector<float>(18u,0.0f)) offset:0u atIndex:6u];
        [enc setBytes:&allowConvergedState length:sizeof(allowConvergedState)
               atIndex:7u];
        [enc dispatchThreads:MTLSizeMake(6u,1u,1u)
              threadsPerThreadgroup:MTLSizeMake(6u,1u,1u)];
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        need(cmd.status == MTLCommandBufferStatusCompleted,
                "rigid contact operator command failed");
        const auto* rows = static_cast<const nm_float4*>(workB.contents);
        std::array<double, kUnknowns> out{};
        for (unsigned n = 0u; n < kFEMNodes; ++n)
            for (unsigned c = 0u; c < 3u; ++c)
                out[3u*n+c] = (&rows[n].x)[c];
        for (unsigned c = 0u; c < 6u; ++c)
            out[12u+c] = rows[rigidBase+c].x;
        return out;
    }

};

} // namespace

namespace {

M3 matrixFromTet(const NMTetrahedronGPU& tet) {
    return {tet.inverseRestRow0.x, tet.inverseRestRow0.y, tet.inverseRestRow0.z,
            tet.inverseRestRow1.x, tet.inverseRestRow1.y, tet.inverseRestRow1.z,
            tet.inverseRestRow2.x, tet.inverseRestRow2.y, tet.inverseRestRow2.z};
}

M3 deformationFromPositions(const std::array<V3,4>& x, const M3& invRest) {
    const M3 ds{ x[1][0]-x[0][0], x[2][0]-x[0][0], x[3][0]-x[0][0],
                 x[1][1]-x[0][1], x[2][1]-x[0][1], x[3][1]-x[0][1],
                 x[1][2]-x[0][2], x[2][2]-x[0][2], x[3][2]-x[0][2] };
    return multiply(ds, invRest);
}

M3 neoHookeanFirstPiola(const M3& f, double mu, double lambda) {
    const double j = f[0]*(f[4]*f[8]-f[5]*f[7]) -
                     f[1]*(f[3]*f[8]-f[5]*f[6]) +
                     f[2]*(f[3]*f[7]-f[4]*f[6]);
    need(j > 0.0 && std::isfinite(j), "FP64 oracle deformation has nonpositive J");
    const M3 invT = transpose(inverse(f));
    const double logJ = std::log(j);
    M3 p{};
    for (unsigned i=0;i<9u;++i)
        p[i] = mu*(f[i]-invT[i]) + lambda*logJ*invT[i];
    return p;
}

std::array<V3,4> neoHookeanNodalForces(const M3& p,
                                      const NMTetrahedronGPU& tet) {
    const M3 invRest = matrixFromTet(tet);
    M3 a = multiply(p, transpose(invRest));
    const double volume = tet.inverseRestRow0.w;
    for (double& v : a) v *= -volume;
    std::array<V3,4> force{};
    for (unsigned component=0;component<3u;++component) {
        force[1][component] = a[3u*component+0u];
        force[2][component] = a[3u*component+1u];
        force[3][component] = a[3u*component+2u];
        force[0][component] = -force[1][component]-force[2][component]-force[3][component];
    }
    return force;
}

using R18 = std::array<double,kUnknowns>;

// The inherited angular BOX oracle keeps its production state in FP32 structs.
// This variant keeps the perturbed FEM point and velocity in FP64 all the way
// through the contact residual, avoiding a false derivative floor from
// rounding an epsilon perturbation back into the uploaded node struct.
FP64ContactProbe fp64ContinuousBoxContactAt(
    const ProductionContact& scene,
    const ProductionContact::State& state,
    const std::array<double,6>& candidate,
    const V3& point,
    const V3& nodeVelocity) {
    const FP64BoxCandidate rigid =
        fp64ProjectBoxCandidate(scene, state, candidate);
    const M3 proxyRotation = rotation(rigid.proxyOrientation);
    const M3 proxyRotationT = transpose(proxyRotation);
    const V3 local = multiply(proxyRotationT, sub(point, rigid.center));
    V3 closest{}, signedOutside{}, outside{};
    uint32_t feature = 0u;
    bool featureMargin = true;
    const double featureScale = std::max({maxAbs3(point),
        maxAbs3(rigid.bodyPosition), maxAbs3(rigid.half),
        scene.thickness(), 1.0e-12});
    const double featureTolerance = 64.0 *
        double(std::numeric_limits<float>::epsilon()) * featureScale;
    for (int axis = 0; axis < 3; ++axis) {
        signedOutside[axis] = std::abs(local[axis]) - rigid.half[axis];
        closest[axis] = std::clamp(local[axis], -rigid.half[axis], rigid.half[axis]);
        outside[axis] = local[axis] - closest[axis];
        if (signedOutside[axis] > 0.0) feature |= 1u << axis;
        if (std::abs(signedOutside[axis]) <= featureTolerance)
            featureMargin = false;
    }
    const double separation = norm(outside);
    if (!(separation > 1.0e-12) || !std::isfinite(separation)) return {};
    const V3 normal = rotateQ(rigid.proxyOrientation,
        mul(outside, 1.0 / separation));
    const V3 witness = add(rigid.center,
        rotateQ(rigid.proxyOrientation, closest));
    const V3 lever = sub(witness, rigid.bodyPosition);
    const V3 rigidPointVelocity = add(rigid.linearVelocity,
        cross(rigid.angularVelocity, sub(witness, rigid.bodyPosition)));
    const double normalVelocity = dot(sub(nodeVelocity, rigidPointVelocity), normal);
    const V3 angularArm = cross(lever, normal);
    const double inverseResponse = scene.nodeInverseMass(state) +
        rigid.bodyInverseMass + dot(angularArm,
            multiply(rigid.inverseInertiaWorld, angularArm));
    const double dt = scene.timestep();
    const double thickness = scene.thickness();
    const double distance = std::max(separation, 1.0e-4 * thickness);
    const double distance2 = distance * distance;
    const double thickness2 = thickness * thickness;
    const double offset = distance2 - thickness2;
    const double logarithm = std::log(distance2 / thickness2);
    const double remaining = std::max(thickness - distance, 1.0e-3 * thickness);
    const double closing = std::max(-normalVelocity, 0.0);
    const double inertialDenominator = inverseResponse * dt * dt * thickness2;
    const double impactDenominator = inverseResponse * dt * remaining * thickness2;
    const double inertial = 10.0 / std::max(inertialDenominator, 1.0e-20);
    const double impact = closing / std::max(impactDenominator, 1.0e-20);
    const double stiffness = std::max(inertial, impact);
    const double first = -stiffness *
        (2.0 * offset * logarithm + offset * offset / distance2);
    const double second = -stiffness *
        (2.0 * logarithm + 4.0 * offset / distance2 -
         offset * offset / (distance2 * distance2));
    const double radialRaw = 4.0 * distance2 * second + 2.0 * first;
    const double radialProjected = separation < 1.0e-4 * thickness
        ? 0.0 : std::max(radialRaw, 0.0);
    const double force = -2.0 * distance * first;
    const double lambda = std::max(force, 0.0) * dt;
    const bool strictInertial = std::isfinite(inertial) &&
        std::isfinite(impact) && inertial > 0.0 &&
        impact < inertial * (1.0 -
            32.0 * double(std::numeric_limits<float>::epsilon()));
    const bool barrierMargin = separation > 1.0e-4 * thickness &&
        separation < thickness && remaining > 1.0e-3 * thickness &&
        inertialDenominator > 1.0e-20 && impactDenominator > 1.0e-20 &&
        force > 0.0 && lambda > 0.0 && inverseResponse > 1.0e-12;
    OracleSample sample{};
    sample.witness = witness;
    sample.normal = normal;
    sample.impulse = mul(normal, -lambda);
    sample.torque = cross(lever, sample.impulse);
    sample.separation = separation;
    sample.normalVelocity = normalVelocity;
    sample.lambda = lambda;
    sample.inverseResponse = inverseResponse;
    sample.stiffness = stiffness;
    sample.forceN = force;
    sample.firstSquared = first;
    sample.secondSquared = second;
    sample.radialCurvatureRaw = radialRaw;
    sample.radialCurvatureProjected = radialProjected;
    sample.stiffnessBranch = strictInertial ? 0 : (impact > inertial ? 1 : -1);
    sample.feature = feature;
    const bool supported = featureMargin && barrierMargin && strictInertial &&
        std::isfinite(norm(sample.impulse)) && std::isfinite(norm(sample.torque));
    return {sample, supported, featureMargin, barrierMargin, strictInertial};
}

FP64ContactProbe fp64ContactAtState(const CoupledFixture& fixture,
                                    const R18& x) {
    const V3 rest = v3(fixture.world.fem.nodes[0].positionAndMass);
    const V3 oldVelocity = v3(fixture.world.fem.nodes[0].velocityAndInverseMass);
    const V3 delta{x[0], x[1], x[2]};
    const V3 nodeVelocity = add(oldVelocity, delta);
    const V3 point = add(rest, mul(nodeVelocity, kDt));
    return fp64ContinuousBoxContactAt(fixture.contactOracle,
        fixture.baseContact, fixture.rigidCandidate(x), point, nodeVelocity);
}

R18 fp64CompleteResidual(const CoupledFixture& fixture, const R18& x) {
    R18 residual{};
    const double dt = kDt;
    std::array<V3,4> positions{};
    for (unsigned n=0;n<4u;++n) {
        const V3 rest = v3(fixture.world.fem.nodes[n].positionAndMass);
        const V3 oldVelocity = v3(fixture.world.fem.nodes[n].velocityAndInverseMass);
        const V3 delta{x[3u*n],x[3u*n+1u],x[3u*n+2u]};
        positions[n] = add(rest,mul(add(oldVelocity,delta),dt));
    }
    const auto& tet = fixture.world.fem.tetrahedra[0];
    const M3 f = deformationFromPositions(positions,matrixFromTet(tet));
    const auto p = neoHookeanFirstPiola(f,
        fixture.world.materials[0].bulk.z,
        fixture.world.materials[0].bulk.w);
    const auto forces = neoHookeanNodalForces(p,tet);
    for (unsigned n=0;n<4u;++n) {
        const double mass = fixture.world.fem.nodes[n].positionAndMass.w;
        for (unsigned c=0;c<3u;++c)
            residual[3u*n+c] = dt*forces[n][c] - mass*x[3u*n+c];
    }

    const auto contact = fp64ContactAtState(fixture, x);
    need(contact.sample.separation > 0.0 &&
         std::isfinite(contact.sample.lambda),
         "FP64 contact residual is unsupported at a perturbed state");
    const auto impulse = contact.sample.impulse;
    // The authored contact sample stores impulse ON the rigid body. The FEM
    // residual receives its opposite; the six rigid residual rows receive the
    // impulse and its COM torque with the production signs.
    for (unsigned c=0;c<3u;++c) {
        residual[c] -= impulse[c];
        residual[12u+c] -= x[12u+c] /
            fixture.baseContact.body.linearVelocityAndInverseMass.w;
        residual[12u+c] += impulse[c];
        residual[15u+c] += contact.sample.torque[c];
    }
    const V3 angularDelta{x[15],x[16],x[17]};
    const V3 angularInertiaResidual = fp64FullInertiaResidual(
        fixture.baseContact.body,angularDelta,dt);
    for (unsigned c=0;c<3u;++c) residual[15u+c] += angularInertiaResidual[c];
    return residual;
}

double scaledNorm(const R18& x) {
    double total=0.0;
    for (unsigned i=0;i<x.size();++i) {
        const double v = i>=15u ? x[i]/kCharacteristicLength : x[i];
        total += v*v;
    }
    return std::sqrt(total);
}

double blockNorm(const R18& x, unsigned begin, unsigned end) {
    double total=0.0;
    for (unsigned i=begin;i<end;++i) {
        const double v = i>=15u ? x[i]/kCharacteristicLength : x[i];
        total += v*v;
    }
    return std::sqrt(total);
}

R18 difference(const R18& a,const R18& b) {
    R18 out{};for(unsigned i=0;i<out.size();++i)out[i]=a[i]-b[i];return out;
}

struct FDResult {
    R18 action{};
    std::array<double,4> refinement{};
    bool branchStable = true;
    bool valid = true;
};

FDResult fp64Directional(const CoupledFixture& fixture,
                         const R18& center,
                         const R18& direction,
                         const unsigned epsilonScale) {
    constexpr std::array<double,4> epsilons{1.0e-3,3.0e-4,1.0e-4,3.0e-5};
    FDResult result;
    std::array<R18,4> derivatives{};
    std::array<uint32_t,4> plusFeatures{},minusFeatures{};
    std::array<int,4> plusBranches{},minusBranches{};
    for (unsigned i=0;i<epsilons.size();++i) {
        const double epsilon=epsilons[i]*epsilonScale;
        R18 plus=center,minus=center;
        for(unsigned j=0;j<center.size();++j){
            plus[j]+=epsilon*float(direction[j]);
            minus[j]-=epsilon*float(direction[j]);
        }
        const auto rp=fp64CompleteResidual(fixture,plus);
        const auto rm=fp64CompleteResidual(fixture,minus);
        for(unsigned j=0;j<center.size();++j)
            derivatives[i][j]=-(rp[j]-rm[j])/(2.0*epsilon);
        const auto plusProbe=fp64ContactAtState(fixture, plus);
        const auto minusProbe=fp64ContactAtState(fixture, minus);
        plusFeatures[i]=plusProbe.sample.feature;minusFeatures[i]=minusProbe.sample.feature;
        plusBranches[i]=plusProbe.sample.stiffnessBranch;minusBranches[i]=minusProbe.sample.stiffnessBranch;
        result.branchStable=result.branchStable&&plusProbe.supportedSmoothBranch&&
            minusProbe.supportedSmoothBranch;
    }
    for(unsigned i=1;i<epsilons.size();++i) {
        const R18 delta=difference(derivatives[i],derivatives[i-1]);
        result.refinement[i-1]=scaledNorm(delta);
    }
    result.action=derivatives.back();
    for(unsigned i=1;i<epsilons.size();++i)
        result.branchStable=result.branchStable&&
            plusFeatures[i]==plusFeatures[0]&&minusFeatures[i]==minusFeatures[0]&&
            plusBranches[i]==plusBranches[0]&&minusBranches[i]==minusBranches[0];
    result.valid=result.branchStable&&std::isfinite(scaledNorm(result.action));
    return result;
}

R18 nativeDirection(CoupledFixture& fixture, const R18& center, const R18& direction) {
    const auto raw=fixture.nativeAction(center,direction);
    return raw;
}

void runCoupledCheck(CoupledDevice& device) {
    CoupledFixture fixture(device);
    R18 center{};
    // Small, nonzero deformation and free-body rotation at the linearization
    // point. These remain inside one BOX face feature and one barrier branch.
    center[0]=0.0001; center[1]=-0.00005;
    center[3]=0.006; center[4]=0.0004;
    center[6]=-0.0003; center[7]=0.005;
    center[9]=0.0002; center[11]=0.004;
    center[12]=0.00015; center[13]=-0.00008; center[14]=0.00011;
    center[15]=0.018; center[16]=-0.012; center[17]=0.009;
    for(double& v:center)v=float(v);

    auto centerNodes=fixture.nodesFor(center);
    const auto centerSample=fixture.evaluateSample(center,centerNodes);
    const auto centerSampleB=centerSample.sample;
    const NMContactSampleGPU nativeSample=*static_cast<const NMContactSampleGPU*>(centerSampleB.contents);
    need((nativeSample.identity.w & NM_CONTACT_VALID)!=0u,
         "center production BOX sample is not valid");
    need(nativeSample.barrierHessianRow0.w>0.0f,
         "center sample did not select the free-BOX angular derivative path");
    const auto fp64Center=fp64ContactAtState(fixture, center);
    need(fp64Center.supportedSmoothBranch,
         "center FP64 contact sample is not on the supported smooth branch");
    const V3 nativeNormal{nativeSample.normalAndVelocity.x,
                          nativeSample.normalAndVelocity.y,
                          nativeSample.normalAndVelocity.z};
    const V3 oracleNormal = fp64Center.sample.normal;
    need(norm(sub(nativeNormal, oracleNormal)) <= 2.0e-5,
         "native and FP64 BOX contact normals differ");

    std::vector<std::pair<std::string,R18>> tests;
    R18 femOnly{}; femOnly[0]=0.2; femOnly[1]=-0.1; femOnly[3]=0.1;
    femOnly[6]=-0.08; femOnly[8]=0.05; femOnly[11]=0.07;
    tests.push_back({"fem_only_including_contact_node",femOnly});
    R18 rigidTranslation{}; rigidTranslation[12]=0.15; rigidTranslation[13]=-0.07; rigidTranslation[14]=0.04;
    tests.push_back({"rigid_translation_only",rigidTranslation});
    R18 rigidRotation{}; rigidRotation[15]=0.35; rigidRotation[16]=-0.25; rigidRotation[17]=0.2;
    tests.push_back({"rigid_rotation_only",rigidRotation});
    R18 mixed=femOnly; for(unsigned i=12u;i<18u;++i)mixed[i]=rigidTranslation[i]+rigidRotation[i];
    tests.push_back({"mixed_fem_and_free_box",mixed});

    bool allPass=true,allResolved=true;
    double worstRelative=0.0,worstAbsolute=0.0,worstFEM=0.0,worstRigid=0.0;
    for(const auto& [name,direction]:tests) {
        const R18 native=nativeDirection(fixture,center,direction);
        const FDResult reference=fp64Directional(fixture,center,direction,1u);
        if(!reference.valid){allResolved=false;allPass=false;
            std::cout<<"coupled_operator name="<<name<<" result=INCONCLUSIVE branch_or_fp64_reference_unresolved\n";
            continue;}
        const R18 error=difference(native,reference.action);
        const double n=scaledNorm(native),r=scaledNorm(reference.action),e=scaledNorm(error);
        const double fErr=blockNorm(error,0u,12u),rErr=blockNorm(error,12u,18u);
        const double rel=e/std::max(r,1.0e-30);
        worstRelative=std::max(worstRelative,rel);worstAbsolute=std::max(worstAbsolute,e);
        worstFEM=std::max(worstFEM,fErr);worstRigid=std::max(worstRigid,rErr);
        // FP64 target must demonstrate a converged center derivative, and the
        // native operator must meet 1e-3 in full scaled norm and each block.
        const bool fp64Resolved=reference.refinement[2] <=
            1.0e-6*std::max(r,1.0e-30);
        const bool pass=fp64Resolved && rel<=1.0e-3 &&
            fErr<=1.0e-3*std::max(blockNorm(reference.action,0u,12u),1.0e-30) &&
            rErr<=1.0e-3*std::max(blockNorm(reference.action,12u,18u),1.0e-30);
        allResolved=allResolved&&fp64Resolved;
        allPass=allPass&&pass;
        std::cout<<std::setprecision(10)
            <<"coupled_operator name="<<name
            <<" sign=A_minus_dR dt="<<kDt
            <<" native_scaled_l2="<<n<<" fp64_scaled_l2="<<r
            <<" relative="<<rel<<" fem_block_abs="<<fErr
            <<" rigid_block_abs="<<rErr
            <<" refinement_abs="<<reference.refinement[0]<<","<<reference.refinement[1]<<","<<reference.refinement[2]
            <<" feature="<<fp64Center.sample.feature
            <<" stiffness_branch="<<fp64Center.sample.stiffnessBranch
            <<" node_contact_impulse="<<nativeSample.impulseAndNormal.x<<","<<nativeSample.impulseAndNormal.y<<","<<nativeSample.impulseAndNormal.z
            <<" rigid_contact_torque="<<nativeSample.angularImpulseAndTangent.x<<","<<nativeSample.angularImpulseAndTangent.y<<","<<nativeSample.angularImpulseAndTangent.z
            <<" outcome="<<(pass?"PASS":(fp64Resolved?"FAIL":"INCONCLUSIVE"))<<"\n";
    }
    std::cout<<"coupled_fem_free_box_operator directions="<<tests.size()
        <<" worst_scaled_relative="<<worstRelative
        <<" worst_scaled_abs="<<worstAbsolute
        <<" worst_fem_block_abs="<<worstFEM
        <<" worst_rigid_block_abs="<<worstRigid
        <<" result="<<(allPass?"PASS":(allResolved?"FAIL":"INCONCLUSIVE"))
        <<" qualification=native_operator_fixture_only steps=0 material=source-authored-neohookean-analytic-FP64-oracle contact=source-production-frictionless-BOX-inertial-branch\n";
    if(!allPass) throw std::runtime_error(
        allResolved ? "coupled native operator comparison failed" :
                      "coupled FP64 derivative remained unresolved");
}

} // namespace

int main() { @autoreleasepool {
    try {
        CoupledDevice device;
        std::cout << "coupled_operator_device=" << [[device.base.gpu name] UTF8String]
                  << " operator_binding=current_frozen_build fp64_contact_oracle_binding=same_frozen_source"
                  << " native_steps=0\n";
        runCoupledCheck(device);
        std::cout << "result=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "result=FAIL message=" << error.what() << "\n";
        return 2;
    }
}}
