// Two-donor synthetic continuation of the existing real NumanX owner probe.
#include "metalrobo/NumiHumanTissueBinding.hpp"
#define main embedded_numanx_human_matter_adapter_probe_main
#include "numanx_human_matter_adapter_probe.mm"
#undef main
namespace adapter_fixture {
struct OwnerPhysicalObservation {
    __strong id<MTLBuffer> buffers[4] = {nil, nil, nil, nil};
    std::array<std::size_t,4> sizes{};
    std::uint64_t transaction=0, generation=0, token=0;
    bool encoded=false;
};
bool encodeOwnerObservation(void* opaque,
    const metalrobo::MetalArticulatedOperatorPhysicalStateObserverPass& pass) noexcept {
    auto* o=static_cast<OwnerPhysicalObservation*>(opaque);
    if(!o || pass.abiVersion!=metalrobo::kMetalArticulatedOperatorPhysicalStateObserverABIVersion ||
       pass.environmentCount!=1u || pass.transactionFingerprint!=o->transaction ||
       pass.physicsGeneration!=o->generation || pass.acceptedTokenFingerprint==0u ||
       pass.rootTranslationElementCount!=1u || pass.qElementCount!=161u ||
       pass.vElementCount!=160u || pass.mujocoStateElementCount!=1u || !pass.commandBuffer ||
       !pass.rootTranslations || !pass.q || !pass.v || !pass.mujocoStates) return false;
    id<MTLCommandBuffer> command=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    const std::array<void*,4> sources{pass.rootTranslations,pass.q,pass.v,pass.mujocoStates};
    o->sizes={sizeof(MRCompensatedRootTranslationGPU),161u*sizeof(float),160u*sizeof(float),sizeof(MRMujocoMuscleStateGPU)};
    for(unsigned i=0;i<4;++i) {
        o->buffers[i]=[command.device newBufferWithLength:o->sizes[i] options:MTLResourceStorageModeShared];
        if(o->buffers[i]==nil) return false;
    }
    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
    if(blit==nil) return false;
    for(unsigned i=0;i<4;++i)
        [blit copyFromBuffer:(__bridge id<MTLBuffer>)sources[i] sourceOffset:0 toBuffer:o->buffers[i] destinationOffset:0 size:o->sizes[i]];
    [blit endEncoding];o->token=pass.acceptedTokenFingerprint;o->encoded=true;return true;
}
OwnerPhysicalObservation observeOwner(metalrobo::MetalArticulatedOperatorContext& owner,
    std::uint64_t transaction, std::uint64_t generation) {
    OwnerPhysicalObservation o{};o.transaction=transaction;o.generation=generation;
    std::string error;
    require(owner.flushPhysicalStateObserver(&o,&encodeOwnerObservation,error) && o.encoded,
            "physical owner observation failed: "+error);
    return o;
}
struct CostalReplayEvidence {
    numi::matter::RuntimeStateSnapshot initial, accepted;
    std::array<nm_float4,2> reactions{};
    std::array<float,160> generalized{};
};
void verifyCostalOwnerIntegration(id<MTLDevice> device, bool accept,
    const numi::matter::RuntimeStateSnapshot* initialRestore=nullptr,
    CostalReplayEvidence* evidence=nullptr, bool continueStep=false, bool bounded=false, bool ownerRollback=false) {
    auto model = owner_fixture::makeHumanModel();
    std::fill(model.defaultV.begin(), model.defaultV.end(), 0.0f);
    const auto parsed = numi::matter::parseMatterFile(NUMI_MATTER_MATERIAL);
    require(parsed.succeeded(), "costal material failed");
    metalrobo::NumiHumanCostalBinding binding;
    binding.bodyCount = model.bodies.size(); binding.nodeCount = 4;
    binding.tetrahedronCount = 1;
    binding.atlasToWorld = {1,0,0,0.1,0,1,0,1,0,0,1,-0.2,0,0,0,1};
    binding.regions.resize(14, {MR_INVALID_INDEX, owner_fixture::kFirstBody,
                                owner_fixture::kFirstBody + 1});
    binding.nodeDonorBodies = {owner_fixture::kFirstBody, owner_fixture::kFirstBody,
                               owner_fixture::kFirstBody+1, owner_fixture::kFirstBody+1};
    metalrobo::NumiHumanCostalCartilagePayload cart; cart.regions.resize(14);
    cart.nodes = {{{0,0,0},0,1,0},{{.01f,0,0},0,2,0},
                  {{0,.01f,0},0,0,0},{{0,0,.01f},0,0,0}};
    cart.tetrahedra.push_back({{0,1,2,3},0});
    auto cooked = metalrobo::compileNumiHumanCostalTissue(
        cart, binding, model, parsed.material, 1.0/240.0);
    require(cooked.succeeded(), "costal cook failed: " + cooked.error);
    std::string ownershipError;
    require(metalrobo::validateNumiHumanCostalAttachmentOwnership(
                cart,binding,cooked.matter.world,ownershipError),
            "costal ownership rejected: " + ownershipError);
    require(cooked.mass.partitions.size()==2 &&
            cooked.matter.world.fem.humanAttachments.size()==2,
            "two-owner costal fixture coverage failed");
    const auto world = cooked.matter.world;
    model = cooked.rebasedHuman;
    // Use real source velocity in the joint after the registered rest cook.
    model.defaultV[6] = 0.001f;
    numi::matter::RuntimeConfiguration runtimeConfig;
    runtimeConfig.metallib = NUMI_MATTER_METALLIB;
    runtimeConfig.environmentCount = 1u;
    runtimeConfig.captureEvents = false;
    runtimeConfig.captureDiagnostics = true;
    runtimeConfig.adaptiveTransfer = false;
    runtimeConfig.coupledCandidateCompensatedTranslation = true;
    runtimeConfig.acceptedStateProofMujocoBytesPerEnvironmentCapacity =
        sizeof(MRMujocoMuscleStateGPU);
    numi::matter::Runtime matter;
    const auto matterInit = matter.initialize(world, runtimeConfig);
    require(matterInit.encoded,
        "integrated owner Matter initialization failed: " +
            matterInit.message);

    std::array<NMNumiHumanTendonFEMNodeLoadGPU, 4u> nodeLoads{};
    for (auto& load : nodeLoads) {
        std::fill(std::begin(load.endpointIndex),
                  std::end(load.endpointIndex), NM_INVALID_INDEX);
    }
    nodeLoads[2u].endpointIndex[0u] = 0u;
    nodeLoads[2u].scale.x = 0.1f;
    std::array<NMNumiHumanTendonFEMNodeAnchorGPU, 4u> nodeAnchors{};
    for (auto& anchor : nodeAnchors) anchor.bodyIndex = NM_INVALID_INDEX;
    for (const auto& a : world.fem.humanAttachments) {
        require(a.identity.x < nodeAnchors.size(), "costal anchor node out of range");
        nodeAnchors[a.identity.x].bodyIndex = a.identity.y;
        nodeAnchors[a.identity.x].flags = NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;
        nodeAnchors[a.identity.x].localPoint = a.localPoint;
    }
    std::array<NMNumiHumanTendonFEMEndpointReplacementGPU, 1u>
        replacements{};
    replacements[0u].loadEndpointIndex = 0u;
    replacements[0u].anchorEndpointIndex = 1u;
    replacements[0u].flags =
        NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE;
    replacements[0u].forceOwnerFraction.x = 0.1f;
    numi::matter::NumiHumanTendonFEMLoadSource tendonSource;
    tendonSource.nodeLoads = nodeLoads;
    tendonSource.nodeAnchors = nodeAnchors;
    tendonSource.endpointReplacements = replacements;
    tendonSource.endpointCount = 2u;
    tendonSource.environmentCount = 1u;
    tendonSource.productionForceOwnerFraction = 0.1f;
    numi::matter::NumiHumanTendonFEMLoadConfiguration tendonConfiguration;
    tendonConfiguration.metallib = NUMI_MATTER_METALLIB;
    tendonConfiguration.executionMode = numi::matter::
        NumiHumanTendonFEMLoadExecutionMode::numanXDeferred;
    numi::matter::NumiHumanTendonFEMLoadAdapter tendonAdapter;
    require(tendonAdapter.initialize(
                matter, tendonSource, tendonConfiguration),
        "integrated deferred tendon/FEM adapter initialization failed");
    const auto deferredLoads = tendonAdapter.deferredProgram();
    require(deferredLoads.valid() &&
            deferredLoads.environmentCount == matter.environmentCount() &&
            deferredLoads.femNodeCount == matter.femNodeCount() &&
            deferredLoads.activeAnchorCount ==
                matter.femHumanAttachmentCount() &&
            deferredLoads.runtimeDeviceProgramFingerprint ==
                matter.deviceProgramFingerprint(),
        "integrated deferred force program lost Runtime shape identity");

    metalrobo::MetalNumanXHumanMatterConfig adapterConfig;
    adapterConfig.matterRuntime = &matter;
    adapterConfig.coupledHumanMetallibPath = NUMANX_ADAPTER_METALLIB;
    adapterConfig.adapterMetallibPath = NUMANX_ADAPTER_METALLIB;
    adapterConfig.environmentCapacity = 1u;
    adapterConfig.pointCapacity = 2u;
    adapterConfig.transactionSlotCount = 1u;
    adapterConfig.stateProofProgram.context = &matter;
    adapterConfig.stateProofProgram.encode = &encodeRuntimeProof;
    adapterConfig.stateProofProgram.fingerprint =
        matter.acceptedStateProofProgramFingerprint();
    adapterConfig.tendonFEMDeferredLoadProgram = deferredLoads;
    {
        auto malformedConfig = adapterConfig;
        malformedConfig.tendonFEMDeferredLoadProgram.borrowExternalForces =
            nullptr;
        metalrobo::MetalNumanXHumanMatterContext malformed(
            std::move(malformedConfig));
        require(malformed.initialize().status == metalrobo::
                    MetalNumanXHumanMatterHostStatus::invalidConfiguration,
            "NumanX admitted a partial deferred force program");
    }
    {
        auto mismatchedConfig = adapterConfig;
        ++mismatchedConfig.tendonFEMDeferredLoadProgram.femNodeCount;
        metalrobo::MetalNumanXHumanMatterContext mismatched(
            std::move(mismatchedConfig));
        require(mismatched.initialize().status == metalrobo::
                    MetalNumanXHumanMatterHostStatus::matterRuntimeIncompatible,
            "NumanX admitted a deferred FEM-node count mismatch");
    }
    metalrobo::MetalNumanXHumanMatterContext adapter(adapterConfig);
    const auto adapterInit = adapter.initialize();
    require(adapterInit.succeeded() &&
            adapterInit.acceptedStateProofAvailable,
        "integrated owner adapter initialization failed: " +
            adapterInit.message);

    metalrobo::MetalNumanXHumanMatterTransaction transaction{};
    transaction.environmentCount = 1u;
    transaction.transactionSlot = 0u;
    transaction.controlStep = 47u;
    transaction.physicsSubstep = 0u;
    transaction.physicsSubsteps = 1u;
    transaction.expectedMatterCompletedMicrosteps = 1u;
    transaction.seed = 0x4701u;
    transaction.transactionFingerprint = 0x4702u;
    transaction.substepFingerprint = 0x4703u;
    transaction.acceptedTimestampMicroseconds = 4704u;
    transaction.physicsGeneration = 1u;
    transaction.linearizationEpoch = 0x4705u;
    transaction.slotGeneration = 1u;
    const metalrobo::MetalArticulatedOperatorConfig ownerConfig{
        .pointJacobiansOnly = true,
        .mujocoActivationTimestepSeconds = matter.timestepSeconds(),
        .metallibPath = NUMANX_ADAPTER_METALLIB,
    };
    metalrobo::MetalArticulatedOperatorContext owner(ownerConfig);
    for (unsigned iteration=0; iteration<(bounded?5u:(continueStep?2u:1u)); ++iteration) {
    std::cerr << "attempt=" << iteration << " transaction=" << transaction.transactionFingerprint
              << " generation=" << transaction.physicsGeneration << " slot=" << transaction.slotGeneration
              << " control=" << transaction.controlStep << " timestamp_us=" << transaction.acceptedTimestampMicroseconds << "\n";
    const auto program = adapter.program(transaction);
    require(program.valid(), "integrated owner adapter program is invalid");

    const auto points = owner_fixture::bodyProbes();
    owner_fixture::CandidateAudit ownerFixture;
    owner_fixture::initializeAudit(ownerFixture, device);
    for (auto& site : ownerFixture.muscleSites) {
        for (const auto& partition : cooked.mass.partitions) {
            if (site.bodyIndex != partition.donorBody) continue;
            site.localPoint.x -= static_cast<float>(partition.remainingCOMOffsetM[0]);
            site.localPoint.y -= static_cast<float>(partition.remainingCOMOffsetM[1]);
            site.localPoint.z -= static_cast<float>(partition.remainingCOMOffsetM[2]);
        }
    }
    auto input = owner_fixture::makeInput(model, points, ownerFixture);
    std::array<MRNumiHumanTendonBindingGPU, 2u> tendonBindings{};
    tendonBindings[0u].muscleIndex = 0u;
    tendonBindings[0u].endpointOrdinal = 0u;
    tendonBindings[0u].bodyIndex = owner_fixture::kFirstBody;
    tendonBindings[0u].mode = MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT;
    tendonBindings[0u].envelopeIndex = MR_INVALID_INDEX;
    tendonBindings[0u].boneStableId = 0u;
    tendonBindings[0u].sourceLocalPoint =
        ownerFixture.muscleSites[0u].localPoint;
    tendonBindings[1u].muscleIndex = 0u;
    tendonBindings[1u].endpointOrdinal = 1u;
    tendonBindings[1u].bodyIndex = owner_fixture::kFirstBody;
    tendonBindings[1u].mode = MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT;
    tendonBindings[1u].envelopeIndex = MR_INVALID_INDEX;
    tendonBindings[1u].boneStableId = 0u;
    tendonBindings[1u].sourceLocalPoint =
        ownerFixture.muscleSites[1u].localPoint;
    input.stand.tendonBindings = tendonBindings;
    input.stand.tendonLoadProgram = tendonAdapter.program();
    input.stand.numanXHumanMatterProgram = program;
    if(iteration!=0) {
        input.residentContinuation.previousTransactionFingerprint=transaction.transactionFingerprint-1u;
        input.residentContinuation.previousPhysicsGeneration=transaction.physicsGeneration-1u;
        input.q={};input.v={};input.rootTranslations={};input.stand.v={};input.mujoco.states={};
    }
    if(initialRestore != nullptr && iteration==0) {
        require(matter.restore(*initialRestore).encoded, "initial Matter restore failed");
        require(equalAcceptedAuthority(*initialRestore,matter.snapshot()), "restored initial authority differs");
    }
    OwnerPhysicalObservation ownerBefore{};
    if(ownerRollback && iteration==4u) ownerBefore=observeOwner(owner,transaction.transactionFingerprint-1u,transaction.physicsGeneration-1u);
    const auto before = matter.snapshot();
    require(before.available,
        "integrated owner Matter baseline is unavailable");
    metalrobo::MetalArticulatedOperatorSubmission submission;
    const auto submitted = owner.submit(model, input, submission);
    require(submitted.succeeded() && submitted.dispatched &&
            submission.valid(),
        "real owner rejected adapter+Matter prepare: " + submitted.message +
            "; tendon/FEM: " + tendonAdapter.diagnostics().message);
    metalrobo::MetalNumanXHumanMatterPrepared prepared;
    require(submission.extractPreparedHumanMatter(prepared) &&
            prepared.valid(),
        "real owner did not expose the prepared capability");
    metalrobo::MetalNumanXHumanMatterPreparedView view{};
    require(prepared.view(view) && view.environmentCount == 1u &&
            view.transactionSlot == transaction.transactionSlot &&
            view.stepIndex == 0u && view.substepIndex == 0u &&
            view.physicsSubstepCount == 1u &&
            view.controlStep == transaction.controlStep &&
            view.programFingerprint == program.fingerprint &&
            view.transactionFingerprint ==
                transaction.transactionFingerprint &&
            view.linearizationEpoch == transaction.linearizationEpoch &&
            view.slotGeneration == transaction.slotGeneration,
        "real owner prepared view lost adapter identity");

    id<MTLCommandQueue> applyQueue = [device newCommandQueue];
    require(applyQueue != nil,
        "real owner apply queue allocation failed");
    __unsafe_unretained id<MTLSharedEvent> ownerEvent =
        (__bridge id<MTLSharedEvent>)view.physicalPreparedEvent;
    waitForSharedEventValue(
        ownerEvent, view.physicalPreparedEventValue,
        "real owner physical completion event did not advance");

    auto humanIO = makeSensorPublicationService(
        device, transaction.transactionFingerprint,
        transaction.slotGeneration + 0x4700u);
    auto wrongTransactionCandidate = humanIO->program;
    ++wrongTransactionCandidate.transactionFingerprint;
    wrongTransactionCandidate.identityFingerprint =
        wrongTransactionCandidate.computedIdentityFingerprint();
    require(!prepared.bindHumanIOCandidatePublication(
                wrongTransactionCandidate),
        "real owner admitted a wrong-transaction HumanIO candidate");
    auto wrongDeviceCandidate = humanIO->program;
    ++wrongDeviceCandidate.deviceRegistryID;
    wrongDeviceCandidate.identityFingerprint =
        wrongDeviceCandidate.computedIdentityFingerprint();
    require(!prepared.bindHumanIOCandidatePublication(wrongDeviceCandidate),
        "real owner admitted a wrong-device HumanIO candidate");
    require(prepared.bindHumanIOCandidatePublication(humanIO->program),
        "real owner rejected exact post-physical HumanIO candidate");
    require(!prepared.bindHumanIOCandidatePublication(humanIO->program),
        "real owner admitted a duplicate HumanIO candidate bind");
    require(prepared.view(view) &&
            view.humanIOCandidateKeyFingerprint ==
                humanIO->program.candidateKeyFingerprint &&
            view.acceptedBrainGeneration ==
                humanIO->program.acceptedBrainGeneration &&
            view.humanIOSensorGeneration ==
                humanIO->program.sensorGeneration &&
            view.humanIOProgramFingerprint ==
                humanIO->program.humanIOProgramFingerprint &&
            view.humanIOSensorFingerprint ==
                humanIO->program.sensorFingerprint &&
            view.humanIOTransactionInstanceFingerprint ==
                humanIO->program.transactionInstanceFingerprint &&
            view.humanIOCandidatePublicationFingerprint ==
                humanIO->program.candidatePublicationFingerprint &&
            view.humanIODeviceRegistryID == device.registryID &&
            view.humanIOIdentityFingerprint ==
                humanIO->program.identityFingerprint,
        "real owner prepared view lost bound HumanIO identity");

    if (accept && (!bounded || iteration < 4u)) {
        // Synthetic Brain witness; the physical owner and publication path are real.
        const auto token=value<MRNumanXAcceptedPhysicsStateTokenGPU>((__bridge id<MTLBuffer>)view.preparedPhysicsStateTokens);
        require(token.tokenFingerprint!=0, "owner physical token is zero");
        MRNumanXHumanMatterBrainCommitWitnessGPU w{};
        w.magic=MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_MAGIC;w.abiVersion=MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_ABI_VERSION;
        w.structBytes=MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_BYTES;w.status=MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_PREPARE_COMPLETE;
        w.decision=MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;w.environment=0;w.stepIndex=view.stepIndex;w.substepIndex=view.substepIndex;
        w.transactionSlot=view.transactionSlot;w.physicsSubstepCount=view.physicsSubstepCount;w.controlStep=view.controlStep;
        w.programFingerprint=view.programFingerprint;w.transactionFingerprint=view.transactionFingerprint;w.linearizationEpoch=view.linearizationEpoch;
        w.slotGeneration=view.slotGeneration;w.physicsTokenFingerprint=token.tokenFingerprint;w.brainProgramFingerprint=0xb1000001u;w.brainShadowStateFingerprint=0xb1000002u;
        std::uint64_t wh=14695981039346656037ull;
        for(auto x:{w.magic,w.abiVersion,w.structBytes,w.status,w.decision,w.environment,w.stepIndex,w.substepIndex,w.transactionSlot,w.physicsSubstepCount,w.controlStep,w.reserved0}) mixU32(wh,x);
        for(auto x:{w.programFingerprint,w.transactionFingerprint,w.linearizationEpoch,w.slotGeneration,w.physicsTokenFingerprint,w.brainProgramFingerprint,w.brainShadowStateFingerprint,w.reserved1[0],w.reserved1[1]}) mixU64(wh,x);
        w.witnessFingerprint=wh ? wh : 1;
        id<MTLBuffer> wb=makeBuffer(device,w,@"synthetic Brain witness");id<MTLSharedEvent> we=[device newSharedEvent];we.signaledValue=1;
        id<MTLCommandBuffer> pc=[applyQueue commandBuffer];metalrobo::MetalNumanXHumanMatterProposalRequest pr{};
        pr.commandBuffer=(__bridge void*)pc;pr.brainCommitWitnesses=(__bridge void*)wb;pr.brainPrepareCompleteEvent=(__bridge void*)we;
        pr.brainPrepareCompleteEventValue=1;pr.brainCommitWitnessesGPUAddress=wb.gpuAddress;pr.brainCommitWitnessElementCount=1;pr.brainCommitWitnessStride=1;
        pr.environmentCount=view.environmentCount;pr.transactionSlot=view.transactionSlot;pr.stepIndex=view.stepIndex;pr.substepIndex=view.substepIndex;
        pr.physicsSubstepCount=view.physicsSubstepCount;pr.controlStep=view.controlStep;pr.programFingerprint=view.programFingerprint;
        pr.transactionFingerprint=view.transactionFingerprint;pr.linearizationEpoch=view.linearizationEpoch;pr.slotGeneration=view.slotGeneration;
        const auto proposed=prepared.proposePrepared(pr);require(proposed.succeeded()&&proposed.encoded,"accepted proposal: "+proposed.message);
        finish(pc);waitForSharedEventValue(ownerEvent,view.proposalEventValue,"accepted proposal event");
        const auto proposal=value<MRNumanXHumanMatterProposalGPU>((__bridge id<MTLBuffer>)view.proposals);
        require(proposal.status==MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY && proposal.decision==MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT,"proposal was not accepted");
        PreparedTransaction identity{};identity.program=program;identity.transaction=transaction;
        const auto preflight=makePreflight(identity,proposal,ApplyMode::accept);
        id<MTLBuffer> pb=makeBuffer(device,preflight,@"synthetic Brain preflight");id<MTLSharedEvent> pe=[device newSharedEvent];pe.signaledValue=1;
        metalrobo::MetalNumanXHumanMatterBrainPreflightView pv{};pv.brainCommitPreflights=(__bridge void*)pb;pv.preflightReadyEvent=(__bridge void*)pe;
        pv.brainCommitPreflightsGPUAddress=pb.gpuAddress;pv.brainCommitPreflightElementCount=1;pv.preflightReadyEventValue=1;pv.brainCommitPreflightStride=1;
        pv.environmentCount=view.environmentCount;pv.transactionSlot=view.transactionSlot;pv.stepIndex=view.stepIndex;pv.substepIndex=view.substepIndex;
        pv.physicsSubstepCount=view.physicsSubstepCount;pv.controlStep=view.controlStep;pv.programFingerprint=view.programFingerprint;
        pv.transactionFingerprint=view.transactionFingerprint;pv.linearizationEpoch=view.linearizationEpoch;pv.slotGeneration=view.slotGeneration;
        require(prepared.reservePreparedApplication(pv),"accepted application reservation");
        const auto ack=makeAck(identity,proposal,preflight,ApplyMode::accept);id<MTLBuffer> ab=makeBuffer(device,ack,@"synthetic Brain ACK");
        id<MTLSharedEvent> ae=[device newSharedEvent];ae.signaledValue=1;OwnerApplyCompletionCapture completion;
        id<MTLCommandBuffer> ac=[applyQueue commandBuffer];metalrobo::MetalNumanXHumanMatterApplyRequest ar{};
        ar.commandBuffer=(__bridge void*)ac;ar.brainAcks=(__bridge void*)ab;ar.brainAckEvent=(__bridge void*)ae;ar.brainAckEventValue=1;
        ar.brainAcksGPUAddress=ab.gpuAddress;ar.brainAckElementCount=1;ar.brainAckStride=1;ar.completionContext=&completion;ar.completion=&captureOwnerApplyCompletion;
        ar.environmentCount=view.environmentCount;ar.transactionSlot=view.transactionSlot;ar.stepIndex=view.stepIndex;ar.substepIndex=view.substepIndex;
        ar.physicsSubstepCount=view.physicsSubstepCount;ar.controlStep=view.controlStep;ar.programFingerprint=view.programFingerprint;
        ar.transactionFingerprint=view.transactionFingerprint;ar.linearizationEpoch=view.linearizationEpoch;ar.slotGeneration=view.slotGeneration;
        const auto applied=prepared.applyPrepared(ar);require(applied.succeeded()&&applied.encoded,"accepted apply: "+applied.message);
        finish(ac);waitForOwnerApplyCompletion(completion);
        require(completion.status.load()==static_cast<std::uint32_t>(metalrobo::MetalNumanXHumanMatterApplyTerminalStatus::acceptedPendingPublication),"accepted apply disposition");
        metalrobo::MetalNumanXHumanMatterPublicationReservationRequest reservation{};
        reservation.environmentCount=view.environmentCount;reservation.transactionSlot=view.transactionSlot;reservation.stepIndex=view.stepIndex;
        reservation.substepIndex=view.substepIndex;reservation.physicsSubstepCount=view.physicsSubstepCount;reservation.controlStep=view.controlStep;
        reservation.programFingerprint=view.programFingerprint;reservation.transactionFingerprint=view.transactionFingerprint;
        reservation.linearizationEpoch=view.linearizationEpoch;reservation.slotGeneration=view.slotGeneration;reservation.jointCommitFingerprint=0xd002u;reservation.brainGeneration=19;
        require(prepared.reservePublishedRoot(reservation),"accepted publication reservation");
        auto* fence=static_cast<MRNumanXHumanMatterJointPublicationFenceGPU*>(((__bridge id<MTLBuffer>)view.publicationFences).contents);
        fence->status=MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED;fence->fenceFingerprint=recordFingerprint(*fence);
        metalrobo::MetalNumanXHumanMatterPublicationReleaseRequest release{};
        release.publicationFences=view.publicationFences;release.publicationFencesGPUAddress=view.publicationFencesGPUAddress;
        release.publicationFenceElementCount=view.publicationFenceElementCount;release.publicationFenceStride=view.publicationFenceStride;
        release.environmentCount=view.environmentCount;release.transactionSlot=view.transactionSlot;release.stepIndex=view.stepIndex;release.substepIndex=view.substepIndex;
        release.physicsSubstepCount=view.physicsSubstepCount;release.controlStep=view.controlStep;release.programFingerprint=view.programFingerprint;
        release.transactionFingerprint=view.transactionFingerprint;release.linearizationEpoch=view.linearizationEpoch;release.slotGeneration=view.slotGeneration;
        release.jointCommitFingerprint=reservation.jointCommitFingerprint;release.brainGeneration=reservation.brainGeneration;
        require(prepared.releasePublishedRoot(release)==metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::released,"accepted publication release");
        const auto after=matter.snapshot();require(after.available && after.femNodes.size()==before.femNodes.size(),"accepted Matter snapshot");
        double move=0;for(std::size_t i=0;i<after.femNodes.size();++i){const auto a=after.femNodes[i].positionAndMass,b=before.femNodes[i].positionAndMass;
            move=std::max(move,std::sqrt(double(a.x-b.x)*(a.x-b.x)+double(a.y-b.y)*(a.y-b.y)+double(a.z-b.z)*(a.z-b.z)));}
        require(std::isfinite(move)&&move>0,"accepted tissue did not move");
        id<MTLBuffer> source=(__bridge id<MTLBuffer>)matter.femHumanAttachmentRawReactionBuffer();id<MTLBuffer> rb=[device newBufferWithLength:2*sizeof(nm_float4) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> rc=[applyQueue commandBuffer];id<MTLBlitCommandEncoder> blit=[rc blitCommandEncoder];
        [blit copyFromBuffer:source sourceOffset:0 toBuffer:rb destinationOffset:0 size:rb.length];[blit endEncoding];finish(rc);
        auto* r=static_cast<nm_float4*>(rb.contents);const auto norm=[](nm_float4 x){return std::sqrt(double(x.x)*x.x+double(x.y)*x.y+double(x.z)*x.z);};
        const double r0=norm(r[0]),r1=norm(r[1]);
        for(unsigned i=0;i<4;++i) {
            const auto a=after.femNodes[i].positionAndMass,b=before.femNodes[i].positionAndMass;
            std::cerr<<"node="<<i<<" delta="<<(a.x-b.x)<<","<<(a.y-b.y)<<","<<(a.z-b.z)<<"\n";
        }
        for(unsigned i=0;i<2;++i) {
            const auto& attachment=world.fem.humanAttachments[i];
            std::cerr<<"attachment_row="<<i<<" node="<<attachment.identity.x<<" body="<<attachment.identity.y
                     <<" raw_reaction_n="<<r[i].x<<","<<r[i].y<<","<<r[i].z<<"\n";
        }
        id<MTLBuffer> gs=(__bridge id<MTLBuffer>)program.matterGeneralizedReaction;
        id<MTLBuffer> gb=[device newBufferWithLength:160*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> gc=[applyQueue commandBuffer];id<MTLBlitCommandEncoder> gblit=[gc blitCommandEncoder];
        [gblit copyFromBuffer:gs sourceOffset:0 toBuffer:gb destinationOffset:0 size:gb.length];[gblit endEncoding];finish(gc);
        auto* gf=static_cast<float*>(gb.contents);double gl1=0;for(unsigned i=0;i<160;++i) gl1+=std::abs(gf[i]);
        std::cerr<<"accepted_displacement_m="<<move<<" reaction_l1="<<r0<<","<<r1
                 <<" generalized_l1="<<gl1<<" joint6="<<gf[6]<<"\n";
        require(std::isfinite(r0)&&std::isfinite(r1)&&r0>0&&r1>0,"accepted attachment reactions missing");
        require(humanIO->stage==SensorPublicationService::Stage::released && humanIO->publishCalls==1,
                "accepted HumanIO was not published exactly once");
        const auto adapterSnapshot=tendonAdapter.snapshot();
        require(adapterSnapshot.available, "accepted tendon adapter snapshot missing");
        if(!bounded) {
            if(!continueStep || iteration!=0)
                require(matter.restore(after).encoded && equalAcceptedAuthority(after,matter.snapshot()),
                        "committed Matter restoration differed");
            require(tendonAdapter.restore(adapterSnapshot).succeeded(), "committed tendon adapter restore failed");
            const auto adapterReplay=tendonAdapter.snapshot();
            require(adapterReplay.available && adapterReplay.authoritySHA256==adapterSnapshot.authoritySHA256,
                    "committed tendon adapter authority differs");
        }
        if(evidence) {
            evidence->initial=before;evidence->accepted=after;
            std::copy_n(r,2,evidence->reactions.begin());std::copy_n(gf,160,evidence->generalized.begin());
        }
        std::cout<<"PASS accepted costal owner step="<<iteration<<" displacement_m="<<move<<" reactions_n="<<r0<<","<<r1<<" synthetic_brain_witness=1\n";
        if((continueStep && iteration==0) || (bounded && iteration<4u)) {
            ++transaction.controlStep;++transaction.physicsGeneration;++transaction.slotGeneration;
            ++transaction.transactionFingerprint;++transaction.substepFingerprint;++transaction.linearizationEpoch;
            transaction.acceptedTimestampMicroseconds += 4167;
            continue;
        }
        return;
    }

    id<MTLCommandBuffer> proposalCommand = [applyQueue commandBuffer];
    metalrobo::MetalNumanXHumanMatterProposalRequest proposalRequest{};
    proposalRequest.mode =
        metalrobo::MetalNumanXHumanMatterProposalMode::forceReject;
    proposalRequest.commandBuffer = (__bridge void*)proposalCommand;
    proposalRequest.environmentCount = view.environmentCount;
    proposalRequest.transactionSlot = view.transactionSlot;
    proposalRequest.stepIndex = view.stepIndex;
    proposalRequest.substepIndex = view.substepIndex;
    proposalRequest.physicsSubstepCount = view.physicsSubstepCount;
    proposalRequest.controlStep = view.controlStep;
    proposalRequest.programFingerprint = view.programFingerprint;
    proposalRequest.transactionFingerprint = view.transactionFingerprint;
    proposalRequest.linearizationEpoch = view.linearizationEpoch;
    proposalRequest.slotGeneration = view.slotGeneration;
    const auto proposed = prepared.proposePrepared(proposalRequest);
    require(proposed.succeeded() && proposed.encoded,
        "real owner rejected mutation-free force-reject proposal: " +
            proposed.message);
    require(!prepared.proposePrepared(proposalRequest).succeeded(),
        "real owner admitted a duplicate proposal");
    finish(proposalCommand);
    waitForSharedEventValue(
        ownerEvent, view.proposalEventValue,
        "real owner proposal completion event did not advance");

    const auto proposal = value<MRNumanXHumanMatterProposalGPU>(
        (__bridge id<MTLBuffer>)view.proposals);
    require(proposal.status == MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY &&
            proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
            proposal.candidatePublicationFingerprint ==
                humanIO->program.candidatePublicationFingerprint &&
            proposal.humanIOIdentityFingerprint ==
                humanIO->program.identityFingerprint &&
            proposal.proposalFingerprint == recordFingerprint(proposal),
        "real owner force-reject proposal is malformed");
    MRNumanXHumanMatterBrainCommitPreflightGPU preflight{};
    preflight.abiVersion =
        MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_ABI_VERSION;
    preflight.structBytes = MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_BYTES;
    preflight.status = MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_SUCCESS;
    preflight.environment = 0u;
    preflight.controlStep = view.controlStep;
    preflight.substepIndex = view.substepIndex;
    preflight.physicsSubstepCount = view.physicsSubstepCount;
    preflight.transactionSlot = view.transactionSlot;
    preflight.ownerProgramFingerprint = view.programFingerprint;
    preflight.transactionFingerprint = view.transactionFingerprint;
    preflight.linearizationEpoch = view.linearizationEpoch;
    preflight.slotGeneration = view.slotGeneration;
    preflight.substepFingerprint = transaction.substepFingerprint;
    preflight.physicsTokenFingerprint = proposal.physicsTokenFingerprint;
    preflight.fastTargetGeneration = 1u;
    preflight.cognitiveTargetGeneration = 1u;
    preflight.jointReceiptFingerprint = 0x4706u;
    preflight.fastProgramFingerprint = 0x4707u;
    preflight.brainProgramFingerprint = 0x4708u;
    preflight.preflightFingerprint = recordFingerprint(preflight);
    id<MTLBuffer> preflightBuffer = makeBuffer(
        device, preflight, @"integrated Brain commit preflight");
    id<MTLSharedEvent> preflightReady = [device newSharedEvent];
    require(preflightReady != nil,
        "integrated Brain preflight event allocation failed");
    preflightReady.signaledValue = 1u;
    metalrobo::MetalNumanXHumanMatterBrainPreflightView preflightView{};
    preflightView.brainCommitPreflights =
        (__bridge void*)preflightBuffer;
    preflightView.preflightReadyEvent = (__bridge void*)preflightReady;
    preflightView.brainCommitPreflightsGPUAddress =
        preflightBuffer.gpuAddress;
    preflightView.brainCommitPreflightElementCount = 1u;
    preflightView.preflightReadyEventValue = 1u;
    preflightView.brainCommitPreflightStride = 1u;
    preflightView.environmentCount = view.environmentCount;
    preflightView.transactionSlot = view.transactionSlot;
    preflightView.stepIndex = view.stepIndex;
    preflightView.substepIndex = view.substepIndex;
    preflightView.physicsSubstepCount = view.physicsSubstepCount;
    preflightView.controlStep = view.controlStep;
    preflightView.programFingerprint = view.programFingerprint;
    preflightView.transactionFingerprint = view.transactionFingerprint;
    preflightView.linearizationEpoch = view.linearizationEpoch;
    preflightView.slotGeneration = view.slotGeneration;
    auto stalePreflight = preflightView;
    ++stalePreflight.controlStep;
    require(!prepared.reservePreparedApplication(stalePreflight),
        "real owner admitted a stale preflight reservation");
    require(prepared.reservePreparedApplication(preflightView),
        "real owner rejected exact force-reject reservation");
    require(!prepared.reservePreparedApplication(preflightView),
        "real owner admitted duplicate application reservation");

    OwnerApplyCompletionCapture completion;
    id<MTLCommandBuffer> applyCommand = [applyQueue commandBuffer];
    metalrobo::MetalNumanXHumanMatterApplyRequest applyRequest{};
    applyRequest.mode =
        metalrobo::MetalNumanXHumanMatterApplyMode::forceReject;
    applyRequest.commandBuffer = (__bridge void*)applyCommand;
    applyRequest.completionContext = &completion;
    applyRequest.completion = &captureOwnerApplyCompletion;
    applyRequest.environmentCount = view.environmentCount;
    applyRequest.transactionSlot = view.transactionSlot;
    applyRequest.stepIndex = view.stepIndex;
    applyRequest.substepIndex = view.substepIndex;
    applyRequest.physicsSubstepCount = view.physicsSubstepCount;
    applyRequest.controlStep = view.controlStep;
    applyRequest.programFingerprint = view.programFingerprint;
    applyRequest.transactionFingerprint = view.transactionFingerprint;
    applyRequest.linearizationEpoch = view.linearizationEpoch;
    applyRequest.slotGeneration = view.slotGeneration;
    auto staleApply = applyRequest;
    ++staleApply.slotGeneration;
    require(!prepared.applyPrepared(staleApply).succeeded(),
        "real owner admitted a stale apply generation");
    const auto applied = prepared.applyPrepared(applyRequest);
    require(applied.succeeded() && applied.encoded,
        "real owner rejected forced root apply: " + applied.message);
    require(!prepared.applyPrepared(applyRequest).succeeded(),
        "real owner admitted duplicate apply");
    finish(applyCommand);
    waitForOwnerApplyCompletion(completion);
    require(completion.status.load(std::memory_order_acquire) ==
                static_cast<std::uint32_t>(metalrobo::
                    MetalNumanXHumanMatterApplyTerminalStatus::
                        rejectedReleased) &&
            completion.slotGeneration.load(std::memory_order_acquire) ==
                view.slotGeneration,
        "real owner apply completion lost reject/generation identity");

    const auto disposition = matter.preparedStateDisposition(
        dispositionIdentity(transaction, program));
    const auto after = matter.snapshot();
    require(disposition == numi::matter::PreparedStateDisposition::resolved &&
            !prepared.valid() && equalAcceptedAuthority(before, after) &&
            humanIO->stage == SensorPublicationService::Stage::rejected &&
            humanIO->reserveCalls == 0u &&
            humanIO->publishCalls == 0u &&
            humanIO->rejectCalls == 1u,
        "owner+adapter+Matter reject did not resolve and byte-restore");
    if(ownerRollback) {
        const auto ownerAfter=observeOwner(owner,transaction.transactionFingerprint-1u,transaction.physicsGeneration-1u);
        require(ownerBefore.token==ownerAfter.token && ownerBefore.sizes==ownerAfter.sizes,
                "rejected owner changed accepted token or arena layout");
        for(unsigned i=0;i<4;++i)
            require(std::memcmp(ownerBefore.buffers[i].contents,ownerAfter.buffers[i].contents,ownerBefore.sizes[i])==0,
                    "rejected owner changed accepted physical arena "+std::to_string(i));
        std::cout<<"PASS owner physical rejection bytes root/q/v/mujoco="
                 <<ownerBefore.sizes[0]<<","<<ownerBefore.sizes[1]<<","<<ownerBefore.sizes[2]<<","<<ownerBefore.sizes[3]
                 <<" transaction="<<ownerBefore.transaction<<" generation="<<ownerBefore.generation<<" token="<<ownerBefore.token<<"\n";
    }
    const auto tendonDiagnostics = tendonAdapter.diagnostics();
    require(tendonDiagnostics.initialized &&
            tendonDiagnostics.executionMode == numi::matter::
                NumiHumanTendonFEMLoadExecutionMode::numanXDeferred &&
            tendonDiagnostics.preSourceCorrectionEncodeCount == iteration + 1u &&
            tendonDiagnostics.deferredExternalForceBorrowCount == 2u * (iteration + 1u) &&
            tendonDiagnostics.runtimePreDynamicsEncodeCount == 0u &&
            tendonDiagnostics.runtimePostCommitEncodeCount == 0u &&
            tendonDiagnostics.encodedPassCount == iteration + 1u,
        "deferred tendon/FEM path opened a duplicate Runtime lifecycle or "
        "lost its exact force-view borrows");
    std::cout << "PASS force-reject attempt=" << iteration << " transaction=" << transaction.transactionFingerprint
              << " Matter_authority_restored=1 HumanIO_rejected=1\n";
    }
}

} // namespace adapter_fixture
int main(int argc, const char** argv) {
  @autoreleasepool {
    try {
      id<MTLDevice> device=MTLCreateSystemDefaultDevice();
      adapter_fixture::require(device != nil, "Metal device unavailable");
      const bool accept=argc==2 && std::string(argv[1])=="--accept";
      const bool replay=argc==2 && std::string(argv[1])=="--replay";
      const bool continuation=argc==2 && std::string(argv[1])=="--continue";
      const bool ownerRollback=argc==2 && std::string(argv[1])=="--owner-rollback";
      const bool bounded=(argc==2 && std::string(argv[1])=="--bounded") || ownerRollback;
      adapter_fixture::require(argc==1 || accept || replay || continuation || bounded,"usage: owner_probe [--accept|--replay|--continue|--bounded|--owner-rollback]");
      if(replay) {
        adapter_fixture::CostalReplayEvidence first, second;
        adapter_fixture::verifyCostalOwnerIntegration(device,true,nullptr,&first);
        adapter_fixture::verifyCostalOwnerIntegration(device,true,&first.initial,&second);
        adapter_fixture::require(adapter_fixture::equalAcceptedAuthority(first.accepted,second.accepted),"accepted Matter replay differs");
        adapter_fixture::require(!std::memcmp(first.reactions.data(),second.reactions.data(),sizeof(first.reactions)),"reaction replay differs");
        adapter_fixture::require(!std::memcmp(first.generalized.data(),second.generalized.data(),sizeof(first.generalized)),"generalized reaction replay differs");
        std::cout<<"PASS committed Matter and adapter restore; initial-state replay Matter authority and forces bitwise\n";
      } else adapter_fixture::verifyCostalOwnerIntegration(device,accept || continuation || bounded,nullptr,nullptr,continuation,bounded,ownerRollback);
      if(!accept && !replay && !continuation && !bounded) std::cout << "PASS costal two-owner real NumanX force-reject synthetic chain\n";
      return 0;
    } catch(const std::exception& e) {
      std::cerr << "costal real-owner probe FAILED: " << e.what() << "\n";
      return 1;
    }
  }
}
