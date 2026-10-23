#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "numi/matter/numi_human.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace numi::matter;
static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
int main(int argc, const char** argv) {
    @autoreleasepool {
        try {
            require(argc == 3, "usage: native_initialize_test material metallib");
            auto parsed = parseMatterFile(argv[1]);
            require(parsed.succeeded(), "material parse failed");
            WorldSource source;
            source.environmentCount = 1u;
            source.gravity = {0.0, 0.0, 0.0};
            source.frameTimestep = 1.e-4;
            source.materials.push_back(std::move(parsed.material));
            ObjectSource object;
            object.name = "surface_attachment_native_initialization";
            object.materialIndex = 0u;
            object.representation = Representation::fem;
            object.mixedFEM = false;
            object.deformableContact = false;
            object.deformableSelfContact = false;
            object.characteristicLength = .01;
            object.femNodes = {{0.,0.,0.},{.01,0.,0.},{0.,.01,0.},{0.,0.,.01}};
            object.femFixedNodes = {0u,1u,2u};
            object.tetrahedra.push_back({{0u,1u,2u,3u}});
            source.objects.push_back(std::move(object));
            CompileOptions options;
            options.maximumRateExponent = 0u;
            auto compiled = compileWorld(source, options);
            require(compiled.succeeded(), "world compile failed");
            Runtime runtime;
            auto initialized = runtime.initialize(compiled.world, {
                .metallib = argv[2], .environmentCount = 1u,
                .captureEvents = true, .captureDiagnostics = true,
                .automaticIdentification = false, .adaptiveTransfer = false});
            require(initialized.encoded && runtime.valid(), "runtime initialize failed");
            auto before = runtime.snapshot();
            require(before.available, "runtime initial snapshot missing");
            std::array<NMNumiHumanTendonFEMNodeLoadGPU,4> loads{};
            for (auto& load: loads) std::fill_n(load.endpointIndex,4u,NM_INVALID_INDEX);
            std::array<NMNumiHumanTendonFEMNodeAnchorGPU,4> anchors{};
            for (auto& anchor: anchors) anchor.bodyIndex = NM_INVALID_INDEX;
            for (unsigned i=0; i<3; ++i) {
                anchors[i].bodyIndex = 0u;
                anchors[i].flags = NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;
                anchors[i].localPoint = {i==1 ? .01f:0.f,i==2 ? .01f:0.f,0.f,0.f};
            }
            NMNumiHumanFEMSurfaceAttachmentGPU surface{};
            surface.node[0]=0u; surface.node[1]=1u; surface.node[2]=3u;
            surface.bodyIndex=0u; surface.stableID=731u;
            surface.flags=NM_NUMI_HUMAN_FEM_SURFACE_ATTACHMENT_ACTIVE;
            surface.barycentric={.2f,.3f,.5f,0.f};
            surface.bodyLocalPointAndArea={.003f,0.f,.005f,1.e-5f};
            surface.stiffnessAndStrainPerPressure={1.e6f,1.e-5f,0.f,0.f};
            NMNumiHumanFEMBodyContactSampleGPU legacy{};
            legacy.slaveNode=3u; legacy.bodyIndex=0u;
            legacy.flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE;
            legacy.bodyLocalPointAndArea={0.f,0.f,.009f,1.e-5f};
            legacy.bodyLocalNormalAndReferenceSeparation={0.f,0.f,1.f,0.f};
            legacy.stiffnessAndNormalStrainPerPressure={1.e6f,1.e-5f,0.f,0.f};
            const NumiHumanTendonFEMLoadConfiguration config{.metallib=argv[2]};
            auto makeSource = [&](std::span<const NMNumiHumanFEMBodyContactSampleGPU> body,
                                  std::span<const NMNumiHumanFEMSurfaceAttachmentGPU> surf) {
                NumiHumanTendonFEMLoadSource s;
                s.nodeLoads=loads; s.nodeAnchors=anchors;
                s.femBodyContactSamples=body;
                s.endpointCount=1u; s.environmentCount=1u;
                s.femSurfaceAttachments=surf;
                return s;
            };
            NumiHumanTendonFEMLoadAdapter empty, legacyOnly, surfaceOnly, mixed;
            require(empty.initialize(runtime,makeSource({},{}),config), "empty optional interactions rejected");
            require(legacyOnly.initialize(runtime,makeSource({&legacy,1},{}),config), "legacy-only rejected");
            require(surfaceOnly.initialize(runtime,makeSource({},{&surface,1}),config), "surface-only rejected");
            require(mixed.initialize(runtime,makeSource({&legacy,1},{&surface,1}),config), "mixed rejected");
            auto base = empty.snapshot(); auto one = surfaceOnly.snapshot();
            require(base.available && one.available, "adapter snapshot missing");
            require(base.adapterFingerprint != one.adapterFingerprint, "surface omitted from authority");
            require(base.femBodyContactAcceptedHistory.empty() &&
                    !one.femBodyContactAcceptedHistory.empty(), "surface history extent missing");
            require(surfaceOnly.restore(one).succeeded(), "own snapshot restore rejected");
            require(!empty.restore(one).succeeded(), "cross-source snapshot accepted");
            auto expectRejected = [&](const NMNumiHumanFEMSurfaceAttachmentGPU& bad) {
                NumiHumanTendonFEMLoadAdapter rejected;
                require(!rejected.initialize(runtime,makeSource({},{&bad,1}),config),
                        "malformed surface attachment admitted");
            };
            auto bad=surface; bad.node[2]=4u; expectRejected(bad);
            bad=surface; bad.node[1]=bad.node[0]; expectRejected(bad);
            bad=surface; bad.barycentric.x=-.2f; expectRejected(bad);
            bad=surface; bad.barycentric.x=.4f; expectRejected(bad);
            bad=surface; bad.bodyLocalPointAndArea.w=0.f; expectRejected(bad);
            bad=surface; bad.stiffnessAndStrainPerPressure.x=std::numeric_limits<float>::infinity(); expectRejected(bad);
            bad=surface; bad.reserved0=1u; expectRejected(bad);
            bad=surface; bad.flags=0u; expectRejected(bad);
            bad=surface; bad.stableID=NM_INVALID_INDEX; expectRejected(bad);
            std::array<NMNumiHumanFEMSurfaceAttachmentGPU,2> duplicate{{surface,surface}};
            NumiHumanTendonFEMLoadAdapter duplicateAdapter;
            require(!duplicateAdapter.initialize(runtime,makeSource({},duplicate),config), "duplicate stable ID admitted");
            auto changed=surface; changed.stableID=732u;
            NumiHumanTendonFEMLoadAdapter changedAdapter;
            require(changedAdapter.initialize(runtime,makeSource({},{&changed,1}),config), "changed source rejected");
            require(changedAdapter.snapshot().adapterFingerprint != one.adapterFingerprint,
                    "stable attachment identity omitted from authority");
            auto after=runtime.snapshot();
            require(after.available && before.femNodes.size()==after.femNodes.size(), "runtime snapshot extent changed");
            require(std::memcmp(before.femNodes.data(),after.femNodes.data(),before.femNodes.size()*sizeof(before.femNodes[0]))==0,
                    "adapter validation changed physical nodes");
            std::cout << "PASS native initialization and source authority; physical nodes unchanged\n";
            // Exercise this new span through the actual adapter/Runtime owner.
            // The body fixture is fixed; its wrench is still projected on Metal.
            auto loaded=surface;
            loaded.bodyLocalPointAndArea.x += .002f;
            // Exercise the optional bounded bank through real adapter bindings,
            // not only direct coupon encoders. Both faces stay below the tissue:
            // contact contributes zero, so the original spring-law assertion
            // remains an independent check of the mixed owner.
            NMNumiHumanFEMBodyContactSampleGPU bankSample{};
            bankSample.slaveNode=3u; bankSample.bodyIndex=0u;
            bankSample.flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
                NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
            bankSample.bodyLocalPointAndArea={0.f,0.f,0.f,1.e-5f};
            bankSample.bodyLocalNormalAndReferenceSeparation={.02f,0.f,0.f,1.e6f};
            bankSample.stiffnessAndNormalStrainPerPressure={0.f,.02f,0.f,1.e-5f};
            std::array<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU,2> bank{};
            for (unsigned i=0;i<bank.size();++i) {
                const float z=-.01f*float(i);
                bank[i].bodyLocalVertex0={0.f,0.f,z,0.f};
                bank[i].bodyLocalVertex1={.02f,0.f,z,0.f};
                bank[i].bodyLocalVertex2={0.f,.02f,z,0.f};
                bank[i].authoredOutwardNormal={0.f,0.f,1.f,0.f};
                bank[i].bodyIndex=0u; bank[i].boneStableId=49u;
                bank[i].triangleStableId=100u+i;
            }
            const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU bankRange{0u,2u,.025f,0u};
            auto physicalSource=makeSource({&bankSample,1},{&loaded,1});
            physicalSource.femBodyContactNeighborhoodRanges={&bankRange,1};
            physicalSource.femBodyContactNeighborhoodTriangles=bank;
            NumiHumanTendonFEMLoadAdapter physical;
            require(physical.initialize(runtime,physicalSource,config), "physical adapter initialize failed");
            auto changedBank=bank;
            changedBank[1].triangleStableId=102u;
            auto changedBankSource=physicalSource;
            changedBankSource.femBodyContactNeighborhoodTriangles=changedBank;
            NumiHumanTendonFEMLoadAdapter changedBankAdapter;
            require(changedBankAdapter.initialize(runtime,changedBankSource,config), "changed bank initialize failed");
            require(changedBankAdapter.snapshot().adapterFingerprint != physical.snapshot().adapterFingerprint &&
                    !changedBankAdapter.restore(physical.snapshot()).succeeded(),
                    "bank identity missing from snapshot authority");
            auto initialAuthority=physical.snapshot();
            auto program=physical.program();
            require(program.valid(), "physical adapter program invalid");
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue=[device newCommandQueue];
            auto buffer=[&](const void* data, size_t bytes) {
                auto b=[device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
                require(b!=nil,"physical fixture buffer unavailable");
                return b;
            };
            MRNumiHumanTendonTransferResultGPU transfer{};
            transfer.status=MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS;
            transfer.envelopeIndex=MR_INVALID_INDEX;
            MRNumiHumanTendonBindingGPU binding{};
            binding.mode=MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT;
            binding.envelopeIndex=MR_INVALID_INDEX;
            MRArticulatedBodyPoseGPU pose{};
            pose.orientation={0.f,0.f,0.f,1.f};
            MRNumiHumanStandStatusGPU stand{};
            stand.failingIndex=MR_INVALID_INDEX;
            auto tb=buffer(&transfer,sizeof(transfer)), bb=buffer(&binding,sizeof(binding));
            auto pb=buffer(&pose,sizeof(pose)), sb=buffer(&stand,sizeof(stand));
            std::array<float,72> jac{};
            for (unsigned point=0;point<4;++point) {
                for (unsigned axis=0;axis<3;++axis) jac[point*18+axis*6+axis]=1.f;
            }
            // Unit-offset virtual points define the exact body rotation columns.
            jac[1*18+1*6+5]=1.f; jac[1*18+2*6+4]=-1.f;
            jac[2*18+0*6+5]=-1.f; jac[2*18+2*6+3]=1.f;
            jac[3*18+0*6+4]=1.f; jac[3*18+1*6+3]=-1.f;
            std::array<float,12> generalized{};
            auto jb=buffer(jac.data(),sizeof(jac)), gb=buffer(generalized.data(),sizeof(generalized));
            auto execute=[&](unsigned step,bool accept) {
                stand.code=accept ? MR_NUMI_HUMAN_STAND_SUCCESS:MR_NUMI_HUMAN_STAND_NONFINITE_RESULT;
                stand.completedSteps=accept ? step+1:step;
                std::memcpy(sb.contents,&stand,sizeof(stand));
                std::memset(gb.contents,0,gb.length);
                auto command=[queue commandBuffer];
                require(command!=nil,"physical command unavailable");
                metalrobo::MetalNumiHumanTendonLoadPass pass{};
                pass.commandBuffer=(__bridge void*)command;
                pass.bindings=(__bridge void*)bb; pass.transfers=(__bridge void*)tb;
                pass.generalizedForces=(__bridge void*)gb; pass.bodyPoses=(__bridge void*)pb;
                pass.pointJacobians=(__bridge void*)jb; pass.standStatuses=(__bridge void*)sb;
                pass.stepIndex=step; pass.environmentCount=1u; pass.endpointCount=1u;
                pass.dofCount=6u; pass.muscleCount=1u;
                pass.generalizedForceStride=6u; pass.generalizedForceOffset=6u;
                pass.pointJacobianStride=72u; pass.bodyPoseStride=1u;
                pass.bodyJacobianPointOffset=0u; pass.articulationFirstBody=0u;
                if (!program.encodePreDynamics(program.context,pass)) throw std::runtime_error("physical pre: "+physical.diagnostics().message);
                if (!program.encodePostValidation(program.context,pass)) throw std::runtime_error("physical post: "+physical.diagnostics().message);
                [command commit]; [command waitUntilCompleted];
                require(command.status==MTLCommandBufferStatusCompleted,"physical GPU command failed");
            };
            execute(0u,true);
            auto firstAccepted=runtime.snapshot();
            auto diagnostics=physical.diagnostics();
            const auto* status=static_cast<const MRNumiHumanStandStatusGPU*>(sb.contents);
            require(status->code==MR_NUMI_HUMAN_STAND_SUCCESS,"physical step rejected");
            // Runtime cursor names the accepted frame index, rather than a frame count.
            require(firstAccepted.controlStep==0u,"first accepted cursor is not frame zero");
            require(firstAccepted.femNodes[3].positionAndMass.x>before.femNodes[3].positionAndMass.x,"surface spring did not deform free tissue node");
            require(diagnostics.femSurfaceAttachmentSampleCount==1u && diagnostics.femBodyContactAuditedStepCount==1u,
                    "surface accepted history not published once");
            require(std::abs(diagnostics.femBodyContactNormalForceNewtons-.02)<1.e-6,
                    "surface force differs from original spring law");
            execute(1u,true);
            auto accepted=runtime.snapshot();
            auto acceptedAuthority=physical.snapshot();
            require(accepted.controlStep==1u && accepted.controlStep>firstAccepted.controlStep,
                    "second accepted cursor did not advance to frame one");
            require(physical.diagnostics().femBodyContactAuditedStepCount==2u,
                    "second surface accepted history missing");
            execute(2u,false);
            auto rejected=runtime.snapshot();
            auto rejectedAuthority=physical.snapshot();
            require(rejected.controlStep==accepted.controlStep,"rejected step advanced cursor");
            require(std::memcmp(rejected.femNodes.data(),accepted.femNodes.data(),accepted.femNodes.size()*sizeof(accepted.femNodes[0]))==0,
                    "rejected step changed accepted physical nodes");
            require(rejectedAuthority.femBodyContactAcceptedHistory.size()==acceptedAuthority.femBodyContactAcceptedHistory.size() &&
                    std::memcmp(rejectedAuthority.femBodyContactAcceptedHistory.data(),acceptedAuthority.femBodyContactAcceptedHistory.data(),
                                acceptedAuthority.femBodyContactAcceptedHistory.size()*sizeof(NMNumiHumanArticularContactAuditGPU))==0,
                    "rejected step advanced surface history");
            require(runtime.restore(before).encoded,"physical initial restore failed");
            require(physical.restore(initialAuthority).succeeded(),"surface initial authority restore failed");
            execute(0u,true);
            execute(1u,true);
            auto replay=runtime.snapshot();
            require(replay.controlStep==accepted.controlStep &&
                    std::memcmp(replay.femNodes.data(),accepted.femNodes.data(),accepted.femNodes.size()*sizeof(accepted.femNodes[0]))==0,
                    "surface physical replay differs");
            auto replayAuthority=physical.snapshot();
            require(sameNumiHumanTendonFEMLoadAdapterSnapshotAuthorityV1(acceptedAuthority,replayAuthority),
                    "surface accepted authority replay differs");
            std::cout << "PASS surface spring and inactive bounded bank actual Runtime: force_n=" << diagnostics.femBodyContactNormalForceNewtons
                      << " free_node_dx_m=" << accepted.femNodes[3].positionAndMass.x-before.femNodes[3].positionAndMass.x
                      << " accepted_steps=2 rejected_nodes_cursor_history=bitwise replay=bitwise\n";
        } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
    }
    return 0;
}
