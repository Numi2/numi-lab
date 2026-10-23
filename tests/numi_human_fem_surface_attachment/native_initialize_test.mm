#import <Foundation/Foundation.h>
#include "numi/matter/matter.hpp"
#include "numi/matter/numi_human.hpp"
#include <algorithm>
#include <array>
#include <cstring>
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
            std::cout << "PASS native Source initialization: empty/legacy/surface/mixed, 10 malformed cases, snapshot authority, unchanged physical nodes; no time advanced\n";
        } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
    }
    return 0;
}
