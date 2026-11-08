// Synthetic ownership integration gate. This is not an anatomical or full
// NumanX runtime qualification: poses are prescribed from a two-body model.
#define main costal_cpu_fixture_main
#include "costal_cook_ownership.cpp"
#undef main
#include "numi/matter/numi_human.hpp"
#import <Metal/Metal.h>
#include <cstring>
#include <stdexcept>

static void requireNative(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
int main(int argc, const char** argv) {
 @autoreleasepool {
  try {
   requireNative(argc>=2 && argc<=4,"usage: costal_ownership_native path-to-metallib [--adapter-control] [--body-contact]");
   bool bodyContact=false, adapterControl=false;
   for(int i=2;i<argc;++i) {
     const std::string arg=argv[i];if(arg=="--body-contact") bodyContact=true;
     else if(arg=="--adapter-control") adapterControl=true;else throw std::runtime_error("unknown mode");
   }
   using namespace metalrobo;
   auto model=makePendulumModel(.1);
   auto material=numi::matter::parseMatter(R"MAT(material silicone {
    parameter density : kg/m^3 = 1100 in [900, 1300];
    parameter mu : kPa = 150 in [5, 1000] log identifiable;
    parameter lambda : MPa = 3 in [0.1, 30] log identifiable;
    model neo_hookean; energy = neo_hookean(mu, lambda); valid = J() - 0.05;
    supports mpm, fem, rigid;
    interface { static_friction = 0.80; dynamic_friction = 0.65; restitution = 0.0; adhesion = 0.0; }
    limits { minimum_J = 0.05; maximum_J = 20.0; maximum_stress = 1e9; maximum_energy_density = 1e9; }
   })MAT");
   requireNative(material.succeeded(),"material parse");
   NumiHumanCostalBinding binding;
   binding.bodyCount=2;binding.nodeCount=4;binding.tetrahedronCount=1;
   binding.atlasToWorld={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
   binding.regions.resize(14,{MR_INVALID_INDEX,0,1});binding.nodeDonorBodies={0,0,1,1};
   NumiHumanCostalCartilagePayload cart;cart.regions.resize(14);
   cart.nodes={{{0,0,0},0,1,0},{{.01f,0,0},0,2,0},{{0,.01f,0},0,0,0},{{0,0,.01f},0,0,0}};
   cart.tetrahedra.push_back({{0,1,2,3},0});
   auto cooked=compileNumiHumanCostalTissue(cart,binding,model,material.material,1e-4);
   requireNative(cooked.succeeded(),cooked.error.c_str());
   std::string error;
   requireNative(validateNumiHumanCostalAttachmentOwnership(cart,binding,cooked.matter.world,error),error.c_str());
   requireNative(cooked.mass.partitions.size()==2 && cooked.matter.world.fem.humanAttachments.size()==2,"ownership coverage");
   numi::matter::CompiledWorld world=cooked.matter.world;
   if(adapterControl) {
     // Supported standalone passive-adapter control: independent fixed nodes,
     // no cooked Human attachment and therefore no primal candidate service.
     numi::matter::WorldSource source;source.environmentCount=1;source.frameTimestep=1e-4;
     source.gravity={0,0,0};source.materials.push_back(material.material);
     numi::matter::ObjectSource object;object.name="standalone_adapter_control";
     object.materialIndex=0;object.representation=numi::matter::Representation::fem;
     object.mixedFEM=false;object.deformableContact=false;object.deformableSelfContact=false;
     for(const auto& n:cart.nodes) object.femNodes.push_back({n.restPosition[0],n.restPosition[1],n.restPosition[2]});
     object.femFixedNodes={0,1};object.tetrahedra.push_back({{0,1,2,3}});source.objects.push_back(std::move(object));
     numi::matter::CompileOptions options;options.maximumRateExponent=0;
     auto result=numi::matter::compileWorld(source,options);requireNative(result.succeeded(),"standalone control cook");
     world=std::move(result.world);
   }
   numi::matter::Runtime runtime;
   const auto initialized=runtime.initialize(world,{.metallib=argv[1],.environmentCount=1,.captureEvents=true,.captureDiagnostics=true,.automaticIdentification=false,.adaptiveTransfer=false});
   requireNative(initialized.encoded && runtime.valid(),"Matter runtime admission");
   auto initial=runtime.snapshot();
   requireNative(initial.available && initial.femNodes.size()==4,"initial snapshot");
   std::vector<NMNumiHumanTendonFEMNodeLoadGPU> loads(4);
   std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> anchors(4);
   for(unsigned i=0;i<4;++i) {std::fill_n(loads[i].endpointIndex,4u,NM_INVALID_INDEX);anchors[i].bodyIndex=NM_INVALID_INDEX;}
   for(const auto& a:cooked.matter.world.fem.humanAttachments) {
     requireNative(a.identity.x<4 && a.identity.y<2,"anchor body and node");
     auto& x=anchors[a.identity.x];x.bodyIndex=a.identity.y;
     x.flags=NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;x.localPoint=a.localPoint;
   }
   numi::matter::NumiHumanTendonFEMLoadAdapter adapter;
   std::vector<NMNumiHumanTendonFEMEndpointReplacementGPU> replacements;
   NMNumiHumanFEMBodyContactSampleGPU sample{};
   sample.slaveNode=3;sample.bodyIndex=0;sample.flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE;
   sample.bodyLocalPointAndArea={0,0,.01001f,1e-5f};sample.bodyLocalNormalAndReferenceSeparation={0,0,1,0};
   sample.stiffnessAndNormalStrainPerPressure={1e6f,1e-5f,0,0};
   numi::matter::NumiHumanTendonFEMLoadSource source;source.nodeLoads=loads;source.nodeAnchors=anchors;
   source.endpointReplacements=replacements;source.endpointCount=1;source.environmentCount=1;source.productionForceOwnerFraction=0;
   if(bodyContact) source.femBodyContactSamples={&sample,1};
   requireNative(adapter.initialize(runtime,source,{.metallib=argv[1]}),"adapter admission");
   auto program=adapter.program();requireNative(program.valid(),"adapter program");
   auto& human=cooked.rebasedHuman;
   std::vector<double> q(human.defaultQ.begin(),human.defaultQ.end()), v(human.defaultV.begin(),human.defaultV.end());
   q[7]=1e-4;
   std::vector<ArticulatedBodyKinematics> bodyKinematics(2);
   requireNative(computeArticulatedBodyKinematics(human,0,q,v,bodyKinematics).succeeded(),"body kinematics");
   std::array<MRArticulatedBodyPoseGPU,2> poses{};
   for(unsigned b=0;b<2;++b) {
     auto& k=bodyKinematics[b];poses[b].position=f4(k.centerOfMassPosition[0],k.centerOfMassPosition[1],k.centerOfMassPosition[2]);
     poses[b].orientation=f4(k.orientation[0],k.orientation[1],k.orientation[2],k.orientation[3]);
   }
   std::vector<ArticulatedPointQuery> points;
   for(unsigned b=0;b<2;++b) for(unsigned p=0;p<4;++p) {
     std::array<double,3> local{};if(p>0)local[p-1]=1.0;
     points.push_back({b,local});
   }
   std::vector<ArticulatedPointKinematics> positions(8);std::vector<double> jac64(8*3*7);
   requireNative(computeArticulatedPointJacobians(human,0,q,v,points,positions,jac64).succeeded(),"articulated Jacobians");
   std::vector<float> jac(jac64.begin(),jac64.end());
   id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=[device newCommandQueue];
   MRNumiHumanTendonBindingGPU bind{};MRNumiHumanTendonTransferResultGPU transfer{};
   MRNumiHumanStandStatusGPU status{};status.code=MR_NUMI_HUMAN_STAND_SUCCESS;status.failingIndex=MR_INVALID_INDEX;
   id<MTLBuffer> bindBuffer=[device newBufferWithBytes:&bind length:sizeof(bind) options:MTLResourceStorageModeShared];
   id<MTLBuffer> transferBuffer=[device newBufferWithBytes:&transfer length:sizeof(transfer) options:MTLResourceStorageModeShared];
   id<MTLBuffer> statusBuffer=[device newBufferWithBytes:&status length:sizeof(status) options:MTLResourceStorageModeShared];
   id<MTLBuffer> poseBuffer=[device newBufferWithBytes:poses.data() length:sizeof(poses) options:MTLResourceStorageModeShared];
   id<MTLBuffer> jacBuffer=[device newBufferWithBytes:jac.data() length:jac.size()*sizeof(float) options:MTLResourceStorageModeShared];
   id<MTLBuffer> forces=[device newBufferWithLength:14*sizeof(float) options:MTLResourceStorageModeShared];
   id<MTLBuffer> reaction=(__bridge id<MTLBuffer>)runtime.femConstraintReactionBuffer();
   id<MTLBuffer> readback=[device newBufferWithLength:4*sizeof(nm_float4) options:MTLResourceStorageModeShared];
   requireNative(queue && bindBuffer && transferBuffer && statusBuffer && poseBuffer && jacBuffer && forces && reaction && readback,"Metal buffers");
   auto execute=[&](unsigned step,bool accept){
     status.code=accept?MR_NUMI_HUMAN_STAND_SUCCESS:MR_NUMI_HUMAN_STAND_NONFINITE_RESULT;
     status.completedSteps=accept?step+1:step;std::memcpy(statusBuffer.contents,&status,sizeof(status));std::memset(forces.contents,0,forces.length);
     id<MTLCommandBuffer> command=[queue commandBuffer];MetalNumiHumanTendonLoadPass pass{};
     pass.commandBuffer=(__bridge void*)command;pass.bindings=(__bridge void*)bindBuffer;pass.transfers=(__bridge void*)transferBuffer;
     pass.generalizedForces=(__bridge void*)forces;pass.bodyPoses=(__bridge void*)poseBuffer;pass.pointJacobians=(__bridge void*)jacBuffer;
     pass.standStatuses=(__bridge void*)statusBuffer;pass.stepIndex=step;pass.environmentCount=1;pass.endpointCount=1;
     pass.dofCount=7;pass.muscleCount=1;pass.generalizedForceStride=14;pass.generalizedForceOffset=7;
     pass.pointJacobianStride=168;pass.bodyJacobianPointOffset=0;pass.bodyPoseStride=2;pass.articulationFirstBody=0;
     if (!program.encodePreDynamics(program.context,pass) || !program.encodePostValidation(program.context,pass)) throw std::runtime_error("encode transaction: " + adapter.diagnostics().message);
     id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];[blit copyFromBuffer:reaction sourceOffset:0 toBuffer:readback destinationOffset:0 size:readback.length];[blit endEncoding];
     [command commit];[command waitUntilCompleted];requireNative(command.status==MTLCommandBufferStatusCompleted,"Metal completion");
   };
   execute(0,true);auto accepted=runtime.snapshot();
   requireNative(accepted.available && accepted.statuses.size()==1 && accepted.statuses[0].code==NM_STATUS_SUCCESS,"accepted Matter solve");
   auto adapterAccepted=adapter.snapshot();
   requireNative(adapterAccepted.available,"adapter accepted snapshot");
   if(bodyContact) {
     requireNative(!adapterAccepted.femBodyContactAcceptedHistory.empty(),"body contact audit missing");
     const auto& audit=adapterAccepted.femBodyContactAcceptedHistory[0];
     std::cerr<<"body_contact_audit normal_force="<<audit.normalForceAreaAndCounts.x<<" closed="<<audit.normalForceAreaAndCounts.z<<" accepted="<<audit.energyStrainClosureAndAccepted.w<<"\n";
     requireNative(audit.normalForceAreaAndCounts.x>0 && audit.normalForceAreaAndCounts.z>0 && audit.energyStrainClosureAndAccepted.w==1,"body contact not active and accepted");
   }
   std::array<nm_float4,4> acceptedReactions{};std::memcpy(acceptedReactions.data(),readback.contents,readback.length);
   std::array<float,14> acceptedForces{};std::memcpy(acceptedForces.data(),forces.contents,forces.length);
   auto magnitude=[](nm_float4 a){return std::sqrt(double(a.x)*a.x+double(a.y)*a.y+double(a.z)*a.z);};
   const double r0=magnitude(acceptedReactions[0]),r1=magnitude(acceptedReactions[1]);
   requireNative(std::isfinite(r0)&&std::isfinite(r1)&&r0>0&&r1>0,"both attachment reactions");
   double displacement=0;for(unsigned i=0;i<4;++i) {
     const auto a=accepted.femNodes[i].positionAndMass,b=initial.femNodes[i].positionAndMass;
     displacement=std::max(displacement,magnitude({a.x-b.x,a.y-b.y,a.z-b.z,0}));
   }
   requireNative(displacement>0 && std::isfinite(displacement),"loaded movement");
   execute(1,false);auto rollback=runtime.snapshot();
   requireNative(rollback.available && rollback.femNodes.size()==accepted.femNodes.size() && !std::memcmp(rollback.femNodes.data(),accepted.femNodes.data(),accepted.femNodes.size()*sizeof(NMFEMNodeStateGPU)),"rejected FEM node rollback");
   requireNative(runtime.restore(initial).encoded,"restore initial");execute(0,true);auto replay=runtime.snapshot();
   requireNative(replay.available && replay.femNodes.size()==accepted.femNodes.size() && !std::memcmp(replay.femNodes.data(),accepted.femNodes.data(),accepted.femNodes.size()*sizeof(NMFEMNodeStateGPU)) && !std::memcmp(readback.contents,acceptedReactions.data(),readback.length) && !std::memcmp(forces.contents,acceptedForces.data(),forces.length),"bitwise node, reaction and generalized force replay");
   std::cout<<(adapterControl?"synthetic standalone adapter control passed body_contact=":"synthetic Human-attachment Matter admission passed body_contact=")<<bodyContact<<" standalone_control="<<adapterControl<<" device="<<initialized.device<<" joint_angle_rad=0.0001 displacement_m="<<displacement<<" anchor_reactions_n="<<r0<<","<<r1<<" rejected_fem_nodes=restored replay_nodes_reactions_forces=bitwise\n";
   return 0;
  } catch(const std::exception& e){std::cerr<<"synthetic ownership native gate FAILED: "<<e.what()<<"\n";return 1;}
 }
}
