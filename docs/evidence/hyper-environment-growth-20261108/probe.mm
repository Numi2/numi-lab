#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalHyperPolicy.hpp"
#include "metalrobo/RunProgram.hpp"
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace metalrobo;
struct Capture {
  MetalWorldDeviceActionProgram inner;
  __strong id<MTLBuffer> actions;
  __strong id<MTLBuffer> resets;
  bool reject;
  __strong id<MTLBuffer> policyReset;
};
static bool capturePolicy(void *opaque,
                          const MetalWorldDeviceActionPass &p) noexcept {
  auto *c = static_cast<Capture *>(opaque);
  auto adjusted=p;
  if (c->policyReset != nil) {
    adjusted.resetMasks=(__bridge void*)c->policyReset;
    adjusted.resetMaskOffsetElements=0;
  }
  if (!c->inner.encode(c->inner.context, adjusted))
    return false;
  if (c->reject)
    return false;
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)p.commandBuffer;
  id<MTLBuffer> actions = (__bridge id<MTLBuffer>)p.taskActions;
  id<MTLBuffer> resets = (__bridge id<MTLBuffer>)p.resetMasks;
  id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
  if (!b)
    return false;
  NSUInteger bytes = p.environmentCount * p.actionCount * sizeof(float);
  [b copyFromBuffer:actions
           sourceOffset:p.controlStep * bytes
               toBuffer:c->actions
      destinationOffset:p.controlStep * bytes
                   size:bytes];
  NSUInteger rb = p.environmentCount * sizeof(unsigned);
  [b copyFromBuffer:resets
           sourceOffset:p.resetMaskOffsetElements * sizeof(unsigned)
               toBuffer:c->resets
      destinationOffset:p.controlStep * rb
                   size:rb];
  [b endEncoding];
  return true;
}
static bool
captureAfterStep(void *opaque,
                 const MetalWorldDeviceObservationStepResult &result) noexcept {
  auto *capture = static_cast<Capture *>(opaque);
  return capture->inner.afterStep == nullptr ||
         capture->inner.afterStep(capture->inner.context, result);
}
template <class T> void req(const T &d, const char *where) {
  if (!d.succeeded())
    throw std::runtime_error(std::string(where) + ": " + d.message);
}
void require(bool x, const char *m) {
  if (!x)
    throw std::runtime_error(m);
}
int main(int argc, char **argv) {
  try {
    require(argc == 2, "expected HyperPolicy metallib path");
    RunManifest m;
    m.id = "hyper_capacity_probe";
    m.robot = *builtinRobotPack("franka_panda");
    m.scene = makeFrankaPickPlaceScenePack();
    m.sensors.id = "probe_sensors";
    m.task =
        makeFrankaPickPlaceTaskPack(m.sensors.observation, m.reality.reset);
    m.reality.id = "nominal";
    m.profile.id = "probe";
    m.profile.environmentCount = 1;
    m.profile.controlSteps = 3;
    m.profile.physicsSubsteps = 4;
    m.profile.controlTimestepSeconds = .02f;
    m.profile.capacities.rawContacts = 256;
    m.teacher.id = "none";
    CompiledRun run;
    req(compileRun(m, run), "compile run");
    auto const &t = run.task();
    auto const &l = t.layout();
    std::cerr << "frame " << l.actorObservationSize << " actions "
              << l.actionCount << " contacts " << t.contactGroups().size()
              << "\n";
    require(t.contactGroups().size() > 0, "task needs contact group");
    HyperPolicyPack p;
    p.id = "capacity_diagnostic";
    p.revision = 1;
    p.sourceMotionFingerprint = 1;
    p.contract = {.version = 1,
                  .worldFingerprint = t.worldFingerprint(),
                  .taskFingerprint = t.fingerprint(),
                  .observationFingerprint = t.observationFingerprint(),
                  .actionFingerprint = t.actionFingerprint()};
    p.observationMean.assign(l.actorObservationSize, 0);
    p.observationInverseStandardDeviation.assign(l.actorObservationSize, 1);
    p.actionBias.assign(l.actionCount, 0);
    p.actionScale.assign(l.actionCount, 1);
    p.actionLower.assign(l.actionCount, -1);
    p.actionUpper.assign(l.actionCount, 1);
    p.maximumActionRate.assign(l.actionCount, .25f);
    HyperPolicyLayer layer;
    layer.inputCount = l.actorObservationSize;
    layer.outputCount = l.actionCount;
    layer.rank = 1;
    layer.activation = PolicyActivation::identity;
    layer.weights.assign(layer.inputCount * layer.outputCount, 0);
    layer.bias.assign(l.actionCount, 1);
    layer.adapterDown.assign(layer.inputCount, 0);
    layer.adapterUp.assign(layer.outputCount, 0);
    layer.adapterBiasBasis.assign(layer.outputCount, 0);
    p.layers = {layer};
    p.coefficientLimits = {1};
    p.knotPhases = {0, 1};
    p.coefficientKnots = {0, 0};
    p.coefficientTangents = {0, 0};
    p.authorityKnots.assign(2 * l.actionCount, 1);
    p.authorityTangents.assign(2 * l.actionCount, 0);
    p.phaseRateKnots = {1, 1};
    p.phaseRateTangents = {0, 0};
    p.referencePhases = {0, 1};
    p.referenceActions.assign(2 * l.actionCount, 0);
    p.signatureCount = 2 * l.actionCount + 10;
    p.referenceSignatures.assign(2 * p.signatureCount, 0);
    p.signatureWeights.assign(p.signatureCount, 1);
    p.contactTrackCount = 1;
    p.referenceContactMasks = {0, 0};
    p.contactGroupIndices = {0};
    p.phaseAlignmentBlend = 0;
    p.controlTimeStep = .02f;
    CompiledHyperPolicyProgram cp;
    auto hp = compileHyperPolicyProgram(p, t, cp);
    if (!hp.succeeded())
      throw std::runtime_error("compile policy " + hp.element + ": " +
                               hp.message);

    std::cout << std::setprecision(10) << "{\"runs\":[";
    bool firstOutput=true;
    for (unsigned initial : {1u,2u}) for (unsigned abort : {0u,1u}) for (unsigned repeat=0;repeat<2;++repeat) {
      MetalHyperPolicyRuntime policy;
      MetalHyperPolicyConfiguration pc;pc.metallibPath=argv[1];
      req(createMetalHyperPolicyRuntime(cp,t,pc,policy),"create policy");
      MetalWorldContext world;
      MetalWorldStepConfig cfg;cfg.timestepSeconds=.02f;cfg.physicsSubsteps=4;
      cfg.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
      cfg.taskProgram=t;cfg.ccdMode=MetalWorldCCDMode::disabled;
      auto inner=policy.actionProgram();
      id<MTLDevice> device=MTLCreateSystemDefaultDevice();
      Capture cap{inner,nil,nil,false,nil};
      cfg.deviceActionProgram=inner;cfg.deviceActionProgram.context=&cap;
      cfg.deviceActionProgram.encode=&capturePolicy;
      cfg.deviceActionProgram.afterStep=&captureAfterStep;
      std::vector<float> firstPhase,secondPhase,firstAction,secondAction;
      for (unsigned stage=0;stage<2;++stage) {
        unsigned env=stage==0?initial:2;
        std::vector<float> q,v;
        std::vector<MRBodyStateGPU> scene;
        for(unsigned i=0;i<env;++i) {
          q.insert(q.end(),run.model().defaultQ.begin(),run.model().defaultQ.end());
          v.insert(v.end(),run.world().nv(),0.f);
          auto bodies=run.defaultSceneBodies();
          scene.insert(scene.end(),bodies.begin(),bodies.end());
          scene[i*4].position.z+=2;scene[i*4+3].position.z+=2;
        }
        std::vector<unsigned> reset(env,0u);
        MetalWorldBatch b{.environmentCount=env,.controlStepCount=1,.initialQ=q,.initialV=v,
                          .policyRevision=1,.resetMasks=reset,.initialSceneBodies=scene};
        cap.actions=[device newBufferWithLength:env*l.actionCount*sizeof(float) options:MTLResourceStorageModeShared];
        cap.resets=[device newBufferWithLength:env*sizeof(unsigned) options:MTLResourceStorageModeShared];
        require(cap.actions && cap.resets,"capture");
        if(stage==1) {
          unsigned policyMask[2]={0u,1u};
          cap.policyReset=[device newBufferWithBytes:policyMask length:sizeof(policyMask) options:MTLResourceStorageModeShared];
          require(cap.policyReset != nil,"policy reset buffer");
        }
        if(stage==1 && abort) {
          cap.reject=true;
          MetalWorldResidentState discarded;MetalWorldSubmission ticket;
          auto result=world.initializeResidentState(run.world(),b,cfg,discarded,ticket);
          require(!result.succeeded() && !ticket.valid(),"expected uncommitted reject");
          cap.reject=false;
        }
        MetalWorldResidentState resident;MetalWorldSubmission ticket;
        auto result=world.initializeResidentState(run.world(),b,cfg,resident,ticket);
        req(result,"submit");
        MetalWorldResult output;result=ticket.wait(output);req(result,"wait");
        require(result.published && result.failedStepCount==0 && result.successfulStepCount==env,"physical status");
        std::vector<float> ph,ac,latent,lp;
        require(policy.copyRolloutTrace(1,env,ph,ac,latent,lp),"trace");
        auto data=static_cast<float*>(cap.actions.contents);
        if(stage==0) { firstPhase=ph;firstAction.assign(data,data+env*l.actionCount); }
        else { secondPhase=ph;secondAction.assign(data,data+env*l.actionCount); }
      }
      if(!firstOutput)std::cout << ",";firstOutput=false;
      std::cout << "{\"initial\":" << initial << ",\"abort\":" << abort << ",\"repeat\":" << repeat
                << ",\"first_phase\":" << firstPhase[0] << ",\"first_action\":" << firstAction[0]
                << ",\"phase0\":" << secondPhase[0] << ",\"phase1\":" << secondPhase[1]
                << ",\"action0\":" << secondAction[0] << ",\"action1\":" << secondAction[l.actionCount] << "}";
    }
    std::cout << "]}\n";return 0;
  } catch(const std::exception& e) {std::cerr << "failure: " << e.what() << "\n";return 2;}
}
