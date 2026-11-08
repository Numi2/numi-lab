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
template <class T> void req(const T &d, const char *m) {
  if (!d.succeeded())
    throw std::runtime_error(std::string(m) + ": " + d.message);
}
void require(bool b, const char *m) {
  if (!b)
    throw std::runtime_error(m);
}
constexpr unsigned E = 2, N = 14;
struct Capture {
  MetalWorldDeviceActionProgram inner;
  MetalWorldDeviceActionPass last{};
  __strong id<MTLBuffer> original;
  __strong id<MTLBuffer> diagnostic;
  __strong id<MTLBuffer> reset;
  __strong id<MTLBuffer> originalReset;
  __strong id<MTLBuffer> world;
  __strong id<MTLBuffer> contact;
  bool skipRestore;
};
static bool encode(void *x, const MetalWorldDeviceActionPass &p) noexcept {
  auto *c = static_cast<Capture *>(x);
  c->last = p;
  if (!c->inner.encode(c->inner.context, p))
    return false;
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)p.commandBuffer;
  id<MTLBuffer> actions = (__bridge id<MTLBuffer>)p.taskActions;
  id<MTLBuffer> resets = (__bridge id<MTLBuffer>)p.resetMasks;
  auto b = [cb blitCommandEncoder];
  if (!b)
    return false;
  NSUInteger bytes = E * p.actionCount * sizeof(float);
  [b copyFromBuffer:actions
           sourceOffset:p.controlStep * bytes
               toBuffer:c->original
      destinationOffset:p.controlStep * bytes
                   size:bytes];
  [b copyFromBuffer:resets
           sourceOffset:p.resetMaskOffsetElements * sizeof(unsigned)
               toBuffer:c->originalReset
      destinationOffset:p.controlStep * E * sizeof(unsigned)
                   size:E * sizeof(unsigned)];
  [b endEncoding];
  return true;
}
static bool after(void *x,
                  const MetalWorldDeviceObservationStepResult &p) noexcept {
  auto *c = static_cast<Capture *>(x);
  if (!c->skipRestore && c->inner.afterStep &&
      !c->inner.afterStep(c->inner.context, p))
    return false;
  if (p.controlStep != N - 1)
    return true;
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)p.commandBuffer;
  id<MTLBuffer> world = (__bridge id<MTLBuffer>)p.worldStatuses;
  id<MTLBuffer> contact = (__bridge id<MTLBuffer>)p.contactStatuses;
  auto blit = [cb blitCommandEncoder];
  if (!blit)
    return false;
  [blit copyFromBuffer:world
           sourceOffset:0
               toBuffer:c->world
      destinationOffset:0
                   size:E * sizeof(MRMetalWorldStatusGPU)];
  [blit copyFromBuffer:contact
           sourceOffset:0
               toBuffer:c->contact
      destinationOffset:0
                   size:E * sizeof(MRMetalWorldContactStatusGPU)];
  [blit endEncoding];
  // A diagnostic second inference after the final physical step observes the
  // retained state; it is not a simulated retry or an additional action step.
  auto probe = c->last;
  probe.taskActions = (__bridge void *)c->diagnostic;
  probe.resetMasks = (__bridge void *)c->reset;
  probe.resetMaskOffsetElements = 0;
  return c->inner.encode(c->inner.context, probe);
}
int main(int argc, char **argv) {
  try {
    require(argc == 2, "shader path");
    RunManifest m;
    m.id = "hyper_capacity_probe";
    m.robot = *builtinRobotPack("franka_panda");
    m.scene = makeFrankaPickPlaceScenePack();
    m.sensors.id = "probe_sensors";
    m.task =
        makeFrankaPickPlaceTaskPack(m.sensors.observation, m.reality.reset);
    m.reality.id = "nominal";
    m.profile.id = "probe";
    m.profile.environmentCount = 2;
    m.profile.controlSteps = 30;
    m.profile.physicsSubsteps = 4;
    m.profile.controlTimestepSeconds = .02f;
    m.profile.capacities.rawContacts = 1;
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
    MetalHyperPolicyRuntime policy;
    MetalHyperPolicyConfiguration pc;
    pc.metallibPath = argv[1];
    req(createMetalHyperPolicyRuntime(cp, t, pc, policy), "create policy");
    MetalWorldContext world;
    MetalWorldResidentState state;
    MetalWorldStepConfig cfg;
    cfg.timestepSeconds = .02f;
    cfg.physicsSubsteps = 4;
    cfg.actuationMode = MetalWorldActuationMode::implicitPositionDrive;
    cfg.taskProgram = t;
    cfg.ccdMode = MetalWorldCCDMode::disabled;
    auto inner = policy.actionProgram();
    require(inner.afterStep, "missing policy acceptance callback");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    auto alloc = [&](size_t size) {
      return [dev newBufferWithLength:size
                              options:MTLResourceStorageModeShared];
    };
    Capture c{inner,
              {},
              alloc(N * E * l.actionCount * sizeof(float)),
              alloc(N * E * l.actionCount * sizeof(float)),
              alloc(E * sizeof(unsigned)),
              alloc(N * E * sizeof(unsigned)),
              alloc(E * sizeof(MRMetalWorldStatusGPU)),
              alloc(E * sizeof(MRMetalWorldContactStatusGPU)),
              std::getenv("NUMI_SKIP_POLICY_RESTORE") != nullptr};
    require(c.original && c.diagnostic && c.reset && c.originalReset &&
                c.world && c.contact,
            "capture allocations");
    std::memset(c.reset.contents, 0, c.reset.length);
    cfg.deviceActionProgram = inner;
    cfg.deviceActionProgram.context = &c;
    cfg.deviceActionProgram.encode = &encode;
    cfg.deviceActionProgram.afterStep = &after;
    std::vector<float> q(E * run.world().nq()), v(E * run.world().nv(), 0);
    std::vector<MRBodyStateGPU> scene;
    for (unsigned e = 0; e < E; ++e) {
      std::copy(run.model().defaultQ.begin(), run.model().defaultQ.end(),
                q.begin() + e * run.world().nq());
      scene.insert(scene.end(), run.defaultSceneBodies().begin(),
                   run.defaultSceneBodies().end());
    }
    require(scene.size() == 8, "scene size");
    scene[0].position.z += .15f;
    scene[3].position.z += 2;
    scene[4].position.z += 2;
    scene[7].position.z += 2;
    std::vector<unsigned> reset(N * E, 0);
    MetalWorldBatch batch{.environmentCount = E,
                          .controlStepCount = N,
                          .initialQ = q,
                          .initialV = v,
                          .policyRevision = 1,
                          .resetMasks = reset,
                          .initialSceneBodies = scene};
    MetalWorldSubmission sub;
    auto d = world.initializeResidentState(run.world(), batch, cfg, state, sub);
    req(d, "submit");
    MetalWorldResult out;
    d = sub.wait(out);
    require(d.published, "expected published outcome");
    std::vector<float> phase, action, lat, log;
    require(policy.copyRolloutTrace(N, E, phase, action, lat, log), "trace");
    auto *first = static_cast<float *>(c.original.contents);
    auto *next = static_cast<float *>(c.diagnostic.contents);
    auto *masks = static_cast<unsigned *>(c.originalReset.contents);
    auto *ws = static_cast<MRMetalWorldStatusGPU *>(c.world.contents);
    auto *cs = static_cast<MRMetalWorldContactStatusGPU *>(c.contact.contents);
    std::cout << std::setprecision(10)
              << "{\"success\":" << d.successfulStepCount
              << ",\"failed\":" << d.failedStepCount << ",\"rows\":[";
    for (unsigned e = 0; e < E; ++e) {
      if (e)
        std::cout << ",";
      size_t index = ((N - 1) * E + e) * l.actionCount;
      std::cout << "{\"world\":" << ws[e].code << ",\"contact\":" << cs[e].code
                << ",\"reset\":" << masks[(N - 1) * E + e]
                << ",\"original\":" << first[index]
                << ",\"next\":" << next[index]
                << ",\"phase\":" << phase[(N - 1) * E + e] << "}";
    }
    auto floats = [](const char *name, const std::vector<float> &a) {
      std::cout << "],\"" << name << "\":[";
      for (size_t i = 0; i < a.size(); ++i) {
        if (i)
          std::cout << ",";
        std::cout << a[i];
      }
    };
    floats("q", out.finalQ);
    floats("v", out.finalV);
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
