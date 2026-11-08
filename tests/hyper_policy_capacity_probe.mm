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
};
static bool capturePolicy(void *opaque,
                          const MetalWorldDeviceActionPass &p) noexcept {
  auto *c = static_cast<Capture *>(opaque);
  if (!c->inner.encode(c->inner.context, p))
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
    bool firstRun = true;
    for (auto sizes :
         std::vector<std::vector<unsigned>>{{1, 2}, {2, 1}, {1, 1, 1}})
      for (int repeat = 0; repeat < 2; ++repeat) {
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
        auto original = policy.actionProgram();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        Capture cap{original, nil, nil, false};
        cfg.deviceActionProgram = original;
        cfg.deviceActionProgram.context = &cap;
        cfg.deviceActionProgram.encode = &capturePolicy;
        cfg.ccdMode = MetalWorldCCDMode::disabled;
        std::vector<float> q = run.model().defaultQ, v(run.world().nv(), 0);
        auto sceneSpan = run.defaultSceneBodies();
        std::vector<MRBodyStateGPU> scene(sceneSpan.begin(), sceneSpan.end());
        require(scene.size() == 4, "scene size");
        scene[0].position.z += 2;
        scene[3].position.z += 2;
        std::vector<float> phases, action0, physical0, finalq, finalv;
        std::vector<MRBodyStateGPU> finalScene;
        std::vector<unsigned> statuses, terms, actualResets;
        bool init = true;
        for (unsigned n : sizes) {
          cap.actions =
              [device newBufferWithLength:n * l.actionCount * sizeof(float)
                                  options:MTLResourceStorageModeShared];
          cap.resets =
              [device newBufferWithLength:n * sizeof(unsigned)
                                  options:MTLResourceStorageModeShared];
          require(cap.actions && cap.resets, "capture buffers");
          std::vector<unsigned> reset(n, 0);
          MetalWorldBatch b{.environmentCount = 1,
                            .controlStepCount = n,
                            .policyRevision = 1,
                            .resetMasks = reset};
          if (init) {
            b.initialQ = q;
            b.initialV = v;
            b.initialSceneBodies = scene;
          }
          bool abortNow = (std::getenv("ABORT_FRESH") && init) ||
                          (std::getenv("ABORT_GROW") && !init && n == 2);
          if (abortNow) {
            cap.reject = true;
            MetalWorldSubmission abandoned;
            auto failure = init ? world.initializeResidentState(
                                      run.world(), b, cfg, state, abandoned)
                                : world.submitResident(run.world(), b, cfg,
                                                       state, abandoned);
            require(!failure.succeeded() && !abandoned.valid(),
                    "expected uncommitted rejection");
            cap.reject = false;
            std::cerr << "aborted init=" << init << " count=" << n
                      << " message=" << failure.message << "\n";
          }
          MetalWorldSubmission sub;
          auto d = init ? world.initializeResidentState(run.world(), b, cfg,
                                                        state, sub)
                        : world.submitResident(run.world(), b, cfg, state, sub);
          req(d, "submit");
          MetalWorldResult out;
          d = sub.wait(out);
          req(d, "wait");
          require(d.failedStepCount == 0 && d.successfulStepCount == n &&
                      d.published,
                  "physical failure or unpublished");
          std::vector<float> ph, ac, lat, lp;
          require(policy.copyRolloutTrace(n, 1, ph, ac, lat, lp), "trace");
          require(ph.size() == n && ac.size() == n * l.actionCount,
                  "trace sizes");
          for (unsigned i = 0; i < n; ++i) {
            phases.push_back(ph[i]);
            action0.push_back(ac[i * l.actionCount]);
            physical0.push_back(
                static_cast<float *>(cap.actions.contents)[i * l.actionCount]);
            actualResets.push_back(
                static_cast<unsigned *>(cap.resets.contents)[i]);
          }
          for (auto x : out.statuses)
            statuses.push_back(x.code);
          for (auto x : out.transitions)
            terms.push_back(x.termination.z);
          finalq = out.finalQ;
          finalv = out.finalV;
          finalScene = out.finalSceneBodies;
          require(finalScene.size() == scene.size(), "final scene size");
          init = false;
        }
        if (!firstRun)
          std::cout << ",";
        firstRun = false;
        std::cout << "{\"sizes\":[";
        for (size_t i = 0; i < sizes.size(); ++i) {
          if (i)
            std::cout << ",";
          std::cout << sizes[i];
        }
        std::cout << "],\"repeat\":" << repeat;
        auto floats = [](const char *name, const std::vector<float> &a) {
          std::cout << ",\"" << name << "\":[";
          for (size_t i = 0; i < a.size(); ++i) {
            if (i)
              std::cout << ",";
            std::cout << a[i];
          }
          std::cout << "]";
        };
        floats("phases", phases);
        floats("action0", action0);
        floats("physical0", physical0);
        floats("q", finalq);
        floats("v", finalv);
        std::cout << ",\"scene_bytes\":\"";
        const auto *sceneBytes =
            reinterpret_cast<const unsigned char *>(finalScene.data());
        for (std::size_t i = 0; i < finalScene.size() * sizeof(MRBodyStateGPU);
             ++i) {
          std::cout << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned>(sceneBytes[i]);
        }
        std::cout << std::dec << std::setfill(' ') << "\"";
        std::cout << ",\"statuses\":[";
        for (size_t i = 0; i < statuses.size(); ++i) {
          if (i)
            std::cout << ",";
          std::cout << statuses[i];
        }
        std::cout << "],\"resets\":[";
        for (size_t i = 0; i < actualResets.size(); ++i) {
          if (i)
            std::cout << ",";
          std::cout << actualResets[i];
        }
        std::cout << "],\"physics_errors\":[";
        for (size_t i = 0; i < terms.size(); ++i) {
          if (i)
            std::cout << ",";
          std::cout << terms[i];
        }
        std::cout << "]}";
      }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
