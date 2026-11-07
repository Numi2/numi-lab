// Research-only open-loop replay preparation. Never changes the policy pack.
#include "metalrobo/LearningPacks.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

static float action(float latent, float scale, float bias, float clip) {
    return std::clamp(std::fma(scale, latent, bias), -clip, clip);
}
static void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            check(action(0.75f, 2.f, -0.25f, 1.f) == 1.f, "positive clamp");
            check(action(-0.75f, 2.f, 0.25f, 1.f) == -1.f, "negative clamp");
            check(action(0.5f, 1.f, 0.f, 10.f) == 0.5f, "identity");
            std::cout << "action affine and clipping checks passed\n";
            return 0;
        }
        check(argc == 5, "usage: extract_actions POLICY ROLLOUT EXPECTED_POLICY_FINGERPRINT OUTPUT");
        metalrobo::PolicyPack policy;
        metalrobo::PolicyRolloutPack rollout;
        auto p = metalrobo::readPolicyPack(argv[1], policy);
        auto r = metalrobo::readPolicyRolloutPack(argv[2], rollout);
        check(p.succeeded() && r.succeeded(), "pack read failed");
        const auto expected = std::stoull(argv[3]);
        check(policy.contract.exact(), "policy contract not exact");
        check(policy.contract.worldFingerprint == 3978934575658225183ull &&
              policy.contract.taskFingerprint == 922021427268694894ull &&
              policy.contract.observationFingerprint == 15900562835394482329ull &&
              policy.contract.actionFingerprint == 9497262916478774679ull,
              "unexpected policy contract");
        check(rollout.taskFingerprint == policy.contract.taskFingerprint &&
              rollout.policyFingerprint == expected && expected != 0 &&
              rollout.policyRevision == policy.revision, "rollout identity mismatch");
        check(rollout.environmentCount == 8 && rollout.controlStepCount == 1000 &&
              rollout.actionCount == 29, "unexpected rollout dimensions");
        check(policy.actionBias.empty() || policy.actionBias.size() == rollout.actionCount, "bias width");
        check(policy.actionScale.empty() || policy.actionScale.size() == rollout.actionCount, "scale width");
        check(std::isfinite(policy.actionClip) && policy.actionClip >= 1.f, "clip must admit intervention range");
        const std::size_t count = std::size_t(rollout.environmentCount) * rollout.controlStepCount * rollout.actionCount;
        check(rollout.latents.size() == count, "latent size");
        std::ofstream out(argv[4], std::ios::binary | std::ios::trunc);
        check(out.good(), "output open failed");
        for (std::size_t i=0; i<count; ++i) {
            const auto j = i % rollout.actionCount;
            const float scale = policy.actionScale.empty() ? 1.f : policy.actionScale[j];
            const float bias = policy.actionBias.empty() ? 0.f : policy.actionBias[j];
            check(std::isfinite(rollout.latents[i]) && std::isfinite(scale) && std::isfinite(bias), "nonfinite input");
            const float a = action(rollout.latents[i], scale, bias, policy.actionClip);
            check(std::isfinite(a), "nonfinite action");
            out.write(reinterpret_cast<const char*>(&a), sizeof(a));
        }
        out.close();
        check(out.good(), "output write failed");
        std::cout << "policy_fingerprint=" << rollout.policyFingerprint << " action_count=" << count << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
