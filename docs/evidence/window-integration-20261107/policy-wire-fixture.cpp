#include "metalrobo/LearningPacks.hpp"
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    metalrobo::PolicyPack p;
    p.id = "wire-fixture"; p.revision = 12;
#ifdef LEGACY_POLICY
    p.contract.version = 1;
#else
    p.contract.version = 2;
    p.contract.compatibleTasks = {{11, 22}, {11, 23}};
#endif
    p.contract.worldFingerprint = 11; p.contract.taskFingerprint = 22;
    p.contract.observationFingerprint = 33; p.contract.actionFingerprint = 44;
    p.observationMean = {0.25f, -0.25f};
    p.observationInverseStandardDeviation = {1.0f, 2.0f};
    metalrobo::PolicyDenseLayer layer;
    layer.inputCount = 2; layer.outputCount = 1;
    layer.activation = metalrobo::PolicyActivation::identity;
    layer.weights = {0.5f, -0.5f}; layer.bias = {0.125f};
    p.layers.push_back(layer);
    p.actionBias = {0.0f}; p.actionScale = {1.0f};
    auto written = metalrobo::writePolicyPack(p, argv[1]);
    if (!written.succeeded()) { std::cerr << written.message << "\n"; return 1; }
    metalrobo::PolicyPack restored;
    auto read = metalrobo::readPolicyPack(argv[1], restored);
    if (!read.succeeded() || restored.id != p.id ||
        restored.contract.version != p.contract.version) return 1;
    std::cout << "native writer and reader passed\n";
}
