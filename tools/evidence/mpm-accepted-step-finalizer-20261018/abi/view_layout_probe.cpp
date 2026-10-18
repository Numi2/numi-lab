#include <numi/matter/matter.hpp>
#include <cstddef>
#include <iostream>
using View = numi::matter::AcceptedStepExtensionView;
int main() {
    std::cout << "sizeof=" << sizeof(View) << "\n";
    std::cout << "commandBuffer=" << offsetof(View, commandBuffer) << "\n";
    std::cout << "matterStatuses=" << offsetof(View, matterStatuses) << "\n";
    std::cout << "vascularAccepted=" << offsetof(View, vascularAccepted) << "\n";
    std::cout << "vascularCandidate=" << offsetof(View, vascularCandidate) << "\n";
    std::cout << "vascularCheckpoint=" << offsetof(View, vascularCheckpoint) << "\n";
    std::cout << "vascularUnknowns=" << offsetof(View, vascularUnknowns) << "\n";
    std::cout << "vascularCompartments=" << offsetof(View, vascularCompartments) << "\n";
    std::cout << "vascularElastance=" << offsetof(View, vascularElastance) << "\n";
    std::cout << "vascularConnections=" << offsetof(View, vascularConnections) << "\n";
    std::cout << "vascularTissues=" << offsetof(View, vascularTissues) << "\n";
    std::cout << "vascularExchanges=" << offsetof(View, vascularExchanges) << "\n";
    std::cout << "vascularCavities=" << offsetof(View, vascularCavities) << "\n";
    std::cout << "vascularCompartmentCavity=" << offsetof(View, vascularCompartmentCavity) << "\n";
    std::cout << "vascularClockAccepted=" << offsetof(View, vascularClockAccepted) << "\n";
    std::cout << "vascularClockCandidate=" << offsetof(View, vascularClockCandidate) << "\n";
    std::cout << "vascularClockCheckpoint=" << offsetof(View, vascularClockCheckpoint) << "\n";
    std::cout << "vascularLayout=" << offsetof(View, vascularLayout) << "\n";
    std::cout << "environmentCount=" << offsetof(View, environmentCount) << "\n";
    std::cout << "vascularStateStride=" << offsetof(View, vascularStateStride) << "\n";
    std::cout << "compartmentCount=" << offsetof(View, compartmentCount) << "\n";
    std::cout << "connectionCount=" << offsetof(View, connectionCount) << "\n";
    std::cout << "tissueCount=" << offsetof(View, tissueCount) << "\n";
    std::cout << "exchangeCount=" << offsetof(View, exchangeCount) << "\n";
    std::cout << "cavityCount=" << offsetof(View, cavityCount) << "\n";
    std::cout << "controlStep=" << offsetof(View, controlStep) << "\n";
    std::cout << "physicsSubstep=" << offsetof(View, physicsSubstep) << "\n";
    std::cout << "physicsSubsteps=" << offsetof(View, physicsSubsteps) << "\n";
    std::cout << "microtick=" << offsetof(View, microtick) << "\n";
    std::cout << "microtickCount=" << offsetof(View, microtickCount) << "\n";
    std::cout << "vascularClockQuantumExponent=" << offsetof(View, vascularClockQuantumExponent) << "\n";
    std::cout << "matterPhase=" << offsetof(View, matterPhase) << "\n";
    std::cout << "vascularTimestepTicks=" << offsetof(View, vascularTimestepTicks) << "\n";
    std::cout << "frameTimestepSeconds=" << offsetof(View, frameTimestepSeconds) << "\n";
    std::cout << "microstepTimestepSeconds=" << offsetof(View, microstepTimestepSeconds) << "\n";
#ifdef MPM_EXTENSION_LAYOUT
    std::cout << "particleAccepted=" << offsetof(View, particleAccepted) << "\n";
    std::cout << "particleCandidate=" << offsetof(View, particleCandidate) << "\n";
    std::cout << "particleCheckpoint=" << offsetof(View, particleCheckpoint) << "\n";
    std::cout << "particleMaterialStateDefaults=" << offsetof(View, particleMaterialStateDefaults) << "\n";
    std::cout << "particleMaterialStateAccepted=" << offsetof(View, particleMaterialStateAccepted) << "\n";
    std::cout << "particleMaterialStateCandidate=" << offsetof(View, particleMaterialStateCandidate) << "\n";
    std::cout << "particleMaterialStateCheckpoint=" << offsetof(View, particleMaterialStateCheckpoint) << "\n";
    std::cout << "particleCount=" << offsetof(View, particleCount) << "\n";
    std::cout << "materialStateStride=" << offsetof(View, materialStateStride) << "\n";
#endif
    return 0;
}
