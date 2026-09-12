#pragma once
#include "numi/matter/matter.hpp"

// Synthetic constitutive-verification data, never calibrated Human physiology.
inline numi::matter::VascularNetworkSource vascularFixture() {
    using namespace numi::matter;
    VascularNetworkSource n;
    // SHA256 of the synthetic source label; physical values are separately
    // bound by compileWorld physics identity for every configured variation.
    // Numi Matter synthetic passive two-pool tracer fixture v1; no biological calibration
    n.contentIdentity={0xf31cc66c3aaeed67ull,0x6aced56dc02e7669ull,0xec18d60ef21c6795ull,0x142f2cfaab27369aull};
    n.species.push_back({1,"fixture:tracer",1e-6,1e-5});
    n.compartments.push_back({2,"fixture:blood-a",1e-6,2e-6,0,1e-9,0,1e-6,1e-5,{2e-6}});
    n.compartments.push_back({3,"fixture:blood-b",1e-6,1e-6,0,1e-9,0,1e-6,1e-5,{0}});
    n.connections.push_back({4,2,3,1e9,0,0,1e-6,1000,1e-5});
    VascularTissueSource t;t.stableIdentifier=5;t.anatomicalIdentifier="fixture:organ";t.volume=1e-6;t.initialSpeciesAmounts={0};
    n.tissues.push_back(t);n.exchanges.push_back({6,3,5,1,1e-7,1});
    return n;
}
