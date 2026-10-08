#include "../apps/NumiHumanAcceptedGeometryCadence.hpp"
#include <cstdlib>
#include <iostream>

namespace {
void expect(bool condition,const char* message) {
    if(!condition){std::cerr<<"FAIL: "<<message<<'\n';std::exit(1);}
}
}

int main() {
    using numiHumanRestingAcceptedGeometry::CaptureStepClass;
    using numiHumanRestingAcceptedGeometry::classifyCaptureStep;
    using numiHumanRestingAcceptedGeometry::terminalSnapshotReady;
    expect(classifyCaptureStep(0u,100u,32u)==CaptureStepClass::initial,
        "initial accepted state remains a valid capture");
    expect(classifyCaptureStep(31u,100u,32u)==CaptureStepClass::submission_endpoint,
        "ordinary zero-based submission endpoint remains valid");
    expect(classifyCaptureStep(99u,100u,32u)==CaptureStepClass::invalid,
        "N-1 is not mislabeled terminal when it is not a cadence endpoint");
    expect(classifyCaptureStep(100u,100u,32u)==CaptureStepClass::terminal,
        "true accepted terminal state N is valid");
    expect(classifyCaptureStep(127u,128u,32u)==CaptureStepClass::submission_endpoint,
        "N-1 remains valid when it independently ends a regular submission");
    expect(classifyCaptureStep(128u,128u,32u)==CaptureStepClass::terminal,
        "terminal N remains distinct after a regular N-1 endpoint");
    expect(classifyCaptureStep(101u,100u,32u)==CaptureStepClass::invalid,
        "capture beyond the accepted horizon is rejected");
    expect(classifyCaptureStep(0u,0u,32u)==CaptureStepClass::invalid,
        "empty accepted horizon is rejected");
    expect(classifyCaptureStep(1u,100u,0u)==CaptureStepClass::invalid,
        "zero submission cadence is rejected");
    expect(terminalSnapshotReady(100u,100u,true,100u,true),
        "terminal publication requires exact accepted body and respiration clocks");
    expect(!terminalSnapshotReady(0u,0u,true,0u,true),
        "zero-step horizon cannot be published as a terminal state");
    expect(!terminalSnapshotReady(100u,99u,true,99u,true),
        "failed or incomplete body horizon cannot publish a terminal frame");
    expect(!terminalSnapshotReady(100u,100u,false,100u,true),
        "rejected body endpoint cannot publish a terminal frame");
    expect(!terminalSnapshotReady(100u,100u,true,99u,true),
        "stale respiratory state cannot publish a terminal frame");
    expect(!terminalSnapshotReady(100u,100u,true,100u,false),
        "rejected respiratory state cannot publish a terminal frame");
    return 0;
}
