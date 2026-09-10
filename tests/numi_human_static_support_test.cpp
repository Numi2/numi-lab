#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/NumiHumanSupport.hpp"
#include <cstring>
#include "metalrobo/NumiHumanMuscleEquilibrium.hpp"
#include "metalrobo/NumiHumanCompliantEquilibrium.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {
void require(const bool value, const std::string_view message) {
    if (value) return;
    std::cerr << "numi_human_static_support_test=failed error=\""
              << message << "\"\n";
    std::exit(1);
}
bool near(const double a, const double b, const double tolerance = 1.0e-7) {
    return std::abs(a - b) <= tolerance;
}
mr_float4 f4(
    const double x,
    const double y,
    const double z,
    const double w = 0.0
) {
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z),
        static_cast<float>(w),
    };
}

MRDofPropertiesGPU rootDof(const std::uint32_t localDof) {
    MRDofPropertiesGPU result{};
    result.articulationIndex = 0u;
    result.jointIndex = MR_INVALID_INDEX;
    result.qIndex = localDof < 3u
        ? localDof
        : MR_INVALID_INDEX;
    result.vIndex = localDof;
    result.localDof = localDof;
    result.flags = MR_DOF_FLAG_ROOT;
    return result;
}

MRBodyPropertiesGPU body(
    const std::uint32_t parent,
    const std::uint32_t inboundJoint,
    const double mass,
    const std::array<double, 3> inertia
) {
    MRBodyPropertiesGPU result{};
    result.articulationIndex = 0u;
    result.parentBody = parent;
    result.inboundJoint = inboundJoint;
    result.motionType = MR_MOTION_DYNAMIC;
    result.massAndInverseMass = f4(mass, 1.0 / mass, 0.0, 0.0);
    result.inertiaRow0 = f4(inertia[0], 0.0, 0.0);
    result.inertiaRow1 = f4(0.0, inertia[1], 0.0);
    result.inertiaRow2 = f4(0.0, 0.0, inertia[2]);
    result.inverseInertiaRow0 =
        f4(1.0 / inertia[0], 0.0, 0.0);
    result.inverseInertiaRow1 =
        f4(0.0, 1.0 / inertia[1], 0.0);
    result.inverseInertiaRow2 =
        f4(0.0, 0.0, 1.0 / inertia[2]);
    result.dampingAndSpeedLimits =
        f4(0.0, 0.0, 1.0e6, 1.0e6);
    return result;
}

metalrobo::EngineModel makeFreeBodyModel() {
    metalrobo::EngineModel model;
    model.name = "analytic_contact_free_body";
    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FLOATING;
    articulation.firstBody = 0u;
    articulation.bodyCount = 1u;
    articulation.firstJoint = 0u;
    articulation.jointCount = 0u;
    articulation.qOffset = 0u;
    articulation.nq = 7u;
    articulation.vOffset = 0u;
    articulation.nv = 6u;
    model.articulations.push_back(articulation);
    for (std::uint32_t localDof = 0u;
         localDof < 6u;
         ++localDof) {
        model.dofs.push_back(rootDof(localDof));
    }
    model.bodies.push_back(body(
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        2.5,
        {0.7, 1.1, 1.6}
    ));
    model.defaultQ = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    model.defaultV.assign(6u, 0.0f);
    return model;
}


struct Fixture {
    metalrobo::EngineModel model = makeFreeBodyModel();
    std::vector<double> q;
    const std::vector<metalrobo::MujocoMuscleSite> sites{
        {0u, {-0.5, 0.0, 0.0}}, {0u, {0.5, 0.0, 0.0}}};
    std::vector<metalrobo::MujocoMuscleDefinition> muscles;
    const std::vector<metalrobo::MujocoCompliantMuscleArchitecture> architectures{1u};
    metalrobo::NumiHumanMuscleEquilibriumConfig config;

    Fixture() {
        model.world.gravityAndTimestep = {0.0f, 0.0f, -9.81f, 0.0001f};
        q.assign(model.defaultQ.begin(), model.defaultQ.end());
        metalrobo::MujocoMuscleDefinition muscle;
        muscle.route = {
            {metalrobo::MujocoRouteNodeType::site, 0u},
            {metalrobo::MujocoRouteNodeType::site, 1u}};
        muscle.lengthRange = {0.8, 1.2};
        muscle.accelerationScale = 1.0;
        muscle.controlRange = {0.0, 1.0};
        muscle.gainParameters = {0.75, 1.05, 1.0, 200.0, 0.5, 1.6,
                                 1.5, 1.3, 1.2, 0.0};
        muscle.biasParameters = muscle.gainParameters;
        muscle.dynamicParameters = {0.01, 0.04, 0.0};
        muscles.push_back(muscle);
        config.activationSamples = 3u;
        config.activationSweeps = 8u;
        config.poseSweeps = 0u;
        config.globalActivationPolishIterations = 2u;
    }

    auto compile(const std::vector<metalrobo::NumiHumanStaticSupportContact>& supports,
                 metalrobo::NumiHumanMuscleEquilibriumResult& result) const {
        return metalrobo::compileNumiHumanMuscleEquilibrium(
            model, 0u, q, sites, {}, muscles, architectures, {}, {}, supports,
            result, config);
    }
    double weight() const {
        return -2.5 * static_cast<double>(model.world.gravityAndTimestep.z);
    }
};
} // namespace

void testCurvedSupport() {
    using namespace metalrobo;
    auto model = makeFreeBodyModel();
    std::vector<double> q{0,0,0, std::sqrt(0.5),0,0,std::sqrt(0.5)};
    std::vector<double> v{0,0,0,2,0,0};
    std::vector<ArticulatedPointQuery> queries{{0u, {0,0,0}, 0.2, {0,0,1}}};
    std::vector<ArticulatedPointKinematics> points(1);
    std::vector<double> jac(18);
    require(computeArticulatedPointJacobians(model,0,q,v,queries,points,jac).succeeded(),
        "sphere query rejected");
    require(near(points[0].position[2],-0.2) && near(points[0].position[1],0) &&
            near(points[0].linearVelocity[1],0.4) && near(jac[6+3],0.2),
        "sphere surface or material friction velocity is wrong");
    // Rotating about an offset centre changes normal gap; its derivative is n*J.
    const double angle=0.31, eps=1.0e-6;
    queries[0].localPoint={0.5,0,0};
    q={0,0,0,0,std::sin(angle/2),0,std::cos(angle/2)};
    require(computeArticulatedPointJacobians(model,0,q,v,queries,points,jac).succeeded(),"tilted sphere failed");
    const double derivative=jac[12+4];
    double gap[2];
    for (unsigned i=0;i<2;++i) {
        const double a=angle+(i ? eps : -eps);
        q[4]=std::sin(a/2);q[6]=std::cos(a/2);
        require(computeArticulatedPointJacobians(model,0,q,v,queries,points,jac).succeeded(),"gap derivative failed");
        gap[i]=points[0].position[2];
    }
    require(near((gap[1]-gap[0])/(2*eps),derivative,1.0e-9),"sphere normal-gap Jacobian disagrees with geometry");
    const auto kept=points[0].position;
    for (double bad : {-0.1, std::numeric_limits<double>::quiet_NaN()}) {
        queries[0].supportRadius=bad;
        require(!computeArticulatedPointJacobians(model,0,q,v,queries,points,jac).succeeded() && points[0].position==kept,
            "bad radius published point output");
    }
    queries[0].supportRadius=0.2;queries[0].supportPlaneNormal={0,0,2};
    require(!computeArticulatedPointJacobians(model,0,q,v,queries,points,jac).succeeded(),"nonunit sphere plane admitted");
    queries[0].supportPlaneNormal={0,0.6,0.8};queries[0].localPoint={0,0,0};
    require(computeArticulatedPointJacobians(model,0,q,v,queries,points,jac).succeeded() &&
            near(points[0].position[1],-0.12) && near(points[0].position[2],-0.16),"oblique sphere plane failed");
    const std::vector<NumiHumanStaticSupportContact> supports{
        {.bodyIndex=0,.normal={0,0.6,0.8},.planePoint={0,0.3,0.4},.supportRadius=0.2}};
    const std::vector<NumiHumanSupportPoseCoordinate> coordinates{{1u,2.0},{2u,2.0}};
    const std::vector<std::uint32_t> active{0};
    NumiHumanSupportPoseResult fitted;
    require(compileNumiHumanSupportPose(model,0,q,{},supports,active,coordinates,fitted).succeeded() &&
            near(0.6*fitted.q[1]+0.8*fitted.q[2],0.7),"stance fitted centre instead of surface");

    queries[0].supportRadius=0;
    queries[0].supportRadii={0.2,0.3,0.4}; queries[0].supportOrientation={0,0,0,1};
    queries[0].supportPlaneNormal={0,0,1};
    q={0,0,0,0,std::sqrt(0.5),0,std::sqrt(0.5)};
    require(computeArticulatedPointJacobians(model,0,q,v,queries,points,jac).succeeded() &&
            near(points[0].position[2],-0.2),"ellipsoid did not refresh its rotated semi-axis");
    queries[0].supportRadii[1]=0;
    require(!computeArticulatedPointJacobians(model,0,q,v,queries,points,jac).succeeded(),"degenerate ellipsoid admitted");

    NumiHumanSupportHeader h;
    h.magic={'N','H','C','N','T','2',0,0};h.payloadAbi=2;h.engineBodyCount=1;h.contactCount=1;
    std::vector<std::byte> bytes(84+96);
    std::memcpy(bytes.data(),&h,84);
    const std::array<std::uint32_t,4> identity{0,42,2,0};
    const std::array<float,12> primitive{-0.5,0,0,0.2, 0.5,0,0,0.7, -0.2,-0.2,0,0};
    std::memcpy(bytes.data()+84,identity.data(),16);std::memcpy(bytes.data()+100,primitive.data(),48);
    NumiHumanSupportPayload payload;std::string error;
    require(decodeNumiHumanSupportPayload(bytes,1,h.sourceSha256,payload,error) && payload.contacts.size()==2 &&
            payload.contacts[0].localPointX==-0.5f && payload.contacts[1].localPointX==0.5f &&
            payload.contacts[0].supportRadius==0.2f && payload.header.contactCount==2,
        "capsule did not compile both endpoint-sphere constraints");
    auto bad=bytes;bad.pop_back();
    require(!decodeNumiHumanSupportPayload(bad,1,h.sourceSha256,payload,error) && payload.contacts.size()==2,
        "truncated primitive published output");
    for (std::size_t offset : {std::size_t(92),std::size_t(96),std::size_t(112),std::size_t(140)}) {
        bad=bytes;const std::uint32_t invalid=0xffffffffu;std::memcpy(bad.data()+offset,&invalid,4);
        require(!decodeNumiHumanSupportPayload(bad,1,h.sourceSha256,payload,error),"malformed primitive admitted");
    }
    auto foreign=h.sourceSha256;foreign[0]=1;
    require(!decodeNumiHumanSupportPayload(bytes,1,foreign,payload,error),"foreign source primitive admitted");
    h.magic[5]='1';h.payloadAbi=1;
    bytes.resize(84+48);std::memcpy(bytes.data(),&h,84);
    NumiHumanSupportContact legacy;legacy.bodyIndex=0;legacy.sourceGeometryIndex=42;
    std::memcpy(bytes.data()+84,&legacy,48);
    require(decodeNumiHumanSupportPayload(bytes,1,h.sourceSha256,payload,error) && payload.contacts.size()==1 &&
        payload.contacts[0].supportRadius==0,"legacy witness changed semantics");
}

void testSourceCompliantPreparation() {
    using namespace metalrobo;
    NumiHumanSourceScalarLaw law{{-100,-2,0,0},{0.5f,0.5f,0.1f,0.5f},{2,0,0,0},0.4,true};
    double force=123;
    require(evaluateNumiHumanSourceStaticForce(law,0,1e-4,force) && force==0,
        "zero deformation supplied an ideal static reaction");
    require(evaluateNumiHumanSourceStaticForce(law,-0.02,1e-4,force) && near(force,10,1e-10),
        "source inverse weight or impedance was omitted from static force");
    auto invalid=law;invalid.inverseWeight=0;force=123;
    require(!evaluateNumiHumanSourceStaticForce(invalid,-0.02,1e-4,force) && force==123,
        "invalid source law changed the accepted output");
    auto positive=law;positive.solref={0.001f,1,0,0};
    double safe=0,unsafe=0;
    require(evaluateNumiHumanSourceStaticForce(positive,-0.02,0.01,safe),"REFSAFE scalar failed");
    positive.referenceSafe=false;
    require(evaluateNumiHumanSourceStaticForce(positive,-0.02,0.01,unsafe) &&
        near(unsafe/safe,std::pow(0.02/double(positive.solref.x),2),1e-8),"REFSAFE timestep law changed");
    auto model=makeFreeBodyModel();model.world.gravityAndTimestep={0,0,-9.81f,1e-4f};
    std::vector<double> q(model.defaultQ.begin(),model.defaultQ.end());
    NumiHumanCompliantEquality equality;
    equality.source.indices={2,2,MR_INVALID_INDEX,MR_INVALID_INDEX};
    equality.source.solref=law.solref;equality.source.solimp0=law.solimp0;equality.source.solimp1=law.solimp1;
    equality.inverseWeight=law.inverseWeight;
    const std::vector<NumiHumanCompliantEquality> equalities{equality};
    NumiHumanCompliantEquilibriumConfig config;config.accelerationTolerance=1e-6;
    NumiHumanCompliantEquilibriumResult result;
    const auto run=[&](auto& output) {
        return compileNumiHumanCompliantEquilibrium(model,0,q,{},{},{},{},{},equalities,{},{},output,config);
    };
    auto status=run(result);
    const double expected=2.5*double(model.world.gravityAndTimestep.z)/500.0;
    require(status.succeeded() && status.balanced && near(result.state.q[2],expected,1e-8),
        "compliant preparation did not find the analytic loaded deformation");
    require(result.objectiveHistory.size()>1 && result.objectiveHistory.back()<result.objectiveHistory.front(),
        "preparation trace omitted its converged search");
    NumiHumanCompliantEquilibriumResult replay;
    require(run(replay).succeeded() && replay.state.q==result.state.q && replay.objectiveHistory==result.objectiveHistory,
        "source-compliant preparation replay changed");
    config.maximumCoordinateDisplacement=0.001;
    require(run(replay).succeeded() && !replay.state.diagnostics.balanced && near(replay.state.q[2],-0.001,1e-9),
        "infeasible compliant preparation waived its displacement bound");
    const auto accepted=replay.state.q;q[2]=NAN;
    require(!run(replay).succeeded() && replay.state.q==accepted,"failed preparation overwrote accepted output");
    q[2]=0;config.maximumCoordinateDisplacement=0.15;
    NumiHumanCompliantLimit limit{2,2,0,1,0,law};
    const std::vector<NumiHumanCompliantLimit> limits{limit};
    status=compileNumiHumanCompliantEquilibrium(model,0,q,{},{},{},{},{},{},limits,{},result,config);
    require(status.succeeded() && status.balanced && near(result.state.q[2],expected,1e-8) &&
        result.state.generalizedPositionLimitForce[2]>0,"source lower stop did not acquire its loaded deformation");
    q[2]=5e-7;config.optimizePose=false;
    const std::vector<NumiHumanStaticSupportContact> grounded{{.bodyIndex=0}};
    status=compileNumiHumanCompliantEquilibrium(model,0,q,{},{},{},{},{},{},{},grounded,result,config);
    require(status.succeeded() && status.balanced && result.state.q==q &&
        near(result.state.supportNormalForce[0],-2.5*double(model.world.gravityAndTimestep.z),1e-7),
        "support-only admission moved the fixed prepared state or lost weight");
    q[2]=0;config.optimizePose=true;
    model.world.gravityAndTimestep.z=9.81f;limit.lower=-1;limit.upper=0;
    const std::vector<NumiHumanCompliantLimit> upper{limit};
    status=compileNumiHumanCompliantEquilibrium(model,0,q,{},{},{},{},{},{},upper,{},result,config);
    require(status.succeeded() && status.balanced && near(result.state.q[2],-expected,1e-8) &&
        result.state.generalizedPositionLimitForce[2]<0,"source upper stop did not acquire its loaded deformation");
}

int main() {
    testSourceCompliantPreparation();
    testCurvedSupport();
    using namespace metalrobo;
    Fixture fixture;
    const NumiHumanStaticSupportContact touching{.bodyIndex = 0u};
    auto separated = touching;
    separated.localPoint[2] = 0.02;
    NumiHumanMuscleEquilibriumResult accepted;
    const auto grounded = fixture.compile({touching, separated}, accepted);
    require(grounded.succeeded() && grounded.balanced,
            "analytic grounded mass did not balance");
    require(accepted.supportNormalForce.size() == 2u &&
                accepted.supportPlaneGapMeters.size() == 2u &&
                near(accepted.supportNormalForce[0], fixture.weight()) &&
                accepted.supportNormalForce[1] == 0.0 &&
                near(accepted.supportPlaneGapMeters[0], 0.0) &&
                near(accepted.supportPlaneGapMeters[1], 0.02),
            "separated witness carried weight or lost its source index");
    require(near(accepted.generalizedSupportForce[2], fixture.weight()),
            "support generalized force disagrees with analytic weight");

    NumiHumanMuscleEquilibriumResult replay;
    require(fixture.compile({touching, separated}, replay).succeeded() &&
                replay.q == accepted.q && replay.activation == accepted.activation &&
                replay.supportNormalForce == accepted.supportNormalForce &&
                replay.supportPlaneGapMeters == accepted.supportPlaneGapMeters,
            "static support compile did not replay exactly");

    NumiHumanMuscleEquilibriumResult airborne;
    fixture.config.supportForceRegularization = 0.0;
    require(fixture.compile({separated}, airborne).succeeded() &&
                !airborne.diagnostics.balanced &&
                airborne.supportNormalForce[0] == 0.0 &&
                near(airborne.diagnostics.maximumFloatingRootForceResidual,
                     fixture.weight()),
            "airborne mass received a fictitious ground reaction");

    auto penetrating = touching;
    penetrating.localPoint[2] = -0.014;
    const auto rejected = fixture.compile({penetrating}, accepted);
    require(rejected.status == NumiHumanMuscleEquilibriumStatus::supportPenetration &&
                rejected.failingIndex == 0u && accepted.q == replay.q &&
                accepted.activation == replay.activation &&
                accepted.supportNormalForce == replay.supportNormalForce &&
                accepted.supportPlaneGapMeters == replay.supportPlaneGapMeters,
            "penetrating initial pose was admitted or mutated accepted output");

    // A shared world translation changes neither contact nor supported load.
    fixture.q[2] = 1.25;
    auto elevatedPlane = touching;
    elevatedPlane.planePoint[2] = 1.25;
    NumiHumanMuscleEquilibriumResult translated;
    require(fixture.compile({elevatedPlane}, translated).succeeded() &&
                translated.diagnostics.balanced &&
                near(translated.supportNormalForce[0], fixture.weight()) &&
                near(translated.supportPlaneGapMeters[0], 0.0),
            "authored nonzero plane was ignored");
    fixture.q[2] = 0.0;

    // The gap and force follow the authored normal, not a hard-coded z axis.
    fixture.model.world.gravityAndTimestep = {0.0f, -9.81f, 0.0f, 0.0001f};
    auto sidePlane = touching;
    sidePlane.normal = {0.0, 1.0, 0.0};
    NumiHumanMuscleEquilibriumResult rotated;
    require(fixture.compile({sidePlane}, rotated).succeeded() &&
                rotated.diagnostics.balanced &&
                near(rotated.generalizedSupportForce[1],
                     -2.5 * fixture.model.world.gravityAndTimestep.y),
            "rotated plane did not support the analytic load");

    // Numerical geometry tolerance is an explicit admission bound.
    auto withinTolerance = sidePlane;
    withinTolerance.localPoint[1] = -0.5 * fixture.config.supportGapToleranceMeters;
    require(fixture.compile({withinTolerance}, rotated).succeeded(),
            "bounded geometric roundoff was rejected");
    fixture.config.supportGapToleranceMeters = 0.0;
    require(fixture.compile({withinTolerance}, rotated).status ==
                NumiHumanMuscleEquilibriumStatus::supportPenetration,
            "zero tolerance was not enforced");
    fixture.config.supportGapToleranceMeters = -1.0;
    require(fixture.compile({sidePlane}, rotated).status ==
                NumiHumanMuscleEquilibriumStatus::invalidConfiguration,
            "negative support tolerance was accepted");
    fixture.config.supportGapToleranceMeters = std::numeric_limits<double>::quiet_NaN();
    require(fixture.compile({sidePlane}, rotated).status ==
                NumiHumanMuscleEquilibriumStatus::invalidConfiguration,
            "nonfinite support tolerance was accepted");
    fixture.config.supportGapToleranceMeters = 1.0e-6;
    sidePlane.planePoint[0] = std::numeric_limits<double>::quiet_NaN();
    require(fixture.compile({sidePlane}, rotated).status ==
                NumiHumanMuscleEquilibriumStatus::invalidDimensions,
            "nonfinite authored plane was accepted");
    // Offline placement changes only explicitly bounded coordinates, with
    // every unselected witness still checked against its authored plane.
    Fixture placement;
    placement.q[2] = 0.015;
    const std::vector<NumiHumanStaticSupportContact> placementContacts{touching, separated};
    const std::vector<std::uint32_t> activeContacts{0u};
    const std::vector<NumiHumanSupportPoseCoordinate> vertical{{2u, 0.02}};
    NumiHumanSupportPoseResult fitted;
    require(compileNumiHumanSupportPose(placement.model, 0u, placement.q, {},
                placementContacts, activeContacts, vertical, fitted).succeeded() &&
                near(fitted.q[2], 0.0, 1.0e-8) &&
                near(fitted.supportPlaneGapMeters[1], 0.02, 1.0e-8),
            "explicit offline placement did not reach the authored plane");
    for (std::size_t i = 0; i < fitted.q.size(); ++i) {
        require(i == 2u || fitted.q[i] == placement.q[i],
                "placement altered an unauthorized coordinate");
    }
    NumiHumanSupportPoseResult repeatedFit;
    require(compileNumiHumanSupportPose(placement.model, 0u, placement.q, {},
                placementContacts, activeContacts, vertical, repeatedFit).succeeded() &&
                repeatedFit.q == fitted.q &&
                repeatedFit.supportPlaneGapMeters == fitted.supportPlaneGapMeters &&
                repeatedFit.iterations == fitted.iterations,
            "offline support placement did not replay exactly");
    const std::vector<NumiHumanSupportPoseCoordinate> tooShort{{2u, 0.001}};
    require(compileNumiHumanSupportPose(placement.model, 0u, placement.q, {},
                placementContacts, activeContacts, tooShort, fitted).status ==
                NumiHumanMuscleEquilibriumStatus::supportPoseInfeasible &&
                fitted.q == repeatedFit.q &&
                fitted.supportPlaneGapMeters == repeatedFit.supportPlaneGapMeters,
            "infeasible placement exceeded its bound or changed accepted output");
    auto obstacle = touching;
    obstacle.localPoint[2] = -0.001;
    const std::vector<NumiHumanStaticSupportContact> obstructed{touching, obstacle};
    require(compileNumiHumanSupportPose(placement.model, 0u, placement.q, {},
                obstructed, activeContacts, vertical, fitted).status ==
                NumiHumanMuscleEquilibriumStatus::supportPoseInfeasible &&
                fitted.q == repeatedFit.q,
            "placement ignored an unselected penetrating witness");
    const std::vector<std::uint32_t> duplicates{0u, 0u};
    require(compileNumiHumanSupportPose(placement.model, 0u, placement.q, {},
                placementContacts, duplicates, vertical, fitted).status ==
                NumiHumanMuscleEquilibriumStatus::invalidSelection,
            "duplicate active contacts were admitted");
    const std::vector<NumiHumanSupportPoseCoordinate> rotation{{3u, 0.1}};
    require(compileNumiHumanSupportPose(placement.model, 0u, placement.q, {},
                placementContacts, activeContacts, rotation, fitted).status ==
                NumiHumanMuscleEquilibriumStatus::invalidSelection,
            "offline scalar placement admitted a root rotation");
    NumiHumanSupportPoseConfig badFit;
    badFit.gapToleranceMeters = std::numeric_limits<double>::quiet_NaN();
    require(compileNumiHumanSupportPose(placement.model, 0u, placement.q, {},
                placementContacts, activeContacts, vertical, fitted, badFit).status ==
                NumiHumanMuscleEquilibriumStatus::invalidConfiguration,
            "offline placement admitted a nonfinite tolerance");
    auto rotatedPlane = touching;
    rotatedPlane.normal = {0.0, 1.0, 0.0};
    rotatedPlane.planePoint = {0.0, 1.0, 0.0};
    placement.q[1] = 1.01;
    const std::vector<NumiHumanStaticSupportContact> rotatedContacts{rotatedPlane};
    const std::vector<NumiHumanSupportPoseCoordinate> horizontal{{1u, 0.02}};
    require(compileNumiHumanSupportPose(placement.model, 0u, placement.q, {},
                rotatedContacts, activeContacts, horizontal, fitted).succeeded() &&
                near(fitted.q[1], 1.0, 1.0e-8) && fitted.q[2] == placement.q[2],
            "placement ignored a translated or rotated authored plane");
    // An analytic two-slider chain with q1 = 2*q0 must use the full
    // equality tangent: world witness displacement is 3*q0, not q0.
    EngineModel sliders;
    MRArticulationGPU sliderArt{};
    sliderArt.rootBody = 0u;
    sliderArt.rootType = MR_ROOT_FIXED;
    sliderArt.firstBody = 0u;
    sliderArt.bodyCount = 3u;
    sliderArt.firstJoint = 0u;
    sliderArt.jointCount = 2u;
    sliderArt.nq = 2u;
    sliderArt.nv = 2u;
    sliders.articulations.push_back(sliderArt);
    sliders.bodies.push_back(body(MR_INVALID_INDEX, MR_INVALID_INDEX, 1.0, {1.0, 1.0, 1.0}));
    for (std::uint32_t i = 0; i < 2; ++i) {
        sliders.bodies.push_back(body(i, i, 1.0, {1.0, 1.0, 1.0}));
        MRJointDescriptorGPU joint{};
        joint.parentBody = i;
        joint.childBody = i + 1;
        joint.jointType = MR_JOINT_PRISMATIC;
        joint.qOffset = i;
        joint.vOffset = i;
        joint.nq = 1u;
        joint.nv = 1u;
        joint.axis0 = f4(0.0, 1.0, 0.0);
        joint.parentRotation = f4(0.0, 0.0, 0.0, 1.0);
        joint.childRotation = f4(0.0, 0.0, 0.0, 1.0);
        sliders.joints.push_back(joint);
        MRDofPropertiesGPU dof{};
        dof.articulationIndex = 0;
        dof.jointIndex = i;
        dof.qIndex = i;
        dof.vIndex = i;
        dof.flags = MR_DOF_FLAG_POSITION_LIMIT;
        dof.limits = f4(-0.1, 0.1, 0.0);
        sliders.dofs.push_back(dof);
    }
    MRNumiHumanJointEqualityGPU equality{};
    equality.indices = {1u, 1u, 0u, 0u};
    equality.referencesAndCoefficients0 = f4(0.0, 0.0, 0.0, 2.0);
    const std::vector<MRNumiHumanJointEqualityGPU> sliderEqualities{equality};
    const std::vector<double> sliderQ{0.01, 0.0};
    const std::vector<NumiHumanStaticSupportContact> sliderContacts{
        {.bodyIndex = 2u, .normal = {0.0, 1.0, 0.0}}};
    const std::vector<NumiHumanSupportPoseCoordinate> sliderCoordinates{{0u, 0.08}};
    NumiHumanSupportPoseConfig singleIteration;
    singleIteration.maximumIterations = 1u;
    require(compileNumiHumanSupportPose(sliders, 0u, sliderQ, sliderEqualities,
                sliderContacts, activeContacts, sliderCoordinates, fitted,
                singleIteration).succeeded() && near(fitted.q[0], 0.0, 1.0e-8) &&
                near(fitted.q[1], 0.0, 1.0e-8),
            "placement omitted the exact source equality tangent");
    auto limitedContacts = sliderContacts;
    limitedContacts[0].planePoint[1] = 0.24;
    require(compileNumiHumanSupportPose(sliders, 0u, sliderQ, sliderEqualities,
                limitedContacts, activeContacts, sliderCoordinates, fitted).status ==
                NumiHumanMuscleEquilibriumStatus::supportPoseInfeasible,
            "placement violated an equality-dependent source joint limit");
    const std::vector<NumiHumanSupportPoseCoordinate> dependentSelection{{1u, 0.08}};
    require(compileNumiHumanSupportPose(sliders, 0u, sliderQ, sliderEqualities,
                sliderContacts, activeContacts, dependentSelection, fitted).status ==
                NumiHumanMuscleEquilibriumStatus::invalidSelection,
            "placement admitted direct actuation of an equality dependent");
    // The two serial, unit-mass sliders have M=[[2,1],[1,1]]. These
    // force/acceleration pairs distinguish complementarity from force clipping.
    Fixture reactions;
    reactions.model = sliders;
    reactions.q = {0.0, 0.0};
    reactions.model.world.gravityAndTimestep = f4(0, 0, 0, 0.0001);
    reactions.model.dofs[0].limits = f4(0, 1, 0);
    reactions.model.dofs[1].limits = f4(-10, 10, 0);
    std::vector<double> mass(4u);
    require(computeArticulatedMassMatrix(reactions.model, 0u, reactions.q, mass).succeeded() &&
            near(mass[0], 2) && near(mass[1], 1) && near(mass[2], 1) && near(mass[3], 1),
            "analytic coupled-slider mass changed");
    const auto compileReaction = [&](double f0, double f1,
                                    const std::vector<MRNumiHumanJointEqualityGPU>& eqs = {}) {
        const std::vector<NumiHumanPassiveCoordinateCoupling> loads{
            {0u, 0u, reactions.q[0] + f0, 1.0},
            {1u, 1u, reactions.q[1] + f1, 1.0}};
        NumiHumanMuscleEquilibriumResult result;
        const auto diagnostics = compileNumiHumanMuscleEquilibrium(
            reactions.model, 0u, reactions.q, reactions.sites, {}, reactions.muscles,
            reactions.architectures, eqs, {}, {}, loads, result, reactions.config);
        require(diagnostics.succeeded() && diagnostics.positionLimitKktResidual < 1.0e-8,
                "coupled reaction compile or physical KKT certificate failed");
        require(result.q == reactions.q, "reaction solve clamped coordinates");
        for (std::size_t row = 0u; row < 2u; ++row) {
            const double inertial = mass[2u * row] * result.generalizedAccelerationResidual[0] +
                mass[2u * row + 1u] * result.generalizedAccelerationResidual[1];
            require(near(inertial, result.generalizedForceResidual[row]),
                    "reported equality/stop reaction does not satisfy full M*a=f");
        }
        return result;
    };
    const auto intoStop = compileReaction(1, 3);
    require(near(intoStop.generalizedPositionLimitForce[0], 2) &&
            near(intoStop.generalizedAccelerationResidual[0], 0) &&
            near(intoStop.generalizedAccelerationResidual[1], 3),
            "positive coordinate force hid a mass-coupled stop violation");
    const auto awayFromStop = compileReaction(-1, -3);
    require(near(awayFromStop.generalizedPositionLimitForce[0], 0) &&
            near(awayFromStop.generalizedAccelerationResidual[0], 2) &&
            near(awayFromStop.generalizedAccelerationResidual[1], -5),
            "outward acceleration received a spurious joint-stop force");
    reactions.model.dofs[0].limits = f4(-1, 0, 0);
    const auto upperStop = compileReaction(-1, -3);
    require(near(upperStop.generalizedPositionLimitForce[0], -2) &&
            near(upperStop.generalizedAccelerationResidual[0], 0) &&
            near(upperStop.generalizedAccelerationResidual[1], -3),
            "upper-stop reaction used the wrong sign");
    reactions.model.dofs[0].limits = f4(0, 0, 0);
    const auto collapsedRange = compileReaction(1, 3);
    require(near(collapsedRange.generalizedPositionLimitForce[0], 2) &&
            near(collapsedRange.generalizedAccelerationResidual[0], 0),
            "coincident lower and upper stops failed their unilateral solve");
    reactions.model.dofs[0].limits = f4(0, 1, 0);
    reactions.model.dofs[1].limits = f4(0, 1, 0);
    const auto bothStops = compileReaction(-3, -1);
    require(near(bothStops.generalizedPositionLimitForce[0], 3) &&
            near(bothStops.generalizedPositionLimitForce[1], 1) && bothStops.diagnostics.balanced,
            "reaction at one stop failed to activate the coupled second stop");
    reactions.model.dofs[0].limits = f4(-1, 1, 0);
    auto reversedEquality = equality;
    reversedEquality.referencesAndCoefficients0.w = -2.0f;
    const std::vector<MRNumiHumanJointEqualityGPU> reversedEqualities{reversedEquality};
    const auto dependentStop = compileReaction(2, 0, reversedEqualities);
    require(near(dependentStop.generalizedPositionLimitForce[1], 1) &&
            dependentStop.diagnostics.balanced &&
            near(dependentStop.generalizedJointEqualityForce[0], -2) &&
            near(dependentStop.generalizedJointEqualityForce[1], -1),
            "equality-dependent stop omitted its signed source tangent");
    reactions.model.dofs[1].limits = f4(-1, 1, 0);
    const auto freeEquality = compileReaction(2, 0, reversedEqualities);
    require(near(freeEquality.generalizedAccelerationResidual[0], 1) &&
            near(freeEquality.generalizedAccelerationResidual[1], -2) &&
            near(freeEquality.generalizedJointEqualityForce[0], -2) &&
            near(freeEquality.generalizedJointEqualityForce[1], -1),
            "equality acceleration was zeroed instead of lifted through the tangent");
    reactions.model.dofs[0].limits = f4(0, 1, 0);
    reactions.model.dofs[1].limits = f4(0, 1, 0);
    const auto redundantStops = compileReaction(-2, -1, sliderEqualities);
    require(redundantStops.diagnostics.balanced &&
            near(redundantStops.generalizedPositionLimitForce[0] +
                 2 * redundantStops.generalizedPositionLimitForce[1], 4),
            "redundant equality-linked source stops failed complementarity");
    const auto repeatedStops = compileReaction(-2, -1, sliderEqualities);
    require(redundantStops.generalizedPositionLimitForce == repeatedStops.generalizedPositionLimitForce &&
            redundantStops.generalizedAccelerationResidual == repeatedStops.generalizedAccelerationResidual,
            "coupled source-stop reaction replay changed");
    auto preservedReaction = redundantStops;
    reactions.q[0] = -0.01;
    const auto invalidInitial = compileNumiHumanMuscleEquilibrium(
        reactions.model, 0u, reactions.q, reactions.sites, {}, reactions.muscles,
        reactions.architectures, {}, {}, preservedReaction, reactions.config);
    require(invalidInitial.status == NumiHumanMuscleEquilibriumStatus::positionLimitViolation &&
            invalidInitial.failingIndex == 0u && preservedReaction.q == redundantStops.q &&
            preservedReaction.generalizedPositionLimitForce == redundantStops.generalizedPositionLimitForce,
            "penetrated initial joint received a static certificate or changed the destination");
    reactions.q[0] = 0.0;
    auto invalidEquality = equality;
    invalidEquality.indices.z = MR_INVALID_INDEX;
    invalidEquality.indices.w = MR_INVALID_INDEX;
    invalidEquality.referencesAndCoefficients0 = f4(0, 0, 2, 0);
    const std::vector<MRNumiHumanJointEqualityGPU> invalidEqualities{invalidEquality};
    const auto invalidDependent = compileNumiHumanMuscleEquilibrium(
        reactions.model, 0u, reactions.q, reactions.sites, {}, reactions.muscles,
        reactions.architectures, invalidEqualities, {}, preservedReaction, reactions.config);
    require(invalidDependent.status == NumiHumanMuscleEquilibriumStatus::positionLimitViolation &&
            invalidDependent.failingIndex == 1u && preservedReaction.q == redundantStops.q,
            "fixed equality projected a dependent outside its source limits");
    // Rotate the second slider: M=[[2,.8],[.8,1]] and the spanning
    // muscle changes f by [F,.8F]. The active-stop derivative [0,.8]
    // differs from the free derivative. Balance requires F=-3.75, lambda=2.75;
    // the activation regularizer permits a small, bounded residual.
    reactions.model.joints[1].axis0 = f4(0.6, 0.8, 0.0);
    reactions.muscles[0].gainParameters[2] = 10.0;
    reactions.muscles[0].biasParameters[2] = 10.0;
    reactions.model.dofs[1].limits = f4(-10, 10, 0);
    const std::vector<MujocoMuscleSite> recruitingSites{
        {0u, {0.0, 0.0, 0.0}}, {2u, {0.0, 1.0, 0.0}}};
    const std::vector<NumiHumanPassiveCoordinateCoupling> recruitingLoads{
        {0u, 0u, 1.0, 1.0}, {1u, 1u, 3.0, 1.0}};
    NumiHumanMuscleEquilibriumResult recruitedReaction;
    const auto recruitedDiagnostics = compileNumiHumanMuscleEquilibrium(
            reactions.model, 0u, reactions.q, recruitingSites, {}, reactions.muscles,
            reactions.architectures, {}, {}, {}, recruitingLoads,
            recruitedReaction, reactions.config);
    require(recruitedDiagnostics.succeeded() &&
            recruitedReaction.diagnostics.balanced &&
            recruitedReaction.activation[0] > 0.0 &&
            near(recruitedReaction.generalizedPositionLimitForce[0], 2.75, 5.0e-4),
            "recruitment did not balance a coupled joint-stop load");
    // Two coupled serial sliders: a proximal muscle supplies [-1,0]F and
    // a spanning muscle [-1,-1]F. The exact tension solution for loads [5,2]
    // is [3,2]. A single ordered sweep cannot resolve this force sharing.
    auto coupledModel = reactions.model;
    coupledModel.joints[1].axis0 = f4(0.0, 1.0, 0.0);
    coupledModel.dofs[0].limits = coupledModel.dofs[1].limits = f4(-10, 10, 0);
    const std::vector<MujocoMuscleSite> coupledSites{
        {0u, {0,0,0}}, {1u, {0,1,0}}, {2u, {0,1,0}}};
    std::vector<MujocoMuscleDefinition> coupledMuscles(2u, reactions.muscles[0]);
    coupledMuscles[0].route = {{MujocoRouteNodeType::site, 0u}, {MujocoRouteNodeType::site, 1u}};
    coupledMuscles[1].route = {{MujocoRouteNodeType::site, 0u}, {MujocoRouteNodeType::site, 2u}};
    const std::vector<MujocoCompliantMuscleArchitecture> coupledArchitectures(2u);
    // Recruitment through the compliant preparation owner: gravity supplies
    // [2,1] on two unit-mass sliders. Both muscle tensions must be exactly 1.
    auto compliantModel=coupledModel;compliantModel.world.gravityAndTimestep=f4(0,1,0,0.0001);
    NumiHumanCompliantEquilibriumConfig compliantConfig;
    compliantConfig.optimizeActivation=true;compliantConfig.accelerationTolerance=1e-6;
    const std::vector<double> coldActivation(2,0);
    NumiHumanCompliantEquilibriumResult compliantResult,compliantReplay;
    const auto prepareCompliant=[&](auto& output) {
        return compileNumiHumanCompliantEquilibrium(compliantModel,0,reactions.q,coldActivation,
            coupledSites,{},coupledMuscles,coupledArchitectures,{},{},{},output,compliantConfig);
    };
    require(prepareCompliant(compliantResult).succeeded() && compliantResult.state.diagnostics.balanced &&
        near(compliantResult.state.muscleTendonForce[0],-1,1e-6) && near(compliantResult.state.muscleTendonForce[1],-1,1e-6),
        "source-compliant recruitment missed analytic gravity balance");
    require(prepareCompliant(compliantReplay).succeeded() && compliantReplay.state.activation==compliantResult.state.activation &&
        compliantReplay.state.q==compliantResult.state.q,"compliant recruitment replay changed");
    compliantModel.world.gravityAndTimestep.y=1e6f;
    require(prepareCompliant(compliantReplay).succeeded() && !compliantReplay.state.diagnostics.balanced &&
        std::all_of(compliantReplay.state.activation.begin(),compliantReplay.state.activation.end(),[](double a){return a>=0 && a<=1;}),
        "infeasible compliant recruitment waived actuator bounds");
    const std::vector<NumiHumanPassiveCoordinateCoupling> coupledLoads{
        {0u, 0u, 5.0, 1.0}, {1u, 1u, 2.0, 1.0}};
    auto coupledConfig = reactions.config;
    coupledConfig.activationSweeps = 1u;
    coupledConfig.activationRegularization = 0.0;
    coupledConfig.globalActivationPolishIterations = 12u;
    const auto recruitCoupled = [&](const std::vector<MujocoMuscleDefinition>& definitions) {
        NumiHumanMuscleEquilibriumResult result;
        require(compileNumiHumanMuscleEquilibrium(coupledModel, 0u, reactions.q,
            coupledSites, {}, definitions, coupledArchitectures, {}, {}, {},
            coupledLoads, result, coupledConfig).succeeded(),
            "coupled recruitment proposal failed");
        return result;
    };
    const auto coupledResult = recruitCoupled(coupledMuscles);
    require(coupledResult.diagnostics.normalizedResidualRms < 1.0e-6 &&
            near(coupledResult.muscleTendonForce[0], -3.0, 1.0e-5) &&
            near(coupledResult.muscleTendonForce[1], -2.0, 1.0e-5),
            "coupled polish did not recover analytic muscle force sharing");
    std::swap(coupledMuscles[0], coupledMuscles[1]);
    const auto reversedResult = recruitCoupled(coupledMuscles);
    require(reversedResult.diagnostics.normalizedResidualRms < 1.0e-6 &&
            near(reversedResult.activation[1], coupledResult.activation[0], 1.0e-6) &&
            near(reversedResult.activation[0], coupledResult.activation[1], 1.0e-6),
            "coupled recruitment retained muscle ordering bias");
    const auto coupledReplay = recruitCoupled(coupledMuscles);
    require(coupledReplay.activation == reversedResult.activation &&
            coupledReplay.generalizedAccelerationResidual == reversedResult.generalizedAccelerationResidual,
            "coupled recruitment changed exact replay");
    coupledConfig.activationLimit = 0.05;
    const auto limitedRecruitment = recruitCoupled(coupledMuscles);
    require(!limitedRecruitment.diagnostics.balanced &&
            limitedRecruitment.activation[0] <= coupledConfig.activationLimit &&
            limitedRecruitment.activation[1] <= coupledConfig.activationLimit &&
            limitedRecruitment.q == reactions.q,
            "infeasible recruitment waived activation bounds or changed the posture");
    // With no generalized muscle moment arm, two independent spring loads
    // require both coordinates to move in the same accepted posture update.
    // One scalar trial alone cannot reach the analytic equilibrium [5,2].
    for (auto& definition : coupledMuscles) {
        definition.route = {{MujocoRouteNodeType::site, 0u}, {MujocoRouteNodeType::site, 1u}};
    }
    const std::vector<MujocoMuscleSite> postureSites{{0u,{0,0,0}}, {0u,{0,1,0}}};
    coupledConfig.poseSweeps = 1u;
    coupledConfig.poseStepFraction = 0.3;
    coupledConfig.maximumPoseStep = 6.0;
    coupledConfig.poseRegularization = 0.0;
    NumiHumanMuscleEquilibriumResult postureResult;
    require(compileNumiHumanMuscleEquilibrium(coupledModel, 0u, reactions.q,
        postureSites, {}, coupledMuscles, coupledArchitectures, {}, {}, {},
        coupledLoads, postureResult, coupledConfig).succeeded() &&
        postureResult.diagnostics.normalizedResidualRms < 1.0e-6 &&
        near(postureResult.q[0], 5.0, 1.0e-5) && near(postureResult.q[1], 2.0, 1.0e-5) &&
        postureResult.diagnostics.acceptedPoseSteps == 1u,
        "coupled posture proposal missed the analytic two-coordinate equilibrium");
    require(postureResult.searchTrace.size() == 3u &&
            postureResult.searchTrace[0].kind == 0u && postureResult.searchTrace[1].kind == 1u &&
            postureResult.searchTrace[1].coupledPoseProposal && postureResult.searchTrace[2].kind == 2u &&
            postureResult.searchTrace[1].objective < postureResult.searchTrace[0].objective &&
            near(postureResult.searchTrace.back().normalizedResidualRms, postureResult.diagnostics.normalizedResidualRms),
            "accepted search history lost a coupled update or disagreed with its certificate");
    std::cout << "numi_human_static_support_test=passed"
              << " source_compliant_preparation=passed compliant_recruitment=passed"
              << " coupled_recruitment=passed recruitment_bounds=passed"
              << " coupled_posture=passed"
              << " coupled_limit_reactions=passed dependent_acceleration=passed"
              << " analytic_weight_n=" << replay.supportNormalForce[0]
              << " replay=exact penetration=rejected airborne_force_n=0\n";
}
