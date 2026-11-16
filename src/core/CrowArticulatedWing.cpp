#include "metalrobo/CrowArticulatedWing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace metalrobo {
namespace {

// Related-species engineering priors: Raven bone lengths scaled by body mass.
// These are NOT measured American-crow values; all joint limits, masses,
// cross-sections, and drive gains here are exploratory and uncalibrated.
constexpr std::array<float, 3> lengths{{0.08355912f, 0.09530571f, 0.09089033f}};
constexpr std::array<float, 3> massFractions{{0.40f, 0.35f, 0.25f}};
constexpr std::array<float, 3> chordHalf{{0.035f, 0.035f, 0.030f}};
constexpr std::array<float, 3> thicknessHalf{{0.006f, 0.005f, 0.004f}};
constexpr float wristProxyMass = 0.00001f;
// Exploratory bone proxies, not measurements. The capsule centerline ends
// short of the wrist to keep adjacent joint volumes from intersecting.
constexpr float boneProxyRadius = 0.003f;
constexpr float wristCenterlineInset = 0.010f;

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

void setBoxInertia(MRBodyPropertiesGPU& b, float mass, float hx, float hy, float hz) {
    require(mass > 0 && hx > 0 && hy > 0 && hz > 0, "invalid wing segment inertia");
    const float ix = mass*(hy*hy+hz*hz)/3.0f;
    const float iy = mass*(hx*hx+hz*hz)/3.0f;
    const float iz = mass*(hx*hx+hy*hy)/3.0f;
    b.massAndInverseMass = {mass, 1.0f/mass, 0, 0};
    b.inertiaRow0 = {ix,0,0,0}; b.inertiaRow1 = {0,iy,0,0}; b.inertiaRow2 = {0,0,iz,0};
    b.inverseInertiaRow0 = {1.0f/ix,0,0,0};
    b.inverseInertiaRow1 = {0,1.0f/iy,0,0};
    b.inverseInertiaRow2 = {0,0,1.0f/iz,0};
}

void updateCounts(EngineModel& m) {
    auto& a = m.articulations.at(0);
    a.bodyCount = m.world.bodyCount = static_cast<std::uint32_t>(m.bodies.size());
    a.jointCount = m.world.jointCount = static_cast<std::uint32_t>(m.joints.size());
    a.nq = m.world.nq = static_cast<std::uint32_t>(m.defaultQ.size());
    a.nv = m.world.nv = static_cast<std::uint32_t>(m.defaultV.size());
    m.world.shapeCount = static_cast<std::uint32_t>(m.shapes.size());
}

void setShape(MRShapeGPU& s, std::uint32_t body, float x, float y, float z,
              float localY, std::uint32_t generation) {
    s.bodyIndex = body;
    s.localPosition = {0,localY,0,1}; s.localRotation = {0,0,0,1};
    s.dimensions = {x,y,z,0};
    s.slotGeneration = generation;
    s.contactRestAndBoundingRadius.z = std::sqrt(x*x+y*y+z*z);
}

} // namespace

RobotPack makeCrowArticulatedWingMechanicsFixture(RobotPack p) {
    require(p.id == "crow_v15_experimental_hip_roll", "expected exact v15 hip-roll research body");
    auto& m = p.mechanics;
    require(m.articulations.size() == 1 && m.bodies.size() == 35 && m.joints.size() >= 6,
            "unexpected v15 topology");
    require(m.bodyNames.at(5) == "crow_left_wing" && m.bodyNames.at(6) == "crow_right_wing",
            "unexpected v15 wing body indices");
    require(m.jointNames.at(2) == "crow_left_wing_flap" &&
            m.jointNames.at(3) == "crow_right_wing_flap", "unexpected v15 shoulder frames");
    require(p.flappingWings.has_value(), "expected v15 aerodynamic identity");
    // Existing flapping program assumes one rigid body and cannot represent
    // these wings. Explicitly remove it; this fixture is not flight-qualified.
    p.flappingWings.reset();
    p.id = "american_crow_v15_articulated_wing_mechanics_fixture_v3";
    p.revision += 1;
    p.sourceRevision += "+experimental-articulated-wing-mechanics-v3-no-aero";
    p.capabilities.push_back("experimental_uncalibrated_articulated_wing_mechanics");
    p.capabilities.push_back("flight_aerodynamics_unavailable");
    m.name = p.id;

    for (unsigned side=0; side<2; ++side) {
        const float sign = side == 0 ? 1.0f : -1.0f;
        const std::string label = side == 0 ? "left" : "right";
        const std::uint32_t upper = 5u+side;
        auto wingShape = std::find(m.shapeNames.begin(),m.shapeNames.end(),
                                   "crow_"+label+"_wing/collision");
        require(wingShape != m.shapeNames.end(), "missing original wing collider");
        const auto shapeIndex = static_cast<std::uint32_t>(wingShape-m.shapeNames.begin());
        const MRShapeGPU originalShape = m.shapes.at(shapeIndex);
        require(originalShape.bodyIndex == upper, "wrong original wing collider body");
        const float originalMass = m.bodies.at(upper).massAndInverseMass.x;
        require(std::isfinite(originalMass) && originalMass*massFractions[2] > wristProxyMass,
                "invalid v15 wing mass");
        auto& shoulderJoint = m.joints.at(4u+side);
        require(shoulderJoint.childBody == upper && shoulderJoint.childAnchor.x == 0 &&
                shoulderJoint.childAnchor.y == 0 && shoulderJoint.childAnchor.z == 0,
                "unexpected v15 upper-wing COM frame");
        // State origins and anchors are COM-centred. Move the upper COM from
        // the old pronation joint to the middle of the proposed segment.
        shoulderJoint.childAnchor.y = -sign*lengths[0]/2;
        auto& upperBody = m.bodies.at(upper);
        setBoxInertia(upperBody, originalMass*massFractions[0], chordHalf[0],lengths[0]/2,thicknessHalf[0]);
        upperBody.centerOfMass = {0,0,0,0};
        setShape(m.shapes.at(shapeIndex),upper,chordHalf[0],lengths[0]/2,
                 thicknessHalf[0],0,originalShape.slotGeneration);

        std::uint32_t parent=upper;
        // Three revolutes: elbow flexion, wrist flexion, and wrist deviation.
        // A small nonzero inertial link carries the second wrist axis.
        for (unsigned k=0;k<3;++k) {
            const bool proxy = k==1;
            const std::uint32_t bodyIndex = static_cast<std::uint32_t>(m.bodies.size());
            const std::uint32_t jointIndex = static_cast<std::uint32_t>(m.joints.size());
            const std::uint32_t qi = static_cast<std::uint32_t>(m.defaultQ.size());
            const std::uint32_t vi = static_cast<std::uint32_t>(m.defaultV.size());
            const std::string part = k==0 ? "forearm" : k==1 ? "wrist_axis" : "hand";
            const std::string jointName = "crow_"+label+"_"+(k==0 ? "elbow_flex" : k==1 ? "wrist_flex" : "wrist_deviation");
            MRBodyPropertiesGPU b{};
            b.articulationIndex=0; b.parentBody=parent; b.inboundJoint=jointIndex;
            b.motionType=MR_MOTION_DYNAMIC; b.dampingAndSpeedLimits={0.01f,0.01f,30,30};
            if (k==0) setBoxInertia(b,originalMass*massFractions[1],chordHalf[1],lengths[1]/2,thicknessHalf[1]);
            else if (proxy) setBoxInertia(b,wristProxyMass,0.003f,0.003f,0.003f);
            else setBoxInertia(b,originalMass*massFractions[2]-wristProxyMass,chordHalf[2],lengths[2]/2,thicknessHalf[2]);
            m.bodies.push_back(b); m.bodyNames.push_back("crow_"+label+"_"+part);

            MRJointDescriptorGPU j{};j.parentBody=parent;j.childBody=bodyIndex;
            j.jointType=MR_JOINT_REVOLUTE;j.qOffset=qi;j.nq=1;j.vOffset=vi;j.nv=1;
            j.axis0=k<2 ? mr_float4{0,0,sign,0} : mr_float4{sign,0,0,0};
            j.parentAnchor = k==0 ? mr_float4{0,sign*lengths[0]/2,0,0}
                : k==1 ? mr_float4{0,sign*lengths[1]/2,0,0} : mr_float4{0,0,0,0};
            j.childAnchor = k==0 ? mr_float4{0,-sign*lengths[1]/2,0,0}
                : k==1 ? mr_float4{0,0,0,0} : mr_float4{0,-sign*lengths[2]/2,0,0};
            j.parentRotation=j.childRotation={0,0,0,1};
            m.joints.push_back(j);m.jointNames.push_back(jointName);

            MRDofPropertiesGPU d{};d.articulationIndex=0;d.jointIndex=jointIndex;
            d.qIndex=qi;d.vIndex=vi;
            d.flags=MR_DOF_FLAG_ACTUATED|MR_DOF_FLAG_POSITION_LIMIT|
                    MR_DOF_FLAG_VELOCITY_LIMIT|MR_DOF_FLAG_EFFORT_LIMIT|MR_DOF_FLAG_DRIVE;
            d.limits = k==0 ? mr_float4{0,2.4f,12,0.08f}
                     : k==1 ? mr_float4{0,2.1f,12,0.04f} : mr_float4{-0.5f,0.5f,12,0.02f};
            d.drive={0.12f,0.004f,0.00005f,0};
            m.dofs.push_back(d);m.dofNames.push_back(jointName);
            m.defaultQ.push_back(0);m.defaultV.push_back(0);
            if (!m.actuatorProfiles.empty()) m.actuatorProfiles.emplace_back();
            p.actuators.push_back({.id="wing."+label+"."+(k==0?"elbow":k==1?"wrist_flex":"wrist_deviation"),
                .kind=RobotActuatorKind::jointPosition,.target=jointName,.scale=1.0f,
                .responseTimeSeconds=0.020f,.parameters={0,0,1,0}});
            p.roles.push_back({"wing_"+label+"_"+part,RobotSemanticKind::body,{"crow_"+label+"_"+part}});
            if (!proxy) {
                MRShapeGPU shape=originalShape;
                const unsigned segment=k==0?1u:2u;
                const float localY = (k==0 ? -sign : sign)*(wristCenterlineInset-boneProxyRadius)/2;
                const float halfLine = (lengths[segment]-wristCenterlineInset-boneProxyRadius)/2;
                require(halfLine > 0, "invalid bone proxy length");
                shape.bodyIndex=bodyIndex;shape.shapeType=MR_SHAPE_CAPSULE;
                shape.localPosition={0,localY,0,1};shape.localRotation={0,0,0,1};
                shape.dimensions={boneProxyRadius,halfLine,0,0};
                shape.slotGeneration=static_cast<std::uint32_t>(m.shapes.size()+1);
                shape.contactRestAndBoundingRadius.z=boneProxyRadius+halfLine;
                m.shapes.push_back(shape);m.shapeNames.push_back("crow_"+label+"_"+part+"/collision");
            }
            parent=bodyIndex;
        }
    }
    updateCounts(m);
    for(auto& role:p.roles) {
        if(role.id=="whole_body")role.members=m.bodyNames;
        if(role.id=="all_joints")role.members=m.jointNames;
        if(role.id=="all_dofs")role.members=m.dofNames;
    }
    std::string reason;
    if(!m.valid(&reason)) throw std::logic_error("invalid articulated-wing fixture: "+reason);
    return p;
}
} // namespace metalrobo
