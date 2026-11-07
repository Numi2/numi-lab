#include "metalrobo/RunProgram.hpp"
#include "metalrobo/FiniteResource.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace metalrobo {
namespace {
struct Segment {
    const char* body;
    const char* joint;
    float mass;
    mr_float4 neutralCOM;
    mr_float4 halfExtents;
    mr_float4 axis;
    mr_float4 limits;
};

// These are prospective engineering parameters, explicitly unmeasured.
// Torso mass and spatial inertia are partitioned, never duplicated.
constexpr std::array<Segment, 3u> segments{{
    {"crow_neck_yaw_link", "crow_neck_yaw", 0.001f,
     {0.120f, 0.0f, 0.075f, 0.0f}, {0.008f, 0.008f, 0.012f, 0.0f},
     {0.0f, 0.0f, 1.0f, 0.0f}, {-1.40f, 1.40f, 8.0f, 0.025f}},
    {"crow_head", "crow_neck_pitch", 0.035f,
     {0.160f, 0.0f, 0.090f, 0.0f}, {0.030f, 0.025f, 0.025f, 0.0f},
     {0.0f, 1.0f, 0.0f, 0.0f}, {-1.0f, 1.0f, 8.0f, 0.025f}},
    {"crow_lower_beak", "crow_beak_open", 0.004f,
     {0.205f, 0.0f, 0.079f, 0.0f}, {0.025f, 0.009f, 0.004f, 0.0f},
     {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.65f, 10.0f, 0.010f}},
}};

mr_float4 inertia(const Segment& segment) {
    const auto h = segment.halfExtents;
    return {segment.mass * (h.y*h.y + h.z*h.z) / 3.0f,
            segment.mass * (h.x*h.x + h.z*h.z) / 3.0f,
            segment.mass * (h.x*h.x + h.y*h.y) / 3.0f, 0.0f};
}

void diagonalInertia(MRBodyPropertiesGPU& body, const mr_float4 value) {
    body.inertiaRow0 = {value.x, 0.0f, 0.0f, 0.0f};
    body.inertiaRow1 = {0.0f, value.y, 0.0f, 0.0f};
    body.inertiaRow2 = {0.0f, 0.0f, value.z, 0.0f};
    body.inverseInertiaRow0 = {1.0f/value.x, 0.0f, 0.0f, 0.0f};
    body.inverseInertiaRow1 = {0.0f, 1.0f/value.y, 0.0f, 0.0f};
    body.inverseInertiaRow2 = {0.0f, 0.0f, 1.0f/value.z, 0.0f};
}
} // namespace

RobotPack makeAmericanCrowActiveHeadRobotPack() {
    auto source = builtinRobotPack("birdflow_american_crow_estimated_hybrid");
    if (!source) throw std::logic_error("Parent crow RobotPack is unavailable");
    RobotPack result = std::move(*source);
    result.id = "american_crow_active_head_estimated_v1";
    result.revision = 1u;
    result.sourceRevision += "+estimated-active-head-v1";
    result.capabilities.push_back("experimental_active_head_mechanics");
    result.capabilities.push_back("experimental_lower_beak_mechanics");
    EngineModel& model = result.mechanics;
    model.name = result.id;
    const auto firstBody = static_cast<std::uint32_t>(model.bodies.size());
    const auto firstJoint = static_cast<std::uint32_t>(model.joints.size());
    const auto firstQ = static_cast<std::uint32_t>(model.defaultQ.size());
    const auto firstV = static_cast<std::uint32_t>(model.defaultV.size());
    // No references into vectors survive the appends below.
    auto torso = model.bodies.at(0u);
    float remainingMass = torso.massAndInverseMass.x;
    mr_float4 moment{};
    float ixx = torso.inertiaRow0.x, iyy = torso.inertiaRow1.y;
    float izz = torso.inertiaRow2.z, ixz = torso.inertiaRow0.z;
    for (const auto& segment : segments) {
        remainingMass -= segment.mass;
        const auto r = segment.neutralCOM;
        moment.x += segment.mass*r.x;
        moment.z += segment.mass*r.z;
        const auto local = inertia(segment);
        ixx -= local.x + segment.mass*r.z*r.z;
        iyy -= local.y + segment.mass*(r.x*r.x + r.z*r.z);
        izz -= local.z + segment.mass*r.x*r.x;
        ixz += segment.mass*r.x*r.z;
    }
    if (!(remainingMass > 0.0f)) throw std::logic_error("Invalid crow mass partition");
    const mr_float4 shift{-moment.x/remainingMass, 0.0f,
                         -moment.z/remainingMass, 0.0f};
    ixx -= remainingMass*shift.z*shift.z;
    iyy -= remainingMass*(shift.x*shift.x + shift.z*shift.z);
    izz -= remainingMass*shift.x*shift.x;
    ixz += remainingMass*shift.x*shift.z;
    const float determinant = ixx*izz - ixz*ixz;
    if (!(ixx > 0.0f && iyy > 0.0f && izz > 0.0f && determinant > 0.0f)) {
        throw std::logic_error("Crow head partition produced nonpositive inertia");
    }
    torso.massAndInverseMass = {remainingMass, 1.0f/remainingMass, 0.0f, 0.0f};
    torso.inertiaRow0 = {ixx, 0.0f, ixz, 0.0f};
    torso.inertiaRow1 = {0.0f, iyy, 0.0f, 0.0f};
    torso.inertiaRow2 = {ixz, 0.0f, izz, 0.0f};
    torso.inverseInertiaRow0 = {izz/determinant, 0.0f, -ixz/determinant, 0.0f};
    torso.inverseInertiaRow1 = {0.0f, 1.0f/iyy, 0.0f, 0.0f};
    torso.inverseInertiaRow2 = {-ixz/determinant, 0.0f, ixx/determinant, 0.0f};
    model.bodies[0u] = torso;
    model.defaultQ[0u] += shift.x;
    model.defaultQ[2u] += shift.z;
    for (auto& joint : model.joints) {
        if (joint.parentBody == 0u) {
            joint.parentAnchor.x -= shift.x;
            joint.parentAnchor.z -= shift.z;
        }
    }
    for (auto& shape : model.shapes) {
        if (shape.bodyIndex == 0u) {
            // Replace the inherited whole-head torso proxy with a torso-only
            // box. New physical head/beak proxies are independently owned.
            shape.localPosition.x = -0.030f - shift.x;
            shape.localPosition.z -= shift.z;
            shape.dimensions.x = 0.130f;
            shape.contactRestAndBoundingRadius.z = std::sqrt(
                shape.dimensions.x*shape.dimensions.x +
                shape.dimensions.y*shape.dimensions.y +
                shape.dimensions.z*shape.dimensions.z);
        }
    }
    // These points are authored in the root COM frame. Preserve their neutral
    // physical locations when that frame moves during the mass partition.
    for (auto& wing : result.flappingWings->wings) {
        wing.rootToCenterAndArea.x -= shift.x;
        wing.rootToCenterAndArea.z -= shift.z;
    }
    result.flappingWings->tail.rootToCenterAndArea.x -= shift.x;
    result.flappingWings->tail.rootToCenterAndArea.z -= shift.z;

    for (std::uint32_t index = 0u; index < segments.size(); ++index) {
        const auto& segment = segments[index];
        MRBodyPropertiesGPU body{};
        body.articulationIndex = 0u;
        body.parentBody = index == 0u ? 0u : firstBody + index - 1u;
        body.inboundJoint = firstJoint + index;
        body.motionType = MR_MOTION_DYNAMIC;
        body.massAndInverseMass = {segment.mass, 1.0f/segment.mass, 0.0f, 0.0f};
        diagonalInertia(body, inertia(segment));
        body.dampingAndSpeedLimits = {0.01f, 0.01f, 30.0f, 20.0f};
        model.bodies.push_back(body);
        model.bodyNames.push_back(segment.body);

        MRJointDescriptorGPU joint{};
        joint.parentBody = body.parentBody;
        joint.childBody = firstBody + index;
        joint.jointType = MR_JOINT_REVOLUTE;
        joint.qOffset = firstQ + index; joint.nq = 1u;
        joint.vOffset = firstV + index; joint.nv = 1u;
        joint.axis0 = segment.axis;
        const mr_float4 parentCOM = index == 0u ? shift : segments[index-1u].neutralCOM;
        const mr_float4 hinge = index < 2u ? segments[0u].neutralCOM
                                         : mr_float4{0.180f, 0.0f, 0.083f, 0.0f};
        joint.parentAnchor = {hinge.x-parentCOM.x, 0.0f, hinge.z-parentCOM.z, 0.0f};
        joint.childAnchor = {hinge.x-segment.neutralCOM.x, 0.0f, hinge.z-segment.neutralCOM.z, 0.0f};
        joint.parentRotation = joint.childRotation = {0.0f, 0.0f, 0.0f, 1.0f};
        model.joints.push_back(joint);
        model.jointNames.push_back(segment.joint);
        MRDofPropertiesGPU dof{};
        dof.articulationIndex = 0u; dof.jointIndex = firstJoint + index;
        dof.qIndex = firstQ + index; dof.vIndex = firstV + index;
        dof.flags = MR_DOF_FLAG_ACTUATED | MR_DOF_FLAG_POSITION_LIMIT |
                    MR_DOF_FLAG_VELOCITY_LIMIT | MR_DOF_FLAG_EFFORT_LIMIT | MR_DOF_FLAG_DRIVE;
        dof.limits = segment.limits;
        dof.drive = {0.30f, 0.015f, 0.000002f, 0.0f};
        model.dofs.push_back(dof);
        model.dofNames.push_back(segment.joint);
        model.defaultQ.push_back(0.0f); model.defaultV.push_back(0.0f);
        if (!model.actuatorProfiles.empty()) model.actuatorProfiles.emplace_back();
        result.actuators.push_back({
            .id = index == 0u ? "head.yaw" : index == 1u ? "head.pitch" : "beak.open",
            .kind = RobotActuatorKind::jointPosition, .target = segment.joint,
            .scale = index == 0u ? 1.0f : index == 1u ? 0.8f : 0.65f,
            .responseTimeSeconds = 0.020f,
            .parameters = {0.0f, 0.0f, 1.0f, 0.0f},
        });
        if (index > 0u) {
            auto shape = model.shapes.front();
            shape.bodyIndex = firstBody + index;
            shape.localPosition = {0.0f, 0.0f, 0.0f, 1.0f};
            shape.localRotation = {0.0f, 0.0f, 0.0f, 1.0f};
            shape.dimensions = segment.halfExtents;
            shape.slotGeneration = static_cast<std::uint32_t>(model.shapes.size()+1u);
            shape.contactRestAndBoundingRadius.z = std::sqrt(
                shape.dimensions.x*shape.dimensions.x + shape.dimensions.y*shape.dimensions.y +
                shape.dimensions.z*shape.dimensions.z);
            model.shapes.push_back(shape);
            model.shapeNames.push_back(std::string(segment.body)+"/collision");
            if (index == 1u) {
                // The upper beak is rigidly attached to the head; only the
                // lower beak opens. Its mass is included in the head segment.
                shape.localPosition = {0.045f, 0.0f, -0.003f, 1.0f};
                shape.dimensions = {0.025f, 0.009f, 0.003f, 0.0f};
                shape.slotGeneration = static_cast<std::uint32_t>(model.shapes.size()+1u);
                shape.contactRestAndBoundingRadius.z = std::sqrt(
                    0.025f*0.025f + 0.009f*0.009f + 0.003f*0.003f);
                model.shapes.push_back(shape);
                model.shapeNames.push_back("crow_head/upper_beak_collision");
            }
        }
    }
    auto& articulation = model.articulations.at(0u);
    articulation.bodyCount = model.world.bodyCount = static_cast<std::uint32_t>(model.bodies.size());
    articulation.jointCount = model.world.jointCount = static_cast<std::uint32_t>(model.joints.size());
    articulation.nq = model.world.nq = static_cast<std::uint32_t>(model.defaultQ.size());
    articulation.nv = model.world.nv = static_cast<std::uint32_t>(model.defaultV.size());
    model.world.shapeCount = static_cast<std::uint32_t>(model.shapes.size());
    for (auto& role : result.roles) {
        if (role.id == "whole_body") role.members = model.bodyNames;
        if (role.id == "all_joints") role.members = model.jointNames;
        if (role.id == "all_dofs") role.members = model.dofNames;
    }
    result.roles.push_back({"head", RobotSemanticKind::body, {"crow_head"}});
    result.roles.push_back({"lower_beak", RobotSemanticKind::body, {"crow_lower_beak"}});
    std::string reason;
    if (!model.valid(&reason)) throw std::logic_error("Active-head crow invalid: "+reason);
    return result;
}

TaskPack makeAmericanCrowEmbodiedTaskPack(
    TaskObservationProgram& observations, TaskResetProgram& reset
) {
    TaskPack task = makeBirdFlowAmericanCrowNeuralJourneyTaskPack(observations,reset);
    task.id = "american_crow_embodied_development_v11";
    const auto robot = makeAmericanCrowActiveHeadRobotPack();
    task.baseHeightTarget = robot.mechanics.defaultQ[2u];
    for (const char* actuator : {"head.yaw", "head.pitch", "beak.open"}) {
        task.actions.push_back({.actuator=actuator});
        const TaskObservationOperatorSpec previous{
            .source=TaskObservationSource::previousAction, .target=actuator,
        };
        observations.actorCurrent.push_back(previous);
        observations.critic.push_back(previous);
    }
    for (const auto& segment : segments) {
        const TaskObservationOperatorSpec position{
            .source=TaskObservationSource::jointPositionError, .target=segment.joint,
        };
        const TaskObservationOperatorSpec velocity{
            .source=TaskObservationSource::jointVelocity, .target=segment.joint, .scale=0.125f,
        };
        observations.actorCurrent.push_back(position);
        observations.actorCurrent.push_back(velocity);
        observations.critic.push_back(position);
        observations.critic.push_back(velocity);
    }
    return task;
}

TaskPack makeAmericanCrowFeedingStationTaskPack(
    TaskObservationProgram& observations, TaskResetProgram& reset
) {
    auto task=makeAmericanCrowEmbodiedTaskPack(observations,reset);
    const MRFiniteResourceProgramGPU resourceProgram{{15,22,1,100000},
        {0.065f,0,0,0.08f},{0.20f,0,0,0}};
    const std::array<MRFiniteResourcePatchGPU,1> resourcePatches{{
        {{17,45000,20000,0},{0.225585f,0,0.271365f,0.030f}}}};
    task.deviceStateProgramFingerprint=finiteResourceProgramFingerprint(resourceProgram,resourcePatches,true);
    task.id="american_crow_feeding_station_v12";
    // A canonical feeding-station curriculum admits mouth skills separately
    // from navigation. No episode clock orders takeoff or flight stages.
    reset={};
    task.difficultyBandCount=1u;
    task.maximumEpisodeSteps=1000000u;
    task.commands={};
    task.commands.standingProbability=1.0f;
    task.pushes={};
    task.rewards={{.operation=TaskRewardOperator::nutritionIntake,.weight=100000.0f}};
    task.outcomes={{.id="accepted_intake_utility",.unit="utility",
        .source=TaskOutcomeSource::rewardContribution,
        .direction=TaskOutcomeDirection::higherIsBetter,
        .rewardOperation=TaskRewardOperator::nutritionIntake}};
    for (unsigned component=0;component<4;++component) {
        const TaskObservationOperatorSpec input{.source=TaskObservationSource::nutrition,.component=component,
                                               .scale=component==1u ? 50000.0f : 1.0f};
        observations.actorCurrent.push_back(input);observations.critic.push_back(input);
    }
    return task;
}

FiniteResourceConfig makeAmericanCrowFeedingResourceConfig(const CompiledRun& run,unsigned environments) {
    if (run.task().layout().actorObservationSize!=97u ||
        run.task().actorOperators().empty() ||
        run.task().actorOperators().back().source.x!=MR_TASK_OBSERVE_NUTRITION)
        throw std::invalid_argument("Crow feeding resources require the feeding-station task");
    FiniteResourceConfig config;
    config.worldFingerprint=run.world().fingerprint();config.environmentCount=environments;
    config.bodyCount=run.model().bodies.size();config.nq=run.model().defaultQ.size();
    config.publishTaskState=true;
    const auto head=std::find(run.model().bodyNames.begin(),run.model().bodyNames.end(),"crow_head");
    const auto jaw=std::find(run.model().jointNames.begin(),run.model().jointNames.end(),"crow_beak_open");
    if (head==run.model().bodyNames.end() || jaw==run.model().jointNames.end() || run.world().sceneBodyIndices().empty())
        throw std::invalid_argument("Crow feeding station lacks its physical head or ground anchor");
    config.program.binding={static_cast<unsigned>(head-run.model().bodyNames.begin()),
        run.model().joints[jaw-run.model().jointNames.begin()].qOffset,1,100000};
    config.program.mouthClosed={0.065f,0,0,0.08f};config.program.opening={0.20f,0,0,0};
    config.patches.push_back({{run.world().sceneBodyIndices().front(),45000,20000,0},
                             {0.225585f,0.0f,0.271365f,0.030f}});
    if (finiteResourceProgramFingerprint(config.program,config.patches,true)!=
        run.task().header().deviceStateProgramFingerprint)
        throw std::invalid_argument("Feeding resource semantics differ from the compiled policy task contract");
    return config;
}
} // namespace metalrobo
