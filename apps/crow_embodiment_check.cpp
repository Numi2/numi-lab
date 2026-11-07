#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedABA.hpp"
#include "metalrobo/RunProgram.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void close(double a, double b, const char* message) {
    require(std::abs(a-b) <= 2.0e-7*(1.0+std::abs(a)), message);
}

struct GPUMetrics { double accelerationError, configurationError; std::string device; };
GPUMetrics gpuAdmission(const metalrobo::EngineModel& model) {
    constexpr unsigned environments=12u;
    const unsigned nq=model.defaultQ.size(), nv=model.defaultV.size();
    std::vector<float> q, v, effort;
    for (unsigned environment=0;environment<environments;++environment) {
        q.insert(q.end(),model.defaultQ.begin(),model.defaultQ.end());
        v.insert(v.end(),nv,0.0f); effort.insert(effort.end(),nv,0.0f);
        const float sign=environment%2u==0u ? 1.0f : -1.0f;
        q[environment*nq+nq-3u]=sign*0.1f*(environment/2u);
        q[environment*nq+nq-2u]=sign*0.06f*(environment/2u);
        q[environment*nq+nq-1u]=0.05f*(environment/2u);
        effort[environment*nv+nv-3u]=sign*0.001f;
        effort[environment*nv+nv-2u]=-sign*0.002f;
        effort[environment*nv+nv-1u]=0.0002f;
    }
    const metalrobo::MetalArticulatedABAInput input{
        .articulationIndex=0u, .environmentCount=environments,
        .q=q, .v=v, .effort=effort, .applyBodyDamping=false,
    };
    metalrobo::MetalArticulatedABAContext context;
    metalrobo::MetalArticulatedABAResult actual, replay;
    auto status=context.run(model,input,actual);
    if (!status.succeeded()) throw std::runtime_error("Native crow head ABA failed: "+status.message);
    require(status.published && status.failedEnvironmentCount==0u,"Native head result was not accepted");
    status=context.run(model,input,replay);
    require(status.succeeded() && replay.acceleration==actual.acceleration &&
            replay.nextQ==actual.nextQ && replay.nextV==actual.nextV,"Native head replay differs");
    double accelerationError=0.0, configurationError=0.0;
    for (unsigned environment=0;environment<environments;++environment) {
        std::vector<double> cq(q.begin()+environment*nq,q.begin()+(environment+1u)*nq);
        std::vector<double> cv(v.begin()+environment*nv,v.begin()+(environment+1u)*nv);
        std::vector<double> ce(effort.begin()+environment*nv,effort.begin()+(environment+1u)*nv);
        std::vector<double> acceleration(nv);
        metalrobo::ArticulatedDynamicsConfig config;
        config.timestep=model.world.gravityAndTimestep.w;
        config.applyBodyDamping=false;
        require(metalrobo::computeArticulatedForwardDynamics(model,0u,cq,cv,ce,{},acceleration,config).succeeded(),
                "Head dynamics oracle failed");
        for (unsigned dof=0;dof<nv;++dof) {
            accelerationError=std::max(accelerationError,
                std::abs(actual.acceleration[environment*nv+dof]-acceleration[dof])/(1.0+std::abs(acceleration[dof])));
        }
        require(metalrobo::integrateArticulatedState(model,0u,cq,cv,ce,{},config).succeeded(),"Head integration oracle failed");
        for (unsigned coordinate=0;coordinate<nq;++coordinate) {
            configurationError=std::max(configurationError,std::abs(actual.nextQ[environment*nq+coordinate]-cq[coordinate]));
        }
        for (unsigned dof=nv-3u;dof<nv;++dof) {
            require(std::abs(actual.nextV[environment*nv+dof])>1e-7,"Native effort failed to actuate head/beak");
        }
    }
    require(accelerationError<5e-5 && configurationError<5e-5,"Native head dynamics disagree with independent FP64 oracle");
    return {accelerationError,configurationError,status.deviceName};
}
}

int main() {
    try {
        const auto parent = metalrobo::builtinRobotPack("birdflow_american_crow_estimated_hybrid").value();
        const auto candidate = metalrobo::makeAmericanCrowActiveHeadRobotPack();
        const auto& oldModel = parent.mechanics;
        const auto& model = candidate.mechanics;
        require(model.bodies.size()==17u && model.joints.size()==16u &&
                model.defaultQ.size()==23u && model.defaultV.size()==22u,
                "Incorrect active-head topology");
        require(candidate.actuators.size()==parent.actuators.size()+3u,
                "Head/beak actuator authority missing");
        metalrobo::RunManifest manifest;
        manifest.id="crow_embodied_compiler_admission";
        manifest.robot=candidate;
        manifest.scene=metalrobo::makeBirdFlowAmericanCrowFlightScenePack();
        manifest.sensors.id="crow_embodied_proprioception";
        manifest.task=metalrobo::makeAmericanCrowEmbodiedTaskPack(
            manifest.sensors.observation,manifest.reality.reset);
        manifest.reality.id="crow_embodied_reality";
        manifest.teacher.id="no_teacher";
        manifest.profile.id="crow_embodied_compiler_control";
        manifest.profile.environmentCount=8u;
        manifest.profile.controlSteps=128u;
        manifest.profile.physicsSubsteps=4u;
        manifest.profile.controlTimestepSeconds=0.02f;
        metalrobo::CompiledRun compiled;
        const auto compiledStatus=metalrobo::compileRun(manifest,compiled);
        if (!compiledStatus.succeeded()) throw std::runtime_error(compiledStatus.message);
        require(compiled.task().actionBindings().size()==18u &&
                compiled.task().layout().actorObservationSize==93u &&
                compiled.task().layout().criticObservationSize==93u,
                "Embodied task observation/action contract changed");
        for (unsigned action=0;action<18u;++action) {
            require(compiled.task().actionBindings()[action].actuator.w==
                    (action>=15u ? MR_TASK_ACTUATOR_INDEPENDENT_POSTURE : 0u),
                    "Independent posture leaked into legacy actions or missed head actions");
        }
        const auto preserved=compiled.fingerprint();
        for (const float invalid : {0.5f,-1.0f,2.0f}) {
            manifest.robot.actuators.back().parameters.z=invalid;
            require(!metalrobo::compileRun(manifest,compiled).succeeded() &&
                    compiled.fingerprint()==preserved,
                    "Invalid posture flag was accepted or mutated the published run");
        }
        require(metalrobo::engineModelFingerprint(oldModel) != metalrobo::engineModelFingerprint(model),
                "Embodiment must have a distinct mechanics identity");
        double oldMass=0.0, newMass=0.0;
        for (const auto& body : oldModel.bodies) oldMass+=body.massAndInverseMass.x;
        for (const auto& body : model.bodies) newMass+=body.massAndInverseMass.x;
        close(oldMass,newMass,"Head/beak mass was duplicated or lost");

        // The reference oracle is used only for admission. Production remains
        // native Metal. Compare physical invariants under the exact change of
        // root COM coordinates, including rotating and translating mechanisms.
        for (int trial=0; trial<36; ++trial) {
            std::vector<double> oldQ(oldModel.defaultQ.begin(),oldModel.defaultQ.end());
            std::vector<double> q(model.defaultQ.begin(),model.defaultQ.end());
            std::vector<double> oldV(oldModel.defaultV.size()), v(model.defaultV.size());
            const double yaw=0.17*trial, c=std::cos(yaw), s=std::sin(yaw);
            const double dx=model.defaultQ[0]-oldModel.defaultQ[0];
            const double dz=model.defaultQ[2]-oldModel.defaultQ[2];
            const std::array<double,3> shift{c*dx,s*dx,dz};
            oldQ[0]=0.013*trial; oldQ[1]=-0.01*trial; oldQ[2]+=0.02*trial;
            oldQ[3]=0.0; oldQ[4]=0.0; oldQ[5]=std::sin(yaw/2); oldQ[6]=std::cos(yaw/2);
            std::copy(oldQ.begin()+3,oldQ.end(),q.begin()+3);
            for (unsigned axis=0;axis<3;++axis) q[axis]=oldQ[axis]+shift[axis];
            for (unsigned axis=0;axis<oldV.size();++axis) {
                v[axis]=oldV[axis]=0.3*std::sin(0.23*trial+0.77*axis);
            }
            v[0]+=oldV[4]*shift[2]-oldV[5]*shift[1];
            v[1]+=oldV[5]*shift[0]-oldV[3]*shift[2];
            v[2]+=oldV[3]*shift[1]-oldV[4]*shift[0];
            metalrobo::ArticulatedInvariants oldInvariants, invariants;
            require(metalrobo::computeArticulatedInvariants(oldModel,0u,oldQ,oldV,oldInvariants).succeeded(),
                    "Parent invariant oracle rejected state");
            require(metalrobo::computeArticulatedInvariants(model,0u,q,v,invariants).succeeded(),
                    "Active-head invariant oracle rejected state");
            close(oldInvariants.kineticEnergy,invariants.kineticEnergy,"Neutral kinetic energy changed");
            close(oldInvariants.potentialEnergy,invariants.potentialEnergy,"Neutral gravitational first moment changed");
            for (unsigned axis=0;axis<3;++axis) {
                close(oldInvariants.linearMomentum[axis],invariants.linearMomentum[axis],"Neutral linear momentum changed");
                close(oldInvariants.angularMomentum[axis],invariants.angularMomentum[axis],"Neutral spatial inertia changed");
            }
            std::vector<metalrobo::ArticulatedBodyKinematics> oldBodies(oldModel.bodies.size()), bodies(model.bodies.size());
            require(metalrobo::computeArticulatedBodyKinematics(oldModel,0u,oldQ,oldV,oldBodies).succeeded(),"Parent FK failed");
            require(metalrobo::computeArticulatedBodyKinematics(model,0u,q,v,bodies).succeeded(),"Active-head FK failed");
            for (unsigned body=1;body<oldBodies.size();++body) for (unsigned axis=0;axis<3;++axis) {
                close(oldBodies[body].centerOfMassPosition[axis],bodies[body].centerOfMassPosition[axis],"Inherited anchor moved");
                close(oldBodies[body].linearVelocity[axis],bodies[body].linearVelocity[axis],"Inherited body twist changed");
            }
            q[q.size()-3u]=0.6; q[q.size()-2u]=-0.4; q[q.size()-1u]=0.3;
            std::vector<metalrobo::ArticulatedBodyKinematics> moved(model.bodies.size());
            require(metalrobo::computeArticulatedBodyKinematics(model,0u,q,v,moved).succeeded(),"Active FK failed");
            double headMotion=0.0, jawMotion=0.0;
            for (unsigned axis=0;axis<3;++axis) {
                headMotion+=std::pow(moved[15].centerOfMassPosition[axis]-bodies[15].centerOfMassPosition[axis],2);
                jawMotion+=std::pow(moved[16].centerOfMassPosition[axis]-bodies[16].centerOfMassPosition[axis],2);
            }
            require(headMotion>1e-4 && jawMotion>1e-4,"Head/beak coordinates do not move their physical bodies");
        }
        const auto gpu=gpuAdmission(model);
        std::cout<<std::setprecision(17)
            <<"{\"schema\":\"numi.crow-head-mechanics-admission.v1\",\"status\":\"passed\","
            <<"\"robot\":\"american_crow_active_head_estimated_v1\",\"reference_pose_twist_cases\":36,"
            <<"\"native_environments\":12,\"neutral_mass_kg\":"<<newMass
            <<",\"acceleration_scaled_error\":"<<gpu.accelerationError
            <<",\"configuration_absolute_error\":"<<gpu.configurationError
            <<",\"device\":"<<std::quoted(gpu.device)
            <<",\"exact_replay\":true,\"all_head_beak_joints_actuated\":true,"
            <<"\"compiler_contract_and_rejection_controls\":true,"
            <<"\"boundary\":\"estimated single-step mechanics; no head vision, contacts, biology, or behavioral qualification\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
