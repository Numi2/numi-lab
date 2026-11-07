#include "/Users/numi/numi-g1-tracker-camera-owner-20261107/src/c_api_retained_research.cpp"
#include <array>
#include <iostream>
int main() {
 std::array<MRTaskRolloutDynamicSphereC,6> balls{};
 for(std::size_t i=0;i<6;++i) { balls[i].position[0]=4.f+float(i);balls[i].position[2]=2.f;balls[i].radius=.1f;balls[i].mass=1.f;balls[i].launch_step=0; }
 MRTaskRolloutConfigC cfg{};cfg.environment_count=2;cfg.physics_substeps=4;cfg.velocity_iterations=4;cfg.final_velocity_iterations=2;cfg.control_timestep_seconds=.02f;cfg.seed=4081015;cfg.dynamic_spheres=balls.data();cfg.dynamic_sphere_count=6;
 auto m=makeUnitreeG1RunManifest(metalrobo::LocomotionSurface::ground,metalrobo::UnitreeG1Task::ballDisturbanceRecovery,cfg);
 m.task.capacities.rawContacts=8;
 using O=metalrobo::TaskRandomizationOperator;
 std::erase_if(m.reality.reset.operators,[](const auto& op){return op.operation==O::sceneBodyPosition || op.operation==O::sceneBodyVelocity || op.operation==O::sceneBodyLaunchStep || op.operation==O::sceneBodyEventImpact;});
 m.teacher.id="no_teacher";m.profile.capacities=m.task.capacities;
 metalrobo::CompiledRun run;auto status=metalrobo::compileRun(m,run);
 std::cout<<"status="<<unsigned(status.status)<<" element="<<status.element<<" message="<<status.message<<'\n';
 if(!status.succeeded()) return 1;
 std::cout<<"impact_events="<<run.task().impactEvents().size()<<" actor_frame="<<run.task().layout().actorFrameSize<<'\n';
}
