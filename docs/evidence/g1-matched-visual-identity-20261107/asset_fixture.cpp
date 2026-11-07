// CPU-only authoring of six synthetic, body-bound test visual packs.
#include "/Users/numi/numi-g1-tracker-plane-owner-20261107/src/c_api.cpp"
#include "metalrobo/VisualPlatform.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include "make_probe_visual_pack.inc"
int main(int argc,char** argv) {
 try {
  if(argc!=2) throw std::runtime_error("usage: asset_fixture OUTPUT_DIRECTORY");
  std::array<MRTaskRolloutDynamicSphereC,6> balls{};
  for(std::size_t i=0;i<balls.size();++i) {
   balls[i].position[0]=2.f+0.2f*i; balls[i].position[2]=1.f;
   balls[i].radius=0.1f; balls[i].mass=1.f; balls[i].launch_step=100u+100u*i;
  }
  MRTaskRolloutConfigC config{};config.environment_count=2;config.physics_substeps=4;config.velocity_iterations=4;config.final_velocity_iterations=2;config.control_timestep_seconds=.02f;config.seed=4081015u;config.dynamic_spheres=balls.data();config.dynamic_sphere_count=6;
  auto manifest=makeUnitreeG1RunManifest(metalrobo::LocomotionSurface::ground,metalrobo::UnitreeG1Task::ballDodge,config);
  manifest.teacher.id="no_teacher"; manifest.profile.capacities=manifest.task.capacities;
  metalrobo::CompiledRun run; auto status=metalrobo::compileRun(manifest,run);
  if(!status.succeeded()) throw std::runtime_error(status.element+": "+status.message);
  std::filesystem::create_directories(argv[1]);
  for(std::size_t i=0;i<6;++i) {
   const std::string name="locomotion_dynamic_sphere_"+std::to_string(i);
   auto it=std::ranges::find(run.model().bodyNames,name);
   if(it==run.model().bodyNames.end()) throw std::runtime_error("missing body "+name);
   const auto body=static_cast<std::uint32_t>(it-run.model().bodyNames.begin());
   auto pack=makeAuthoredObjectPack(body);
   pack.instances[0].identity.y=static_cast<std::uint32_t>(100u+i);
   pack.symbolicBindings[0].node=name;
   pack.symbolicBindings[0].link=name;
   pack.contentHash=metalrobo::computeVisualAssetPackContentHash(pack);
   const auto path=std::filesystem::path(argv[1])/(name+".mrvpack");
   std::string reason;
   if(!metalrobo::writeVisualAssetPack(pack,path,&reason)) throw std::runtime_error(reason);
   std::cout << name << " body=" << body << " path=" << path << '\n';
  }
  return 0;
 } catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
