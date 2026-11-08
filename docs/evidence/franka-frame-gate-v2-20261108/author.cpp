#include "/Users/numi/numi-franka-frame-gate-owner-20261108/src/c_api_retained_research.cpp"
#include "metalrobo/VisualPlatform.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include "make_probe_visual_pack.inc"
int main(int argc,char** argv) {try {
 if(argc!=2) throw std::runtime_error("usage: author OUTPUT");
 MRTaskRolloutConfigC cfg{};cfg.environment_count=2;cfg.physics_substeps=4;cfg.velocity_iterations=4;cfg.final_velocity_iterations=2;cfg.control_timestep_seconds=.02f;cfg.seed=20261108;
 auto manifest=makeFrankaPickPlaceRunManifest(cfg);manifest.teacher.id="no_teacher";manifest.profile.capacities=manifest.task.capacities;
 metalrobo::CompiledRun run;auto status=metalrobo::compileRun(manifest,run);
 if(!status.succeeded())throw std::runtime_error(status.element+": "+status.message);
 auto it=std::ranges::find(run.model().bodyNames,"pick_object");
 if(it==run.model().bodyNames.end())throw std::runtime_error("pick_object missing");
 auto body=static_cast<std::uint32_t>(it-run.model().bodyNames.begin());
 auto pack=makeAuthoredObjectPack(body);pack.instances[0].identity.y=105u;pack.symbolicBindings[0].node="pick_object";pack.symbolicBindings[0].link="pick_object";
 pack.contentHash=metalrobo::computeVisualAssetPackContentHash(pack);
 std::string reason;if(!metalrobo::writeVisualAssetPack(pack,argv[1],&reason))throw std::runtime_error(reason);
 std::cout<<"asset=pick_object body="<<body<<" content_hash="<<pack.contentHash<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<"author failure: "<<e.what()<<'\n';return 1;}}

