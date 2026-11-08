#include "metalrobo/CrowManipulation.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace metalrobo;
int main(int argc,char** argv) {
    bool owns=false;std::filesystem::path output;
    try {
        if(argc!=2)throw std::invalid_argument("Usage: metalrobo_crow_object_push_compile NEW_OUTPUT");
        output=std::filesystem::absolute(argv[1]);
        if(!std::filesystem::create_directory(output))throw std::invalid_argument("Output directory exists");
        owns=true;auto m=makeAmericanCrowObjectPushFixtureManifest();
        CompiledRun physical;auto status=compileRun(m,physical);
        if(!status.succeeded())throw std::runtime_error("Compile physical fixture: "+status.message);
        const auto visual=authorAmericanCrowForagingVisual(physical,output,false,crowObjectPushCameraOrientation);
        const auto program=authorAmericanCrowObjectPushRetinalProgram(physical,visual);
        m.sensors.observation={};m.task=makeAmericanCrowObjectPushTaskPack(m.sensors.observation,m.reality.reset,&*program.retinal);
        m.sensors.deviceVisual=program;m.task.capacities=m.profile.capacities;
        CompiledRun run;status=compileRun(m,run);
        if(!status.succeeded())throw std::runtime_error("Compile retinal task: "+status.message);
        if(run.world().fingerprint()!=physical.world().fingerprint())throw std::runtime_error("Sensor authoring changed physics");
        std::ofstream f(output/"retinal-observation.json");f<<std::setprecision(10);
        f<<R"({"format":"numi.retinal-observation.v1","id":"native-object-push-v14-development","packs":[)";
        for(unsigned i=0;i<program.assets.size();++i) {
            const auto& p=program.assets[i];if(i)f<<',';
            f<<"{\"path\":"<<std::quoted(std::filesystem::path(p.path).filename().string())
                <<",\"asset_id\":"<<std::quoted(p.assetId)<<",\"semantic_id\":"<<p.semanticId<<",\"instance_id\":"<<p.instanceId<<'}';
        }
        f<<R"(],"camera":{"parent_body":"crow_head","position":[0.018,0.026,0.012],"orientation":[-0.5416752204,0.5416752204,-0.4545194777,0.4545194777],"width":160,"height":120,"minimum_visible_pixels":1,"vertical_field_of_view_degrees":45,"nominal_rate_hz":15,"maximum_retained_bytes":0},"retinal":{"right_position":[0.018,-0.026,0.012],"right_orientation":[-0.5416752204,0.5416752204,-0.4545194777,0.4545194777],"width":16,"height":12,"sample_numerator":3,"sample_denominator":10,"latency_ticks":2,"minimum_depth_meters":0.01,"maximum_depth_meters":5}})"<<'\n';
        f.close();if(!f)throw std::runtime_error("Write retinal configuration failed");
        std::cout<<"{\"status\":\"passed\",\"task_id\":\""<<m.task.id<<"\",\"world_fingerprint\":"<<run.world().fingerprint()
            <<",\"task_fingerprint\":"<<run.task().fingerprint()<<",\"sensor_fingerprint\":"<<program.fingerprint
            <<",\"actor_features\":"<<run.task().layout().actorObservationSize<<",\"critic_features\":"<<run.task().layout().criticObservationSize
            <<",\"simulation_steps\":0,\"biological_calibration\":false}\n";
        return 0;
    }catch(const std::exception& e){if(owns)std::ofstream(output/"failure.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
