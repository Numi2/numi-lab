#include "metalrobo/CrowForaging.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace metalrobo;
int main(int argc,char** argv) {
    bool owns=false;std::filesystem::path output;
    try {
        if(argc!=5 && argc!=6)throw std::invalid_argument("Usage: metalrobo_crow_foraging_compile CALIBRATED_PORTS_BIN NEW_OUTPUT ALLOCATION_SEED OCCLUDED_0_OR_1 [MEMORY_CHOICE_DECISION_TICK]");
        unsigned decision=0;
        if(argc==6) {
            std::size_t used=0;const std::string text=argv[5];const auto value=std::stoul(text,&used);
            if(used!=text.size() || value<48 || value>192)throw std::invalid_argument("Memory-choice decision tick must be48..192");
            decision=unsigned(value);
        }
        std::array<mr_float4,2> ports{};
        std::ifstream calibration(argv[1],std::ios::binary);
        calibration.read(reinterpret_cast<char*>(ports.data()),sizeof(ports));
        if(!calibration || calibration.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("Calibration must contain exactly two native float4 port records");
        std::size_t consumed=0;const std::string seedText=argv[3];
        const auto seed=std::stoull(seedText,&consumed);
        if(consumed!=seedText.size() || seedText.empty() || seedText.front()=='-')throw std::invalid_argument("Allocation seed must be an unsigned64 integer");
        // The last explicit flag is deliberately required: it is world identity.
        const std::string blockedText=argv[4];
        if(blockedText!="0" && blockedText!="1")throw std::invalid_argument("Occlusion must be0 or1");
        const bool blocked=blockedText=="1";
        output=argv[2];
        if(!std::filesystem::create_directory(output))throw std::invalid_argument("Output directory exists");
        owns=true;
        auto manifest=makeAmericanCrowForagingFixtureManifest(blocked);applyAmericanCrowForagingFixtureLimits(manifest);
        CompiledRun physical;const auto compiled=compileRun(manifest,physical);
        if(!compiled.succeeded())throw std::invalid_argument("Compile physical fixture: "+compiled.message);
        const auto visual=authorAmericanCrowForagingVisual(physical,output,true);
        const auto program=authorAmericanCrowCuedForagingProgram(physical,visual,ports,manifest.profile.environmentCount,seed);
        std::ofstream artifact(output/"retinal-observation.json");artifact<<std::setprecision(10);
        artifact<<R"({"format":"numi.retinal-observation.v1","id":"native-calibrated-cued-foraging-development","packs":[)";
        for(unsigned i=0;i<program.sensor.assets.size();++i) {
            const auto& pack=program.sensor.assets[i];
            if(i)artifact<<',';
            artifact<<"{\"path\":"<<std::quoted(std::filesystem::path(pack.path).filename().string())<<",\"asset_id\":"<<std::quoted(pack.assetId)
                <<",\"semantic_id\":"<<pack.semanticId<<",\"instance_id\":"<<pack.instanceId<<'}';
        }
        artifact<<R"(],"camera":{"parent_body":"crow_head","position":[0.018,0.026,0.012],"orientation":[-0.5,0.5,-0.5,0.5],"width":160,"height":120,"minimum_visible_pixels":1,"vertical_field_of_view_degrees":45,"nominal_rate_hz":15,"maximum_retained_bytes":0},"retinal":{"right_position":[0.018,-0.026,0.012],"right_orientation":[-0.5,0.5,-0.5,0.5],"width":16,"height":12,"sample_numerator":3,"sample_denominator":10,"latency_ticks":2,"minimum_depth_meters":0.01,"maximum_depth_meters":5},"cued_foraging":{"ports":[)";
        for(unsigned i=0;i<2;++i){if(i)artifact<<',';const auto p=ports[i];artifact<<'['<<p.x<<','<<p.y<<','<<p.z<<','<<p.w<<']';}
        artifact<<"],\"cue_mesh_instances\":["<<visual.resourceCues[0].binding.x<<','<<visual.resourceCues[1].binding.x<<"],\"allocation_seed\":"<<seed<<",\"occluded\":"<<(blocked?"true":"false");
        if(decision)artifact<<",\"memory_choice_decision_tick\":"<<decision;
        artifact<<"}}\n";
        artifact.close();if(!artifact)throw std::runtime_error("Write cue environment configuration failed");
        std::cout<<"{\"status\":\"passed\",\"world_fingerprint\":"<<physical.world().fingerprint()<<",\"sensor_fingerprint\":"<<program.sensor.fingerprint
            <<",\"resource_program_fingerprint\":"<<finiteResourceProgramFingerprint(program.resources.program,program.resources.patches,true,program.resources.allocation)
            <<",\"simulation_steps\":0,\"biological_calibration\":false}\n";return 0;
    }catch(const std::exception& e){if(owns)std::ofstream(output/"failure.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
