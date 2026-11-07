#define main numi_suture_probe_entry_for_checkpoint_test
#include "../../apps/dual_psm_suture_handoff_probe.cpp"
#undef main
#include <unistd.h>

int main() {
    using namespace metalrobo;
    char pathTemplate[]="/tmp/numi-suture-twist-test-XXXXXX";
    const char* directory=mkdtemp(pathTemplate);
    if(directory==nullptr) return 1;
    const std::filesystem::path path=std::filesystem::path(directory)/"state.tsv";
    const auto makeWorld=[] {
        HeterogeneousWorld world;
        world.model.world.nq=1;world.model.world.nv=1;
        world.model.defaultQ={7};world.model.defaultV={8};
        world.model.bodies.resize(1);world.sceneBodyIndices={0};
        world.defaultSceneBodies.resize(1);
        world.rods.resize(1);
        auto& rod=world.rods[0];
        rod.model.restPositions.resize(2);rod.model.restTwists.resize(1);
        rod.defaultState.positions.resize(2);rod.defaultState.velocities.resize(2);
        rod.defaultState.twists={9};rod.defaultState.twistRates={10};
        return world;
    };
    const auto write=[&](const char* schema,bool twist,bool duplicate) {
        std::ofstream out(path);
        out << "schema\t" << schema << "\nphase\ttissue-receiver-bridge-start\nstep\t1\n"
            << "q\t0\nv\t0\nneedle_position\t0\t0\t0\n"
            << "needle_orientation_xyzw\t0\t0\t0\t1\n"
            << "needle_linear_velocity\t0\t0\t0\nneedle_angular_velocity\t0\t0\t0\n"
            << "thread_position\t0\t0\t0\t0\nthread_position\t1\t1\t0\t0\n"
            << "thread_velocity\t0\t0\t0\t0\nthread_velocity\t1\t0\t0\t0\n";
        if(twist) out << "thread_twist\t0\t0.25\t0.5\n";
        if(duplicate) out << "thread_twist\t0\t0.75\t1.5\n";
        if(std::string_view(schema).ends_with(".v3"))
            out << "matter_snapshot\tmissing.matter.bin\t1\t1\t1\t1\n";
    };
    auto checkRejected=[&](const char* expected){
        auto world=makeWorld();
        try { (void)loadHandoffState(path,"tissue-receiver-bridge-start",world); }
        catch(const std::exception& e) {
            if(std::string(e.what()).find(expected)!=std::string::npos &&
               world.model.defaultQ[0]==7 && world.rods[0].defaultState.twists[0]==9)
                return true;
        }
        return false;
    };
    write("numi.dual-psm-suture-handoff-state.v3",false,false);
    if(!checkRejected("missing archived thread twist state")) return 2;
    write("numi.dual-psm-suture-handoff-state.v3",true,true);
    if(!checkRejected("duplicate resume thread edge")) return 3;
    write("numi.dual-psm-suture-handoff-state.v3",true,false);
    if(!checkRejected("Matter snapshot archive failed identity validation")) return 4;
    write("numi.dual-psm-suture-handoff-state.v2",true,false);
    auto recorded=makeWorld();
    if(loadHandoffState(path,"tissue-receiver-bridge-start",recorded)!=1 ||
       recorded.rods[0].defaultState.twists[0]!=0.25 ||
       recorded.rods[0].defaultState.twistRates[0]!=0.5) return 5;
    write("numi.dual-psm-suture-handoff-state.v2",false,false);
    auto legacy=makeWorld();
    if(loadHandoffState(path,"tissue-receiver-bridge-start",legacy)!=1 ||
       legacy.rods[0].defaultState.twists[0]!=0) return 6;
    std::filesystem::remove(path);
    std::filesystem::remove(directory);
    std::cout << "v3 missing or duplicate twist rejected before mutation; legacy fallback retained\n";
}
