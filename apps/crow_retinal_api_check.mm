#import <Foundation/Foundation.h>
#include "CrowRetinalFixture.hpp"
using namespace crow_retinal_fixture;
namespace {
struct Stream {std::vector<float> actor,actions,latents,q,v;std::vector<MRTaskTransitionC> transitions;};
Stream execute(Fixture& f,const std::filesystem::path& out,unsigned chunk,const std::filesystem::path& policy) {@autoreleasepool {
    require(std::filesystem::create_directory(out),"Evidence path exists");auto h=f.create();api(mr_task_rollout_load_policy_pack(h.get(),policy.c_str()),"install control actor");
    api(mr_task_rollout_set_state_readback(h.get(),1),"enable final state inspection");Stream result;std::vector<float> previous;
    for (unsigned start=0;start<T;start+=chunk) {
        const unsigned count=std::min(chunk,T-start);std::vector<unsigned> masks(count*N,0);MRTaskRolloutAdvanceC a{};
        api(mr_task_rollout_advance(h.get(),nullptr,0,masks.data(),masks.size(),count,1,1,&a),"terminal-enabled advance");
        require(a.successful_environment_steps==count*N && !a.failed_environment_steps,"C rollout failed physics");
        const auto* actor=mr_task_rollout_actor_observations(h.get());const auto* actions=mr_task_rollout_policy_actions(h.get());
        const auto* latents=mr_task_rollout_policy_latents(h.get());const auto* transitions=mr_task_rollout_transitions(h.get());
        require(actor && actions && latents && transitions,"Missing learning publication");
        if (!previous.empty()) for (unsigned env=0;env<N;++env)
            require(!std::memcmp(&previous[env*F+97],actor+env*F+97,1154*sizeof(float)),"Terminal retinal features differ from next chunk's first actor input");
        previous.assign(actor+count*N*F,actor+(count+1)*N*F);
        save(out/("terminal-"+std::to_string(start+count)+".f32"),previous);
        for (unsigned t=0;t<=count;++t) for (unsigned env=0;env<N;++env) {
            const auto index=(t*N+env)*F;const unsigned tick=start+t;
            require(actor[index+1249]==(tick<2 ? 0 : 1),"Terminal/live validity differs from two-tick delay");
            double sum=0;for (unsigned c=97;c<1249;++c) {require(std::isfinite(actor[index+c]),"Nonfinite actor image");sum+=actor[index+c];}
            require(tick<2 ? sum==0 : sum>1,"Terminal/live slot silently lost its rendered RGB");
        }
        result.actor.insert(result.actor.end(),actor,actor+count*N*F);result.actions.insert(result.actions.end(),actions,actions+count*N*18);
        result.latents.insert(result.latents.end(),latents,latents+count*N*18);result.transitions.insert(result.transitions.end(),transitions,transitions+count*N);
    }
    result.q.assign(mr_task_rollout_final_q(h.get()),mr_task_rollout_final_q(h.get())+N*23);
    result.v.assign(mr_task_rollout_final_v(h.get()),mr_task_rollout_final_v(h.get())+N*22);
    save(out/"actor.f32",result.actor);save(out/"actions.f32",result.actions);save(out/"latents.f32",result.latents);
    save(out/"transitions.bin",result.transitions);save(out/"q.f32",result.q);save(out/"v.f32",result.v);return result;
}}
}
int main(int argc,char** argv) {@autoreleasepool {bool owns=false;try {
    require(argc==2,"Usage: metalrobo_crow_retinal_api_check NEW_OUTPUT");const std::filesystem::path out=argv[1];
    require(std::filesystem::create_directory(out),"Output exists");owns=true;
    std::ofstream(out/"plan.json")<<R"({"schema":"numi.crow-retinal-api-admission.v5","parent":"01 and02 exposed partition-dependent retinal frames with exact physics/actions; phase0 alone did not fix it. Terminal pass used the regular pre-step state selector, whose guard rejects controlStep==count and leaves stale projection input. v3 projects accepted terminal state directly and passed partition/terminal checks, but its immediate open/negative-close gesture consumed nothing. v4 established settling/neutral-close still consumed nothing. Compiler inspection found v13 missing the avian ground-curriculum flag inherited by v12. v5 compiles that supported-station flag; control schedule/predictions unchanged","unit":"two native lanes at50Hz in canonical fixed supported station; explicit generated collision proxies, no biological or visual-learning claim","seed":2651108801,"predictions":{"negative_configs_rejected":6,"terminal_and_next_retina_bit_identical":true,"chunk1_8_complete_learning_stream_bit_identical":true,"cold_start_blind_ticks":2,"zero_sensor_and_physics_errors":true,"world_reset_preserves_consumed_food":true},"stopping":"two16-tick partition runs plus bounded64-settle/32-open/32-neutral-close feeding and8-tick world-reset control; retain failed attempts, no threshold changes"})";
    Fixture f(out);std::ofstream negatives(out/"negative-controls.jsonl");
    for (unsigned test=0;test<6;++test) {
        auto c=f.c;auto retinal=f.retina;const MRRetinalSensorConfigC* input=&retinal;
        if (test==0) retinal.struct_size=0;if (test==1) retinal.abi_version=2;
        if (test==2) input=nullptr;if (test==3) c.visual_sensor_program=nullptr;
        if (test==4) c.source=MR_RUN_SOURCE_BIRDFLOW_DOVE;
        CrowRunOwner h(test==5 ? mr_create_task_rollout(&c) : mr_create_task_rollout_retinal(&c,input),mr_task_rollout_destroy);
        negatives<<"{\"case\":"<<test<<",\"rejected\":"<<(!h ? "true" : "false")<<",\"reason\":"<<std::quoted(mr_last_error())<<"}\n";negatives.flush();require(!h,"Malformed retinal C configuration was accepted");
    }
    const auto zero=f.policy(out,"zero-actor-with-critic");const auto one=execute(f,out/"chunk1",1,zero),eight=execute(f,out/"chunk8",8,zero);
    const auto same=[](const auto& a,const auto& b){return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]));};
    require(same(one.actor,eight.actor) && same(one.actions,eight.actions) && same(one.latents,eight.latents) && same(one.transitions,eight.transitions) && same(one.q,eight.q) && same(one.v,eight.v),"Terminal-enabled learning stream depends on submission partition");
    {auto h=f.create();const auto open=f.policy(out,"open-beak",1),close=f.policy(out,"close-beak",0);
        std::array<MRTaskResourceAccountingC,N> before{},after{};
        std::vector<unsigned> masks(32*N,0);MRTaskRolloutAdvanceC a{};
        api(mr_task_rollout_load_policy_pack(h.get(),close.c_str()),"install settling control");
        for (unsigned i=0;i<2;++i) api(mr_task_rollout_advance(h.get(),nullptr,0,masks.data(),masks.size(),32,1,1,&a),"settle supported body before feeding");
        api(mr_task_rollout_load_policy_pack(h.get(),open.c_str()),"install open-beak control");
        api(mr_task_rollout_advance(h.get(),nullptr,0,masks.data(),masks.size(),32,1,1,&a),"open beak");
        api(mr_task_rollout_load_policy_pack(h.get(),close.c_str()),"install close-beak control");
        api(mr_task_rollout_advance(h.get(),nullptr,0,masks.data(),masks.size(),32,1,1,&a),"close beak");
        api(mr_task_rollout_copy_resource_accounting(h.get(),before.data(),N),"publish pre-reset food");
        api(mr_task_rollout_reset(h.get(),2651108802),"reset world and retina");masks.resize(8*N);
        api(mr_task_rollout_advance(h.get(),nullptr,0,masks.data(),masks.size(),8,1,1,&a),"advance after world reset");
        api(mr_task_rollout_copy_resource_accounting(h.get(),after.data(),N),"publish post-reset food");
        save(out/"resource-before-reset.bin",std::vector(before.begin(),before.end()));save(out/"resource-after-reset.bin",std::vector(after.begin(),after.end()));
        for (unsigned env=0;env<N;++env) {
            require(before[env].consumed_micrograms>0,"Controlled gesture did not consume food before reset");
            require(before[env].remaining_micrograms==after[env].remaining_micrograms && before[env].crop_micrograms==after[env].crop_micrograms &&
                before[env].consumed_micrograms==after[env].consumed_micrograms && after[env].accepted_control_ticks==before[env].accepted_control_ticks+8,"World reset refilled food or changed lifetime accounting");
        }
        const auto* actor=mr_task_rollout_actor_observations(h.get());
        for (unsigned t=0;t<=8;++t) for (unsigned env=0;env<N;++env) require(actor[(t*N+env)*F+1249]==(t<2 ? 0 : 1),"Whole-world reset retained stale retinal frames");
    }
    const std::string result=R"({"schema":"numi.crow-retinal-api-evidence.v1","status":"passed","malformed_configs_rejected":6,"terminal_retinal_slots_valid":true,"terminal_next_retina_bit_identical":true,"chunk1_8_bit_identical":true,"world_reset_preserves_food":true,"visual_learning_qualified":false,"ecological_environment_qualified":false})";
    std::ofstream(out/"result.json")<<result<<'\n';std::cout<<result<<'\n';return 0;
} catch (const std::exception& e) {if (owns) std::ofstream(std::filesystem::path(argv[1])/"failure.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}}}
