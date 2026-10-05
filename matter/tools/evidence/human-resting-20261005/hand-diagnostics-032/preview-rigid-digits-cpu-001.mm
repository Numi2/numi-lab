// Static geometry diagnostic using the existing FP64 kinematics owner only.
// No physical stepping, controller updates, or Metal dispatches occur here.
#define main unused_native_entry_point
#include "/Users/n/numi-human-resting-hand-reduction-source-018/apps/numilab_human_myosim_visual_probe.mm"
#undef main

int main(int argc, char** argv) {
    @autoreleasepool { try {
        require(argc==5,"usage: rigid equality retained-native-log output-json");
        const auto rigid=loadRigid(argv[1]);
        const auto equality=loadJointEqualities(argv[2],rigid.header);
        std::vector<MRNumiHumanJointEqualityGPU> reduced;std::string error;
        require(metalrobo::compileNumiHumanRestingHandReduction(equality.payload,
            rigid.model.defaultQ,rigid.model.dofs,reduced,error),error);
        std::ifstream input(argv[3]);require(input.good(),"native log unavailable");
        std::string line,receipt;bool matched=false;
        while(std::getline(input,line)) {
            if(line.starts_with("rigid_source_rest_frames=")&&line.find(
                "rigid_sha256="+loadedKneeSHA256Hex(rigid.payloadSha256))!=std::string::npos)matched=true;
            if(line.starts_with("stand_terminal_state=")) {
                require(receipt.empty(),"multiple terminal states");receipt=line.substr(21);
            }
        }
        require(matched&&!receipt.empty(),"missing source-bound terminal state");
        NSData* bytes=[NSData dataWithBytes:receipt.data() length:receipt.size()];
        NSDictionary* state=[NSJSONSerialization JSONObjectWithData:bytes options:0 error:nil];
        require([state[@"schema"] isEqual:@"numi.human.legacy-stand-terminal.v1"],"wrong terminal schema");
        NSArray* coordinates=state[@"q"];require(coordinates.count==rigid.model.defaultQ.size(),"wrong q count");
        std::vector<double> q(coordinates.count),v(rigid.model.dofs.size(),0.);
        for(unsigned i=0;i<q.size();++i)q[i]=double([coordinates[i] floatValue]);
        NSMutableDictionary* result=[@{@"physical_steps_advanced":@0,@"metal_dispatches":@0,
            @"scope":@"Static counterfactual geometry preview only. Rigid digits reset to source reference; retained remaining q is unchanged. Not a dynamic result."} mutableCopy];
        for(unsigned mode=0;mode<2;++mode) {
            if(mode)for(unsigned i=51;i<reduced.size();++i)
                q[reduced[i].indices.x]=reduced[i].referencesAndCoefficients0.x;
            std::vector<metalrobo::ArticulatedBodyKinematics> poses(rigid.model.bodies.size());
            require(metalrobo::computeArticulatedBodyKinematics(rigid.model,0,q,v,poses).succeeded(),"existing kinematics failed");
            NSMutableArray* bodies=[NSMutableArray array];
            for(const auto& p:poses) [bodies addObject:@{@"body_index":@(p.bodyIndex),
                @"position_world_m":@[@(p.centerOfMassPosition[0]),@(p.centerOfMassPosition[1]),@(p.centerOfMassPosition[2])],
                @"orientation_world_xyzw":@[@(p.orientation[0]),@(p.orientation[1]),@(p.orientation[2]),@(p.orientation[3])]}];
            result[mode?@"rigid_digits":@"retained_original"]=bodies;
        }
        NSData* output=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil];
        require([output writeToFile:@(argv[4]) atomically:YES],"cannot write geometry diagnostic");
        std::cout<<"static_pose_count=2 physical_steps_advanced=0 metal_dispatches=0\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}}
}
