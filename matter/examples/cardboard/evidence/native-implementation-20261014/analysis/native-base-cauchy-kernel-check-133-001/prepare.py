from pathlib import Path
B=Path('/Users/home/cardboard-implementation-20261014');O=B/'analysis/native-base-cauchy-kernel-check-133-001';O.mkdir(exist_ok=False)
p=(B/'analysis/native-residual-direction-replay-419-001/native_residual_direction.mm').read_text()
p=p[:p.index('\n\nstruct ResidualResult')]
p=p.replace('const bool massOnly = false)', 'const bool massOnly = false,\n                            const bool trueResidual = true,\n                            const bool requireSuccess = true)')
p=p.replace('micro.flags = NM_MICROSTEP_FGMRES_OPERATOR;', 'micro.flags = NM_MICROSTEP_FGMRES_OPERATOR |\n        (trueResidual ? NM_MICROSTEP_FGMRES_TRUE_RESIDUAL : 0u);')
p=p.replace('const auto fgmresB = makeBuffer(device.gpu,\n        std::vector<NMFGMRESStateGPU>(eCount));', 'std::vector<NMFGMRESStateGPU> latchedStates(eCount);\n    for (auto& state : latchedStates) state.diagnostics.z = 1.0f;\n    const auto fgmresB = makeBuffer(device.gpu, latchedStates);\n    std::vector<nm_float4> fallbackMetrics(eCount);\n    fallbackMetrics[0].w = 3.0f; fallbackMetrics[1].w = 1.0f;\n    const auto fallbackMetricsB = makeBuffer(device.gpu, fallbackMetrics);')
p=p.replace('id<MTLBuffer> args[26] = {};','id<MTLBuffer> args[27] = {};').replace('args[24] = cacheValidB; args[25] = useCacheB;', 'args[24] = cacheValidB; args[25] = useCacheB; args[26] = fallbackMetricsB;').replace('i < 26u; ++i','i < 27u; ++i').replace('need(statuses[e].code == NM_STATUS_SUCCESS,','need(!requireSuccess || statuses[e].code == NM_STATUS_SUCCESS,')
p+='''

void save(const std::filesystem::path& path, id<MTLBuffer> buffer) {
    std::ofstream out(path,std::ios::binary); need(out.good(),"output unavailable");
    out.write(static_cast<const char*>(buffer.contents),buffer.length);
    need(out.good(),"output write failed");
}
template<class T> void saveVector(const std::filesystem::path& path,const std::vector<T>& values) {
    std::ofstream out(path,std::ios::binary);need(out.good(),"output unavailable");
    out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));need(out.good(),"output write failed");
}
void kernelCheck(Device& device,const CompiledWorld& world,const Payload& p,const std::filesystem::path& output) {
    const auto E=p.header.environments,N=p.header.nodes;
    const auto count=std::size_t(E)*N;
    auto dispatch=world.dispatch;
    std::vector<nm_float4> metrics(E);metrics[0]={1.0f,2.0f,0.0f,3.0f};metrics[1]={4.0f,5.0f,0.75f,1.0f};
    const auto metricsB=makeBuffer(device.gpu,metrics);
    const auto nodesB=makeBuffer(device.gpu,p.candidate);
    const auto rhsB=makeBuffer(device.gpu,p.rhs);
    const auto baseB=makeBuffer(device.gpu,std::vector<nm_float4>(count));
    const auto emptyB=makeBuffer(device.gpu,std::vector<nm_float4>(count));
    const auto directionB=makeBuffer(device.gpu,std::vector<nm_float4>(count));
    auto invoke=[&](NSString* name,NSUInteger rows,auto bind){
        auto command=[device.queue commandBuffer];auto encoder=[command computeCommandEncoder];
        need(command&&encoder,"kernel encoder failed");[encoder setComputePipelineState:device.makePipeline(name)];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];bind(encoder);
        [encoder dispatchThreads:MTLSizeMake(rows,1u,1u) threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        need(command.status==MTLCommandBufferStatusCompleted,"kernel command did not complete");
    };
    invoke(@"numi_matter_metal::nm_fgmres_capture_base_cauchy_rhs",count,[&](id<MTLComputeCommandEncoder> enc){
        [enc setBuffer:rhsB offset:0u atIndex:1u];[enc setBuffer:nodesB offset:0u atIndex:2u];[enc setBuffer:baseB offset:0u atIndex:3u];
    });save(output/"base-rhs.bin",baseB);
    auto prepare=[&](id<MTLBuffer> gradient,std::uint32_t useGradient){
        invoke(@"numi_matter_metal::nm_fgmres_prepare_base_cauchy_action",count,[&](id<MTLComputeCommandEncoder> enc){
            [enc setBuffer:metricsB offset:0u atIndex:1u];[enc setBuffer:baseB offset:0u atIndex:2u];[enc setBuffer:gradient offset:0u atIndex:3u];
            [enc setBuffer:nodesB offset:0u atIndex:4u];[enc setBuffer:directionB offset:0u atIndex:5u];[enc setBytes:&useGradient length:sizeof(useGradient) atIndex:6u];
        });
        const auto* d=static_cast<const nm_float4*>(directionB.contents);return std::vector<nm_float4>(d,d+count);
    };
    Payload working=p;working.direction=prepare(emptyB,0u);save(output/"transpose-input.bin",directionB);
    auto fem=device.makePipeline(@"numi_matter_metal::nm_fem_apply_operator_elements");auto gather=device.makePipeline(@"numi_matter_metal::nm_fgmres_gather_nodes");
    auto layoutFunction=[device.library newFunctionWithName:@"numi_matter_metal::nm_primal_contact_argument_layout"];
    need(layoutFunction!=nil,"contact layout absent");auto contactEncoder=[layoutFunction newArgumentEncoderWithBufferIndex:0u];
    auto transpose=evaluateNative(device,fem,gather,world,working,contactEncoder,3u);saveVector(output/"gradient.bin",transpose.action);
    auto gradientB=makeBuffer(device.gpu,transpose.action);
    auto latch=evaluateNative(device,fem,gather,world,working,contactEncoder,3u,false,false);saveVector(output/"latch-control.bin",latch.action);
    Payload invalidUnselected=working;invalidUnselected.tangentValid[p.header.tetrahedra]=0u;
    auto unselected=evaluateNative(device,fem,gather,world,invalidUnselected,contactEncoder,3u);saveVector(output/"unselected-invalid-cache.bin",unselected.action);
    Payload invalidSelected=working;invalidSelected.tangentValid[0]=0u;
    auto selected=evaluateNative(device,fem,gather,world,invalidSelected,contactEncoder,3u,false,true,false);saveVector(output/"selected-invalid-cache-status.bin",selected.status);
    need(selected.status[0].code!=NM_STATUS_SUCCESS&&selected.status[1].code==NM_STATUS_SUCCESS,"strict cache mask failure");
    working.direction=prepare(gradientB,1u);save(output/"forward-input.bin",directionB);
    auto forward=evaluateNative(device,fem,gather,world,working,contactEncoder,4u);saveVector(output/"gradient-action.bin",forward.action);
    auto forwardB=makeBuffer(device.gpu,forward.action);
    std::vector<nm_float4> solution(2u*count,nm_float4{0.125f,-0.25f,0.5f,0.0f});
    const auto solutionB=makeBuffer(device.gpu,solution);
    std::vector<nm_float4> line(E,nm_float4{0.25f,3.0f,0.5f,48.0f});
    const auto lineB=makeBuffer(device.gpu,line);const auto alphaB=makeBuffer(device.gpu,std::vector<float>(E,0.75f));
    const auto traceB=makeBuffer(device.gpu,std::vector<nm_float4>(E));
    std::vector<NMFGMRESStateGPU> states(E);for(auto& s:states)s.diagnostics.z=1.0f;
    const auto statesB=makeBuffer(device.gpu,states);const auto statusB=makeBuffer(device.gpu,std::vector<NMMatterStatusGPU>(E));
    const auto trialStatusB=makeBuffer(device.gpu,std::vector<NMMatterStatusGPU>(E));
    NMFGMRESLayoutGPU layout{};layout.unknownCount=static_cast<std::uint32_t>(2u*count);
    const std::uint32_t iteration=48u;
    invoke(@"numi_matter_metal::nm_fgmres_build_base_cauchy_direction",E,[&](id<MTLComputeCommandEncoder> enc){
        [enc setBytes:&world.mixedSolver length:sizeof(world.mixedSolver) atIndex:1u];
        id<MTLBuffer> buffers[]={metricsB,nodesB,gradientB,forwardB,solutionB,lineB,alphaB,traceB,statesB,statusB,trialStatusB};
        for(NSUInteger i=0u;i<11u;++i)[enc setBuffer:buffers[i] offset:0u atIndex:i+2u];
        [enc setBytes:&iteration length:sizeof(iteration) atIndex:13u];[enc setBytes:&layout length:sizeof(layout) atIndex:30u];
    });
    save(output/"solution.bin",solutionB);save(output/"metrics.bin",metricsB);save(output/"line-search.bin",lineB);save(output/"environment-alpha.bin",alphaB);
    save(output/"fallback-trace.bin",traceB);save(output/"states.bin",statesB);save(output/"statuses.bin",statusB);save(output/"trial-statuses.bin",trialStatusB);
    need(std::memcmp(nodesB.contents,p.candidate.data(),nodesB.length)==0,"candidate inputs changed");
    need(std::memcmp(rhsB.contents,p.rhs.data(),rhsB.length)==0,"RHS inputs changed");
    const auto* actual=static_cast<const nm_float4*>(solutionB.contents);
    need(std::memcmp(actual+N,solution.data()+N,N*sizeof(nm_float4))==0,"unselected solution changed");
    need(std::memcmp(actual+count+N,solution.data()+count+N,N*sizeof(nm_float4))==0,"unselected field rows changed");
    need(std::memcmp(static_cast<const nm_float4*>(metricsB.contents)+1,metrics.data()+1,sizeof(nm_float4))==0,"unselected metrics changed");
    need(std::memcmp(static_cast<const NMFGMRESStateGPU*>(statesB.contents)+1,states.data()+1,sizeof(NMFGMRESStateGPU))==0,"unselected state changed");
    need(std::memcmp(static_cast<const nm_float4*>(lineB.contents)+1,line.data()+1,sizeof(nm_float4))==0,"unselected line search changed");
    const auto* statuses=static_cast<const NMMatterStatusGPU*>(statusB.contents);need(statuses[0].code==0u&&statuses[1].code==0u,"direction status failure");
    std::ofstream report(output/"native-contracts.json");report<<"{\\\"native_commands_complete\\\":true,\\\"unselected_environment_unchanged\\\":true,\\\"immutable_inputs_unchanged\\\":true,\\\"strict_cache_mask_control\\\":true,\\\"physical_transactions\\\":0}\\n";
}
} // namespace
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==4,"usage: source133 kernel check <package> <payload> <output>");auto p=readPayload(argv[2]);CompiledWorld world;std::string error;
    need(readPackage(argv[1],world,nullptr,&error),"package rejected: "+error);checkCompiledBinding(world,p);
    std::filesystem::create_directories(argv[3]);Device device;kernelCheck(device,world,p,argv[3]);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\\n";return 1;}}}
'''
(O/'native_cauchy_kernel_check.mm').write_text(p)
(O/'prepare.py').write_bytes(Path(__file__).read_bytes())
print(O)
