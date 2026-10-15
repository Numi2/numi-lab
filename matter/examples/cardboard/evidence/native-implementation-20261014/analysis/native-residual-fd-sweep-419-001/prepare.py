from pathlib import Path
import hashlib,json,struct,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');S=B/'snapshots/source-status5-vtccd-diagnostics-125';P=B/'analysis/native-operator-term-replay-419-001';C=B/'solver/two-stage-first-failure-rich-fd-419';O=B/'analysis/native-residual-fd-sweep-419-001';O.mkdir();(O/'build').mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'manifest.json')=='3be02f1a7509836fa6ccc6b5528aba5b2c68f40589dbf7bc54d1745ee8f4dd5f'
records=[json.loads(line,parse_int=lambda s:-0.0 if s=='-0' else int(s)) for line in (C/'operator-capture').read_text().splitlines()]
ns=sorted([x for x in records if x['record']=='node'],key=lambda x:(x['environment'],x['node_id']))
assert len(ns)==1764
payload=(P/'prepared/replay-input.bin').read_bytes();payload=b'NMFDSWP1'+payload[8:]
for n in ns:payload+=struct.pack('<20f',*[v for row in n['accepted']['raw_state_float4s'] for v in row])
(O/'replay-input.bin').write_bytes(payload)
text=(P/'operator_terms_replay.mm').read_text()
text=text.replace('    std::vector<nm_float4> capturedFreshAction;','    std::vector<nm_float4> capturedFreshAction;\n    std::vector<NMFEMNodeStateGPU> acceptedNodes;')
text=text.replace('"NMCACHE1", 8u','"NMFDSWP1", 8u')
a='    need(stream.peek() == std::ifstream::traits_type::eof(),'
assert text.count(a)==1;text=text.replace(a,'    p.acceptedNodes = readVector<NMFEMNodeStateGPU>(stream, e * n, "accepted nodes");\n'+a)
new=r'''
struct ResidualResult {
    std::vector<NMFEMNodeStateGPU> nodes;
    std::vector<NMFEMElementVectorGPU> forces;
    std::vector<nm_float4> residual;
    std::vector<NMMatterStatusGPU> statuses;
};

ResidualResult evaluateResidual(Device& device, const CompiledWorld& world,
                                const Payload& p, float epsilon, bool negative) {
    const auto e = p.header.environments, n = p.header.nodes, t = p.header.tetrahedra;
    auto dispatch = world.dispatch;
    dispatch.gravityAndTimestep.w = p.header.dt;
    NMMicrostepGPU micro{};
    micro.microtickCount = 1u; micro.solverIteration = 48u;
    micro.time = {p.header.dt, 1.0f / p.header.dt, 0.0f, p.header.dt};
    const auto objectB = makeBuffer(device.gpu, world.objects);
    const auto materialsB = makeBuffer(device.gpu, world.materials);
    const auto programsB = makeBuffer(device.gpu, world.scalarPrograms);
    const auto instructionsB = makeBuffer(device.gpu, world.instructions);
    const auto parametersB = makeBuffer(device.gpu, p.parameters);
    const auto acceptedB = makeBuffer(device.gpu, p.acceptedNodes);
    const auto candidateB = makeBuffer(device.gpu, p.candidate);
    auto signedDirection = p.direction;
    if (negative) for (auto& d : signedDirection) { d.x = -d.x; d.y = -d.y; d.z = -d.z; }
    const auto directionB = makeBuffer(device.gpu, signedDirection);
    const auto rangesB = makeBuffer(device.gpu, repeatRanges(world, e));
    const auto incidenceB = makeBuffer(device.gpu, repeatIncidence(world, e));
    const auto schedulersB = makeBuffer(device.gpu, repeatObjectRows(world.schedulers, e, "scheduler"));
    const auto adaptiveB = makeBuffer(device.gpu, repeatObjectRows(world.adaptive, e, "adaptive"));
    const auto tetB = makeBuffer(device.gpu, repeatTetrahedra(world, e));
    const auto historyB = makeBuffer(device.gpu, p.acceptedHistory);
    const auto statusB = makeBuffer(device.gpu, std::vector<NMMatterStatusGPU>(e));
    const auto trialB = makeBuffer(device.gpu, std::vector<NMFEMNodeStateGPU>(std::size_t(e)*n));
    const auto forceB = makeBuffer(device.gpu, std::vector<NMFEMElementVectorGPU>(std::size_t(e)*t));
    std::vector<nm_float4> metrics(e); for (auto& m : metrics) m.z = epsilon;
    const auto metricsB = makeBuffer(device.gpu, metrics);
    const auto mixedB = makeBuffer(device.gpu, world.mixedMaterials);
    const auto fieldsB = makeBuffer(device.gpu, world.fem.fields);
    const auto learnedB = makeBuffer(device.gpu, world.learnedMaterials);
    const auto layersB = makeBuffer(device.gpu, world.learnedLayers);
    const auto weightsB = makeBuffer(device.gpu, world.learnedWeights);
    const auto solverB = makeBuffer(device.gpu, world.mixedSolver);
    const auto residualB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    const auto solutionB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    const auto zB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    const auto dB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    const auto aB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    const auto reactionsB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    const auto externalB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    auto command = [device.queue commandBuffer];
    auto encode = [&](NSString* name, NSUInteger count, auto bind) {
        auto encoder = [command computeCommandEncoder];
        auto pipeline = device.makePipeline(name);
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
        bind(encoder);
        [encoder dispatchThreads:MTLSizeMake(count,1u,1u) threadsPerThreadgroup:MTLSizeMake(64u,1u,1u)];
        [encoder endEncoding];
    };
    encode(@"numi_matter_metal::nm_fem_armijo_prepare_candidate",std::size_t(e)*n,[&](id<MTLComputeCommandEncoder> enc){
        id<MTLBuffer> args[]={objectB,rangesB,schedulersB,adaptiveB,acceptedB,candidateB,directionB,metricsB,statusB,trialB};
        for(NSUInteger i=0;i<10u;++i)[enc setBuffer:args[i] offset:0u atIndex:i+2u];
    });
    encode(@"numi_matter_metal::nm_fem_armijo_validate_geometry",e,[&](id<MTLComputeCommandEncoder> enc){
        id<MTLBuffer> args[]={objectB,materialsB,tetB,schedulersB,adaptiveB,trialB,statusB,metricsB};
        for(NSUInteger i=0;i<8u;++i)[enc setBuffer:args[i] offset:0u atIndex:i+2u];
    });
    encode(@"numi_matter_metal::nm_fem_internal_forces",std::size_t(e)*t,[&](id<MTLComputeCommandEncoder> enc){
        id<MTLBuffer> args[]={objectB,materialsB,programsB,instructionsB,parametersB,trialB,tetB,historyB,schedulersB,adaptiveB,forceB,statusB,mixedB,fieldsB,learnedB,layersB,weightsB};
        for(NSUInteger i=0;i<17u;++i)[enc setBuffer:args[i] offset:0u atIndex:i+2u];
    });
    encode(@"numi_matter_metal::nm_fem_build_mechanical_residual",std::size_t(e)*n,[&](id<MTLComputeCommandEncoder> enc){
        id<MTLBuffer> args[]={objectB,acceptedB,trialB,tetB,forceB,incidenceB,rangesB,schedulersB,adaptiveB,solutionB,residualB,zB,dB,aB,statusB,solverB,mixedB,fieldsB};
        for(NSUInteger i=0;i<18u;++i)[enc setBuffer:args[i] offset:0u atIndex:i+2u];
        const std::uint32_t preserve=1u,hasExternal=0u;
        [enc setBytes:&preserve length:sizeof(preserve) atIndex:20u];
        [enc setBuffer:externalB offset:0u atIndex:21u];
        [enc setBytes:&hasExternal length:sizeof(hasExternal) atIndex:22u];
        [enc setBuffer:reactionsB offset:0u atIndex:23u];
    });
    [command commit];[command waitUntilCompleted];
    need(command.status==MTLCommandBufferStatusCompleted,"native residual command did not complete");
    ResidualResult r;
    const auto* nodes=static_cast<const NMFEMNodeStateGPU*>(trialB.contents);r.nodes.assign(nodes,nodes+std::size_t(e)*n);
    const auto* forces=static_cast<const NMFEMElementVectorGPU*>(forceB.contents);r.forces.assign(forces,forces+std::size_t(e)*t);
    const auto* res=static_cast<const nm_float4*>(residualB.contents);r.residual.assign(res,res+std::size_t(e)*n);
    const auto* status=static_cast<const NMMatterStatusGPU*>(statusB.contents);r.statuses.assign(status,status+e);
    need(std::memcmp(historyB.contents,p.acceptedHistory.data(),historyB.length)==0,"accepted material history mutated");
    need(std::memcmp(acceptedB.contents,p.acceptedNodes.data(),acceptedB.length)==0,"accepted node history mutated");
    need(std::memcmp(candidateB.contents,p.candidate.data(),candidateB.length)==0,"center candidate mutated");
    return r;
}
'''
text=text.replace('struct Metrics {',new+'\nstruct Metrics {')
at='            writeReport(outputDir + "/replay-report.json", p, env0, env1);'
extra=r'''
            std::ofstream summary(outputDir + "/residual-sweep.jsonl");
            summary << std::setprecision(17);
            const float epsilons[]={0.0f,0.03125f,0.015625f,0.00390625f,0.0009765625f,0.000244140625f,0.00006103515625f,0.0000152587890625f,0.000003814697265625f,0.00000095367431640625f,0.0000002384185791015625f,0.000000059604644775390625f};
            for(std::size_t k=0;k<std::size(epsilons);++k)for(unsigned side=0;side<(k==0?1u:2u);++side){
                auto r=evaluateResidual(device,world,p,epsilons[k],side!=0u);
                const auto prefix=outputDir+"/sample-"+std::to_string(k)+"-"+std::to_string(side);
                const auto* nodeWords=reinterpret_cast<const nm_float4*>(r.nodes.data());
                writeFloat4File(prefix+"-nodes.bin",std::vector<nm_float4>(nodeWords,nodeWords+r.nodes.size()*5u));
                writeFloat4File(prefix+"-residual.bin",r.residual);
                writeElementFile(prefix+"-forces.bin",r.forces);
                summary<<"{\"sample\":"<<k<<",\"side\":"<<side<<",\"epsilon\":"<<epsilons[k]<<",\"status\":["<<r.statuses[0].code<<","<<r.statuses[1].code<<"],\"history_unchanged\":true}\n";
                summary.flush();
            }
'''
assert text.count(at)==1;text=text.replace(at,at+'\n'+extra)
(O/'native_residual_sweep.mm').write_text(text)
m=json.loads((P/'manifest.json').read_text());cmd=[x.replace(str(P/'operator_terms_replay.mm'),str(O/'native_residual_sweep.mm')).replace(str(P/'build/operator_terms_replay'),str(O/'build/native_residual_sweep')) for x in m['compile_argv']]
def run(name,argv):
 with (O/f'{name}.stdout.log').open('x') as out,(O/f'{name}.stderr.log').open('x') as err:q=subprocess.run(argv,stdout=out,stderr=err,timeout=180)
 assert q.returncode==0,(name,q.returncode)
run('compile',cmd)
argv=[str(O/'build/native_residual_sweep'),str(C/'output/compiled.nmatterpack'),str(O/'replay-input.bin')]
run('preflight',argv+['/tmp/cardboard424-preflight-unused','--preflight-only'])
(O/'prepare.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps(dict(source_snapshot=str(S),source_build_sha256=sha(R/'build-cardboard-provenance-125/source-build.json'),capture_sha256=sha(C/'operator-capture'),compile_argv=cmd,native_argv=argv+['OUTPUT_DIRECTORY'],preflight_pass=True,physical_transactions=0,files={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file()}),indent=2)+'\n')
print('prepared native FD sweep',sha(O/'manifest.json'))
