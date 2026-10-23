from pathlib import Path
p=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/sparse-bone-contact-135/source-overlay/tests/numi_human_fem_body_triangle_contact/fem_body_triangle_contact_coupon.mm")
s=p.read_text(); marker="        std::printf(\"{\\\"status\\\":\\\"PASS\\\",\\\"device\\\":\\\"%s\\\",\""
assert marker in s
code=r"""
        // Compare the sparse owner to the preserved pre-change GPU owner.
        // Unsorted slave IDs, inactive gaps, nonzero incoming loads and three
        // environments expose indexing/overwrite errors that one node cannot.
        constexpr unsigned envs=3u, nodeCount=17u, sampleCount=5u;
        NMNumiHumanTendonFEMLoadDispatchGPU sparseDispatch=dispatch;
        sparseDispatch.environmentCount=envs;
        sparseDispatch.femNodeCount=nodeCount;
        sparseDispatch.bodyPoseStride=2u;
        sparseDispatch.femBodyContactSampleCount=sampleCount;
        sparseDispatch.femBodyContactCandidateTriangleCount=0u;
        std::array<NMNumiHumanFEMBodyContactSampleGPU,sampleCount> sparseSamples{};
        const std::array<unsigned,sampleCount> slaveIds{{12u,1u,16u,4u,9u}};
        for (unsigned i=0;i<sampleCount;++i) {
            sparseSamples[i]=samples[i%3u==0u ? 5u : (i%3u==1u ? 6u : 1u)];
            sparseSamples[i].slaveNode=slaveIds[i];
            sparseSamples[i].bodyIndex=1u;
        }
        std::array<NMFEMNodeStateGPU,envs*nodeCount> sparseNodes{};
        std::array<MRArticulatedBodyPoseGPU,envs*2u> sparsePoses{};
        std::array<nm_float4,envs*nodeCount> initialLoads{};
        for (unsigned e=0;e<envs;++e) {
            for (unsigned b=0;b<2u;++b) {
                sparsePoses[e*2u+b].position=f4(0,0,0,0);
                sparsePoses[e*2u+b].orientation=f4(0,0,0,1);
            }
            for (unsigned n=0;n<nodeCount;++n) {
                const unsigned row=e*nodeCount+n;
                sparseNodes[row].positionAndMass=f4(.4f,.4f,
                    .01f*float(int(n%5u)-2)-.004f*float(e),1);
                initialLoads[row]=f4(.125f*float(n),-.25f*float(e),.03125f,77.0f);
            }
        }
        auto sdBuffer=dispatchBuffer(&sparseDispatch,sizeof(sparseDispatch));
        auto ssBuffer=dispatchBuffer(sparseSamples.data(),sizeof(sparseSamples));
        auto snBuffer=dispatchBuffer(sparseNodes.data(),sizeof(sparseNodes));
        auto spBuffer=dispatchBuffer(sparsePoses.data(),sizeof(sparsePoses));
        auto sparseLoads=dispatchBuffer(initialLoads.data(),sizeof(initialLoads));
        auto denseLoads=dispatchBuffer(initialLoads.data(),sizeof(initialLoads));
        id<MTLCommandBuffer> parityCommand=[queue commandBuffer];
        auto parityEncode=[&](NSString* name,id<MTLBuffer> result,unsigned threads) {
            NSError* error=nil;
            auto fn=[library newFunctionWithName:[@"numi_matter_metal::" stringByAppendingString:name]];
            auto pipeline=[device newComputePipelineStateWithFunction:fn error:&error];
            require(pipeline!=nil,"sparse/dense oracle pipelines");
            auto encoder=[parityCommand computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:sdBuffer offset:0 atIndex:0];
            [encoder setBuffer:ssBuffer offset:0 atIndex:1];
            [encoder setBuffer:snBuffer offset:0 atIndex:2];
            [encoder setBuffer:spBuffer offset:0 atIndex:3];
            [encoder setBuffer:result offset:0 atIndex:4];
            [encoder setBuffer:ownerRangeBuffer offset:0 atIndex:5];
            [encoder setBuffer:ownerCandidateBuffer offset:0 atIndex:6];
            [encoder dispatchThreads:MTLSizeMake(threads,1,1)
                threadsPerThreadgroup:MTLSizeMake(8,1,1)];
            [encoder endEncoding];
        };
        parityEncode(@"nm_numi_human_assemble_fem_body_contact_loads",sparseLoads,envs*sampleCount);
        parityEncode(@"nm_numi_human_assemble_fem_body_contact_loads_dense_oracle",denseLoads,envs*nodeCount);
        [parityCommand commit];[parityCommand waitUntilCompleted];
        require(parityCommand.status==MTLCommandBufferStatusCompleted,"sparse/dense oracle completes");
        const auto* sparseResult=static_cast<const nm_float4*>(sparseLoads.contents);
        const auto* denseResult=static_cast<const nm_float4*>(denseLoads.contents);
        unsigned changedNodes=0;
        for (unsigned row=0;row<envs*nodeCount;++row) {
            for (unsigned axis=0;axis<4u;++axis) {
                const auto* a=reinterpret_cast<const float*>(&sparseResult[row]);
                const auto* b=reinterpret_cast<const float*>(&denseResult[row]);
                require(std::isfinite(a[axis]) && std::abs(a[axis]-b[axis])<=1.0e-6f,
                    "sparse force matches old GPU kernel including untouched rows and w");
            }
            changedNodes+=sparseResult[row].x!=initialLoads[row].x ||
                sparseResult[row].y!=initialLoads[row].y || sparseResult[row].z!=initialLoads[row].z;
        }
        require(changedNodes>=envs,"oracle must exercise actual nonzero contact/attachment forces");
        std::printf("sparse_dense_gpu_parity=PASS environments=%u nodes=%u contacts=%u changed=%u\n",
                    envs,nodeCount,sampleCount,changedNodes);

"""
s=s.replace(marker,code+marker);p.write_text(s)
