from pathlib import Path
p=Path('/Users/numi/rod-paired-step-20261108');s=(p/'base-host.mm').read_text()
def change(old,new):
 global s
 assert s.count(old)==1, (old[:100],s.count(old));s=s.replace(old,new)
change('''        std::vector<mr_float4> positions(nodeElements);
        std::vector<mr_float4> velocities(nodeElements);''','''        std::vector<mr_float4> positions(nodeElements);
        std::vector<mr_float4> positionsLow(nodeElements);
        std::vector<mr_float4> velocities(nodeElements);''')
needle='''                velocities[environment * nodeCount + node] = {
                    static_cast<float>(velocity[0]),'''
insert='''                const mr_float4 high = positions[environment * nodeCount + node];
                positionsLow[environment * nodeCount + node] = {
                    static_cast<float>(position[0] - static_cast<double>(high.x)),
                    static_cast<float>(position[1] - static_cast<double>(high.y)),
                    static_cast<float>(position[2] - static_cast<double>(high.z)),
                    0.0f,
                };
'''
change(needle,insert+needle)
change('''        std::vector<MRRodGPUAttachment> attachments;
        attachments.reserve(input.attachments.size());''','''        std::vector<MRRodGPUAttachment> attachments;
        std::vector<mr_float4> attachmentLow;
        attachments.reserve(input.attachments.size());
        attachmentLow.reserve(input.attachments.size());''')
needle='''            record.nodeIndex = attachment.nodeIndex;
            attachments.push_back(record);'''
replace='''            record.nodeIndex = attachment.nodeIndex;
            attachmentLow.push_back({
                static_cast<float>(attachment.targetPosition[0] - static_cast<double>(record.targetAndCompliance.x)),
                static_cast<float>(attachment.targetPosition[1] - static_cast<double>(record.targetAndCompliance.y)),
                static_cast<float>(attachment.targetPosition[2] - static_cast<double>(record.targetAndCompliance.z)),
                0.0f,
            });
            attachments.push_back(record);''';change(needle,replace)
needle='''        id<MTLBuffer> inputAttachmentBuffer =
            inputBuffer(device, attachments);'''
insert='''        std::vector<float> restLow(edgeCount), stiffnessLow(edgeCount), inverseMassLow(nodeCount);
        for (std::size_t i = 0; i < edgeCount; ++i) {
            restLow[i] = static_cast<float>(model.restLengths[i] - static_cast<double>(restLengths[i]));
            stiffnessLow[i] = static_cast<float>(model.stretchStiffness[i] - static_cast<double>(stretchStiffness[i]));
        }
        for (std::size_t i = 0; i < nodeCount; ++i) {
            inverseMassLow[i] = static_cast<float>(1.0 / model.nodeMasses[i] - static_cast<double>(inverseMasses[i]));
        }
        id<MTLBuffer> pairBuffers[6] = {
            inputBuffer(device, positionsLow), inputBuffer(device, attachmentLow),
            inputBuffer(device, restLow), inputBuffer(device, stiffnessLow),
            inputBuffer(device, inverseMassLow), outputBuffer<mr_float4>(device, nodeElements),
        };
        for (id<MTLBuffer> buffer : pairBuffers) {
            if (buffer == nil) return reject(std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::metalBufferFailure,
                "failed to allocate research paired-position buffer");
        }
''';change(needle,insert+needle)
needle='''        [encoder setBuffer:referenceDirectorBuffer
                    offset:0u
                   atIndex:23u];'''
replace=needle+'''
        for (NSUInteger i = 0; i < 6; ++i) {
            [encoder setBuffer:pairBuffers[i] offset:0u atIndex:24u+i];
        }
        const float timestepLow = static_cast<float>(config.step.timestep -
            static_cast<double>(dispatch.gravityAndTimestep.w));
        [encoder setBytes:&timestepLow length:sizeof(timestepLow) atIndex:30u];''';change(needle,replace)
needle='''                state.positions[node] = {
                    position.x,
                    position.y,
                    position.z,
                };'''
replace='''                const mr_float4 low = static_cast<const mr_float4*>(pairBuffers[5].contents)[environment * nodeCount + node];
                state.positions[node] = {
                    static_cast<double>(position.x) + (toolPairs.empty() ? low.x : 0.0f),
                    static_cast<double>(position.y) + (toolPairs.empty() ? low.y : 0.0f),
                    static_cast<double>(position.z) + (toolPairs.empty() ? low.z : 0.0f),
                };''';change(needle,replace)
(p/'candidate-host.mm').write_text(s)
