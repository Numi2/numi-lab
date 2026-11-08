from pathlib import Path
p=Path('/Users/numi/tracker-masked-policy-20261108/probe.cpp')
s=p.read_text()
def rep(a,b):
 global s
 assert s.count(a)==1
 s=s.replace(a,b)
rep('''            require(policyTrace != nil && policyInputs != nil, "policy trace allocation");''','''            id<MTLBuffer> actionOutputs=[policyDevice newBufferWithLength:30u*kEnvironmentCount*layout.actionCount*sizeof(float) options:MTLResourceStorageModeShared];
            require(policyTrace != nil && policyInputs != nil && actionOutputs != nil, "policy trace allocation");''')
rep('''                __strong id<MTLBuffer> inputs;
                std::uint32_t pixels;''','''                __strong id<MTLBuffer> inputs;
                __strong id<MTLBuffer> outputs;
                std::uint32_t pixels;''')
rep('''            } action{pipeline,policyTrace,policyInputs,visualPixels,''','''            } action{pipeline,policyTrace,policyInputs,actionOutputs,visualPixels,''')
rep('''                    [enc endEncoding]; return true;
                }, .policyRevision''','''                    [enc endEncoding];
                    id<MTLBlitCommandEncoder> outputCopy=[cb blitCommandEncoder];
                    NSUInteger outputBytes=pass.environmentCount*pass.actionCount*sizeof(float);
                    if (!outputCopy || (pass.controlStep+1u)*outputBytes>owner->outputs.length) return false;
                    [outputCopy copyFromBuffer:actions sourceOffset:0 toBuffer:owner->outputs
                        destinationOffset:pass.controlStep*outputBytes size:outputBytes];
                    [outputCopy endEncoding]; return true;
                }, .policyRevision''')
rep('''                saveBuffer(".policy",policyTrace); saveBuffer(".inputs",policyInputs);''','''                saveBuffer(".policy",policyTrace); saveBuffer(".inputs",policyInputs); saveBuffer(".actions",actionOutputs);''')
p.write_text(s)
