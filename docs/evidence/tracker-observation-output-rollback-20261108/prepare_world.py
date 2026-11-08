from pathlib import Path
s=Path('world-original.cpp').read_text()
def change(a,b):
 global s
 assert s.count(a)==1,(a[:60],s.count(a));s=s.replace(a,b)
change('''            require(captures != nil, "actor capture allocation");''','''            require(captures != nil, "actor capture allocation");
            const std::size_t outputSnapshotBytes = kEnvironmentCount * layout.actorObservationSize * sizeof(float);
            id<MTLBuffer> outputCaptures = [captureDevice newBufferWithLength:30u*3u*outputSnapshotBytes options:MTLResourceStorageModeShared];
            require(outputCaptures != nil, "output capture allocation");''')
change('''                __strong id<MTLBuffer> history;
                NSUInteger bytes;
            } capture{resetTrace, trackerProgram, captures, nil, actorSnapshotBytes};''','''                __strong id<MTLBuffer> history;
                __strong id<MTLBuffer> outputSnapshots;
                __strong id<MTLBuffer> output;
                NSUInteger outputOffset;
                NSUInteger outputBytes;
                NSUInteger bytes;
            } capture{resetTrace, trackerProgram, captures, nil, outputCaptures, nil, 0u, outputSnapshotBytes, actorSnapshotBytes};''')
change('''                    c->history=(__bridge id<MTLBuffer>)pass.actorHistory;
                    id<MTLCommandBuffer> cmd''','''                    c->history=(__bridge id<MTLBuffer>)pass.actorHistory;
                    c->output=(__bridge id<MTLBuffer>)pass.actorObservations;
                    c->outputOffset=pass.actorObservationOffsetElements*sizeof(float);
                    id<MTLCommandBuffer> cmd''')
change('''                        if (!blit || !c->history || c->history.length < c->bytes) return false;
                        [blit copyFromBuffer:c->history sourceOffset:0 toBuffer:c->snapshots
                            destinationOffset:(pass.controlStep*3u+phase)*c->bytes size:c->bytes];''','''                        if (!blit || !c->history || c->history.length < c->bytes || !c->output ||
                            c->outputOffset > c->output.length || c->outputBytes > c->output.length-c->outputOffset) return false;
                        [blit copyFromBuffer:c->history sourceOffset:0 toBuffer:c->snapshots
                            destinationOffset:(pass.controlStep*3u+phase)*c->bytes size:c->bytes];
                        [blit copyFromBuffer:c->output sourceOffset:c->outputOffset toBuffer:c->outputSnapshots
                            destinationOffset:(pass.controlStep*3u+phase)*c->outputBytes size:c->outputBytes];''')
change('''                    if (!blit || !c->history) return false;
                    [blit copyFromBuffer:c->history sourceOffset:0 toBuffer:c->snapshots
                        destinationOffset:(pass.controlStep*3u+2u)*c->bytes size:c->bytes];''','''                    if (!blit || !c->history || !c->output) return false;
                    [blit copyFromBuffer:c->history sourceOffset:0 toBuffer:c->snapshots
                        destinationOffset:(pass.controlStep*3u+2u)*c->bytes size:c->bytes];
                    [blit copyFromBuffer:c->output sourceOffset:c->outputOffset toBuffer:c->outputSnapshots
                        destinationOffset:(pass.controlStep*3u+2u)*c->outputBytes size:c->outputBytes];''')
change('''                std::cout << "actor_snapshot_bytes=" << actorSnapshotBytes << "\\n";''','''                std::ofstream outputFile(std::string(prefix)+".observation",std::ios::binary|std::ios::trunc);
                outputFile.write(static_cast<const char*>(outputCaptures.contents), static_cast<std::streamsize>(outputCaptures.length));
                outputFile.close(); require(outputFile.good(), "observation trace write");
                std::cout << "output_per_env=" << layout.actorObservationSize << "\\n";
                std::cout << "actor_snapshot_bytes=" << actorSnapshotBytes << "\\n";''')
Path('world_probe.cpp').write_text(s)
