from pathlib import Path
s=Path('/Users/numi/tracker-actor-history-20261108/probe.cpp').read_text()
def repl(a,b):
 global s
 assert s.count(a)==1,(s.count(a),a[:80]);s=s.replace(a,b)
repl('        trackerConfig.timestepSeconds = 0.02f;','        trackerConfig.timestepSeconds = 0.02f;\n        trackerConfig.nominalSensorRateHz = 10.0f;')
repl('''            struct CaptureOwner {
                MetalWorldDeviceObservationProgram tracker;''','''            id<MTLBuffer> resetTrace = [captureDevice newBufferWithLength:30u*kEnvironmentCount*sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
            require(resetTrace != nil, "reset capture allocation");
            struct CaptureOwner {
                __strong id<MTLBuffer> resets;
                MetalWorldDeviceObservationProgram tracker;''')
repl('''            } capture{trackerProgram, captures, nil, actorSnapshotBytes};''','''            } capture{resetTrace, trackerProgram, captures, nil, actorSnapshotBytes};''')
repl('''                    return copy(0u) && c->tracker.encode(c->tracker.context,pass) && copy(1u);''','''                    id<MTLBuffer> resets=(__bridge id<MTLBuffer>)pass.resetMasks;
                    id<MTLBlitCommandEncoder> resetCopy=[cmd blitCommandEncoder];
                    if (!resetCopy || !resets) return false;
                    [resetCopy copyFromBuffer:resets sourceOffset:pass.controlStep*pass.environmentCount*sizeof(std::uint32_t)
                        toBuffer:c->resets destinationOffset:pass.controlStep*pass.environmentCount*sizeof(std::uint32_t)
                        size:pass.environmentCount*sizeof(std::uint32_t)];
                    [resetCopy endEncoding];
                    return copy(0u) && c->tracker.encode(c->tracker.context,pass) && copy(1u);''')
repl('''            require(rendered.metadata.identity.z==(residentMode ? 30u : 29u),''','''            require(rendered.metadata.identity.z==(residentMode ? 6u : 5u),''')
repl('''                save(".status", result.statuses);
                std::ofstream capturesFile''','''                save(".status", result.statuses);
                std::ofstream resetFile(std::string(prefix)+".reset",std::ios::binary|std::ios::trunc);
                resetFile.write(static_cast<const char*>(resetTrace.contents), static_cast<std::streamsize>(resetTrace.length));
                resetFile.close(); require(resetFile.good(), "reset trace write");
                std::ofstream capturesFile''')
Path('/Users/numi/tracker-reset-cadence-20261108/world_probe.cpp').write_text(s)
