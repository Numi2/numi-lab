from pathlib import Path
import difflib,shlex,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-corrected-wing-action-observer-20261110';B=R/'build/crow-v15-corrected-wing-action-observer-20261110';base=(R/'build/crow-v15-grip-search-20261109/complete-observer/MetalWorld.mm').read_text();s=base
needle='bool privatePersistentBuffer(const std::size_t index) {\n    switch (index) {';pos=s.index(needle);end=s.index('\n}',pos);part=s[pos:end]
for key in ('    case kTaskState:\n','    case kTaskActionHistory:\n'):
 assert part.count(key)==1;part=part.replace(key,'')
s=s[:pos]+part+s[end:]
s=s.replace('    bool nativeTask = false;\n    bool captureContactEvidence', '    bool nativeTask = false;\n    std::size_t observerActionCount = 0, observerDelayCount = 0;\n    bool captureContactEvidence', 1)
s=s.replace('            pending->nativeTask = nativeTask;', '            pending->nativeTask = nativeTask;\n            if (nativeTask) { pending->observerActionCount=config.taskProgram.layout().actionCount; pending->observerDelayCount=config.taskProgram.layout().delayStateCount; }',1)
needle='            const auto end = std::chrono::steady_clock::now();'
insert='''            if (std::getenv("CROW_ACTION_OBSERVER") != nullptr) {
                id<MTLBuffer> sb = pending->context->buffers[kTaskState];
                id<MTLBuffer> hb = pending->context->buffers[kTaskActionHistory];
                const auto* st = static_cast<const MRTaskStateGPU*>(sb.contents);
                const auto* ah = static_cast<const float*>(hb.contents);
                const auto actionCount = pending->observerActionCount, delayCount = pending->observerDelayCount;
                const auto lanes = pending->diagnostics.layout.dispatch.environmentCount;
                if (st && ah && delayCount && actionCount) for (std::uint32_t lane=0;lane<lanes;++lane) {
                    if (st[lane].episode.x % 100u != 0u) continue;
                    std::fprintf(stderr,"CROW_ACTION_SAMPLE {\\\"episode_step\\\":%u,\\\"lane\\\":%u,\\\"filtered\\\":[",st[lane].episode.x,lane);
                    const auto start=(lane*delayCount+delayCount-1u)*actionCount;
                    for (std::uint32_t j=0;j<actionCount;++j) std::fprintf(stderr,"%s%.9g",j?",":"",ah[start+j]);
                    std::fprintf(stderr,"]}\\n");
                }
            }
''';assert s.count(needle)==1;s=s.replace(needle,insert+needle);src=B/'MetalWorld.mm';src.write_text(s);(A/'observer.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/complete-observer.mm',tofile='b/action-observer.mm')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);args=shlex.split(line);obj=B/'MetalWorld.mm.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'src/metal'));args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(R/'build/crow-v15-wing-inertia-prototype-20261110/c_api.cpp.o');args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(obj);subprocess.run(args,cwd=R/'build',check=True);print(hashlib.sha256((B/'libmetalrobo.dylib').read_bytes()).hexdigest())
