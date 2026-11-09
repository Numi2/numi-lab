from pathlib import Path
import difflib,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-toe-support-20261109';B=R/'build/crow-v15-toe-support-20261109'
needle='''        if (feeding && !cueConfig) {
            handle->resourceRuntime='''
insert='''        if (std::getenv("CROW_CONTRACT_DIAGNOSTIC")) {
            auto signature=[](const auto& span) {
                const auto* p=reinterpret_cast<const unsigned char*>(span.data());
                std::uint64_t h=1469598103934665603ull;
                for (std::size_t i=0;i<span.size_bytes();++i) {h^=p[i];h*=1099511628211ull;}
                return h;
            };
            const auto& p=handle->taskProgram;
            const auto& l=p.layout();
            std::fprintf(stderr,"CROW_CONTRACT layout %u %u %u %u %u %u %u %u %u %u %u %u %u\\n",l.actionCount,l.actorFrameSize,l.actorHistoryLength,l.actorObservationSize,l.criticFrameSize,l.criticHistoryLength,l.criticObservationSize,l.contactMetricCount,l.biasCount,l.delayStateCount,l.motionFeatureCount,l.interactionFrameCount,l.interactionContactCount);
            auto dump=[&](const char* name,const auto& value){std::fprintf(stderr,"CROW_CONTRACT %s %zu %llu\\n",name,value.size(),static_cast<unsigned long long>(signature(value)));};
            dump("actions",p.actionBindings());dump("actor",p.actorOperators());dump("critic",p.criticOperators());dump("contacts",p.contactGroups());dump("members",p.contactMembers());dump("radii",p.contactMemberRadii());dump("contact_prefix",p.contactGroups().first(3));dump("member_prefix",p.contactMembers().first(10));dump("radii_prefix",p.contactMemberRadii().first(10));dump("bias",p.biasSpecs());dump("terrain",p.terrainSampleOffsets());dump("motion",p.motionBodies());
            const auto& q=handle->run.model().defaultQ;
            dump("first_q",std::span<const float>(q.data(),std::min(q.size(),std::size_t(23))));
        }
        if (feeding && !cueConfig) {
            handle->resourceRuntime='''
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines()
for name,source in [('support',B/'c_api.cpp'),('branch',R/'build/crow-v15-branch-perch-20261109/c_api.cpp')]:
 C=B/('contract-debug-'+name);C.mkdir(exist_ok=True);s=source.read_text();assert s.count(needle)==1;s=s.replace(needle,insert);src=C/'c_api.cpp';src.write_text(s)
 (A/('contract-debug-'+name+'.patch')).write_text(''.join(difflib.unified_diff(source.read_text().splitlines(True),s.splitlines(True),fromfile='a/'+name+'.cpp',tofile='b/'+name+'-debug.cpp')))
 line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);obj=C/'c_api.cpp.o'
 for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
 args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
 line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(C/'libmetalrobo.dylib');args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(R/'build/crow-v15-grip-search-20261109/complete-observer/MetalWorld.mm.o');subprocess.run(args,cwd=R/'build',check=True)
 print('built',name,flush=True)
