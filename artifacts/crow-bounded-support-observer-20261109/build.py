from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd(); A=R/'artifacts/crow-bounded-support-observer-20261109'; B=R/'build/crow-bounded-support-observer-20261109'; B.mkdir(parents=True,exist_ok=True)
base=(R/'build/crow-extended-flight-20261109/c_api.cpp').read_text()
needle='''            for (size_t lane=0; lane<environmentCount && lane<published.transitions.size(); ++lane) {
                if (published.transitions[lane].termination.w != MR_TASK_TERMINATION_CONTACT) continue;'''
insert=r'''
            if (std::getenv("CROW_SUPPORT_TRACE") != nullptr) {
                for (size_t lane=0; lane<environmentCount && lane<published.contactStatuses.size(); ++lane) {
                    const auto stride = published.layout.contactDispatch.constraintStride;
                    const auto count = published.contactStatuses[lane].activeContacts;
                    const auto name=[&](uint32_t b) { return b<handle->model.bodyNames.size()
                        ? handle->model.bodyNames[b].c_str() : "invalid"; };
                    bool opened=false, first=true;
                    for (size_t i=0; i<count && lane*stride+i<published.contactEvidence.contacts.size(); ++i) {
                        const auto& c=published.contactEvidence.contacts[lane*stride+i];
                        const std::string a=name(c.bodyA), b=name(c.bodyB);
                        if (a.find("crow_course_perch")==std::string::npos &&
                            b.find("crow_course_perch")==std::string::npos) continue;
                        if (!opened) {
                            std::fprintf(stderr, "CROW_SUPPORT_SAMPLE {\"step\":%llu,\"lane\":%zu",
                                static_cast<unsigned long long>(handle->submittedControlSteps+1u), lane);
                            array("q",published.finalQ,lane*handle->world.nq(),handle->world.nq());
                            array("v",published.finalV,lane*handle->world.nv(),handle->world.nv());
                            std::fprintf(stderr,",\"contacts\":[");
                            opened=true;
                        }
                        std::fprintf(stderr,"%s{\"a\":%u,\"a_name\":\"%s\",\"b\":%u,\"b_name\":\"%s\",\"normal\":[%.9g,%.9g,%.9g],\"tangent_u\":[%.9g,%.9g,%.9g],\"tangent_v\":[%.9g,%.9g,%.9g],\"impulses\":[%.9g,%.9g,%.9g],\"separation\":%.9g}",
                            first?"":",",c.bodyA,a.c_str(),c.bodyB,b.c_str(),
                            c.normal.x,c.normal.y,c.normal.z,c.tangent.x,c.tangent.y,c.tangent.z,
                            c.normal.y*c.tangent.z-c.normal.z*c.tangent.y,c.normal.z*c.tangent.x-c.normal.x*c.tangent.z,c.normal.x*c.tangent.y-c.normal.y*c.tangent.x,c.impulses.x,c.impulses.y,c.impulses.z,c.pointAndSeparation.w);
                        first=false;
                    }
                    if (opened) std::fprintf(stderr,"]}\n");
                }
            }
'''
assert base.count(needle)==1
s=base.replace(needle,insert+needle);
marker='''                    bool opened=false, first=true;'''
bounded=r'''                    // Bound repetitive stable-support evidence, while keeping
                    // all contact interruptions and the final 25 steps.
                    static std::vector<unsigned> consecutive;
                    static std::vector<unsigned long long> lastStep;
                    if (consecutive.size()!=environmentCount) {
                        consecutive.assign(environmentCount,0u);
                        lastStep.assign(environmentCount,0ull);
                    }
                    bool hasFoot=false, hasOther=false;
                    double upward=0.0;
                    for (size_t i=0;i<count && lane*stride+i<published.contactEvidence.contacts.size();++i) {
                        const auto& c=published.contactEvidence.contacts[lane*stride+i];
                        const std::string a=name(c.bodyA), b=name(c.bodyB);
                        if (a.find("crow_course_perch")==std::string::npos &&
                            b.find("crow_course_perch")==std::string::npos) continue;
                        const bool footA=a.find("foot")!=std::string::npos;
                        const bool footB=b.find("foot")!=std::string::npos;
                        if (footA || footB) {
                            hasFoot=true;
                            upward+=(footA ? -1.0 : 1.0)*c.normal.z*c.impulses.x;
                        } else hasOther=true;
                    }
                    const auto step=static_cast<unsigned long long>(handle->submittedControlSteps+1u);
                    const auto vb=lane*handle->world.nv();
                    const bool velocityOK=vb+2u<published.finalV.size() &&
                        std::hypot(published.finalV[vb],published.finalV[vb+1u])<=0.12f &&
                        std::abs(published.finalV[vb+2u])<=0.12f;
                    if (hasFoot && !hasOther && upward>0.0 && velocityOK)
                        consecutive[lane]=(lastStep[lane]+1ull==step) ? consecutive[lane]+1u : 1u;
                    else consecutive[lane]=0u;
                    lastStep[lane]=step;
                    if (consecutive[lane]>50u && step<2976u && step%100u!=0u) continue;
'''
assert s.count(marker)==1
s=s.replace(marker,bounded+marker)
src=B/'c_api.cpp'; src.write_text(s)
(A/'observer.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/extended-c_api.cpp',tofile='b/extended-c_api.cpp')))
lines=subprocess.check_output(['ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines()
compile_line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x)
args=shlex.split(compile_line);obj=B/'c_api.o'
for k,v in [('-o',str(obj)),('-MF',str(B/'c_api.d')),('-MT',str(obj))]: args[args.index(k)+1]=v
args[-1]=str(src);subprocess.run(args,cwd=R/'build',check=True)
link_line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x)
args=shlex.split(link_line[len(': && '):-len(' && :')]); lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib)
args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj)
subprocess.run(args,cwd=R/'build',check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'build.json').write_text(json.dumps({'source_sha256':h(src),'library_sha256':h(lib),'extended_source_sha256':h(R/'build/crow-extended-flight-20261109/c_api.cpp')},indent=2)+'\n')
