from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-landing-approach-dev-20261111';B=R/'build/crow-v15-landing-approach-dev-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';V=R/'build/crow-v15-navigation-collider-observer-20261111';base=(R/'build/crow-v15-course-collisions-enabled-20261111/enabled/c_api.cpp').read_text();assert base.count('manifest.task.maximumEpisodeSteps = 3000u;')==1;base=base.replace('manifest.task.maximumEpisodeSteps = 3000u;','manifest.task.maximumEpisodeSteps = 5000u;');lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();c=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);ln=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out=[]
for name,damping in [('trace','.00005f')]:
 D=B/name;D.mkdir(exist_ok=True);old='''                        if (a.find("crow_course_perch")==std::string::npos &&
                            b.find("crow_course_perch")==std::string::npos &&
                            a.find("locomotion_ground")==std::string::npos &&
                            b.find("locomotion_ground")==std::string::npos) continue;
''';assert base.count(old)==2;s=''.join(base.rsplit(old,1));src=D/'c_api.cpp';src.write_text(s);(A/name/'body.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/reference.cpp',tofile='b/'+name+'.cpp')))
 args=shlex.split(c);args[1:1]=['-I'+str(W/'include'),'-I'+str(R/'artifacts/crow-v15-two-segment-frozen-flight-contact-map-20261111'),'-I'+str(R/'artifacts/crow-clearance-observer-20261109')];obj=D/'c_api.cpp.o'
 for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
 args[-1]=str(src);subprocess.run(args,cwd=R/'build',check=True)
 args=shlex.split(ln[len(': && '):-len(' && :')]);lib=D/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);files=['src/core/CompiledMetalMultiArticulatedProgram.cpp','src/core/CompiledMetalMultiArticulatedContactProgram.cpp','src/metal/MetalArticulatedInverseMass.mm','src/metal/MetalArticulatedABA.mm','src/metal/MetalWorld.mm','src/metal/MetalMultiArticulatedContact.mm','src/metal/MetalMultiArticulatedABA.mm','src/metal/MetalMultiArticulatedInverseMass.mm']
 for i,n in enumerate(files):
  old='CMakeFiles/metalrobo.dir/'+n+'.o';assert old in args;args[args.index(old)]=str(V/'MetalWorld.mm.o' if i==4 else W/(str(i)+'.o'))
 subprocess.run(args,cwd=R/'build',check=True);out.append({'name':name,'toe_segment_mass_kg':float(damping[:-1]),'library_sha256':h(lib),'source_sha256':h(src),'observer_sha256':h(V/'MetalWorld.mm.o')});print('built',name,flush=True)
(A/'build-hashes.json').write_text(json.dumps(out,indent=2)+'\n')
