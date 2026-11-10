from pathlib import Path
import difflib,shlex,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-rigid-toe-geometry-20261110';B=R/'build/crow-v15-rigid-toe-geometry-20261110';base=(R/'build/crow-v15-original-free-flight-20261110/c_api.cpp').read_text();needle='''static MRTaskRolloutHandle* createBirdFlowAmericanCrowRun(''';assert base.count(needle)==1
model='''namespace {
static metalrobo::RobotPack rigidCrowToeGeometry() {
  auto p=metalrobo::makeAmericanCrowActiveHeadRobotPack(); auto& m=p.mechanics;
  p.id="american_crow_rigid_toe_geometry_control";m.name=p.id;p.sourceRevision+="+rigid-toe-geometry-control";
  constexpr float hx=.018f,hy=.0025f,hz=.0092f;
  for (int side=0;side<2;++side) {
    const std::string footName=side==0?"crow_left_foot":"crow_right_foot";
    auto it=std::find(m.bodyNames.begin(),m.bodyNames.end(),footName);if(it==m.bodyNames.end())throw std::runtime_error("foot missing");const unsigned foot=static_cast<unsigned>(it-m.bodyNames.begin());
    for(unsigned i=0;i<m.shapeNames.size();++i) if(m.shapeNames[i]==(side==0?"crow_left_foot/sole":"crow_right_foot/sole"))m.shapes[i].dimensions.x=.010f;
    const float dx[4]={.020666667f,.020666667f,.020666667f,-.062f},dy[4]={-.012f,0,.012f,0};
    for(int t=0;t<4;++t) {
      MRShapeGPU shape=m.shapes.front();shape.bodyIndex=foot;shape.localPosition={dx[t],dy[t],0,1};shape.localRotation={0,0,0,1};shape.dimensions={hx,hy,hz,0};shape.slotGeneration=m.shapes.size()+1;shape.contactRestAndBoundingRadius.z=std::sqrt(hx*hx+hy*hy+hz*hz);m.shapes.push_back(shape);
      m.shapeNames.push_back(std::string("crow_")+(side==0?"left_":"right_")+(t==3?"rear_toe":"front_toe_"+std::to_string(t+1))+"/collision");
    }
  }
  m.world.shapeCount=m.shapes.size();std::string why;if(!m.valid(&why))throw std::runtime_error("invalid rigid control "+why);return p;
}
}

''';s=base.replace(needle,model+needle);old='        manifest.robot = std::move(*robot);\n        manifest.scene = cueConfig';assert s.count(old)==1;s=s.replace(old,'        manifest.robot = activeNavigation ? rigidCrowToeGeometry() : std::move(*robot);\n        manifest.scene = cueConfig');src=B/'c_api.cpp';src.write_text(s);(A/'rigid-geometry.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/original-body.cpp',tofile='b/rigid-geometry.cpp')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);obj=B/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(R/'build/crow-v15-grip-search-20261109/complete-observer/MetalWorld.mm.o');subprocess.run(args,cwd=R/'build',check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
