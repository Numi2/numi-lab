from pathlib import Path
s=Path('/Users/numi/frictionless-active-normal-20261108/probe.cpp').read_text()
def rep(a,b):
 global s
 assert s.count(a)==1,(a[:80],s.count(a));s=s.replace(a,b)
rep('for(float angle : {0.3f,-0.41f}) for(int iterations : {2,8,32}) for(int testcase=0;testcase<3;testcase++) {','for(int active : {0,1}) for(int testcase=0;testcase<3;testcase++) {\n  constexpr float angle=0.3f; constexpr int iterations=2;')
rep('''  m.defaultV[0]=float(testcase-1);
  m.defaultV[1]=-1.f;
  metalrobo::CompiledWorld world;''','''  m.defaultV[0]=0.f; m.defaultV[1]=0.f;
  const float cubeY=m.defaultQ[1];
  MRBodyPropertiesGPU sceneProps=b; sceneProps.articulationIndex=MR_INVALID_INDEX;
  sceneProps.parentBody=MR_INVALID_INDEX; sceneProps.inboundJoint=MR_INVALID_INDEX;
  MRShapeGPU sceneShape=sh;sceneShape.bodyIndex=2u;
  m.bodies.push_back(sceneProps); m.shapes.push_back(sceneShape);
  if (!m.bodyNames.empty()) m.bodyNames.push_back("scene_box");
  if (!m.shapeNames.empty()) m.shapeNames.push_back("scene_box_shape");
  m.world.bodyCount=static_cast<mr_u32>(m.bodies.size());m.world.shapeCount=static_cast<mr_u32>(m.shapes.size());
  m.bodies[0].motionType=MR_MOTION_KINEMATIC;
  m.defaultQ[0]=10.f;m.defaultQ[1]=10.f;m.defaultQ[5]=0.f;m.defaultQ[6]=1.f;
  metalrobo::CompiledWorld world;''')
rep('''  MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;
  std::vector<MRBodyStateGPU> scene{ground};''','''  MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_KINEMATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;ground.linearVelocityAndInverseMass.x=0.4f;
  MRBodyStateGPU cube{};cube.position={0,cubeY,0,1};cube.orientation={0,0,std::sin(angle/2),std::cos(angle/2)};
  cube.flagsAndIndices[0]=MR_MOTION_DYNAMIC;cube.flagsAndIndices[1]=MR_INVALID_INDEX;cube.flagsAndIndices[2]=2u;
  cube.linearVelocityAndInverseMass.x=float(testcase-1);cube.linearVelocityAndInverseMass.y=active?-1.f:0.f;
  std::vector<MRBodyStateGPU> scene{ground,cube};''')
rep('''same(r.finalQ,rr.finalQ)&&same(r.finalV,rr.finalV)&&s.firstGPUStatusCode==ss.firstGPUStatusCode''','''same(r.finalQ,rr.finalQ)&&same(r.finalV,rr.finalV)&&same(r.finalSceneBodies,rr.finalSceneBodies)&&s.firstGPUStatusCode==ss.firstGPUStatusCode''')
rep('''<<",\\"case\\":"<<testcase<<",\\"initial_y\\":"<<m.defaultQ[1]''','''<<",\\"active\\":"<<active<<",\\"case\\":"<<testcase<<",\\"initial_y\\":"<<cubeY''')
rep('''  std::cout<<"],\\"solver_residual\\":"''','''  if(r.finalSceneBodies.size()!=2) return 5;
  const auto &fg=r.finalSceneBodies[0]; const auto &fc=r.finalSceneBodies[1];
  std::cout<<"],\\"ground_velocity\\":["<<fg.linearVelocityAndInverseMass.x<<","<<fg.linearVelocityAndInverseMass.y<<","<<fg.linearVelocityAndInverseMass.z<<"],\\"cube_velocity\\":["<<fc.linearVelocityAndInverseMass.x<<","<<fc.linearVelocityAndInverseMass.y<<","<<fc.linearVelocityAndInverseMass.z<<","<<fc.angularVelocity.x<<","<<fc.angularVelocity.y<<","<<fc.angularVelocity.z<<"],\\"solver_residual\\":"''')
Path('/Users/numi/frictionless-scene-body-20261108/probe.cpp').write_text(s)
