from pathlib import Path
root=Path('/Users/numi/numi-lab-reconciliation-20261108')
p=root/'docs/evidence/frictionless-two-dynamic-20261108'
s=(root/'docs/evidence/frictionless-active-mixed-20261108/active/probe.cpp').read_text()
def rep(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
rep('for(float angle : {0.3f,-0.41f}) for(int iterations : {2,8,32}) for(int testcase=0;testcase<3;testcase++) {','for(float angle : {-0.37f,0.29f}) for(int iterations : {2,32}) for(int active : {0,1}) for(int testcase=0;testcase<3;testcase++) {')
rep('m.defaultQ[1]=hx*std::abs(std::sin(angle))+hy*std::abs(std::cos(angle))+gap;','m.defaultQ[1]=1.05f+hx*std::abs(std::sin(angle))+hy*std::abs(std::cos(angle))+gap;')
rep('m.defaultV[1]=-1.f;','m.defaultV[1]=active ? -0.5f : 0.1f;')
needle='  metalrobo::CompiledWorld world;auto compiled=metalrobo::compileMetalWorld(m,0,world);'
insert='''  MRBodyPropertiesGPU support=b;support.articulationIndex=MR_INVALID_INDEX;support.parentBody=MR_INVALID_INDEX;support.inboundJoint=MR_INVALID_INDEX;
  support.massAndInverseMass={2.f,.5f,0,0};constexpr float sx=.5f,sy=.05f,sz=.5f;const float six=2*(sy*sy+sz*sz)/3.f,siy=2*(sx*sx+sz*sz)/3.f,siz=2*(sx*sx+sy*sy)/3.f;
  support.inertiaRow0={six,0,0,0};support.inertiaRow1={0,siy,0,0};support.inertiaRow2={0,0,siz,0};support.inverseInertiaRow0={1/six,0,0,0};support.inverseInertiaRow1={0,1/siy,0,0};support.inverseInertiaRow2={0,0,1/siz,0};
  MRShapeGPU supportShape=sh;supportShape.bodyIndex=2;supportShape.dimensions={sx,sy,sz,0};supportShape.contactRestAndBoundingRadius={.001f,0,std::sqrt(sx*sx+sy*sy+sz*sz),0};supportShape.slotGeneration=512;
  m.bodies.push_back(support);m.shapes.push_back(supportShape);if(!m.bodyNames.empty())m.bodyNames.push_back("support");if(!m.shapeNames.empty())m.shapeNames.push_back("support");m.world.bodyCount=m.bodies.size();m.world.shapeCount=m.shapes.size();
'''
rep(needle,insert+needle)
rep('  std::vector<MRBodyStateGPU> scene{ground}; std::vector<float> efforts(world.nv(),0.f);','''  MRBodyStateGPU floor{};floor.position={0,1,0,1};floor.orientation.w=1;floor.linearVelocityAndInverseMass={0,.1f,0,.5f};floor.inverseInertiaWorldRow0={1/six,0,0,0};floor.inverseInertiaWorldRow1={0,1/siy,0,0};floor.inverseInertiaWorldRow2={0,0,1/siz,0};floor.flagsAndIndices[0]=MR_MOTION_DYNAMIC;floor.flagsAndIndices[1]=MR_INVALID_INDEX;floor.flagsAndIndices[2]=2;
  std::vector<MRBodyStateGPU> scene{ground,floor}; std::vector<float> efforts(world.nv(),0.f);''')
rep('<<",\\"case\\":"<<testcase','<<",\\"active\\":"<<active<<",\\"case\\":"<<testcase')
rep('same(r.finalV,rr.finalV)&&s.firstGPUStatusCode','same(r.finalV,rr.finalV)&&same(r.finalSceneBodies,rr.finalSceneBodies)&&s.firstGPUStatusCode')
rep('  std::cout<<"],\\"solver_residual\\":"','''  if(r.finalSceneBodies.size()!=2)return 5;const auto& fs=r.finalSceneBodies[1];
  std::cout<<"],\\"support_v\\":["<<fs.linearVelocityAndInverseMass.x<<","<<fs.linearVelocityAndInverseMass.y<<","<<fs.linearVelocityAndInverseMass.z<<","<<fs.angularVelocity.x<<","<<fs.angularVelocity.y<<","<<fs.angularVelocity.z<<"],\\"support_q\\":["<<fs.position.x<<","<<fs.position.y<<","<<fs.position.z<<","<<fs.orientation.x<<","<<fs.orientation.y<<","<<fs.orientation.z<<","<<fs.orientation.w;
  std::cout<<"],\\"solver_residual\\":"''')
(p/'probe.cpp').write_text(s)
