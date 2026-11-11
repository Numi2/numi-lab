#define main toe_probe_main
#include "../crow-v15-two-segment-toe-prototype-20261111/probe.cpp"
#undef main
#include "../crow-v15-landing-com-diagnostic-20261111/wing_model.hpp"
#include <iomanip>
static RobotPack hipRollModel(){
 auto p=toeModel();crowResearchWingInertia(p);auto &m=p.mechanics;p.id="crow_v15_experimental_hip_roll";p.sourceRevision+="+hip_roll_experiment";
 auto oldBodies=m.bodies;auto oldNames=m.bodyNames;auto oldJoints=m.joints;
 auto map=[](unsigned i){return i+(i>=8u)+(i>=11u);};
 m.bodies.clear();m.bodyNames.clear();
 const unsigned newJointBase=m.joints.size(),newQ=m.defaultQ.size(),newV=m.defaultV.size();
 for(unsigned i=0;i<oldBodies.size();++i){
  if(i==8u||i==11u){unsigned side=i==8u?0u:1u;auto proxy=oldBodies[side==0?1:2];proxy.parentBody=0;proxy.inboundJoint=newJointBase+side;proxy.centerOfMass={0,0,0,0};m.bodies.push_back(proxy);m.bodyNames.push_back(side==0?"crow_left_hip_roll_link":"crow_right_hip_roll_link");}
  auto b=oldBodies[i];if(b.parentBody!=MR_INVALID_INDEX)b.parentBody=map(b.parentBody);if(i==8u||i==11u)b.parentBody=map(i)-1u;m.bodies.push_back(b);m.bodyNames.push_back(oldNames[i]);
 }
 for(unsigned i=0;i<m.joints.size();++i){auto &j=m.joints[i];j.parentBody=map(j.parentBody);j.childBody=map(j.childBody);if(i==7u||i==10u){j.parentBody=j.childBody-1u;j.parentAnchor={0,0,0,0};}}
 for(auto& s:m.shapes)s.bodyIndex=map(s.bodyIndex);
 // Preserve precisely the torso/thigh shape exclusions that came from direct
 // adjacency before the added hip-roll proxy links changed that topology.
 for(unsigned i=0;i<m.shapes.size();++i)for(unsigned j=i+1;j<m.shapes.size();++j){const auto a=m.shapes[i].bodyIndex,b=m.shapes[j].bodyIndex;if((a==0u && (b==map(8u)||b==map(11u))) || (b==0u && (a==map(8u)||a==map(11u))))m.collisionExclusions.push_back({i,j});}
 std::sort(m.collisionExclusions.begin(),m.collisionExclusions.end(),[](const auto&a,const auto&b){return a.colliderA<b.colliderA || (a.colliderA==b.colliderA && a.colliderB<b.colliderB);});m.collisionExclusions.erase(std::unique(m.collisionExclusions.begin(),m.collisionExclusions.end(),[](const auto&a,const auto&b){return a.colliderA==b.colliderA && a.colliderB==b.colliderB;}),m.collisionExclusions.end());
 for(unsigned side=0;side<2;++side){unsigned oldJ=side==0?7u:10u;auto j=oldJoints[oldJ];j.childBody=map(j.childBody)-1u;j.parentBody=0;j.qOffset=newQ+side;j.vOffset=newV+side;j.axis0={1,0,0,0};j.childAnchor={0,0,0,0};m.joints.push_back(j);m.jointNames.push_back(side==0?"crow_left_hip_roll":"crow_right_hip_roll");auto d=m.dofs.at(oldJoints[oldJ].vOffset);d.jointIndex=newJointBase+side;d.qIndex=newQ+side;d.vIndex=newV+side;d.limits={-.35f,.35f,8.f,.02f};d.drive={.08f,.003f,.00005f,0};m.dofs.push_back(d);m.dofNames.push_back(side==0?"crow_left_hip_roll":"crow_right_hip_roll");m.defaultQ.push_back(0);m.defaultV.push_back(0);if(!m.actuatorProfiles.empty())m.actuatorProfiles.emplace_back();}
 auto&a=m.articulations.at(0);a.bodyCount=m.world.bodyCount=m.bodies.size();a.jointCount=m.world.jointCount=m.joints.size();a.nq=m.world.nq=m.defaultQ.size();a.nv=m.world.nv=m.defaultV.size();
 for(auto& r:p.roles){if(r.id=="whole_body")r.members=m.bodyNames;if(r.id=="all_joints")r.members=m.jointNames;if(r.id=="all_dofs")r.members=m.dofNames;}
 std::string why;need(m.valid(&why),"model invalid: "+why);return p;
}
#ifndef CROW_HIP_ROLL_LIBRARY
int main(int argc,char** argv){try{auto old=toeModel();crowResearchWingInertia(old);auto model=hipRollModel();auto &a=old.mechanics;auto &b=model.mechanics;std::cout<<std::setprecision(12)<<"model bodies "<<b.bodies.size()<<" nq "<<b.defaultQ.size()<<" nv "<<b.defaultV.size()<<"\n";
 double maxPose=0,maxEnergy=0,maxMomentum=0;for(unsigned n=0;n<16;++n){std::vector<double> aq(a.defaultQ.begin(),a.defaultQ.end()),av(a.defaultV.size()),bq(b.defaultQ.begin(),b.defaultQ.end()),bv(b.defaultV.size());aq[0]=.01*n;aq[5]=std::sin(.02*n);aq[6]=std::cos(.02*n);for(size_t k=0;k<av.size();++k)av[k]=.1*std::sin(.23*n+.77*k);std::copy(aq.begin(),aq.end(),bq.begin());std::copy(av.begin(),av.end(),bv.begin());ArticulatedInvariants ai,bi;need(computeArticulatedInvariants(a,0,aq,av,ai).succeeded(),"old invariants");need(computeArticulatedInvariants(b,0,bq,bv,bi).succeeded(),"new invariants");maxEnergy=std::max(maxEnergy,std::abs(ai.kineticEnergy-bi.kineticEnergy));for(int k=0;k<3;++k)maxMomentum=std::max(maxMomentum,std::abs(ai.linearMomentum[k]-bi.linearMomentum[k]));std::vector<ArticulatedBodyKinematics> ak(a.bodies.size()),bk(b.bodies.size());need(computeArticulatedBodyKinematics(a,0,aq,av,ak).succeeded(),"old kinematics");need(computeArticulatedBodyKinematics(b,0,bq,bv,bk).succeeded(),"new kinematics");for(unsigned i=0;i<ak.size();++i){unsigned j=i+(i>=8u)+(i>=11u);for(int k=0;k<3;++k)maxPose=std::max(maxPose,std::abs(ak[i].centerOfMassPosition[k]-bk[j].centerOfMassPosition[k]));}}
 std::cout<<"max_position_difference_m "<<maxPose<<" max_kinetic_energy_difference_j "<<maxEnergy<<" max_linear_momentum_difference_kg_m_s "<<maxMomentum<<"\n";
 if(argc>1 && std::string(argv[1])=="--gpu"){
  const unsigned n=12,nq=b.defaultQ.size(),nv=b.defaultV.size();std::vector<float> q,v,e;
  for(unsigned k=0;k<n;k++){q.insert(q.end(),b.defaultQ.begin(),b.defaultQ.end());v.insert(v.end(),nv,0);e.insert(e.end(),nv,0);q[k*nq+nq-2]=.02f*(int(k%5)-2);q[k*nq+nq-1]=.015f*(int(k%7)-3);v[k*nv+nv-2]=.04f*(int(k%3)-1);v[k*nv+nv-1]=.03f*(int(k%5)-2);e[k*nv+nv-2]=.0001f*(int(k%3)-1);e[k*nv+nv-1]=.0001f*(int(k%5)-2);}
  MetalArticulatedABAInput in{.articulationIndex=0u,.environmentCount=n,.q=q,.v=v,.effort=e,.applyBodyDamping=false};MetalArticulatedABAConfig gpuConfig;gpuConfig.metallibPath="/Users/n/Developer/numi-crow-rl-20261107/build/crow-v15-wide-contact-isolation-20261111/MetalRobo.metallib";MetalArticulatedABAContext ctx(gpuConfig);MetalArticulatedABAResult out,again;auto s=ctx.run(b,in,out);need(s.succeeded()&&s.published&&s.failedEnvironmentCount==0,"GPU ABA failed "+s.message+" first_status="+std::to_string(s.firstGPUStatusCode)+" first_environment="+std::to_string(s.firstFailingEnvironment)+" failed_count="+std::to_string(s.failedEnvironmentCount));s=ctx.run(b,in,again);need(s.succeeded()&&again.acceleration==out.acceleration&&again.nextQ==out.nextQ&&again.nextV==out.nextV,"GPU repeat mismatch");double err=0;for(unsigned k=0;k<n;k++){std::vector<double> qq(q.begin()+k*nq,q.begin()+(k+1)*nq),vv(v.begin()+k*nv,v.begin()+(k+1)*nv),ee(e.begin()+k*nv,e.begin()+(k+1)*nv),acc(nv);ArticulatedDynamicsConfig cfg;cfg.timestep=b.world.gravityAndTimestep.w;cfg.applyBodyDamping=false;auto x=computeArticulatedForwardDynamics(b,0,qq,vv,ee,{},acc,cfg);need(x.succeeded(),"CPU ABA failed");for(unsigned j=0;j<nv;j++)err=std::max(err,std::abs(out.acceleration[k*nv+j]-acc[j])/(1+std::abs(acc[j])));}std::cout<<"gpu_scaled_acceleration_error "<<err<<" lanes "<<n<<" device "<<s.deviceName<<"\n";need(err<5e-5,"GPU/CPU ABA mismatch");
 }
 return 0;}catch(const std::exception&e){std::cerr<<"failure "<<e.what()<<"\n";return 1;}}

#endif
