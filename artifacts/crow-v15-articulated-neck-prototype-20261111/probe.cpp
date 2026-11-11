#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"

static RobotPack neckModel() {
 auto p=hipRollModel(); auto& m=p.mechanics;
 p.id="crow_v15_experimental_two_pitch_neck"; m.name=p.id; p.sourceRevision+="+experimental-two-pitch-neck";
 const auto it=std::find(m.bodyNames.begin(),m.bodyNames.end(),"crow_head"); need(it!=m.bodyNames.end(),"head missing");
 const unsigned h=unsigned(it-m.bodyNames.begin()), ji=m.bodies[h].inboundJoint;
 need(m.jointNames[ji]=="crow_neck_pitch","pitch missing");
 auto original=m.bodies; auto names=m.bodyNames; auto originalJoint=m.joints[ji];
 const unsigned newJ=unsigned(m.joints.size()),newQ=unsigned(m.defaultQ.size()),newV=unsigned(m.defaultV.size());
 auto map=[h](unsigned i){return i==MR_INVALID_INDEX?i:i+(i>=h);};
 auto scale=[](MRBodyPropertiesGPU& b, float f){
  b.massAndInverseMass.x*=f;b.massAndInverseMass.y=1.f/b.massAndInverseMass.x;
  b.inertiaRow0.x*=f;b.inertiaRow1.y*=f;b.inertiaRow2.z*=f;
  b.inverseInertiaRow0.x=1.f/b.inertiaRow0.x;b.inverseInertiaRow1.y=1.f/b.inertiaRow1.y;b.inverseInertiaRow2.z=1.f/b.inertiaRow2.z;
 };
 m.bodies.clear();m.bodyNames.clear();
 for(unsigned i=0;i<original.size();++i){
  if(i==h){auto proxy=original[i]; scale(proxy,3.f/35.f);proxy.parentBody=map(original[i].parentBody);proxy.inboundJoint=newJ;m.bodies.push_back(proxy);m.bodyNames.push_back("crow_neck_proximal_pitch_link");}
  auto b=original[i];b.parentBody=map(b.parentBody);if(i==h){scale(b,32.f/35.f);b.parentBody=h;}m.bodies.push_back(b);m.bodyNames.push_back(names[i]);
 }
 for(auto& j:m.joints){j.parentBody=map(j.parentBody);j.childBody=map(j.childBody);}
 for(auto& s:m.shapes)s.bodyIndex=map(s.bodyIndex);
 auto& distal=m.joints[ji];distal.parentBody=h;distal.parentAnchor.x=originalJoint.childAnchor.x+.025f;distal.parentAnchor.y=originalJoint.childAnchor.y;distal.parentAnchor.z=originalJoint.childAnchor.z;distal.childAnchor=distal.parentAnchor;
 auto proximal=originalJoint;proximal.parentBody=map(originalJoint.parentBody);proximal.childBody=h;proximal.qOffset=newQ;proximal.vOffset=newV;m.joints.push_back(proximal);m.jointNames.push_back("crow_neck_proximal_pitch");
 auto d=m.dofs.at(originalJoint.vOffset);d.jointIndex=newJ;d.qIndex=newQ;d.vIndex=newV;d.limits={-.5f,.5f,8.f,.025f};m.dofs.push_back(d);m.dofNames.push_back("crow_neck_proximal_pitch");m.defaultQ.push_back(0);m.defaultV.push_back(0);if(!m.actuatorProfiles.empty())m.actuatorProfiles.emplace_back();
 auto& a=m.articulations.at(0);a.bodyCount=m.world.bodyCount=m.bodies.size();a.jointCount=m.world.jointCount=m.joints.size();a.nq=m.world.nq=m.defaultQ.size();a.nv=m.world.nv=m.defaultV.size();
 for(auto& role:p.roles){if(role.id=="whole_body")role.members=m.bodyNames;if(role.id=="all_joints")role.members=m.jointNames;if(role.id=="all_dofs")role.members=m.dofNames;}
 std::string why;need(m.valid(&why),"neck model invalid: "+why);return p;
}
int main(){try{
 auto old=hipRollModel();auto neo=neckModel();auto& a=old.mechanics;auto& b=neo.mechanics;
 auto oldHead=unsigned(std::find(a.bodyNames.begin(),a.bodyNames.end(),"crow_head")-a.bodyNames.begin());
 auto head=unsigned(std::find(b.bodyNames.begin(),b.bodyNames.end(),"crow_head")-b.bodyNames.begin());
 auto j=a.joints[a.bodies[oldHead].inboundJoint]; const unsigned pq=j.qOffset,pv=j.vOffset;
 double maxPose=0,maxInvariant=0;
 for(unsigned n=0;n<16;n++){
  std::vector<double> aq(a.defaultQ.begin(),a.defaultQ.end()),av(a.defaultV.size()),bq(b.defaultQ.begin(),b.defaultQ.end()),bv(b.defaultV.size());
  aq[0]=.01*n;aq[5]=std::sin(.02*n);aq[6]=std::cos(.02*n);
  for(size_t k=7;k<aq.size();k++)aq[k]=.05*std::sin(.23*n+.31*k);
  for(size_t k=0;k<av.size();k++)av[k]=.1*std::sin(.23*n+.77*k);aq[pq]=0;av[pv]=0;
  std::copy(aq.begin(),aq.end(),bq.begin());std::copy(av.begin(),av.end(),bv.begin());
  ArticulatedInvariants ai,bi;need(computeArticulatedInvariants(a,0,aq,av,ai).succeeded(),"old invariants");need(computeArticulatedInvariants(b,0,bq,bv,bi).succeeded(),"neck invariants");
  auto check=[&](double x,double y){double e=std::abs(x-y);maxInvariant=std::max(maxInvariant,e);need(e<=2e-7*(1+std::abs(x)),"invariant mismatch");};
  check(ai.kineticEnergy,bi.kineticEnergy);check(ai.potentialEnergy,bi.potentialEnergy);for(int k=0;k<3;k++){check(ai.linearMomentum[k],bi.linearMomentum[k]);check(ai.angularMomentum[k],bi.angularMomentum[k]);}
  std::vector<ArticulatedBodyKinematics> ak(a.bodies.size()),bk(b.bodies.size());need(computeArticulatedBodyKinematics(a,0,aq,av,ak).succeeded(),"old kinematics");need(computeArticulatedBodyKinematics(b,0,bq,bv,bk).succeeded(),"neck kinematics");
  for(unsigned i=0;i<ak.size();i++){unsigned t=i+(i>=oldHead);for(int k=0;k<3;k++){double diff=std::abs(ak[i].centerOfMassPosition[k]-bk[t].centerOfMassPosition[k]);maxPose=std::max(maxPose,diff);need(diff<=2e-7,"pose mismatch");}}
 }
 std::vector<double> q(b.defaultQ.begin(),b.defaultQ.end()),v(b.defaultV.size());std::vector<ArticulatedBodyKinematics> zero(b.bodies.size()),posed(b.bodies.size());
 need(computeArticulatedBodyKinematics(b,0,q,v,zero).succeeded(),"neutral FK");q.back()=.4;q[pq]=-.4;need(computeArticulatedBodyKinematics(b,0,q,v,posed).succeeded(),"reach FK");
 double dx=posed[head].centerOfMassPosition[0]-zero[head].centerOfMassPosition[0],dz=posed[head].centerOfMassPosition[2]-zero[head].centerOfMassPosition[2],dist=std::hypot(dx,dz),rot=0;for(int k=0;k<4;k++)rot=std::max(rot,std::abs(posed[head].orientation[k]-zero[head].orientation[k]));
 std::cout<<std::setprecision(12)<<"locked_max_pose_m "<<maxPose<<" max_invariant_abs "<<maxInvariant<<" reach_m "<<dist<<" reach_dx_m "<<dx<<" reach_dz_m "<<dz<<" orientation_error "<<rot<<"\n";need(dist>=.001&&rot<1e-7,"independent reach failed");
 const unsigned n=12,nq=b.defaultQ.size(),nv=b.defaultV.size();std::vector<float> gq,gv,e;
 for(unsigned k=0;k<n;k++){gq.insert(gq.end(),b.defaultQ.begin(),b.defaultQ.end());gv.insert(gv.end(),nv,0);e.insert(e.end(),nv,0);gq[k*nq+nq-1]=.02f*(int(k%5)-2);gq[k*nq+pq]=-.015f*(int(k%5)-2);gv[k*nv+nv-1]=.04f*(int(k%3)-1);e[k*nv+nv-1]=.0001f*(int(k%3)-1);}
 MetalArticulatedABAInput in{.articulationIndex=0u,.environmentCount=n,.q=gq,.v=gv,.effort=e,.applyBodyDamping=false};MetalArticulatedABAConfig gc;gc.metallibPath="/Users/n/Developer/numi-crow-rl-20261107/build/crow-v15-wide-contact-isolation-20261111/MetalRobo.metallib";MetalArticulatedABAContext ctx(gc);MetalArticulatedABAResult out,again;auto s=ctx.run(b,in,out);need(s.succeeded()&&s.published&&s.failedEnvironmentCount==0,"GPU ABA failed "+s.message+" status="+std::to_string(s.firstGPUStatusCode));s=ctx.run(b,in,again);need(s.succeeded()&&again.acceleration==out.acceleration&&again.nextQ==out.nextQ&&again.nextV==out.nextV,"GPU repeat mismatch");
 double err=0;for(unsigned k=0;k<n;k++){std::vector<double> qq(gq.begin()+k*nq,gq.begin()+(k+1)*nq),vv(gv.begin()+k*nv,gv.begin()+(k+1)*nv),ee(e.begin()+k*nv,e.begin()+(k+1)*nv),acc(nv);ArticulatedDynamicsConfig cfg;cfg.timestep=b.world.gravityAndTimestep.w;cfg.applyBodyDamping=false;auto x=computeArticulatedForwardDynamics(b,0,qq,vv,ee,{},acc,cfg);need(x.succeeded(),"CPU ABA failed");for(unsigned z=0;z<nv;z++)err=std::max(err,std::abs(out.acceleration[k*nv+z]-acc[z])/(1+std::abs(acc[z])));}
 std::cout<<"gpu_scaled_acceleration_error "<<err<<" lanes "<<n<<" device "<<s.deviceName<<"\n";need(err<5e-5,"GPU/CPU ABA mismatch");
 return 0;
 }catch(const std::exception& e){std::cerr<<"failure "<<e.what()<<"\n";return 1;}}
