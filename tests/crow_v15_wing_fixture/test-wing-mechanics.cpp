#include "v15_fixture.hpp"
#include <metalrobo/CrowArticulatedWing.hpp>

int main() {
 using namespace metalrobo;
 auto original=crowHipRollModel();
 const auto before=original.mechanics;
 auto candidate=makeCrowArticulatedWingMechanicsFixture(original);
 const auto& m=candidate.mechanics;
 auto total=[](const EngineModel& x){double sum=0;for(const auto& b:x.bodies)sum+=b.massAndInverseMass.x;return sum;};
 crowToeNeed(m.bodies.size()==41 && m.joints.size()==before.joints.size()+6,"unexpected body/joint count");
 crowToeNeed(m.defaultQ.size()==before.defaultQ.size()+6 && m.defaultV.size()==before.defaultV.size()+6,"unexpected state count");
 crowToeNeed(!candidate.flappingWings && original.flappingWings.has_value(),"aerodynamic guard failed");
 crowToeNeed(original.mechanics.bodyNames==before.bodyNames,"input was modified");
 crowToeNeed(std::abs(total(m)-total(before))<1e-7,"mass not conserved");
 for(unsigned side=0;side<2;++side){
  const auto upper=5u+side;const auto first=35u+3u*side;
  const double mass=m.bodies[upper].massAndInverseMass.x+m.bodies[first].massAndInverseMass.x+m.bodies[first+1].massAndInverseMass.x+m.bodies[first+2].massAndInverseMass.x;
  crowToeNeed(std::abs(mass-before.bodies[upper].massAndInverseMass.x)<1e-8,"wing mass changed");
  crowToeNeed(m.bodies[first].parentBody==upper && m.bodies[first+1].parentBody==first && m.bodies[first+2].parentBody==first+1,"bad parent chain");
  for(unsigned i: {upper,first,first+1,first+2}){
   const auto& b=m.bodies[i];
   crowToeNeed(b.massAndInverseMass.x>0 && b.inertiaRow0.x>0 && b.inertiaRow1.y>0 && b.inertiaRow2.z>0,"nonpositive mass or inertia");
  }
  for(unsigned j=0;j<3;++j){
   const auto& d=m.dofs[before.dofs.size()+3*side+j];
   crowToeNeed(d.limits.x<d.limits.y && d.limits.z>0 && d.limits.w>0,"invalid joint limit");
  }
 }
 crowToeNeed(m.shapes.size()==before.shapes.size()+4,"unexpected collider count");
 crowToeNeed(m.joints[before.joints.size()].axis0.z==-m.joints[before.joints.size()+3].axis0.z,"elbow axes not mirrored");
 crowToeNeed(m.joints[before.joints.size()+2].axis0.x==-m.joints[before.joints.size()+5].axis0.x,"wrist axes not mirrored");
 auto bad=original;bad.id="wrong-version";bool rejected=false;
 try{(void)makeCrowArticulatedWingMechanicsFixture(std::move(bad));}catch(const std::invalid_argument&){rejected=true;}
 crowToeNeed(rejected,"incompatible input accepted");
 std::cout<<"PASS versioned articulated wing structural checks; flight aerodynamics unavailable\n";
 std::cout<<"mass_before="<<total(before)<<" mass_after="<<total(m)<<" bodies="<<m.bodies.size()<<" joints="<<m.joints.size()<<" shapes="<<m.shapes.size()<<"\n";
}
