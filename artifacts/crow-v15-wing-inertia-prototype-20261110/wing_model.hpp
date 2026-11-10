#pragma once
#include <cmath>
#include <stdexcept>
inline void crowResearchWingInertia(metalrobo::RobotPack& p) {
  auto& m=p.mechanics;
  if(m.bodies.size()!=25 || m.joints.size()<6 || m.defaultQ.size()!=31) throw std::logic_error("unexpected crow v15 topology");
  constexpr double referenceBodyMass=0.3876, referenceWingMass=0.0308, referenceWingMoment=0.0002613;
  const double actualMass=0.450039991512, ratio=actualMass/referenceBodyMass;
  // The literature row includes scaled/related-species values; this is a bounded research model, not specimen calibration.
  const double wingChainMass=referenceWingMass*ratio;
  const double newDistalMass=wingChainMass-0.000020;
  const double added=2.0*(newDistalMass-m.bodies[5].massAndInverseMass.x);
  auto& torso=m.bodies[0];const double oldMass=torso.massAndInverseMass.x,newMass=oldMass-added;
  if(!(newMass>0 && newDistalMass>0)) throw std::logic_error("invalid wing mass partition");
  const float f=static_cast<float>(newMass/oldMass);
  torso.massAndInverseMass.x=static_cast<float>(newMass);torso.massAndInverseMass.y=static_cast<float>(1/newMass);
  for(float* x : {&torso.inertiaRow0.x,&torso.inertiaRow0.y,&torso.inertiaRow0.z,&torso.inertiaRow1.x,&torso.inertiaRow1.y,&torso.inertiaRow1.z,&torso.inertiaRow2.x,&torso.inertiaRow2.y,&torso.inertiaRow2.z}) *x*=f;
  for(float* x : {&torso.inverseInertiaRow0.x,&torso.inverseInertiaRow0.y,&torso.inverseInertiaRow0.z,&torso.inverseInertiaRow1.x,&torso.inverseInertiaRow1.y,&torso.inverseInertiaRow1.z,&torso.inverseInertiaRow2.x,&torso.inverseInertiaRow2.y,&torso.inverseInertiaRow2.z}) *x/=f;
  for(unsigned side=0;side<2;side++) {
    auto& sweep=m.joints[side];auto& flap=m.joints[2+side];auto& wing=m.bodies[5+side];
    const auto pa=sweep.childAnchor;const double angle=m.defaultQ[flap.qOffset]*(side==0?1.0:-1.0),c=std::cos(angle),s=std::sin(angle);
    flap.parentAnchor=pa;flap.childAnchor={pa.x,static_cast<float>(c*pa.y+s*pa.z),static_cast<float>(-s*pa.y+c*pa.z),0};
    const double shoulderMoment=referenceWingMoment*std::pow(ratio,5.0/3.0),r2=pa.y*pa.y+pa.z*pa.z;
    const double ix=shoulderMoment-newDistalMass*r2,iy=ix*.25,iz=ix;
    if(!(ix>0&&iy>0&&iz>0))throw std::logic_error("source moment below parallel-axis mass contribution");
    wing.massAndInverseMass.x=static_cast<float>(newDistalMass);wing.massAndInverseMass.y=static_cast<float>(1/newDistalMass);
    wing.inertiaRow0={static_cast<float>(ix),0,0,0};wing.inertiaRow1={0,static_cast<float>(iy),0,0};wing.inertiaRow2={0,0,static_cast<float>(iz),0};
    wing.inverseInertiaRow0={static_cast<float>(1/ix),0,0,0};wing.inverseInertiaRow1={0,static_cast<float>(1/iy),0,0};wing.inverseInertiaRow2={0,0,static_cast<float>(1/iz),0};
  }
  p.sourceRevision+="+research-wing-mass-v1";p.capabilities.push_back("research_scaled_wing_mass_not_specimen_calibrated");
}
