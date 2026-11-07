#include "numi/matter/matter.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <cmath>
#include <iostream>
int main(){
 using namespace numi::matter;
 auto material=parseMatter(R"(
 material pad {
 parameter density : kg/m^3 = 1100 in [1000,1200];
 parameter mu : Pa = 2000 in [1000,3000];
 parameter lambda : Pa = 5000 in [4000,6000];
 model neo_hookean; energy = neo_hookean(mu,lambda); valid = J()-0.05;
 supports fem;
 interface { static_friction = 0; dynamic_friction = 0; restitution = 0; adhesion = 0; }
 limits { minimum_J = 0.05; maximum_J = 20; maximum_stress = 1e9; maximum_energy_density = 1e9; }
 })");if(!material.succeeded())return 1;
 WorldSource s;s.materials.push_back(material.material);
 ObjectSource o;o.name="pad";o.materialIndex=0;o.representation=Representation::fem;o.mixedFEM=false;
 o.femNodes={{0,0,0},{.02,0,0},{0,.02,0},{0,0,.02}};
 o.tetrahedra.push_back({{0,1,2,3}});s.objects.push_back(o);
 auto compile=[&](){return compileWorld(s,{});};
 auto legacy=compile();if(!legacy.succeeded()){for(const auto& d:legacy.diagnostics)std::cerr<<d.message<<"\n";return 2;}
 s.mixedSolver.mechanicalRelativeImpulseTolerance=.001;
 if(compile().succeeded())return 3;
 s.mixedSolver.mechanicalAbsoluteImpulseTolerance=1e-10;
 s.mixedSolver.mechanicalImpulseScale=1e-5;
 auto physical=compile();if(!physical.succeeded())return 4;
 if(physical.world.physicsFingerprint==legacy.world.physicsFingerprint)return 5;
 const auto first=physical.world.physicsFingerprint;
 s.mixedSolver.mechanicalImpulseScale=2e-5;
 if(compile().world.physicsFingerprint==first)return 6;
 s.mixedSolver.mechanicalImpulseScale=NAN;if(compile().succeeded())return 7;
 s.mixedSolver.mechanicalImpulseScale=1e-5;s.objects[0].mixedFEM=true;
 if(compile().succeeded())return 8;
 if(RuntimeConfiguration{}.experimentalFEMPlaneAugmentedLagrangian)return 9;
 if(metalrobo::MetalWorldStepConfig{}.directionalCCDAdvancement||
    metalrobo::MetalWorldStepConfig{}.intervalLocalCCDAdvancement)return 10;
 std::cout<<"physical residual policy admission, identity and disabled defaults passed\n";
}
