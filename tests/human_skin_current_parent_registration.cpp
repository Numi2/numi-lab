#include "NumiHumanPhysicalSkinChildRegisteredSurfaceSidecars.hpp"
#include <iostream>
using namespace metalrobo::human::coarse_tissue;
int main(){
 PhysicalSkinParentMassPacket parent;
 parent.currentWorldPoints={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
 parent.tetrahedra={{0,1,2,3},{0,1,2,3}};
 parent.materialIndices={15,16};parent.donorBodyIndices={23,23};parent.anatomicalRegionIndices={23,23};
 PhysicalSkinRegisteredSurfaceCSVSet s;
 for(unsigned l=0;l<3;++l){
  s.surfaces[l].worldPoints={{.25,.25,.25}};s.surfaces[l].donorBodyIndices={23};
  PhysicalSkinParentRegistrationRow row;row.available=true;row.tet=l==2?1:0;row.bary={.25,.25,.25,.25};
  row.inferredCurrentParent=l==0;s.nodeParent[l]={row};
 }
 PhysicalSkinSurfaceParentRegistrationReport report;std::string error;
 if(!validatePhysicalSkinSurfaceParentRegistration(s,parent,report,error)||report.checkedNodes[0]!=1)return 1;
 const auto good=report;
 parent.anatomicalRegionIndices[0]=24;
 if(validatePhysicalSkinSurfaceParentRegistration(s,parent,report,error)||error.find("same-donor-region")==std::string::npos||report.checkedNodes!=good.checkedNodes)return 2;
 parent.anatomicalRegionIndices[0]=23;parent.donorBodyIndices[0]=24;
 if(validatePhysicalSkinSurfaceParentRegistration(s,parent,report,error)||error.find("same-donor-region")==std::string::npos)return 3;
 parent.donorBodyIndices[0]=23;s.surfaces[0].donorBodyIndices.clear();
 if(validatePhysicalSkinSurfaceParentRegistration(s,parent,report,error))return 4;
 s.surfaces[0].donorBodyIndices={23};s.nodeParent[0][0].residual[0]=.001;s.nodeParent[0][0].distance=.001;
 if(validatePhysicalSkinSurfaceParentRegistration(s,parent,report,error)||error.find("reconstruct")==std::string::npos)return 5;
 s.nodeParent[0][0].residual={};s.nodeParent[0][0].distance=0;
 if(!validatePhysicalSkinSurfaceParentRegistration(s,parent,report,error))return 6;
 std::cout<<"current_parent_registration=PASS same_material_donor_region=checked different_region=reject different_donor=reject missing_donor=reject wrong_reconstruction=reject atomic_output=true physical_steps=0\n";
}
