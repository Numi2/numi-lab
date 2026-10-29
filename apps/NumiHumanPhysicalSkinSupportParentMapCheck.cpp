#include "NumiHumanPhysicalSkinSupportParentMap.hpp"
#include <iostream>
#include <limits>
using namespace metalrobo::human::coarse_tissue;
int main() {
 PhysicalSkinChildMassMappingBundle base;
 base.parent.currentWorldPoints={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
 base.parent.referenceWorldPoints=base.parent.currentWorldPoints;
 base.parent.tetrahedra={{0,1,2,3}};
 base.childTetrahedra=base.parent.tetrahedra;
 base.parent.materialIndices=base.childMaterialIndices={15};
 base.parent.donorBodyIndices=base.childDonorBodyIndices={7};
 base.parent.anatomicalRegionIndices=base.childAnatomicalRegionIndices={9};
 base.parent.stableSourceFaceIdentities=base.childStableSourceFaceIdentities={123};
 base.volumeNodeParentTetrahedra={0,0,0,0};
 base.volumeNodeParentBarycentric={{{1,0,0,0}},{{0,1,0,0}},{{0,0,1,0}},{{0,0,0,1}}};
 base.volumeNodeParentResidualWorldM.assign(4,V3{.0001,0,0});
 base.currentWorldPoints=base.referenceWorldPoints=base.parent.currentWorldPoints;
 for(auto& x:base.currentWorldPoints)x[0]+=.0001;
 base.referenceWorldPoints=base.currentWorldPoints;
 base.spatialMassTransportResult.childParentOffsets={0,1};
 base.spatialMassTransportResult.childParentRows={0};
 std::vector<std::array<float,3>> actual{{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
 std::string error;PhysicalSkinSupportParentMap out;
 if(!provePhysicalSkinSupportParentMap(base,actual,base.parent.tetrahedra,.0004,out,error)){
  std::cerr<<error;return 1;
 }
 const auto saved=out.childNodeToParent;const auto savedD=out.maximumNodeDisplacementM;
 unsigned rejected=0;
 auto reject=[&](const auto& bad,double bound=.0004){
  if(provePhysicalSkinSupportParentMap(bad,actual,base.parent.tetrahedra,bound,out,error)||
     out.childNodeToParent!=saved||out.maximumNodeDisplacementM!=savedD)return false;
  ++rejected;return true;
 };
 auto bad=base;bad.volumeNodeParentBarycentric[0]={.5,.5,0,0};if(!reject(bad))return 2;
 bad=base;bad.volumeNodeParentBarycentric[0]={0,1,0,0};if(!reject(bad))return 3;
 bad=base;bad.volumeNodeParentTetrahedra[0]=1;if(!reject(bad))return 4;
 bad=base;bad.volumeNodeParentResidualWorldM[0][0]=0;if(!reject(bad))return 5;
 bad=base;bad.childMaterialIndices[0]=16;if(!reject(bad))return 6;
 bad=base;bad.childDonorBodyIndices[0]=8;if(!reject(bad))return 7;
 bad=base;bad.childAnatomicalRegionIndices[0]=8;if(!reject(bad))return 8;
 bad=base;bad.childStableSourceFaceIdentities[0]=8;if(!reject(bad))return 9;
 bad=base;std::swap(bad.childTetrahedra[0][0],bad.childTetrahedra[0][1]);if(!reject(bad))return 10;
 bad=base;bad.childTetrahedra[0][0]=4;if(!reject(bad))return 11;
 bad=base;bad.spatialMassTransportResult.childParentRows[0]=1;if(!reject(bad))return 12;
 bad=base;bad.spatialMassTransportResult.childParentOffsets[1]=2;if(!reject(bad))return 13;
 if(!reject(base,.00001))return 14;
 bad=base;bad.parent.currentWorldPoints[0][0]=.0000001;if(!reject(bad))return 15;
 bad=base;bad.referenceWorldPoints[0][0]=std::numeric_limits<double>::quiet_NaN();if(!reject(bad))return 16;
 bad=base;bad.parent.referenceWorldPoints[0][0]=std::numeric_limits<double>::quiet_NaN();if(!reject(bad))return 17;
 bad=base;bad.childStableSourceFaceIdentities.clear();if(!reject(bad))return 18;
 auto reordered=base;
 for(auto* points:{&reordered.currentWorldPoints,&reordered.referenceWorldPoints,&reordered.volumeNodeParentResidualWorldM})
  std::swap((*points)[0],(*points)[1]);
 std::swap(reordered.volumeNodeParentBarycentric[0],reordered.volumeNodeParentBarycentric[1]);
 reordered.childTetrahedra[0]={1,0,2,3};
 PhysicalSkinSupportParentMap perm;
 if(!provePhysicalSkinSupportParentMap(reordered,actual,base.parent.tetrahedra,.0004,perm,error)||
    perm.childNodeToParent!=std::vector<std::uint32_t>{1,0,2,3})return 19;
 auto attachment=base;
 attachment.currentWorldPoints=attachment.parent.currentWorldPoints;
 attachment.referenceWorldPoints=attachment.parent.referenceWorldPoints;
 if(!provePhysicalSkinInheritedAttachmentNode(attachment,actual,0,0))return 20;
 attachment.currentWorldPoints[0][0]=.00000001;
 if(provePhysicalSkinInheritedAttachmentNode(attachment,actual,0,0))return 21;
 attachment.currentWorldPoints=attachment.parent.currentWorldPoints;
 attachment.referenceWorldPoints[0][0]=.00000001;
 if(provePhysicalSkinInheritedAttachmentNode(attachment,actual,0,0))return 22;
 attachment.referenceWorldPoints=attachment.parent.referenceWorldPoints;
 if(provePhysicalSkinInheritedAttachmentNode(attachment,actual,0,1))return 23;
 attachment.volumeNodeParentBarycentric[0]={.5,.5,0,0};
 if(provePhysicalSkinInheritedAttachmentNode(attachment,actual,0,0))return 24;
 attachment=base;attachment.currentWorldPoints=attachment.parent.currentWorldPoints;
 attachment.referenceWorldPoints=attachment.parent.referenceWorldPoints;
 attachment.parent.referenceWorldPoints[0][0]=std::numeric_limits<double>::quiet_NaN();
 if(provePhysicalSkinInheritedAttachmentNode(attachment,actual,0,0))return 25;
 std::cout<<"support_parent_topology=PASS bounded_geometry_change=PASS reordered_nodes=PASS invalid_cases="
          <<rejected<<" failure_preserves_state=PASS unchanged_attachment_node=PASS moved_attachment_rejected=PASS contact_forces_created=0 physical_admission=false\n";
}
