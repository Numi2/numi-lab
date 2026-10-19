#include "NumiHumanCoarseTissuePreparation.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <string>
using namespace metalrobo::human::coarse_tissue;
static V3 rotate(const V3 p,double x,double y,double z,double w){
    const V3 q{x,y,z};const auto c=cross(q,p);
    return add(p,add(mul(c,2*w),mul(cross(q,c),2)));
}
static bool near(double a,double b,double e=2e-11){return std::abs(a-b)<=e*std::max({1.0,std::abs(a),std::abs(b)});}
int main(){
    numi::matter::WorldSource world;
    numi::matter::MaterialProgram material;material.name="proxy";
    material.supportedRepresentations.push_back(numi::matter::Representation::fem);
    numi::matter::Parameter density;density.name="density";density.defaultValue=0;
    density.lower=0;density.upper=2000;density.identifiable=false;
    material.parameters.push_back(density);world.materials.push_back(material);
    RegisteredSurface source;source.name="prostate";source.provider="BodyParts3D";
    source.sourceMember="FJ3139";
    source.sourceMeshSHA256=std::string(64,'a');source.sourcePayloadSHA256=std::string(64,'b');
    source.stableIdentifier=463;source.sourceBodyIndex=20;source.sourceLayer=3;
    source.donorLocalPoints={{0.2,0.1,-0.3},{1.2,0.1,-0.3},{0.2,2.1,-0.3},{0.2,0.1,2.7}};
    source.faces={{1,2,3},{0,3,2},{0,1,3},{0,2,1}};
    source.sourceVertexIndices={41,47,53,59};source.sourceFaceIndices={1001,1002,1003,1004};
    double qz=double(float(std::sin(0.35))),qw=double(float(std::cos(0.35)));
    double qn=std::sqrt(qz*qz+qw*qw);qz/=qn;qw/=qn;
    const V3 translation{17.25,-8.75,4.5};
    for(const auto p:source.donorLocalPoints)source.worldPoints.push_back(add(translation,rotate(p,0,0,qz,qw)));
    GeometryAudit audit;std::string error;
    if(!auditClosedSurface(source,audit,error)){std::cerr<<"source audit: "<<error<<"\n";return 1;}
    PreparedOrganicFEM prepared;
    if(!appendRegisteredStarKernelFEM(world,source,0,"prostate","FJ3139",463,3,
            "proxy","CPU reference regression",audit.volumeM3*1000.0,
            "test_prostate",prepared,error)){std::cerr<<"prepare: "<<error<<"\n";return 2;}
    const V3 localCenter{0.45,0.6,0.45};
    const auto expectedCenter=add(translation,rotate(localCenter,0,0,qz,qw));
    const auto actualCenter=world.objects[0].femNodes.back();
    for(int a=0;a<3;a++)if(!near(actualCenter[a],expectedCenter[a])){
        std::cerr<<"translated centroid mismatch axis="<<a<<" actual="<<actualCenter[a]<<" expected="<<expectedCenter[a]<<"\n";return 3;
    }
    for(int a=0;a<3;a++)if(!near(prepared.donorMassNodes.back().localPosition[a],localCenter[a])){
        std::cerr<<"donor local centroid mismatch axis="<<a<<" actual="<<prepared.donorMassNodes.back().localPosition[a]<<" expected="<<localCenter[a]<<"\n";return 4;
    }
    if(prepared.nodeSourceVertex.size()!=5||prepared.nodeSourceVertex[0]!=41||
       prepared.nodeSourceVertex[1]!=47||prepared.nodeSourceVertex[2]!=53||
       prepared.nodeSourceVertex[3]!=59||prepared.nodeSourceVertex[4]!=NM_INVALID_INDEX){
        std::cerr<<"source vertex ancestry mismatch\n";return 5;
    }
    if(prepared.tetSourceFace!=std::vector<std::uint32_t>{1001,1002,1003,1004}){
        std::cerr<<"source face ancestry mismatch\n";return 6;
    }
    double mass=0;for(const auto& row:prepared.donorMassNodes)mass+=row.massKg;
    if(!near(prepared.volumeM3,audit.volumeM3)||!near(mass,prepared.massDebitKg)||
       std::abs(prepared.massResidualKg)>1e-12*prepared.massDebitKg){
        std::cerr<<"volume/mass closure mismatch\n";return 7;
    }
    std::cout<<"translated_star_centroid=PASS world_center_m="
             <<actualCenter[0]<<","<<actualCenter[1]<<","<<actualCenter[2]
             <<" world_local_volume_relative_error="
             <<std::abs(prepared.volumeM3-audit.volumeM3)/audit.volumeM3
             <<" mass_residual_kg="<<prepared.massResidualKg
             <<" node_ancestry=4/4 face_ancestry=4/4\n";
}
