#pragma once
#include "numi/matter/matter.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

// Synthetic hollow tissue shell. The central cell is deliberately absent:
// cavity blood volume never overlaps the FEM tissue volume.
namespace vascular_cavity_fixture {
using Vec=std::array<double,3>;
using Face=std::array<std::uint32_t,3>;
inline Vec add(Vec a,const Vec& b){for(unsigned i=0;i<3;++i)a[i]+=b[i];return a;}
inline Vec subtract(Vec a,const Vec& b){for(unsigned i=0;i<3;++i)a[i]-=b[i];return a;}
inline Vec scaled(Vec a,double s){for(double& v:a)v*=s;return a;}
inline double dot(const Vec& a,const Vec& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline Vec cross(const Vec& a,const Vec& b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline std::uint32_t node(unsigned x,unsigned y,unsigned z){return x+4u*y+16u*z;}
struct Shell {
    numi::matter::ObjectSource object;
    std::vector<Face> lumenFaces;
    std::vector<std::uint32_t> innerNodes;
};
inline Shell hollowShell(bool pinInner=false) {
    using namespace numi::matter;
    Shell shell;
    auto& object=shell.object;
    object.name="synthetic:hollow_tissue_shell";object.representation=Representation::fem;
    object.characteristicLength=.01;object.deformableContact=false;object.mixedFEM=false;
    for(unsigned z=0;z<4;++z)for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x){
        object.femNodes.push_back({(double(x)-1.5)*.01,(double(y)-1.5)*.01,(double(z)-1.5)*.01});
        if(x==0||y==0||z==0||x==3||y==3||z==3||pinInner)object.femFixedNodes.push_back(node(x,y,z));
        if(x>0&&x<3&&y>0&&y<3&&z>0&&z<3)shell.innerNodes.push_back(node(x,y,z));
    }
    for(unsigned z=0;z<3;++z)for(unsigned y=0;y<3;++y)for(unsigned x=0;x<3;++x){
        if(x==1&&y==1&&z==1)continue;
        std::array<unsigned,3> axes{0,1,2};
        do {
            std::array<unsigned,3> corner{x,y,z};
            std::array<std::uint32_t,4> tet{node(x,y,z),0,0,node(x+1,y+1,z+1)};
            ++corner[axes[0]];tet[1]=node(corner[0],corner[1],corner[2]);
            ++corner[axes[1]];tet[2]=node(corner[0],corner[1],corner[2]);
            const auto a=subtract(object.femNodes[tet[1]],object.femNodes[tet[0]]);
            const auto b=subtract(object.femNodes[tet[2]],object.femNodes[tet[0]]);
            const auto c=subtract(object.femNodes[tet[3]],object.femNodes[tet[0]]);
            if(dot(a,cross(b,c))<0)std::swap(tet[1],tet[2]);
            object.tetrahedra.push_back({tet});
        } while(std::next_permutation(axes.begin(),axes.end()));
    }
    struct Occurrence {unsigned count=0;Face oriented{};};
    std::map<Face,Occurrence> boundary;
    for(const auto& tet:object.tetrahedra){const auto& t=tet.nodes;
        for(const Face face:std::array<Face,4>{{{t[1],t[2],t[3]},{t[0],t[3],t[2]},{t[0],t[1],t[3]},{t[0],t[2],t[1]}}}){
            auto key=face;std::sort(key.begin(),key.end());auto& entry=boundary[key];++entry.count;entry.oriented=face;
        }
    }
    for(const auto& [key,entry]:boundary)if(entry.count==1){
        const bool inner=std::all_of(key.begin(),key.end(),[&](auto i){return std::find(shell.innerNodes.begin(),shell.innerNodes.end(),i)!=shell.innerNodes.end();});
        if(inner){auto face=entry.oriented;std::swap(face[1],face[2]);shell.lumenFaces.push_back(face);}
    }
    if(shell.lumenFaces.size()!=12||shell.innerNodes.size()!=8||object.tetrahedra.size()!=156)
        throw std::runtime_error("hollow shell generator topology differs");
    return shell;
}
inline double volume(const std::vector<Vec>& x,const std::vector<Face>& faces){
    double value=0;for(const auto& f:faces)value+=dot(x[f[0]],cross(x[f[1]],x[f[2]]))/6.;return value;
}
inline std::vector<Vec> gradient(const std::vector<Vec>& x,const std::vector<Face>& faces){
    std::vector<Vec> result(x.size());
    for(const auto& f:faces)for(unsigned i=0;i<3;++i)result[f[i]]=add(result[f[i]],scaled(cross(x[f[(i+1)%3]],x[f[(i+2)%3]]),1./6));
    return result;
}
inline std::vector<Vec> discreteGradient(const std::vector<Vec>& old,const std::vector<Vec>& next,const std::vector<Face>& faces){
    std::vector<Vec> mid(old.size());for(unsigned i=0;i<old.size();++i)mid[i]=scaled(add(old[i],next[i]),.5);
    auto a=gradient(old,faces),b=gradient(mid,faces),c=gradient(next,faces);
    for(unsigned i=0;i<a.size();++i)a[i]=scaled(add(add(a[i],scaled(b[i],4)),c[i]),1./6);
    return a;
}
inline numi::matter::WorldSource world(bool pinInner=false,double dt=.001){
    using namespace numi::matter;
    WorldSource s;s.environmentCount=2;s.frameTimestep=dt;s.gravity={0,0,0};
    s.mixedSolver.relativeResidual=1e-7;s.mixedSolver.newtonIterations=12;s.mixedSolver.fgmresIterations=64;
    const auto material=parseMatter(R"(material hollow_shell_fixture {
      parameter density : kg/m^3 = 1000 in [500,1500];
      parameter mu : Pa = 1000 in [1,100000];
      parameter lambda : Pa = 10000 in [1,1000000];
      model neo_hookean; energy = neo_hookean(mu,lambda); valid = J() - 0.1;
      supports fem;
      interface {static_friction=0;dynamic_friction=0;restitution=0;adhesion=0;}
      limits {minimum_J=0.1;maximum_J=10;maximum_stress=1e9;maximum_energy_density=1e9;}
    })");
    if(!material.succeeded())throw std::runtime_error("synthetic hollow shell material parse failed");
    s.materials.push_back(material.material);auto shell=hollowShell(pinInner);s.objects.push_back(shell.object);
    auto& n=s.vascular;
    // Explicit synthetic identities; all exact fixture data are also covered
    // by normal compiled-world fingerprints. No source-calibration claim.
    n.contentIdentity={0x686f6c6c6f772d63ULL,0x61766974792d7631ULL,0,1};
    n.species.push_back({1,"synthetic:tracer",1e-6,1e-5});
    VascularCompartmentSource cavity;cavity.stableIdentifier=11;cavity.anatomicalIdentifier="synthetic:hollow_lumen";
    cavity.initialVolume=volume(shell.object.femNodes,shell.lumenFaces);cavity.volumeScale=1e-6;cavity.volumeResidualTolerance=1e-5;
    cavity.initialSpeciesAmounts={0};cavity.pressureLaw=VascularPressureLaw::deformingCavity;
    n.compartments.push_back(cavity);
    VascularCompartmentSource reservoir;reservoir.stableIdentifier=12;reservoir.anatomicalIdentifier="synthetic:linear_reservoir";
    reservoir.referenceVolume=4e-6;reservoir.initialVolume=4e-6;reservoir.referencePressure=100;reservoir.compliance=1e-8;
    reservoir.volumeScale=1e-6;reservoir.volumeResidualTolerance=1e-5;reservoir.initialSpeciesAmounts={2e-6};n.compartments.push_back(reservoir);
    VascularConnectionSource edge;edge.stableIdentifier=21;edge.fromCompartment=12;edge.toCompartment=11;
    edge.resistance=1e9;edge.flowScale=1e-7;edge.pressureScale=100;edge.flowResidualTolerance=1e-5;n.connections.push_back(edge);
    VascularCavitySource boundary;boundary.stableIdentifier=31;boundary.compartment=11;boundary.objectIndex=0;
    boundary.initialPressure=0;boundary.pressureScale=100;boundary.geometryResidualTolerance=1e-5;
    boundary.sourceIdentity={0x73796e7468657469ULL,0x632d636176697479ULL,0,1};boundary.mechanicalIdentity={0x686f6c6c6f772d66ULL,0x656d2d7368656c6cULL,0,1};
    for(unsigned i=0;i<shell.lumenFaces.size();++i)boundary.faces.push_back({100+i,shell.lumenFaces[i],VascularCavityFaceRole::materialWall});
    n.cavities.push_back(boundary);return s;
}
} // namespace vascular_cavity_fixture
