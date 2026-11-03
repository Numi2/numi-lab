#include "NumiHumanSkinFatFixedCavityReconditioning.hpp"
#include <cassert>
#include <iostream>

using namespace metalrobo::human::coarse_tissue;
namespace sc=skin_fat_cavity_detail;
namespace {
void put16(std::vector<std::uint8_t>& b,std::uint16_t x){b.push_back(x&255u);b.push_back(x>>8u);}
void put32(std::vector<std::uint8_t>& b,std::uint32_t x){put16(b,x&65535u);put16(b,x>>16u);}
std::vector<std::uint8_t> npy(std::string descriptor,std::string shape,const std::vector<std::uint32_t>& values) {
    std::string h="{'descr': '"+descriptor+"', 'fortran_order': False, 'shape': "+shape+", }";
    while((10u+h.size()+1u)%64u)h+=' ';h+='\n';
    std::vector<std::uint8_t>b{0x93u,'N','U','M','P','Y',1u,0u};put16(b,h.size());
    b.insert(b.end(),h.begin(),h.end());for(auto x:values)put32(b,x);return b;
}
void writeProposal(const std::filesystem::path& path,const std::array<sc::Tet,2>& next,
                   bool wrongParents=false,bool sharedBoundary=false) {
    std::map<std::string,std::vector<std::uint8_t>> entries;
    entries["retained_input_cell_rows"]=npy("<u4","(0,)",{});
    std::vector<std::uint32_t> cells;for(const auto& t:next)cells.insert(cells.end(),t.begin(),t.end());
    if(sharedBoundary)for(const auto& t:next)for(auto v:t)cells.push_back(v==0u?0u:v+4u);
    entries["replacement_tetrahedra"]=npy("<u4",sharedBoundary?"(4, 4)":"(2, 4)",cells);
    entries["replacement_tet_offsets"]=npy("<u4",sharedBoundary?"(3,)":"(2,)",sharedBoundary?std::vector<std::uint32_t>{0,2,4}:std::vector<std::uint32_t>{0,2});
    entries["replacement_parent_offsets"]=npy("<u4",sharedBoundary?"(5,)":"(3,)",sharedBoundary?std::vector<std::uint32_t>{0,3,6,9,12}:std::vector<std::uint32_t>{0,3,6});
    entries["replacement_parent_cell_rows"]=npy("<u4",sharedBoundary?"(12,)":"(6,)",sharedBoundary?std::vector<std::uint32_t>{0,1,2,0,1,2,3,4,5,3,4,5}:std::vector<std::uint32_t>{0,1,2,0,1,wrongParents?1u:2u});
    entries["source_volume_sha256"]=npy("<U64","()",std::vector<std::uint32_t>(64u,'a'));
    std::vector<std::uint8_t> bytes,directory;
    for(const auto& [key,raw]:entries) {
        const std::string name=key+".npy";const auto start=bytes.size();
        const auto crc=static_cast<std::uint32_t>(crc32(0L,raw.data(),raw.size()));
        put32(bytes,0x04034b50u);put16(bytes,20);put16(bytes,0);put16(bytes,0);
        put32(bytes,0);put32(bytes,crc);put32(bytes,raw.size());put32(bytes,raw.size());
        put16(bytes,name.size());put16(bytes,0);bytes.insert(bytes.end(),name.begin(),name.end());bytes.insert(bytes.end(),raw.begin(),raw.end());
        put32(directory,0x02014b50u);put16(directory,20);put16(directory,20);put16(directory,0);put16(directory,0);
        put32(directory,0);put32(directory,crc);put32(directory,raw.size());put32(directory,raw.size());
        put16(directory,name.size());put16(directory,0);put16(directory,0);put16(directory,0);put16(directory,0);
        put32(directory,0);put32(directory,start);directory.insert(directory.end(),name.begin(),name.end());
    }
    const auto begin=bytes.size();bytes.insert(bytes.end(),directory.begin(),directory.end());
    put32(bytes,0x06054b50u);put16(bytes,0);put16(bytes,0);put16(bytes,entries.size());put16(bytes,entries.size());
    put32(bytes,directory.size());put32(bytes,begin);put16(bytes,0);
    std::ofstream f(path,std::ios::binary|std::ios::trunc);f.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());assert(f.good());
}
sc::Tet orient(const std::vector<V3>& p,sc::Tet t) {
    if(sc::volume(p,t)<0.0)std::swap(t[0],t[1]);return t;
}
struct Fixture {
    numi::matter::WorldSource world;
    PhysicalSkinChildMassMappingBundle bundle;
    PreparedOrganicFEM skin,fat;
    RegisteredSkinFatReductionSourceLabels labels;
    std::array<sc::Tet,2> next;
    Fixture() {
        std::vector<V3> points{{0,0,0},{.002,0,0},{0,.002,0},{.001,.0000002,.002},{.001,.0000002,-.002}};
        for(auto& p:points)for(auto& x:p)x=float(x);
        std::array<sc::Tet,3> old{{orient(points,{3,4,0,1}),orient(points,{3,4,1,2}),orient(points,{3,4,2,0})}};
        next={orient(points,{0,1,2,3}),orient(points,{0,2,1,4})};
        bundle.currentWorldPoints=points;bundle.referenceWorldPoints=points;
        bundle.childTetrahedra.assign(old.begin(),old.end());bundle.childMaterialIndices.assign(3,15);
        bundle.childDonorBodyIndices.assign(3,37);bundle.childAnatomicalRegionIndices.assign(3,58);
        bundle.fractionalSemanticMassRouteRequired=true;bundle.massMappingPathSHA256=std::string(64,'b');
        bundle.parent.referenceWorldPoints=points;bundle.parent.tetrahedra.assign(old.begin(),old.end());
        auto& object=world.objects.emplace_back();object.name="fixture";object.mixedFEM=false;object.representation=numi::matter::Representation::fem;
        object.femNodes=points;object.femReferenceNodes=points;object.femMaterialIndices.assign(3,15);
        object.femReferenceSourceIdentity={1,2,3,4};object.femTetrahedronMassSourceIdentity={5,6,7,8};
        world.materials.resize(17);numi::matter::Parameter density;density.name="density";density.lower=900;density.upper=1200;density.defaultValue=1000;
        world.materials[15].parameters.push_back(density);world.materials[16].parameters.push_back(density);
        skin.materialIndex=15;skin.firstGlobalFEMNode=42;fat.materialIndex=16;fat.firstGlobalFEMNode=42;
        skin.tetChildIndex={0,1,2};skin.tetParentTetrahedron={0,1,2};skin.tetParentSurfaceFace={7,7,7};
        skin.tetParentMaterialIndex={15,15,15};skin.tetSourceFaceIdentity={7,7,7};
        auto& transfer=bundle.spatialMassTransportResult;transfer.usesFractionalSemanticSupport=true;
        bundle.maximumSpatialMassTransportDistanceM=.01;transfer.maximumTransportDistanceM=.01;
        transfer.childParentOffsets={0,1,2,3};transfer.childParentRows={0,1,2};
        std::vector<double> nodal(5,0.0);
        for(unsigned i=0;i<3u;++i) {
            const double mass=sc::volume(points,old[i])*(950.0+100.0*i);
            object.tetrahedra.push_back({old[i]});object.femTetrahedronMassesKg.push_back(mass);
            transfer.childMassKg.push_back(mass);transfer.childParentTransportMassKg.push_back(mass);
            for(auto node:old[i])nodal[node]+=mass*.25;
            for(auto node:old[i])for(auto parentNode:old[i])
                skin.donorMassContributions.push_back({42+node,37,mass*.0625,points[parentNode]});
        }
        for(unsigned node=0;node<5u;++node) {
            skin.donorMassNodes.push_back({42+node,37,nodal[node],points[node]});
            bundle.parent.sourceMassContributions.push_back({node,15,37,nodal[node],points[node]});
        }
        labels.sourceTetrahedronCount=3;labels.explicitUnpairedVolumeTopology=true;
        labels.tetrahedronParentTetrahedronIndices={0,1,2};
        labels.massTransferKind=UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
        labels.tetrahedronRegionLabels={37,37,37};labels.tetrahedronBasisRegionLabels={0,0,0};
        labels.tetrahedronSourceFaceIndices={7,7,7};labels.tetrahedronOriginalSourceFaceIndices={7,7,7};
        labels.tetrahedronSourceDonorBodyIndices={37,37,37};labels.tetrahedronSourceParentBodyIndices={2,2,2};
        labels.tetrahedronRegisteredLayerFacetSurfaceFaces.assign(12,NM_INVALID_INDEX);
        labels.tetrahedronRegisteredLayerFacetSourceFaceIdentities.assign(12,NM_INVALID_INDEX);
        labels.tetrahedronAuthoredHeadSidewallFacetFlags.assign(12,0);
        labels.tetrahedronAuthoredHeadSidewallStableFaceIdentities.assign(12,NM_INVALID_INDEX);
    }
};
}
int main() {
    const auto path=std::filesystem::temp_directory_path()/("numi-cavity32-"+std::to_string(getpid())+".npz");
    Fixture original;
    sc::Edge edge;sc::Triangle ring;double before,after,fraction;std::string error;
    assert(sc::certifyGeometry(original.bundle.referenceWorldPoints,original.bundle.childTetrahedra,
        original.next,edge,ring,before,after,fraction,error));
    assert(before>=1e-5&&after>before*100.0&&std::abs(fraction-.5)<1e-12);
    for(int trial=0;trial<13;++trial) {
        Fixture f;writeProposal(path,f.next,trial==1);
        std::string sha;assert(physical_skin_child_mass_detail::sha256File(path,sha,error));
        if(trial==2)f.bundle.childMaterialIndices[1]=16;
        if(trial==3)f.bundle.childDonorBodyIndices[1]=38;
        if(trial==4)f.bundle.childAnatomicalRegionIndices[1]=59;
        if(trial==5)f.world.objects[0].femTetrahedronMassesKg[0]*=1.1;
        if(trial==6)sha[0]=sha[0]=='a'?'b':'a';
        if(trial==7)f.skin.donorMassNodes[0].massKg*=1.1;
        if(trial==8)f.bundle.maximumSpatialMassTransportDistanceM=0.0;
        if(trial==9)f.bundle.maximumSpatialMassTransportDistanceM=NAN;
        if(trial==10)f.bundle.maximumSpatialMassTransportDistanceM=
            f.bundle.spatialMassTransportResult.maximumTransportDistanceM=1e-5;
        if(trial==11)f.bundle.spatialMassTransportResult.maximumTransportDistanceM=.02;
        if(trial==12)f.bundle.parent.referenceWorldPoints[0][0]=INFINITY;
        SkinFatCavityReconditioningReport report;
        const auto oldTets=f.world.objects[0].tetrahedra;
        const auto oldIdentity=f.world.objects[0].femReferenceSourceIdentity;
        const bool pass=reconditionPreparedSkinFatFixedCavities(f.world,f.world.objects[0],f.bundle,
            f.skin,f.fat,f.labels,path,sha,std::string(64,'a'),report,error);
        if(trial==0) {
            if(!pass)std::cerr<<error<<"\n";
            assert(pass&&f.world.objects[0].tetrahedra.size()==2&&report.cavities==1);
            assert(report.maximumTransportDistanceUsedM>0.0&&report.maximumTransportDistanceUsedM<=.01);
            assert(report.maximumSourceMassChangeKg<1e-18&&report.maximumSourceMomentChangeKgM<1e-20);
            assert(f.world.objects[0].femNodes==original.world.objects[0].femNodes);
            assert(f.labels.sourceTetrahedronCount==2&&f.labels.tetrahedronParentAncestryIndices.empty());
            const double beforeMass=std::accumulate(original.world.objects[0].femTetrahedronMassesKg.begin(),original.world.objects[0].femTetrahedronMassesKg.end(),0.0);
            const double afterMass=std::accumulate(f.world.objects[0].femTetrahedronMassesKg.begin(),f.world.objects[0].femTetrahedronMassesKg.end(),0.0);
            assert(std::abs(beforeMass-afterMass)<1e-20);
        } else {
            if(pass)std::cerr<<"unexpected pass "<<trial<<"\n";
            assert(!pass&&f.world.objects[0].tetrahedra.size()==oldTets.size()&&f.world.objects[0].femReferenceSourceIdentity==oldIdentity);
        }
    }
    {
        Fixture f;auto& o=f.world.objects[0];auto& b=f.bundle;auto& t=b.spatialMassTransportResult;
        const auto mapNode=[](std::uint32_t v){return v==0u?0u:v+4u;};
        for(unsigned v=1;v<5;++v) {
            auto x=o.femNodes[v];x[0]=-x[0];x[1]=-x[1];
            o.femNodes.push_back(x);o.femReferenceNodes.push_back(x);
            b.referenceWorldPoints.push_back(x);b.currentWorldPoints.push_back(x);
            b.parent.referenceWorldPoints.push_back(x);
            auto mass=f.skin.donorMassNodes[v];mass.nodeIndex=42+mapNode(v);mass.localPosition=x;
            f.skin.donorMassNodes.push_back(mass);
            auto source=b.parent.sourceMassContributions[v];source.parentNodeIndex=mapNode(v);
            source.donorLocalPosition=x;b.parent.sourceMassContributions.push_back(source);
        }
        f.skin.donorMassNodes[0].massKg*=2;
        b.parent.sourceMassContributions[0].massKg*=2;
        for(unsigned i=0;i<3;++i) {
            auto cell=o.tetrahedra[i].nodes;for(auto& v:cell)v=mapNode(v);
            o.tetrahedra.push_back({cell});b.childTetrahedra.push_back(cell);b.parent.tetrahedra.push_back(cell);
            const auto mass=o.femTetrahedronMassesKg[i];o.femTetrahedronMassesKg.push_back(mass);
            o.femMaterialIndices.push_back(15);b.childMaterialIndices.push_back(15);
            b.childDonorBodyIndices.push_back(37);b.childAnatomicalRegionIndices.push_back(58);
            t.childMassKg.push_back(mass);t.childParentOffsets.push_back(4+i);
            t.childParentRows.push_back(3+i);t.childParentTransportMassKg.push_back(mass);
            f.skin.tetChildIndex.push_back(3+i);f.skin.tetParentTetrahedron.push_back(3+i);
            f.skin.tetParentSurfaceFace.push_back(7);f.skin.tetParentMaterialIndex.push_back(15);
            f.skin.tetSourceFaceIdentity.push_back(7);
            for(auto v:cell)for(auto parent:cell)
                f.skin.donorMassContributions.push_back({42+v,37,mass*.0625,o.femNodes[parent]});
        }
        auto& l=f.labels;l.sourceTetrahedronCount=6;l.tetrahedronParentTetrahedronIndices={0,1,2,3,4,5};
        l.tetrahedronRegionLabels.resize(6,37);l.tetrahedronBasisRegionLabels.resize(6,0);
        l.tetrahedronSourceFaceIndices.resize(6,7);l.tetrahedronOriginalSourceFaceIndices.resize(6,7);
        l.tetrahedronSourceDonorBodyIndices.resize(6,37);l.tetrahedronSourceParentBodyIndices.resize(6,2);
        l.tetrahedronRegisteredLayerFacetSurfaceFaces.resize(24,NM_INVALID_INDEX);
        l.tetrahedronRegisteredLayerFacetSourceFaceIdentities.resize(24,NM_INVALID_INDEX);
        l.tetrahedronAuthoredHeadSidewallFacetFlags.resize(24,0);
        l.tetrahedronAuthoredHeadSidewallStableFaceIdentities.resize(24,NM_INVALID_INDEX);
        writeProposal(path,f.next,false,true);std::string sha;
        assert(physical_skin_child_mass_detail::sha256File(path,sha,error));
        const double massBefore=std::accumulate(o.femTetrahedronMassesKg.begin(),o.femTetrahedronMassesKg.end(),0.0);
        SkinFatCavityReconditioningReport report;
        const bool pass=reconditionPreparedSkinFatFixedCavities(f.world,o,b,f.skin,f.fat,l,path,sha,
            std::string(64,'a'),report,error);
        if(!pass)std::cerr<<"shared-boundary cavity: "<<error<<"\n";
        assert(pass&&report.cavities==2&&o.tetrahedra.size()==4&&report.affectedNodes==9);
        const double massAfter=std::accumulate(o.femTetrahedronMassesKg.begin(),o.femTetrahedronMassesKg.end(),0.0);
        assert(std::abs(massBefore-massAfter)<1e-20&&report.maximumSourceMassChangeKg<1e-18);
    }
    auto invalid=original.next;std::swap(invalid[0][0],invalid[0][1]);
    assert(!sc::certifyGeometry(original.bundle.referenceWorldPoints,original.bundle.childTetrahedra,
        invalid,edge,ring,before,after,fraction,error));
    auto points=original.bundle.referenceWorldPoints;points[3][2]=INFINITY;
    assert(!sc::certifyGeometry(points,original.bundle.childTetrahedra,
        original.next,edge,ring,before,after,fraction,error));
    std::filesystem::remove(path);
    std::cout<<"skin_fat_fixed_cavity PASS exact_geometry conservative_heterogeneous_mass complete_ancestry frozen_transport_radius shared_boundary_vertices rollback negative_cases=14\n";
}
