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
void writeEntries(const std::filesystem::path&,const std::map<std::string,std::vector<std::uint8_t>>&);
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
    writeEntries(path,entries);
}
void writeEntries(const std::filesystem::path& path,const std::map<std::string,std::vector<std::uint8_t>>& entries) {
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

void writeConnectedProposal(const std::filesystem::path& path,std::span<const sc::Tet> next,
                            std::uint32_t oldCount,int bad=0) {
    std::map<std::string,std::vector<std::uint8_t>> entries;
    std::vector<std::uint32_t> cells,parents,offsets{0u};
    for(const auto& t:next) {
        cells.insert(cells.end(),t.begin(),t.end());
        for(std::uint32_t k=0;k<oldCount;++k)parents.push_back(k);
        offsets.push_back(parents.size());
    }
    if(bad==1)parents.back()=parents.front();
    if(bad==2)offsets[1]=static_cast<std::uint32_t>(parents.size()+1u);
    entries["retained_input_cell_rows"]=npy("<u4","(0,)",{});
    entries["replacement_tetrahedra"]=npy("<u4","("+std::to_string(next.size())+", 4)",cells);
    entries["replacement_tet_offsets"]=npy("<u4","(2,)",{0u,static_cast<std::uint32_t>(next.size())});
    entries["replacement_parent_offsets"]=npy("<u4","("+std::to_string(offsets.size())+",)",offsets);
    entries["replacement_parent_cell_rows"]=npy("<u4","("+std::to_string(parents.size())+",)",parents);
    entries["source_volume_sha256"]=npy("<U64","()",std::vector<std::uint32_t>(64u,'a'));
    writeEntries(path,entries);
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
    Fixture(std::vector<V3> points={},std::vector<sc::Tet> old={}) {
        if(points.empty())points={{0,0,0},{.002,0,0},{0,.002,0},{.001,.0000002,.002},{.001,.0000002,-.002}};
        for(auto& p:points)for(auto& x:p)x=float(x);
        if(old.empty())old={orient(points,{3,4,0,1}),orient(points,{3,4,1,2}),orient(points,{3,4,2,0})};
        const auto count=old.size(),nodes=points.size();
        next={orient(points,{0,1,2,3}),orient(points,{0,2,1,4})};
        bundle.currentWorldPoints=points;bundle.referenceWorldPoints=points;
        bundle.childTetrahedra.assign(old.begin(),old.end());bundle.childMaterialIndices.assign(count,15);
        bundle.childDonorBodyIndices.assign(count,37);bundle.childAnatomicalRegionIndices.assign(count,58);
        bundle.fractionalSemanticMassRouteRequired=true;bundle.massMappingPathSHA256=std::string(64,'b');
        bundle.parent.referenceWorldPoints=points;bundle.parent.tetrahedra.assign(old.begin(),old.end());
        auto& object=world.objects.emplace_back();object.name="fixture";object.mixedFEM=false;object.representation=numi::matter::Representation::fem;
        object.femNodes=points;object.femReferenceNodes=points;object.femMaterialIndices.assign(count,15);
        object.femReferenceSourceIdentity={1,2,3,4};object.femTetrahedronMassSourceIdentity={5,6,7,8};
        world.materials.resize(17);numi::matter::Parameter density;density.name="density";density.lower=900;density.upper=1200;density.defaultValue=1000;
        world.materials[15].parameters.push_back(density);world.materials[16].parameters.push_back(density);
        skin.materialIndex=15;skin.firstGlobalFEMNode=42;fat.materialIndex=16;fat.firstGlobalFEMNode=42;
        skin.tetChildIndex.resize(count);std::iota(skin.tetChildIndex.begin(),skin.tetChildIndex.end(),0u);
        skin.tetParentTetrahedron=skin.tetChildIndex;skin.tetParentSurfaceFace.assign(count,7);
        skin.tetParentMaterialIndex.assign(count,15);skin.tetSourceFaceIdentity.assign(count,7);
        auto& transfer=bundle.spatialMassTransportResult;transfer.usesFractionalSemanticSupport=true;
        bundle.maximumSpatialMassTransportDistanceM=.01;transfer.maximumTransportDistanceM=.01;
        transfer.childParentOffsets.resize(count+1u);std::iota(transfer.childParentOffsets.begin(),transfer.childParentOffsets.end(),0u);
        transfer.childParentRows=skin.tetChildIndex;
        std::vector<double> nodal(nodes,0.0);
        for(unsigned i=0;i<count;++i) {
            const double mass=sc::volume(points,old[i])*(950.0+100.0*(i%3u));
            object.tetrahedra.push_back({old[i]});object.femTetrahedronMassesKg.push_back(mass);
            transfer.childMassKg.push_back(mass);transfer.childParentTransportMassKg.push_back(mass);
            for(auto node:old[i])nodal[node]+=mass*.25;
            for(auto node:old[i])for(auto parentNode:old[i])
                skin.donorMassContributions.push_back({42+node,37,mass*.0625,points[parentNode]});
        }
        for(unsigned node=0;node<nodes;++node) {
            skin.donorMassNodes.push_back({42+node,37,nodal[node],points[node]});
            bundle.parent.sourceMassContributions.push_back({node,15,37,nodal[node],points[node]});
        }
        labels.sourceTetrahedronCount=count;labels.explicitUnpairedVolumeTopology=true;
        labels.tetrahedronParentTetrahedronIndices=skin.tetChildIndex;
        labels.massTransferKind=UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
        labels.tetrahedronRegionLabels.assign(count,37);labels.tetrahedronBasisRegionLabels.assign(count,0);
        labels.tetrahedronSourceFaceIndices.assign(count,7);labels.tetrahedronOriginalSourceFaceIndices.assign(count,7);
        labels.tetrahedronSourceDonorBodyIndices.assign(count,37);labels.tetrahedronSourceParentBodyIndices.assign(count,2);
        labels.tetrahedronRegisteredLayerFacetSurfaceFaces.assign(4u*count,NM_INVALID_INDEX);
        labels.tetrahedronRegisteredLayerFacetSourceFaceIdentities.assign(4u*count,NM_INVALID_INDEX);
        labels.tetrahedronAuthoredHeadSidewallFacetFlags.assign(4u*count,0);
        labels.tetrahedronAuthoredHeadSidewallStableFaceIdentities.assign(4u*count,NM_INVALID_INDEX);
    }
};
}

int geometryProposal(const std::filesystem::path& source,const std::filesystem::path& proposal) {
    namespace npz=current_fractional_support_npz_detail;
    std::string error,sourceSHA,proposalSHA,embedded;
    const auto fatal=[&](const std::string& reason){std::cerr<<"cavity_geometry_input_rejected="<<std::quoted(reason)<<"\n";return 2;};
    std::map<std::string,npz::NpyArray> a,b;
    std::vector<double> xyz;std::vector<std::uint32_t> cells,replacements,patchOffsets,parentOffsets,parentRows;
    if(!npz::readRequestedArrays(source,{"points_reference_world_m_f64","volume_tetrahedra"},a,error)||
       !npz::readF64(a.at("points_reference_world_m_f64"),xyz,error)||
       !npz::readU32(a.at("volume_tetrahedra"),cells,error))return fatal(error);
    if(a.at("points_reference_world_m_f64").shape.size()!=2u||
       a.at("points_reference_world_m_f64").shape[1]!=3u||
       a.at("volume_tetrahedra").shape.size()!=2u||a.at("volume_tetrahedra").shape[1]!=4u)
        return fatal("bad source coordinate/cell domains");
    a.clear();std::vector<V3> points(xyz.size()/3u);
    for(std::size_t i=0;i<xyz.size();++i) {
        const float value=static_cast<float>(xyz[i]);if(!std::isfinite(value))return fatal("nonfinite emitted coordinate");
        points[i/3u][i%3u]=value;
    }
    if(!npz::readRequestedArrays(proposal,{"replacement_tetrahedra","replacement_tet_offsets",
       "replacement_parent_offsets","replacement_parent_cell_rows","source_volume_sha256"},b,error)||
       !npz::readU32(b.at("replacement_tetrahedra"),replacements,error)||
       !npz::readU32(b.at("replacement_tet_offsets"),patchOffsets,error)||
       !npz::readU32(b.at("replacement_parent_offsets"),parentOffsets,error)||
       !npz::readU32(b.at("replacement_parent_cell_rows"),parentRows,error)||
       !npz::readSHA256(b.at("source_volume_sha256"),embedded,error)||
       !physical_skin_child_mass_detail::sha256File(source,sourceSHA,error)||
       !physical_skin_child_mass_detail::sha256File(proposal,proposalSHA,error))return fatal(error);
    if(sourceSHA!=embedded||b.at("replacement_tetrahedra").shape.size()!=2u||
       b.at("replacement_tetrahedra").shape[1]!=4u||replacements.size()%4u||
       patchOffsets.size()<2u||patchOffsets.size()>50001u||patchOffsets.front()!=0u||
       patchOffsets.back()*4ull!=replacements.size()||
       parentOffsets.size()!=replacements.size()/4u+1u||parentOffsets.front()!=0u||
       parentOffsets.back()!=parentRows.size())return fatal("bad proposal identity or offset domains");
    b.clear();unsigned failures=0;std::cout<<"skin_fat_cavity_geometry={\"diagnostic_only\":true,\"source_sha256\":"
        <<std::quoted(sourceSHA)<<",\"proposal_sha256\":"<<std::quoted(proposalSHA)
        <<",\"cavities\":"<<patchOffsets.size()-1u<<",\"failures\":[";
    for(std::size_t i=0;i+1u<patchOffsets.size();++i) {
        std::vector<sc::Tet> old,next;bool valid=true;
        const auto first=patchOffsets[i],last=patchOffsets[i+1u];
        if(first>last||last>replacements.size()/4u||last-first<2u||last-first>64u)
            return fatal("bad bounded patch offsets");
        std::vector<std::uint32_t> rows;
        for(auto j=first;j<last;++j) {
            const auto lo=parentOffsets[j],hi=parentOffsets[j+1u];
            if(lo>hi||hi>parentRows.size()||hi-lo<2u||hi-lo>32u)return fatal("bad bounded ancestry offsets");
            if(j==first)rows.assign(parentRows.begin()+lo,parentRows.begin()+hi);
            else if(hi-lo!=rows.size()||!std::equal(rows.begin(),rows.end(),parentRows.begin()+lo))
                return fatal("inconsistent complete patch ancestry");
            sc::Tet t;std::copy_n(replacements.begin()+4ull*j,4u,t.begin());next.push_back(t);
        }
        for(auto row:rows) {
            if(row>=cells.size()/4u)return fatal("old cell out of range");
            sc::Tet t;std::copy_n(cells.begin()+4ull*row,4u,t.begin());
            if(*std::max_element(t.begin(),t.end())>=points.size())return fatal("old vertex out of range");
            old.push_back(t);
        }
        for(const auto& t:next)if(*std::max_element(t.begin(),t.end())>=points.size())
            return fatal("new vertex out of range");
        double before,after;std::vector<double> fractions;
        if(old.size()==3u&&next.size()==2u) {
            sc::Edge edge;sc::Triangle ring;double fraction;
            valid=sc::certifyGeometry(points,old,{next[0],next[1]},edge,ring,before,after,fraction,error);
        } else valid=sc::certifyConnectedGeometry(points,old,next,before,after,fractions,error);
        if(!valid) {if(failures++)std::cout<<",";std::cout<<"{\"cavity\":"<<i<<",\"reason\":"<<std::quoted(error)<<"}";}
    }
    std::cout<<"],\"failure_count\":"<<failures<<",\"mass_qualified\":false,\"native_steps\":0}\n";
    return failures?1:0;
}
int main(int argc,char** argv) {
    if(argc==4&&std::string_view(argv[1])=="--geometry-proposal")return geometryProposal(argv[2],argv[3]);
    if(argc!=1){std::cerr<<"usage: "<<argv[0]<<" [--geometry-proposal source-volume.npz proposal.npz]\n";return 2;}

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

    {
        const std::map<std::uint32_t,double> source{{7,.4},{11,.6}};
        std::vector<std::map<std::uint32_t,double>> flows;
        const std::vector<double> demand{.5,.5},distance{0.,2.,0.,0.};
        assert(sc::routeRestrictedParentMass(source,demand,distance,1.,flows,error));
        assert(flows.size()==2&&std::abs(flows[0].at(7)-.4)<1e-15&&
               std::abs(flows[0].at(11)-.1)<1e-15&&std::abs(flows[1].at(11)-.5)<1e-15);
        const auto previous=flows;
        assert(!sc::routeRestrictedParentMass(source,demand,std::vector<double>{0,2,2,0},1.,flows,error));
        assert(flows==previous); // Infeasibility must not publish partial source flow.
        assert(!sc::routeRestrictedParentMass(source,demand,std::vector<double>{0,2,NAN,0},1.,flows,error));
        assert(!sc::routeRestrictedParentMass(source,std::vector<double>{.5,.6},distance,1.,flows,error));
        assert(flows==previous);
    }
    auto invalid=original.next;std::swap(invalid[0][0],invalid[0][1]);
    assert(!sc::certifyGeometry(original.bundle.referenceWorldPoints,original.bundle.childTetrahedra,
        invalid,edge,ring,before,after,fraction,error));
    auto points=original.bundle.referenceWorldPoints;points[3][2]=INFINITY;
    assert(!sc::certifyGeometry(points,original.bundle.childTetrahedra,
        original.next,edge,ring,before,after,fraction,error));

    {
        // Exact emitted Human fat cavity: the old sliver stalled root 20 at J=2.5.
        const std::vector<V3> p={{-0.0067697777412831783,-0.14630590379238129,0.036408413201570511},{-0.0067686568945646286,-0.14619013667106628,0.036485228687524796},{-0.0067681982181966305,-0.14613674581050873,0.036564752459526062},{-0.0063064750283956528,-0.14636923372745514,0.036472655832767487},{-0.0063050007447600365,-0.14623261988162994,0.036499418318271637},{-0.006303825881332159,-0.14611682295799255,0.036576222628355026},{-0.0063033462502062321,-0.146063432097435,0.036655742675065994},{-0.0067642475478351116,-0.14656788110733032,0.036415576934814453}};
        const std::vector<sc::Tet> old={{2,1,5,4},{2,4,5,6},{0,5,1,4},{7,0,2,1},{7,1,4,0},{7,0,4,3},{7,2,4,1}},good={{0,1,2,5},{0,2,4,5},{2,0,4,7},{0,3,4,7},{2,4,5,6}};
        double b,a;std::vector<double> weights;
        assert(sc::certifyConnectedGeometry(p,old,good,b,a,weights,error));
        assert(a>.015&&b<.000013&&weights.size()==5);
        for(int trial=0;trial<11;++trial) {
            Fixture f(p,old);auto next=good;
            if(trial==3)next.pop_back();
            if(trial==4)next[1]=next[0];
            if(trial==5)std::swap(next[0][0],next[0][1]);
            if(trial==6)next[0][0]=999999u;
            writeConnectedProposal(path,next,old.size(),trial==1?1:trial==2?2:0);
            if(trial==7)f.world.objects[0].femMaterialIndices[0]=16;
            if(trial==8)f.skin.donorMassNodes[0].massKg*=1.1;
            if(trial==9)f.bundle.maximumSpatialMassTransportDistanceM=
                f.bundle.spatialMassTransportResult.maximumTransportDistanceM=1e-9;
            if(trial==10)f.bundle.childAnatomicalRegionIndices[0]=99;
            std::string sha;assert(physical_skin_child_mass_detail::sha256File(path,sha,error));
            const auto initial=f.world.objects[0];const auto initialLabels=f.labels;
            SkinFatCavityReconditioningReport report;
            const bool pass=reconditionPreparedSkinFatFixedCavities(f.world,f.world.objects[0],f.bundle,
                f.skin,f.fat,f.labels,path,sha,std::string(64,'a'),report,error);
            if(trial==0) {
                if(!pass)std::cerr<<"connected cavity: "<<error<<"\n";
                assert(pass&&report.inputCells==7&&report.outputCells==5&&report.cavities==1);
                const auto& o=f.world.objects[0];
                assert(o.femNodes==initial.femNodes&&o.femReferenceNodes==initial.femReferenceNodes);
                const auto total=[](const auto& mass){return std::accumulate(mass.begin(),mass.end(),0.0);};
                assert(std::abs(total(o.femTetrahedronMassesKg)-total(initial.femTetrahedronMassesKg))<1e-24);
                assert(report.maximumSourceMassChangeKg<1e-12*total(initial.femTetrahedronMassesKg)&&
                       report.maximumSourceMomentChangeKgM<1e-12*total(initial.femTetrahedronMassesKg));
                assert(report.minimumQualityAfter>.015&&f.labels.sourceTetrahedronCount==5);
            } else {
                if(pass)std::cerr<<"unexpected connected pass "<<trial<<"\n";
                assert(!pass&&f.world.objects[0].femTetrahedronMassesKg==initial.femTetrahedronMassesKg);
                assert(f.world.objects[0].femReferenceSourceIdentity==initial.femReferenceSourceIdentity);
                assert(f.labels.sourceTetrahedronCount==initialLabels.sourceTetrahedronCount);
            }
        }

        {
            Fixture f(p,old);
            const double total=std::accumulate(f.world.objects[0].femTetrahedronMassesKg.begin(),
                f.world.objects[0].femTetrahedronMassesKg.end(),0.0);
            std::vector<double> distances,demand;double assigned=0;
            for(std::size_t j=0;j<weights.size();++j) {
                const double value=j+1u==weights.size()?total-assigned:weights[j]*total;
                demand.push_back(value);assigned+=value;
            }
            for(const auto& t:old) {
                V3 parent{};for(auto v:t)for(unsigned k=0;k<3u;++k)parent[k]+=.25*p[v][k];
                for(const auto& next:good) {
                    V3 child{};for(auto v:next)for(unsigned k=0;k<3u;++k)child[k]+=.25*p[v][k];
                    double d2=0;for(unsigned k=0;k<3u;++k){const double d=child[k]-parent[k];d2+=d*d;}
                    distances.push_back(std::sqrt(d2));
                }
            }
            auto candidates=distances;std::sort(candidates.begin(),candidates.end());
            std::map<std::uint32_t,double> source;
            for(std::uint32_t j=0;j<old.size();++j)source[j]=f.world.objects[0].femTetrahedronMassesKg[j];
            std::vector<std::map<std::uint32_t,double>> routed;double radius=0;
            for(auto r:candidates) {
                if(r>=candidates.back())break;
                if(sc::routeRestrictedParentMass(source,demand,distances,r,routed,error)){radius=r;break;}
            }
            assert(radius>0&&radius<candidates.back());
            f.bundle.maximumSpatialMassTransportDistanceM=
                f.bundle.spatialMassTransportResult.maximumTransportDistanceM=radius;
            writeConnectedProposal(path,good,old.size());std::string sha;
            assert(physical_skin_child_mass_detail::sha256File(path,sha,error));
            SkinFatCavityReconditioningReport report;
            const bool pass=reconditionPreparedSkinFatFixedCavities(f.world,f.world.objects[0],f.bundle,
                f.skin,f.fat,f.labels,path,sha,std::string(64,'a'),report,error);
            if(!pass)std::cerr<<"restricted connected mass: "<<error<<"\n";
            assert(pass&&report.restrictedTransportCavities==1&&report.maximumTransportDistanceUsedM<=radius);
            assert(report.maximumSourceMassChangeKg<=1e-12*total);
        }
        auto overlap=good;overlap.push_back(good[0]);
        assert(!sc::certifyConnectedGeometry(p,old,overlap,b,a,weights,error));
        std::vector<sc::Tet> overCapacity(65u,good[0]);
        assert(!sc::certifyConnectedGeometry(p,old,overCapacity,b,a,weights,error));
    }

    std::filesystem::remove(path);
    std::cout<<"skin_fat_fixed_cavity PASS exact_geometry conservative_heterogeneous_mass complete_ancestry frozen_transport_radius shared_boundary_vertices rollback connected_7_to_5 fixed_boundary_float32_mass_ancestry restricted_source_routing infeasible_routing_rollback negative_cases=29\n";
}
