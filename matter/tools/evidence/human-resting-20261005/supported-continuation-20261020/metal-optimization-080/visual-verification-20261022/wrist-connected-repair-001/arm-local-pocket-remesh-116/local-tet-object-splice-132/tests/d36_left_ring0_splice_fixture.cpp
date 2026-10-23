#include "NumiHumanD36LeftRing0LocalTetSplice.hpp"
#include "NumiHumanWholeBodySkinReduction.hpp"
#include "NumiHumanWholeBodySkinTransferAncestry.hpp"
#include "metalrobo/EngineModel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <array>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
using V3 = std::array<double,3>;

struct Csv {
    std::vector<std::string> header;
    std::unordered_map<std::string,std::size_t> index;
    std::vector<std::vector<std::string>> rows;
};

static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream in(line);
    std::string item;
    while(std::getline(in,item,',')) out.push_back(item);
    if(!line.empty()&&line.back()==',') out.emplace_back();
    return out;
}
static Csv readCsv(const std::string& path) {
    std::ifstream file(path);
    if(!file) throw std::runtime_error("cannot open CSV: "+path);
    Csv out;
    std::string line;
    if(!std::getline(file,line)) throw std::runtime_error("empty CSV: "+path);
    out.header=split(line);
    for(std::size_t i=0;i<out.header.size();++i) out.index.emplace(out.header[i],i);
    while(std::getline(file,line)) {
        if(line.empty()) continue;
        auto row=split(line);
        if(row.size()!=out.header.size()) throw std::runtime_error("CSV row has wrong width: "+path);
        out.rows.push_back(std::move(row));
    }
    return out;
}
static double number(const Csv& csv,const std::vector<std::string>& row,const std::string& key) {
    const auto it=csv.index.find(key);
    if(it==csv.index.end()) throw std::runtime_error("missing CSV column: "+key);
    return std::stod(row[it->second]);
}
static std::uint32_t uintNumber(const Csv& csv,const std::vector<std::string>& row,const std::string& key) {
    const double value=number(csv,row,key);
    if(value<0.0||value>static_cast<double>(UINT32_MAX)||std::floor(value)!=value)
        throw std::runtime_error("invalid uint CSV value: "+key);
    return static_cast<std::uint32_t>(value);
}
static V3 vec(const Csv& c,const std::vector<std::string>& r,const char* x,const char* y,const char* z) {
    return {number(c,r,x),number(c,r,y),number(c,r,z)};
}
static V3 matVec(const std::array<float,9>& m,const V3& v) {
    return {double(m[0])*v[0]+double(m[1])*v[1]+double(m[2])*v[2],
            double(m[3])*v[0]+double(m[4])*v[1]+double(m[5])*v[2],
            double(m[6])*v[0]+double(m[7])*v[1]+double(m[8])*v[2]};
}
static double volume6(const V3& a,const V3& b,const V3& c,const V3& d) {
    const auto u=sub(b,a),v=sub(c,a),w=sub(d,a);
    return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0]);
}
static void require(bool ok,const std::string& what) { if(!ok) throw std::runtime_error(what); }
#pragma pack(push,1)
struct RigidHeader {
    std::array<char,8> magic{};
    std::uint32_t payloadAbi=0,engineAbi=0,sourceBodyCount=0,engineBodyCount=0;
    std::uint32_t jointCount=0,nq=0,nv=0,rootBodyIndex=0,virtualBodyCount=0,reserved0=0;
    std::array<std::uint8_t,32> sourceSha256{};
};
#pragma pack(pop)
static_assert(sizeof(RigidHeader)==80);
static std::vector<std::uint32_t> readSourceParents(const std::string& path) {
    std::ifstream in(path,std::ios::binary); if(!in) throw std::runtime_error("cannot open pinned NHRIGID");
    RigidHeader h{}; in.read(reinterpret_cast<char*>(&h),sizeof(h));
    const std::array<char,8> magic{'N','H','R','I','G','I','D','2'};
    require(bool(in)&&h.magic==magic&&h.payloadAbi==1u&&h.sourceBodyCount>0u&&
            h.engineBodyCount>=h.sourceBodyCount&&h.jointCount+1u==h.engineBodyCount,
            "pinned D36 NHRIGID header mismatch");
    in.seekg(static_cast<std::streamoff>(sizeof(MRWorldGPU)+sizeof(MRArticulationGPU)),std::ios::cur);
    std::vector<MRBodyPropertiesGPU> bodies(h.engineBodyCount);
    in.read(reinterpret_cast<char*>(bodies.data()),static_cast<std::streamsize>(bodies.size()*sizeof(bodies[0])));
    require(bool(in),"cannot read NHRIGID body table");
    const auto skip=[&](std::uint64_t n) { in.seekg(static_cast<std::streamoff>(n),std::ios::cur); require(bool(in),"truncated NHRIGID arrays"); };
    skip(std::uint64_t(h.jointCount)*sizeof(MRJointDescriptorGPU));
    skip(std::uint64_t(h.nv)*sizeof(MRDofPropertiesGPU));
    skip(std::uint64_t(h.nq)*sizeof(float)); skip(std::uint64_t(h.nv)*sizeof(float));
    std::vector<std::uint32_t> sourceToCore(h.sourceBodyCount);
    in.read(reinterpret_cast<char*>(sourceToCore.data()),static_cast<std::streamsize>(sourceToCore.size()*sizeof(sourceToCore[0])));
    require(bool(in),"cannot read NHRIGID source-to-core table");
    std::vector<std::uint32_t> parents(h.engineBodyCount,NM_INVALID_INDEX);
    for(std::uint32_t core=0;core<h.engineBodyCount;++core) {
        const auto parent=bodies[core].parentBody;
        if(parent<h.engineBodyCount) parents[core]=parent;
    }
    return parents;
}

int main(int argc,char** argv) try {
    if(argc!=2) throw std::runtime_error("usage: d36_left_ring0_splice_fixture <D36-generated-dir>");
    const std::string g=argv[1];
    const auto nodeCsv=readCsv(g+"/nodes.csv");
    const auto tetCsv=readCsv(g+"/tetrahedra.csv");
    const auto frameCsv=readCsv(g+"/donor-frames-fullweights.csv");
    const auto sourceParents=readSourceParents("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/joint-functional-owner-audit-20261023/knee-projection-envelope-001/myosim-fullbody-core-reference.nhrigid");
    constexpr std::size_t N=kWholeBodySkinSourceVertexCount;
    constexpr std::size_t T=kElbowRefinementChildTetCount;
    require(nodeCsv.rows.size()==N,"D36 node CSV must contain one exact O/I/D record per surface row");
    require(tetCsv.rows.size()==T,"D36 tet CSV must contain exact current topology");
    require(frameCsv.rows.size()==N,"D36 donor frame CSV must contain one record per surface row");

    numi::matter::ObjectSource object;
    object.name="D36 actual whole-body skin-fat source fixture";
    object.representation=numi::matter::Representation::fem;
    object.femNodes.reserve(3u*N);
    for(std::size_t i=0;i<nodeCsv.rows.size();++i) {
        require(uintNumber(nodeCsv,nodeCsv.rows[i],"row")==i,"D36 node CSV order changed");
        object.femNodes.push_back(vec(nodeCsv,nodeCsv.rows[i],"ox","oy","oz"));
    }
    for(const auto& r:nodeCsv.rows) object.femNodes.push_back(vec(nodeCsv,r,"ix","iy","iz"));
    for(const auto& r:nodeCsv.rows) object.femNodes.push_back(vec(nodeCsv,r,"dx","dy","dz"));
    object.tetrahedra.reserve(T);
    object.femMaterialIndices.reserve(T);
    for(std::size_t i=0;i<T;++i) {
        const auto& r=tetCsv.rows[i];
        require(uintNumber(tetCsv,r,"tet")==i,"D36 tet CSV order changed");
        object.tetrahedra.push_back({{uintNumber(tetCsv,r,"a"),uintNumber(tetCsv,r,"b"),
                                      uintNumber(tetCsv,r,"c"),uintNumber(tetCsv,r,"d")}});
        object.femMaterialIndices.push_back(uintNumber(tetCsv,r,"material"));
    }

    RegisteredSurface surface;
    surface.name="D36 exact registered whole-body skin surface";
    surface.provider="D36 frozen production reduction tables";
    surface.sourceMember="wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001";
    surface.sourceMeshSHA256=std::string(kWholeBodySkinMechanicalCSV_SHA256);
    surface.sourcePayloadSHA256=std::string(kWholeBodySkinSourcePayloadSHA256);
    surface.aggregateIdentity=true;
    surface.sourceBodyIndex=NM_INVALID_INDEX;
    surface.worldPoints.reserve(N); surface.donorLocalPoints.reserve(N);
    surface.sourceVertexIndices.assign(kWholeBodySkinSourceVertices.begin(),kWholeBodySkinSourceVertices.end());
    surface.sourceFaceIndices.assign(kWholeBodySkinSourceFaces.begin(),kWholeBodySkinSourceFaces.end());
    surface.faces.assign(kWholeBodySkinFaces.begin(),kWholeBodySkinFaces.end());
    surface.sourceEmbeddingFaceIndices.assign(kWholeBodySkinSourceEmbeddingFaces.begin(),kWholeBodySkinSourceEmbeddingFaces.end());
    surface.sourceEmbeddingBarycentric.assign(kWholeBodySkinSourceEmbeddingBarycentric.begin(),kWholeBodySkinSourceEmbeddingBarycentric.end());
    surface.sourceEmbeddingResidualWorldM.assign(kWholeBodySkinSourceEmbeddingResidualWorldM.begin(),kWholeBodySkinSourceEmbeddingResidualWorldM.end());
    surface.donorBodyIndices.reserve(N); surface.worldToDonorLocalLinear.reserve(N);
    surface.outwardWorldNormals.reserve(N);
    std::vector<V3> localByObjectNode(3u*N);
    for(std::size_t v=0;v<N;++v) {
        const auto& csv=frameCsv.rows[v];
        require(uintNumber(frameCsv,csv,"row")==v,"D36 donor frame row order changed");
        const auto& frame=kWholeBodySkinDonorFrames[v];
        surface.worldPoints.push_back(object.femNodes[v]);
        surface.donorLocalPoints.push_back({double(frame.donorLocalPoint[0]),double(frame.donorLocalPoint[1]),double(frame.donorLocalPoint[2])});
        surface.donorBodyIndices.push_back(frame.donorBodyIndex);
        surface.worldToDonorLocalLinear.push_back({double(frame.worldToDonorLocalLinear[0]),double(frame.worldToDonorLocalLinear[1]),double(frame.worldToDonorLocalLinear[2]),
            double(frame.worldToDonorLocalLinear[3]),double(frame.worldToDonorLocalLinear[4]),double(frame.worldToDonorLocalLinear[5]),
            double(frame.worldToDonorLocalLinear[6]),double(frame.worldToDonorLocalLinear[7]),double(frame.worldToDonorLocalLinear[8])});
        surface.outwardWorldNormals.push_back({double(frame.registeredWorldNormal[0]),double(frame.registeredWorldNormal[1]),double(frame.registeredWorldNormal[2])});
        for(std::size_t layer=0;layer<3u;++layer) {
            const V3 world=object.femNodes[layer*N+v];
            const V3 delta=sub(world,object.femNodes[v]);
            localByObjectNode[layer*N+v]=add(surface.donorLocalPoints.back(),matVec(frame.worldToDonorLocalLinear,delta));
        }
    }
    require(surface.faces.size()==kWholeBodySkinFaceCount,"D36 face table size changed");

    std::vector<V3> parentWorldPoints;
    std::vector<std::uint32_t> parentMaterial,childMaterial;
    parentWorldPoints.reserve(3u*kElbowRefinementParentVertexCount);
    for(std::size_t layer=0;layer<3u;++layer)
        for(std::size_t v=0;v<kElbowRefinementParentVertexCount;++v)
            parentWorldPoints.push_back(object.femNodes[layer*N+v]);
    parentMaterial.reserve(kElbowRefinementParentTetCount);
    childMaterial.reserve(T);
    for(auto code:kElbowRefinementParentMaterialCodes) {
        require(code==1u||code==2u,"unexpected parent material code");
        parentMaterial.push_back(code);
    }
    for(auto code:kElbowRefinementChildMaterialCodes) {
        require(code==1u||code==2u,"unexpected child material code");
        childMaterial.push_back(code);
    }
    ExplicitPairedLayerTopology topology{
        std::span<const std::array<std::uint32_t,4>>(kElbowRefinementChildTetrahedra),
        std::span<const std::uint32_t>(childMaterial),
        std::span<const std::uint32_t>(kElbowRefinementChildParentTetrahedra),
        std::span<const std::array<std::uint32_t,4>>(kElbowRefinementParentTetrahedra),
        std::span<const V3>(parentWorldPoints),
        std::span<const std::uint32_t>(parentMaterial),
        std::span<const std::uint32_t>(kElbowRefinementChildParentSurfaceFaces),
        std::span<const std::uint32_t>(kElbowRefinementChildSourceFaceIdentities)};

    RegisteredSkinFatReductionSourceLabels labels;
    std::string error;
    std::map<std::uint32_t,std::uint32_t> directDonor;
    for(std::size_t v=0;v<N;++v) {
        const auto source=surface.sourceVertexIndices[v];
        const auto donor=surface.donorBodyIndices[v];
        if(source!=NM_INVALID_INDEX) {
            auto [it,inserted]=directDonor.emplace(source,donor);
            require(inserted||it->second==donor,"one source vertex has conflicting D36 donor labels");
        }
    }
    const auto verifiesBound=[&](std::uint32_t source,std::uint32_t donor) {
        const auto it=directDonor.find(source); return it!=directDonor.end()&&it->second==donor;
    };
    const auto verifiesEmbedded=[&](std::uint32_t row,std::uint32_t donor) {
        return row<N&&surface.sourceEmbeddingFaceIndices[row]!=NM_INVALID_INDEX&&surface.donorBodyIndices[row]==donor;
    };
    const auto parentBodyAt=[&](std::uint32_t body) { return body<sourceParents.size()?sourceParents[body]:NM_INVALID_INDEX; };
    require(buildRegisteredSkinFatReductionSourceLabels(
        surface,object,static_cast<std::uint32_t>(N),static_cast<std::uint32_t>(sourceParents.size()),verifiesBound,parentBodyAt,
        labels,error,36u,std::span<const std::uint32_t>{},0u,verifiesEmbedded,
        std::span<const std::uint32_t>(kWholeBodySkinMechanicalFaceIdentities),
        &topology,1u,2u,false),"actual D36 label builder rejected fixture: "+error);

    PreparedOrganicFEM skin,fat;
    skin.materialIndex=1u; fat.materialIndex=2u;
    skin.firstGlobalFEMNode=0u; fat.firstGlobalFEMNode=0u;
    skin.densityKgM3=1000.0; fat.densityKgM3=900.0;
    std::array<double,2> volume{};
    std::map<std::uint32_t,metalrobo::NumiHumanTissueMassNode> massRows;
    for(std::size_t ti=0;ti<T;++ti) {
        const auto material=object.femMaterialIndices[ti];
        require(material==1u||material==2u,"unknown D36 material");
        const auto t=object.tetrahedra[ti].nodes;
        const double vol=volume6(object.femNodes[t[0]],object.femNodes[t[1]],object.femNodes[t[2]],object.femNodes[t[3]])/6.0;
        require(std::isfinite(vol)&&vol>0.0,"D36 source contains nonpositive reference cell");
        const std::size_t mi=material==1u?0u:1u;
        const double rho=mi==0u?skin.densityKgM3:fat.densityKgM3;
        volume[mi]+=vol;
        for(const auto node:t) {
            const auto body=surface.donorBodyIndices[node%N];
            auto [it,inserted]=massRows.emplace(node,metalrobo::NumiHumanTissueMassNode{.nodeIndex=node,.donorBody=body,.massKg=rho*vol/4.0,.localPosition=localByObjectNode[node]});
            if(!inserted) {
                require(it->second.donorBody==body,"D36 shared node has inconsistent donor ownership");
                require(norm(sub(it->second.localPosition,localByObjectNode[node]))<1e-12,"D36 shared node donor frame differs by material");
                it->second.massKg+=rho*vol/4.0;
            }
        }
    }
    skin.volumeM3=volume[0]; fat.volumeM3=volume[1];
    skin.massDebitKg=skin.densityKgM3*skin.volumeM3; fat.massDebitKg=fat.densityKgM3*fat.volumeM3;
    for(const auto& [id,row]:massRows) { (void)id; skin.donorMassNodes.push_back(row); }
    require(skin.donorMassNodes.size()==object.femNodes.size(),"D36 source mass does not cover every ObjectSource node");

    d36_left_ring0::Storage storage;
    require(storage.prepare(object,skin,fat,labels,error),"ring0 data adapter rejected exact D36 ObjectSource: "+error);
    const auto input=storage.input(object,skin,fat);

    // Fail-closed transaction checks against the exact same D36 source object.
    // Every rejection must leave the ObjectSource, both prepared reports, and
    // caller-owned report/label outputs unchanged.
    const auto sameObjectCore=[](const numi::matter::ObjectSource& a,const numi::matter::ObjectSource& b) {
        if(a.femNodes!=b.femNodes||a.femReferenceNodes!=b.femReferenceNodes||
           a.femMaterialIndices!=b.femMaterialIndices||a.femPreconditionerRegionIds!=b.femPreconditionerRegionIds||a.femContactNodes!=b.femContactNodes||
           a.femDisabledContactSurfaceFaces!=b.femDisabledContactSurfaceFaces||
           a.femMaterialFrameRotations!=b.femMaterialFrameRotations||
           a.femMaterialSourceIdentity!=b.femMaterialSourceIdentity||
           a.femMaterialFrameSourceIdentity!=b.femMaterialFrameSourceIdentity||
           a.femReferenceSourceIdentity!=b.femReferenceSourceIdentity||
           a.femCapacity.nodes!=b.femCapacity.nodes||a.femCapacity.tetrahedra!=b.femCapacity.tetrahedra)
            return false;
        if(a.tetrahedra.size()!=b.tetrahedra.size())return false;
        for(std::size_t i=0;i<a.tetrahedra.size();++i)if(a.tetrahedra[i].nodes!=b.tetrahedra[i].nodes)return false;
        return true;
    };
    const auto sameMassRows=[](const auto& a,const auto& b) {
        if(a.size()!=b.size())return false;
        for(std::size_t i=0;i<a.size();++i)if(a[i].nodeIndex!=b[i].nodeIndex||a[i].donorBody!=b[i].donorBody||
            a[i].massKg!=b[i].massKg||a[i].localPosition!=b[i].localPosition)return false;
        return true;
    };
    auto expectRejected=[&](const char* tag,LocalSkinFatTetSpliceInput badInput,auto mutateObject) {
        auto badObject=object; auto badSkin=skin; auto badFat=fat; auto badLabels=labels;
        mutateObject(badObject);
        const auto objectAtCall=badObject;
        LocalSkinFatTetSpliceReport outReport; outReport.originalNodeCount=777u; outReport.originalTetCount=888u;
        outReport.skinVolumeBeforeM3=9.5; outReport.oldTetToNewTet={5u,7u};
        RegisteredSkinFatReductionSourceLabels outLabels; outLabels.sourceTetrahedronCount=321u;
        outLabels.sourceNodeDonorBodyIndices={17u}; outLabels.tetrahedronSourceFaceIndices={23u};
        std::string rejectError;
        const bool accepted=spliceRegisteredSkinFatLocalTets(badObject,badSkin,badFat,badLabels,badInput,outReport,outLabels,rejectError);
        require(!accepted,std::string("negative transaction unexpectedly accepted: ")+tag);
        require(sameObjectCore(badObject,objectAtCall),std::string("rejected splice mutated ObjectSource: ")+tag);
        require(badSkin.materialIndex==skin.materialIndex&&badSkin.firstGlobalFEMNode==skin.firstGlobalFEMNode&&
                badSkin.densityKgM3==skin.densityKgM3&&badSkin.volumeM3==skin.volumeM3&&badSkin.massDebitKg==skin.massDebitKg&&
                sameMassRows(badSkin.donorMassNodes,skin.donorMassNodes),std::string("rejected splice mutated skin report: ")+tag);
        require(badFat.materialIndex==fat.materialIndex&&badFat.firstGlobalFEMNode==fat.firstGlobalFEMNode&&
                badFat.densityKgM3==fat.densityKgM3&&badFat.volumeM3==fat.volumeM3&&badFat.massDebitKg==fat.massDebitKg&&
                sameMassRows(badFat.donorMassNodes,fat.donorMassNodes),std::string("rejected splice mutated fat report: ")+tag);
        require(outReport.originalNodeCount==777u&&outReport.originalTetCount==888u&&outReport.skinVolumeBeforeM3==9.5&&
                outReport.oldTetToNewTet==std::vector<std::uint32_t>{5u,7u},std::string("rejected splice mutated report output: ")+tag);
        require(outLabels.sourceTetrahedronCount==321u&&outLabels.sourceNodeDonorBodyIndices==std::vector<std::uint32_t>{17u}&&
                outLabels.tetrahedronSourceFaceIndices==std::vector<std::uint32_t>{23u},std::string("rejected splice mutated label output: ")+tag);
        require(!rejectError.empty(),std::string("rejected splice omitted diagnostic: ")+tag);
    };
    const auto noObjectChange=[](auto&){};
    {
        auto bad=input; bad.childMaterialIndices=std::span<const std::uint32_t>{};
        expectRejected("missing material metadata",bad,noObjectChange);
    }
    {
        auto bad=input; bad.childParentTetRows=std::span<const std::uint32_t>{};
        expectRejected("missing parent ancestry",bad,noObjectChange);
    }
    {
        std::vector<LocalSkinFatTetNode> changed(storage.nodes);
        const auto it=std::find_if(changed.begin(),changed.end(),[](const auto& n){return n.existingObjectNode!=NM_INVALID_INDEX;});
        require(it!=changed.end(),"fixture has no retained node for transaction test");
        it->world[0]+=1.0e-3;
        auto bad=input; bad.nodes=changed;
        expectRejected("changed retained node",bad,noObjectChange);
    }
    {
        std::vector<std::array<std::uint32_t,4>> t(input.childTetrahedra.begin(),input.childTetrahedra.end());
        std::vector<std::uint32_t> mats(input.childMaterialIndices.begin(),input.childMaterialIndices.end());
        std::vector<std::uint32_t> parents(input.childParentTetRows.begin(),input.childParentTetRows.end());
        std::vector<std::uint32_t> faces(input.childParentSurfaceFaceRows.begin(),input.childParentSurfaceFaceRows.end());
        std::vector<std::uint32_t> ids(input.childSourceFaceIdentities.begin(),input.childSourceFaceIdentities.end());
        std::vector<std::uint32_t> donors(input.childDonorBodyLabels.begin(),input.childDonorBodyLabels.end());
        std::vector<std::uint32_t> bases(input.childBasisRegionLabels.begin(),input.childBasisRegionLabels.end());
        t.pop_back(); mats.pop_back(); parents.pop_back(); faces.pop_back(); ids.pop_back(); donors.pop_back(); bases.pop_back();
        auto bad=input; bad.childTetrahedra=t; bad.childMaterialIndices=mats; bad.childParentTetRows=parents;
        bad.childParentSurfaceFaceRows=faces; bad.childSourceFaceIdentities=ids; bad.childDonorBodyLabels=donors; bad.childBasisRegionLabels=bases;
        expectRejected("open cavity",bad,noObjectChange);
    }
    {
        std::vector<std::uint32_t> donors(input.childDonorBodyLabels.begin(),input.childDonorBodyLabels.end());
        donors[0]=donors[0]==0u?1u:0u;
        auto bad=input; bad.childDonorBodyLabels=donors;
        expectRejected("donor mismatch",bad,noObjectChange);
    }
    {
        auto bad=input;
        expectRejected("insufficient authored tet capacity",bad,[&](auto& candidate){
            candidate.femCapacity.tetrahedra=static_cast<std::uint32_t>(object.tetrahedra.size());
        });
    }
    {
        const auto replaced=d36_left_ring0::kReplacedTetRows.front();
        auto bad=input;
        expectRejected("disabled contact face on replaced tet",bad,[&](auto& candidate){
            candidate.femDisabledContactSurfaceFaces.push_back(4u*replaced);
        });
    }

    const auto oldNodes=object.femNodes;
    const auto oldTets=object.tetrahedra;
    const auto oldMaterials=object.femMaterialIndices;
    LocalSkinFatTetSpliceReport report;
    RegisteredSkinFatReductionSourceLabels outputLabels;
    require(spliceRegisteredSkinFatLocalTets(object,skin,fat,labels,input,report,outputLabels,error),
            "actual D36 ring0 local splice failed: "+error);
    require(object.femNodes==oldNodes,"ring0 changed an existing D36 node");
    require(object.tetrahedra.size()==oldTets.size()-520u+578u,"ring0 output tet count is wrong");
    require(object.femMaterialIndices.size()==object.tetrahedra.size(),"ring0 output material array is incomplete");
    require(outputLabels.sourceTetrahedronCount==object.tetrahedra.size(),"ring0 labels do not cover output tets");
    require(outputLabels.sourceNodeDonorBodyIndices==labels.sourceNodeDonorBodyIndices,"ring0 changed retained node donor labels");
    require(skin.donorMassNodes.size()==object.femNodes.size()&&fat.donorMassNodes.empty(),"ring0 broke unique aggregate mass-node ownership");
    const double materialVolumeClosureError=std::max(
        std::abs(report.skinVolumeBeforeM3-report.skinVolumeAfterM3),
        std::abs(report.fatVolumeBeforeM3-report.fatVolumeAfterM3));
    require(std::abs(report.skinVolumeBeforeM3-report.skinVolumeAfterM3)<=1.0e-6*report.skinVolumeBeforeM3&&
            std::abs(report.fatVolumeBeforeM3-report.fatVolumeAfterM3)<=1.0e-6*report.fatVolumeBeforeM3,
            "ring0 material volume exceeds the explicit owner closure tolerance");
    require(report.aggregateDonorMassNodes.size()==object.femNodes.size(),"ring0 aggregate mass map lost physical nodes");
    std::size_t retained=0u;
    for(std::size_t old=0;old<oldTets.size();++old) {
        const bool selected=std::binary_search(d36_left_ring0::kReplacedTetRows.begin(),d36_left_ring0::kReplacedTetRows.end(),static_cast<std::uint32_t>(old));
        if(!selected) {
            const auto now=report.oldTetToNewTet[old];
            require(now<object.tetrahedra.size(),"ring0 omitted an unmodified old tet");
            require(object.tetrahedra[now].nodes==oldTets[old].nodes&&object.femMaterialIndices[now]==oldMaterials[old],
                    "ring0 changed an outside parent tet or material");
            ++retained;
        }
    }
    require(retained==oldTets.size()-520u,"ring0 retained tet count mismatch");
    for(const auto& [body,dm]:report.donorMassDeltaKg) {
        require(std::isfinite(dm),"ring0 donor mass delta is nonfinite");
        const auto first=report.donorFirstMomentDeltaKgM.find(body);
        const auto second=report.donorSecondMomentDeltaKgM2.find(body);
        require(first!=report.donorFirstMomentDeltaKgM.end()&&second!=report.donorSecondMomentDeltaKgM2.end(),
                "ring0 donor first/second moment delta missing");
        for(double x:first->second) require(std::isfinite(x),"ring0 donor first moment is nonfinite");
        for(double x:second->second) require(std::isfinite(x),"ring0 donor second moment is nonfinite");
    }
    std::cout<<"PASS d36_actual_label_build local_splice exact_cavity_and_interface material_volume_mass output_label_remap aggregate_mass_moments transactional_rejections=7"
             <<" nodes="<<object.femNodes.size()<<" tets="<<object.tetrahedra.size()
             <<" outside_tets_preserved="<<retained
             <<" replaced_tets="<<report.originalTetCount-retained
             <<" children="<<input.childTetrahedra.size()
             <<" material_volume_closure_error_m3="<<materialVolumeClosureError
             <<" skin_mass_kg="<<report.skinMassAfterKg
             <<" fat_mass_kg="<<report.fatMassAfterKg
             <<" donor_mass_bodies="<<report.donorMassDeltaKg.size()<<"\n";
    return 0;
} catch(const std::exception& e) {
    std::cerr<<"FAIL "<<e.what()<<"\n";
    return 1;
}
