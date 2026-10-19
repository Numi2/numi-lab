#include "NumiHumanTetGenOrganicFEM.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
using namespace metalrobo::human::coarse_tissue;

static std::vector<std::string> dataLines(const std::string& path) {
    std::ifstream f(path); if(!f) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> out; std::string line;
    while(std::getline(f,line)){auto p=line.find('#');if(p!=std::string::npos)line.resize(p);std::istringstream s(line);std::string x;if(s>>x)out.push_back(line);}
    return out;
}
static void makeIdentity(std::array<std::uint64_t,4>& x,const std::string& hash){if(hash.size()!=64)throw std::runtime_error("identity hash must be SHA256");for(unsigned i=0;i<4;i++)x[i]=std::stoull(hash.substr(16*i,16),nullptr,16);}
static RegisteredTetrahedralVolume readTetgen(const std::string& dir,const std::string& prefix,
        const std::string& member,const std::string& anatomicalConcept,const std::string& meshHash,
        const std::string& payloadHash,std::uint32_t nodeCount,std::uint32_t faceCount) {
    RegisteredTetrahedralVolume s;
    s.name=member+"_body23_passive_volume"; s.provider="BodyParts3D v4.0"; s.hierarchy="is_a"; s.sourceMember=member;
    s.anatomicalConceptIdentifier=anatomicalConcept; s.stableIdentifier=0; s.sourceBodyIndex=23; s.sourceLayer=0; s.sourceVertexCount=nodeCount; s.sourceFaceCount=faceCount;
    s.sourceMeshSHA256=meshHash; s.sourcePayloadSHA256=payloadHash;
    s.geometryProvenance="TetGen 1.6 body-23-local exact-coordinate-welded closed source PLC; no repair/caps";
    s.frameTransformProvenance="CPU geometry fixture uses identity world/reference/donor-local body23 coordinates; not a runtime pose registration";
    s.intersectionAuditProvenance="TetGen -pd intersection-only audit exit=0; retained report optic-nerve-tetgen-preparation.json";
    s.referenceConfigurationProvenance="registered body-23-local source coordinates; reference-only geometry fixture, no unloaded-state claim";
    makeIdentity(s.materialSourceIdentity,"f9fc009dc6ee8b110d475a87ea3ea3320c02095f07c8944e85b3c7f1d6967d33"); makeIdentity(s.referenceSourceIdentity,"31c77660a16e8688fe68489eadff9d3861ceb27ed3b73d0288bda330d2a13b8c");
    auto nodeRows=dataLines(dir+"/"+prefix+".1.node"); std::istringstream nh(nodeRows[0]); unsigned n,dim,attrs,markers; nh>>n>>dim>>attrs>>markers;
    std::unordered_map<unsigned,unsigned> index;
    for(unsigned i=0;i<n;i++){std::istringstream row(nodeRows[i+1]);unsigned id,marker=0;V3 p{};row>>id>>p[0]>>p[1]>>p[2];if(markers)row>>marker;if(dim!=3||id==0)throw std::runtime_error("bad node header/id");index[id]=i;s.worldPoints.push_back(p);s.referenceWorldPoints.push_back(p);s.donorLocalPoints.push_back(p);s.nodeSourceVertex.push_back(marker?marker-1:NM_INVALID_INDEX);}
    s.sourceBodyIndex=23;
    auto eleRows=dataLines(dir+"/"+prefix+".1.ele");std::istringstream eh(eleRows[0]);unsigned nt,nper,ea;eh>>nt>>nper>>ea;if(nper!=4)throw std::runtime_error("bad tet width");
    for(unsigned i=0;i<nt;i++){std::istringstream row(eleRows[i+1]);unsigned id;std::array<std::uint32_t,4> t{};row>>id;for(auto& x:t){unsigned raw;row>>raw;if(!index.contains(raw))throw std::runtime_error("missing tet node id");x=index[raw];}s.tetrahedra.push_back(t);}
    auto faceRows=dataLines(dir+"/"+prefix+".1.face");std::istringstream fh(faceRows[0]);unsigned nf,fm;fh>>nf>>fm;
    for(unsigned i=0;i<nf;i++){std::istringstream row(faceRows[i+1]);unsigned id,a,b,c,mark;row>>id>>a>>b>>c;if(fm)row>>mark;else mark=0;if(!index.contains(a)||!index.contains(b)||!index.contains(c))throw std::runtime_error("missing face node");s.boundaryFaces.push_back({index[a],index[c],index[b]});s.boundarySourceFaceIndices.push_back(mark?mark-1:NM_INVALID_INDEX);}
    if(s.boundarySourceFaceIndices.size()!=nf||nf!=faceCount)throw std::runtime_error("unexpected boundary marker cardinality");
    std::uint32_t maxVertex=0;for(auto x:s.nodeSourceVertex)if(x!=NM_INVALID_INDEX)maxVertex=std::max(maxVertex,x+1);
    if(maxVertex!=nodeCount)throw std::runtime_error("unexpected canonical source vertex ancestry count");
    return s;
}
static void appendMaterial(numi::matter::WorldSource& w) {
    numi::matter::MaterialProgram m; m.name="human_optic_nerve_longitudinal_low_strain_isotropic_effective";
    m.parameters.push_back(numi::matter::Parameter{"density",{},1000.0,1000.0,1000.0,false,false});
    m.supportedRepresentations.push_back(numi::matter::Representation::fem);
    m.staticFriction=0.05;m.dynamicFriction=0.03;m.restitution=0;m.adhesion=0;
    w.materials.push_back(m);
}
static bool run(const std::string& dir,const std::string& prefix,const std::string& member,const std::string& anatomicalConcept,
        const std::string& meshHash,std::uint32_t sourceNodes,std::uint32_t boundaryFaces) {
    constexpr const char* payload="3b0acb6146358f3302f2b85e2214bafa9f2fae7b0e51af70fd0838d6609da4a4";
    auto s=readTetgen(dir,prefix,member,anatomicalConcept,meshHash,payload,sourceNodes,boundaryFaces);
    double volume=0;for(auto t:s.tetrahedra)volume+=tetVolume(s.referenceWorldPoints[t[0]],s.referenceWorldPoints[t[1]],s.referenceWorldPoints[t[2]],s.referenceWorldPoints[t[3]]);
    numi::matter::WorldSource w;appendMaterial(w);RegisteredTetMaterial binding;binding.materialIndex=0;binding.materialName=w.materials[0].name;binding.provenance="Park et al. 2021 human ON low-strain E3% effective isotropic proxy; see material file";
    s.materials.push_back(binding);
    PreparedOrganicFEM prepared;PreparedTetrahedralAncestry ancestry;std::string error;
    const auto expectedName=s.name;
    bool ok=appendRegisteredTetrahedralFEM(w,s,expectedName,member,anatomicalConcept,0,0,member+"_test_FEM",1000.0*volume,prepared,ancestry,error);
    if(!ok){std::cerr<<member<<" append failed: "<<error<<"\n";return false;}
    if(w.objects.size()!=1||w.objects[0].tetrahedra.size()!=s.tetrahedra.size()||prepared.donorMassNodes.size()!=s.worldPoints.size()||ancestry.boundarySourceFaceIndices.size()!=s.boundaryFaces.size())return false;
    double mass=0;for(const auto& m:prepared.donorMassNodes)mass+=m.massKg;
    std::cout<<member<<" PASS nodes="<<s.worldPoints.size()<<" tets="<<s.tetrahedra.size()<<" boundary="<<s.boundaryFaces.size()<<" volume_m3="<<prepared.volumeM3<<" mass_kg="<<mass<<" residual_kg="<<prepared.massResidualKg<<"\n";
    auto expectReject=[&](const char* label,RegisteredTetrahedralVolume bad){
        numi::matter::WorldSource badWorld;appendMaterial(badWorld);PreparedOrganicFEM ignored;PreparedTetrahedralAncestry ignoredA;std::string badError;
        bool rejected=appendRegisteredTetrahedralFEM(badWorld,bad,expectedName,member,anatomicalConcept,0,0,"bad",1000.0*volume,ignored,ignoredA,badError);
        if(rejected||!badWorld.objects.empty()||badError.empty()){std::cerr<<member<<" "<<label<<" regression failed; error="<<badError<<"\n";return false;}
        std::cout<<member<<" "<<label<<" fail-closed PASS: "<<badError<<"\n";return true;
    };
    auto inverted=s;std::swap(inverted.tetrahedra[0][0],inverted.tetrahedra[0][1]);if(!expectReject("inverted tet",std::move(inverted)))return false;
    auto duplicate=s;duplicate.tetrahedra.push_back(duplicate.tetrahedra[0]);if(!expectReject("duplicate tet",std::move(duplicate)))return false;
    auto collapsed=s;for(auto& p:collapsed.worldPoints)for(auto& x:p)x+=1.0e8;if(!expectReject("current Float32 collapse",std::move(collapsed)))return false;
    auto badShared=s;std::array<std::uint32_t,4> baseTet{};Face sharedFace{};bool foundBoundaryFace=false;for(const auto& candidate:badShared.tetrahedra){for(const auto& face:badShared.boundaryFaces){if(std::all_of(face.begin(),face.end(),[&](auto node){return std::find(candidate.begin(),candidate.end(),node)!=candidate.end();})){baseTet=candidate;sharedFace=face;foundBoundaryFace=true;break;}}if(foundBoundaryFace)break;}if(!foundBoundaryFace){std::cerr<<member<<" has no tet with a marked boundary face\n";return false;}V3 center{};for(auto i:baseTet)center=add(center,mul(badShared.worldPoints[i],0.25));const auto centerIndex=static_cast<std::uint32_t>(badShared.worldPoints.size());badShared.worldPoints.push_back(center);badShared.referenceWorldPoints.push_back(center);badShared.donorLocalPoints.push_back(center);badShared.nodeSourceVertex.push_back(NM_INVALID_INDEX);std::array<std::uint32_t,4> extra{sharedFace[0],sharedFace[1],sharedFace[2],centerIndex};if(tetVolume(badShared.worldPoints[extra[0]],badShared.worldPoints[extra[1]],badShared.worldPoints[extra[2]],badShared.worldPoints[extra[3]])<0)std::swap(extra[0],extra[1]);badShared.tetrahedra.push_back(extra);if(!expectReject("same-side shared face",std::move(badShared)))return false;
    return true;
}
int main(int argc,char** argv){
 const std::string base=(argc>1?argv[1]:"/Users/n/numi-human-separation-20261018/C/nerve-volume-240/assets");
 bool a=run(base+"/FJ1772","FJ1772","FJ1772","FMA50878","314b4a735672c1ce6690d29199856569fa3db4864f2d94690e091432e085e057",1190,2376);
 bool b=run(base+"/FJ1819","FJ1819","FJ1819","FMA50875","3fde0d489978522bb0c916db61287c42aedfb4c6660b6f8016a148ffccef89f4",1191,2378);
 return a&&b?0:1;
}
