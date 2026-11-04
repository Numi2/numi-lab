#pragma once

#include "NumiHumanCurrentFractionalSupportNPZ.hpp"
#include "NumiHumanPhysicalSkinReplacement.hpp"
#include "NumiHumanSkinFatReductionLabels.hpp"
#include <bit>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace metalrobo::human::coarse_tissue {

// Initialization-only composition of an authenticated mass mapping with exact
// fixed-boundary cavities (legacy 3-to-2 and bounded connected retessellations). The imported bundle remains immutable provenance.
// No runtime force, state owner, clearance, or acceptance threshold is changed.
struct SkinFatCavityReconditioningReport {
    std::size_t cavities=0u,inputCells=0u,outputCells=0u,affectedNodes=0u,restrictedTransportCavities=0u;
    double minimumQualityBefore=INFINITY,minimumQualityAfter=INFINITY;
    double maximumNodalMassChangeKg=0.0,nodalMassL1ChangeKg=0.0;
    double maximumSourceMassChangeKg=0.0,maximumSourceMomentChangeKgM=0.0;
    double maximumTransportDistanceUsedM=0.0;
    std::array<double,3> lumpedFirstMomentChangeKgM{};
    std::string proposalSHA256,compositionSHA256,inputReferenceSHA256;
};

namespace skin_fat_cavity_detail {
using Tet=std::array<std::uint32_t,4>;
using Triangle=std::array<std::uint32_t,3>;
using Edge=std::array<std::uint32_t,2>;
template<class Key> struct ArrayHash {
    std::size_t operator()(const Key& k) const noexcept {
        std::size_t h=1469598103934665603ull;
        for(auto v:k)h=(h^v)*1099511628211ull;
        return h;
    }
};
inline constexpr unsigned faces[4][3]={{1,2,3},{0,3,2},{0,1,3},{0,2,1}};
inline constexpr unsigned edges[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
inline std::map<Triangle,int> chain(std::span<const Tet> cells) {
    std::map<Triangle,int> result;
    for(const auto& t:cells)for(const auto& f:faces) {
        Triangle key{t[f[0]],t[f[1]],t[f[2]]};
        const auto inversions=(key[0]>key[1])+(key[0]>key[2])+(key[1]>key[2]);
        std::sort(key.begin(),key.end());result[key]+=inversions%2?-1:1;
    }
    std::erase_if(result,[](const auto& p){return p.second==0;});
    return result;
}

// Float32 coordinates are dyadic. This bounded integer lattice makes orientation
// and cavity-volume equality exact without another arithmetic dependency.
// Refuse a cavity if its local dynamic range cannot fit signed 128-bit products.
struct Lattice {
    std::map<std::uint32_t,std::array<std::int64_t,3>> points;
    bool prepare(std::span<const V3> x,const std::set<std::uint32_t>& nodes) {
        int minimumExponent=128;
        for(auto n:nodes)for(double value:x[n]) {
            const float f=static_cast<float>(value);
            if(!std::isfinite(f))return false;
            const auto bits=std::bit_cast<std::uint32_t>(f)&0x7fffffffu;
            if(bits)minimumExponent=std::min(minimumExponent,
                bits>>23u?int(bits>>23u)-127-23:-149);
        }
        if(minimumExponent==128)return false;
        for(auto n:nodes)for(unsigned a=0;a<3u;++a) {
            const auto bits=std::bit_cast<std::uint32_t>(static_cast<float>(x[n][a]));
            const auto magnitude=bits&0x7fffffffu;std::int64_t integer=0;
            if(magnitude) {
                const unsigned e=magnitude>>23u;
                const std::uint64_t significand=(magnitude&0x7fffffu)|(e?0x800000u:0u);
                const int shift=(e?int(e)-127-23:-149)-minimumExponent;
                if(shift<0||shift>38||significand>(std::uint64_t(INT64_MAX)>>shift))return false;
                integer=static_cast<std::int64_t>(significand<<shift);
                if(bits>>31u)integer=-integer;
            }
            points[n][a]=integer;
        }
        const auto origin=points.begin()->second;
        for(auto& [n,p]:points)for(unsigned a=0;a<3u;++a) {
            const __int128 difference=__int128(p[a])-origin[a];
            if(difference<=-(__int128(1)<<39)||difference>=(__int128(1)<<39))return false;
            p[a]=static_cast<std::int64_t>(difference);
        }
        return true;
    }
    __int128 determinant(const Tet& t) const {
        std::array<std::array<__int128,3>,3> e{};
        for(unsigned i=0;i<3u;++i)for(unsigned a=0;a<3u;++a)
            e[i][a]=__int128(points.at(t[i+1u])[a])-points.at(t[0])[a];
        return e[0][0]*(e[1][1]*e[2][2]-e[1][2]*e[2][1])
             - e[0][1]*(e[1][0]*e[2][2]-e[1][2]*e[2][0])
             + e[0][2]*(e[1][0]*e[2][1]-e[1][1]*e[2][0]);
    }
};
inline double volume(std::span<const V3> points,const Tet& t) {
    return physical_skin_child_mass_detail::signedTetVolume(
        points[t[0]],points[t[1]],points[t[2]],points[t[3]]);
}
inline double quality(std::span<const V3> points,const Tet& t) {
    double longest2=0.0;
    for(const auto& e:edges) {
        double d2=0.0;
        for(unsigned a=0;a<3u;++a) {
            const double d=double(float(points[t[e[0]]][a]))-double(float(points[t[e[1]]][a]));
            d2+=d*d;
        }
        longest2=std::max(longest2,d2);
    }
    V3 p[4];for(unsigned i=0;i<4u;++i)for(unsigned a=0;a<3u;++a)p[i][a]=float(points[t[i]][a]);
    const double v=physical_skin_child_mass_detail::signedTetVolume(p[0],p[1],p[2],p[3]);
    return 6.0*v/(longest2*std::sqrt(longest2));
}
struct Patch {
    std::vector<std::uint32_t> oldRows;
    std::vector<Tet> cells;
    Edge oldEdge{};Triangle ring{};
    std::map<std::uint32_t,double> parentMass;
    std::map<std::uint32_t,double> ancestry;
    std::vector<double> mass,fractions;
    std::vector<std::map<std::uint32_t,double>> restrictedParentMass;
    double fraction=0.0;
    bool edgeStar=false;
};

inline bool certifyGeometry(std::span<const V3> points,
        std::span<const Tet> old,const std::array<Tet,2>& next,Edge& edge,Triangle& ring,
        double& minimumBefore,double& minimumAfter,double& fraction,std::string& error) {
    const auto fail=[&](const char* message){error=message;return false;};
    if(old.size()!=3u)return fail("cavity must contain three cells");
    std::set<std::uint32_t> nodes;
    for(const auto& t:old) {
        std::set<std::uint32_t> unique(t.begin(),t.end());
        if(unique.size()!=4u)return fail("repeated cavity vertex");
        nodes.insert(t.begin(),t.end());
    }
    if(nodes.size()!=5u)return fail("3-to-2 cavity must have five vertices");
    std::vector<std::uint32_t> common;
    for(auto n:nodes)if(std::all_of(old.begin(),old.end(),[&](const auto& t){
        return std::find(t.begin(),t.end(),n)!=t.end();}))common.push_back(n);
    if(common.size()!=2u)return fail("3-to-2 cavity lacks its complete edge star");
    edge={common[0],common[1]};
    unsigned j=0u;for(auto n:nodes)if(n!=edge[0]&&n!=edge[1])ring[j++]=n;
    std::set<Edge> links;
    for(const auto& t:old) {
        Edge link{};j=0u;for(auto n:t)if(n!=edge[0]&&n!=edge[1])link[j++]=n;
        std::sort(link.begin(),link.end());links.insert(link);
    }
    if(links!=std::set<Edge>{{ring[0],ring[1]},{ring[0],ring[2]},{ring[1],ring[2]}})
        return fail("edge star is open or duplicated");
    std::set<Tet> expected,actual;
    for(auto n:edge) {Tet t{ring[0],ring[1],ring[2],n};std::sort(t.begin(),t.end());expected.insert(t);}
    for(auto t:next) {std::sort(t.begin(),t.end());actual.insert(t);}
    if(actual!=expected||chain(old)!=chain(next))return fail("oriented cavity boundary changed");
    Lattice lattice;
    if(!lattice.prepare(points,nodes))return fail("cavity exceeds exact lattice arithmetic range");
    __int128 oldSum=0,newSum=0;
    for(const auto& t:old) {const auto det=lattice.determinant(t);if(det<=0)return fail("nonpositive exact old cell");oldSum+=det;}
    for(const auto& t:next) {const auto det=lattice.determinant(t);if(det<=0)return fail("nonpositive exact replacement");newSum+=det;}
    if(oldSum!=newSum)return fail("exact cavity volume changed");
    minimumBefore=INFINITY;minimumAfter=INFINITY;
    for(const auto& t:old)minimumBefore=std::min(minimumBefore,quality(points,t));
    for(const auto& t:next) {
        const double v=volume(points,t),q=quality(points,t);
        // The original skin/fat compiler's positive-volume and quality gates.
        if(!(std::isfinite(v)&&v>1e-18&&std::isfinite(q)&&q>=1e-5))
            return fail("replacement fails unchanged native cell quality");
        minimumAfter=std::min(minimumAfter,q);
    }
    if(minimumAfter<1.02*minimumBefore)return fail("replacement does not improve worst conditioning");
    fraction=double(lattice.determinant(next[0]))/double(newSum);
    if(!(std::isfinite(fraction)&&fraction>0.0&&fraction<1.0))return fail("invalid exact split fraction");
    return true;
}


// General cavities use the same immutable coordinates and existing CSR proposal.
// Their bounded exact certificate is stricter than a floating volume comparison.
inline bool cavityBall(std::span<const Tet> cells,std::string& error) {
    struct Use {std::size_t row;int sign;};
    std::map<Triangle,std::vector<Use>> use;
    std::set<Tet> unique;
    for(std::size_t i=0;i<cells.size();++i) {
        auto sorted=cells[i];std::sort(sorted.begin(),sorted.end());
        if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end()||!unique.insert(sorted).second) {
            error="duplicate cell or repeated vertex";return false;
        }
        for(const auto& f:faces) {
            Triangle key{cells[i][f[0]],cells[i][f[1]],cells[i][f[2]]};
            const int sign=((key[0]>key[1])+(key[0]>key[2])+(key[1]>key[2]))%2?-1:1;
            std::sort(key.begin(),key.end());use[key].push_back({i,sign});
        }
    }
    std::vector<std::vector<std::size_t>> adjacency(cells.size());
    std::map<Edge,unsigned> shellEdges;std::set<std::uint32_t> shellNodes;
    std::size_t shellFaces=0;
    for(const auto& [face,owners]:use) {
        if(owners.size()==2u&&owners[0].sign==-owners[1].sign) {
            adjacency[owners[0].row].push_back(owners[1].row);
            adjacency[owners[1].row].push_back(owners[0].row);
        } else if(owners.size()==1u) {
            ++shellFaces;shellNodes.insert(face.begin(),face.end());
            for(unsigned j=0;j<3u;++j) {
                Edge e{face[j],face[(j+1u)%3u]};std::sort(e.begin(),e.end());++shellEdges[e];
            }
        } else {error="nonmanifold or inconsistently oriented cavity face";return false;}
    }
    if(shellNodes.size()+shellFaces!=shellEdges.size()+2u||
       std::any_of(shellEdges.begin(),shellEdges.end(),[](const auto& p){return p.second!=2u;})) {
        error="cavity boundary is not a closed sphere";return false;
    }
    std::set<std::size_t> seen;std::vector<std::size_t> todo{0};
    while(!todo.empty()) {const auto i=todo.back();todo.pop_back();if(!seen.insert(i).second)continue;
        todo.insert(todo.end(),adjacency[i].begin(),adjacency[i].end());}
    if(seen.size()!=cells.size()){error="disconnected cavity";return false;}
    return true;
}
inline bool disjointInteriors(const Lattice& lattice,const Tet& a,const Tet& b) {
    using I3=std::array<__int128,3>;
    const auto point=[&](std::uint32_t v){const auto& p=lattice.points.at(v);return I3{p[0],p[1],p[2]};};
    const auto sub=[](const I3& u,const I3& v){return I3{u[0]-v[0],u[1]-v[1],u[2]-v[2]};};
    const auto cross=[](const I3& u,const I3& v){return I3{u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};};
    const auto separated=[&](const I3& n) {
        if(n==I3{})return false;
        const auto dot=[&](std::uint32_t v){const auto p=point(v);return p[0]*n[0]+p[1]*n[1]+p[2]*n[2];};
        auto amin=dot(a[0]),amax=amin,bmin=dot(b[0]),bmax=bmin;
        for(unsigned k=1;k<4u;++k){const auto av=dot(a[k]),bv=dot(b[k]);
            amin=std::min(amin,av);amax=std::max(amax,av);bmin=std::min(bmin,bv);bmax=std::max(bmax,bv);}
        return amax<=bmin||bmax<=amin;
    };
    // Bounds from Lattice::prepare keep all cross/dot products inside int128.
    for(const auto& t:{a,b})for(const auto& f:faces)
        if(separated(cross(sub(point(t[f[1]]),point(t[f[0]])),sub(point(t[f[2]]),point(t[f[0]])))))return true;
    for(const auto& ea:edges)for(const auto& eb:edges)
        if(separated(cross(sub(point(a[ea[1]]),point(a[ea[0]])),sub(point(b[eb[1]]),point(b[eb[0]])))))return true;
    return false;
}
inline bool certifyConnectedGeometry(std::span<const V3> points,std::span<const Tet> old,
        std::span<const Tet> next,double& before,double& after,
        std::vector<double>& fractions,std::string& error) {
    const auto fail=[&](const char* message){error=message;return false;};
    if(old.size()<2u||old.size()>32u||next.size()<2u||next.size()>64u)
        return fail("connected cavity exceeds bounded cell domain");
    std::set<std::uint32_t> nodes,nextNodes;
    for(const auto& t:old)nodes.insert(t.begin(),t.end());
    for(const auto& t:next)nextNodes.insert(t.begin(),t.end());
    if(nodes!=nextNodes||nodes.size()>32u||nodes.empty()||*nodes.rbegin()>=points.size())
        return fail("connected cavity changes its vertex domain");
    if(!cavityBall(old,error)||!cavityBall(next,error))return false;
    if(chain(old)!=chain(next))return fail("oriented connected cavity boundary changed");
    Lattice lattice;if(!lattice.prepare(points,nodes))return fail("cavity exceeds exact lattice arithmetic range");
    __int128 oldSum=0,newSum=0;
    for(const auto& t:old) {const auto d=lattice.determinant(t);
        if(d<=0||__builtin_add_overflow(oldSum,d,&oldSum))return fail("invalid exact old cavity volume");}
    for(const auto& t:next) {const auto d=lattice.determinant(t);
        if(d<=0||__builtin_add_overflow(newSum,d,&newSum))return fail("invalid exact new cavity volume");}
    if(oldSum!=newSum)return fail("exact connected cavity volume changed");
    for(const auto group:{old,next})for(std::size_t i=0;i<group.size();++i)for(std::size_t j=0;j<i;++j)
        if(!disjointInteriors(lattice,group[i],group[j]))return fail("overlapping cavity cell interiors");
    before=INFINITY;after=INFINITY;fractions.clear();
    for(const auto& t:old)before=std::min(before,quality(points,t));
    for(const auto& t:next) {
        const double v=volume(points,t),q=quality(points,t);
        if(!(std::isfinite(v)&&v>1e-18&&std::isfinite(q)&&q>=1e-5))
            return fail("replacement fails unchanged native cell quality");
        after=std::min(after,q);
        const double fraction=double(lattice.determinant(t))/double(newSum);
        if(!(std::isfinite(fraction)&&fraction>0.0&&fraction<1.0))return fail("invalid exact split fraction");
        fractions.push_back(fraction);
    }
    if(after<1.02*before)return fail("replacement does not improve worst conditioning");
    return true;
}


// Initialization-only reuse of the existing compensated sparse feasibility owner.
// Row/column closures are certified separately; no radius or density bound changes.
inline bool routeRestrictedParentMass(const std::map<std::uint32_t,double>& source,
        std::span<const double> demand,std::span<const double> distance,
        double radius,std::vector<std::map<std::uint32_t,double>>& output,
        std::string& error) {
    using namespace spatial_mass_transport_detail;
    const std::size_t np=source.size(),nc=demand.size();
    const auto fail=[&](const char* reason){error=reason;return false;};
    if(np==0u||np>4096u||nc<2u||nc>64u||distance.size()!=np*nc||
       !std::isfinite(radius)||radius<=0.0)return fail("invalid bounded cavity transport domain");
    const auto sink=static_cast<std::uint32_t>(1u+np+nc);
    MinCostFlow graph(sink+1u);std::vector<EdgeRef> refs;
    std::size_t i=0;double total=0.0;
    for(const auto& [parent,mass]:source) {
        if(!(std::isfinite(mass)&&mass>0.0))return fail("invalid cavity source mass");
        const auto from=static_cast<std::uint32_t>(1u+i);graph.addEdge(0u,from,mass,0.0);total+=mass;
        for(std::size_t j=0;j<nc;++j) {
            const double d=distance[i*nc+j];
            if(!std::isfinite(d)||d<0.0)return fail("invalid cavity source distance");
            if(d<=radius) {
                const auto edge=graph.addEdge(from,static_cast<std::uint32_t>(1u+np+j),mass,0.0);
                refs.push_back({parent,static_cast<std::uint32_t>(j),from,edge,d});
            }
        }
        ++i;
    }
    for(std::size_t j=0;j<nc;++j) {
        if(!(std::isfinite(demand[j])&&demand[j]>0.0))return fail("invalid cavity cell demand");
        graph.addEdge(static_cast<std::uint32_t>(1u+np+j),sink,demand[j],0.0);
    }
    std::size_t augmentations=0;double sent=0.0;
    if(!graph.runFeasible(0u,sink,4u*(refs.size()+np+nc),augmentations,sent,error))return false;
    std::vector<std::map<std::uint32_t,double>> next(nc);
    std::map<std::uint32_t,double> used;std::vector<double> received(nc,0.0);
    for(const auto& ref:refs) {
        const double mass=graph.flow(ref);
        if(!std::isfinite(mass)||mass<0.0)return fail("invalid cavity transport result");
        if(mass==0.0)continue;
        next[ref.child][ref.parent]=mass;used[ref.parent]+=mass;received[ref.child]+=mass;
    }
    for(const auto& [parent,mass]:source)
        if(std::abs(used[parent]-mass)>1e-12*mass)return fail("cavity source cannot close within frozen transport radius");
    for(std::size_t j=0;j<nc;++j)
        if(std::abs(received[j]-demand[j])>1e-12*demand[j])
            return fail("cavity cell cannot close within frozen transport radius");
    if(!std::isfinite(sent)||std::abs(sent-total)>1e-12*total)
        return fail("cavity total transport does not close");
    output=std::move(next);return true;
}

struct PreparedCells {
    std::vector<std::uint32_t> child,parent,face,material,source,offsets,ancestry;
    std::vector<double> fractions;
    void append(const PreparedOrganicFEM& old,std::size_t r,std::uint32_t row) {
        child.push_back(row);parent.push_back(old.tetParentTetrahedron[r]);
        face.push_back(old.tetParentSurfaceFace[r]);material.push_back(old.tetParentMaterialIndex[r]);
        source.push_back(old.tetSourceFaceIdentity[r]);

    }
    void append(const Patch& p,const PreparedOrganicFEM& old,std::size_t r,std::uint32_t row) {
        child.push_back(row);
        (void)p;parent.push_back(old.tetParentTetrahedron[r]);
        face.push_back(old.tetParentSurfaceFace[r]);material.push_back(old.materialIndex);
        source.push_back(old.tetSourceFaceIdentity[r]);
    }
    void commit(PreparedOrganicFEM& p) {
        p.tetChildIndex=std::move(child);p.tetParentTetrahedron=std::move(parent);
        p.tetParentSurfaceFace=std::move(face);p.tetParentMaterialIndex=std::move(material);
        p.tetSourceFaceIdentity=std::move(source);p.tetSourceFace=p.tetSourceFaceIdentity;
        p.tetParentAncestryOffsets=std::move(offsets);p.tetParentAncestryTetrahedra=std::move(ancestry);
        p.tetParentAncestryVolumeFractions=std::move(fractions);
    }
};
} // namespace skin_fat_cavity_detail

[[nodiscard]] inline bool reconditionPreparedSkinFatFixedCavities(
        const numi::matter::WorldSource& world,numi::matter::ObjectSource& object,
        const PhysicalSkinChildMassMappingBundle& bundle,PreparedOrganicFEM& skin,PreparedOrganicFEM& fat,
        RegisteredSkinFatReductionSourceLabels& labels,
        const std::filesystem::path& proposalPath,std::string_view expectedProposalSHA,
        std::string_view sourceVolumeSHA,SkinFatCavityReconditioningReport& output,std::string& error) {
    using namespace skin_fat_cavity_detail;
    using namespace current_fractional_support_npz_detail;
    const auto fail=[&](std::string message){error="skin/fat cavity composition: "+message;return false;};
    SkinFatCavityReconditioningReport report;report.inputCells=object.tetrahedra.size();
    if(!physical_skin_child_mass_detail::validSHA256(expectedProposalSHA)||
       !physical_skin_child_mass_detail::validSHA256(sourceVolumeSHA)||
       !physical_skin_child_mass_detail::sha256File(proposalPath,report.proposalSHA256,error)||
       report.proposalSHA256!=expectedProposalSHA)return fail("proposal SHA mismatch");
    if(object.representation!=numi::matter::Representation::fem||object.mixedFEM||
       object.femNodes!=object.femReferenceNodes||object.femNodes!=bundle.currentWorldPoints||
       object.femReferenceNodes!=bundle.referenceWorldPoints||
       object.tetrahedra.size()!=bundle.childTetrahedra.size()||
       object.femMaterialIndices!=bundle.childMaterialIndices||
       object.femTetrahedronMassesKg!=bundle.spatialMassTransportResult.childMassKg||
       !bundle.fractionalSemanticMassRouteRequired||!bundle.spatialMassTransportResult.usesFractionalSemanticSupport||
       skin.donorMassContributions.empty()||!fat.donorMassContributions.empty()||
       !fat.donorMassNodes.empty()||skin.donorMassNodes.size()!=object.femNodes.size())
        return fail("requires the complete verified initial-state fractional skin/fat owner");
    const auto n=object.femNodes.size(),nt=object.tetrahedra.size();
    std::map<std::string,NpyArray> arrays;
    if(!readRequestedArrays(proposalPath,{"retained_input_cell_rows","replacement_tetrahedra",
        "replacement_tet_offsets","replacement_parent_offsets","replacement_parent_cell_rows",
        "source_volume_sha256"},arrays,error))return false;
    std::string embeddedSource;
    if(!readSHA256(arrays.at("source_volume_sha256"),embeddedSource,error)||embeddedSource!=sourceVolumeSHA)
        return fail("proposal is not bound to the authenticated source volume");
    std::vector<std::uint32_t> retained,replacements,patchOffsets,parentOffsets,parentRows;
    if(!readU32(arrays.at("retained_input_cell_rows"),retained,error)||
       !readU32(arrays.at("replacement_tetrahedra"),replacements,error)||
       !readU32(arrays.at("replacement_tet_offsets"),patchOffsets,error)||
       !readU32(arrays.at("replacement_parent_offsets"),parentOffsets,error)||
       !readU32(arrays.at("replacement_parent_cell_rows"),parentRows,error))return false;
    if(arrays.at("replacement_tetrahedra").shape.size()!=2u||
       arrays.at("replacement_tetrahedra").shape[1]!=4u||replacements.size()%4u||
       patchOffsets.size()<2u||patchOffsets.size()>50001u||
       patchOffsets.front()!=0u||patchOffsets.back()*4ull!=replacements.size()||
       parentOffsets.size()!=replacements.size()/4u+1u||parentOffsets.front()!=0u||
       parentOffsets.back()!=parentRows.size())return fail("malformed bounded fixed-cavity proposal");
    arrays.clear();
    const auto patchCount=patchOffsets.size()-1u;
    std::vector<Patch> patches(patchCount);
    std::vector<std::uint8_t> removed(nt,0u),affected(n,0u);
    std::vector<std::uint32_t> preparedRow(nt,NM_INVALID_INDEX);
    const auto indexPrepared=[&](const PreparedOrganicFEM& p)->bool {
        const auto count=p.tetChildIndex.size();
        if(p.tetParentTetrahedron.size()!=count||p.tetParentSurfaceFace.size()!=count||
           p.tetParentMaterialIndex.size()!=count||p.tetSourceFaceIdentity.size()!=count||
           !p.tetParentAncestryOffsets.empty()||!p.tetParentAncestryTetrahedra.empty()||
           !p.tetParentAncestryVolumeFractions.empty())return false;
        for(std::size_t r=0;r<count;++r) {
            const auto c=p.tetChildIndex[r];
            if(c>=nt||preparedRow[c]!=NM_INVALID_INDEX||object.femMaterialIndices[c]!=p.materialIndex)return false;
            preparedRow[c]=static_cast<std::uint32_t>(r);
        }
        return true;
    };
    if(!indexPrepared(skin)||!indexPrepared(fat)||
       std::find(preparedRow.begin(),preparedRow.end(),NM_INVALID_INDEX)!=preparedRow.end())
        return fail("incomplete input material ancestry");
    std::unordered_map<Edge,std::size_t,ArrayHash<Edge>> edgeToPatch;
    std::unordered_set<Triangle,ArrayHash<Triangle>> newFaces;
    std::vector<std::uint32_t> starCounts(patchCount,0u);
    std::unordered_map<Triangle,std::size_t,ArrayHash<Triangle>> connectedInteriorFaces;
    std::unordered_map<std::uint32_t,std::size_t> connectedOldRows;
    std::vector<std::uint32_t> transportRejectedCavities;
    const auto& transfer=bundle.spatialMassTransportResult;
    const double maximumTransportDistance=bundle.maximumSpatialMassTransportDistanceM;
    if(!std::isfinite(maximumTransportDistance)||maximumTransportDistance<=0.0||
       transfer.maximumTransportDistanceM!=maximumTransportDistance)
        return fail("cavity transport radius differs from the frozen mass mapping");
    for(std::size_t i=0;i<patchCount;++i) {
        auto& p=patches[i];const auto first=patchOffsets[i];
        if(patchOffsets[i+1u]<first||patchOffsets[i+1u]>replacements.size()/4u)
            return fail("invalid replacement patch offsets");
        const auto count=patchOffsets[i+1u]-first;
        if(count<2u||count>64u)return fail("replacement patch exceeds bounded cell domain");
        p.cells.resize(count);p.mass.resize(count);
        for(std::size_t j=0;j<count;++j) {
            const auto r=first+j;
            if(parentOffsets[r+1u]<parentOffsets[r]||parentOffsets[r+1u]>parentRows.size())
                return fail("invalid replacement ancestry offsets");
            const auto ancestors=parentOffsets[r+1u]-parentOffsets[r];
            if(ancestors<2u||ancestors>32u)return fail("replacement lacks bounded complete cavity ancestry");
            if(j==0u)p.oldRows.assign(parentRows.begin()+parentOffsets[r],parentRows.begin()+parentOffsets[r+1u]);
            else if(ancestors!=p.oldRows.size()||
                    !std::equal(p.oldRows.begin(),p.oldRows.end(),parentRows.begin()+parentOffsets[r]))
                return fail("replacements have different cavity ancestry");
            std::copy_n(replacements.begin()+4u*r,4u,p.cells[j].begin());
        }
        auto sorted=p.oldRows;std::sort(sorted.begin(),sorted.end());
        if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end()||sorted.back()>=nt)
            return fail("invalid retired cell rows");
        const auto seed=p.oldRows[0];const auto material=object.femMaterialIndices[seed];
        for(auto row:p.oldRows) {
            if(removed[row]||object.tetrahedra[row].nodes!=bundle.childTetrahedra[row]||
               object.femMaterialIndices[row]!=material||
               bundle.childDonorBodyIndices[row]!=bundle.childDonorBodyIndices[seed]||
               bundle.childAnatomicalRegionIndices[row]!=bundle.childAnatomicalRegionIndices[seed])
                return fail("cavity crosses a material, donor, region or retired-cell boundary");
            removed[row]=1u;
        }
        std::vector<Tet> old;for(auto row:p.oldRows)old.push_back(object.tetrahedra[row].nodes);
        std::set<std::uint32_t> nodes;for(const auto& t:old)nodes.insert(t.begin(),t.end());
        for(auto v:nodes)if(v>=n)return fail("cavity node outside the reference domain");
        for(const auto& t:p.cells)for(auto v:t)if(!nodes.contains(v))
            return fail("replacement introduces a vertex outside its cavity");
        for(auto v:nodes)affected[v]=1u;
        double before,after;
        p.edgeStar=old.size()==3u&&p.cells.size()==2u;
        if(p.edgeStar) {
            const std::array<Tet,2> next{p.cells[0],p.cells[1]};
            if(!certifyGeometry(object.femReferenceNodes,old,next,p.oldEdge,p.ring,before,after,p.fraction,error))
                return false;
            p.fractions={p.fraction,1.0-p.fraction};
            if(!edgeToPatch.emplace(p.oldEdge,i).second||!newFaces.insert(p.ring).second)
                return fail("duplicate patch edge or separator face");
        } else {
            if(!certifyConnectedGeometry(object.femReferenceNodes,old,p.cells,before,after,p.fractions,error))
                return false;
            for(auto row:p.oldRows)connectedOldRows.emplace(row,i);
            const auto boundary=chain(p.cells);
            for(const auto& t:p.cells)for(const auto& f:faces) {
                Triangle key{t[f[0]],t[f[1]],t[f[2]]};std::sort(key.begin(),key.end());
                if(!boundary.contains(key)) {
                    const auto [it,inserted]=connectedInteriorFaces.emplace(key,i);
                    if(!inserted&&it->second!=i)return fail("connected cavities share an interior face");
                }
            }
        }
        report.minimumQualityBefore=std::min(report.minimumQualityBefore,before);
        report.minimumQualityAfter=std::min(report.minimumQualityAfter,after);
        double totalMass=0.0,totalVolume=0.0;
        for(auto row:p.oldRows) {
            const auto v=volume(object.femReferenceNodes,object.tetrahedra[row].nodes);totalVolume+=v;
            totalMass+=object.femTetrahedronMassesKg[row];
            for(auto e=transfer.childParentOffsets[row];e<transfer.childParentOffsets[row+1u];++e)
                p.parentMass[transfer.childParentRows[e]]+=transfer.childParentTransportMassKg[e];

        }
        // Repartitioning moves cell centroids. Retain the frozen source radius;
        // general cavities may conservatively reroute only their existing sources.
        std::vector<V3> childCenters(p.cells.size());
        for(std::size_t j=0;j<p.cells.size();++j)for(auto node:p.cells[j])for(unsigned a=0;a<3u;++a)
            childCenters[j][a]+=0.25*object.femReferenceNodes[node][a];
        std::vector<double> distances;distances.reserve(p.parentMass.size()*p.cells.size());
        bool needsRestrictedTransport=false;double maximumAllDistance=0.0;
        for(const auto& [parent,mass]:p.parentMass) {
            if(parent>=bundle.parent.tetrahedra.size()||!std::isfinite(mass)||mass<=0.0)
                return fail("invalid cavity source flow");
            V3 parentCenter{};
            for(auto node:bundle.parent.tetrahedra[parent]) {
                if(node>=bundle.parent.referenceWorldPoints.size())return fail("cavity parent node outside reference domain");
                for(unsigned a=0;a<3u;++a)parentCenter[a]+=0.25*bundle.parent.referenceWorldPoints[node][a];
            }
            for(const auto& center:childCenters) {
                double d2=0.0;for(unsigned a=0;a<3u;++a){const double d=center[a]-parentCenter[a];d2+=d*d;}
                const double distance=std::sqrt(d2);
                if(!std::isfinite(distance))return fail("nonfinite cavity source distance");
                distances.push_back(distance);maximumAllDistance=std::max(maximumAllDistance,distance);
                needsRestrictedTransport|=distance>maximumTransportDistance;
            }
        }
        const double density=totalMass/totalVolume;
        const auto& parameters=world.materials[material].parameters;
        const auto bounds=std::find_if(parameters.begin(),parameters.end(),[](const auto& x){return x.name=="density";});
        if(bounds==parameters.end())return fail("replacement material has no density bounds");
        const double densityRoundoff=spatial_mass_transport_detail::densityBoundRoundoff(
            bounds->lower,bounds->upper,density);
        if(!std::isfinite(density)||density<bounds->lower-densityRoundoff||
           density>bounds->upper+densityRoundoff)
            return fail("conservative replacement density exceeds unchanged material bounds at cavity "+
                std::to_string(i)+" density="+std::to_string(density));
        double assignedMass=0.0;
        for(std::size_t j=0;j<p.cells.size();++j) {
            p.mass[j]=j+1u==p.cells.size()?totalMass-assignedMass:p.fractions[j]*totalMass;
            if(!(std::isfinite(p.mass[j])&&p.mass[j]>0.0))return fail("invalid composed cell mass");
            assignedMass+=p.mass[j];
        }
        const double edgeTotal=std::accumulate(p.parentMass.begin(),p.parentMass.end(),0.0,
            [](double s,const auto& x){return s+x.second;});
        if(std::abs(edgeTotal-totalMass)>1e-10*totalMass)return fail("input mass and parent flows disagree");
        if(needsRestrictedTransport) {
            if(p.edgeStar||!routeRestrictedParentMass(p.parentMass,p.mass,distances,
                    maximumTransportDistance,p.restrictedParentMass,error)) {
                transportRejectedCavities.push_back(static_cast<std::uint32_t>(i));
            } else {
                ++report.restrictedTransportCavities;
                std::size_t parentIndex=0;
                for(const auto& [parent,mass]:p.parentMass) {
                    (void)mass;
                    for(std::size_t j=0;j<p.cells.size();++j)
                        if(p.restrictedParentMass[j].contains(parent))
                            report.maximumTransportDistanceUsedM=std::max(
                                report.maximumTransportDistanceUsedM,distances[parentIndex*p.cells.size()+j]);
                    ++parentIndex;
                }
            }
        } else report.maximumTransportDistanceUsedM=std::max(report.maximumTransportDistanceUsedM,maximumAllDistance);
    }
    if(!transportRejectedCavities.empty()) {
        std::ostringstream message;
        message<<"skin_fat_cavity_transport_failures={\"diagnostic_only\":true,"
            <<"\"maximum_transport_distance_m\":"<<std::setprecision(17)<<maximumTransportDistance
            <<",\"cavity_indices\":[";
        for(std::size_t i=0;i<transportRejectedCavities.size();++i) {
            if(i)message<<",";message<<transportRejectedCavities[i];
        }
        message<<"],\"complete\":true}\n";
        const auto text=message.str();std::fwrite(text.data(),1u,text.size(),stderr);
        return fail("source flows exceed the frozen transport radius in "+
            std::to_string(transportRejectedCavities.size())+" cavities; no state changed");
    }
    // One streaming pass proves complete edge stars and absence of new faces.
    // Avoid each patch rescanning all 1.5 million cells.
    std::vector<double> oldAffectedMass(n,0.0);
    for(std::size_t row=0;row<nt;++row) {
        const auto& t=object.tetrahedra[row].nodes;
        if(t!=bundle.childTetrahedra[row])return fail("input object differs from its authenticated topology");
        for(const auto& e:edges) {
            Edge key{t[e[0]],t[e[1]]};std::sort(key.begin(),key.end());
            if(auto hit=edgeToPatch.find(key);hit!=edgeToPatch.end())++starCounts[hit->second];
        }
        for(const auto& f:faces) {
            Triangle key{t[f[0]],t[f[1]],t[f[2]]};std::sort(key.begin(),key.end());
            if(newFaces.contains(key))return fail("new separator face already exists in retained topology");
            if(const auto hit=connectedInteriorFaces.find(key);hit!=connectedInteriorFaces.end()) {
                const auto owner=connectedOldRows.find(static_cast<std::uint32_t>(row));
                if(owner==connectedOldRows.end()||owner->second!=hit->second)
                    return fail("connected interior face already exists outside its cavity");
            }
        }
        for(auto node:t)if(affected[node])oldAffectedMass[node]+=0.25*object.femTetrahedronMassesKg[row];
    }
    if(std::any_of(edgeToPatch.begin(),edgeToPatch.end(),[&](const auto& entry){return starCounts[entry.second]!=3u;}))
        return fail("retired rows are not the complete edge star");
    std::size_t k=0u;
    for(std::size_t row=0;row<nt;++row)if(!removed[row]) {
        if(k>=retained.size()||retained[k++]!=row)return fail("retained row map drops or reorders cells");
    }
    if(k!=retained.size())return fail("extra retained rows");
    std::vector<numi::matter::TetrahedronSource> nextTets;
    std::vector<std::uint32_t> nextMaterials;
    std::vector<double> nextMass;
    nextTets.reserve(retained.size()+replacements.size()/4u);nextMaterials.reserve(nextTets.capacity());nextMass.reserve(nextTets.capacity());
    PreparedCells skinCells,fatCells;
    // Rebuild only affected-node donor contributions. Current source debit
    // positions remain the immutable parent positions; mechanics keeps one owner.
    using SourceKey=std::pair<std::uint32_t,std::uint32_t>;
    std::map<SourceKey,std::vector<std::uint32_t>> sourceRows;
    std::map<SourceKey,double> sourceMass;
    for(std::uint32_t r=0;r<bundle.parent.sourceMassContributions.size();++r) {
        const auto& s=bundle.parent.sourceMassContributions[r];
        const SourceKey key{s.parentNodeIndex,s.materialIndex};
        sourceRows[key].push_back(r);sourceMass[key]+=s.massKg;
    }
    std::map<SourceKey,double> nextContributions;
    std::vector<double> nextAffectedMass(n,0.0);
    const auto addFlow=[&](const Tet& t,std::uint32_t material,std::uint32_t parent,double mass) {
        const auto& pt=bundle.parent.tetrahedra[parent];
        for(auto node:t)if(affected[node])for(auto parentNode:pt) {
            const SourceKey key{parentNode,material};const auto denominator=sourceMass.at(key);
            for(auto r:sourceRows.at(key))
                nextContributions[{node,r}]+=mass*0.0625*(bundle.parent.sourceMassContributions[r].massKg/denominator);
        }
    };
    for(auto old:retained) {
        const auto row=static_cast<std::uint32_t>(nextTets.size());
        const auto material=object.femMaterialIndices[old];const auto& t=object.tetrahedra[old];
        nextTets.push_back(t);nextMaterials.push_back(material);nextMass.push_back(object.femTetrahedronMassesKg[old]);
        auto& pc=material==skin.materialIndex?skinCells:fatCells;
        pc.append(material==skin.materialIndex?skin:fat,preparedRow[old],row);
        bool touches=false;for(auto node:t.nodes)if(affected[node]) {
            touches=true;nextAffectedMass[node]+=0.25*object.femTetrahedronMassesKg[old];
        }
        if(touches)for(auto e=transfer.childParentOffsets[old];e<transfer.childParentOffsets[old+1u];++e)
            addFlow(t.nodes,material,transfer.childParentRows[e],transfer.childParentTransportMassKg[e]);
    }
    for(const auto& p:patches)for(std::size_t j=0;j<p.cells.size();++j) {
        const auto row=static_cast<std::uint32_t>(nextTets.size());
        const auto material=object.femMaterialIndices[p.oldRows[0]];
        nextTets.push_back({p.cells[j]});nextMaterials.push_back(material);nextMass.push_back(p.mass[j]);
        auto& pc=material==skin.materialIndex?skinCells:fatCells;
        pc.append(p,material==skin.materialIndex?skin:fat,preparedRow[p.oldRows[0]],row);
        for(auto node:p.cells[j])nextAffectedMass[node]+=0.25*p.mass[j];
        if(!p.restrictedParentMass.empty()) {
            for(const auto& [parent,mass]:p.restrictedParentMass[j])addFlow(p.cells[j],material,parent,mass);
        } else for(const auto& [parent,mass]:p.parentMass) {
            double assigned=0.0;
            for(std::size_t k=0;k<j;++k)assigned+=p.fractions[k]*mass;
            const double part=j+1u==p.cells.size()?mass-assigned:p.fractions[j]*mass;
            if(!(std::isfinite(part)&&part>0.0))return fail("invalid composed parent flow");
            addFlow(p.cells[j],material,parent,part);
        }
    }
    std::map<std::uint32_t,double> newClosure;
    for(const auto& [key,mass]:nextContributions)newClosure[key.first]+=mass;
    for(std::size_t node=0;node<n;++node)if(affected[node]) {
        const double oldMass=skin.donorMassNodes[node].massKg;
        if(skin.donorMassNodes[node].nodeIndex!=skin.firstGlobalFEMNode+node||
           std::abs(oldMass-oldAffectedMass[node])>1e-9*std::max(oldMass,1e-12)||
           std::abs(newClosure[node]-nextAffectedMass[node])>1e-9*std::max(nextAffectedMass[node],1e-12)||
           !(nextAffectedMass[node]>0.0))return fail("node/cell/source mass closure failed");
        const double change=nextAffectedMass[node]-oldMass;
        report.maximumNodalMassChangeKg=std::max(report.maximumNodalMassChangeKg,std::abs(change));
        report.nodalMassL1ChangeKg+=std::abs(change);
        for(unsigned a=0;a<3u;++a)report.lumpedFirstMomentChangeKgM[a]+=change*object.femReferenceNodes[node][a];
    }
    using Moment=std::array<double,4>;
    std::map<std::uint32_t,Moment> oldMoments,newMoments;
    const auto addMoment=[](auto& moments,std::uint32_t donor,double mass,const auto& x) {
        auto& m=moments[donor];m[0]+=mass;for(unsigned a=0;a<3u;++a)m[a+1u]+=mass*x[a];
    };
    std::vector<NumiHumanTissueMassContribution> contributions;
    contributions.reserve(skin.donorMassContributions.size()+nextContributions.size());
    for(const auto& c:skin.donorMassContributions) {
        const auto local=c.nodeIndex-skin.firstGlobalFEMNode;
        if(local>=n)return fail("source contribution node outside skin/fat domain");
        if(affected[local])addMoment(oldMoments,c.donorBody,c.massKg,c.localPosition);
        else contributions.push_back(c);
    }
    for(const auto& [key,mass]:nextContributions) {
        const auto& source=bundle.parent.sourceMassContributions[key.second];
        addMoment(newMoments,source.donorBodyIndex,mass,source.donorLocalPosition);
        contributions.push_back({skin.firstGlobalFEMNode+key.first,source.donorBodyIndex,mass,source.donorLocalPosition});
    }
    for(const auto& [donor,old]:oldMoments) {
        const auto& next=newMoments[donor];
        for(unsigned a=0;a<4u;++a) {
            const double change=std::abs(next[a]-old[a]);
            if(change>1e-10*std::max({std::abs(old[a]),std::abs(old[0]),1e-12}))
                return fail("source donor mass/first moment changed");
            if(a==0u)report.maximumSourceMassChangeKg=std::max(report.maximumSourceMassChangeKg,change);
            else report.maximumSourceMomentChangeKgM=std::max(report.maximumSourceMomentChangeKgM,change);
        }
    }
    for(const auto& [donor,m]:newMoments)if(!oldMoments.contains(donor))return fail("introduced source donor");
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("numi.human.fixed-cavity-initialization-composition.v1");
    hash.string(bundle.massMappingPathSHA256);hash.string(report.proposalSHA256);
    for(auto word:object.femReferenceSourceIdentity)hash.u64(word);
    for(auto word:object.femTetrahedronMassSourceIdentity)hash.u64(word);
    for(const auto& p:patches) {
        for(auto row:p.oldRows)hash.u64(row);
        for(const auto& t:p.cells)for(auto v:t)hash.u64(v);
        for(double mass:p.mass)hash.u64(std::bit_cast<std::uint64_t>(mass));
        for(const auto& [row,mass]:p.parentMass){hash.u64(row);hash.u64(std::bit_cast<std::uint64_t>(mass));}
        if(!p.restrictedParentMass.empty()) {
            hash.string("numi.human.fixed-cavity-restricted-source-routing.v1");
            for(const auto& cell:p.restrictedParentMass) {
                hash.u64(cell.size());
                for(const auto& [row,mass]:cell){hash.u64(row);hash.u64(std::bit_cast<std::uint64_t>(mass));}
            }
        }
    }
    report.compositionSHA256=hash.finish();
    std::array<std::uint64_t,4> identity{};
    for(unsigned w=0;w<4u;++w)
        if(std::from_chars(report.compositionSHA256.data()+16u*w,report.compositionSHA256.data()+16u*(w+1u),identity[w],16).ec!=std::errc{})
            return fail("invalid composed identity");
    RegisteredSkinFatReductionSourceLabels nextLabels=labels;
    const auto remap=[&](auto& target,const auto& original) {
        if(original.size()!=nt)return false;
        target.clear();target.reserve(nextTets.size());
        for(auto row:retained)target.push_back(original[row]);
        for(const auto& p:patches)for(std::size_t j=0;j<p.cells.size();++j)target.push_back(original[p.oldRows[0]]);
        return true;
    };
    if(!remap(nextLabels.tetrahedronRegionLabels,labels.tetrahedronRegionLabels)||
       !remap(nextLabels.tetrahedronBasisRegionLabels,labels.tetrahedronBasisRegionLabels)||
       !remap(nextLabels.tetrahedronSourceFaceIndices,labels.tetrahedronSourceFaceIndices)||
       !remap(nextLabels.tetrahedronOriginalSourceFaceIndices,labels.tetrahedronOriginalSourceFaceIndices)||
       !remap(nextLabels.tetrahedronSourceDonorBodyIndices,labels.tetrahedronSourceDonorBodyIndices)||
       !remap(nextLabels.tetrahedronSourceParentBodyIndices,labels.tetrahedronSourceParentBodyIndices))
        return fail("source-label cell domains are incomplete");
    if(!labels.tetrahedronParentAncestryOffsets.empty()||
       !labels.tetrahedronParentAncestryIndices.empty()||
       !labels.tetrahedronParentAncestryVolumeFractions.empty()||
       labels.ancestryVerificationKind!=VolumeAncestryVerificationKind::unverified||
       !labels.ancestryVerificationReceiptSHA256.empty()||
       !remap(nextLabels.tetrahedronParentTetrahedronIndices,labels.tetrahedronParentTetrahedronIndices))
        return fail("requires spatial mass labels without a geometric-overlap ancestry claim");
    // Complete immediate construction ancestry is the authenticated proposal
    // CSR plus the original imported source, not a fabricated 330-overlap CSR.
    // Registered O/I/D and authored head-wall facets remain exact triangles,
    // independent of which interior tetrahedron now owns each facet.
    const auto facetCount=4u*nt;
    if(labels.tetrahedronRegisteredLayerFacetSurfaceFaces.size()!=facetCount||
       labels.tetrahedronRegisteredLayerFacetSourceFaceIdentities.size()!=facetCount||
       labels.tetrahedronAuthoredHeadSidewallFacetFlags.size()!=facetCount||
       labels.tetrahedronAuthoredHeadSidewallStableFaceIdentities.size()!=facetCount)
        return fail("missing exact registered/head-wall facet metadata");
    struct Facet {std::uint32_t face=NM_INVALID_INDEX,source=NM_INVALID_INDEX,wall=NM_INVALID_INDEX;std::uint8_t flag=0u;};
    std::map<Triangle,Facet> patchFacets;
    for(const auto& p:patches)for(auto row:p.oldRows)for(unsigned f=0;f<4u;++f) {
        Triangle key;for(unsigned k=0;k<3u;++k)key[k]=object.tetrahedra[row].nodes[faces[f][k]];
        std::sort(key.begin(),key.end());const auto e=4u*row+f;
        const Facet value{labels.tetrahedronRegisteredLayerFacetSurfaceFaces[e],
            labels.tetrahedronRegisteredLayerFacetSourceFaceIdentities[e],
            labels.tetrahedronAuthoredHeadSidewallStableFaceIdentities[e],
            labels.tetrahedronAuthoredHeadSidewallFacetFlags[e]};
        const auto [it,inserted]=patchFacets.emplace(key,value);
        if(!inserted&&(it->second.face!=value.face||it->second.source!=value.source||
                      it->second.wall!=value.wall||it->second.flag!=value.flag))
            return fail("shared cavity facet has contradictory metadata");
    }
    const auto remapFacets=[&](auto& target,const auto& original,auto field) {
        target.clear();target.reserve(4u*nextTets.size());
        for(auto row:retained)for(unsigned f=0;f<4u;++f)target.push_back(original[4u*row+f]);
        for(const auto& p:patches)for(const auto& t:p.cells)for(const auto& f:faces) {
            Triangle key{t[f[0]],t[f[1]],t[f[2]]};std::sort(key.begin(),key.end());
            const auto found=patchFacets.find(key);
            target.push_back(field(found==patchFacets.end()?Facet{}:found->second));
        }
    };
    remapFacets(nextLabels.tetrahedronRegisteredLayerFacetSurfaceFaces,
        labels.tetrahedronRegisteredLayerFacetSurfaceFaces,[](const Facet& f){return f.face;});
    remapFacets(nextLabels.tetrahedronRegisteredLayerFacetSourceFaceIdentities,
        labels.tetrahedronRegisteredLayerFacetSourceFaceIdentities,[](const Facet& f){return f.source;});
    remapFacets(nextLabels.tetrahedronAuthoredHeadSidewallFacetFlags,
        labels.tetrahedronAuthoredHeadSidewallFacetFlags,[](const Facet& f){return f.flag;});
    remapFacets(nextLabels.tetrahedronAuthoredHeadSidewallStableFaceIdentities,
        labels.tetrahedronAuthoredHeadSidewallStableFaceIdentities,[](const Facet& f){return f.wall;});
    nextLabels.sourceTetrahedronCount=nextTets.size();
    nextLabels.massTransferReceiptSHA256=report.compositionSHA256;
    if(!validateRegisteredSkinFatSourceDomainMetadata(nextLabels,error))return false;
    std::string finalProposalSHA;
    if(!physical_skin_child_mass_detail::sha256File(proposalPath,finalProposalSHA,error)||
       finalProposalSHA!=report.proposalSHA256)return fail("proposal changed during composition");
    // All checks completed. Commit the initialization transformation atomically.
    object.tetrahedra=std::move(nextTets);object.femMaterialIndices=std::move(nextMaterials);
    object.femTetrahedronMassesKg=std::move(nextMass);
    object.femReferenceSourceIdentity=identity;object.femTetrahedronMassSourceIdentity=identity;
    skinCells.commit(skin);fatCells.commit(fat);skin.donorMassContributions=std::move(contributions);
    labels=std::move(nextLabels);
    for(std::size_t node=0;node<n;++node)if(affected[node])skin.donorMassNodes[node].massKg=nextAffectedMass[node];
    report.cavities=patchCount;report.outputCells=object.tetrahedra.size();
    report.affectedNodes=std::count(affected.begin(),affected.end(),std::uint8_t(1));
    output=std::move(report);error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue
