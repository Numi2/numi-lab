#pragma once
// One-time donor-aware transfer from reduced accepted FEM shell displacement
// to detailed render skin. No simulation/physics is performed here.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace numi::human {
class AcceptedFEMSkinTransfer {
public:
    using Point = std::array<double,3>;
    using Face = std::array<std::uint32_t,3>;
    struct alignas(16) GPUTransferRow {
        std::uint32_t coarseVertex[3]{};
        std::uint32_t donorBodyIndex = 0;
        float weight[3]{};
        float restDistance = 0;
    };
    struct Pose { Point translation{}; std::array<double,4> rotation{0,0,0,1}; };
    // Compiler-authored physical refinement ancestry. Identity rows retain the
    // existing source baseline; nonidentity rows prolong earlier coarse rows.
    // This is presentation metadata, not another mechanical state or anchor.
    struct BaselineProlongationRow {
        std::array<std::uint32_t,3> parent{};
        Point weight{};
    };
    // Stable admission witness: detail indices address the input arrays, sourceVertexIndex is detailSource[detailVertexIndex], and coarseFaceIndex addresses faces. A donor-tree projection has no coarse representative.
    struct ProjectionWitness {
        bool valid=false;
        std::size_t detailVertexIndex=std::numeric_limits<std::size_t>::max();
        std::uint32_t sourceVertexIndex=std::numeric_limits<std::uint32_t>::max();
        std::uint32_t donorBodyIndex=std::numeric_limits<std::uint32_t>::max();
        std::uint32_t coarseRepresentativeIndex=std::numeric_limits<std::uint32_t>::max();
        std::size_t coarseFaceIndex=std::numeric_limits<std::size_t>::max();
        Face coarseFaceVertices{};
        double distance=0;
    };
    struct BuildReport {
        std::size_t detailVertexCount=0, coarseVertexCount=0, faceCount=0;
        std::size_t exactAncestryRows=0, projectedRows=0, donorTreeCount=0, inactiveRows=0;
        std::size_t ancestryFallbackRows=0;
        double maximumProjectionDistance=0, rmsProjectionDistance=0;
        double maximumAncestryFallbackDistance=0, rmsAncestryFallbackDistance=0;
        ProjectionWitness maximumProjectionWitness{};
        ProjectionWitness maximumAncestryFallbackWitness{};
    };

    // Build once while registered triangles are in scope. Source IDs provide
    // exact one-hot ancestry when available. Otherwise a donor-filtered
    // triangle BVH supplies the nearest row; donors absent from the reduced
    // mesh require a source-connected coarse representative. When that map is
    // supplied it selects the local one-ring for every projected row, avoiding
    // a donor-wide nearest point that could jump across separate body patches.
    bool build(const std::vector<Point>& detailRest,
               const std::vector<std::uint32_t>& detailSource,
               const std::vector<std::uint32_t>& detailDonor,
               const std::vector<std::uint8_t>& detailUsed,
               const std::vector<Point>& coarseRest,
               const std::vector<std::uint32_t>& coarseSource,
               const std::vector<std::uint32_t>& coarseDonor,
               const std::vector<Face>& faces, BuildReport* report,
               std::string* error,
               double maxProjection=std::numeric_limits<double>::infinity(),
               const std::vector<std::uint32_t>* detailRepresentativeCoarseIndex=nullptr,
               const std::vector<Face>* coarseSourceFaces=nullptr,
               const std::vector<Point>* coarseSourceBarycentric=nullptr,
               const std::vector<BaselineProlongationRow>* coarseBaselineProlongation=nullptr) {
        clear();
        auto fail=[&](const char* s){ if(error)*error=s; clear(); return false; };
        if(detailRest.size()!=detailSource.size() ||
           detailRest.size()!=detailDonor.size() || detailRest.size()!=detailUsed.size() ||
           (detailRepresentativeCoarseIndex && detailRepresentativeCoarseIndex->size()!=detailRest.size()) ||
           coarseRest.size()!=coarseSource.size() ||
           coarseRest.size()!=coarseDonor.size())
            return fail("parallel vertex arrays differ in size");
        if(detailRest.empty() || coarseRest.empty() || faces.empty())
            return fail("empty detailed or coarse surface");
        if(!(maxProjection>=0) || std::isnan(maxProjection))
            return fail("invalid maximum projection distance");
        for(const auto& p:detailRest) if(!finite(p)) return fail("nonfinite detail position");
        for(const auto& p:coarseRest) if(!finite(p)) return fail("nonfinite coarse position");

        // Optional source-triangle ancestry registers interpolated coarse palm
        // nodes without inventing a nearest source vertex. Presentation uses the
        // same current detailed source points; FEM/contact keep their own state.
        if(bool(coarseSourceFaces)!=bool(coarseSourceBarycentric))
            return fail("coarse source embedding requires faces and barycentrics");
        if(coarseSourceFaces) {
            if(coarseSourceFaces->size()!=coarseRest.size() ||
               coarseSourceBarycentric->size()!=coarseRest.size())
                return fail("coarse source embedding size mismatch");
            for(std::size_t i=0;i<coarseRest.size();++i) {
                const auto& f=(*coarseSourceFaces)[i];
                const auto& b=(*coarseSourceBarycentric)[i];
                if(!finite(b) || b[0]<-1.0e-8 || b[1]<-1.0e-8 || b[2]<-1.0e-8 ||
                   std::abs(b[0]+b[1]+b[2]-1.0)>1.0e-8)
                    return fail("invalid coarse source barycentrics");
                for(auto v:f) if(v>=detailRest.size())
                    return fail("coarse source embedding index out of range");
                if(coarseSource[i]!=std::numeric_limits<std::uint32_t>::max()) {
                    bool exactSource=false;
                    for(int k=0;k<3;++k)
                        exactSource|=f[k]==coarseSource[i] && std::abs(b[k]-1.0)<=1.0e-8;
                    if(!exactSource)
                        return fail("coarse exact source identity differs from its embedding");
                }
            }
            coarseSourceFaces_=*coarseSourceFaces;
            coarseSourceBarycentric_=*coarseSourceBarycentric;
        }
        if(coarseBaselineProlongation) {
            if(coarseBaselineProlongation->size()!=coarseRest.size())
                return fail("coarse baseline prolongation size mismatch");
            for(std::size_t i=0;i<coarseRest.size();++i) {
                const auto& row=(*coarseBaselineProlongation)[i];
                double sum=0.0;bool self=false,other=false;Point reference{};
                for(unsigned k=0;k<3;++k) {
                    const auto parent=row.parent[k];const double w=row.weight[k];
                    if(parent>=coarseRest.size()||!std::isfinite(w)||w<0.0||w>1.0)
                        return fail("invalid coarse baseline prolongation parent or weight");
                    if(w>0.0) {
                        if(parent>i)return fail("coarse baseline prolongation has a forward dependency");
                        self|=parent==i;other|=parent!=i;
                    }
                    sum+=w;
                    for(unsigned a=0;a<3;++a)reference[a]+=w*coarseRest[parent][a];
                }
                if(std::abs(sum-1.0)>1.0e-12|| (self&&other))
                    return fail("coarse baseline prolongation is not normalized or has a self dependency");
                for(unsigned a=0;a<3;++a)
                    if(!std::isfinite(reference[a])||std::abs(reference[a]-coarseRest[i][a])>1.0e-10)
                        return fail("coarse baseline prolongation does not reproduce its physical rest point");
            }
            coarseBaselineProlongation_=*coarseBaselineProlongation;
        }
        coarseRest_=coarseRest;
        coarseSource_=coarseSource;
        coarseDonor_=coarseDonor;
        rows_.resize(detailRest.size());
        std::unordered_map<Key,std::uint32_t,KeyHash> exact;
        exact.reserve(coarseRest.size()*2);
        for(std::size_t i=0;i<coarseRest.size();++i)
            exact.emplace(Key{coarseDonor[i],coarseSource[i]},static_cast<std::uint32_t>(i));

        std::unordered_map<std::uint32_t,std::vector<Face>> donorFaces;
        std::unordered_map<std::uint32_t,std::vector<std::size_t>> donorFaceIndices;
        std::vector<std::vector<Face>> incidentFaces(coarseRest.size());
        std::vector<std::vector<std::size_t>> incidentFaceIndices(coarseRest.size());
        std::vector<Face> incidentFace(coarseRest.size());
        std::vector<std::uint8_t> hasIncident(coarseRest.size(),0), incidentCorner(coarseRest.size(),0);
        for(std::size_t faceIndex=0;faceIndex<faces.size();++faceIndex) {
            const Face& f=faces[faceIndex];
            if(f[0]>=coarseRest.size() || f[1]>=coarseRest.size() || f[2]>=coarseRest.size())
                return fail("coarse face index out of range");
            for(std::uint8_t k=0;k<3;++k) {
                incidentFaces[f[k]].push_back(f);
                incidentFaceIndices[f[k]].push_back(faceIndex);
                if(!hasIncident[f[k]]) { hasIncident[f[k]]=1; incidentFace[f[k]]=f; incidentCorner[f[k]]=k; }
            }
            // A mixed-donor triangle is valid transfer support for each donor
            // actually represented at one of its corners. Never promote it to
            // an unrelated donor tree or use a scene-wide fallback.
            for(std::uint8_t k=0;k<3;++k) {
                const auto donor=coarseDonor[f[k]];
                bool alreadyAdded=false;
                for(std::uint8_t j=0;j<k;++j) alreadyAdded|=(coarseDonor[f[j]]==donor);
                if(!alreadyAdded) {
                    donorFaces[donor].push_back(f);
                    donorFaceIndices[donor].push_back(faceIndex);
                }
            }
        }
        std::unordered_map<std::uint32_t,std::size_t> treeIndex;
        for(auto& entry:donorFaces) {
            if(entry.second.empty()) continue;
            treeIndex[entry.first]=trees_.size();
            trees_.emplace_back();
            trees_.back().faces=std::move(entry.second);
            trees_.back().faceIndices=std::move(donorFaceIndices[entry.first]);
            trees_.back().build(coarseRest);
        }
        BuildReport r;
        r.detailVertexCount=detailRest.size(); r.coarseVertexCount=coarseRest.size();
        r.faceCount=faces.size(); r.donorTreeCount=trees_.size();
        double sum2=0, maxd=0, ancestrySum2=0, ancestryMax=0;
        const auto makeProjectionWitness=[&](std::size_t detailIndex,
                                             std::uint32_t representativeIndex,
                                             const Face& face,
                                             std::size_t faceIndex,
                                             double distance) {
            ProjectionWitness witness{};
            witness.valid=true;
            witness.detailVertexIndex=detailIndex;
            witness.sourceVertexIndex=detailSource[detailIndex];
            witness.donorBodyIndex=detailDonor[detailIndex];
            witness.coarseRepresentativeIndex=representativeIndex;
            witness.coarseFaceIndex=faceIndex;
            witness.coarseFaceVertices=face;
            witness.distance=distance;
            return witness;
        };
        for(std::size_t i=0;i<detailRest.size();++i) {
            if(!detailUsed[i]) { rows_[i].donorBodyIndex=detailDonor[i]; ++r.inactiveRows; continue; }
            auto it=exact.find(Key{detailDonor[i],detailSource[i]});
            if(it!=exact.end()) {
                if(hasIncident[it->second]) setExactOnFace(rows_[i],incidentFace[it->second],incidentCorner[it->second],detailDonor[i]);
                else setExact(rows_[i],it->second,detailDonor[i]);
                ++r.exactAncestryRows;
                continue;
            }
            Hit h;
            std::uint32_t representativeIndex=std::numeric_limits<std::uint32_t>::max();
            auto ti=treeIndex.find(detailDonor[i]);
            if(!detailRepresentativeCoarseIndex && ti!=treeIndex.end()) {
                if(!trees_[ti->second].nearest(detailRest[i],coarseRest,h))
                    return fail("donor BVH found no valid triangle");
            } else {
                if(!detailRepresentativeCoarseIndex)
                    return fail("no coarse faces for detail donor and no source ancestry map");
                representativeIndex=(*detailRepresentativeCoarseIndex)[i];
                if(representativeIndex==std::numeric_limits<std::uint32_t>::max())
                    return fail("active detail vertex has no source-connected coarse representative");
                if(representativeIndex>=coarseRest.size())
                    return fail("source ancestry representative index is out of range");
                const std::uint32_t coarseVertex=representativeIndex;
                if(incidentFaces[coarseVertex].empty())
                    return fail("source ancestry representative has no incident coarse face");
                for(std::size_t localFace=0;localFace<incidentFaces[coarseVertex].size();++localFace) {
                    const Face& f=incidentFaces[coarseVertex][localFace];
                    std::array<double,3> bary{};
                    const Point q=closest(detailRest[i],coarseRest[f[0]],coarseRest[f[1]],coarseRest[f[2]],bary);
                    const double d=std::sqrt(dist2(detailRest[i],q));
                    if(d<h.distance) {
                        h.face=f; h.faceIndex=incidentFaceIndices[coarseVertex][localFace];
                        h.bary=bary; h.distance=d;
                    }
                }
                const double parentDistance=std::sqrt(dist2(detailRest[i],coarseRest[coarseVertex]));
                const double boundTolerance=1.0e-10*std::max(1.0,parentDistance);
                if(!std::isfinite(h.distance) || h.distance>parentDistance+boundTolerance)
                    return fail("incident-face projection exceeds source representative distance bound");
                ++r.ancestryFallbackRows;
                ancestrySum2+=h.distance*h.distance;
                if(!r.maximumAncestryFallbackWitness.valid || h.distance>ancestryMax) {
                    ancestryMax=h.distance;
                    r.maximumAncestryFallbackWitness=makeProjectionWitness(
                        i,representativeIndex,h.face,h.faceIndex,h.distance);
                }
            }
            if(h.distance>maxProjection) return fail("projection exceeds configured distance");
            auto& row=rows_[i];
            for(int k=0;k<3;++k) { row.coarseVertex[k]=h.face[k]; row.weight[k]=float(h.bary[k]); }
            row.donorBodyIndex=detailDonor[i]; row.restDistance=float(h.distance);
            ++r.projectedRows; sum2+=h.distance*h.distance;
            if(!r.maximumProjectionWitness.valid || h.distance>maxd) {
                maxd=h.distance;
                r.maximumProjectionWitness=makeProjectionWitness(
                    i,representativeIndex,h.face,h.faceIndex,h.distance);
            }
        }
        r.maximumProjectionDistance=maxd;
        r.rmsProjectionDistance=r.projectedRows ? std::sqrt(sum2/r.projectedRows):0;
        r.maximumAncestryFallbackDistance=ancestryMax;
        r.rmsAncestryFallbackDistance=r.ancestryFallbackRows ? std::sqrt(ancestrySum2/r.ancestryFallbackRows):0;
        if(report)*report=r;
        if(error)error->clear();
        return true;
    }

    const std::vector<GPUTransferRow>& rows() const { return rows_; }
    std::size_t detailVertexCount() const { return rows_.size(); }

    // Carry each coarse source-to-shell rest offset in its donor body's local
    // frame. This removes remeshing offsets at zero FEM displacement.
    bool makeCurrentCoarseBaseline(const std::vector<Point>& sourceRestByID,
             const std::vector<Point>& sourceCurrentByID,
             const std::vector<Pose>& donorRestPose,
             const std::vector<Pose>& donorCurrentPose,
             std::vector<Point>& out, std::string* error) const {
        out.resize(coarseSource_.size());
        for(std::size_t i=0;i<coarseSource_.size();++i) {
            const auto donor=coarseDonor_[i];
            Point sourceRest{},sourceCurrent{};
            if(coarseSourceFaces_.empty()) {
                const auto source=coarseSource_[i];
                if(source>=sourceRestByID.size() || source>=sourceCurrentByID.size()) {
                    if(error)*error="coarse source is out of range";
                    out.clear(); return false;
                }
                sourceRest=sourceRestByID[source]; sourceCurrent=sourceCurrentByID[source];
            } else {
                const auto& f=coarseSourceFaces_[i];
                const auto& b=coarseSourceBarycentric_[i];
                for(int k=0;k<3;++k) {
                    if(f[k]>=sourceRestByID.size() || f[k]>=sourceCurrentByID.size() ||
                       !finite(sourceRestByID[f[k]]) || !finite(sourceCurrentByID[f[k]])) {
                        if(error)*error="coarse embedded source is invalid";
                        out.clear(); return false;
                    }
                    for(int j=0;j<3;++j) {
                        sourceRest[j]+=b[k]*sourceRestByID[f[k]][j];
                        sourceCurrent[j]+=b[k]*sourceCurrentByID[f[k]][j];
                    }
                }
            }
            if(donor>=donorRestPose.size() || donor>=donorCurrentPose.size()) {
                if(error)*error="coarse donor pose is out of range";
                out.clear(); return false;
            }
            const Pose& rest=donorRestPose[donor];
            const Pose& current=donorCurrentPose[donor];
            if(!finite(rest.translation)||!finite(current.translation)||
               !finite(sourceRest)||!finite(sourceCurrent)||
               !finite(coarseRest_[i])||!finite(rest.rotation)||!finite(current.rotation)) {
                if(error)*error="nonfinite pose or baseline position";
                out.clear(); return false;
            }
            auto qr=rest.rotation, qc=current.rotation;
            auto norm=[](const std::array<double,4>& q){double n=0;for(double v:q)n+=v*v;return std::sqrt(n);};
            const double nr=norm(qr), nc=norm(qc);
            if(!(nr>1e-12) || !(nc>1e-12)) {
                if(error)*error="invalid donor quaternion"; out.clear(); return false;
            }
            for(double& v:qr)v/=nr;
            for(double& v:qc)v/=nc;
            Point offset=sub(coarseRest_[i],sourceRest);
            Point local=rotate(Point{-qr[0],-qr[1],-qr[2]},qr[3],offset);
            Point world=rotate(Point{qc[0],qc[1],qc[2]},qc[3],local);
            out[i]={sourceCurrent[0]+world[0],sourceCurrent[1]+world[1],sourceCurrent[2]+world[2]};
        }
        // A refined physical midpoint must use the same prolongation for its
        // presentation baseline as for accepted positions. Re-evaluating a new
        // source projection with one donor rotation need not commute with a
        // mixed-owner parent edge, and would otherwise change the detailed skin
        // merely by subdividing an unchanged physical deformation.
        for(std::size_t i=0;i<coarseBaselineProlongation_.size();++i) {
            const auto& row=coarseBaselineProlongation_[i];
            bool identity=false;
            for(unsigned k=0;k<3;++k)identity|=row.weight[k]>0.0&&row.parent[k]==i;
            if(identity)continue; // Preserve legacy arithmetic exactly.
            Point value{};
            for(unsigned k=0;k<3;++k)if(row.weight[k]>0.0)
                for(unsigned a=0;a<3;++a)value[a]+=row.weight[k]*out[row.parent[k]][a];
            out[i]=value;
        }
        if(error)error->clear(); return true;
    }

    // acceptedOuterPositions use coarse-shell local order. Baseline positions
    // come from makeCurrentCoarseBaseline at the same accepted presentation pose.
    bool makeCoarseDisplacements(const std::vector<Point>& acceptedOuterPositions,
             const std::vector<Point>& currentMechanicalBaseline,
             std::vector<std::array<float,4>>& out, std::string* error) const {
        if(acceptedOuterPositions.size()!=coarseSource_.size() ||
           currentMechanicalBaseline.size()!=coarseSource_.size()) {
            if(error)*error="accepted outer FEM or current baseline count mismatch";
            return false;
        }
        out.resize(coarseSource_.size());
        for(std::size_t i=0;i<coarseSource_.size();++i) {
            const auto& a=acceptedOuterPositions[i]; const auto& b=currentMechanicalBaseline[i];
            if(!finite(a)||!finite(b)) {
                if(error)*error="nonfinite accepted or baseline position"; out.clear(); return false;
            }
            out[i]={float(a[0]-b[0]),float(a[1]-b[1]),float(a[2]-b[2]),0};
        }
        if(error)error->clear(); return true;
    }
    // CPU reference for parity tests; production applies the same interpolation
    // in a render-only Metal pass after ordinary skeletal skinning.
    bool applyReference(const std::vector<Point>& skeletalDetail,
             const std::vector<std::array<float,4>>& delta,
             std::vector<Point>& out, std::string* error) const {
        if(skeletalDetail.size()!=rows_.size()) {
            if(error)*error="detailed skin vertex count mismatch"; return false;
        }
        out.resize(rows_.size());
        for(std::size_t i=0;i<rows_.size();++i) {
            Point p=skeletalDetail[i]; const auto& row=rows_[i];
            for(int k=0;k<3;++k) {
                if(row.coarseVertex[k]>=delta.size()) {
                    if(error)*error="transfer coarse index out of range"; out.clear(); return false;
                }
                for(int a=0;a<3;++a) p[a]+=double(row.weight[k])*delta[row.coarseVertex[k]][a];
            }
            out[i]=p;
        }
        if(error)error->clear(); return true;
    }

private:
    struct Key { std::uint32_t donor,source; bool operator==(const Key& x)const{return donor==x.donor&&source==x.source;} };
    struct KeyHash { std::size_t operator()(const Key& x)const{return (std::size_t(x.donor)<<32)^x.source;} };
    struct Box {
        Point lo{INFINITY,INFINITY,INFINITY}, hi{-INFINITY,-INFINITY,-INFINITY};
    };
    struct Hit {
        Face face{};
        std::size_t faceIndex=std::numeric_limits<std::size_t>::max();
        std::array<double,3> bary{};
        double distance=INFINITY;
    };
    struct Node { Box box; std::uint32_t begin=0,end=0; int left=-1,right=-1; };
    struct Tree {
        std::vector<Face> faces;
        std::vector<std::size_t> faceIndices;
        std::vector<std::uint32_t> order;
        std::vector<Node> nodes;
        void build(const std::vector<Point>& p) {
            order.resize(faces.size()); std::iota(order.begin(),order.end(),0);
            nodes.clear(); buildNode(0,std::uint32_t(order.size()),p);
        }
        int buildNode(std::uint32_t b,std::uint32_t e,const std::vector<Point>& p) {
            Node n; n.begin=b;n.end=e; Box cent;
            for(auto j=b;j<e;++j) {
                const auto& f=faces[order[j]]; Point c{};
                for(int k=0;k<3;++k) for(int a=0;a<3;++a) {
                    n.box.lo[a]=std::min(n.box.lo[a],p[f[k]][a]);
                    n.box.hi[a]=std::max(n.box.hi[a],p[f[k]][a]);
                    c[a]+=p[f[k]][a]/3.0;
                }
                for(int a=0;a<3;++a){cent.lo[a]=std::min(cent.lo[a],c[a]);cent.hi[a]=std::max(cent.hi[a],c[a]);}
            }
            int id=int(nodes.size()); nodes.push_back(n);
            if(e-b<=8)return id;
            int axis=0;
            for(int a=1;a<3;++a)if(cent.hi[a]-cent.lo[a]>cent.hi[axis]-cent.lo[axis])axis=a;
            auto m=b+(e-b)/2;
            std::nth_element(order.begin()+b,order.begin()+m,order.begin()+e,[&](auto x,auto y){
                const auto& a=faces[x];const auto& c=faces[y];
                return (p[a[0]][axis]+p[a[1]][axis]+p[a[2]][axis]) <
                       (p[c[0]][axis]+p[c[1]][axis]+p[c[2]][axis]);
            });
            nodes[id].left=buildNode(b,m,p); nodes[id].right=buildNode(m,e,p); return id;
        }
        bool nearest(const Point& q,const std::vector<Point>& p,Hit& h)const {
            if(nodes.empty())return false; visit(0,q,p,h); return std::isfinite(h.distance);
        }
        void visit(int id,const Point& q,const std::vector<Point>& p,Hit& h)const {
            const auto& n=nodes[id]; if(boxDist2(n.box,q)>h.distance*h.distance)return;
            if(n.left<0) {
                for(auto j=n.begin;j<n.end;++j) {
                    const auto faceSlot=order[j];
                    const auto& f=faces[faceSlot]; std::array<double,3>w{};
                    auto x=closest(q,p[f[0]],p[f[1]],p[f[2]],w); double d=dist2(q,x);
                    if(d<h.distance*h.distance) {
                        h.face=f; h.faceIndex=faceIndices[faceSlot];
                        h.bary=w; h.distance=std::sqrt(d);
                    }
                } return;
            }
            auto dl=boxDist2(nodes[n.left].box,q),dr=boxDist2(nodes[n.right].box,q);
            if(dl<dr){visit(n.left,q,p,h);visit(n.right,q,p,h);}
            else{visit(n.right,q,p,h);visit(n.left,q,p,h);}
        }
    };
    static bool finite(const Point&p){return std::isfinite(p[0])&&std::isfinite(p[1])&&std::isfinite(p[2]);}
    static bool finite(const std::array<double,4>& q){for(double x:q)if(!std::isfinite(x))return false;return true;}
    static Point rotate(const Point&q,double w,const Point&v){
        Point a{q[1]*v[2]-q[2]*v[1],q[2]*v[0]-q[0]*v[2],q[0]*v[1]-q[1]*v[0]};
        Point b{q[1]*a[2]-q[2]*a[1],q[2]*a[0]-q[0]*a[2],q[0]*a[1]-q[1]*a[0]};
        return {v[0]+2*(b[0]+w*a[0]),v[1]+2*(b[1]+w*a[1]),v[2]+2*(b[2]+w*a[2])};
    }
    static double dist2(const Point&a,const Point&b){double d=0;for(int i=0;i<3;++i){double x=a[i]-b[i];d+=x*x;}return d;}
    static double boxDist2(const Box&b,const Point&p){double d=0;for(int i=0;i<3;++i){double x=p[i]<b.lo[i]?b.lo[i]-p[i]:p[i]>b.hi[i]?p[i]-b.hi[i]:0;d+=x*x;}return d;}
    static Point sub(const Point&a,const Point&b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
    static double dot(const Point&a,const Point&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
    static Point closestSegment(const Point&p,const Point&a,const Point&b,double& t){
        auto ab=sub(b,a);double d=dot(ab,ab);t=d>0?std::clamp(dot(sub(p,a),ab)/d,0.0,1.0):0.0;
        return {a[0]+t*ab[0],a[1]+t*ab[1],a[2]+t*ab[2]};
    }
    static Point closest(const Point&p,const Point&a,const Point&b,const Point&c,std::array<double,3>&w){
        auto ab=sub(b,a),ac=sub(c,a),ap=sub(p,a);double d1=dot(ab,ap),d2=dot(ac,ap);
        if(d1<=0&&d2<=0){w={1,0,0};return a;}
        auto bp=sub(p,b);double d3=dot(ab,bp),d4=dot(ac,bp);
        if(d3>=0&&d4<=d3){w={0,1,0};return b;}
        double vc=d1*d4-d3*d2;
        if(vc<=0&&d1>=0&&d3<=0){double v=d1/(d1-d3);w={1-v,v,0};return {a[0]+v*ab[0],a[1]+v*ab[1],a[2]+v*ab[2]};}
        auto cp=sub(p,c);double d5=dot(ab,cp),d6=dot(ac,cp);
        if(d6>=0&&d5<=d6){w={0,0,1};return c;}
        double vb=d5*d2-d1*d6;
        if(vb<=0&&d2>=0&&d6<=0){double v=d2/(d2-d6);w={1-v,0,v};return {a[0]+v*ac[0],a[1]+v*ac[1],a[2]+v*ac[2]};}
        double va=d3*d6-d5*d4;
        if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){double v=(d4-d3)/((d4-d3)+(d5-d6));w={0,1-v,v};return {b[0]+v*(c[0]-b[0]),b[1]+v*(c[1]-b[1]),b[2]+v*(c[2]-b[2])};}
        double den=va+vb+vc;
        if(!(std::abs(den)>1e-30)) {
            double t0,t1,t2; auto q0=closestSegment(p,a,b,t0),q1=closestSegment(p,b,c,t1),q2=closestSegment(p,c,a,t2);
            double e0=dist2(p,q0),e1=dist2(p,q1),e2=dist2(p,q2);
            if(e0<=e1&&e0<=e2){w={1-t0,t0,0};return q0;}
            if(e1<=e2){w={0,1-t1,t1};return q1;}
            w={t2,0,1-t2};return q2;
        }
        double v=vb/den,z=vc/den,u=1-v-z; w={u,v,z};
        return {u*a[0]+v*b[0]+z*c[0],u*a[1]+v*b[1]+z*c[1],u*a[2]+v*b[2]+z*c[2]};
    }
    static void setExactOnFace(GPUTransferRow&r,const Face&f,std::uint8_t corner,std::uint32_t d){
        for(int k=0;k<3;++k)r.coarseVertex[k]=f[k];r.donorBodyIndex=d;
        r.weight[0]=r.weight[1]=r.weight[2]=0;r.weight[corner]=1;r.restDistance=0;
    }
    static void setExact(GPUTransferRow&r,std::uint32_t v,std::uint32_t d){
        r.coarseVertex[0]=r.coarseVertex[1]=r.coarseVertex[2]=v;r.donorBodyIndex=d;
        r.weight[0]=1;r.weight[1]=r.weight[2]=0;r.restDistance=0;
    }
    void clear(){rows_.clear();coarseRest_.clear();coarseSource_.clear();coarseDonor_.clear();coarseSourceFaces_.clear();coarseSourceBarycentric_.clear(); coarseBaselineProlongation_.clear();trees_.clear();}
    std::vector<GPUTransferRow> rows_;
    std::vector<Point> coarseRest_;
    std::vector<std::uint32_t> coarseSource_,coarseDonor_;
    std::vector<Face> coarseSourceFaces_;
    std::vector<Point> coarseSourceBarycentric_;
    std::vector<BaselineProlongationRow> coarseBaselineProlongation_;
    std::vector<Tree> trees_;
};
static_assert(sizeof(AcceptedFEMSkinTransfer::GPUTransferRow)==32,"Metal row ABI");
static_assert(alignof(AcceptedFEMSkinTransfer::GPUTransferRow)==16,"Metal row alignment");
} // namespace numi::human

