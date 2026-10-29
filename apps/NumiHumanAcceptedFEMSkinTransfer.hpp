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
#include <sstream>
#include <unordered_map>
#include <unordered_set>
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
    // Compiler-authored affine embedding into one accepted physical tetrahedron.
    // Indices address the full physical volume, never detailed/source row IDs.
    // Effective weights include the reference residual and sum to one.
    struct GPUTetrahedralTransferRow {
        std::uint32_t node[4]{};
        float weight[4]{};
    };
    static_assert(sizeof(GPUTetrahedralTransferRow)==32, "tetrahedral transfer row ABI");
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

    // Opt-in prepared refined O domain. Transfer-row vertex indices address
    // the full accepted FEM node domain, while faces/source embeddings use the
    // compact coarse-vertex order. Digests are caller-pinned source/candidate
    // identities; unknown or mismatched identities fail closed.
    struct PreparedRefinedDomain {
        std::size_t originalCoarseVertexCount=0;
        std::size_t mechanicalNodeCount=0;
        // Source identity is pinned when the original owner is built. The
        // candidate identity is checked against a separately trusted manifest
        // value supplied to installPreparedRefinedDomain.
        std::array<std::uint64_t,4> sourceIdentity{};
        std::array<std::uint64_t,4> candidateIdentity{};
        std::vector<Point> coarseRest;
        std::vector<std::uint32_t> coarseSource,coarseDonor;
        std::vector<std::uint32_t> coarseMechanicalNodeRows;
        std::vector<Face> faces;
        std::vector<std::uint32_t> originalFaceStableIDs,refinedFaceStableIDs;
        std::vector<std::uint32_t> detailSourceIDs,detailParentStableFaceIDs;
        std::vector<std::uint32_t> detailSelectedFaceRows;
        std::vector<Face> coarseSourceFaces;
        std::vector<Point> coarseSourceBarycentric;
        std::vector<BaselineProlongationRow> coarseBaselineProlongation;
        std::vector<std::uint32_t> coarseFaceSourceOffsets;
        std::vector<std::uint32_t> coarseFaceStableSourceIDs;
        std::vector<GPUTransferRow> rows;
        double baselineToleranceMeters=1.0e-8;
        double maximumTransferPointErrorMeters=2.0e-7;
    };
    struct RefinedDomainReport {
        std::size_t originalCoarseVertexCount=0,refinedCoarseVertexCount=0;
        std::size_t mechanicalNodeCount=0,detailRowCount=0;
        std::size_t activeRows=0,inactiveRows=0,generatedCoarseVertices=0;
        double maximumBaselineReconstructionErrorMeters=0;
        double maximumTransferPointErrorMeters=0;
        std::array<std::uint64_t,4> sourceIdentity{},candidateIdentity{};
    };
    using DetailFaceEligibilityPredicate=bool(*)(const void*,std::size_t,std::size_t);

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
               const std::vector<BaselineProlongationRow>* coarseBaselineProlongation=nullptr,
               std::size_t sourceConnectedFaceRings=1,
               const std::vector<std::uint32_t>* stableFaceIds=nullptr,
               const std::array<std::uint64_t,4>* sourceIdentity=nullptr,
               DetailFaceEligibilityPredicate detailFaceEligibility=nullptr,
               const void* detailFaceEligibilityContext=nullptr) {
        clear();
        auto fail=[&](const char* s){ if(error)*error=s; clear(); return false; };
        if(detailRest.size()!=detailSource.size() ||
           detailRest.size()!=detailDonor.size() || detailRest.size()!=detailUsed.size() ||
           (detailRepresentativeCoarseIndex && detailRepresentativeCoarseIndex->size()!=detailRest.size()) ||
           coarseRest.size()!=coarseSource.size() ||
           coarseRest.size()!=coarseDonor.size()||
           (detailFaceEligibility==nullptr&&detailFaceEligibilityContext!=nullptr))
            return fail("parallel vertex arrays or typed face-eligibility context differ in size");
        if(detailRest.empty() || coarseRest.empty() || faces.empty())
            return fail("empty detailed or coarse surface");
        if(!(maxProjection>=0) || std::isnan(maxProjection))
            return fail("invalid maximum projection distance");
        if(sourceConnectedFaceRings<1 || sourceConnectedFaceRings>3)
            return fail("source-connected face ring bound must be in [1,3]");
        if(stableFaceIds && stableFaceIds->size()!=faces.size())
            return fail("stable coarse face identity count differs from topology");
        if(sourceIdentity) {
            bool nonzero=false;
            for(const auto word:*sourceIdentity)nonzero|=word!=0;
            if(!nonzero)return fail("source identity pin is empty");
        }
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
        baseCoarseRest_=coarseRest;
        baseCoarseSource_=coarseSource;
        baseCoarseDonor_=coarseDonor;
        originalFaces_=faces;
        detailSource_=detailSource;
        detailDonor_=detailDonor;
        detailUsed_=detailUsed;
        detailVertexCount_=detailRest.size();
        baseCoarseSourceFaces_=coarseSourceFaces_;
        baseCoarseSourceBarycentric_=coarseSourceBarycentric_;
        baseCoarseBaselineProlongation_=coarseBaselineProlongation_;
        if(stableFaceIds)baseStableFaceIds_=*stableFaceIds;
        hasBaseSourceIdentity_=sourceIdentity!=nullptr;
        if(sourceIdentity)baseSourceIdentity_=*sourceIdentity;
        rows_.resize(detailRest.size());
        selectedFaceIndices_.assign(detailRest.size(),std::numeric_limits<std::uint32_t>::max());
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
        // Shared-edge adjacency keeps expanded candidates in the source
        // representative's connected coarse-surface component. Ring 1 is
        // exactly the legacy incident-face set; higher rings are opt-in.
        std::vector<std::vector<std::size_t>> faceNeighbors(faces.size());
        std::unordered_map<std::uint64_t,std::vector<std::size_t>> edgeFaces;
        edgeFaces.reserve(faces.size()*2);
        const auto edgeKey=[](std::uint32_t a,std::uint32_t b) {
            if(a>b)std::swap(a,b);
            return (std::uint64_t(a)<<32)|std::uint64_t(b);
        };
        for(std::size_t fi=0;fi<faces.size();++fi) {
            const auto& f=faces[fi];
            edgeFaces[edgeKey(f[0],f[1])].push_back(fi);
            edgeFaces[edgeKey(f[1],f[2])].push_back(fi);
            edgeFaces[edgeKey(f[2],f[0])].push_back(fi);
        }
        for(const auto& edge:edgeFaces) {
            const auto& owners=edge.second;
            for(std::size_t a=0;a<owners.size();++a)
                for(std::size_t b=a+1;b<owners.size();++b) {
                    faceNeighbors[owners[a]].push_back(owners[b]);
                    faceNeighbors[owners[b]].push_back(owners[a]);
                }
        }
        for(auto& adjacent:faceNeighbors) {
            std::sort(adjacent.begin(),adjacent.end());
            adjacent.erase(std::unique(adjacent.begin(),adjacent.end()),adjacent.end());
        }
        const auto stableFaceId=[&](std::size_t faceIndex) -> std::uint32_t {
            return stableFaceIds ? (*stableFaceIds)[faceIndex] : std::uint32_t(faceIndex);
        };
        const auto stableTieBetter=[&](std::size_t candidate,std::size_t incumbent) {
            if(incumbent>=faces.size())return true;
            const auto a=stableFaceId(candidate),b=stableFaceId(incumbent);
            return a<b || (a==b && candidate<incumbent);
        };
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
        std::vector<std::uint32_t> faceVisitStamp(faces.size(),0), faceDepth(faces.size(),0);
        std::uint32_t faceVisitEpoch=0;
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
        std::size_t eligibilityFailures=0u;
        std::ostringstream eligibilityWitnesses;
        for(std::size_t i=0;i<detailRest.size();++i) {
            if(!detailUsed[i]) { rows_[i].donorBodyIndex=detailDonor[i]; ++r.inactiveRows; continue; }
            auto it=exact.find(Key{detailDonor[i],detailSource[i]});
            if(it!=exact.end()) {
                bool exactAccepted=false;
                if(hasIncident[it->second]) {
                    for(std::size_t q=0;q<incidentFaceIndices[it->second].size();++q) {
                        const auto faceIndex=incidentFaceIndices[it->second][q];
                        if(detailFaceEligibility&&!detailFaceEligibility(
                                detailFaceEligibilityContext,i,faceIndex))continue;
                        const auto corner=std::find(faces[faceIndex].begin(),
                            faces[faceIndex].end(),it->second);
                        if(corner==faces[faceIndex].end())continue;
                        setExactOnFace(rows_[i],faces[faceIndex],
                            std::uint8_t(corner-faces[faceIndex].begin()),detailDonor[i]);
                        selectedFaceIndices_[i]=std::uint32_t(faceIndex);
                        exactAccepted=true;break;
                    }
                } else if(!detailFaceEligibility) {
                    setExact(rows_[i],it->second,detailDonor[i]);exactAccepted=true;
                }
                if(exactAccepted) {
                    ++r.exactAncestryRows;continue;
                }
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
                if(++faceVisitEpoch==0) {
                    std::fill(faceVisitStamp.begin(),faceVisitStamp.end(),0);
                    faceVisitEpoch=1;
                }
                std::vector<std::size_t> frontier;
                for(const auto faceIndex:incidentFaceIndices[coarseVertex]) {
                    if(faceVisitStamp[faceIndex]!=faceVisitEpoch) {
                        faceVisitStamp[faceIndex]=faceVisitEpoch;
                        faceDepth[faceIndex]=0;
                        frontier.push_back(faceIndex);
                    }
                }
                std::vector<std::size_t> candidateFaces=frontier;
                for(std::size_t ring=1;ring<sourceConnectedFaceRings;++ring) {
                    std::vector<std::size_t> next;
                    for(const auto faceIndex:frontier) for(const auto adjacent:faceNeighbors[faceIndex]) {
                        if(faceVisitStamp[adjacent]==faceVisitEpoch)continue;
                        faceVisitStamp[adjacent]=faceVisitEpoch;
                        faceDepth[adjacent]=std::uint32_t(ring);
                        next.push_back(adjacent);
                        candidateFaces.push_back(adjacent);
                    }
                    frontier.swap(next);
                    if(frontier.empty())break;
                }
                if(sourceConnectedFaceRings>1)
                    std::sort(candidateFaces.begin(),candidateFaces.end(),[&](auto a,auto b) {
                        if(stableFaceId(a)!=stableFaceId(b))return stableFaceId(a)<stableFaceId(b);
                        return a<b;
                    });
                for(const auto faceIndex:candidateFaces) {
                    if(detailFaceEligibility&&!detailFaceEligibility(
                            detailFaceEligibilityContext,i,faceIndex))continue;
                    const Face& f=faces[faceIndex];
                    bool sameDonor=false;
                    for(unsigned k=0;k<3;++k)sameDonor|=coarseDonor[f[k]]==detailDonor[i];
                    // Preserve the full legacy seed set, including mixed
                    // donor faces incident on its representative. Only newly
                    // reached faces must directly support the detail donor.
                    if(faceDepth[faceIndex]>0 && !sameDonor)continue;
                    std::array<double,3> bary{};
                    const Point q=closest(detailRest[i],coarseRest[f[0]],coarseRest[f[1]],coarseRest[f[2]],bary);
                    const double d=std::sqrt(dist2(detailRest[i],q));
                    const bool deterministicTie=sourceConnectedFaceRings>1
                        ? stableTieBetter(faceIndex,h.faceIndex) : faceIndex<h.faceIndex;
                    if(d<h.distance || (d==h.distance && deterministicTie)) {
                        h.face=f; h.faceIndex=faceIndex; h.bary=bary; h.distance=d;
                    }
                }
                const double parentDistance=std::sqrt(dist2(detailRest[i],coarseRest[coarseVertex]));
                const double boundTolerance=1.0e-10*std::max(1.0,parentDistance);
                if(!std::isfinite(h.distance)||h.faceIndex>=faces.size()) {
                    if(detailFaceEligibility) {
                        ++eligibilityFailures;
                        if(eligibilityFailures<=20u)eligibilityWitnesses<<" row="<<i<<":no eligible face in bounded source-connected neighborhood;";
                        continue;
                    }
                    return fail("incident-face projection exceeds source representative distance bound");
                }
                if(h.distance>parentDistance+boundTolerance) {
                    if(detailFaceEligibility) {
                        ++eligibilityFailures;
                        if(eligibilityFailures<=20u)eligibilityWitnesses<<" row="<<i<<":eligible projection exceeds representative-distance bound;";
                        continue;
                    }
                    return fail("incident-face projection exceeds source representative distance bound");
                }
                ++r.ancestryFallbackRows;
                ancestrySum2+=h.distance*h.distance;
                if(!r.maximumAncestryFallbackWitness.valid || h.distance>ancestryMax) {
                    ancestryMax=h.distance;
                    r.maximumAncestryFallbackWitness=makeProjectionWitness(
                        i,representativeIndex,h.face,h.faceIndex,h.distance);
                }
            }
            if(h.distance>maxProjection) {
                if(detailFaceEligibility) {
                    ++eligibilityFailures;
                    if(eligibilityFailures<=20u)eligibilityWitnesses<<" row="<<i<<":eligible projection exceeds configured distance;";
                    continue;
                }
                return fail("projection exceeds configured distance");
            }
            auto& row=rows_[i];
            selectedFaceIndices_[i]=std::uint32_t(h.faceIndex);
            for(int k=0;k<3;++k) { row.coarseVertex[k]=h.face[k]; row.weight[k]=float(h.bary[k]); }
            row.donorBodyIndex=detailDonor[i]; row.restDistance=float(h.distance);
            ++r.projectedRows; sum2+=h.distance*h.distance;
            if(!r.maximumProjectionWitness.valid || h.distance>maxd) {
                maxd=h.distance;
                r.maximumProjectionWitness=makeProjectionWitness(
                    i,representativeIndex,h.face,h.faceIndex,h.distance);
            }
        }
        if(eligibilityFailures) {
            if(error)*error="typed per-detail face eligibility rejected "+
                std::to_string(eligibilityFailures)+" rows; first witnesses:"+
                eligibilityWitnesses.str();
            clear();return false;
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
    std::uint32_t selectedFaceIndex(std::size_t i) const { return selectedFaceIndices_.at(i); }
    bool hasPreparedRefinedDomain() const { return refinedDomainInstalled_; }
    std::size_t mechanicalNodeCount() const {
        return refinedDomainInstalled_ ? mechanicalNodeCount_ : coarseRest_.size();
    }
    const std::vector<std::uint32_t>& coarseMechanicalNodeRows() const {
        return coarseMechanicalNodeRows_;
    }

    // Installs only a fully prepared, typed refined surface. This method never
    // projects or repairs rows: every active detail row must already be bound
    // to a child face carrying the same stable source-face identity.
    bool installPreparedRefinedDomain(const PreparedRefinedDomain& in,
                                      const std::array<std::uint64_t,4>& expectedCandidateIdentity,
                                      RefinedDomainReport* report,
                                      std::string* error) {
        auto fail=[&](const char* text){ if(error)*error=text; return false; };
        const auto invalid32=std::numeric_limits<std::uint32_t>::max();
        if(refinedDomainInstalled_)return fail("a refined domain is already installed");
        const std::size_t oldN=baseCoarseRest_.size(),newN=in.coarseRest.size();
        if(oldN==0||originalFaces_.empty()||rows_.empty())
            return fail("the existing owner build must succeed before refined installation");
        if(in.originalCoarseVertexCount!=oldN||newN<oldN||in.mechanicalNodeCount==0||
           in.coarseSource.size()!=newN||in.coarseDonor.size()!=newN||
           in.coarseMechanicalNodeRows.size()!=newN||in.faces.empty()||
           in.refinedFaceStableIDs.size()!=in.faces.size()||
           in.originalFaceStableIDs.size()!=originalFaces_.size()||
           in.detailSourceIDs.size()!=rows_.size()||
           in.detailParentStableFaceIDs.size()!=rows_.size()||
           in.detailSelectedFaceRows.size()!=rows_.size()||
           in.coarseSourceFaces.size()!=newN||in.coarseSourceBarycentric.size()!=newN||
           in.coarseBaselineProlongation.size()!=newN||
           in.coarseFaceSourceOffsets.size()!=newN+1||in.rows.size()!=rows_.size())
            return fail("refined-domain arrays have inconsistent dimensions");
        if(!std::isfinite(in.baselineToleranceMeters)||in.baselineToleranceMeters<0)
            return fail("invalid refined baseline tolerance");
        const auto anyIdentity=[](const std::array<std::uint64_t,4>& identity) {
            for(const auto word:identity)if(word!=0)return true;
            return false;
        };
        if(!hasBaseSourceIdentity_||baseStableFaceIds_.size()!=originalFaces_.size())
            return fail("original source build lacks pinned source and stable-face identities");
        if(!anyIdentity(in.sourceIdentity)||in.sourceIdentity!=baseSourceIdentity_)
            return fail("refined candidate source identity differs from the built source");
        if(!anyIdentity(expectedCandidateIdentity)||
           !anyIdentity(in.candidateIdentity)||
           in.candidateIdentity!=expectedCandidateIdentity)
            return fail("refined candidate identity differs from the trusted candidate pin");
        if(!std::isfinite(in.maximumTransferPointErrorMeters)||
           in.maximumTransferPointErrorMeters<0.0)
            return fail("invalid refined transfer-point tolerance");
        if(newN>std::numeric_limits<std::uint32_t>::max()||
           in.mechanicalNodeCount>std::numeric_limits<std::uint32_t>::max())
            return fail("refined domain exceeds GPU row index range");
        for(const auto& p:in.coarseRest)if(!finite(p))
            return fail("nonfinite refined coarse point");
        for(std::size_t i=0;i<oldN;++i) {
            if(in.coarseRest[i]!=baseCoarseRest_[i]||
               in.coarseSource[i]!=baseCoarseSource_[i]||
               in.coarseDonor[i]!=baseCoarseDonor_[i])
                return fail("original coarse prefix identity or coordinates changed");
            if(!baseCoarseSourceFaces_.empty()) {
                if(in.coarseSourceFaces[i]!=baseCoarseSourceFaces_[i]||
                   in.coarseSourceBarycentric[i]!=baseCoarseSourceBarycentric_[i])
                    return fail("original coarse source embedding changed");
            } else if(in.coarseSourceFaces[i]!=Face{invalid32,invalid32,invalid32}||
                      in.coarseSourceBarycentric[i]!=Point{}) {
                return fail("unembedded original coarse prefix changed its source representation");
            }
        }
        if(in.originalFaceStableIDs!=baseStableFaceIds_)
            return fail("original coarse stable face identities changed");
        if(std::unordered_set<std::uint32_t>(baseStableFaceIds_.begin(),
                    baseStableFaceIds_.end()).size()!=baseStableFaceIds_.size())
            return fail("original stable face identities are not unique");

        std::unordered_map<std::uint32_t,std::uint32_t> mechanicalToCoarse;
        mechanicalToCoarse.reserve(newN*2);
        for(std::size_t i=0;i<newN;++i) {
            const auto m=in.coarseMechanicalNodeRows[i];
            if(m>=in.mechanicalNodeCount||
               !mechanicalToCoarse.emplace(m,static_cast<std::uint32_t>(i)).second)
                return fail("invalid or duplicate coarse-to-mechanical row mapping");
            if(i<oldN&&m!=i)
                return fail("original coarse prefix no longer maps to its original mechanics row");
            if(i>=oldN&&in.coarseSource[i]!=invalid32)
                return fail("generated coarse vertex has a fabricated direct source vertex identity");
            if(in.coarseDonor[i]!=invalid32&&
               std::find(detailDonor_.begin(),detailDonor_.end(),in.coarseDonor[i])==detailDonor_.end())
                return fail("coarse vertex donor identity is not present in the built source domain");
            if(i<oldN&&in.coarseDonor[i]==invalid32)
                return fail("original coarse prefix lost its donor identity");
        }

        std::unordered_set<std::uint32_t> knownFaceIDs;
        std::unordered_set<std::uint32_t> representedChildFaceIDs;
        std::unordered_set<std::string> duplicateFaceCheck;
        std::vector<std::unordered_set<std::uint32_t>> expectedNodeFaces(oldN);
        std::unordered_map<std::uint32_t,std::unordered_set<std::uint32_t>> parentVertices;
        for(std::size_t fi=0;fi<originalFaces_.size();++fi) {
            const auto id=in.originalFaceStableIDs[fi];
            if(id==invalid32)return fail("unknown original stable face identity");
            knownFaceIDs.insert(id);
            for(auto v:originalFaces_[fi]) {
                if(v>=oldN)return fail("stored original face row is invalid");
                expectedNodeFaces[v].insert(id);
                parentVertices[id].insert(v);
            }
        }
        if(in.coarseFaceSourceOffsets.front()!=0||
           in.coarseFaceSourceOffsets.back()!=in.coarseFaceStableSourceIDs.size())
            return fail("invalid per-node source-face CSR endpoints");
        for(std::size_t i=0;i<newN;++i) {
            const auto b=in.coarseFaceSourceOffsets[i],e=in.coarseFaceSourceOffsets[i+1];
            if(e<b||e>in.coarseFaceStableSourceIDs.size()||b==e)
                return fail("invalid or empty per-node source-face ancestry row");
            std::unordered_set<std::uint32_t> ids;
            for(auto k=b;k<e;++k) {
                const auto id=in.coarseFaceStableSourceIDs[k];
                if(id==invalid32||!knownFaceIDs.count(id)||!ids.insert(id).second)
                    return fail("unknown or duplicate per-node source-face identity");
            }
            if(i<oldN&&ids!=expectedNodeFaces[i])
                return fail("original coarse per-node source-face ancestry changed");
        }
        for(std::size_t fi=0;fi<in.faces.size();++fi) {
            const auto id=in.refinedFaceStableIDs[fi];
            const auto& f=in.faces[fi];
            if(id==invalid32||!knownFaceIDs.count(id)||f[0]>=newN||f[1]>=newN||f[2]>=newN||
               f[0]==f[1]||f[1]==f[2]||f[2]==f[0])
                return fail("invalid refined face or unknown stable parent face");
            std::array<std::uint32_t,3> key=f;
            std::sort(key.begin(),key.end());
            const std::string faceKey=std::to_string(key[0])+":"+std::to_string(key[1])+":"+std::to_string(key[2]);
            if(!duplicateFaceCheck.insert(faceKey).second)
                return fail("duplicate refined child face");
            representedChildFaceIDs.insert(id);
            for(auto v:f) {
                bool carries=false;
                for(auto k=in.coarseFaceSourceOffsets[v];k<in.coarseFaceSourceOffsets[v+1];++k)
                    carries|=in.coarseFaceStableSourceIDs[k]==id;
                if(!carries)return fail("refined face vertex lacks its parent-face identity");
            }
        }
        if(representedChildFaceIDs!=knownFaceIDs)
            return fail("refined faces do not cover every original stable parent face");

        double maximumBaselineError=0;
        for(std::size_t i=0;i<newN;++i) {
            const auto& row=in.coarseBaselineProlongation[i];
            double sum=0;bool self=false,other=false;Point reconstructed{};
            for(unsigned k=0;k<3;++k) {
                const auto parent=row.parent[k];const double w=row.weight[k];
                if(parent>=oldN||!std::isfinite(w)||w<0||w>1)
                    return fail("invalid refined baseline parent or weight");
                if(w>0){self|=parent==i;other|=parent!=i;}
                sum+=w;
                for(unsigned a=0;a<3;++a)reconstructed[a]+=w*baseCoarseRest_[parent][a];
            }
            if(std::abs(sum-1.0)>1.0e-12||(self&&other))
                return fail("refined baseline ancestry is not a normalized convex partition");
            if(i<oldN) {
                if(self==false||other)
                    return fail("original prefix baseline row is not preserved");
                for(unsigned k=0;k<3;++k) {
                    const auto& base=baseCoarseBaselineProlongation_;
                    const std::uint32_t expectedParent=base.empty()
                        ? static_cast<std::uint32_t>(i):base[i].parent[k];
                    const double expectedWeight=base.empty()?(k==0?1.0:0.0):base[i].weight[k];
                    if(row.parent[k]!=expectedParent||row.weight[k]!=expectedWeight)
                        return fail("original prefix baseline prolongation changed");
                }
            } else {
                for(unsigned a=0;a<3;++a) {
                    const double d=std::abs(reconstructed[a]-in.coarseRest[i][a]);
                    maximumBaselineError=std::max(maximumBaselineError,d);
                }
                if(maximumBaselineError>in.baselineToleranceMeters)
                    return fail("generated baseline ancestry does not reproduce its rest point");
                bool supportsFace=false;
                for(auto k=in.coarseFaceSourceOffsets[i];k<in.coarseFaceSourceOffsets[i+1];++k) {
                    const auto id=in.coarseFaceStableSourceIDs[k];
                    const auto it=parentVertices.find(id);
                    if(it==parentVertices.end())continue;
                    bool all=true;
                    for(unsigned j=0;j<3;++j)
                        if(row.weight[j]>0)all&=it->second.count(row.parent[j])!=0;
                    supportsFace|=all;
                }
                if(!supportsFace)
                    return fail("generated baseline ancestry leaves its declared source-face domain");
            }
            const auto& sf=in.coarseSourceFaces[i];
            const auto& sb=in.coarseSourceBarycentric[i];
            const bool absentEmbedding=sf[0]==invalid32&&sf[1]==invalid32&&sf[2]==invalid32;
            if(absentEmbedding) {
                if(i<oldN||sb!=Point{}||(row.weight[0]==1.0&&row.parent[0]==i))
                    return fail("source embedding is absent without generated baseline ancestry");
                if(in.coarseDonor[i]!=invalid32)
                    return fail("unembedded generated node must not invent a donor frame");
            } else {
                if(!finite(sb)||sb[0]<-1e-8||sb[1]<-1e-8||sb[2]<-1e-8||
                   std::abs(sb[0]+sb[1]+sb[2]-1.0)>1e-8)
                    return fail("invalid refined source-face barycentrics");
                for(auto v:sf)if(v>=detailVertexCount_)
                    return fail("refined source-face vertex index is out of range");
                if(in.coarseDonor[i]==invalid32)
                    return fail("embedded refined node lacks a donor identity");
            }
            if(i>=oldN&&in.coarseSource[i]!=invalid32)
                return fail("generated vertex direct source identity is not a sentinel");
        }

        double maximumTransferPointError=0.0;
        for(std::size_t i=0;i<in.rows.size();++i) {
            const auto& row=in.rows[i];
            if(in.detailSourceIDs[i]!=detailSource_[i])
                return fail("prepared detail source identity differs from the existing owner");
            if(row.donorBodyIndex!=detailDonor_[i]||!std::isfinite(row.restDistance)||row.restDistance<0)
                return fail("prepared transfer donor identity or projection distance is invalid");
            const auto faceRow=in.detailSelectedFaceRows[i];
            bool any=false;double sum=0;
            if(detailUsed_[i]) {
                if(faceRow>=in.faces.size()||in.detailParentStableFaceIDs[i]==invalid32||
                   in.refinedFaceStableIDs[faceRow]!=in.detailParentStableFaceIDs[i])
                    return fail("active detail row has unknown or mismatched stable parent face");
                const auto baseFace=selectedFaceIndices_[i];
                bool baseParentMatches=false;
                std::uint32_t directBaseVertex=invalid32;
                for(std::size_t v=0;v<baseCoarseSource_.size();++v) {
                    if(baseCoarseSource_[v]==detailSource_[i]&&
                       baseCoarseDonor_[v]==detailDonor_[i]) {
                        directBaseVertex=static_cast<std::uint32_t>(v);
                        break; // build() resolves duplicate keys to the first row.
                    }
                }
                bool baseIsExact=false;
                if(directBaseVertex!=invalid32) {
                    for(unsigned k=0;k<3;++k) {
                        bool oneHot=rows_[i].weight[k]==1.0f;
                        for(unsigned j=0;j<3;++j)if(j!=k)
                            oneHot&=rows_[i].weight[j]==0.0f;
                        baseIsExact|=oneHot&&rows_[i].coarseVertex[k]==directBaseVertex;
                    }
                }
                if(baseIsExact) {
                    for(std::size_t f=0;f<originalFaces_.size();++f) {
                        bool contains=false;
                        for(auto v:originalFaces_[f])contains|=v==directBaseVertex;
                        if(contains&&baseStableFaceIds_[f]==in.detailParentStableFaceIDs[i])
                            baseParentMatches=true;
                    }
                } else if(baseFace<baseStableFaceIds_.size()) {
                    baseParentMatches=baseStableFaceIds_[baseFace]==in.detailParentStableFaceIDs[i];
                }
                if(!baseParentMatches)
                    return fail("prepared parent face is not the existing owner row's source face");
            }
            for(unsigned k=0;k<3;++k) {
                const auto node=row.coarseVertex[k];const float w=row.weight[k];
                if(node>=in.mechanicalNodeCount||!std::isfinite(w)||w<0)
                    return fail("prepared GPU row has invalid mechanics index or weight");
                sum+=w;
                if(w>0) {
                    any=true;
                    const auto it=mechanicalToCoarse.find(node);
                    if(it==mechanicalToCoarse.end())
                        return fail("prepared transfer support is not a refined outer node");
                    if(detailUsed_[i]&&std::find(in.faces[faceRow].begin(),in.faces[faceRow].end(),it->second)==in.faces[faceRow].end())
                        return fail("transfer support is outside the selected child face");
                }
            }
            if(detailUsed_[i]) {
                if(!any||std::abs(sum-1.0)>2e-6)
                    return fail("active detailed transfer weights are not a convex partition");
                Point oldPoint{},newPoint{};
                for(unsigned k=0;k<3;++k) {
                    const auto oldNode=rows_[i].coarseVertex[k];
                    const double oldWeight=rows_[i].weight[k];
                    if(oldNode>=oldN)return fail("built owner transfer support is outside original coarse domain");
                    for(unsigned a=0;a<3;++a)oldPoint[a]+=oldWeight*baseCoarseRest_[oldNode][a];
                    if(row.weight[k]>0.0f) {
                        const auto it=mechanicalToCoarse.find(row.coarseVertex[k]);
                        if(it==mechanicalToCoarse.end())return fail("new transfer support is not an outer mechanics node");
                        for(unsigned a=0;a<3;++a)newPoint[a]+=double(row.weight[k])*in.coarseRest[it->second][a];
                    }
                }
                const double pointError=std::sqrt(dist2(oldPoint,newPoint));
                maximumTransferPointError=std::max(maximumTransferPointError,pointError);
                if(pointError>in.maximumTransferPointErrorMeters)
                    return fail("prepared transfer does not preserve the original baseline point");
            } else if(any||sum!=0.0) {
                return fail("inactive detailed row acquired transfer support");
            }
        }

        coarseRest_=in.coarseRest;coarseSource_=in.coarseSource;coarseDonor_=in.coarseDonor;
        rows_=in.rows;selectedFaceIndices_=in.detailSelectedFaceRows;
        coarseSourceFaces_=in.coarseSourceFaces;
        coarseSourceBarycentric_=in.coarseSourceBarycentric;
        coarseBaselineProlongation_=in.coarseBaselineProlongation;
        coarseMechanicalNodeRows_=in.coarseMechanicalNodeRows;
        mechanicalNodeCount_=in.mechanicalNodeCount;
        refinedDomainInstalled_=true;
        trees_.clear();
        if(report) {
            *report=RefinedDomainReport{};
            report->originalCoarseVertexCount=oldN;
            report->refinedCoarseVertexCount=newN;
            report->mechanicalNodeCount=in.mechanicalNodeCount;
            report->detailRowCount=rows_.size();
            report->generatedCoarseVertices=newN-oldN;
            report->maximumBaselineReconstructionErrorMeters=maximumBaselineError;
            report->maximumTransferPointErrorMeters=maximumTransferPointError;
            report->sourceIdentity=in.sourceIdentity;
            report->candidateIdentity=in.candidateIdentity;
            for(auto used:detailUsed_)used?++report->activeRows:++report->inactiveRows;
        }
        if(error)error->clear();
        return true;
    }

    // Carry each coarse source-to-shell rest offset in its donor body's local
    // frame. This removes remeshing offsets at zero FEM displacement.
    bool makeCurrentCoarseBaseline(const std::vector<Point>& sourceRestByID,
             const std::vector<Point>& sourceCurrentByID,
             const std::vector<Pose>& donorRestPose,
             const std::vector<Pose>& donorCurrentPose,
             std::vector<Point>& out, std::string* error) const {
        out.resize(coarseSource_.size());
        for(std::size_t i=0;i<coarseSource_.size();++i) {
            if(refinedDomainInstalled_&&i>=baseCoarseRest_.size()&&
               !coarseBaselineProlongation_.empty()&&
               coarseSourceFaces_[i][0]==std::numeric_limits<std::uint32_t>::max())
                continue; // An unembedded generated vertex is defined by its pinned prefix prolongation below.
            const auto donor=coarseDonor_[i];
            Point sourceRest{},sourceCurrent{};
            if(coarseSourceFaces_.empty()||
               coarseSourceFaces_[i][0]==std::numeric_limits<std::uint32_t>::max()) {
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
        if(refinedDomainInstalled_) {
            if(error)*error="use makeCoarseDisplacementsFromMechanical for refined transfer rows";
            out.clear();return false;
        }
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

    // Full mechanics-domain adapters for prepared refined rows. Non-O nodes
    // carry zero deltas and therefore cannot affect detailed skin transfer.
    bool makeCurrentMechanicalBaseline(const std::vector<Point>& sourceRestByID,
             const std::vector<Point>& sourceCurrentByID,
             const std::vector<Pose>& donorRestPose,
             const std::vector<Pose>& donorCurrentPose,
             std::vector<Point>& out, std::string* error) const {
        if(!refinedDomainInstalled_) {
            if(error)*error="prepared refined domain is not installed";out.clear();return false;
        }
        std::vector<Point> compact;
        if(!makeCurrentCoarseBaseline(sourceRestByID,sourceCurrentByID,
                                      donorRestPose,donorCurrentPose,compact,error)) {
            out.clear();return false;
        }
        out.assign(mechanicalNodeCount_,Point{});
        for(std::size_t i=0;i<compact.size();++i)out[coarseMechanicalNodeRows_[i]]=compact[i];
        if(error)error->clear();return true;
    }
    bool makeCoarseDisplacementsFromMechanical(const std::vector<Point>& acceptedMechanical,
             const std::vector<Point>& baselineMechanical,
             std::vector<std::array<float,4>>& out,std::string* error) const {
        if(!refinedDomainInstalled_||acceptedMechanical.size()!=mechanicalNodeCount_||
           baselineMechanical.size()!=mechanicalNodeCount_) {
            if(error)*error="refined mechanical arrays have invalid dimensions";out.clear();return false;
        }
        out.assign(mechanicalNodeCount_,std::array<float,4>{0,0,0,0});
        for(auto m:coarseMechanicalNodeRows_) {
            if(!finite(acceptedMechanical[m])||!finite(baselineMechanical[m])) {
                if(error)*error="nonfinite refined mechanical position";out.clear();return false;
            }
            out[m]={float(acceptedMechanical[m][0]-baselineMechanical[m][0]),
                    float(acceptedMechanical[m][1]-baselineMechanical[m][1]),
                    float(acceptedMechanical[m][2]-baselineMechanical[m][2]),0};
        }
        if(error)error->clear();return true;
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
    void clear(){
        rows_.clear();selectedFaceIndices_.clear();coarseRest_.clear();coarseSource_.clear();coarseDonor_.clear();
        coarseSourceFaces_.clear();coarseSourceBarycentric_.clear();coarseBaselineProlongation_.clear();trees_.clear();
        baseCoarseRest_.clear();baseCoarseSource_.clear();baseCoarseDonor_.clear();originalFaces_.clear();
        detailSource_.clear();detailDonor_.clear();detailUsed_.clear();baseCoarseSourceFaces_.clear();
        baseCoarseSourceBarycentric_.clear();baseCoarseBaselineProlongation_.clear();baseStableFaceIds_.clear();
        coarseMechanicalNodeRows_.clear();detailVertexCount_=0;mechanicalNodeCount_=0;refinedDomainInstalled_=false;
        baseSourceIdentity_={};hasBaseSourceIdentity_=false;
    }
    std::vector<GPUTransferRow> rows_;
    std::vector<std::uint32_t> selectedFaceIndices_;
    std::vector<Point> coarseRest_;
    std::vector<std::uint32_t> coarseSource_,coarseDonor_;
    std::vector<Face> coarseSourceFaces_;
    std::vector<Point> coarseSourceBarycentric_;
    std::vector<BaselineProlongationRow> coarseBaselineProlongation_;
    std::vector<Tree> trees_;
    std::vector<Point> baseCoarseRest_;
    std::vector<std::uint32_t> baseCoarseSource_,baseCoarseDonor_;
    std::vector<Face> originalFaces_;
    std::vector<std::uint32_t> detailSource_,detailDonor_;
    std::vector<std::uint8_t> detailUsed_;
    std::vector<Face> baseCoarseSourceFaces_;
    std::vector<Point> baseCoarseSourceBarycentric_;
    std::vector<BaselineProlongationRow> baseCoarseBaselineProlongation_;
    std::vector<std::uint32_t> baseStableFaceIds_;
    std::array<std::uint64_t,4> baseSourceIdentity_{};
    bool hasBaseSourceIdentity_=false;
    std::vector<std::uint32_t> coarseMechanicalNodeRows_;
    std::size_t detailVertexCount_=0,mechanicalNodeCount_=0;
    bool refinedDomainInstalled_=false;
};
static_assert(sizeof(AcceptedFEMSkinTransfer::GPUTransferRow)==32,"Metal row ABI");
static_assert(alignof(AcceptedFEMSkinTransfer::GPUTransferRow)==16,"Metal row alignment");
} // namespace numi::human

