#pragma once
// Coupled outer-skin displacement transfer for an offline candidate.
// This does not alter FEM topology, material thickness, or native runtime state.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace numi::human::connected_skin_transfer013 {

using Point = std::array<double, 3>;
using Face = std::array<std::uint32_t, 3>;

struct TransferRow {
    std::array<std::uint32_t, 3> physicalVertex{};
    std::array<float, 3> weight{};
    bool active = true;
};

struct Pins {
    std::string sourceMeshSha256;
    std::string physicalMeshSha256;
    std::string poseSha256;
};

struct Input {
    // Source mesh is the registered detailed surface vertex domain (the 294 input).
    std::span<const Point> sourceRestWorldM;
    std::span<const Face> sourceFaces;
    std::span<const std::uint32_t> detailSourceVertexIds;
    std::span<const Point> detailRestWorldM;
    std::span<const Face> detailFaces;
    std::span<const std::uint8_t> detailMayMove;
    std::span<const std::uint8_t> detailProtected;

    // Physical O/FEM surface, with the candidate's exact source-face embedding.
    std::span<const Point> physicalOuterRestWorldM;
    std::span<const Face> physicalOuterFaces;
    std::span<const std::uint32_t> physicalSourceFaceRows;
    std::span<const Point> physicalSourceBarycentric;
    std::span<const Point> physicalSourceResidualWorldM;
    std::span<const std::uint8_t> physicalMayMove;
    std::span<const std::uint8_t> physicalProtected;

    // Rows are the existing AcceptedFEMSkinTransfer mapping: detail row -> 3 physical O rows.
    std::span<const TransferRow> detailFromPhysical;
    std::span<const Point> physicalDeltaWorldM;

    // Existing physical node masses are copied through exactly; this helper never remaps them.
    std::span<const double> physicalNodeMassKg;
    Pins pins;
    double reconstructionToleranceM = 1.0e-8;
    double displacementToleranceM = 1.0e-12;
};

struct Report {
    std::size_t sourceVertexCount = 0;
    std::size_t detailVertexCount = 0;
    std::size_t physicalVertexCount = 0;
    std::size_t activeDetailRows = 0;
    std::size_t inactiveDetailRows = 0;
    std::size_t movedDetailRows = 0;
    std::size_t movedPhysicalRows = 0;
    std::size_t protectedDetailRows = 0;
    std::size_t protectedPhysicalRows = 0;
    double maxRestEmbeddingErrorM = 0.0;
    double maxDetailSourceIdentityErrorM = 0.0;
    double maxMovedSourceResidualErrorM = 0.0;
    double maxSourceResidualChangeM = 0.0;
    double maxPhysicalDeltaErrorM = 0.0;
    double maxDetailDeltaErrorM = 0.0;
    double detailedSurfaceAreaM2 = 0.0;
    double unmappedDetailedAreaM2 = 0.0;
    double inducedPhysicalSurfaceAreaM2 = 0.0;
    Point detailAreaDisplacementMomentM3{};
    Point inducedPhysicalAreaDisplacementMomentM3{};
    double momentClosureErrorM3 = 0.0;
    double physicalMassBeforeKg = 0.0;
    double physicalMassAfterKg = 0.0;
    bool physicalMassRowsBitwiseUnchanged = false;
    Pins pins;
};

struct Output {
    std::vector<Point> detailWorldM;
    std::vector<Point> physicalOuterWorldM;
    std::vector<Point> physicalSourceResidualWorldM;
    std::vector<Point> detailDeltaWorldM;
    std::vector<double> physicalNodeMassKg;
    // The dual surface-area measure induced on physical rows by the detail->physical map.
    // This is a transfer quadrature, not an anatomical or FEM mass ledger.
    std::vector<double> inducedPhysicalSurfaceMeasureM2;
    Report report;
};

// Adapts the exact row ABI exposed by AcceptedFEMSkinTransfer::rows() without
// depending on that owner's header. Pass the same detailUsed mask used to build it.
template<class AcceptedOwnerRow>
inline bool copyAcceptedOwnerRows(std::span<const AcceptedOwnerRow> source,
        std::span<const std::uint8_t> active, std::vector<TransferRow>& out,
        std::string& error) {
    if(source.size()!=active.size()) { error="accepted transfer and active mask row counts differ";out.clear();return false; }
    out.resize(source.size());
    for(std::size_t i=0;i<source.size();++i) {
        TransferRow row{};row.active=active[i]!=0;
        for(unsigned k=0;k<3;++k) {
            row.physicalVertex[k]=source[i].coarseVertex[k];
            row.weight[k]=source[i].weight[k];
        }
        out[i]=row;
    }
    error.clear();return true;
}

namespace detail {
inline bool finite(const Point& p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}
inline Point sub(const Point& a, const Point& b) {
    return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}
inline Point add(const Point& a, const Point& b) {
    return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}
inline double dot(const Point& a, const Point& b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
inline Point cross(const Point& a, const Point& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
inline double norm(const Point& a) { return std::sqrt(dot(a,a)); }
inline bool validSha256(const std::string& s) {
    if(s.size()!=64) return false;
    return std::all_of(s.begin(),s.end(),[](unsigned char c) {
        return (c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F');
    });
}
inline bool maskValue(std::span<const std::uint8_t> mask, std::size_t i, bool emptyDefault) {
    return mask.empty() ? emptyDefault : mask[i] != 0;
}
inline void addScaled(Point& dst, const Point& p, double w) {
    for(unsigned a=0;a<3;++a) dst[a] += w*p[a];
}
inline double triangleArea(const Point& a, const Point& b, const Point& c) {
    return 0.5*norm(cross(sub(b,a),sub(c,a)));
}
} // namespace detail

inline bool transfer(const Input& in, Output& out, std::string& error) {
    using namespace detail;
    auto fail=[&](const char* why){ error=why; out={}; return false; };
    const std::size_t ns=in.sourceRestWorldM.size();
    const std::size_t nd=in.detailRestWorldM.size();
    const std::size_t np=in.physicalOuterRestWorldM.size();
    if(ns==0||nd==0||np==0||in.sourceFaces.empty()||in.detailFaces.empty()||
       in.physicalOuterFaces.empty())
        return fail("empty source, detail, or physical surface");
    if(in.detailSourceVertexIds.size()!=nd||in.detailMayMove.size()!=nd||
       in.detailProtected.size()!=nd||in.detailFromPhysical.size()!=nd||
       in.physicalOuterFaces.empty()||in.physicalSourceFaceRows.size()!=np||
       in.physicalSourceBarycentric.size()!=np||in.physicalSourceResidualWorldM.size()!=np||
       in.physicalMayMove.size()!=np||in.physicalProtected.size()!=np||
       in.physicalDeltaWorldM.size()!=np||in.physicalNodeMassKg.size()!=np)
        return fail("row-aligned mapping, mask, displacement, or mass arrays differ in size");
    if(!validSha256(in.pins.sourceMeshSha256)||!validSha256(in.pins.physicalMeshSha256)||
       !validSha256(in.pins.poseSha256))
        return fail("source mesh, physical mesh, and pose SHA-256 pins are required");
    if(!std::isfinite(in.reconstructionToleranceM)||in.reconstructionToleranceM<=0.0||
       !std::isfinite(in.displacementToleranceM)||in.displacementToleranceM<0.0)
        return fail("invalid finite reconstruction or displacement tolerance");

    for(const auto& p:in.sourceRestWorldM) if(!finite(p)) return fail("nonfinite source point");
    for(const auto& p:in.detailRestWorldM) if(!finite(p)) return fail("nonfinite detailed point");
    for(const auto& p:in.physicalOuterRestWorldM) if(!finite(p)) return fail("nonfinite physical O point");
    for(const auto& f:in.sourceFaces) for(auto v:f) if(v>=ns) return fail("source face index out of range");
    for(const auto& f:in.detailFaces) for(auto v:f) if(v>=nd) return fail("detail face index out of range");
    for(const auto& f:in.physicalOuterFaces) for(auto v:f) if(v>=np) return fail("physical O face index out of range");

    std::vector<Point> sourceDelta(ns, Point{});
    std::vector<std::uint8_t> sourceAssigned(ns, 0);
    Output candidate;
    candidate.detailWorldM.assign(in.detailRestWorldM.begin(),in.detailRestWorldM.end());
    candidate.physicalOuterWorldM.assign(in.physicalOuterRestWorldM.begin(),in.physicalOuterRestWorldM.end());
    candidate.physicalSourceResidualWorldM.assign(in.physicalSourceResidualWorldM.begin(),in.physicalSourceResidualWorldM.end());
    candidate.detailDeltaWorldM.assign(nd, Point{});
    candidate.physicalNodeMassKg.assign(in.physicalNodeMassKg.begin(),in.physicalNodeMassKg.end());
    candidate.inducedPhysicalSurfaceMeasureM2.assign(np,0.0);
    Report r;
    r.sourceVertexCount=ns;r.detailVertexCount=nd;r.physicalVertexCount=np;r.pins=in.pins;
    double massBefore=0.0;
    for(double m:in.physicalNodeMassKg) {
        if(!std::isfinite(m)||m<0.0) return fail("physical nodal mass must be finite and nonnegative");
        massBefore+=m;
    }
    if(!(massBefore>0.0)||!std::isfinite(massBefore)) return fail("physical nodal mass total must be positive finite");
    r.physicalMassBeforeKg=massBefore;

    // The detailed 294 vertex identity must resolve to the same registered world row;
    // a numerically nearby but differently posed/source-indexed mesh is not interchangeable.
    for(std::size_t i=0;i<nd;++i) {
        const auto sid=in.detailSourceVertexIds[i];
        if(sid>=ns) return fail("detail-to-source vertex identity is outside source domain");
        const double e=norm(sub(in.detailRestWorldM[i],in.sourceRestWorldM[sid]));
        r.maxDetailSourceIdentityErrorM=std::max(r.maxDetailSourceIdentityErrorM,e);
        if(!std::isfinite(e)||e>in.reconstructionToleranceM)
            return fail("detailed row and direct source identity disagree in the pinned frame");
    }

    // Validate the physical O -> detailed-source face/bary/residual map in the pinned frame.
    for(std::size_t i=0;i<np;++i) {
        const auto fr=in.physicalSourceFaceRows[i];
        const auto& b=in.physicalSourceBarycentric[i];
        if(fr>=in.sourceFaces.size()||!finite(b)||!finite(in.physicalSourceResidualWorldM[i]))
            return fail("physical O row has invalid typed source-face/bary/residual ancestry");
        double sum=0.0;Point q{};
        const auto& f=in.sourceFaces[fr];
        for(unsigned k=0;k<3;++k) {
            if(b[k]<-1.0e-10||b[k]>1.0+1.0e-10) return fail("physical O barycentric weight is outside [0,1]");
            sum+=b[k];addScaled(q,in.sourceRestWorldM[f[k]],b[k]);
        }
        if(std::abs(sum-1.0)>1.0e-10) return fail("physical O barycentric row does not sum to one");
        q=add(q,in.physicalSourceResidualWorldM[i]);
        const double e=norm(sub(q,in.physicalOuterRestWorldM[i]));
        r.maxRestEmbeddingErrorM=std::max(r.maxRestEmbeddingErrorM,e);
        if(!std::isfinite(e)||e>in.reconstructionToleranceM)
            return fail("physical O source point plus residual does not reconstruct its pinned rest point");
    }
    if(r.maxRestEmbeddingErrorM>in.reconstructionToleranceM)
        return fail("physical O rest embedding exceeds reconstruction tolerance");

    // The accepted owner uses a three-corner barycentric map from physical FEM O to detail rows.
    // Partition of unity preserves a rigid translation exactly.
    for(std::size_t i=0;i<nd;++i) {
        const auto sid=in.detailSourceVertexIds[i];
        if(sid>=ns) return fail("detail-to-source vertex identity is outside source domain");
        const auto& row=in.detailFromPhysical[i];
        double sum=0.0;Point d{};
        if(row.active) {
            for(unsigned k=0;k<3;++k) {
                if(row.physicalVertex[k]>=np||!std::isfinite(row.weight[k])||row.weight[k]<-1.0e-7f)
                    return fail("active accepted-transfer row has invalid physical index or weight");
                sum+=row.weight[k];
                addScaled(d,in.physicalDeltaWorldM[row.physicalVertex[k]],double(row.weight[k]));
            }
            if(std::abs(sum-1.0)>1.0e-4) return fail("active accepted-transfer row is not a partition of unity");
            ++r.activeDetailRows;
        } else {
            for(unsigned k=0;k<3;++k) {
                if(row.weight[k]!=0.0f) return fail("inactive accepted-transfer row carries nonzero weights");
                if(row.physicalVertex[k]!=0u) return fail("inactive accepted-transfer row must use zero index padding");
            }
            ++r.inactiveDetailRows;
        }
        if(!maskValue(in.detailMayMove,i,true) && norm(d)>in.displacementToleranceM)
            return fail("detailed row outside the declared repair patch would move");
        if(maskValue(in.detailProtected,i,false)) {
            ++r.protectedDetailRows;
            if(norm(d)>in.displacementToleranceM) return fail("protected detailed back/attachment row would move");
        }
        if(!row.active && norm(d)>0.0) return fail("inactive detailed row received a displacement");
        candidate.detailDeltaWorldM[i]=d;
        if(norm(d)>0.0) {
            candidate.detailWorldM[i]=add(in.detailRestWorldM[i],d);
            ++r.movedDetailRows;
        }
        if(sourceAssigned[sid]) {
            if(norm(sub(sourceDelta[sid],d))>in.displacementToleranceM)
                return fail("duplicate detailed source identity receives inconsistent displacement");
        } else {
            sourceDelta[sid]=d;sourceAssigned[sid]=1;
        }
    }

    // Physical O positions are the constrained mechanical candidate; never overwrite them with
    // projected render coordinates. Their source residual is updated explicitly to close the
    // source embedding after the accepted FEM displacement has moved the detailed surface.
    for(std::size_t i=0;i<np;++i) {
        const auto& d=in.physicalDeltaWorldM[i];
        if(!finite(d)) return fail("nonfinite physical O displacement");
        const double dn=norm(d);
        if(!maskValue(in.physicalMayMove,i,true)&&dn>in.displacementToleranceM)
            return fail("physical O row outside the declared repair patch would move");
        if(maskValue(in.physicalProtected,i,false)) {
            ++r.protectedPhysicalRows;
            if(dn>in.displacementToleranceM) return fail("protected physical back/attachment row would move");
        }
        if(dn>0.0) {
            candidate.physicalOuterWorldM[i]=add(in.physicalOuterRestWorldM[i],d);
            ++r.movedPhysicalRows;
        }
        const auto& f=in.sourceFaces[in.physicalSourceFaceRows[i]];
        Point mapped{};bool sourceMoved=false;
        for(unsigned k=0;k<3;++k) {
            sourceMoved |= norm(sourceDelta[f[k]])>0.0;
            addScaled(mapped,add(in.sourceRestWorldM[f[k]],sourceDelta[f[k]]),
                      in.physicalSourceBarycentric[i][k]);
        }
        const Point newResidual=sub(candidate.physicalOuterWorldM[i],mapped);
        const bool protectedPhysical=maskValue(in.physicalProtected,i,false);
        if(protectedPhysical &&
           (norm(sub(newResidual,in.physicalSourceResidualWorldM[i]))>in.reconstructionToleranceM || sourceMoved))
            return fail("protected physical row would change its detailed-source embedding residual");
        if(dn>0.0 || sourceMoved) {
            candidate.physicalSourceResidualWorldM[i]=newResidual;
            r.maxSourceResidualChangeM=std::max(r.maxSourceResidualChangeM,
                norm(sub(newResidual,in.physicalSourceResidualWorldM[i])));
        }
        const Point check=add(mapped,candidate.physicalSourceResidualWorldM[i]);
        const double e=norm(sub(check,candidate.physicalOuterWorldM[i]));
        r.maxMovedSourceResidualErrorM=std::max(r.maxMovedSourceResidualErrorM,e);
        if(!std::isfinite(e)||e>in.reconstructionToleranceM)
            return fail("updated source residual does not reconstruct the moved physical O point");
        r.maxPhysicalDeltaErrorM=std::max(r.maxPhysicalDeltaErrorM,
            norm(sub(sub(candidate.physicalOuterWorldM[i],in.physicalOuterRestWorldM[i]),d)));
    }
    for(std::size_t i=0;i<nd;++i)
        r.maxDetailDeltaErrorM=std::max(r.maxDetailDeltaErrorM,
            norm(sub(sub(candidate.detailWorldM[i],in.detailRestWorldM[i]),candidate.detailDeltaWorldM[i])));

    // Detailed-surface lumped areas define a dual measure on physical O through R^T.
    // This proves conservative displacement-moment transfer without pretending the render mesh
    // owns FEM tissue mass. Actual physical node masses are passed through unchanged below.
    std::vector<double> detailArea(nd,0.0);
    for(const auto& f:in.detailFaces) {
        const double a=triangleArea(in.detailRestWorldM[f[0]],in.detailRestWorldM[f[1]],in.detailRestWorldM[f[2]]);
        if(!std::isfinite(a)||!(a>0.0)) return fail("detail area quadrature has a zero or nonfinite face");
        const double w=a/3.0;
        detailArea[f[0]]+=w;detailArea[f[1]]+=w;detailArea[f[2]]+=w;
    }
    for(std::size_t i=0;i<nd;++i) {
        if(detailArea[i]>0.0 && !in.detailFromPhysical[i].active) {
            r.unmappedDetailedAreaM2+=detailArea[i];
            continue;
        }
        if(detailArea[i]>0.0) {
            const auto& row=in.detailFromPhysical[i];
            for(unsigned k=0;k<3;++k)
                candidate.inducedPhysicalSurfaceMeasureM2[row.physicalVertex[k]]+=detailArea[i]*double(row.weight[k]);
        }
    }
    for(double a:detailArea)r.detailedSurfaceAreaM2+=a;
    for(double a:candidate.inducedPhysicalSurfaceMeasureM2)r.inducedPhysicalSurfaceAreaM2+=a;
    for(std::size_t i=0;i<nd;++i) if(in.detailFromPhysical[i].active)
        addScaled(r.detailAreaDisplacementMomentM3,candidate.detailDeltaWorldM[i],detailArea[i]);
    for(std::size_t j=0;j<np;++j)
        addScaled(r.inducedPhysicalAreaDisplacementMomentM3,in.physicalDeltaWorldM[j],
                  candidate.inducedPhysicalSurfaceMeasureM2[j]);
    r.momentClosureErrorM3=norm(sub(r.detailAreaDisplacementMomentM3,r.inducedPhysicalAreaDisplacementMomentM3));
    const double momentScale=std::max(1.0,norm(r.detailAreaDisplacementMomentM3));
    if(!std::isfinite(r.momentClosureErrorM3)||r.momentClosureErrorM3>2.0e-12*momentScale)
        return fail("area-weighted displacement moment does not close under the induced physical dual measure");

    double massAfter=0.0;
    for(double m:candidate.physicalNodeMassKg)massAfter+=m;
    r.physicalMassAfterKg=massAfter;
    r.physicalMassRowsBitwiseUnchanged=
        candidate.physicalNodeMassKg.size()==in.physicalNodeMassKg.size() &&
        (candidate.physicalNodeMassKg.empty() ||
         std::memcmp(candidate.physicalNodeMassKg.data(),in.physicalNodeMassKg.data(),
                     candidate.physicalNodeMassKg.size()*sizeof(double))==0);
    if(!r.physicalMassRowsBitwiseUnchanged||massAfter!=massBefore)
        return fail("physical FEM node masses changed during displacement-only transfer");
    r.physicalMassAfterKg=massAfter;
    candidate.report=r;out=std::move(candidate);error.clear();return true;
}

} // namespace numi::human::connected_skin_transfer013
