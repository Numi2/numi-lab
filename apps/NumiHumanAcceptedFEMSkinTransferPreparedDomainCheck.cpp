#include "NumiHumanAcceptedFEMSkinTransfer.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using Transfer = numi::human::AcceptedFEMSkinTransfer;
using Point = Transfer::Point;
using Face = Transfer::Face;

static bool near(double a, double b, double tol=1e-12) {
    return std::abs(a-b) <= tol;
}
static bool require(bool ok, const char* message) {
    if (!ok) std::cerr << "FAIL: " << message << "\n";
    return ok;
}

static Transfer::GPUTransferRow rowOnFace(Face f, unsigned corner, unsigned donor) {
    Transfer::GPUTransferRow r{};
    for (unsigned k=0;k<3;++k) r.coarseVertex[k]=f[k];
    r.donorBodyIndex=donor;
    r.weight[corner]=1.0f;
    r.restDistance=0.0f;
    return r;
}


static bool parentFaceClassificationRegression() {
    const std::array<std::uint64_t,4> sourceID{0x901,0x902,0x903,0x904};
    const std::array<std::uint64_t,4> candidateID{0xa01,0xa02,0xa03,0xa04};
    const auto invalid=std::numeric_limits<std::uint32_t>::max();
    const std::vector<Point> detailRest{
        {0,0,0},{1,0,0},{0,1,0},{1,1,0},
        {1.0-1.0e-8,0,0},{0.5,0,0}
    };
    const std::vector<std::uint32_t> detailSource{0,1,2,3,4,5};
    const std::vector<std::uint32_t> detailDonor(6,7);
    const std::vector<std::uint8_t> detailUsed(6,1);
    const std::vector<Point> coarseRest{{0,0,0},{1,0,0},{0,1,0},{1,1,0}};
    const std::vector<std::uint32_t> coarseSource{0,1,2,3};
    const std::vector<std::uint32_t> coarseDonor(4,7);
    // The first face puts exact vertex 0 in weight slot 2. The near-vertex
    // projected detail row later puts vertex 1 in slot 0.
    const std::vector<Face> originalFaces{{1,2,0},{0,2,3},{1,3,2}};
    const std::vector<std::uint32_t> stableFaceIDs{100,101,102};
    const std::vector<Face> sourceFaces{
        Face{0,1,2},Face{0,1,2},Face{0,1,2},Face{1,3,2}
    };
    const std::vector<Point> sourceBary{
        Point{1,0,0},Point{0,1,0},Point{0,0,1},Point{0,1,0}
    };

    Transfer owner;
    Transfer::BuildReport buildReport;
    std::string error;
    if (!require(owner.build(detailRest,detailSource,detailDonor,detailUsed,
                             coarseRest,coarseSource,coarseDonor,originalFaces,
                             &buildReport,&error,1.0,nullptr,&sourceFaces,
                             &sourceBary,nullptr,1,&stableFaceIDs,&sourceID),
                 "classification fixture source build")) {
        std::cerr << error << "\n"; return false;
    }
    if (!require(buildReport.exactAncestryRows==4 && buildReport.projectedRows==2,
                 "classification fixture contains exact and near-one projected rows")) return false;
    const auto& baseRows=owner.rows();
    if (!require(baseRows[0].weight[2]==1.0f &&
                 baseRows[4].weight[0]>0.999999f,
                 "fixture exercises permuted exact slot and near-one projected slot")) return false;

    Transfer::PreparedRefinedDomain domain;
    domain.originalCoarseVertexCount=4;
    domain.mechanicalNodeCount=6;
    domain.sourceIdentity=sourceID;
    domain.candidateIdentity=candidateID;
    domain.coarseRest={{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0.5,0,0}};
    domain.coarseSource={0,1,2,3,invalid};
    domain.coarseDonor={7,7,7,7,invalid};
    domain.coarseMechanicalNodeRows={0,1,2,3,5};
    domain.faces={{1,2,4},{4,2,0},{0,2,3},{1,3,2}};
    domain.originalFaceStableIDs={100,101,102};
    domain.refinedFaceStableIDs={100,100,101,102};
    domain.detailSourceIDs=detailSource;
    domain.detailParentStableFaceIDs={101,102,100,101,100,100};
    domain.detailSelectedFaceRows={2,3,0,2,0,0};
    domain.coarseSourceFaces={
        Face{0,1,2},Face{0,1,2},Face{0,1,2},Face{1,3,2},
        Face{invalid,invalid,invalid}
    };
    domain.coarseSourceBarycentric={
        Point{1,0,0},Point{0,1,0},Point{0,0,1},Point{0,1,0},Point{0,0,0}
    };
    domain.coarseBaselineProlongation.resize(5);
    for (std::uint32_t i=0;i<4;++i) {
        domain.coarseBaselineProlongation[i].parent={i,i,i};
        domain.coarseBaselineProlongation[i].weight={1,0,0};
    }
    domain.coarseBaselineProlongation[4].parent={0,1,0};
    domain.coarseBaselineProlongation[4].weight={0.5,0.5,0};
    domain.coarseFaceSourceOffsets={0,2,4,7,9,10};
    domain.coarseFaceStableSourceIDs={100,101,100,102,100,101,102,101,102,100};
    domain.rows={
        rowOnFace(Face{0,2,3},0,7), // exact vertex 0; dominant old slot was 2
        rowOnFace(Face{1,3,2},0,7), // exact vertex 1; another incident parent
        rowOnFace(Face{1,2,5},1,7),
        rowOnFace(Face{0,2,3},2,7),
        rowOnFace(Face{1,2,5},0,7), // near-one projection stays on parent 100
        rowOnFace(Face{1,2,5},2,7)
    };

    auto switchedProjectedParent=domain;
    switchedProjectedParent.detailParentStableFaceIDs[4]=102;
    switchedProjectedParent.detailSelectedFaceRows[4]=3;
    switchedProjectedParent.rows[4]=rowOnFace(Face{1,3,2},0,7);
    Transfer::RefinedDomainReport report;
    if (!require(!owner.installPreparedRefinedDomain(switchedProjectedParent,candidateID,
                                                       &report,&error),
                 "near-one projected row cannot switch away from its selected source parent")) return false;
    if (!require(!owner.hasPreparedRefinedDomain(),
                 "wrong-parent rejection leaves classification owner unmodified")) return false;
    if (!require(owner.installPreparedRefinedDomain(domain,candidateID,&report,&error),
                 "exact one-hot rows may use any incident parent face regardless of weight slot")) {
        std::cerr << error << "\n"; return false;
    }
    return true;
}

int main() {
    if (!parentFaceClassificationRegression()) return 1;
    const std::array<std::uint64_t,4> sourceID{0x101,0x202,0x303,0x404};
    const std::array<std::uint64_t,4> candidateID{0x505,0x606,0x707,0x808};
    const auto invalid=std::numeric_limits<std::uint32_t>::max();
    const std::vector<Point> detailRest{{0,0,0},{1,0,0},{0,1,0},{0.5,0,0}};
    const std::vector<std::uint32_t> detailSource{0,1,2,3};
    const std::vector<std::uint32_t> detailDonor(4,7);
    const std::vector<std::uint8_t> detailUsed(4,1);
    const std::vector<Point> coarseRest{{0,0,0},{1,0,0},{0,1,0}};
    const std::vector<std::uint32_t> coarseSource{0,1,2};
    const std::vector<std::uint32_t> coarseDonor(3,7);
    const std::vector<Face> originalFaces{{0,1,2}};
    const std::vector<std::uint32_t> stableFaceIDs{100};
    const std::vector<Face> sourceFaces(3,Face{0,1,2});
    const std::vector<Point> sourceBary{{1,0,0},{0,1,0},{0,0,1}};

    Transfer owner;
    Transfer::BuildReport buildReport;
    std::string error;
    if (!require(owner.build(detailRest,detailSource,detailDonor,detailUsed,
                             coarseRest,coarseSource,coarseDonor,originalFaces,
                             &buildReport,&error,1.0,nullptr,&sourceFaces,
                             &sourceBary,nullptr,1,&stableFaceIDs,&sourceID),
                 "base source-bound owner build")) {
        std::cerr << error << "\n"; return 1;
    }
    if (!require(buildReport.exactAncestryRows==3 && buildReport.projectedRows==1,
                 "base includes exact and projected transfer rows")) return 1;

    Transfer::PreparedRefinedDomain domain;
    domain.originalCoarseVertexCount=3;
    domain.mechanicalNodeCount=5;
    domain.sourceIdentity=sourceID;
    domain.candidateIdentity=candidateID;
    domain.coarseRest={{0,0,0},{1,0,0},{0,1,0},{0.5,0,0}};
    domain.coarseSource={0,1,2,invalid};
    domain.coarseDonor={7,7,7,invalid};
    domain.coarseMechanicalNodeRows={0,1,2,4};
    domain.faces={{0,3,2},{3,1,2}};
    domain.originalFaceStableIDs={100};
    domain.refinedFaceStableIDs={100,100};
    domain.detailSourceIDs=detailSource;
    domain.detailParentStableFaceIDs=std::vector<std::uint32_t>(4,100);
    domain.detailSelectedFaceRows={0,1,0,0};
    domain.coarseSourceFaces={Face{0,1,2},Face{0,1,2},Face{0,1,2},Face{invalid,invalid,invalid}};
    domain.coarseSourceBarycentric={{1,0,0},{0,1,0},{0,0,1},{0,0,0}};
    domain.coarseBaselineProlongation.resize(4);
    for (std::uint32_t i=0;i<3;++i) {
        domain.coarseBaselineProlongation[i].parent={i,i,i};
        domain.coarseBaselineProlongation[i].weight={1,0,0};
    }
    domain.coarseBaselineProlongation[3].parent={0,1,0};
    domain.coarseBaselineProlongation[3].weight={0.5,0.5,0};
    domain.coarseFaceSourceOffsets={0,1,2,3,4};
    domain.coarseFaceStableSourceIDs={100,100,100,100};
    domain.rows={rowOnFace(Face{0,4,2},0,7),
                 rowOnFace(Face{4,1,2},1,7),
                 rowOnFace(Face{0,4,2},2,7),
                 rowOnFace(Face{0,4,2},1,7)};

    auto wrongCandidate=candidateID;
    ++wrongCandidate[0];
    Transfer::RefinedDomainReport refinedReport;
    if (!require(!owner.installPreparedRefinedDomain(domain,wrongCandidate,&refinedReport,&error),
                 "rejects a prepared candidate with the wrong trusted identity")) return 1;
    if (!require(!owner.hasPreparedRefinedDomain(),
                 "failed prepared import leaves owner unchanged")) return 1;
    auto wrongSource=domain;
    ++wrongSource.sourceIdentity[1];
    if (!require(!owner.installPreparedRefinedDomain(wrongSource,candidateID,&refinedReport,&error),
                 "rejects prepared rows from a different pinned source")) return 1;
    auto badPrefix=domain;
    badPrefix.coarseRest[0][0]=1.0e-12;
    if (!require(!owner.installPreparedRefinedDomain(badPrefix,candidateID,&refinedReport,&error),
                 "rejects a modified original coarse prefix")) return 1;
    auto bad=domain;
    bad.rows[3].coarseVertex[1]=3;
    if (!require(!owner.installPreparedRefinedDomain(bad,candidateID,&refinedReport,&error),
                 "rejects a transfer support outside its declared child face")) return 1;
    if (!require(!owner.hasPreparedRefinedDomain(),
                 "invalid prepared rows do not partially install")) return 1;
    if (!require(owner.installPreparedRefinedDomain(domain,candidateID,&refinedReport,&error),
                 "accepts validated midpoint refinement rows")) {
        std::cerr << error << "\n"; return 1;
    }
    if (!require(owner.hasPreparedRefinedDomain() &&
                 owner.mechanicalNodeCount()==5 &&
                 owner.coarseMechanicalNodeRows()==domain.coarseMechanicalNodeRows &&
                 refinedReport.generatedCoarseVertices==1 &&
                 refinedReport.activeRows==4 && refinedReport.inactiveRows==0,
                 "installed domain reports its full mechanics indexing")) return 1;
    if (!require(near(refinedReport.maximumBaselineReconstructionErrorMeters,0.0),
                 "generated baseline point is reproduced from source parents")) return 1;

    std::vector<Point> sourceCurrent=detailRest;
    for (auto& p:sourceCurrent) { p[0]+=0.125; p[1]-=0.25; p[2]+=0.5; }
    std::vector<Transfer::Pose> poses(8);
    std::vector<Point> baseline;
    if (!require(owner.makeCurrentMechanicalBaseline(detailRest,sourceCurrent,poses,poses,
                                                     baseline,&error),
                 "full mechanics baseline uses source-face embeddings and prolongation")) {
        std::cerr << error << "\n"; return 1;
    }
    if (!require(baseline.size()==5 && baseline[3]==Point{} &&
                 near(baseline[4][0],0.625) && near(baseline[4][1],-0.25) &&
                 near(baseline[4][2],0.5),
                 "generated outer node baseline is prolonged and nonouter node is zero")) return 1;

    std::vector<Point> accepted=baseline;
    accepted[4][2]+=0.25;
    std::vector<std::array<float,4>> deltas;
    if (!require(owner.makeCoarseDisplacementsFromMechanical(accepted,baseline,deltas,&error),
                 "mechanics displacement adapter accepts the full refined node domain")) {
        std::cerr << error << "\n"; return 1;
    }
    if (!require(deltas.size()==5 && deltas[3]==std::array<float,4>{0,0,0,0} &&
                 near(deltas[4][2],0.25),
                 "only declared coarse mechanics rows carry transfer displacement")) return 1;
    // The original source vertices translate with the source; the projected
    // midpoint row must remain affine-consistent at the unchanged baseline.
    const std::vector<Point>& detailedBaseline=sourceCurrent;
    std::vector<Point> unchanged;
    if (!require(owner.applyReference(detailedBaseline,
                         std::vector<std::array<float,4>>(5),unchanged,&error),
                 "zero displacement applies over the full mechanics domain")) return 1;
    if (!require(unchanged==detailedBaseline,
                 "zero accepted displacement preserves the detailed baseline bitwise")) return 1;
    std::vector<Point> moved;
    if (!require(owner.applyReference(detailedBaseline,deltas,moved,&error),
                 "prepared rows apply refined mechanics displacement")) return 1;
    if (!require(near(moved[3][2],detailedBaseline[3][2]+0.25),
                 "midpoint detailed row follows its imported refined mechanics node")) return 1;
    const std::vector<Point> detailForce{{0.4,-0.2,0.7},{-0.3,0.5,0.1},
                                         {0.8,0.2,-0.4},{-0.6,0.3,0.9}};
    std::vector<Point> mechanicsForce(5,Point{});
    for (std::size_t i=0;i<owner.rows().size();++i) {
        const auto& r=owner.rows()[i];
        for (unsigned k=0;k<3;++k) for (unsigned a=0;a<3;++a)
            mechanicsForce[r.coarseVertex[k]][a] +=
                double(r.weight[k])*detailForce[i][a];
    }
    double detailWork=0.0,mechanicsWork=0.0;
    for (std::size_t i=0;i<moved.size();++i) for(unsigned a=0;a<3;++a)
        detailWork += detailForce[i][a]*(moved[i][a]-detailedBaseline[i][a]);
    for (std::size_t i=0;i<deltas.size();++i) for(unsigned a=0;a<3;++a)
        mechanicsWork += mechanicsForce[i][a]*deltas[i][a];
    if (!require(near(detailWork,mechanicsWork,2e-7),
                 "imported interpolation and its transpose preserve virtual work")) return 1;

    std::cout << "prepared-domain owner check passed: 1 generated node, 2 child faces, "
              << refinedReport.activeRows << " active rows, mechanics domain "
              << owner.mechanicalNodeCount() << "\n";
    return 0;
}
