#include "ConnectedSkinMechanicalTransfer013.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace numi::human::connected_skin_transfer013;

struct Fixture {
    std::vector<Point> source{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{.5,.5,0}};
    std::vector<Face> sourceFaces{{0,1,4},{1,2,4},{2,3,4},{3,0,4}};
    std::vector<std::uint32_t> detailSource{0,1,2,3,4};
    std::vector<Point> detail=source;
    std::vector<Face> detailFaces{{0,1,4},{1,2,4},{2,3,4},{3,0,4}};
    std::vector<std::uint8_t> detailMayMove{1,1,1,1,1}, detailProtected{0,0,0,0,0};
    std::vector<Point> physical{{0,0,0},{1,0,0},{0,1,0}};
    std::vector<Face> physicalFaces{{0,1,2}};
    // Physical points address detailed source face rows using the registered dense face domain.
    std::vector<std::uint32_t> physicalFaceRows{0,0,3};
    std::vector<Point> physicalBary{{1,0,0},{0,1,0},{1,0,0}};
    std::vector<Point> sourceResidual{Point{},Point{},Point{}};
    std::vector<std::uint8_t> physicalMayMove{1,1,1}, physicalProtected{0,0,0};
    std::vector<TransferRow> rows{
        {{{0,0,0}},{{1,0,0}},true},
        {{{1,0,0}},{{1,0,0}},true},
        {{{2,0,0}},{{1,0,0}},true},
        {{{2,0,0}},{{1,0,0}},true},
        {{{0,1,2}},{{1.0f/3.0f,1.0f/3.0f,1.0f/3.0f}},true}
    };
    std::vector<Point> delta{{0,0,0},{.1,0,0},{0,.2,0}};
    std::vector<double> mass{1.25,2.5,3.75};
    Input makeInput() const {
        Input x;
        x.sourceRestWorldM=source;x.sourceFaces=sourceFaces;x.detailSourceVertexIds=detailSource;
        x.detailRestWorldM=detail;x.detailFaces=detailFaces;x.detailMayMove=detailMayMove;
        x.detailProtected=detailProtected;x.physicalOuterRestWorldM=physical;
        x.physicalOuterFaces=physicalFaces;x.physicalSourceFaceRows=physicalFaceRows;
        x.physicalSourceBarycentric=physicalBary;x.physicalSourceResidualWorldM=sourceResidual;
        x.physicalMayMove=physicalMayMove;x.physicalProtected=physicalProtected;
        x.detailFromPhysical=rows;x.physicalDeltaWorldM=delta;x.physicalNodeMassKg=mass;
        x.pins={std::string(64,'a'),std::string(64,'b'),std::string(64,'c')};
        x.reconstructionToleranceM=1e-9;x.displacementToleranceM=1e-12;
        return x;
    }
};

static bool run(Fixture& f, Output& out, std::string& error) {
    auto in=f.makeInput();
    return transfer(in,out,error);
}

static void testCoupledConservativeTransfer() {
    Fixture f;Output out;std::string error;
    assert(run(f,out,error));
    assert(out.report.activeDetailRows==5);
    assert(out.report.movedPhysicalRows==2);
    assert(out.report.movedDetailRows==4);
    assert(out.report.physicalMassRowsBitwiseUnchanged);
    assert(out.physicalNodeMassKg==f.mass);
    assert(out.report.physicalMassBeforeKg==out.report.physicalMassAfterKg);
    assert(out.report.maxRestEmbeddingErrorM==0.0);
    assert(out.report.maxMovedSourceResidualErrorM<1e-12);
    assert(out.report.momentClosureErrorM3<1e-12);
    // Rigid translation is reproduced exactly by every active partition-of-unity row.
    for(auto& d:f.delta)d={.002,-.003,.004};
    assert(run(f,out,error));
    for(const auto& d:out.detailDeltaWorldM)
        for(unsigned a=0;a<3;++a)assert(std::abs(d[a]-f.delta[0][a])<1e-8);
}

static void testResidualRecordsNoncommutingFit() {
    Fixture f;f.delta[2]={0,.3,0};f.rows[3].physicalVertex={1,0,0};Output out;std::string error;
    assert(run(f,out,error));
    assert(std::abs(out.physicalSourceResidualWorldM[2][1]-.3)<1e-12);
    assert(std::abs(out.report.maxSourceResidualChangeM-std::sqrt(.10))<1e-12);
    // The physical fit is retained; the source residual records the exact gap.
    assert(std::abs(out.physicalOuterWorldM[2][1]-1.3)<1e-12);
}

static void testProtectedRowsFailClosed() {
    Fixture f;f.physicalProtected[1]=1;Output out;std::string error;
    assert(!run(f,out,error));
    assert(error.find("protected physical")!=std::string::npos);
    f=Fixture{};f.detailProtected[4]=1;error.clear();
    assert(!run(f,out,error));
    assert(error.find("protected detailed")!=std::string::npos);
    f=Fixture{};f.detailMayMove[4]=0;error.clear();
    assert(!run(f,out,error));
    assert(error.find("outside the declared repair patch")!=std::string::npos);
}

static void testMappingAndIdentityValidation() {
    Fixture f;Output out;std::string error;
    f.sourceResidual[0][0]=1e-4;
    assert(!run(f,out,error));
    assert(error.find("does not reconstruct")!=std::string::npos);
    f=Fixture{};f.rows[4].weight[0]=.5f;
    assert(!run(f,out,error));
    assert(error.find("partition of unity")!=std::string::npos);
    f=Fixture{};auto in=f.makeInput();in.pins.poseSha256="not-a-hash";
    assert(!transfer(in,out,error));
    assert(error.find("SHA-256 pins")!=std::string::npos);
}

struct AcceptedOwnerRowFixture {
    std::uint32_t coarseVertex[3]{};
    float weight[3]{};
};

static void testAcceptedOwnerRowAdapterAndSourceMismatch() {
    Fixture f;
    std::vector<AcceptedOwnerRowFixture> ownerRows(f.rows.size());
    std::vector<std::uint8_t> active(f.rows.size(),1);
    for(std::size_t i=0;i<f.rows.size();++i) {
        for(unsigned k=0;k<3;++k) {
            ownerRows[i].coarseVertex[k]=f.rows[i].physicalVertex[k];
            ownerRows[i].weight[k]=f.rows[i].weight[k];
        }
    }
    std::vector<TransferRow> adapted;
    std::string error;
    assert(copyAcceptedOwnerRows(std::span<const AcceptedOwnerRowFixture>(ownerRows),
                                 std::span<const std::uint8_t>(active),adapted,error));
    auto in=f.makeInput();
    in.detailFromPhysical=adapted;
    Output out;
    assert(transfer(in,out,error));
    assert(out.report.activeDetailRows==f.rows.size());

    f=Fixture{};
    f.detail[3][0]+=0.01;
    error.clear();
    assert(!run(f,out,error));
    assert(error.find("direct source identity disagree")!=std::string::npos);
}

static void testInactiveDetailAreaIsReportedNotInvented() {
    Fixture f;f.rows[4].active=false;f.rows[4].physicalVertex={0,0,0};
    f.rows[4].weight={0,0,0};f.delta[0]={0,0,0};f.delta[1]={0,0,0};f.delta[2]={0,0,0};
    Output out;std::string error;
    assert(run(f,out,error));
    assert(out.report.inactiveDetailRows==1);
    assert(out.report.unmappedDetailedAreaM2>0.0);
    assert(out.report.detailedSurfaceAreaM2>
           out.report.inducedPhysicalSurfaceAreaM2);
    assert(out.detailWorldM[4]==f.detail[4]);
}

int main() {
    testCoupledConservativeTransfer();
    testResidualRecordsNoncommutingFit();
    testProtectedRowsFailClosed();
    testMappingAndIdentityValidation();
    testAcceptedOwnerRowAdapterAndSourceMismatch();
    testInactiveDetailAreaIsReportedNotInvented();
    std::cout << "PASS coupled O/FEM-to-detail transfer, source residual closure, protected back/attachments, area-moment conservation, exact mass pass-through, fail-closed ancestry and inactive-area reporting\n";
}
