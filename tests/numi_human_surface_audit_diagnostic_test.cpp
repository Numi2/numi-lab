#include "../apps/NumiHumanRestingSurfaceAuditDiagnostic.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {
void expect(bool condition,const char* message) {
    if(!condition){std::cerr<<"FAIL: "<<message<<'\n';std::exit(1);}
}
}

int main() {
    static_assert(sizeof(MRHumanRestingSurfaceFailureGPU)==80);
    static_assert(alignof(MRHumanRestingSurfaceFailureGPU)==16);
    const mr_float4 a{0,0,0,0},b{1,0,0,0},c{0,1,0,0};
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(a,b,c)==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE,"valid triangle is not rejected");
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(a,b,b)==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA,"duplicate-point triangle is exact zero area");
    const mr_float4 collinear{2,0,0,0};
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(a,b,collinear)==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA,"collinear triangle is exact zero area");
    const mr_float4 witnessA{0.03748244047164917f,-0.39636877179145813f,0.09854736924171448f,1};
    const mr_float4 witnessB{0.03771711140871048f,-0.3969446122646332f,0.09887465834617615f,1};
    const mr_float4 witnessC{0.03795178234577179f,-0.3975204527378082f,0.09920194745063782f,1};
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(witnessA,witnessB,witnessC)==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA,
        "fused cross residual does not hide an exact binary32 collinear triangle");
    const float positiveInfinity=std::numeric_limits<float>::infinity();
    auto oneUlpC=witnessC;
    oneUlpC.x=std::nextafter(oneUlpC.x,positiveInfinity);
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(witnessA,witnessB,oneUlpC)==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE,"one-ULP noncollinear x perturbation stays valid");
    oneUlpC=witnessC;oneUlpC.y=std::nextafter(oneUlpC.y,positiveInfinity);
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(witnessA,witnessB,oneUlpC)==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE,"one-ULP noncollinear y perturbation stays valid");
    oneUlpC=witnessC;oneUlpC.z=std::nextafter(oneUlpC.z,positiveInfinity);
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(witnessA,witnessB,oneUlpC)==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE,"one-ULP noncollinear z perturbation stays valid");
    const mr_float4 scaledA{0.07496488094329834f,-0.7927375435829163f,0.19709473848342896f,1};
    const mr_float4 scaledB{0.07543422281742096f,-0.7938892245292664f,0.1977493166923523f,1};
    const mr_float4 scaledC{0.07590356469154358f,-0.7950409054756165f,0.19840389490127563f,1};
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(scaledA,scaledB,scaledC)==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA,"power-of-two rescaling preserves exact collinearity");
    const mr_float4 translatedScaledA{-0.23125877976417542f,0.051815614104270935f,-0.07572631537914276f,1};
    const mr_float4 translatedScaledB{-0.23114144802093506f,0.05152769386768341f,-0.07556267082691193f,1};
    const mr_float4 translatedScaledC{-0.2310241162776947f,0.051239773631095886f,-0.07539902627468109f,1};
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(
        translatedScaledA,translatedScaledB,translatedScaledC)==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA,
        "translated power-of-two rescaling preserves exact collinearity");
    const mr_float4 roundedTranslatedC{0.5759035348892212f,-1.7950408458709717f,0.44840389490127563f,1};
    const mr_float4 roundedTranslatedA{0.5749648809432983f,-1.7927374839782715f,0.44709473848342896f,1};
    const mr_float4 roundedTranslatedB{0.5754342079162598f,-1.7938892841339111f,0.4477493166923523f,1};
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(
        roundedTranslatedA,roundedTranslatedB,roundedTranslatedC)==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE,
        "rounded translated near-collinear control remains nonzero");
    const float tinyEdge=0x1p-80f;
    const mr_float4 tinyA{0,0,0,1},tinyB{tinyEdge,0,0,1},tinyC{0,tinyEdge,0,1};
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(tinyA,tinyB,tinyC)==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE,
        "nonzero exact area is not called collapsed when its float cross underflows");
    const mr_float4 tinyCollinearC{2*tinyEdge,2*tinyEdge,2*tinyEdge,1};
    const mr_float4 tinyCollinearB{tinyEdge,tinyEdge,tinyEdge,1};
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(
        tinyA,tinyCollinearB,tinyCollinearC)==MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA,
        "integer fallback still recognizes collinear subnormal products");
    const float large=0x1p60f;
    const mr_float4 translatedA{large,0,0,1},translatedB{1,1,0,1},translatedC{-large,2,0,1};
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(
        translatedA,translatedB,translatedC)==MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE,
        "inexact origin subtraction does not turn a rationally nonzero triangle into a false zero");
    const mr_float4 largeCollinearA{large,large,0,1},largeCollinearB{1,1,0,1},largeCollinearC{-large,-large,0,1};
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(
        largeCollinearA,largeCollinearB,largeCollinearC)==MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA,
        "exact point-coordinate fallback recognizes collinearity after inexact origin subtraction");
    const float extreme=0x1p100f;
    expect(numiHumanRestingSurfaceAudit::exactBinary32TriangleIsZero(
        extreme,extreme,0.0f,1.0f,1.0f,0.0f,-extreme,-extreme,0.0f),
        "exact point-coordinate predicate handles finite binary32 products across the full exponent range");
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(
        {extreme,extreme,0.0f,1.0f},{1.0f,1.0f,0.0f,1.0f},{-extreme,-extreme,0.0f,1.0f})==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA,
        "overflowing float cross remains rejected even when exact point determinant is zero");
    const mr_float4 nonfinite{0,0,NAN,0};
    expect(numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(a,b,nonfinite)==
        MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA,"non-finite rendered position is distinguished");

    MRHumanRestingSurfaceFailureGPU failure{};
    failure.surfaceTriangleKind={4,17,MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA,0};
    failure.vertexIndices={100,101,102,0};
    failure.renderedPositions[0]={0.125f,-0.25f,0.5f,0};
    failure.renderedPositions[1]={0.125f,-0.25f,0.5f,0};
    failure.renderedPositions[2]={0.125f,-0.25f,0.50000006f,0};
    const std::string zeroMessage=numiHumanRestingSurfaceAudit::describeSurfaceFailure(
        305,4,900,2,7.9e-7f,failure,"0123456789abcdef");
    expect(zeroMessage.find("physical_endpoint=accepted surface_endpoint=rejected")!=std::string::npos,
        "failure text separates accepted physics from rejected presentation");
    expect(zeroMessage.find("first_invalid_triangle_local=17")!=std::string::npos,
        "failure text retains the first local triangle index");
    expect(zeroMessage.find("index_buffer_offset=951")!=std::string::npos,
        "failure text maps local triangle back to source index buffer");
    expect(zeroMessage.find("area_failure=exact_zero_area")!=std::string::npos,
        "failure text distinguishes exact zero area");
    expect(zeroMessage.find("mesh_vertex_indices=[100,101,102]")!=std::string::npos,
        "failure text retains source mesh vertex indices");
    expect(zeroMessage.find("rendered_positions_f32_bits=[")!=std::string::npos,
        "failure text retains exact rendered binary32 positions");

    failure.surfaceTriangleKind.z=MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA;
    failure.renderedPositions[2]=nonfinite;
    const std::string nonfiniteMessage=numiHumanRestingSurfaceAudit::describeSurfaceFailure(
        305,4,900,2,7.9e-7f,failure,"0123456789abcdef");
    expect(nonfiniteMessage.find("area_failure=nonfinite_area")!=std::string::npos,
        "failure text distinguishes non-finite area");
    expect(nonfiniteMessage.find("7fc00000")!=std::string::npos,
        "non-finite diagnostic preserves the rendered float payload bits");

    failure.surfaceTriangleKind.x=MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE;
    const std::string volumeMessage=numiHumanRestingSurfaceAudit::describeSurfaceFailure(
        305,4,900,1,3e-4f,failure,"0123456789abcdef");
    expect(volumeMessage.find("failure=volume_owner_mismatch")!=std::string::npos,
        "volume-owner-only failure remains distinct from triangle failure");
    expect(volumeMessage.find("first_invalid_triangle=not_applicable")!=std::string::npos,
        "volume-only failure does not invent a triangle witness");
    return 0;
}
