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
