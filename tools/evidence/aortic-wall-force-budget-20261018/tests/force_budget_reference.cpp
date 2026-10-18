#include "NumiHumanRestingAorticWall.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace numi::human::aortic_wall;

int main() {
    SectionProof proof;
    proof.stableId=9u;
    proof.sourceBody=7u;
    proof.sourcePayloadSHA256="1a191fe2f6c4a1cb03ee290bc850b80011cc267577992ea86abb3a4cb39f4108";
    proof.planePoint={-0.03372473683003,0.05810004997253,-0.00946755672337};
    proof.planeNormal={0.0,1.0,0.0};
    proof.externalAreaM2=2.4192601236904974e-4;
    #include "actual_nha_contour.inc"
    constexpr double pressurePa=11223.5;
    constexpr double targetReservoirVolumeM3=1.84183e-5;
    Config cfg;
    cfg.circumferentialSegments=12u;
    cfg.axialSegments=8u;
    MRBodyStateGPU identity{};
    identity.orientation={0.0f,0.0f,0.0f,1.0f};
    const Built built=build(proof,identity,{0.0,0.0,0.0},7u,
                            identity,7u,targetReservoirVolumeM3,pressurePa,cfg);
    auto annular=makeAnnularMechanicalWall(built,proof,pressurePa);

    std::array<double,3> centroid{};
    for(const Vec& p:annular.object.femNodes)
        for(unsigned k=0;k<3u;++k) centroid[k]+=p[k]/annular.object.femNodes.size();
    double maxOriginalRound=0.0, sumOriginalRound2=0.0, maxOriginalUlp=0.0;
    double maxCenteredRound=0.0, sumCenteredRound2=0.0, maxCenteredUlp=0.0;
    std::size_t coordinateCount=0;
    auto recordRound=[&](double x,double& maximum,double& sum2,double& ulpMax) {
        const float f=static_cast<float>(x);
        const double error=std::abs(double(f)-x);
        maximum=std::max(maximum,error);
        sum2+=error*error;
        const float next=std::nextafterf(f,std::numeric_limits<float>::infinity());
        ulpMax=std::max(ulpMax,double(next)-double(f));
        ++coordinateCount;
    };
    for(const Vec& p:annular.object.femNodes) for(unsigned k=0;k<3u;++k) {
        recordRound(p[k],maxOriginalRound,sumOriginalRound2,maxOriginalUlp);
        recordRound(p[k]-centroid[k],maxCenteredRound,sumCenteredRound2,maxCenteredUlp);
    }

    double forceNorm2=0.0,maxNodeForce=0.0;
    std::array<double,3> floatNet{};
    for(const auto& f:annular.externalForces) {
        const double x=f[0],y=f[1],z=f[2];
        forceNorm2+=x*x+y*y+z*z;
        maxNodeForce=std::max(maxNodeForce,std::sqrt(x*x+y*y+z*z));
        floatNet[0]+=x;floatNet[1]+=y;floatNet[2]+=z;
    }
    const double forceL2=std::sqrt(forceNorm2);
    constexpr double dt=1.0/4000.0;
    const double strictImpulse[]={4.17371922e-7,9.99744998e-8,1.07705631e-7};
    const char* labels[]={"uncentered_step0_reject","centered_step0_accept","centered_step1_reject"};
    std::cout<<std::setprecision(15)
        <<"nodes="<<annular.object.femNodes.size()<<"\n"
        <<"tets="<<annular.object.tetrahedra.size()<<"\n"
        <<"source_centroid_m="<<centroid[0]<<","<<centroid[1]<<","<<centroid[2]<<"\n"
        <<"coordinate_count="<<coordinateCount/2<<"\n"
        <<"uncentered_fp32_max_rounding_m="<<maxOriginalRound<<"\n"
        <<"uncentered_fp32_rms_rounding_m="<<std::sqrt(sumOriginalRound2/double(coordinateCount/2))<<"\n"
        <<"uncentered_fp32_max_ulp_m="<<maxOriginalUlp<<"\n"
        <<"centered_fp32_max_rounding_m="<<maxCenteredRound<<"\n"
        <<"centered_fp32_rms_rounding_m="<<std::sqrt(sumCenteredRound2/double(coordinateCount/2))<<"\n"
        <<"centered_fp32_max_ulp_m="<<maxCenteredUlp<<"\n"
        <<"external_nodal_force_l2_N="<<forceL2<<"\n"
        <<"external_nodal_force_max_N="<<maxNodeForce<<"\n"
        <<"external_nodal_force_float_net_N="<<std::sqrt(floatNet[0]*floatNet[0]+floatNet[1]*floatNet[1]+floatNet[2]*floatNet[2])<<"\n"
        <<"pressure_Pa="<<pressurePa<<"\n"
        <<"timestep_s="<<dt<<"\n";
    for(unsigned i=0;i<3u;++i) {
        const double forceResidual=strictImpulse[i]/dt;
        std::cout<<labels[i]<<"_certificate_impulse_norm_Ns="<<strictImpulse[i]<<"\n"
            <<labels[i]<<"_force_scale_N="<<forceResidual<<"\n"
            <<labels[i]<<"_force_scale_over_external_L2="<<forceResidual/forceL2<<"\n";
    }
}
