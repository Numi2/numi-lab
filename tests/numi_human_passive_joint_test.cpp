#include "metalrobo/NumiHumanPassiveJoint.hpp"
#include "metalrobo/numi_human_passive_joint.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using metalrobo::NumiHumanPassiveCoordinateCoupling;
    using metalrobo::compileNumiHumanPassiveJointProgram;
    using metalrobo::validateNumiHumanPassiveJointProgram;
    using metalrobo::validateNumiHumanHipCapsuleTerms;
    std::size_t checks = 0;
    const auto require = [&](bool value, const char* message) {
        ++checks;
        if (!value) throw std::runtime_error(message);
    };
    const auto near = [&](double a, double b, double absolute, const char* label) {
        require(std::isfinite(a) && std::abs(a-b) <= absolute, label);
    };
    std::vector<NumiHumanPassiveCoordinateCoupling> rows{
        {40,40,0.1,1.74}, {40,41,-0.2,-0.18},
        {41,40,0.1,-0.18}, {41,41,-0.2,1.28},
        {46,46,0.0,0.054261}, {47,47,0.0,0.1779},
        {48,48,0.0,0.0231}, {49,49,0.0,0.0037206},
    };
    constexpr std::size_t nv = 128;
    std::vector<float> program;
    std::string error;
    require(compileNumiHumanPassiveJointProgram(rows,nv,program,error),
            "authored wrist/finger program failed");
    require(program.size() == nv*(nv+1), "program shape");
    require(error.empty(), "success error not cleared");
    const auto accepted = program;
    auto mutated = rows;
    const auto rejected = [&](const std::vector<NumiHumanPassiveCoordinateCoupling>& candidate) {
        program = accepted;
        require(!compileNumiHumanPassiveJointProgram(candidate,nv,program,error),
                "invalid stiffness admitted");
        require(program == accepted && !error.empty(), "failure mutated accepted program");
    };
    mutated[0].targetDofIndex = 2; rejected(mutated);
    mutated = rows; mutated[0].sourceDofIndex = 128; rejected(mutated);
    mutated = rows; mutated[1].stiffness = 0.18; rejected(mutated);
    mutated = rows; mutated[0].stiffness = -1; rejected(mutated);
    mutated = rows; mutated[0].sourceRestPosition = 0.2; rejected(mutated);
    mutated = rows; mutated[0].stiffness = std::numeric_limits<double>::infinity(); rejected(mutated);
    mutated = rows; mutated[0].sourceRestPosition = std::numeric_limits<double>::quiet_NaN(); rejected(mutated);
    mutated = rows; mutated[0].stiffness = 1e50; rejected(mutated);
    mutated = rows; mutated[0].stiffness = 1e-50; rejected(mutated);
    mutated = {{6,6,0,1},{6,7,0,2},{7,6,0,2},{7,7,0,1}}; rejected(mutated);
    require(compileNumiHumanPassiveJointProgram({},nv,program,error), "zero law failed");
    require(compileNumiHumanPassiveJointProgram(
        std::vector<NumiHumanPassiveCoordinateCoupling>{{6,6,0,1},{6,7,0,-1},{7,6,0,-1},{7,7,0,1}},
        nv,program,error), "valid null direction rejected");
    require(!validateNumiHumanPassiveJointProgram(program,127,error), "wrong shape admitted");
    program[6*nv+7] = -1.01f; program[7*nv+6] = -1.01f;
    require(!validateNumiHumanPassiveJointProgram(program,nv,error), "negative eigenvalue admitted");
    program = accepted;
    program[0] = 1;
    require(!validateNumiHumanPassiveJointProgram(program,nv,error), "root spring admitted");
    program = accepted;
    program[6*nv+40] = program[40*nv+6] = 0.01f;
    require(!validateNumiHumanPassiveJointProgram(program,nv,error), "zero diagonal coupling admitted");
    program = accepted;
    const auto force = [&](double x, double y) {
        const double dx=x-program[nv*nv+40], dy=y-program[nv*nv+41];
        return std::array<double,2>{-(program[40*nv+40]*dx+program[40*nv+41]*dy),
                                   -(program[41*nv+40]*dx+program[41*nv+41]*dy)};
    };
    const auto energy = [&](double x,double y) {
        auto f=force(x,y);
        return -0.5*((x-program[nv*nv+40])*f[0]+(y-program[nv*nv+41])*f[1]);
    };
    near(force(0.3,0.4)[0], -(1.74*0.2-0.18*0.6), 1e-7, "source force mismatch");
    near(force(0.3,0.4)[0], -(energy(0.300001,0.4)-energy(0.299999,0.4))/0.000002,
         1e-9, "force is not the potential gradient");
    require(force(0.3,0.4) != force(0.4,0.4), "force remained frozen after movement");
    near(energy(program[nv*nv+40],program[nv*nv+41]),0,0,"rest energy");

    // Independent scalar oscillator oracle for the acceleration-form update.
    // At a deliberately stiff timestep, explicit spring integration is not
    // adequate. The production terms must include both h*K*v and h^2*K.
    for (float h : {0.0000125f,0.0001f,0.02f}) {
        const float k=100000.0f,mass=0.01f;
        double q=0.01,v=0.1;
        double previous=0.5*mass*v*v+0.5*k*q*q;
        for (int step=0;step<100;++step) {
            const float bias=mrNumiHumanPassiveImplicitBias(k,static_cast<float>(q),static_cast<float>(v),h);
            const float effective=mass+mrNumiHumanPassiveEffectiveInertia(k,h);
            const double actualV=v+h*(-static_cast<double>(bias)/effective);
            const double expectedV=(mass*v-h*k*q)/(mass+h*h*static_cast<double>(k));
            near(actualV,expectedV,1e-5,"backward-Euler velocity disagrees with closed form");
            q+=h*actualV;v=actualV;
            const double current=0.5*mass*v*v+0.5*k*q*q;
            require(current <= previous+1e-7,"passive update generated energy");
            previous=current;
        }
    }
    near(mrNumiHumanPassiveImplicitBias(2,0.5,0.25,0.1),1.05,1e-7,"implicit RHS");
    near(mrNumiHumanPassiveEffectiveInertia(2,0.1),0.02,1e-8,"implicit tangent");

    // Convex unilateral periarticular toe: C2 at slack, with an analytic
    // gradient and positive-semidefinite rank-one tangent.
    MRNumiHumanHipCapsuleTermGPU hip{};
    hip.dofIndex0=100; hip.dofIndex1=102;
    hip.coordinate0=2.0f; hip.coordinate1=-3.0f;
    hip.threshold=0.25f; hip.toeQuadratic=100.0f; hip.toeCubic=200.0f;
    std::array<MRNumiHumanHipCapsuleTermGPU,1> hipTerms{hip};
    require(validateNumiHumanHipCapsuleTerms(hipTerms,128,error),
            "valid hip-capsule descriptor rejected");
    const auto gap = [&](double x,double y) {
        return 2.0*x-3.0*y-0.25;
    };
    const auto energyHip = [&](double x,double y) {
        const double g=std::max(0.0,gap(x,y));
        return (100.0/3.0)*g*g*g+(200.0/4.0)*g*g*g*g;
    };
    const auto restoring = [&](double g) {
        return 100.0*g*g+200.0*g*g*g;
    };
    const auto tangent = [&](double g) {
        return 200.0*g+600.0*g*g;
    };
    require(mrNumiHumanHipCapsulePotential(hip,-0.1f)==0.0f &&
                mrNumiHumanHipCapsuleRestoringMagnitude(hip,0.0f)==0.0f &&
                mrNumiHumanHipCapsuleTangent(hip,0.0f)==0.0f,
            "hip-capsule slack branch is not exactly force/tangent free");
    const double x=0.3,y=0.1,g=gap(x,y);
    near(mrNumiHumanHipCapsuleGap(hip,static_cast<float>(x),static_cast<float>(y)),
         g,1e-7,"hip-capsule gap mismatch");
    near(mrNumiHumanHipCapsulePotential(hip,static_cast<float>(g)),
         energyHip(x,y),2e-7,"hip-capsule energy mismatch");
    const double force0=-hip.coordinate0*restoring(g);
    const double force1=-hip.coordinate1*restoring(g);
    near(force0,-(energyHip(x+1e-6,y)-energyHip(x-1e-6,y))/2e-6,
         3e-5,"hip-capsule q0 force is not the negative potential gradient");
    near(force1,-(energyHip(x,y+1e-6)-energyHip(x,y-1e-6))/2e-6,
         3e-5,"hip-capsule q1 force is not the negative potential gradient");
    const double k=tangent(g);
    const double h=0.01, qd=hip.coordinate0*0.2+hip.coordinate1*(-0.1);
    const double bias0=hip.coordinate0*(restoring(g)+h*k*qd);
    near(mrNumiHumanHipCapsuleEffectiveInertia(
             hip,static_cast<float>(g),static_cast<float>(h),100,102),
         h*h*k*hip.coordinate0*hip.coordinate1,2e-7,
         "hip-capsule tangent cross-block mismatch");
    near(mrNumiHumanHipCapsuleImplicitBias(
             hip,static_cast<float>(g),static_cast<float>(qd),
             static_cast<float>(h),100),
         bias0,2e-6,"shared implicit bias helper mismatch");
    const double k00=mrNumiHumanHipCapsuleEffectiveInertia(
        hip,static_cast<float>(g),static_cast<float>(h),100,100);
    const double k01=mrNumiHumanHipCapsuleEffectiveInertia(
        hip,static_cast<float>(g),static_cast<float>(h),100,102);
    const double k11=mrNumiHumanHipCapsuleEffectiveInertia(
        hip,static_cast<float>(g),static_cast<float>(h),102,102);
    require(k00>=0.0 && k11>=0.0 && k00*k11-k01*k01 >= -1e-8,
            "hip-capsule implicit tangent is not positive semidefinite");
    const double dE00=(energyHip(x+1e-5,y)-2*energyHip(x,y)+energyHip(x-1e-5,y))/(1e-10);
    near(dE00,hip.coordinate0*hip.coordinate0*k,0.02,
         "hip-capsule potential Hessian mismatch");
    require(k>=0.0 && hip.coordinate0*hip.coordinate0*k>=0.0 &&
                hip.coordinate1*hip.coordinate1*k>=0.0,
            "hip-capsule tangent is not positive semidefinite");

    const auto invalidHip = [&](MRNumiHumanHipCapsuleTermGPU bad,
                                const char* message) {
        std::array<MRNumiHumanHipCapsuleTermGPU,1> candidate{bad};
        require(!validateNumiHumanHipCapsuleTerms(candidate,128,error),message);
        require(!error.empty(),"invalid hip descriptor did not explain rejection");
    };
    auto badHip=hip; badHip.dofIndex0=5;
    invalidHip(badHip,"floating root hip coordinate admitted");
    badHip=hip; badHip.dofIndex1=100;
    invalidHip(badHip,"duplicate hip coordinate admitted");
    badHip=hip; badHip.toeCubic=-1.0f;
    invalidHip(badHip,"negative hip toe coefficient admitted");
    badHip=hip; badHip.coordinate0=std::numeric_limits<float>::infinity();
    invalidHip(badHip,"nonfinite hip coordinate coefficient admitted");
    badHip=hip; badHip.reserved0=1u;
    invalidHip(badHip,"nonzero hip reserved field admitted");
    require(!validateNumiHumanHipCapsuleTerms(hipTerms,161,error),
            "hip descriptor exceeded fixed DoF limit");
    std::array<MRNumiHumanHipCapsuleTermGPU,
        MR_NUMI_HUMAN_HIP_CAPSULE_MAX_TERMS+1u> oversizedHip{};
    oversizedHip.fill(hip);
    require(!validateNumiHumanHipCapsuleTerms(
                std::span<const MRNumiHumanHipCapsuleTermGPU>(oversizedHip),
                128,error),
            "hip descriptor exceeded fixed term capacity");

    std::cout << "Human passive joints: " << checks << " checks passed\n";
}
