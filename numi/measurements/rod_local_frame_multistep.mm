#include "rod_local_frame.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

static std::vector<double> solve(std::vector<std::vector<double>> a, std::vector<double> b) {
    const std::size_t n=b.size();
    for (std::size_t j=0;j<n;++j) {
        std::size_t p=j;
        for (std::size_t i=j+1;i<n;++i) if (std::abs(a[i][j])>std::abs(a[p][j])) p=i;
        if (!(std::abs(a[p][j])>1e-18)) return {};
        std::swap(a[p],a[j]); std::swap(b[p],b[j]);
        for (std::size_t i=j+1;i<n;++i) {
            const double f=a[i][j]/a[j][j];
            for (std::size_t k=j;k<n;++k) a[i][k]-=f*a[j][k];
            b[i]-=f*b[j];
        }
    }
    std::vector<double> x(n);
    for (std::size_t j=n;j-- >0;) {
        double r=b[j];
        for (std::size_t k=j+1;k<n;++k) r-=a[j][k]*x[k];
        x[j]=r/a[j][j];
    }
    return x;
}
int main(int argc,char**argv) {
    if(argc!=2) return 2;
    constexpr unsigned n=128;constexpr double rest=.11758755785232791,ea=2.3818718058345718;
    constexpr double radius=.000020,h=.1,speed=.0005;
    metalrobo::DiscreteRodMaterial mat;mat.radius=radius;mat.density=1100;
    mat.youngModulus=ea/(std::numbers::pi*radius*radius);
    const auto model=metalrobo::makeStraightSutureRod(n,rest,mat);
    const std::array<double,2> origin{0,16};
    std::vector<metalrobo::DiscreteElasticRodState> states;
    std::vector<metalrobo::DiscreteRodAttachment> at(4);
    for(unsigned e=0;e<2;++e) {
        auto x=metalrobo::makeDiscreteElasticRodDefaultState(model);
        for(unsigned i=0;i<n;++i) {
            x.positions[i][0]=origin[e]+.122*double(i)/(n-1);
            x.velocities[i][0]=speed*double(i)/(n-1);
        }
        states.push_back(x);
        at[2*e].nodeIndex=0;at[2*e+1].nodeIndex=n-1;
        at[2*e+1].targetVelocity={speed,0,0};
    }
    metalrobo::MetalDiscreteElasticRodConfig cfg;cfg.metallibPath=argv[1];cfg.step.timestep=h;
    cfg.step.gravity={0,0,0};cfg.step.linearDamping=0;cfg.step.twistDamping=0;
    cfg.step.solverIterations=128;cfg.step.constraintTolerance=1e-7;
    std::array<double,2> work{},firstEnergy{},oldForce{},oldTarget{},maxPe{},maxVe{},maxFe{};
    bool all=true;
    std::cout<<std::setprecision(17);
    for(unsigned step=0;step<=40;++step) {
        for(unsigned e=0;e<2;++e) {
            at[2*e].targetPosition={origin[e],0,0};
            at[2*e+1].targetPosition={origin[e]+.122+double(step)*h*speed,0,0};
        }
        std::vector<std::vector<double>> refs(2,std::vector<double>(n));
        std::array<double,2> leftForce{},rightForce{};
        for(unsigned e=0;e<2;++e) {
            auto& r=refs[e];r[0]=at[2*e].targetPosition[0];r[n-1]=at[2*e+1].targetPosition[0];
            std::vector<std::vector<double>> A(n-2,std::vector<double>(n-2));
            std::vector<double> B(n-2);
            for(unsigned i=1;i+1<n;++i) {
                const double kl=model.stretchStiffness[i-1]/model.restLengths[i-1];
                const double kr=model.stretchStiffness[i]/model.restLengths[i];
                const double mass=model.nodeMasses[i]/(h*h);
                A[i-1][i-1]=mass+kl+kr;
                B[i-1]=mass*(states[e].positions[i][0]+h*states[e].velocities[i][0])+
                    kl*model.restLengths[i-1]-kr*model.restLengths[i];
                if(i>1) A[i-1][i-2]=-kl; else B[i-1]+=kl*r[0];
                if(i+2<n) A[i-1][i]=-kr; else B[i-1]+=kr*r[n-1];
            }
            const auto inner=solve(A,B);
            if(inner.size()!=n-2) return 2;
            for(unsigned i=1;i+1<n;++i) r[i]=inner[i-1];
            for(unsigned i=0;i+1<n;++i) if(!(r[i+1]>r[i])) return 2;
            const double kl=model.stretchStiffness[0]/model.restLengths[0];
            const double kr=model.stretchStiffness[n-2]/model.restLengths[n-2];
            leftForce[e]=kl*(r[1]-r[0]-model.restLengths[0])+
                model.nodeMasses[0]*states[e].velocities[0][0]/h;
            rightForce[e]=-kr*(r[n-1]-r[n-2]-model.restLengths[n-2])+
                model.nodeMasses[n-1]*(states[e].velocities[n-1][0]-speed)/h;
        }
        metalrobo::MetalDiscreteElasticRodInput input;
        input.states=states;input.attachmentCount=2;input.attachments=at;
        metalrobo::MetalDiscreteElasticRodResult out;
        const auto d=rod_local_research::run(model,input,out,cfg);
        if(!d.succeeded()||!d.published||out.states.size()!=2||out.reactions.size()!=4) {
            std::cerr<<"step="<<step<<" host="<<metalrobo::metalDiscreteElasticRodHostStatusName(d.status)
                     <<" gpu="<<d.firstGPUStatusCode<<" message="<<d.message<<"\n";
            return 1;
        }
        for(unsigned e=0;e<2;++e) {
            for(unsigned i=0;i<n;++i) {
                const double expectedV=i==0?0:(i==n-1?speed:(refs[e][i]-states[e].positions[i][0])/h);
                maxPe[e]=std::max(maxPe[e],std::abs(out.states[e].positions[i][0]-refs[e][i]));
                maxVe[e]=std::max(maxVe[e],std::abs(out.states[e].velocities[i][0]-expectedV));
            }
            const double actualLeft=out.reactions[2*e].averageForceOnTarget[0];
            const double actualRight=out.reactions[2*e+1].averageForceOnTarget[0];
            const double errL=std::abs(actualLeft-leftForce[e]),errR=std::abs(actualRight-rightForce[e]);
            maxFe[e]=std::max({maxFe[e],errL,errR});
            const bool force=errL<=std::max(1e-6,.01*std::abs(leftForce[e]))&&
                             errR<=std::max(1e-6,.01*std::abs(rightForce[e]));
            all &= force;
            double energy=0;
            for(unsigned i=0;i+1<n;++i) {
                const double extension=out.states[e].positions[i+1][0]-out.states[e].positions[i][0]-
                                       model.restLengths[i];
                energy+=.5*(model.stretchStiffness[i]/model.restLengths[i])*extension*extension;
            }
            for(unsigned i=0;i<n;++i) energy+=.5*model.nodeMasses[i]*
                out.states[e].velocities[i][0]*out.states[e].velocities[i][0];
            const double target=at[2*e+1].targetPosition[0];
            if(step==0) firstEnergy[e]=energy;
            else work[e]+=.5*(-actualRight-oldForce[e])*(target-oldTarget[e]);
            oldForce[e]=-actualRight;oldTarget[e]=target;
            if(step==40) {
                const double delta=energy-firstEnergy[e];
                const double workError=std::abs(work[e]-delta)/std::abs(delta);
                const bool pass=force&&maxPe[e]<=1e-6&&maxVe[e]<=.001&&std::isfinite(workError)&&workError<=.01;
                all &= pass;
                std::cout<<"offset_m="<<origin[e]<<" accepted=41 max_position_error_m="<<maxPe[e]
                         <<" max_velocity_error_m_s="<<maxVe[e]<<" max_force_error_N="<<maxFe[e]
                         <<" work_J="<<work[e]<<" energy_change_J="<<delta
                         <<" work_relative_error="<<workError<<" passed="<<pass<<"\n";
            }
        }
        states=out.states;
    }
    return all?0:1;
}
