#include "metalrobo/MetalDiscreteElasticRod.hpp"
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
    if (argc!=2) return 2;
    constexpr double rest=.11758755785232791, ea=2.3818718058345718, radius=.000020;
    const double h=static_cast<float>(.001);
    metalrobo::DiscreteRodMaterial mat; mat.radius=radius;mat.density=1100;
    mat.youngModulus=ea/(std::numbers::pi*radius*radius);
    bool all=true;
    std::cout<<std::setprecision(17);
    for (unsigned n:{5u,17u,128u}) for (double offset:{0.0,1.0,16.0}) {
        auto model=metalrobo::makeStraightSutureRod(n,rest,mat);
        for (unsigned e=0;e+1<n;++e) model.stretchStiffness[e]*=1.0+.07*(e%4);
        for (unsigned i=0;i<n;++i) model.nodeMasses[i]*=1.0+.11*(i%3);
        for (unsigned e=0;e+1<n;++e) {
            model.restLengths[e]=static_cast<float>(model.restLengths[e]);
            model.stretchStiffness[e]=static_cast<float>(model.stretchStiffness[e]);
        }
        for (unsigned i=0;i<n;++i)
            model.nodeMasses[i]=1.0/static_cast<double>(static_cast<float>(1.0/model.nodeMasses[i]));
        auto initial=metalrobo::makeDiscreteElasticRodDefaultState(model);
        for (unsigned i=0;i<n;++i) {
            initial.positions[i][0]=offset+.122*double(i)/(n-1)+
                (i==0||i+1==n?0.0:.00008*std::sin(double(i)));
            initial.velocities[i][0]=i==0||i+1==n?0.0:.003*std::cos(double(i));
        }
        for (unsigned i=0;i<n;++i) {
            initial.positions[i][0]=static_cast<float>(initial.positions[i][0]);
            initial.velocities[i][0]=static_cast<float>(initial.velocities[i][0]);
        }
        const double g=static_cast<float>(-1.2);
        const double left=static_cast<float>(offset+.00001), right=static_cast<float>(offset+.12202);
        const double vl=static_cast<float>(.01), vr=static_cast<float>(.02);
        std::array<metalrobo::DiscreteRodAttachment,2> attachments{};
        attachments[0].nodeIndex=0; attachments[0].targetPosition={left,0,0};
        attachments[0].targetVelocity={vl,0,0};
        attachments[1].nodeIndex=n-1; attachments[1].targetPosition={right,0,0};
        attachments[1].targetVelocity={vr,0,0};
        std::array<metalrobo::DiscreteElasticRodState,1> in{initial};
        metalrobo::MetalDiscreteElasticRodInput input;
        input.states=in; input.attachmentCount=2; input.attachments=attachments;
        metalrobo::MetalDiscreteElasticRodConfig cfg;
        cfg.metallibPath=argv[1];cfg.step.timestep=h;cfg.step.gravity={g,0,0};
        cfg.step.linearDamping=0;cfg.step.twistDamping=0;
        cfg.step.solverIterations=128;cfg.step.constraintTolerance=1e-7;
        std::vector<std::vector<double>> A(n-2,std::vector<double>(n-2,0));
        std::vector<double> B(n-2), ref(n);
        ref[0]=left;ref[n-1]=right;
        for (unsigned i=1;i+1<n;++i) {
            double kl=model.stretchStiffness[i-1]/model.restLengths[i-1];
            double kr=model.stretchStiffness[i]/model.restLengths[i];
            double mass=model.nodeMasses[i]/(h*h);
            A[i-1][i-1]=mass+kl+kr;
            B[i-1]=mass*(initial.positions[i][0]+h*(initial.velocities[i][0]+h*g))
                +kl*model.restLengths[i-1]-kr*model.restLengths[i];
            if (i>1) A[i-1][i-2]=-kl; else B[i-1]+=kl*left;
            if (i+2<n) A[i-1][i]=-kr; else B[i-1]+=kr*right;
        }
        const auto inner=solve(A,B);
        if (inner.size()!=n-2) return 2;
        for (unsigned i=1;i+1<n;++i) ref[i]=inner[i-1];
        bool valid=true;
        for (unsigned i=0;i+1<n;++i) valid &= ref[i+1]>ref[i];
        const double kl=model.stretchStiffness[0]/model.restLengths[0];
        const double kr=model.stretchStiffness[n-2]/model.restLengths[n-2];
        const double fl=kl*(ref[1]-left-model.restLengths[0])+
            model.nodeMasses[0]*(initial.velocities[0][0]+h*g-vl)/h;
        const double fr=-kr*(right-ref[n-2]-model.restLengths[n-2])+
            model.nodeMasses[n-1]*(initial.velocities[n-1][0]+h*g-vr)/h;
        metalrobo::MetalDiscreteElasticRodResult first,second;
        const auto d1=metalrobo::runMetalDiscreteElasticRod(model,input,first,cfg);
        const auto d2=metalrobo::runMetalDiscreteElasticRod(model,input,second,cfg);
        const bool accepted=d1.succeeded()&&d1.published&&d2.succeeded()&&d2.published&&
            first.states.size()==1&&second.states.size()==1&&first.reactions.size()==2&&second.reactions.size()==2;
        double pe=0,ve=0,fe=0; bool replay=accepted;
        if (accepted) {
            for (unsigned i=0;i<n;++i) {
                const double vref=i==0?vl:(i+1==n?vr:(ref[i]-initial.positions[i][0])/h);
                pe=std::max(pe,std::abs(first.states[0].positions[i][0]-ref[i]));
                ve=std::max(ve,std::abs(first.states[0].velocities[i][0]-vref));
                replay &= first.states[0].positions[i]==second.states[0].positions[i] &&
                          first.states[0].velocities[i]==second.states[0].velocities[i];
            }
            fe=std::max(std::abs(first.reactions[0].averageForceOnTarget[0]-fl),
                        std::abs(first.reactions[1].averageForceOnTarget[0]-fr));
            replay &= first.reactions[0].averageForceOnTarget==second.reactions[0].averageForceOnTarget &&
                      first.reactions[1].averageForceOnTarget==second.reactions[1].averageForceOnTarget;
        }
        const bool pass=valid&&accepted&&pe<=1e-6&&ve<=.001&&
            std::abs(first.reactions[0].averageForceOnTarget[0]-fl)<=std::max(1e-6,.01*std::abs(fl))&&
            std::abs(first.reactions[1].averageForceOnTarget[0]-fr)<=std::max(1e-6,.01*std::abs(fr))&&replay;
        all &= pass;
        std::cout<<"nodes="<<n<<" offset_m="<<offset<<" reference_valid="<<valid<<" accepted="<<accepted
                 <<" host="<<metalrobo::metalDiscreteElasticRodHostStatusName(d1.status)
                 <<" gpu="<<d1.firstGPUStatusCode<<" max_position_error_m="<<pe
                 <<" max_velocity_error_m_s="<<ve<<" max_force_error_N="<<fe
                 <<" left_ref_N="<<fl<<" right_ref_N="<<fr<<" replay="<<replay<<" passed="<<pass<<"\n";
    }
    auto model=metalrobo::makeStraightSutureRod(5,rest,mat);
    auto state=metalrobo::makeDiscreteElasticRodDefaultState(model);
    auto before=state;
    std::array<metalrobo::DiscreteElasticRodState,1> in{state};
    std::array<metalrobo::DiscreteRodAttachment,2> at{};
    at[0].nodeIndex=0; at[0].targetPosition={.13,0,0};
    at[1].nodeIndex=4; at[1].targetPosition={0,0,0};
    metalrobo::MetalDiscreteElasticRodInput input;
    input.states=in;input.attachmentCount=2;input.attachments=at;
    metalrobo::MetalDiscreteElasticRodConfig cfg;
    cfg.metallibPath=argv[1];cfg.step.timestep=h;cfg.step.gravity={0,0,0};
    cfg.step.linearDamping=0;cfg.step.twistDamping=0;cfg.step.solverIterations=128;
    cfg.step.constraintTolerance=1e-7;
    metalrobo::MetalDiscreteElasticRodResult out;
    auto d=metalrobo::runMetalDiscreteElasticRod(model,input,out,cfg);
    const bool rollback=!d.succeeded()&&!d.published&&in[0].positions==before.positions&&
        in[0].velocities==before.velocities&&in[0].twists==before.twists&&in[0].twistRates==before.twistRates;
    all &= rollback;
    std::cout<<"reversed_targets_rejected="<<rollback<<" host="
             <<metalrobo::metalDiscreteElasticRodHostStatusName(d.status)
             <<" gpu="<<d.firstGPUStatusCode<<"\n";
    return all?0:1;
}
