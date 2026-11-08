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
    if(argc!=2) return 2;
    constexpr double rest=.11758755785232791, ea=2.3818718058345718, radius=.000020, h=.001, g=-1.2;
    metalrobo::DiscreteRodMaterial mat;mat.radius=radius;mat.density=1100;
    mat.youngModulus=ea/(std::numbers::pi*radius*radius);
    bool all=true;
    std::cout<<std::setprecision(17);
    for(unsigned n:{3u,7u,17u}) {
        auto model=metalrobo::makeStraightSutureRod(n,rest,mat);
        for(unsigned e=0;e+1<n;++e) model.stretchStiffness[e]*=1.0+.07*(e%4);
        for(unsigned i=0;i<n;++i) model.nodeMasses[i]*=1.0+.11*(i%3);
        auto initial=metalrobo::makeDiscreteElasticRodDefaultState(model);
        for(unsigned i=0;i<n;++i) {
            initial.positions[i][0]=.122*double(i)/(n-1)+
                (i==0||i+1==n?0.0:.00008*std::sin(double(i)));
            initial.velocities[i][0]=.002*std::cos(double(i));
        }
        std::vector<unsigned> pin=n==3?std::vector<unsigned>{0,1,2}:
            (n==7?std::vector<unsigned>{0,3,6}:std::vector<unsigned>{0,1,5,12,16});
        std::vector<bool> pinned(n,false);
        std::vector<metalrobo::DiscreteRodAttachment> at(pin.size());
        for(unsigned j=0;j<pin.size();++j) {
            const unsigned i=pin[j];pinned[i]=true;
            const double shift=.00001*std::cos(.4*double(i));
            at[j].nodeIndex=i;
            at[j].targetPosition={initial.positions[i][0]+shift,0,0};
            at[j].targetVelocity={shift/h,0,0};
        }
        std::vector<std::vector<double>> A(n,std::vector<double>(n,0));
        std::vector<double> B(n);
        for(unsigned i=0;i<n;++i) {
            if(pinned[i]) {
                for(unsigned j=0;j<pin.size();++j) if(pin[j]==i) B[i]=at[j].targetPosition[0];
                A[i][i]=1.0;continue;
            }
            const double kl=model.stretchStiffness[i-1]/model.restLengths[i-1];
            const double kr=model.stretchStiffness[i]/model.restLengths[i];
            const double m=model.nodeMasses[i]/(h*h);
            A[i][i]=m+kl+kr;A[i][i-1]=-kl;A[i][i+1]=-kr;
            B[i]=m*(initial.positions[i][0]+h*(initial.velocities[i][0]+h*g))+
                kl*model.restLengths[i-1]-kr*model.restLengths[i];
        }
        auto ref=solve(A,B);
        if(ref.size()!=n) return 2;
        bool valid=true;for(unsigned i=0;i+1<n;++i) valid &= ref[i+1]>ref[i];
        std::array<metalrobo::DiscreteElasticRodState,1> in{initial};
        metalrobo::MetalDiscreteElasticRodInput input;
        input.states=in;input.attachmentCount=at.size();input.attachments=at;
        metalrobo::MetalDiscreteElasticRodConfig cfg;
        cfg.metallibPath=argv[1];cfg.step.timestep=h;cfg.step.gravity={g,0,0};
        cfg.step.linearDamping=0;cfg.step.twistDamping=0;
        cfg.step.solverIterations=128;cfg.step.constraintTolerance=1e-7;
        metalrobo::MetalDiscreteElasticRodResult first,second;
        const auto d1=metalrobo::runMetalDiscreteElasticRod(model,input,first,cfg);
        const auto d2=metalrobo::runMetalDiscreteElasticRod(model,input,second,cfg);
        const bool accepted=d1.succeeded()&&d1.published&&d2.succeeded()&&d2.published&&
            first.states.size()==1&&second.states.size()==1&&first.reactions.size()==at.size()&&
            second.reactions.size()==at.size();
        double pe=0,ve=0,fe=0;bool replay=accepted,forcePass=accepted;
        if(accepted) {
            for(unsigned i=0;i<n;++i) {
                double v=(ref[i]-initial.positions[i][0])/h;
                for(unsigned j=0;j<pin.size();++j) if(pin[j]==i) v=at[j].targetVelocity[0];
                pe=std::max(pe,std::abs(first.states[0].positions[i][0]-ref[i]));
                ve=std::max(ve,std::abs(first.states[0].velocities[i][0]-v));
                replay &= first.states[0].positions[i]==second.states[0].positions[i]&&
                          first.states[0].velocities[i]==second.states[0].velocities[i];
            }
            for(unsigned j=0;j<pin.size();++j) {
                const unsigned i=pin[j];double f=0;
                if(i>0) f-=(model.stretchStiffness[i-1]/model.restLengths[i-1])*
                    (ref[i]-ref[i-1]-model.restLengths[i-1]);
                if(i+1<n) f+=(model.stretchStiffness[i]/model.restLengths[i])*
                    (ref[i+1]-ref[i]-model.restLengths[i]);
                f+=model.nodeMasses[i]*(initial.velocities[i][0]+h*g-at[j].targetVelocity[0])/h;
                const double error=std::abs(first.reactions[j].averageForceOnTarget[0]-f);
                fe=std::max(fe,error);
                forcePass &= error<=std::max(1e-6,.01*std::abs(f));
                replay &= first.reactions[j].averageForceOnTarget==second.reactions[j].averageForceOnTarget;
            }
        }
        const bool pass=valid&&accepted&&pe<=1e-6&&ve<=.001&&forcePass&&replay;
        all &= pass;
        std::cout<<"nodes="<<n<<" anchors="<<pin.size()<<" reference_valid="<<valid
                 <<" accepted="<<accepted<<" gpu="<<d1.firstGPUStatusCode
                 <<" max_position_error_m="<<pe<<" max_velocity_error_m_s="<<ve
                 <<" max_force_error_N="<<fe<<" force_pass="<<forcePass<<" replay="<<replay
                 <<" passed="<<pass<<"\n";
    }
    return all?0:1;
}
