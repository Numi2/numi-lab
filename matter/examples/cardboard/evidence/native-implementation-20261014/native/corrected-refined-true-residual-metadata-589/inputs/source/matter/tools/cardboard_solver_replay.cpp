#include "cardboard_solver_reference.hpp"
#include <cctype>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace {
namespace paper=numi_cardboard_paper;
namespace oracle=numi_cardboard_solver;
template<class T> void array(const T& values,double scale=1) {
    std::cout<<'[';
    bool first=true;
    for(double v:values) { if(!first)std::cout<<','; first=false; std::cout<<v*scale; }
    std::cout<<']';
}
double difference(const paper::Matrix& a,const paper::Matrix& b) {
    double error=0,scale=1e-10;
    for(unsigned i=0;i<9;++i) {error=std::max(error,std::abs(a[i]-b[i]));scale=std::max(scale,std::abs(b[i]));}
    return error/scale;
}
void checks() {
    unsigned count=0;
    double worst=0;
    const paper::Matrix h{.001,.0003,-.0002,-.0001,-.0007,.0004,.0002,-.0003,.0005};
    for(auto kind:{paper::MaterialKind::liner,paper::MaterialKind::medium}) {
        paper::State old{};
        for(unsigned trial=0;trial<6;++trial) {
            auto f=paper::identity();
            if(trial==0) f[0]=.9998;
            if(trial==1) f[0]=.994;
            if(trial==2) f[1]=.004;
            if(trial==3) f[2]=.001;
            if(trial==4) {f[0]=.992;f[5]=.0006;}
            if(trial==5) {f[0]=.997;f[1]=.005;}
            const auto center=oracle::project(kind,f,old);
            const auto tangent=oracle::directionalTangent(kind,f,h,old,center.state);
            paper::Matrix plus=f,minus=f;
            constexpr double epsilon=1e-4;
            for(unsigned i=0;i<9;++i) {plus[i]+=epsilon*h[i];minus[i]-=epsilon*h[i];}
            const auto pp=oracle::project(kind,plus,old),pm=oracle::project(kind,minus,old);
            paper::Matrix fd{};
            for(unsigned i=0;i<9;++i) fd[i]=(pp.firstPiola[i]-pm.firstPiola[i])/(2*epsilon);
            const double error=difference(tangent,fd);
            worst=std::max(worst,error);
            if(error>1e-6) throw std::runtime_error("FP64 independent derivative gate failed: "+std::to_string(error));
            // An independent finite-Jacobian legacy oracle verifies the analytic
            // projection too; retain a history-bearing case on the final leg.
            const auto legacy=paper::project(kind,f,old,64,1e-11);
            if(difference(center.firstPiola,legacy.firstPiola)>1e-7)
                throw std::runtime_error("analytic projection disagrees with independent FD-Jacobian oracle");
            if(trial==4) old=center.state;
            ++count;
        }
    }
    std::cout<<"{\"schema\":\"numi.cardboard.solver-reference-check.v1\",\"cases\":"<<count
             <<",\"maximum_directional_derivative_relative_error\":"<<worst
             <<",\"gate\":1e-6,\"passed\":true,\"physical_validation\":false}\n";
}
}

int main(int argc,char** argv) {
    std::cout<<std::setprecision(17);
    try {
        if(argc==2 && std::string(argv[1])=="--check") {checks();return 0;}
        std::array<paper::Parameters,2> captured{paper::liner,paper::medium};
        const bool capturedFP32=argc==3 && std::string(argv[1])=="--parameters";
        if(argc==3 && std::string(argv[1])=="--parameters") {
            std::ifstream parameters(argv[2]);
            if(!parameters) throw std::runtime_error("cannot read captured parameters");
            std::string name;
            unsigned count=0;
            std::array<bool,2> seen{};
            while(parameters>>name) {
                if(name!="liner" && name!="medium") throw std::runtime_error("unknown parameter material");
                unsigned index=name=="liner"?0:1;
                if(seen[index]) throw std::runtime_error("duplicate parameter material");
                std::array<double,20> values{};
                for(auto& v:values) {
                    if(!(parameters>>v)||!std::isfinite(v)) throw std::runtime_error("invalid captured parameters");
                    v=static_cast<float>(v);
                    if(!std::isfinite(v)) throw std::runtime_error("captured parameter outside FP32 range");
                }
                auto& p=captured[index];
                p.density=values[0];
                for(unsigned i=0;i<6;++i) p.c[i]=values[1+i]*1e-6;
                for(unsigned i=0;i<3;++i) p.shear[i]=values[7+i]*1e-6;
                p.sigmaReference=values[10]*1e-6;
                for(unsigned i=0;i<6;++i) p.hill[i]=values[11+i];
                p.qFloor=values[17];p.fischerBurmeisterEpsilon=values[18];
                if(!(p.sigmaReference>0 && p.qFloor>0 && p.fischerBurmeisterEpsilon>0 && values[19]>0))
                    throw std::runtime_error("nonpositive captured constitutive scale");
                seen[index]=true;++count;
            }
            if(count!=2) throw std::runtime_error("both captured paper parameter sets required");
        } else if(argc!=1) throw std::runtime_error("usage: cardboard-solver-replay [--check|--parameters FILE]; stdin rows: key liner|medium F[9] H[9] accepted[13] metal_projected[13]");
        std::string line;
        unsigned rows=0,failures=0;
        while(std::getline(std::cin,line)) {
            if(line.empty() || line[0]=='#') continue;
            std::istringstream in(line);
            std::string key,kindName;
            in>>key>>kindName;
            if(key.empty() || key.size()>128 || !std::all_of(key.begin(),key.end(),[](unsigned char c){return std::isalnum(c)||c=='_'||c=='-'||c=='.';}))
                throw std::runtime_error("invalid replay row key");
            std::cout<<"{\"schema\":\"numi.cardboard.solver-material-replay.v1\",\"key\":\""<<key<<"\"";
            try {
                if(kindName!="liner" && kindName!="medium") throw std::runtime_error("unsupported material");
                const auto kind=kindName=="liner"?paper::MaterialKind::liner:paper::MaterialKind::medium;
                const auto& parameters=captured[kind==paper::MaterialKind::liner?0:1];
                paper::Matrix f{},h{};
                paper::State old{},metal{};
                const auto read=[&](auto& values){for(auto& v:values) {
                    if(!(in>>v)||!std::isfinite(v))throw std::runtime_error("invalid numeric input");
                    if(capturedFP32)v=static_cast<float>(v);
                    if(!std::isfinite(v))throw std::runtime_error("captured input outside FP32 range");
                }};
                read(f);read(h);read(old);read(metal);
                std::string extra;if(in>>extra)throw std::runtime_error("extra input columns");
                const auto p=oracle::project(parameters,f,old);
                const auto tangent=oracle::directionalTangent(parameters,f,h,old,p.state);
                std::cout<<",\"valid\":true,\"input_contract\":\""
                    <<(capturedFP32?"exact_widened_fp32":"fp64")<<"\",\"oracle_P_Pa\":";array(p.firstPiola,1e6);
                std::cout<<",\"oracle_tangent_Pa\":";array(tangent,1e6);
                std::cout<<",\"oracle_state\":";array(p.state);
                std::cout<<",\"oracle_residual\":"<<p.residual<<",\"iterations\":"<<p.iterations
                         <<",\"q_dimensionless\":"<<p.equivalentStress
                         <<",\"elastic_trial_q_dimensionless\":"
                         <<paper::hillEquivalent(parameters,
                             paper::normalizedSecondPiola(parameters,f,
                                 paper::plasticStrain(old)))
                         <<",\"lambda_increment\":"<<p.state[12]-old[12]
                         <<",\"independent_residual_at_metal_state\":"
                         <<paper::residualNorm(paper::residual(parameters,f,old,metal));
            } catch(const std::exception& e) {
                // Messages below are fixed code strings without untrusted quotes.
                ++failures;std::cout<<",\"valid\":false,\"reason\":\""<<e.what()<<"\"";
            }
            std::cout<<"}\n";++rows;
        }
        if(rows==0) throw std::runtime_error("no replay rows");
        return failures?2:0;
    } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
