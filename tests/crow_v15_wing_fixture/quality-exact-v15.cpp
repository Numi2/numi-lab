#include "metalrobo/MetalUnifiedQualitySolver.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("expected input file");
    constexpr std::size_t n=73, m=3, cases=4, floatsPer=5624;
    std::vector<float> input(cases*floatsPer);
    std::ifstream file(argv[1], std::ios::binary);
    file.read(reinterpret_cast<char*>(input.data()), input.size()*sizeof(float));
    if (!file || file.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("invalid input");
    std::array<MRUnifiedQualityBlockGPU,m> blocks{};
    for (std::size_t i=0;i<m;++i) {
      auto& block=blocks[i];
      block.layout={static_cast<mr_u32>(i),1u,MR_UNIFIED_QUALITY_SCALAR_INTERVAL,MR_UNIFIED_QUALITY_BLOCK_HARD_EQUALITY};
      block.stableKey={static_cast<mr_u32>(100+i),0u,0u,0u};
      block.scale0=block.scale1={1,1,1,1};
      block.regularization0=block.regularization1={1e-5f,1,1,1};
      block.boundsAndShift={-std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),0,0};
    }
    std::vector<float> A(cases*n*n), J(cases*m*n), bias(cases*m), freev(cases*n), warm(cases*n,0), impulses(cases*m,0);
    for (std::size_t c=0;c<cases;++c) {
      const float* p=input.data()+c*floatsPer;
      std::copy(p,p+n*n,A.begin()+c*n*n);
      std::copy(p+n*n+n,p+n*n+n+m*n,J.begin()+c*m*n);
      for (std::size_t i=0;i<m;++i) bias[c*m+i]=-p[n*n+n+m*n+i];
      std::vector<double> L(n*n), y(n), x(n);
      for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<=i;++j) {
        double sum=A[c*n*n+i*n+j];
        for (std::size_t k=0;k<j;++k) sum-=L[i*n+k]*L[j*n+k];
        if (i==j) {if (!(sum>0)) throw std::runtime_error("reference mass not SPD"); L[i*n+j]=std::sqrt(sum);}
        else L[i*n+j]=sum/L[j*n+j];
      }
      for (std::size_t i=0;i<n;++i) { double v=p[n*n+i]; for (std::size_t j=0;j<i;++j) v-=L[i*n+j]*y[j]; y[i]=v/L[i*n+i]; }
      for (std::size_t i=n;i-->0;) { double v=y[i]; for (std::size_t j=i+1;j<n;++j) v-=L[j*n+i]*x[j]; x[i]=v/L[i*n+i]; }
      for (std::size_t i=0;i<n;++i) freev[c*n+i]=static_cast<float>(x[i]);
    }
    metalrobo::MetalUnifiedQualityProblem problem;
    problem.problemCount=cases; problem.generalizedVelocityCount=n; problem.rowCount=m;
    problem.blocks=blocks; problem.dynamics=A; problem.jacobian=J; problem.bias=bias;
    problem.freeVelocity=freev; problem.warmVelocity=warm; problem.warmImpulses=impulses;
    metalrobo::MetalUnifiedQualityConfig config;
    config.maximumNewtonIterations=40; config.feasibilityTolerance=1e-9f;
    metalrobo::MetalUnifiedQualityResult result;
    const auto diagnostic=metalrobo::solveMetalUnifiedQuality(problem,result,config);
    if (!diagnostic.succeeded()) {std::cerr<<diagnostic.message<<'\n';return 1;}
    std::cout<<std::setprecision(10)<<"{\"cases\":[";
    for (std::size_t c=0;c<cases;++c) {
      if(c) std::cout<<',';
      std::cout<<"{\"path\":"<<result.statuses[c].solvePath<<",\"velocity\":[";
      for (std::size_t i=0;i<n;++i) {if(i)std::cout<<',';std::cout<<result.velocity[c*n+i];}
      std::cout<<"],\"impulse\":[";
      for (std::size_t i=0;i<m;++i) {if(i)std::cout<<',';std::cout<<result.impulses[c*m+i];}
      std::cout<<"]}";
    }
    std::cout<<"]}\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<"exact_v15_probe: "<<e.what()<<'\n';return 2;}
}
