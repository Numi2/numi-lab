#include "metalrobo/numi_human_bilateral.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {
unsigned checks=0;
void require(bool value, const char* message) {
    ++checks;
    if (!value) {std::cerr<<message<<'\n';std::exit(1);}
}
void checkSystem(const std::vector<float>& matrix, const std::vector<float>& expected) {
    const auto n=static_cast<unsigned>(expected.size());
    std::vector<float> factor=matrix,scale(n),pivots(n),rhs(n);
    for (unsigned i=0;i<n;++i) {
        double value=0;
        for (unsigned j=0;j<n;++j) value+=double(matrix[i*n+j])*expected[j];
        rhs[i]=static_cast<float>(value);
    }
    const auto original=rhs;
    require(mrNumiHumanBilateralFactor(factor.data(),scale.data(),pivots.data(),n),"factor rejected finite independent block");
    require(mrNumiHumanBilateralSolve(factor.data(),scale.data(),pivots.data(),rhs.data(),n),"solve rejected finite block");
    for (unsigned i=0;i<n;++i) {
        double sum=0, norm=std::abs(double(original[i]));
        for (unsigned j=0;j<n;++j) {sum+=double(matrix[i*n+j])*rhs[j];norm+=std::abs(double(matrix[i*n+j])*rhs[j]);}
        require(std::abs(sum-original[i])<=3e-6*std::max(norm,1e-20),"bilateral original-unit backward error");
        require(std::abs(rhs[i]-expected[i])<=3e-4*(1+std::abs(expected[i])),"bilateral multiplier differs from manufactured exact solution");
    }
    // Factors are reusable and do not retain a previous RHS or impulse.
    std::fill(rhs.begin(),rhs.end(),0.0f);
    require(mrNumiHumanBilateralSolve(factor.data(),scale.data(),pivots.data(),rhs.data(),n),"zero RHS solve failed");
    require(std::all_of(rhs.begin(),rhs.end(),[](float v){return v==0;}),"zero residual produced impulse");
}
}
int main() {
    checkSystem({2.0f},{-3.0f});
    // Two successive pivot swaps, including previously computed L columns.
    checkSystem({1,9,1, 2,1,8, 5,2,1},{1,-2,3});
    checkSystem({1,0.999f,0.999001f,1},{0.4f,-0.3f});
    std::mt19937 generator(718);
    std::uniform_real_distribution<float> random(-1,1);
    for(unsigned n: {2u,3u,7u,51u,128u,160u}) {
        for(unsigned trial=0;trial<6;++trial) {
            std::vector<float> raw(n*n),matrix(n*n),x(n),scale(n);
            for(auto& v:raw)v=random(generator);
            for(unsigned i=0;i<n;++i) {x[i]=random(generator);scale[i]=std::pow(10.0f,float(int(i%7)-3));}
            for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j) {
                double v=i==j ? double(n) : 0.0;
                for(unsigned k=0;k<n;++k)v+=double(raw[i*n+k])*raw[j*n+k];
                matrix[i*n+j]=static_cast<float>(v*scale[i]*scale[j]);
            }
            // Manufactured x scaled to keep contributions observable.
            for(unsigned i=0;i<n;++i)x[i]/=scale[i];
            checkSystem(matrix,x);
        }
    }
    std::vector<float> s(2),p(2),rhs{1,2};
    std::vector<float> singular{1,1,1,1};
    require(!mrNumiHumanBilateralFactor(singular.data(),s.data(),p.data(),2),"singular rows silently regularized");
    for(float bad: {0.0f,-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        std::vector<float> a{bad};
        require(!mrNumiHumanBilateralFactor(a.data(),s.data(),p.data(),1),"invalid diagonal accepted");
    }
    require(!mrNumiHumanBilateralFactor(nullptr,nullptr,nullptr,0),"zero dimension accepted");
    require(!mrNumiHumanBilateralFactor(nullptr,nullptr,nullptr,161),"oversized dimension accepted");
    std::vector<float> a{1,0,0,1};
    require(mrNumiHumanBilateralFactor(a.data(),s.data(),p.data(),2),"identity factor failed");
    p[0]=0.5f;
    require(!mrNumiHumanBilateralSolve(a.data(),s.data(),p.data(),rhs.data(),2),"nonintegral pivot accepted");
    std::cout<<"Human bilateral block: "<<checks<<" checks passed\n";
}
