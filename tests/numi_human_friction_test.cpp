#include "metalrobo/numi_human_friction.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
std::size_t checks = 0;
void require(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
// Independent long-double eigenspace secular equation; no production helper.
std::array<long double, 2> oracle(long double a, long double b, long double d,
                                 long double rx, long double ry, long double r) {
    if (r == 0) return {0, 0};
    const long double theta = 0.5L * std::atan2(2*b, a-d);
    const long double c = std::cos(theta), s = std::sin(theta);
    const long double e0 = c*c*a+2*c*s*b+s*s*d;
    const long double e1 = s*s*a-2*c*s*b+c*c*d;
    const long double g0 = c*rx+s*ry, g1 = -s*rx+c*ry;
    auto solve = [&](long double shift) {
        const long double u = g0/(e0+shift), v = g1/(e1+shift);
        return std::array<long double,2>{c*u-s*v, s*u+c*v};
    };
    auto p = solve(0);
    if (std::hypot(p[0],p[1]) <= r) return p;
    long double low = 0, high = std::hypot(rx,ry)/r;
    for (int i=0;i<150;++i) {
        const long double mid = (low+high)/2;
        p = solve(mid);
        if (std::hypot(p[0],p[1]) > r) low = mid;
        else high = mid;
    }
    return solve(high);
}
void exercise(float a,float b,float d,float rx,float ry,float r) {
    auto result=mrNumiHumanSolveFrictionDisk(a,b,d,rx,ry,r);
    require(result.valid,"SPD friction solve rejected valid inputs");
    const auto expected=oracle(a,b,d,rx,ry,r);
    const long double norm=std::hypot(result.x,result.y);
    const long double referenceNorm=std::hypot(expected[0],expected[1]);
    const long double error=std::hypot(result.x-expected[0],result.y-expected[1]);
    if(error>3e-5L*std::max(1e-20L,referenceNorm)) {
        std::cerr<<"a="<<a<<" b="<<b<<" d="<<d<<" rhs="<<rx<<","<<ry
                 <<" radius="<<r<<" error="<<double(error)<<" reference="
                 <<double(expected[0])<<","<<double(expected[1])<<'\n';
    }
    require(error<=3e-5L*std::max(1e-20L,referenceNorm),"friction differs from spectral oracle");
    require(norm<=r*(1+3e-7L),"friction exceeds the Coulomb disk");
    if (r == 0.0f) { require(result.x == 0 && result.y == 0, "zero disk carries friction"); return; }
    const long double vx=a*result.x+b*result.y-rx;
    const long double vy=b*result.x+d*result.y-ry;
    const long double residualScale=1+std::hypot(rx,ry)+std::max(a,d)*norm;
    if(norm>0 && norm>r*(1-1e-5L)) {
        require(std::abs(result.x*vy-result.y*vx)<=2e-5L*norm*residualScale,
                "sliding impulse is not parallel to terminal tangential velocity");
        require(result.x*vx+result.y*vy<=2e-5L*norm*residualScale,
                "sliding friction does positive terminal work");
    } else {
        require(std::hypot(vx,vy)<=2e-5L*residualScale,"sticking velocity does not vanish");
    }
}
}
int main() {
    try {
        for(float scale:{1e-6f,1.0f,1e6f})
            for(float angle:{0.0f,0.3f,1.2f})
                for(float eigen:{0.01f,0.3f,1.0f,10.0f,100.0f})
                    for(float radius:{0.0f,1e-4f,0.1f,1.0f,1e4f})
                        for(auto rhs: {std::array<float,2>{-2,-1},{0,0},{1,-3},{0,2}}) {
                            float c=std::cos(angle),s=std::sin(angle);
                            exercise(scale*(c*c+eigen*s*s),scale*(1-eigen)*c*s,
                                scale*(s*s+eigen*c*c),scale*rhs[0],scale*rhs[1],radius);
                        }
        // A concrete counterexample to the old Euclidean clipping update.
        auto correct=mrNumiHumanSolveFrictionDisk(0.65f,0.0f,2.9f,-2.0f,-1.0f,1.0f);
        double x=-2.0/0.65,y=-1.0/2.9,n=std::hypot(x,y);x/=n;y/=n;
        double oldCross=x*(2.9*y+1)-y*(0.65*x+2);
        require(std::abs(oldCross)>0.4,"anisotropic regression no longer distinguishes radial clipping");
        require(std::hypot(correct.x-x,correct.y-y)>0.1,"anisotropic solution matches incorrect radial clipping");
        for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
            require(!mrNumiHumanSolveFrictionDisk(bad,0,1,1,1,1).valid,"nonfinite metric accepted");
            require(!mrNumiHumanSolveFrictionDisk(1,0,1,bad,1,1).valid,"nonfinite force accepted");
            require(!mrNumiHumanSolveFrictionDisk(1,0,1,1,1,bad).valid,"nonfinite radius accepted");
        }
        require(!mrNumiHumanSolveFrictionDisk(1,2,1,1,1,1).valid,"indefinite metric accepted");
        require(!mrNumiHumanSolveFrictionDisk(1,1,1,1,1,1).valid,"singular metric accepted");
        require(!mrNumiHumanSolveFrictionDisk(1,0,1,1,1,-1).valid,"negative radius accepted");
        std::cout<<"Human friction metric: "<<checks<<" checks passed\n";
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
