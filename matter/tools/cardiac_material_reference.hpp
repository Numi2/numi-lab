#pragma once

// Independent FP64 tensor derivation of Rodero et al., S4 equations 3-8,
// DOI 10.1371/journal.pcbi.1008851. No native bytecode/AD/Metal evaluator is
// reused. Static constitutive oracle only: no physical time integration.
#include <array>
#include <cmath>
#include <initializer_list>
#include <stdexcept>

namespace numi_cardiac_reference {
using Matrix = std::array<double, 9>; // row-major
inline constexpr Matrix identity{1,0,0,0,1,0,0,0,1};
inline Matrix add(const Matrix& a, const Matrix& b) {
    Matrix c{}; for(unsigned i=0;i<9;++i)c[i]=a[i]+b[i]; return c;
}
inline Matrix scale(const Matrix& a, double s) {
    Matrix c{}; for(unsigned i=0;i<9;++i)c[i]=a[i]*s; return c;
}
inline Matrix sub(const Matrix& a, const Matrix& b) { return add(a,scale(b,-1)); }
inline Matrix transpose(const Matrix& a) {
    Matrix c{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)c[3*i+j]=a[3*j+i];return c;
}
inline Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix c{};
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)
        for(unsigned k=0;k<3;++k)c[3*i+j]+=a[3*i+k]*b[3*k+j];
    return c;
}
inline double contract(const Matrix& a, const Matrix& b) {
    double c=0;for(unsigned i=0;i<9;++i)c+=a[i]*b[i];return c;
}
inline double determinant(const Matrix& a) {
    return a[0]*(a[4]*a[8]-a[5]*a[7])-a[1]*(a[3]*a[8]-a[5]*a[6])
        +a[2]*(a[3]*a[7]-a[4]*a[6]);
}
inline Matrix inverseTranspose(const Matrix& a) {
    return scale(Matrix{a[4]*a[8]-a[5]*a[7],a[5]*a[6]-a[3]*a[8],a[3]*a[7]-a[4]*a[6],
        a[2]*a[7]-a[1]*a[8],a[0]*a[8]-a[2]*a[6],a[1]*a[6]-a[0]*a[7],
        a[1]*a[5]-a[2]*a[4],a[2]*a[3]-a[0]*a[5],a[0]*a[4]-a[1]*a[3]},1/determinant(a));
}
inline bool finite(const Matrix& a) {
    for(double v:a)if(!std::isfinite(v))return false;return true;
}
struct Result { double energy=0; Matrix stress{}, tangent{}; };
struct GuccioneParameters { double a=1700, bf=8, bt=3, bfs=4, kappa=1e6; };
inline void requireInputs(const Matrix& f,const Matrix& h) {
    if(!finite(f)||!finite(h)||!(determinant(f)>0))
        throw std::invalid_argument("cardiac source oracle requires finite F/H and J>0");
}
inline void requireResult(const Result& r) {
    if(!std::isfinite(r.energy)||!finite(r.stress)||!finite(r.tangent))
        throw std::overflow_error("cardiac source oracle result is nonfinite");
}
inline Matrix strainMetric(const Matrix& e,const GuccioneParameters& p) {
    return {p.bf*e[0],p.bfs*e[1],p.bfs*e[2],p.bfs*e[3],p.bt*e[4],
        p.bt*e[5],p.bfs*e[6],p.bt*e[7],p.bt*e[8]};
}
inline Result guccione(const Matrix& f,const Matrix& h,
                       const GuccioneParameters& p={}) {
    // B=M:Ebar is the symmetric source strain metric. Q=Ebar:B.
    // Piso=a exp(Q) alpha [F B - (B:C)/3 F^-T], alpha=J^-2/3.
    // Differentiate that tensor expression directly along H, including
    // dB, d(alpha), dQ and d(F^-T); no finite-difference tangent is used.
    requireInputs(f,h);
    for(double x:{p.a,p.bf,p.bt,p.bfs,p.kappa})if(!(x>0)||!std::isfinite(x))
        throw std::invalid_argument("invalid Guccione source parameter");
    const double j=determinant(f),logj=std::log(j),alpha=std::pow(j,-2.0/3.0);
    const Matrix it=inverseTranspose(f),c=multiply(transpose(f),f);
    const double ell=contract(it,h); // d(log J)
    const Matrix dc=add(multiply(transpose(h),f),multiply(transpose(f),h));
    const Matrix dit=scale(multiply(multiply(it,transpose(h)),it),-1);
    const Matrix e=scale(sub(scale(c,alpha),identity),.5);
    const Matrix de=scale(sub(dc,scale(c,(2.0/3.0)*ell)),.5*alpha);
    const Matrix b=strainMetric(e,p),db=strainMetric(de,p);
    const double q=contract(e,b),dq=2*contract(b,de),s=contract(b,c);
    const double ds=contract(db,c)+contract(b,dc);
    const double prefactor=p.a*std::exp(q)*alpha;
    const Matrix u=sub(multiply(f,b),scale(it,s/3));
    const Matrix du=sub(add(multiply(h,b),multiply(f,db)),
                        add(scale(it,ds/3),scale(dit,s/3)));
    Result r;
    r.energy=.5*p.kappa*logj*logj+.5*p.a*std::expm1(q);
    r.stress=add(scale(u,prefactor),scale(it,p.kappa*logj));
    r.tangent=add(add(scale(u,prefactor*(dq-(2.0/3.0)*ell)),scale(du,prefactor)),
                  scale(add(scale(it,ell),scale(dit,logj)),p.kappa));
    requireResult(r);return r;
}
inline Result neoHookean(const Matrix& f,const Matrix& h,double c=7450,double kappa=1e6) {
    requireInputs(f,h);
    if(!(c>0)||!std::isfinite(c)||!(kappa>0)||!std::isfinite(kappa))
        throw std::invalid_argument("invalid neo-Hookean source parameter");
    const double j=determinant(f),alpha=std::pow(j,-2.0/3.0),i1=contract(f,f);
    const Matrix it=inverseTranspose(f);
    const double ell=contract(it,h),di1=2*contract(f,h);
    const Matrix dit=scale(multiply(multiply(it,transpose(h)),it),-1);
    const Matrix u=sub(f,scale(it,i1/3));
    const Matrix du=sub(h,add(scale(it,di1/3),scale(dit,i1/3)));
    Result r;
    r.energy=.5*kappa*(j-1)*(j-1)+.5*c*(alpha*i1-3);
    r.stress=add(scale(u,c*alpha),scale(it,kappa*(j-1)*j));
    r.tangent=add(scale(sub(du,scale(u,(2.0/3.0)*ell)),c*alpha),
        scale(add(scale(it,(2*j-1)*j*ell),scale(dit,(j-1)*j)),kappa));
    requireResult(r);return r;
}
inline void requireFrame(const Matrix& q) {
    if(!finite(q)||std::abs(determinant(q)-1)>1e-10)
        throw std::invalid_argument("oracle frame must be a proper orthonormal rotation");
    const Matrix d=sub(multiply(transpose(q),q),identity);
    for(double x:d)if(std::abs(x)>1e-10)
        throw std::invalid_argument("oracle frame is not orthonormal");
}
inline Result framedGuccione(const Matrix& f,const Matrix& h,const Matrix& q,
                             const GuccioneParameters& p={}) {
    requireFrame(q);
    Result r=guccione(multiply(f,q),multiply(h,q),p);
    r.stress=multiply(r.stress,transpose(q));
    r.tangent=multiply(r.tangent,transpose(q));return r;
}
} // namespace numi_cardiac_reference
