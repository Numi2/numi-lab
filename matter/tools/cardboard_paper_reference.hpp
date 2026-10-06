#pragma once

// Independent FP64 material-point oracle for the authored Haj-Ali ideal-Hill
// assets. This deliberately does not call Matter's generated constitutive
// bytecode, Metal kernels, or material solver. Axes are 11=MD, 22=CD, 33=ZD.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace numi_cardboard_paper {

using Vec3 = std::array<double, 3>;
using Matrix = std::array<double, 9>; // row-major
using Vector6 = std::array<double, 6>; // 11,22,33,23,13,12
using State = std::array<double, 13>;  // Ep[6], trial stress/sigma_ref[6], lambda

enum class MaterialKind { liner, medium };

struct Parameters {
    std::string_view name;
    double density;
    std::array<double, 6> c; // C11,C22,C33,C12,C13,C23, MPa
    std::array<double, 3> shear; // G12,G13,G23, MPa
    std::array<double, 3> compressiveYield; // MD,CD,ZD, MPa
    double tau12;
    double tau13;
    double tau23;
    double sigmaReference;
    std::array<double, 6> hill; // F,G,H,L,M,N
    double qFloor = 2.0e-6;
    double fischerBurmeisterEpsilon = 1.0e-8;
};

inline constexpr Parameters liner{
    "hajali2009_liner_hill_ideal", 766.0,
    {4843.6261790411845, 2118.009773355429, 19.100028262603168,
     381.24384915298583, 0.22861936478052688, 0.20899490085830671},
    {1222.0, 166.28, 137.28},
    {22.92, 11.02, 7.52},
    2.1, 0.024, 0.024, 22.92,
    {6.307660915935419, 2.981864887595591, -1.9818648875955915,
     456012.5, 456012.5, 59.56081632653061}};

inline constexpr Parameters medium{
    "hajali2009_medium_hill_ideal", 679.0,
    {4524.848543681839, 1631.0512147406462, 17.90002997076348,
     293.59114732344125, 0.21371636195674576, 0.19286701249506086},
    {1039.0, 226.34, 198.72},
    {9.391, 4.836, 3.23},
    2.1, 0.024, 0.024, 9.391,
    {5.612051993569077, 2.841099191623919, -1.8410991916239194,
     76554.58420138889, 76554.58420138889, 9.998966099773243}};

inline const Parameters& parameters(MaterialKind kind) {
    return kind == MaterialKind::liner ? liner : medium;
}

inline Matrix identity() { return {1,0,0, 0,1,0, 0,0,1}; }

inline Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix c{};
    for (unsigned i=0; i<3; ++i) for (unsigned j=0; j<3; ++j)
        for (unsigned k=0; k<3; ++k) c[3*i+j] += a[3*i+k]*b[3*k+j];
    return c;
}

inline Matrix transpose(const Matrix& a) {
    Matrix b{};
    for (unsigned i=0; i<3; ++i) for (unsigned j=0; j<3; ++j)
        b[3*i+j] = a[3*j+i];
    return b;
}

inline double hillEquivalent(const Parameters& p, const Vector6& normalizedStress) {
    const auto& h = p.hill;
    const double s1=normalizedStress[0], s2=normalizedStress[1], s3=normalizedStress[2];
    const double q2 = h[0]*(s2-s3)*(s2-s3) + h[1]*(s3-s1)*(s3-s1)
        + h[2]*(s1-s2)*(s1-s2) + 2*h[3]*normalizedStress[3]*normalizedStress[3]
        + 2*h[4]*normalizedStress[4]*normalizedStress[4]
        + 2*h[5]*normalizedStress[5]*normalizedStress[5]
        + p.qFloor*p.qFloor;
    if (q2 < -1.0e-12) throw std::runtime_error("Hill quadratic became negative");
    return std::sqrt(std::max(0.0, q2));
}

inline Vector6 hillFlow(const Parameters& p, const Vector6& s) {
    const auto& h=p.hill;
    const double q=hillEquivalent(p,s);
    return {
        (h[1]*(s[0]-s[2])+h[2]*(s[0]-s[1]))/q,
        (h[0]*(s[1]-s[2])+h[2]*(s[1]-s[0]))/q,
        (h[0]*(s[2]-s[1])+h[1]*(s[2]-s[0]))/q,
        h[3]*s[3]/q, h[4]*s[4]/q, h[5]*s[5]/q};
}

inline Matrix rightCauchyGreen(const Matrix& f) { return multiply(transpose(f), f); }

inline Vector6 normalizedSecondPiola(const Parameters& p, const Matrix& f,
                                     const Vector6& ep) {
    const Matrix c=rightCauchyGreen(f);
    const double e11=0.5*(c[0]-1.0)-ep[0];
    const double e22=0.5*(c[4]-1.0)-ep[1];
    const double e33=0.5*(c[8]-1.0)-ep[2];
    const double e23=0.5*c[5]-ep[3];
    const double e13=0.5*c[2]-ep[4];
    const double e12=0.5*c[1]-ep[5];
    const auto& a=p.c;
    return {
        (a[0]*e11+a[3]*e22+a[4]*e33)/p.sigmaReference,
        (a[3]*e11+a[1]*e22+a[5]*e33)/p.sigmaReference,
        (a[4]*e11+a[5]*e22+a[2]*e33)/p.sigmaReference,
        (2*p.shear[2]*e23)/p.sigmaReference,
        (2*p.shear[1]*e13)/p.sigmaReference,
        (2*p.shear[0]*e12)/p.sigmaReference};
}

inline Vector6 plasticStrain(const State& state) {
    Vector6 result{};
    for (unsigned i=0;i<6;++i) result[i]=state[i];
    return result;
}

inline Vector6 returnedNormalizedStress(const Parameters& p, const Matrix& f,
                                        const State& accepted,
                                        const State& candidate) {
    (void)accepted;
    return normalizedSecondPiola(p,f,plasticStrain(candidate));
}

inline Matrix secondPiola(const Parameters& p, const Matrix& f, const Vector6& ep) {
    const Vector6 normalized=normalizedSecondPiola(p,f,ep);
    const double s=p.sigmaReference;
    return {s*normalized[0],s*normalized[5],s*normalized[4],
            s*normalized[5],s*normalized[1],s*normalized[3],
            s*normalized[4],s*normalized[3],s*normalized[2]};
}

inline Matrix firstPiola(const Parameters& p, const Matrix& f, const Vector6& ep) {
    return multiply(f,secondPiola(p,f,ep));
}

inline double residualNorm(const std::array<double,13>& r) {
    double result=0.0;
    for (double value:r) result=std::max(result,std::abs(value));
    return result;
}

inline std::array<double,13> residual(const Parameters& p, const Matrix& f,
                                     const State& accepted, const State& candidate) {
    Vector6 ep=plasticStrain(candidate);
    const Vector6 calculated=normalizedSecondPiola(p,f,ep);
    std::array<double,13> r{};
    for (unsigned i=0;i<6;++i) r[6+i]=candidate[6+i]-calculated[i];
    Vector6 s{};
    for (unsigned i=0;i<6;++i) s[i]=candidate[6+i];
    const Vector6 flow=hillFlow(p,s);
    const double delta=candidate[12]-accepted[12];
    for (unsigned i=0;i<6;++i) r[i]=candidate[i]-accepted[i]-delta*flow[i];
    const double b=1.0-hillEquivalent(p,s);
    r[12]=std::sqrt(delta*delta+b*b+p.fischerBurmeisterEpsilon*p.fischerBurmeisterEpsilon)
        -delta-b;
    return r;
}

inline bool solveLinear(std::array<double,169> a, std::array<double,13> b,
                        std::array<double,13>& x) {
    for (unsigned k=0;k<13;++k) {
        unsigned pivot=k;
        for (unsigned i=k+1;i<13;++i)
            if (std::abs(a[13*i+k])>std::abs(a[13*pivot+k])) pivot=i;
        if (!(std::abs(a[13*pivot+k])>1.0e-18)) return false;
        if (pivot!=k) {
            for (unsigned j=k;j<13;++j) std::swap(a[13*k+j],a[13*pivot+j]);
            std::swap(b[k],b[pivot]);
        }
        for (unsigned i=k+1;i<13;++i) {
            const double ratio=a[13*i+k]/a[13*k+k];
            for (unsigned j=k+1;j<13;++j) a[13*i+j]-=ratio*a[13*k+j];
            b[i]-=ratio*b[k];
        }
    }
    for (int i=12;i>=0;--i) {
        double value=b[static_cast<unsigned>(i)];
        for (unsigned j=static_cast<unsigned>(i)+1;j<13;++j)
            value-=a[13*static_cast<unsigned>(i)+j]*x[j];
        x[static_cast<unsigned>(i)]=value/a[13*static_cast<unsigned>(i)+static_cast<unsigned>(i)];
    }
    return true;
}

struct Result {
    State state{};
    Matrix secondPiola{};
    Matrix firstPiola{};
    double equivalentStress = 0.0;
    double residual = 0.0;
    double plasticWorkMPa = 0.0;
    unsigned iterations = 0;
};

inline Result project(MaterialKind kind, const Matrix& f, const State& accepted,
                      unsigned maxIterations=24, double tolerance=1.0e-11) {
    const Parameters& p=parameters(kind);
    State x=accepted;
    const Vector6 trial=normalizedSecondPiola(p,f,plasticStrain(accepted));
    for (unsigned i=0;i<6;++i) x[6+i]=trial[i];
    unsigned iterations=0;
    for (;iterations<maxIterations;++iterations) {
        const auto r=residual(p,f,accepted,x);
        const double norm=residualNorm(r);
        double stateNorm=0.0;
        for (double value:x) stateNorm=std::max(stateNorm,std::abs(value));
        if (norm<=tolerance*(1.0+stateNorm)) break;
        std::array<double,169> jac{};
        for (unsigned column=0;column<13;++column) {
            State plus=x,minus=x;
            const double h=1.0e-7*(1.0+std::abs(x[column]));
            plus[column]+=h; minus[column]-=h;
            const auto rp=residual(p,f,accepted,plus);
            const auto rm=residual(p,f,accepted,minus);
            for (unsigned row=0;row<13;++row)
                jac[13*row+column]=(rp[row]-rm[row])/(2*h);
        }
        std::array<double,13> rhs{},step{};
        for (unsigned i=0;i<13;++i) rhs[i]=-r[i];
        if (!solveLinear(jac,rhs,step)) throw std::runtime_error("paper oracle singular local Jacobian");
        double scale=1.0;
        bool acceptedStep=false;
        for (unsigned backtrack=0;backtrack<16;++backtrack) {
            State trial=x;
            for (unsigned i=0;i<13;++i) trial[i]+=scale*step[i];
            const double trialNorm=residualNorm(residual(p,f,accepted,trial));
            if (trialNorm<(1.0-1.0e-4*scale)*norm) {
                x=trial; acceptedStep=true; break;
            }
            scale*=0.5;
        }
        if (!acceptedStep) {
            unsigned worst=0;
            for (unsigned i=1;i<13;++i) if (std::abs(r[i])>std::abs(r[worst])) worst=i;
            throw std::runtime_error(
                "paper oracle line search failed material="+std::string(p.name)+
                " iteration="+std::to_string(iterations)+" norm="+std::to_string(norm)+
                " worst_row="+std::to_string(worst)+" residual="+std::to_string(r[worst])+
                " multiplier="+std::to_string(x[12])+" q="+
                std::to_string(hillEquivalent(p,returnedNormalizedStress(p,f,accepted,x)))+
                " s="+std::to_string(x[6])+","+std::to_string(x[7])+","+std::to_string(x[8])+
                " ep="+std::to_string(x[0])+","+std::to_string(x[1])+","+std::to_string(x[2])+
                " oldep="+std::to_string(accepted[0])+","+std::to_string(accepted[1])+","+
                    std::to_string(accepted[2])+" oldlambda="+std::to_string(accepted[12]));
        }
    }
    const double finalNorm=residualNorm(residual(p,f,accepted,x));
    double stateNorm=0.0;
    for (double value:x) stateNorm=std::max(stateNorm,std::abs(value));
    if (finalNorm>tolerance*(1.0+stateNorm)) throw std::runtime_error("paper oracle did not converge");
    const Vector6 ep=plasticStrain(x);
    const Vector6 s=returnedNormalizedStress(p,f,accepted,x);
    Vector6 dep{};
    for (unsigned i=0;i<6;++i) dep[i]=x[i]-accepted[i];
    const Matrix second=secondPiola(p,f,ep);
    const Matrix first=multiply(f,second);
    const double work=second[0]*dep[0]+second[4]*dep[1]+second[8]*dep[2]
        +2.0*(second[5]*dep[3]+second[2]*dep[4]+second[1]*dep[5]);
    return {x,second,first,hillEquivalent(p,s),finalNorm,work,iterations};
}

inline Matrix directionalTangent(MaterialKind kind, const Matrix& f, const Matrix& h,
                                const State& accepted, double step=1.0e-6) {
    Matrix plus=f,minus=f;
    for (unsigned i=0;i<9;++i) { plus[i]+=step*h[i]; minus[i]-=step*h[i]; }
    const Matrix pp=project(kind,plus,accepted).firstPiola;
    const Matrix pm=project(kind,minus,accepted).firstPiola;
    Matrix result{};
    for (unsigned i=0;i<9;++i) result[i]=(pp[i]-pm[i])/(2*step);
    return result;
}

inline double maxAbs(const Matrix& a) {
    double result=0.0;
    for (double x:a) result=std::max(result,std::abs(x));
    return result;
}

} // namespace numi_cardboard_paper
