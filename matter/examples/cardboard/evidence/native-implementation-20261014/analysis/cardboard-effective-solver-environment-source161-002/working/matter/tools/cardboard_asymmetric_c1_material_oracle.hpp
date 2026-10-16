#pragma once

// Independent FP64 material-point oracle for the explicit C1 tensile/compressive
// transfer assets. This evaluates the declared StVK/Hill/Huber equations below
// directly; it does not interpret the authored bytecode or call the Metal law.

#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace numi_cardboard_asymmetric_c1_oracle {

constexpr unsigned kStateCount=15u;
constexpr unsigned kDerivativeCount=24u; // 15 local unknowns + 9 F entries
using Matrix=std::array<double,9>;
using State=std::array<double,kStateCount>;
using Residual=std::array<double,kStateCount>;
using Jacobian=std::array<double,kStateCount*kStateCount>;

struct Point {
    State state{};
    Matrix firstPiolaPa{};
    double residualInfinity=0.0;
    unsigned iterations=0u;
};

// Material-point tangent audit data. This is computed directly from the
// independent FP64 equation evaluator, not from compiled bytecode. The
// deformation residual derivative uses row-major F directions and is kept
// separate from the local Jacobian so native diagnostics can identify which
// side of the implicit chain first diverges.
struct Linearization {
    Residual residual{};
    Jacobian jacobian{};
    State deformationResidualDerivative{};
    std::array<double,kStateCount*9u> stressStateDerivative{};
    Matrix fixedStateTangent{};
};

template<std::size_t N> struct Dual {
    double value=0.0;
    std::array<double,N> derivative{};
    Dual()=default;
    Dual(const double x):value(x) {}
    static Dual variable(const double x,const std::size_t index) {
        Dual result{x}; result.derivative.at(index)=1.0; return result;
    }
};

template<std::size_t N> inline Dual<N> operator+(const Dual<N>& a,const Dual<N>& b) {
    Dual<N> r{a.value+b.value};
    for(std::size_t i=0;i<N;++i)r.derivative[i]=a.derivative[i]+b.derivative[i];
    return r;
}
template<std::size_t N> inline Dual<N> operator-(const Dual<N>& a,const Dual<N>& b) {
    Dual<N> r{a.value-b.value};
    for(std::size_t i=0;i<N;++i)r.derivative[i]=a.derivative[i]-b.derivative[i];
    return r;
}
template<std::size_t N> inline Dual<N> operator-(const Dual<N>& a) {
    Dual<N> r{-a.value};
    for(std::size_t i=0;i<N;++i)r.derivative[i]=-a.derivative[i];
    return r;
}
template<std::size_t N> inline Dual<N> operator*(const Dual<N>& a,const Dual<N>& b) {
    Dual<N> r{a.value*b.value};
    for(std::size_t i=0;i<N;++i)
        r.derivative[i]=a.derivative[i]*b.value+a.value*b.derivative[i];
    return r;
}
template<std::size_t N> inline Dual<N> operator/(const Dual<N>& a,const Dual<N>& b) {
    if(!(std::abs(b.value)>1.0e-30))throw std::runtime_error("C1 oracle division by zero");
    Dual<N> r{a.value/b.value};
    const double denominator=b.value*b.value;
    for(std::size_t i=0;i<N;++i)
        r.derivative[i]=(a.derivative[i]*b.value-a.value*b.derivative[i])/denominator;
    return r;
}
template<std::size_t N> inline Dual<N> sqrt(const Dual<N>& a) {
    if(!(a.value>0.0))throw std::runtime_error("C1 oracle sqrt requires positive input");
    Dual<N> r{std::sqrt(a.value)};
    const double scale=0.5/r.value;
    for(std::size_t i=0;i<N;++i)r.derivative[i]=scale*a.derivative[i];
    return r;
}
template<std::size_t N> inline Dual<N> operator+(const Dual<N>& a,const double b) { return a+Dual<N>{b}; }
template<std::size_t N> inline Dual<N> operator+(const double a,const Dual<N>& b) { return Dual<N>{a}+b; }
template<std::size_t N> inline Dual<N> operator-(const Dual<N>& a,const double b) { return a-Dual<N>{b}; }
template<std::size_t N> inline Dual<N> operator-(const double a,const Dual<N>& b) { return Dual<N>{a}-b; }
template<std::size_t N> inline Dual<N> operator*(const Dual<N>& a,const double b) { return a*Dual<N>{b}; }
template<std::size_t N> inline Dual<N> operator*(const double a,const Dual<N>& b) { return Dual<N>{a}*b; }
template<std::size_t N> inline Dual<N> operator/(const Dual<N>& a,const double b) { return a/Dual<N>{b}; }
template<std::size_t N> inline Dual<N> operator/(const double a,const Dual<N>& b) { return Dual<N>{a}/b; }

inline double parameter(const numi::matter::MaterialProgram& material,
                        const std::string& name) {
    const auto found=std::find_if(material.parameters.begin(),material.parameters.end(),
        [&](const numi::matter::Parameter& p){return p.name==name;});
    if(found==material.parameters.end())throw std::runtime_error("C1 oracle missing parameter: "+name);
    return found->defaultValue;
}

template<class T> struct Evaluation {
    std::array<T,kStateCount> residual{};
    std::array<T,9> firstPiola{};
};

template<std::size_t N> inline Evaluation<Dual<N>> evaluateSystem(
    const numi::matter::MaterialProgram& material,const Matrix& f,
    const State& accepted,const State& candidate,const bool seed) {
    if(material.internalState.size()!=kStateCount)
        throw std::runtime_error("C1 oracle requires exactly fifteen states");
    std::array<Dual<N>,kStateCount> z{};
    std::array<Dual<N>,9> F{};
    for(unsigned i=0;i<kStateCount;++i)
        z[i]=seed?Dual<N>::variable(candidate[i],i):Dual<N>{candidate[i]};
    for(unsigned i=0;i<9u;++i)
        F[i]=seed?Dual<N>::variable(f[i],kStateCount+i):Dual<N>{f[i]};

    const Dual<N> e11=0.5*(F[0]*F[0]+F[3]*F[3]+F[6]*F[6]-1.0);
    const Dual<N> e22=0.5*(F[1]*F[1]+F[4]*F[4]+F[7]*F[7]-1.0);
    const Dual<N> e33=0.5*(F[2]*F[2]+F[5]*F[5]+F[8]*F[8]-1.0);
    const Dual<N> c12=F[0]*F[1]+F[3]*F[4]+F[6]*F[7];
    const Dual<N> c13=F[0]*F[2]+F[3]*F[5]+F[6]*F[8];
    const Dual<N> c23=F[1]*F[2]+F[4]*F[5]+F[7]*F[8];
    const Dual<N> ee11=e11-z[0],ee22=e22-z[1],ee33=e33-z[2];
    const Dual<N> g12=c12-2.0*z[5],g13=c13-2.0*z[4],g23=c23-2.0*z[3];

    const double C11=parameter(material,"c11"),C22=parameter(material,"c22");
    const double C33=parameter(material,"c33"),C12=parameter(material,"c12");
    const double C13=parameter(material,"c13"),C23=parameter(material,"c23");
    const double G12=parameter(material,"g12"),G13=parameter(material,"g13");
    const double G23=parameter(material,"g23"),sigma=parameter(material,"sigma_ref");
    const Dual<N> S11=C11*ee11+C12*ee22+C13*ee33;
    const Dual<N> S22=C12*ee11+C22*ee22+C23*ee33;
    const Dual<N> S33=C13*ee11+C23*ee22+C33*ee33;
    const Dual<N> S12=G12*g12,S13=G13*g13,S23=G23*g23;

    const Dual<N> s11=z[6],s22=z[7],s33=z[8],s23=z[9],s13=z[10],s12=z[11];
    const double hillF=parameter(material,"hill_F"),hillG=parameter(material,"hill_G");
    const double hillH=parameter(material,"hill_H"),hillL=parameter(material,"hill_L");
    const double hillM=parameter(material,"hill_M"),hillN=parameter(material,"hill_N");
    const double qFloor=parameter(material,"q_floor");
    const double delta=parameter(material,"compression_smoothing_width");
    const Dual<N> qSquared=hillF*(s22-s33)*(s22-s33)+hillG*(s33-s11)*(s33-s11)
        +hillH*(s11-s22)*(s11-s22)+2.0*hillL*s23*s23+2.0*hillM*s13*s13
        +2.0*hillN*s12*s12+qFloor*qFloor;
    const Dual<N> qHill=sqrt(qSquared);
    const auto rDelta=[&](const Dual<N>& s) {
        if(s.value>=delta)return Dual<N>{0.0};
        if(s.value<=-delta)return -s;
        const Dual<N> widthMinusS=delta-s;
        return widthMinusS*widthMinusS/(4.0*delta);
    };
    const auto rPrime=[&](const Dual<N>& s) {
        if(s.value>=delta)return Dual<N>{0.0};
        if(s.value<=-delta)return Dual<N>{-1.0};
        return (s-delta)/(2.0*delta);
    };
    const double k11=parameter(material,"compression_k11");
    const double k22=parameter(material,"compression_k22");
    const double k33=parameter(material,"compression_k33");
    const Dual<N> q=qHill+k11*rDelta(s11)+k22*rDelta(s22)+k33*rDelta(s33);
    const Dual<N> flow11=(hillG*(s11-s33)+hillH*(s11-s22))/qHill+k11*rPrime(s11);
    const Dual<N> flow22=(hillF*(s22-s33)+hillH*(s22-s11))/qHill+k22*rPrime(s22);
    const Dual<N> flow33=(hillF*(s33-s22)+hillG*(s33-s11))/qHill+k33*rPrime(s33);
    const Dual<N> flow23=hillL*s23/qHill;
    const Dual<N> flow13=hillM*s13/qHill;
    const Dual<N> flow12=hillN*s12/qHill;

    const Dual<N> lambdaIncrement=z[12]-accepted[12];
    const double smoothWidth=parameter(material,"hardening_smoothing_width");
    const auto smoothPositive=[&](const Dual<N>& x) {
        return 0.5*(x+sqrt(x*x+smoothWidth*smoothWidth));
    };
    const Dual<N> positive11=smoothPositive(s11),positive22=smoothPositive(s22);
    const Dual<N> positive33=smoothPositive(s33);
    const double weightFloor=qFloor;
    const Dual<N> weightDenominator=positive11*positive11+positive22*positive22+
        positive33*positive33+weightFloor*weightFloor;
    const Dual<N> weightMD=positive11*positive11/weightDenominator;
    const Dual<N> weightCD=positive22*positive22/weightDenominator;
    const double Hmd=parameter(material,"hardening_md");
    const double Hcd=parameter(material,"hardening_cd");
    const double scaleMD=parameter(material,"fit_transfer_scale_md")*
                         parameter(material,"hardening_md_flow_scale");
    const double scaleCD=parameter(material,"fit_transfer_scale_cd")*
                         parameter(material,"hardening_cd_flow_scale");
    const Dual<N> effectiveRadius=1.0+Hmd*scaleMD*z[13]/sigma+Hcd*scaleCD*z[14]/sigma;
    const Dual<N> complementarityGap=effectiveRadius-q;
    const double fb=parameter(material,"fb_epsilon");
    const double residualScale=parameter(material,"residual_scale");

    Evaluation<Dual<N>> result;
    result.residual[0]=residualScale*(z[0]-accepted[0]-lambdaIncrement*flow11);
    result.residual[1]=residualScale*(z[1]-accepted[1]-lambdaIncrement*flow22);
    result.residual[2]=residualScale*(z[2]-accepted[2]-lambdaIncrement*flow33);
    result.residual[3]=residualScale*(z[3]-accepted[3]-lambdaIncrement*flow23);
    result.residual[4]=residualScale*(z[4]-accepted[4]-lambdaIncrement*flow13);
    result.residual[5]=residualScale*(z[5]-accepted[5]-lambdaIncrement*flow12);
    result.residual[6]=residualScale*(z[6]-S11/sigma);
    result.residual[7]=residualScale*(z[7]-S22/sigma);
    result.residual[8]=residualScale*(z[8]-S33/sigma);
    result.residual[9]=residualScale*(z[9]-S23/sigma);
    result.residual[10]=residualScale*(z[10]-S13/sigma);
    result.residual[11]=residualScale*(z[11]-S12/sigma);
    result.residual[12]=residualScale*(sqrt(lambdaIncrement*lambdaIncrement+
        complementarityGap*complementarityGap+fb*fb)-lambdaIncrement-complementarityGap);
    result.residual[13]=residualScale*(z[13]-accepted[13]-lambdaIncrement*weightMD);
    result.residual[14]=residualScale*(z[14]-accepted[14]-lambdaIncrement*weightCD);

    const std::array<Dual<N>,9> S{S11,S12,S13,S12,S22,S23,S13,S23,S33};
    for(unsigned row=0;row<3u;++row)for(unsigned column=0;column<3u;++column)
        for(unsigned k=0;k<3u;++k)
            result.firstPiola[row*3u+column]=result.firstPiola[row*3u+column]+
                F[row*3u+k]*S[k*3u+column];
    return result;
}

inline Linearization linearize(const numi::matter::ConstitutiveProgram& program,
                               const Matrix& f,const Matrix& h,
                               const State& accepted,const State& candidate) {
    const auto system=evaluateSystem<kDerivativeCount>(program.material,f,
                                                        accepted,candidate,true);
    Linearization result;
    for(unsigned row=0;row<kStateCount;++row) {
        result.residual[row]=system.residual[row].value;
        for(unsigned column=0;column<kStateCount;++column)
            result.jacobian[kStateCount*row+column]=
                system.residual[row].derivative[column];
        double deformationDerivative=0.0;
        for(unsigned component=0;component<9u;++component)
            deformationDerivative+=system.residual[row].derivative[
                kStateCount+component]*h[component];
        result.deformationResidualDerivative[row]=deformationDerivative;
    }
    for(unsigned component=0;component<9u;++component) {
        for(unsigned state=0;state<kStateCount;++state)
            result.stressStateDerivative[component*kStateCount+state]=
                system.firstPiola[component].derivative[state];
        for(unsigned deformationComponent=0;deformationComponent<9u;
            ++deformationComponent)
            result.fixedStateTangent[component]+=
                system.firstPiola[component].derivative[
                    kStateCount+deformationComponent]*h[deformationComponent];
    }
    return result;
}

inline double normInfinity(const Residual& r) {
    double value=0.0;for(double x:r)value=std::max(value,std::abs(x));return value;
}

inline bool solve(Jacobian matrix,State rhs,State& solution) {
    constexpr unsigned n=kStateCount;
    for(unsigned col=0;col<n;++col) {
        unsigned pivot=col;double magnitude=std::abs(matrix[n*col+col]);
        for(unsigned row=col+1u;row<n;++row) {
            const double candidate=std::abs(matrix[n*row+col]);
            if(candidate>magnitude){magnitude=candidate;pivot=row;}
        }
        if(!std::isfinite(magnitude)||magnitude<1.0e-14)return false;
        if(pivot!=col) {
            for(unsigned j=col;j<n;++j)std::swap(matrix[n*col+j],matrix[n*pivot+j]);
            std::swap(rhs[col],rhs[pivot]);
        }
        const double diagonal=matrix[n*col+col];
        for(unsigned row=col+1u;row<n;++row) {
            const double factor=matrix[n*row+col]/diagonal;
            for(unsigned j=col+1u;j<n;++j)matrix[n*row+j]-=factor*matrix[n*col+j];
            rhs[row]-=factor*rhs[col];
        }
    }
    for(int row=static_cast<int>(n)-1;row>=0;--row) {
        double value=rhs[static_cast<unsigned>(row)];
        for(unsigned col=static_cast<unsigned>(row)+1u;col<n;++col)
            value-=matrix[n*static_cast<unsigned>(row)+col]*solution[col];
        solution[static_cast<unsigned>(row)]=
            value/matrix[n*static_cast<unsigned>(row)+static_cast<unsigned>(row)];
        if(!std::isfinite(solution[static_cast<unsigned>(row)]))return false;
    }
    return true;
}

inline Point project(const numi::matter::ConstitutiveProgram& program,
                     const Matrix& f,const State& accepted) {
    const auto& material=program.material;
    if(material.internalState.size()!=kStateCount)
        throw std::runtime_error("C1 oracle requires a 15-state constitutive program");
    State z=accepted;
    // Elastic trial stress is a useful, explicit initial iterate for the six
    // normalized second-Piola stress unknowns; all plastic states start accepted.
    const auto trial=evaluateSystem<kDerivativeCount>(material,f,accepted,z,false);
    for(unsigned s=0;s<6u;++s) z[6u+s]=-trial.residual[6u+s].value;
    constexpr double tolerance=2.0e-13;
    Point result;bool converged=false;
    for(unsigned iteration=0;iteration<160u;++iteration) {
        const auto system=evaluateSystem<kDerivativeCount>(material,f,accepted,z,true);
        Residual residual{};Jacobian jac{};
        for(unsigned row=0;row<kStateCount;++row) {
            residual[row]=system.residual[row].value;
            for(unsigned col=0;col<kStateCount;++col)
                jac[kStateCount*row+col]=system.residual[row].derivative[col];
        }
        const double current=normInfinity(residual);
        if(current<tolerance){result.iterations=iteration;converged=true;break;}
        State rhs{},correction{};
        for(unsigned i=0;i<kStateCount;++i)rhs[i]=-residual[i];
        if(!solve(jac,rhs,correction))throw std::runtime_error("C1 oracle local Jacobian singular");
        bool acceptedStep=false;double alpha=1.0;
        for(unsigned backtrack=0;backtrack<28u;++backtrack) {
            State candidate=z;
            for(unsigned i=0;i<kStateCount;++i)candidate[i]+=alpha*correction[i];
            const auto trialSystem=evaluateSystem<kDerivativeCount>(material,f,accepted,candidate,false);
            Residual trialResidual{};
            for(unsigned i=0;i<kStateCount;++i)trialResidual[i]=trialSystem.residual[i].value;
            if(normInfinity(trialResidual)<current) {z=candidate;acceptedStep=true;break;}
            alpha*=0.5;
        }
        if(!acceptedStep)throw std::runtime_error("C1 oracle residual line search stalled");
    }
    const auto final=evaluateSystem<kDerivativeCount>(material,f,accepted,z,false);
    Residual finalResidual{};
    for(unsigned i=0;i<kStateCount;++i)finalResidual[i]=final.residual[i].value;
    result.residualInfinity=normInfinity(finalResidual);
    if(!converged&&result.residualInfinity<tolerance){result.iterations=160u;converged=true;}
    if(!converged)throw std::runtime_error("C1 oracle projection did not converge");
    result.state=z;
    for(unsigned i=0;i<9u;++i)result.firstPiolaPa[i]=final.firstPiola[i].value;
    return result;
}

inline Matrix tangent(const numi::matter::ConstitutiveProgram& program,
                     const Matrix& f,const Matrix& h,
                     const State& accepted,const State& candidate) {
    const auto system=evaluateSystem<kDerivativeCount>(program.material,f,accepted,candidate,true);
    Jacobian jac{};State rhs{},stateDirection{};
    for(unsigned row=0;row<kStateCount;++row) {
        for(unsigned col=0;col<kStateCount;++col)
            jac[kStateCount*row+col]=system.residual[row].derivative[col];
        for(unsigned j=0;j<9u;++j)
            rhs[row]-=system.residual[row].derivative[kStateCount+j]*h[j];
    }
    if(!solve(jac,rhs,stateDirection))throw std::runtime_error("C1 oracle tangent Jacobian singular");
    Matrix result{};
    for(unsigned p=0;p<9u;++p) {
        result[p]=0.0;
        for(unsigned j=0;j<9u;++j)
            result[p]+=system.firstPiola[p].derivative[kStateCount+j]*h[j];
        for(unsigned s=0;s<kStateCount;++s)
            result[p]+=system.firstPiola[p].derivative[s]*stateDirection[s];
    }
    return result;
}

} // namespace numi_cardboard_asymmetric_c1_oracle
