#pragma once

// Reduced scalar return map for the source-authored 13-state associative Hill
// ideal-plastic system. The equations and state ordering match the two
// hajali2009_*_hill_ideal.nmatter materials. No hardening or tolerance change
// is introduced here.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace numi::cardboard::hill_scalar {

template <class T> using Vector6 = std::array<T, 6>;
template <class T> using Matrix6 = std::array<T, 36>;
template <class T> using State13 = std::array<T, 13>;

template <class T>
struct Parameters {
    // Captured table convention: stiffness and sigmaReference share units;
    // the return map only uses their ratios.
    std::array<T, 6> stiffness{}; // c11,c22,c33,c12,c13,c23
    std::array<T, 3> shear{};      // g12,g13,g23
    T sigmaReference{};
    std::array<T, 6> hill{};       // F,G,H,L,M,N
    T qFloor{};
    T fischerBurmeisterEpsilon{};
};

enum class Status : unsigned char {
    success,
    invalid_input,
    singular_stress_solve,
    no_bracket,
    nonfinite
};

template <class T>
struct Result {
    Status status = Status::invalid_input;
    State13<T> state{};
    Vector6<T> trialStress{};
    Vector6<T> returnedStress{};
    T t = T(0);
    T lambdaIncrement = T(0);
    T equivalentStress = T(0);
    T scalarResidual = T(0);
    T flowResidual = T(0);
    T stressResidual = T(0);
    T fischerBurmeisterResidual = T(0);
    T bracketLow = T(0);
    T bracketHigh = T(0);
    unsigned iterations = 0;
};

template <class T>
inline bool finite(T value) {
    return std::isfinite(static_cast<double>(value));
}

template <class T>
inline bool allFinite(const Parameters<T>& p, const State13<T>& oldState,
                      const Vector6<T>& trialStress) {
    for (T v : p.stiffness) if (!finite(v)) return false;
    for (T v : p.shear) if (!finite(v)) return false;
    for (T v : p.hill) if (!finite(v)) return false;
    for (T v : oldState) if (!finite(v)) return false;
    for (T v : trialStress) if (!finite(v)) return false;
    return finite(p.sigmaReference) && finite(p.qFloor) &&
        finite(p.fischerBurmeisterEpsilon) && p.sigmaReference > T(0) &&
        p.qFloor > T(0) && p.fischerBurmeisterEpsilon > T(0);
}

template <class T>
inline Matrix6<T> flowMatrix(const Parameters<T>& p) {
    const T F=p.hill[0], G=p.hill[1], H=p.hill[2];
    Matrix6<T> D{};
    // This is the exact associated-flow matrix in source state order
    // 11,22,33,23,13,12. The shear entries are L,M,N (not 2L,2M,2N);
    // the Hill norm below carries the source's factor of two.
    D[0*6+0]=G+H; D[0*6+1]=-H;  D[0*6+2]=-G;
    D[1*6+0]=-H;  D[1*6+1]=F+H; D[1*6+2]=-F;
    D[2*6+0]=-G;  D[2*6+1]=-F;  D[2*6+2]=F+G;
    D[3*6+3]=p.hill[3];
    D[4*6+4]=p.hill[4];
    D[5*6+5]=p.hill[5];
    return D;
}

template <class T>
inline Matrix6<T> normalizedElasticMatrix(const Parameters<T>& p) {
    Matrix6<T> C{};
    const T inverseSigma=T(1)/p.sigmaReference;
    C[0*6+0]=p.stiffness[0]*inverseSigma;
    C[1*6+1]=p.stiffness[1]*inverseSigma;
    C[2*6+2]=p.stiffness[2]*inverseSigma;
    C[0*6+1]=C[1*6+0]=p.stiffness[3]*inverseSigma;
    C[0*6+2]=C[2*6+0]=p.stiffness[4]*inverseSigma;
    C[1*6+2]=C[2*6+1]=p.stiffness[5]*inverseSigma;
    C[3*6+3]=T(2)*p.shear[2]*inverseSigma; // 23
    C[4*6+4]=T(2)*p.shear[1]*inverseSigma; // 13
    C[5*6+5]=T(2)*p.shear[0]*inverseSigma; // 12
    return C;
}

template <class T>
inline Matrix6<T> multiply(const Matrix6<T>& a, const Matrix6<T>& b) {
    Matrix6<T> c{};
    for (unsigned i=0;i<6;++i)
        for (unsigned j=0;j<6;++j)
            for (unsigned k=0;k<6;++k)
                c[6*i+j]+=a[6*i+k]*b[6*k+j];
    return c;
}

template <class T>
inline T hillEquivalent(const Parameters<T>& p, const Vector6<T>& s) {
    const T F=p.hill[0], G=p.hill[1], H=p.hill[2];
    const T q2=F*(s[1]-s[2])*(s[1]-s[2])
        +G*(s[2]-s[0])*(s[2]-s[0])
        +H*(s[0]-s[1])*(s[0]-s[1])
        +T(2)*p.hill[3]*s[3]*s[3]
        +T(2)*p.hill[4]*s[4]*s[4]
        +T(2)*p.hill[5]*s[5]*s[5]
        +p.qFloor*p.qFloor;
    if (!(q2>=T(0)) || !finite(q2)) return std::numeric_limits<T>::quiet_NaN();
    return std::sqrt(q2);
}

template <class T>
inline bool solve6(Matrix6<T> a, Vector6<T> b, Vector6<T>& x) {
    for (unsigned k=0;k<6;++k) {
        unsigned pivot=k;
        T largest=std::abs(a[6*k+k]);
        for (unsigned i=k+1;i<6;++i) {
            const T candidate=std::abs(a[6*i+k]);
            if (candidate>largest) { largest=candidate; pivot=i; }
        }
        if (!(largest>T(0)) || !finite(largest)) return false;
        if (pivot!=k) {
            for (unsigned j=k;j<6;++j) std::swap(a[6*k+j],a[6*pivot+j]);
            std::swap(b[k],b[pivot]);
        }
        const T diagonal=a[6*k+k];
        for (unsigned i=k+1;i<6;++i) {
            const T factor=a[6*i+k]/diagonal;
            a[6*i+k]=T(0);
            for (unsigned j=k+1;j<6;++j) a[6*i+j]-=factor*a[6*k+j];
            b[i]-=factor*b[k];
        }
    }
    for (int i=5;i>=0;--i) {
        T value=b[static_cast<unsigned>(i)];
        for (unsigned j=static_cast<unsigned>(i)+1;j<6;++j)
            value-=a[6*static_cast<unsigned>(i)+j]*x[j];
        const T diagonal=a[6*static_cast<unsigned>(i)+static_cast<unsigned>(i)];
        if (!(std::abs(diagonal)>T(0)) || !finite(diagonal)) return false;
        x[static_cast<unsigned>(i)]=value/diagonal;
        if (!finite(x[static_cast<unsigned>(i)])) return false;
    }
    return true;
}

template <class T>
inline T maxAbs(const Vector6<T>& x) {
    T result=T(0);
    for (T value:x) result=std::max(result,std::abs(value));
    return result;
}

template <class T>
struct ScalarEvaluation {
    bool valid=false;
    Vector6<T> stress{};
    T q=T(0);
    T residual=T(0);
};

template <class T>
inline ScalarEvaluation<T> evaluateT(const Parameters<T>& p,
    const Vector6<T>& trialStress,const Matrix6<T>& elasticTimesFlow,T t) {
    ScalarEvaluation<T> result;
    Matrix6<T> system{};
    for (unsigned i=0;i<6;++i) for (unsigned j=0;j<6;++j)
        system[6*i+j]=(i==j?T(1):T(0))+t*elasticTimesFlow[6*i+j];
    if (!solve6(system,trialStress,result.stress)) return result;
    result.q=hillEquivalent(p,result.stress);
    const T lambda=t*result.q;
    const T gap=T(1)-result.q;
    result.residual=lambda*gap-(p.fischerBurmeisterEpsilon*p.fischerBurmeisterEpsilon/T(2));
    result.valid=finite(result.q)&&finite(result.residual);
    return result;
}

// Safeguarded one-dimensional reduction. It brackets the exact smoothed FB
// equation lambda*(1-q)=epsilon^2/2, lambda=t*q, then returns the associated
// stress/plastic-strain state. The accepted state is const and only its
// accumulated multiplier is incremented in the returned candidate.
template <class T>
inline Result<T> returnMap(const Parameters<T>& p,const Vector6<T>& trialStress,
                           const State13<T>& accepted) {
    Result<T> out;
    if (!allFinite(p,accepted,trialStress)) return out;
    const Matrix6<T> C=normalizedElasticMatrix(p);
    const Matrix6<T> D=flowMatrix(p);
    const Matrix6<T> CD=multiply(C,D);
    const T q0=hillEquivalent(p,trialStress);
    if (!finite(q0)) { out.status=Status::nonfinite; return out; }
    const T halfEpsilonSquared=p.fischerBurmeisterEpsilon*p.fischerBurmeisterEpsilon/T(2);
    if (!(halfEpsilonSquared>T(0)) || !finite(halfEpsilonSquared)) {
        out.status=Status::invalid_input; return out;
    }

    T scale=T(0);
    for (T v:CD) scale=std::max(scale,std::abs(v));
    if (!(scale>T(0)) || !finite(scale)) { out.status=Status::invalid_input; return out; }
    T high=T(1)/scale;
    if (q0<T(1)) {
        const T denominator=q0*(T(1)-q0);
        if (denominator>T(0)) high=std::max(high,T(2)*halfEpsilonSquared/denominator);
    }
    if (!(high>T(0)) || !finite(high)) high=T(1)/scale;

    ScalarEvaluation<T> upper=evaluateT(p,trialStress,CD,high);
    unsigned bracketExpansions=0;
    // Starting at t=0 the scalar residual is strictly negative. Grow the
    // upper endpoint until it reaches the positive branch. This handles both
    // the elastic tiny-multiplier root and the post-yield root without a
    // Newton step through the yield kink.
    while (upper.valid && upper.residual<T(0) && bracketExpansions<256) {
        high*=T(2);
        if (!finite(high)) break;
        upper=evaluateT(p,trialStress,CD,high);
        ++bracketExpansions;
    }
    if (!upper.valid) { out.status=Status::singular_stress_solve; return out; }
    if (!(upper.residual>=T(0))) { out.status=Status::no_bracket; return out; }

    T low=T(0);
    T bestT=high;
    ScalarEvaluation<T> best=upper;
    for (unsigned iteration=0;iteration<192;++iteration) {
        const T middle=low+(high-low)/T(2);
        if (!(middle>low) || !(middle<high)) break; // adjacent representables
        const ScalarEvaluation<T> current=evaluateT(p,trialStress,CD,middle);
        if (!current.valid) { out.status=Status::singular_stress_solve; return out; }
        ++out.iterations;
        if (std::abs(current.residual)<std::abs(best.residual)) {
            best=current; bestT=middle;
        }
        if (current.residual<T(0)) low=middle;
        else high=middle;
    }
    const ScalarEvaluation<T> atLow=evaluateT(p,trialStress,CD,low);
    const ScalarEvaluation<T> atHigh=evaluateT(p,trialStress,CD,high);
    if (!atLow.valid || !atHigh.valid) { out.status=Status::singular_stress_solve; return out; }
    if (std::abs(atLow.residual)<std::abs(best.residual)) { best=atLow; bestT=low; }
    if (std::abs(atHigh.residual)<std::abs(best.residual)) { best=atHigh; bestT=high; }

    out.bracketLow=low;
    out.bracketHigh=high;
    out.t=bestT;
    out.trialStress=trialStress;
    out.returnedStress=best.stress;
    out.equivalentStress=best.q;
    out.lambdaIncrement=bestT*best.q;
    out.scalarResidual=best.residual;
    for (unsigned i=0;i<13;++i) out.state[i]=accepted[i];
    const T inverseQ=T(1)/best.q;
    for (unsigned i=0;i<6;++i) {
        T flowNumerator=T(0);
        for (unsigned j=0;j<6;++j) flowNumerator+=D[6*i+j]*best.stress[j];
        const T deltaPlastic=bestT*flowNumerator;
        out.state[i]=accepted[i]+deltaPlastic;
        out.flowResidual=std::max(out.flowResidual,
            std::abs(out.state[i]-accepted[i]-out.lambdaIncrement*flowNumerator*inverseQ));
    }
    for (unsigned i=0;i<6;++i) out.state[6+i]=best.stress[i];
    out.state[12]=accepted[12]+out.lambdaIncrement;

    // Residual of the reduced linear stress equation, evaluated in T.
    for (unsigned i=0;i<6;++i) {
        T value=best.stress[i]-trialStress[i];
        for (unsigned j=0;j<6;++j) value+=bestT*CD[6*i+j]*best.stress[j];
        out.stressResidual=std::max(out.stressResidual,std::abs(value));
    }
    const T gap=T(1)-best.q;
    out.fischerBurmeisterResidual=
        std::sqrt(out.lambdaIncrement*out.lambdaIncrement+gap*gap+
                  p.fischerBurmeisterEpsilon*p.fischerBurmeisterEpsilon)
        -out.lambdaIncrement-gap;
    if (!finite(out.lambdaIncrement) || !finite(out.state[12]) ||
        !finite(out.fischerBurmeisterResidual)) {
        out.status=Status::nonfinite; return out;
    }
    out.status=Status::success;
    return out;
}

// Exact source trial stress from F and the accepted additive Green-plastic
// strain. F is row-major; state order is Ep11,Ep22,Ep33,Ep23,Ep13,Ep12.
template <class T>
inline Vector6<T> elasticTrialStress(const Parameters<T>& p,
    const std::array<T,9>& F,const State13<T>& accepted) {
    // Keep the source expression order: sum the three squared F components,
    // subtract one, multiply by one half, subtract accepted Ep; then form the
    // unnormalized stress numerator and divide by sigmaReference. This matters
    // in FP32 near a rotated, nearly isochoric state.
    const T e11=T(0.5)*((F[0]*F[0]+F[3]*F[3]+F[6]*F[6])-T(1))-accepted[0];
    const T e22=T(0.5)*((F[1]*F[1]+F[4]*F[4]+F[7]*F[7])-T(1))-accepted[1];
    const T e33=T(0.5)*((F[2]*F[2]+F[5]*F[5]+F[8]*F[8])-T(1))-accepted[2];
    const T e23=F[1]*F[2]+F[4]*F[5]+F[7]*F[8]-T(2)*accepted[3];
    const T e13=F[0]*F[2]+F[3]*F[5]+F[6]*F[8]-T(2)*accepted[4];
    const T e12=F[0]*F[1]+F[3]*F[4]+F[6]*F[7]-T(2)*accepted[5];
    const auto& c=p.stiffness;
    return {
        (c[0]*e11+c[3]*e22+c[4]*e33)/p.sigmaReference,
        (c[3]*e11+c[1]*e22+c[5]*e33)/p.sigmaReference,
        (c[4]*e11+c[5]*e22+c[2]*e33)/p.sigmaReference,
        p.shear[2]*e23/p.sigmaReference,
        p.shear[1]*e13/p.sigmaReference,
        p.shear[0]*e12/p.sigmaReference
    };
}

} // namespace numi::cardboard::hill_scalar
