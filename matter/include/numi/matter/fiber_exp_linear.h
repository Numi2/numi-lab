#pragma once

// FEBio FEUncoupledFiberExpLinear: q=lambda*dW/dlambda. The toe-energy
// integral is evaluated by 16-point Gauss-Legendre quadrature. Analytic
// derivatives preserve the source's right-branch tangent at both seams.
// The quadrature certificate is restricted to lambdaMax<=1.25 and
// c4*(lambdaMax-1)<=16. Values outside it fail, never extrapolate silently.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define NM_FIBER_REF thread
#else
#include <cmath>
#define NM_FIBER_REF
#endif
namespace numi_matter_fiber {
template<typename T> inline bool finite(T x) {
#ifdef __METAL_VERSION__
    return metal::isfinite(x);
#else
    return std::isfinite(x);
#endif
}
template<typename T> inline T exponential(T x) {
#ifdef __METAL_VERSION__
    return metal::exp(x);
#else
    return std::exp(x);
#endif
}
template<typename T> inline T logarithm(T x) {
#ifdef __METAL_VERSION__
    return metal::log(x);
#else
    return std::log(x);
#endif
}
template<typename T> inline T exponentialMinusOne(T x) {
#ifdef __METAL_VERSION__
    // Avoid cancellation near the tension threshold.
    if (metal::abs(x)<T(0.01)) return x*(T(1)+x*(T(.5)+x*(T(1.0/6)+x*(T(1.0/24)+x*T(1.0/120)))));
    return metal::exp(x)-T(1);
#else
    return std::expm1(x);
#endif
}
template<typename T> inline T evaluate(
    T x,T c3,T c4,T c5,T limit,unsigned order,NM_FIBER_REF bool& valid
) {
    if (!finite(x)||!finite(c3)||!finite(c4)||!finite(c5)||!finite(limit)||
        !(x>T(0))||c3<T(0)||!(c4>T(0))||c5<T(0)||!(limit>T(1))||
        limit>T(1.25)||c4*(limit-T(1))>T(16)||order>2) {
        valid=false;return T(0);
    }
    // FEBio's c3==0 sentinel chooses a C2 transition using c5.
    if(c3==T(0)) c3=c5/(c4*exponential(c4*(limit-T(1))));
    if(x<T(1)) return T(0);
    const T qLimit=c3*exponentialMinusOne(c4*(limit-T(1)));
    const bool toe=x<limit;
    const T q=toe ? c3*exponentialMinusOne(c4*(x-T(1))) : qLimit+c5*(x-limit);
    if(order==1) { const T value=q/x;valid=valid&&finite(value);return value; }
    if(order==2) {
        const T derivative=toe ? c3*c4*exponential(c4*(x-T(1))) : c5;
        const T value=(derivative-q/x)/x;valid=valid&&finite(value);return value;
    }
    const T top=toe?x:limit, intervalHalf=(top-T(1))/T(2);
    const T nodes[8]={T(.0950125098376374402),T(.281603550779258913),T(.458016777657227386),T(.617876244402643748),T(.755404408355003034),T(.865631202387831744),T(.944575023073232576),T(.989400934991649933)};
    const T weights[8]={T(.189450610455068496),T(.182603415044923589),T(.169156519395002538),T(.149595988816576732),T(.124628971255533872),T(.0951585116824927848),T(.0622535239386478929),T(.0271524594117540949)};
    T sum=T(0);
    for(unsigned i=0;i<8;++i) {
        const T a=intervalHalf*(T(1)-nodes[i]), b=intervalHalf*(T(1)+nodes[i]);
        sum+=weights[i]*(exponentialMinusOne(c4*a)/(T(1)+a)+exponentialMinusOne(c4*b)/(T(1)+b));
    }
    T energy=c3*intervalHalf*sum;
    if(!toe) {
        const T z=(x-limit)/limit;
        // x-limit-limit*log(x/limit) has severe FP32 cancellation near seam.
        T remainder;
        if(z<T(.01)) remainder=limit*z*z*(T(.5)+z*(T(-1.0/3)+z*(T(.25)+z*(T(-.2)+z*T(1.0/6)))));
        else remainder=(x-limit)-limit*logarithm(x/limit);
        energy+=c5*remainder+qLimit*logarithm(x/limit);
    }
    valid=valid&&finite(energy);return energy;
}
} // namespace numi_matter_fiber
#undef NM_FIBER_REF
