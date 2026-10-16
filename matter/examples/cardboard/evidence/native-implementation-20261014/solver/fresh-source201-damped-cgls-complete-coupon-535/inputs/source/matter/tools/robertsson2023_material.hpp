#pragma once
#include "numi/matter/matter.hpp"
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

// Robertsson et al., DOI 10.1002/pts.2774, equations 18--27, B1 and C1--C3.
// Native authoring of the full multiplicative, twelve-subsurface model.
// This is the source's single-ply softwood grade, NOT corrugated kraft/starch.
// Stress/plastic work are normalized in MPa inside the implicit system;
// the stored energy is converted to Pa. Fp is row-major. State 21 stores
// accumulated Delta-lambda in MPa; states 22--25 store onset work in MPa.
namespace numi_cardboard::robertsson2023 {
using namespace numi::matter;
using R = std::uint32_t;
using M = std::array<R,9>;
using V = std::array<R,3>;
struct Graph {
    MaterialProgram p;
    R c(double x) { return p.expressions.constant(x); }
    R var(ExprKind k, unsigned i) {
        Expr e; e.kind=k; e.index=i; return p.expressions.append(e);
    }
    R op(ExprKind k,R a,R b=NM_INVALID_INDEX,int n=0) {
        Expr e; e.kind=k; e.arguments[0]=a; e.arguments[1]=b;
        e.integer=n; e.dimension=p.expressions.nodes[a].dimension;
        if(k==ExprKind::multiply) e.dimension=e.dimension+p.expressions.nodes[b].dimension;
        if(k==ExprKind::divide) e.dimension=e.dimension-p.expressions.nodes[b].dimension;
        return p.expressions.append(e);
    }
    R add(R a,R b) {return op(ExprKind::add,a,b);}
    R sub(R a,R b) {return op(ExprKind::subtract,a,b);}
    R mul(R a,R b) {return op(ExprKind::multiply,a,b);}
    R div(R a,R b) {return op(ExprKind::divide,a,b);}
    R scale(R a,double s) {return mul(c(s),a);}
    R pow(R a,int n) {return op(ExprKind::integerPower,a,NM_INVALID_INDEX,n);}
    R sqrt(R a) {return op(ExprKind::squareRoot,a);}
    R log(R a) {return op(ExprKind::logarithm,a);}
    R exp(R a) {return op(ExprKind::exponential,a);}
    R choose(R condition,R yes,R no) {Expr e;e.kind=ExprKind::selectNonnegative;e.arguments[0]=condition;e.arguments[1]=yes;e.arguments[2]=no;e.dimension=p.expressions.nodes[yes].dimension;return p.expressions.append(e);}
    R min(R a,R b) {return choose(sub(b,a),a,b);}
    R max(R a,R b) {return choose(sub(a,b),a,b);}
    R sum(std::initializer_list<R> values) {R s=c(0);for(R v:values)s=add(s,v);return s;}
    R det(const M&a) {return sum({mul(a[0],sub(mul(a[4],a[8]),mul(a[5],a[7]))),
       scale(mul(a[1],sub(mul(a[3],a[8]),mul(a[5],a[6]))),-1),
       mul(a[2],sub(mul(a[3],a[7]),mul(a[4],a[6])))});}
    M inverse(const M&a) {
        M b{};R d=det(a);
        for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j) {
            unsigned r1=(j+1)%3,r2=(j+2)%3,c1=(i+1)%3,c2=(i+2)%3;
            b[3*i+j]=div(sub(mul(a[3*r1+c1],a[3*r2+c2]),mul(a[3*r1+c2],a[3*r2+c1])),d);
        }
        return b;
    }
    M product(const M&a,const M&b) {
        M q{};
        for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j) {
           q[3*i+j]=sum({mul(a[3*i],b[j]),mul(a[3*i+1],b[3+j]),mul(a[3*i+2],b[6+j])});
        }
        return q;
    }
    V col(const M&a,unsigned j) {return {a[j],a[3+j],a[6+j]};}
    R dot(const V&a,const V&b) {return sum({mul(a[0],b[0]),mul(a[1],b[1]),mul(a[2],b[2])});}
    V cross(const V&a,const V&b) {return {sub(mul(a[1],b[2]),mul(a[2],b[1])),
       sub(mul(a[2],b[0]),mul(a[0],b[2])),sub(mul(a[0],b[1]),mul(a[1],b[0]))};}
    V unit(V a) {R n=sqrt(dot(a,a));for(auto&v:a)v=div(v,n);return a;}
    M outer(const V&a,const V&b) {M q{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)q[3*i+j]=mul(a[i],b[j]);return q;}
    M combine(const M&a,const M&b,double s) {M q{};for(unsigned i=0;i<9;++i)q[i]=add(a[i],scale(b[i],s));return q;}
    R inner(const M&a,const M&b) {R q=c(0);for(unsigned i=0;i<9;++i)q=add(q,mul(a[i],b[i]));return q;}
    unsigned state(const char* name,double initial=0) {
        unsigned i=p.internalState.size();p.internalState.push_back({name,{},initial,InternalState::Transfer::average});
        p.stateUpdateRoots.push_back(NM_INVALID_INDEX);p.stateImplicitRoots.push_back(NM_INVALID_INDEX);return i;
    }
    R parameter(const char* name,double value,Dimension d={}) {
        unsigned i=p.parameters.size();p.parameters.push_back({name,d,value,value,value,false,false});
        R r=var(ExprKind::parameter,i);p.expressions.nodes[r].dimension=d;return r;
    }
};

// The primary paper prints -2 in the ZD-compression bracket. Replacing it
// with -1-1/E7 is an engineering recentering that makes the compressed-side
// limit zero; no author executable source was available to establish that
// alternate as the paper's implementation. Keep both conventions explicit.
enum class CompressionEnergyConvention {
    paper_absolute, // literal printed Eq. 18; compressed-side limit is negative
    recentered,     // zero ZD compression energy at I13 -> 1-, preserving legacy output
};

inline const char* conventionName(CompressionEnergyConvention convention) {
    switch (convention) {
        case CompressionEnergyConvention::paper_absolute: return "paper_absolute";
        case CompressionEnergyConvention::recentered: return "recentered";
    }
    throw std::invalid_argument("unknown Robertsson compression energy convention");
}

struct StrengthHardeningParameters {
    // The source law uses numerical MPa values in its normalized local system.
    std::array<double,12> initialStrengthMPa{};
    std::array<double,12> hardeningAmplitudeMPa{};
    // q(kappa)=A*log(1+B*kappa); B=0 with A>0 is a linear hardening branch.
    std::array<double,12> hardeningRate{};
};

inline StrengthHardeningParameters publishedStrengthHardening() {
    return {{20.4,11.7,9.91,22.0,16.7,9.91,1.0,0.4,2.51,2.51,2.51,2.51},
            {12.2,5.10,6.63,0.0,0.0,6.63,49.0,0.0,0.0,0.0,0.0,0.0},
            {690.0,435.0,334.0,0.0,0.0,334.0,0.0,0.0,0.0,0.0,0.0,0.0}};
}

// Every damage-driving plane has its own reference area in mm^2. The source
// brick recipe supplies equal MD areas for nu=9,10 and equal CD areas for
// nu=11,12; a tetrahedral selected-plane adaptation can supply four different
// element-local areas, but is not source calibration.
inline MaterialProgram makeMaterialFromModeAreas(
    CompressionEnergyConvention energyConvention,
    const std::array<double,4>& selectedPlaneAreaMm2,
    double damageCap,
    bool selectedPlaneCandidate,
    const std::optional<std::array<double,4>>& softeningEnergyJPerM2 = std::nullopt,
    const std::optional<std::array<double,4>>& selectedPlaneLengthM = std::nullopt,
    const std::optional<StrengthHardeningParameters>& strengthHardening = std::nullopt) {
    for (double area : selectedPlaneAreaMm2) {
        if (!(std::isfinite(area) && area > 0.0))
            throw std::invalid_argument(
                "Robertsson selected damage-plane areas must be finite and positive");
    }
    if (!(std::isfinite(damageCap) && damageCap >= 0.0 && damageCap <= 0.96))
        throw std::invalid_argument("invalid Robertsson damage cap");
    const bool hasAnyCandidateInput = softeningEnergyJPerM2.has_value() ||
        selectedPlaneLengthM.has_value() || strengthHardening.has_value();
    const bool hasAllCandidateInputs = softeningEnergyJPerM2.has_value() &&
        selectedPlaneLengthM.has_value() && strengthHardening.has_value();
    if ((selectedPlaneCandidate && !hasAllCandidateInputs) ||
        (!selectedPlaneCandidate && hasAnyCandidateInput))
        throw std::invalid_argument(
            "selected-plane candidate requires explicit Gc, element lengths, strengths, and hardening");
    if (softeningEnergyJPerM2 && selectedPlaneLengthM) {
        for (unsigned i=0;i<4;++i) {
            if (!(std::isfinite((*softeningEnergyJPerM2)[i]) &&
                  (*softeningEnergyJPerM2)[i] > 0.0 &&
                  std::isfinite((*selectedPlaneLengthM)[i]) &&
                  (*selectedPlaneLengthM)[i] > 0.0))
                throw std::invalid_argument(
                    "selected-plane Gc and lengths must be finite and positive");
        }
    }
    const StrengthHardeningParameters modeParameters =
        strengthHardening.value_or(publishedStrengthHardening());
    if (selectedPlaneCandidate) {
        for (unsigned i=0;i<12;++i) {
            if (!(std::isfinite(modeParameters.initialStrengthMPa[i]) &&
                  modeParameters.initialStrengthMPa[i] > 0.0 &&
                  std::isfinite(modeParameters.hardeningAmplitudeMPa[i]) &&
                  modeParameters.hardeningAmplitudeMPa[i] >= 0.0 &&
                  std::isfinite(modeParameters.hardeningRate[i]) &&
                  modeParameters.hardeningRate[i] >= 0.0))
                throw std::invalid_argument(
                    "invalid explicit source damage strength/hardening mode input");
        }
    }
    const std::string convention=conventionName(energyConvention);
    Graph g;g.p.cacheExpressions=true;
    g.p.name="robertsson2023_single_ply_softwood_full_"+convention+
        (selectedPlaneCandidate ? "_tet_selected_plane_candidate" : "");
    g.p.supportedRepresentations={Representation::fem};
    g.p.staticFriction=0;g.p.dynamicFriction=0;
    g.parameter("density",788,{-3,1,0,0});
    R stressUnit=g.parameter("stress_unit",1e6,{-1,1,-2,0});
    std::array<R,4> areaRatio{};
    constexpr const char* areaNames[]={"area_ratio_nu9","area_ratio_nu10",
                                       "area_ratio_nu11","area_ratio_nu12"};
    for (unsigned i=0;i<4;++i)
        areaRatio[i]=g.parameter(areaNames[i],selectedPlaneAreaMm2[i]/0.0011);
    std::array<R,4> softeningEnergyDensity{};
    if (selectedPlaneCandidate) {
        constexpr const char* gcNames[]={"fracture_energy_nu9_J_m2",
                                         "fracture_energy_nu10_J_m2",
                                         "fracture_energy_nu11_J_m2",
                                         "fracture_energy_nu12_J_m2"};
        constexpr const char* lengthNames[]={"selected_plane_length_nu9_m",
                                             "selected_plane_length_nu10_m",
                                             "selected_plane_length_nu11_m",
                                             "selected_plane_length_nu12_m"};
        for (unsigned i=0;i<4;++i) {
            R gc=g.parameter(gcNames[i],(*softeningEnergyJPerM2)[i],
                             {0,1,-2,0});
            R length=g.parameter(lengthNames[i],(*selectedPlaneLengthM)[i],
                                 {1,0,0,0});
            softeningEnergyDensity[i]=g.div(
                g.div(gc,length),stressUnit);
        }
    }
    R cap=g.parameter("damage_cap",damageCap);
    std::array<R,12> initialStrength{},hardeningAmplitude{},hardeningRate{};
    for (unsigned n=0;n<12;++n) {
        if (selectedPlaneCandidate) {
            const std::string suffix = "_nu" + std::to_string(n + 1u);
            initialStrength[n]=g.parameter(
                ("initial_strength" + suffix + "_MPa").c_str(),
                modeParameters.initialStrengthMPa[n]);
            hardeningAmplitude[n]=g.parameter(
                ("hardening_amplitude" + suffix + "_MPa").c_str(),
                modeParameters.hardeningAmplitudeMPa[n]);
            hardeningRate[n]=g.parameter(
                ("hardening_rate" + suffix).c_str(),
                modeParameters.hardeningRate[n]);
        } else {
            initialStrength[n]=g.c(modeParameters.initialStrengthMPa[n]);
            hardeningAmplitude[n]=g.c(modeParameters.hardeningAmplitudeMPa[n]);
            hardeningRate[n]=g.c(modeParameters.hardeningRate[n]);
        }
    }
    constexpr const char* fpNames[]={"fp00","fp01","fp02","fp10","fp11","fp12","fp20","fp21","fp22"};
    constexpr const char* kNames[]={"kappa1","kappa2","kappa3","kappa4","kappa5","kappa6","kappa7","kappa8","kappa9","kappa10","kappa11","kappa12"};
    for(unsigned i=0;i<9;++i)g.state(fpNames[i],i%4==0?1:0);
    for(auto name:kNames)g.state(name);
    g.state("accumulated_multiplier_MPa");
    for(auto name:{"onset_work9_MPa","onset_work10_MPa","onset_work11_MPa","onset_work12_MPa"})g.state(name);
    g.state("damage");
    std::array<R,27> y{},old{};
    for(unsigned i=0;i<27;++i){y[i]=g.var(ExprKind::candidateState,i);old[i]=g.var(ExprKind::internalState,i);}
    M f{},fp{};for(unsigned i=0;i<9;++i){f[i]=g.var(ExprKind::deformation,i);fp[i]=y[i];}
    M fe=g.product(f,g.inverse(fp));
    V v1=g.col(fe,0),v2=g.col(fe,1),v3=g.col(fe,2),n3=g.cross(v1,v2);
    R i11=g.sqrt(g.dot(v1,v1)),i12=g.sqrt(g.dot(v2,v2)),i13=g.sqrt(g.dot(v3,v3));
    R i23=g.sqrt(g.dot(n3,n3)),je=g.det(fe);
    // The printed and recentered forms have identical deformation derivatives
    // on this branch but different absolute energies. Since the full selected
    // psi0 below multiplies (1-damage), its derivative with respect to damage
    // is the corresponding full damage-conjugate force A=Psi0. Keep the
    // convention in the energy itself; stress parity alone is insufficient.
    R zdT=g.scale(g.sub(g.add(i13,g.div(g.c(1),i13)),g.c(2)),28.4);
    constexpr double E6_MPa=0.365;
    constexpr double E7=11.5;
    const double compressionConstant =
        energyConvention==CompressionEnergyConvention::paper_absolute
            ? -2.0 : -1.0-1.0/E7;
    R zdC=g.scale(g.sum({g.pow(i13,2),g.scale(g.exp(g.scale(g.sub(g.pow(i13,2),g.c(1)),-E7)),1/E7),g.c(compressionConstant)}),E6_MPa);
    R zd=g.choose(g.sub(i13,g.c(1)),zdT,zdC);
    R psi=g.sum({g.scale(g.sub(g.add(i11,g.div(g.c(1),i11)),g.c(2)),1690),
       g.scale(g.sub(g.add(i12,g.div(g.c(1),i12)),g.c(2)),292),
       g.scale(g.sum({i11,i12,g.div(g.c(1),i23),g.c(-3)}),1333),zd,
       g.scale(g.sub(g.mul(g.mul(i11,i12),i13),je),78)});
    for(unsigned n=0;n<12;++n) {
        if(modeParameters.hardeningRate[n]>0.0) {
            R x=g.add(g.mul(y[9+n],hardeningRate[n]),g.c(1));
            psi=g.add(psi,g.scale(
                g.sum({g.mul(x,g.log(x)),g.scale(x,-1),g.c(1)}),
                modeParameters.hardeningAmplitudeMPa[n] /
                    modeParameters.hardeningRate[n]));
        } else if(modeParameters.hardeningAmplitudeMPa[n]>0.0) {
            psi=g.add(psi,g.scale(g.mul(hardeningAmplitude[n],
                                      g.pow(y[9+n],2)),0.5));
        }
    }
    g.p.energyRoot=g.mul(stressUnit,g.mul(g.sub(g.c(1),y[26]),psi));
    // Eq A1 and 20, in normalized MPa. ZD branch derivatives below use
    // min/max arguments; at the join the source chooses the tension modulus.
    R p5ZD=g.choose(g.sub(i13,g.c(1)),
       g.scale(g.sub(g.div(g.c(1),i13),g.div(g.c(1),g.pow(i13,3))),28.4),
       g.scale(g.sub(g.c(1),g.exp(g.scale(g.sub(g.pow(i13,2),g.c(1)),-11.5))),0.730));
    M m1=g.outer(v1,v1),m2=g.outer(v2,v2),m3=g.outer(n3,n3),v33=g.outer(v3,v3),tau{};
    // Algebraically collect each modulus before multiplying by MPa. This
    // preserves Eq20 while avoiding cancellation between large E4/E5 terms
    // after each has already been rounded to a floating-point stress.
    R axial1=g.sub(g.div(g.c(1),i11),g.div(g.c(1),g.pow(i11,3)));
    R axial2=g.sub(g.div(g.c(1),i12),g.div(g.c(1),g.pow(i12,3)));
    for(unsigned i=0;i<9;++i) {
        R e4=g.sum({g.div(m1[i],i11),g.div(m2[i],i12),g.div(m3[i],g.pow(i23,3)),
            i%4==0?g.div(g.c(-1),i23):g.c(0)});
        R e5=g.sum({g.mul(g.div(g.mul(i12,i13),i11),m1[i]),
            g.mul(g.div(g.mul(i11,i13),i12),m2[i]),
            g.mul(g.div(g.mul(i11,i12),i13),v33[i]),i%4==0?g.scale(je,-1):g.c(0)});
        tau[i]=g.sum({g.scale(g.mul(axial1,m1[i]),1690),
            g.scale(g.mul(axial2,m2[i]),292),g.scale(e4,1333),g.scale(e5,78),g.mul(p5ZD,v33[i])});
    }
    V u1=g.unit(v1),u2=g.unit(v2),u3=g.unit(n3);
    M u11=g.outer(u1,u1),u22=g.outer(u2,u2),u33=g.outer(u3,u3);
    std::array<M,12> ns{};
    ns[0]=g.combine(u11,u22,-.46);ns[1]=g.combine(u22,u11,-.19);
    for(auto&r:ns[0])r=g.scale(r,1/std::sqrt(1+.46*.46));
    for(auto&r:ns[1])r=g.scale(r,1/std::sqrt(1+.19*.19));
    ns[2]=g.combine(g.outer(u1,u2),g.outer(u2,u1),1);
    for(auto&r:ns[2])r=g.scale(r,1/std::sqrt(2.));
    for(unsigned i=0;i<9;++i){ns[3][i]=g.scale(u11[i],-1);ns[4][i]=g.scale(u22[i],-1);ns[5][i]=g.scale(ns[2][i],-1);ns[6][i]=g.scale(u33[i],-1);ns[7][i]=u33[i];}
    for(unsigned n=8;n<12;++n){ns[n]=g.combine(g.outer(n<10?u1:u2,u3),u33,.7);if(n%2==1)ns[n]=g.combine(g.outer(n<10?u1:u2,u3),u33,-.7);
       for(auto&r:ns[n])r=g.scale(r,(n%2==1?-1:1)/std::sqrt(1+.7*.7));}
    std::array<R,12> omega{},lambda{};
    R yield=g.c(-1),dl=g.sub(y[21],old[21]);
    M flow{};for(auto&r:flow)r=g.c(0);
    M fpInvF=g.product(fp,g.inverse(f));
    for(unsigned n=0;n<12;++n){
       R hard=g.c(0);
       if(modeParameters.hardeningRate[n]>0.0)
           hard=g.mul(hardeningAmplitude[n],g.log(g.add(
               g.mul(y[9+n],hardeningRate[n]),g.c(1))));
       else if(modeParameters.hardeningAmplitudeMPa[n]>0.0)
           hard=g.mul(hardeningAmplitude[n],y[9+n]);
       R strength=g.add(initialStrength[n],hard);
       omega[n]=g.div(g.max(g.inner(tau,ns[n]),g.c(0)),strength);
       lambda[n]=g.div(g.scale(g.pow(omega[n],5),6),strength);
       yield=g.add(yield,g.pow(omega[n],6));
       M transformed=g.product(g.product(fpInvF,ns[n]),f);
       for(unsigned i=0;i<9;++i)flow[i]=g.add(flow[i],g.mul(lambda[n],transformed[i]));
       g.p.stateImplicitRoots[9+n]=g.scale(g.sub(g.sub(y[9+n],old[9+n]),g.mul(dl,g.mul(lambda[n],omega[n]))),10);
    }
    for(unsigned i=0;i<9;++i)g.p.stateImplicitRoots[i]=g.scale(g.sub(g.sub(y[i],old[i]),g.mul(dl,flow[i])),10);
    // Fischer-Burmeister numerical regularization; exact KKT zero set as
    // epsilon tends to zero, not a physical viscosity or fitted parameter.
    R margin=g.scale(yield,-1);
    g.p.stateImplicitRoots[21]=g.scale(g.sub(g.sub(g.sqrt(g.sum({g.pow(dl,2),g.pow(margin,2),g.c(1e-16)})),dl),margin),10);
    std::array<R,4> work{},prevWork{},beta{},dc{};
    R onsetNow=g.c(0),onsetPrev=g.c(0),sumCritical=g.c(0);
    for(unsigned j=0;j<4;++j){work[j]=g.scale(y[17+j],2.51);prevWork[j]=g.scale(old[17+j],2.51);
       beta[j]=g.scale(areaRatio[j],j<2?2.85:3.98);
       dc[j]=g.scale(areaRatio[j],j<2?8:9);
       onsetNow=g.add(onsetNow,g.mul(beta[j],work[j]));onsetPrev=g.add(onsetPrev,g.mul(beta[j],prevWork[j]));sumCritical=g.add(sumCritical,old[22+j]);}
    // Freeze onset mode proportions at the first threshold crossing, with
    // linear interpolation in dissipated work within the accepted increment.
    R theta=g.min(g.c(1),g.max(g.c(0),g.div(g.sub(g.c(1),onsetPrev),g.choose(g.sub(g.sub(onsetNow,onsetPrev),g.c(1e-12)),g.sub(onsetNow,onsetPrev),g.c(1)))));
    R reached=g.choose(g.sub(onsetNow,g.c(1)),g.c(1),g.c(0));
    R already=g.choose(g.sub(sumCritical,g.c(1e-12)),g.c(1),g.c(0));
    R numerator=g.c(0),denominator=g.c(0);
    for(unsigned j=0;j<4;++j){R crossing=g.add(prevWork[j],g.mul(theta,g.sub(work[j],prevWork[j])));
       R critical=g.add(g.mul(already,old[22+j]),g.mul(g.sub(g.c(1),already),g.mul(reached,crossing)));
       g.p.stateImplicitRoots[22+j]=g.scale(g.sub(y[22+j],critical),10);
       numerator=g.add(numerator,g.sub(work[j],y[22+j]));
       if (selectedPlaneCandidate) {
           R modeActive=g.choose(g.sub(y[22+j],g.c(1e-12)),g.c(1),g.c(0));
           denominator=g.add(denominator,
                             g.mul(modeActive,softeningEnergyDensity[j]));
       } else {
           denominator=g.add(denominator,g.mul(dc[j],y[22+j]));
       }
    }
    R alpha=g.min(cap,g.max(old[26],g.mul(reached,g.div(numerator,g.choose(g.sub(denominator,g.c(1e-12)),denominator,g.c(1))))));
    g.p.stateImplicitRoots[26]=g.scale(g.sub(y[26],alpha),10);
    g.p.minimumDeterminant=.05;g.p.maximumDeterminant=20;
    return std::move(g.p);
}

// Source-equivalent area map for the published brick law: the two MD shear
// subsurfaces share the source MD projected area, and the two CD subsurfaces
// share the source CD projected area. The convention is always explicit.
inline MaterialProgram makeMaterial(CompressionEnergyConvention convention,
                                    double areaMDmm2,
                                    double areaCDmm2,
                                    double damageCap) {
    if (!(std::isfinite(areaMDmm2) && areaMDmm2 > 0.0 &&
          std::isfinite(areaCDmm2) && areaCDmm2 > 0.0))
        throw std::invalid_argument(
            "Robertsson MD/CD projected areas must be finite and positive");
    return makeMaterialFromModeAreas(
        convention, {areaMDmm2, areaMDmm2, areaCDmm2, areaCDmm2},
        damageCap, false);
}

// Candidate mapping for a tetrahedral crack-band adaptation. Each of the four
// source damage subsurfaces receives its own actual selected-plane area.
// This is a changed regularization geometry, not the paper's brick calibration.
inline MaterialProgram makeSelectedPlaneCandidate(
    CompressionEnergyConvention convention,
    const std::array<double,4>& selectedPlaneAreaMm2,
    const std::array<double,4>& fractureEnergyJPerM2,
    const std::array<double,4>& selectedPlaneLengthM,
    const StrengthHardeningParameters& strengthHardening,
    double damageCap) {
    return makeMaterialFromModeAreas(convention, selectedPlaneAreaMm2,
                                     damageCap, true, fractureEnergyJPerM2,
                                     selectedPlaneLengthM, strengthHardening);
}

inline MaterialProgram makePublishedReferenceMaterial(
    CompressionEnergyConvention convention) {
    constexpr double sourceReferenceAreaMm2 = 0.0011;
    constexpr double sourceDamageCap = 0.96;
    return makeMaterial(convention, sourceReferenceAreaMm2,
                        sourceReferenceAreaMm2, sourceDamageCap);
}

// Preserve the historical fixture as an explicit, named regression mode.
inline MaterialProgram makeLegacyRecenteredMaterial() {
    return makePublishedReferenceMaterial(CompressionEnergyConvention::recentered);
}
}
