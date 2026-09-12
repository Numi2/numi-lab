#include "numi/matter/detail.hpp"
#include "cardiac_material_reference.hpp"
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {
namespace ref=numi_cardiac_reference;
using ref::Matrix;
using ref::Result;
using numi::matter::MaterialProgram;
void require(bool condition,const std::string& message) {
    if(!condition)throw std::runtime_error(message);
}
std::string precise(double x) {
    std::ostringstream out;out<<std::setprecision(17)<<x;return out.str();
}
double maxAbs(const Matrix& x) {
    double v=0;for(double a:x)v=std::max(v,std::abs(a));return v;
}
double matrixError(const Matrix& a,const Matrix& b,double scale=1) {
    return maxAbs(ref::sub(a,b))/std::max({scale,maxAbs(a),maxAbs(b)});
}
double sourceExpression(const MaterialProgram& material,const Matrix& f) {
    using numi::matter::ExprKind;
    std::vector<double> v;
    for(const auto& e:material.expressions.nodes) {
        const auto a=[&](unsigned i){return v.at(e.arguments[i]);};
        double x=0;
        switch(e.kind) {
        case ExprKind::constant:x=e.constant;break;
        case ExprKind::parameter:x=material.parameters.at(e.index).defaultValue;break;
        case ExprKind::deformation:x=f.at(e.index);break;
        case ExprKind::add:x=a(0)+a(1);break;
        case ExprKind::subtract:x=a(0)-a(1);break;
        case ExprKind::multiply:x=a(0)*a(1);break;
        case ExprKind::divide:x=a(0)/a(1);break;
        case ExprKind::negate:x=-a(0);break;
        case ExprKind::logarithm:x=std::log(a(0));break;
        case ExprKind::exponential:x=std::exp(a(0));break;
        case ExprKind::integerPower:x=std::pow(a(0),e.integer);break;
        default:throw std::runtime_error("unexpected cardiac source expression");
        }
        require(std::isfinite(x),"nonfinite cardiac source expression");v.push_back(x);
    }
    return v.at(material.energyRoot);
}
double bytecode(const numi::matter::ScalarBytecode& program,
                const MaterialProgram& material,const Matrix& f,const Matrix& h) {
    std::vector<double> stack;
    const auto pop=[&](){require(!stack.empty(),"bytecode stack underflow");
        const double v=stack.back();stack.pop_back();return v;};
    for(const auto& op:program.instructions) {
        double v=0;
        switch(op.opcode) {
        case NM_EXPR_CONSTANT:v=op.immediate.x;break;
        case NM_EXPR_PARAMETER:v=material.parameters.at(op.index).defaultValue;break;
        case NM_EXPR_F:v=f.at(op.index);break;
        case NM_EXPR_DF:v=h.at(op.index);break;
        case NM_EXPR_ADD:{double b=pop();v=pop()+b;break;}
        case NM_EXPR_SUBTRACT:{double b=pop();v=pop()-b;break;}
        case NM_EXPR_MULTIPLY:{double b=pop();v=pop()*b;break;}
        case NM_EXPR_DIVIDE:{double b=pop();v=pop()/b;break;}
        case NM_EXPR_NEGATE:v=-pop();break;
        case NM_EXPR_LOG:v=std::log(pop());break;
        case NM_EXPR_EXP:v=std::exp(pop());break;
        case NM_EXPR_POW_INTEGER:v=std::pow(pop(),op.integer);break;
        default:throw std::runtime_error("unexpected cardiac bytecode opcode");
        }
        require(std::isfinite(v),"nonfinite cardiac bytecode");stack.push_back(v);
    }
    require(stack.size()==1,"bytecode stack did not reduce to one value");return stack.front();
}
void set(MaterialProgram& m,const std::string& name,double value) {
    for(auto& p:m.parameters)if(p.name==name){p.defaultValue=value;return;}
    throw std::runtime_error("missing parameter "+name);
}
double get(const MaterialProgram& m,const std::string& name) {
    for(const auto& p:m.parameters)if(p.name==name)return p.defaultValue;
    throw std::runtime_error("missing parameter "+name);
}
std::string diagnostics(const std::vector<numi::matter::Diagnostic>& values) {
    std::string s;for(const auto& d:values)s+=d.message+"\n";return s;
}
Matrix rotation(double angle,unsigned axis) {
    Matrix r=ref::identity;const unsigned i=(axis+1)%3,j=(axis+2)%3;
    r[3*i+i]=r[3*j+j]=std::cos(angle);r[3*i+j]=-std::sin(angle);r[3*j+i]=std::sin(angle);return r;
}
struct Metrics {
    std::size_t states=0,scalarComparisons=0;
    double energyError=0,stressError=0,tangentError=0,stressFdError=0,tangentFdError=0,covarianceError=0;
};
using Oracle=std::function<Result(const Matrix&,const Matrix&)>;
void compare(const MaterialProgram& m,const Oracle& oracle,Metrics& metrics) {
    const auto compiled=numi::matter::detail::compileConstitutive(m,NM_EXPRESSION_STACK_CAPACITY);
    require(compiled.succeeded(),"source constitutive compile failed: "+diagnostics(compiled.diagnostics));
    const Matrix r=ref::multiply(rotation(.53,2),rotation(-.31,1));
    const Matrix h{.17,-.23,.11,.31,-.19,.07,-.13,.29,.21};
    std::vector<Matrix> states{ref::identity,
        Matrix{1.13,0,0,0,.93,0,0,0,1.04},Matrix{.91,.17,0,0,1.06,.12,.04,0,1.02},
        Matrix{1,.35,.05,0,1,0,0,0,1},Matrix{1.2,0,0,0,1/std::sqrt(1.2),0,0,0,1/std::sqrt(1.2)},
        Matrix{.9,0,0,0,.9,0,0,0,.9},Matrix{1.1,0,0,0,1.1,0,0,0,1.1}};
    // Deterministic multiaxial cases; no statistical or biological claim.
    for(unsigned n=1;n<=12;++n) {
        Matrix f=ref::identity;
        for(unsigned i=0;i<9;++i)f[i]+=.13*std::sin(double(17*n+11*i));
        states.push_back(f);
    }
    for(const Matrix& f:states) {
        const Result expected=oracle(f,h);
        const double e=sourceExpression(m,f);
        const double ee=std::abs(e-expected.energy)/std::max(1.,std::abs(expected.energy));
        metrics.energyError=std::max(metrics.energyError,ee);
        require(ee<2e-11,"authored source energy differs from independent tensor formula");
        Matrix stress{},tangent{};
        for(unsigned i=0;i<9;++i) {
            stress[i]=bytecode(compiled.program.stress[i],m,f,h);
            tangent[i]=bytecode(compiled.program.tangentVector[i],m,f,h);
        }
        // Bytecode immediates are FP32. Compare its FP64 evaluation to exact
        // source coefficients with an explicit coefficient-rounding budget.
        const double pe=matrixError(stress,expected.stress,1000);
        const double he=matrixError(tangent,expected.tangent,1000);
        metrics.stressError=std::max(metrics.stressError,pe);
        metrics.tangentError=std::max(metrics.tangentError,he);
        require(pe<3e-6&&he<3e-6,"symbolic source stress/tangent differs from independent FP64 tensor derivative");
        const double step=2e-6;
        Matrix fdStress{};
        for(unsigned i=0;i<9;++i) {
            Matrix unit{};unit[i]=step;
            fdStress[i]=(oracle(ref::add(f,unit),{}).energy-oracle(ref::sub(f,unit),{}).energy)/(2*step);
        }
        const Matrix fdTangent=ref::scale(ref::sub(oracle(ref::add(f,ref::scale(h,step)),{}).stress,
            oracle(ref::sub(f,ref::scale(h,step)),{}).stress),1/(2*step));
        const double pfd=matrixError(fdStress,expected.stress,1000);
        const double hfd=matrixError(fdTangent,expected.tangent,1000);
        metrics.stressFdError=std::max(metrics.stressFdError,pfd);
        metrics.tangentFdError=std::max(metrics.tangentFdError,hfd);
        require(pfd<5e-7&&hfd<5e-7,"independent analytic derivative failed finite difference");
        const Result rotated=oracle(ref::multiply(r,f),ref::multiply(r,h));
        // At an exact rotation, trace(Cbar)-3 can round by one ULP. Use
        // the same explicit 1 kPa floor as the stress/tangent comparison,
        // so multiplying that cancellation by the 1 MPa valve coefficient
        // does not turn an undeformed zero into a false relative failure.
        const double covariance=std::max({matrixError(rotated.stress,ref::multiply(r,expected.stress),1000),
            matrixError(rotated.tangent,ref::multiply(r,expected.tangent),1000),
            std::abs(rotated.energy-expected.energy)/std::max(1000.,std::abs(expected.energy))});
        metrics.covarianceError=std::max(metrics.covarianceError,covariance);
        require(covariance<2e-11,"source law violates superposed spatial-rotation covariance: material="+
            m.name+" state="+std::to_string(metrics.states)+" error="+precise(covariance)+
            " energy="+precise(expected.energy)+" rotated_energy="+precise(rotated.energy));
        // Hessian reciprocity provides a check independent of FD step size.
        const Matrix k{-.11,.21,.03,-.17,.09,.31,.14,-.07,.13};
        const double lhs=ref::contract(k,expected.tangent),rhs=ref::contract(h,oracle(f,k).tangent);
        require(std::abs(lhs-rhs)<2e-11*std::max({1000.,std::abs(lhs),std::abs(rhs)}),
            "source tangent violates major symmetry");
        ++metrics.states;metrics.scalarComparisons+=18;
    }
}
void checkDensity(const MaterialProgram& m) {
    using namespace numi::matter;
    require(get(m,"density")==0,"source asset has an invented density default");
    WorldSource world;world.materials={m};world.gravity={0,0,0};
    ObjectSource object;object.name="synthetic_constitutive_admission_tet";
    object.representation=Representation::fem;object.mixedFEM=false;
    object.deformableContact=false;object.deformableSelfContact=false;
    object.femNodes={{{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,.01}}};
    object.tetrahedra={{{0,1,2,3}}};world.objects={object};
    CompileOptions options;options.maximumRateExponent=0;options.emitSpecializedMetal=false;
    const auto unresolved=compileWorld(world,options);
    require(!unresolved.succeeded(),"unresolved source density was accepted as executable mass");
    require(diagnostics(unresolved.diagnostics).find("physical limits")!=std::string::npos,
        "zero-density rejection did not reach material physical admission");
    // Test-only positive density, explicitly not a sourced myocardial value.
    set(world.materials[0],"density",1000);
    const auto resolved=compileWorld(world,options);
    require(resolved.succeeded(),"synthetic explicit-density non-mixed fixture failed: "+diagnostics(resolved.diagnostics));
}
void frameChecks() {
    const Matrix f{1.16,.17,.03,-.04,.93,.11,.02,-.05,1.04};
    const Matrix h{.17,-.23,.11,.31,-.19,.07,-.13,.29,.21};
    const Matrix q=ref::multiply(rotation(.71,2),rotation(-.37,1));
    const Matrix r=ref::multiply(rotation(-.42,0),rotation(.27,2));
    const auto expected=ref::framedGuccione(f,h,q);
    const auto transformed=ref::framedGuccione(ref::multiply(f,ref::transpose(r)),
        ref::multiply(h,ref::transpose(r)),ref::multiply(r,q));
    require(matrixError(transformed.stress,ref::multiply(expected.stress,ref::transpose(r)),1000)<2e-11,
        "material-frame reference rotation does not covary");
    require(matrixError(transformed.tangent,ref::multiply(expected.tangent,ref::transpose(r)),1000)<2e-11,
        "material-frame tangent reference rotation does not covary");
    const auto sheetRotated=ref::framedGuccione(f,h,ref::multiply(q,rotation(.81,0)));
    require(matrixError(sheetRotated.stress,expected.stress,1000)<2e-11&&
        matrixError(sheetRotated.tangent,expected.tangent,1000)<2e-11,
        "source transverse isotropy depends on sheet rotation about fibre");
    const auto wrong=ref::framedGuccione(f,h,ref::transpose(q));
    require(matrixError(wrong.stress,expected.stress,1000)>1e-3,
        "frame fixture cannot distinguish Q from Q transpose");
    const double step=2e-6;
    const Matrix fd=ref::scale(ref::sub(ref::framedGuccione(ref::add(f,ref::scale(h,step)),{},q).stress,
        ref::framedGuccione(ref::sub(f,ref::scale(h,step)),{},q).stress),1/(2*step));
    require(matrixError(fd,expected.tangent,1000)<5e-7,"framed tangent failed direct global-F finite difference");
    for(const Matrix& invalid:{Matrix{1,0,0,0,1,0,0,0,-1},Matrix{1,.1,0,0,1,0,0,0,1}}) {
        bool rejected=false;try{(void)ref::framedGuccione(f,h,invalid);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"improper/nonorthogonal source frame admitted by oracle");
    }
    for(Matrix invalid:{Matrix{},Matrix{1,0,0,0,1,0,0,0,-1},
                         Matrix{std::numeric_limits<double>::infinity(),0,0,0,1,0,0,0,1}}) {
        bool rejected=false;try{(void)ref::guccione(invalid,h);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"nonphysical deformation admitted by source oracle");
    }
    // The present mixed implementation would eliminate p using logJ+p/k=0
    // then apply -p*cofF. That differs from BOTH source volume potentials.
    // This is a non-equivalence regression, not a claim of mixed admission.
    const double s=1.1,j=s*s*s;
    const Matrix dilation=ref::scale(ref::identity,s);
    const double mixed=1e6*std::log(j)*j/s;
    require(std::abs(ref::guccione(dilation,{}).stress[0]-mixed)>1e4,
        "fixture cannot distinguish source Guccione from current mixed closure");
    require(std::abs(ref::neoHookean(dilation,{}).stress[0]-mixed)>1e4,
        "fixture cannot distinguish source neo-Hookean from current mixed closure");
}
} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==3,"usage: cardiac-material-check GUCCIONE.nmatter NEO_HOOKEAN.nmatter");
        auto guccione=numi::matter::parseMatterFile(argv[1]);
        auto neo=numi::matter::parseMatterFile(argv[2]);
        require(guccione.succeeded(),"Guccione parse failed: "+diagnostics(guccione.diagnostics));
        require(neo.succeeded(),"neo-Hookean parse failed: "+diagnostics(neo.diagnostics));
        for(const auto* m:{&guccione.material,&neo.material})
            require(m->hint==numi::matter::ConstitutiveHint::generic&&m->internalState.empty()&&
                m->dissipationRoot==NM_INVALID_INDEX&&m->mixed.maximumActiveTension==0,
                "passive source law acquired a specialized/state/viscous/active approximation");
        require(get(guccione.material,"a")==1700&&get(guccione.material,"bf")==8&&
            get(guccione.material,"bt")==3&&get(guccione.material,"bfs")==4&&
            get(guccione.material,"kappa")==1e6&&get(neo.material,"kappa")==1e6,
            "source Table B/C parameter mismatch");
        Metrics metrics;
        compare(guccione.material,[](const Matrix& f,const Matrix& h){return ref::guccione(f,h);},metrics);
        for(double c:{7450.,26660.,3700.,1e6}) {
            auto m=neo.material;set(m,"c",c);
            compare(m,[c](const Matrix& f,const Matrix& h){return ref::neoHookean(f,h,c);},metrics);
        }
        checkDensity(guccione.material);checkDensity(neo.material);frameChecks();
        std::cout<<std::setprecision(17)<<"{\"status\":\"pass\",\"physical_steps\":0,\"metal_executed\":false,"
            <<"\"states\":"<<metrics.states<<",\"scalar_comparisons\":"<<metrics.scalarComparisons
            <<",\"maximum_energy_error\":"<<metrics.energyError<<",\"maximum_stress_error\":"<<metrics.stressError
            <<",\"maximum_tangent_error\":"<<metrics.tangentError<<",\"maximum_stress_fd_error\":"<<metrics.stressFdError
            <<",\"maximum_tangent_fd_error\":"<<metrics.tangentFdError<<",\"maximum_covariance_error\":"<<metrics.covarianceError
            <<",\"density_sentinel_rejected\":true,\"mixed_source_equivalent\":false,\"subject_calibrated\":false}\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"cardiac material check failed: "<<error.what()<<'\n';return 1;
    }
}
