#include "numi/matter/detail.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace numi::matter;
using Matrix = std::array<double, 9>;
constexpr Matrix identity{1,0,0,0,1,0,0,0,1};
unsigned checks = 0;

void require(bool ok, const std::string& message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}

std::string number(double x) {
    std::ostringstream out;
    out << std::setprecision(17) << x;
    return out.str();
}

std::string diagnostics(const std::vector<Diagnostic>& values) {
    std::string out;
    for (const auto& value : values) out += value.message + "; ";
    return out;
}

double maxAbs(const Matrix& a) {
    double result = 0.0;
    for (const double x : a) result = std::max(result, std::abs(x));
    return result;
}

double relativeError(const Matrix& a, const Matrix& b) {
    Matrix difference{};
    for (unsigned i = 0; i < 9; ++i) difference[i] = a[i] - b[i];
    return maxAbs(difference) / std::max({1.0, maxAbs(a), maxAbs(b)});
}

Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix c{};
    for (unsigned i = 0; i < 3; ++i) for (unsigned j = 0; j < 3; ++j)
        for (unsigned k = 0; k < 3; ++k) c[3*i+j] += a[3*i+k] * b[3*k+j];
    return c;
}

Matrix transpose(const Matrix& a) {
    Matrix b{};
    for (unsigned i = 0; i < 3; ++i) for (unsigned j = 0; j < 3; ++j)
        b[3*i+j] = a[3*j+i];
    return b;
}

Matrix add(const Matrix& a, const Matrix& b) {
    Matrix c{};
    for (unsigned i = 0; i < 9; ++i) c[i] = a[i] + b[i];
    return c;
}

Matrix rotation(const double angle, const unsigned axis) {
    Matrix r = identity;
    const unsigned i = (axis + 1u) % 3u;
    const unsigned j = (axis + 2u) % 3u;
    r[3*i+i] = r[3*j+j] = std::cos(angle);
    r[3*i+j] = -std::sin(angle);
    r[3*j+i] = std::sin(angle);
    return r;
}

struct OrthotropicConstants {
    // SI values, axes M,C,T. The conventional axial-first interpretation is
    // nu_ij = lateral strain j under axial stress i; the displayed paper table
    // lists symbols but does not independently spell out this definition.
    std::array<double, 3> young{};
    double nuMT = 0.0;
    double nuTC = 0.0;
    double nuCM = 0.0;
    std::array<double, 3> shear{}; // CT, MT, CM
    double density = 0.0;
    Matrix compliance{};
    Matrix normalStiffness{};

    OrthotropicConstants(std::array<double,3> moduli, double mt, double tc,
                         double cm, std::array<double,3> shearModuli,
                         double massDensity)
        : young(moduli), nuMT(mt), nuTC(tc), nuCM(cm), shear(shearModuli),
          density(massDensity) {
        compliance[0] = 1.0 / young[0];
        compliance[4] = 1.0 / young[1];
        compliance[8] = 1.0 / young[2];
        // Apply the published axial-first convention, then fill the symmetric
        // compliance by reciprocity. No symmetric-Poisson approximation.
        compliance[1] = compliance[3] = -nuCM / young[1]; // C load, M strain
        compliance[2] = compliance[6] = -nuMT / young[0]; // M load, T strain
        compliance[5] = compliance[7] = -nuTC / young[2]; // T load, C strain
        const double a = compliance[0], b = compliance[1], c = compliance[2];
        const double d = compliance[4], e = compliance[5], f = compliance[8];
        const double determinant = a*(d*f-e*e) - b*(b*f-c*e) + c*(b*e-c*d);
        require(compliance[0] > 0.0 &&
                compliance[0]*compliance[4]-compliance[1]*compliance[1] > 0.0 &&
                determinant > 0.0,
            "source orthotropic compliance fails Sylvester positive-definiteness checks");
        normalStiffness = {
            (d*f-e*e)/determinant, (c*e-b*f)/determinant, (b*e-c*d)/determinant,
            (c*e-b*f)/determinant, (a*f-c*c)/determinant, (b*c-a*e)/determinant,
            (b*e-c*d)/determinant, (b*c-a*e)/determinant, (a*d-b*b)/determinant};
    }
};

OrthotropicConstants linerConstants() {
    return {{5020e6,1884e6,20e6},.20,.01,.20,{20e6,20e6,1175e6},840.0};
}

OrthotropicConstants mediumOrthotropicConstants() {
    return {{4727e6,2046e6,16e6},.20,.01,.20,{16e6,16e6,1250e6},720.0};
}

struct Law {
    Matrix normalStiffness{};
    Matrix compliance{};
    std::array<double, 3> shear{}; // CT, MT, CM
    double density = 0.0;
};

Law orthotropicLaw(const OrthotropicConstants& source) {
    return {source.normalStiffness, source.compliance, source.shear, source.density};
}

Law mediumRctLaw() {
    constexpr double e = 189.0e6;
    constexpr double nu = 0.30; // Explicit unreported 3-D isotropy assumption.
    constexpr double mu = e / (2.0 * (1.0 + nu));
    constexpr double lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Law law;
    law.normalStiffness = {lambda+2*mu,lambda,lambda,
        lambda,lambda+2*mu,lambda,lambda,lambda,lambda+2*mu};
    law.compliance = {1/e,-nu/e,-nu/e,
        -nu/e,1/e,-nu/e,-nu/e,-nu/e,1/e};
    law.shear = {mu,mu,mu};
    law.density = 720.0;
    return law;
}

struct Result {
    double energy = 0.0;
    Matrix stress{};
    Matrix tangent{};
    Matrix secondPiola{};
};

Result reference(const Law& law, const Matrix& f, const Matrix& h) {
    // Independent FP64 StVK tensor oracle. It does not reuse Matter's AD,
    // generated bytecode or evaluator: P=F S, S=C:E, and
    // DP[H]=H S+F C:sym(F^T H).
    const Matrix c = multiply(transpose(f), f);
    const std::array<double, 3> en{
        0.5*(c[0]-1.0), 0.5*(c[4]-1.0), 0.5*(c[8]-1.0)};
    const std::array<double, 3> gamma{c[5], c[2], c[1]}; // CT, MT, CM
    std::array<double, 3> sn{};
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            sn[i] += law.normalStiffness[3*i+j] * en[j];
    const Matrix second{sn[0],law.shear[2]*c[1],law.shear[1]*c[2],
        law.shear[2]*c[1],sn[1],law.shear[0]*c[5],
        law.shear[1]*c[2],law.shear[0]*c[5],sn[2]};
    Result out;
    out.secondPiola = second;
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            out.energy += 0.5 * en[i] * law.normalStiffness[3*i+j] * en[j];
    for (unsigned i = 0; i < 3; ++i)
        out.energy += 0.5 * law.shear[i] * gamma[i] * gamma[i];
    out.stress = multiply(f, second);

    const Matrix df = multiply(transpose(f), h);
    const Matrix hd = multiply(transpose(h), f);
    const Matrix dc = add(df, hd);
    const std::array<double, 3> den{0.5*dc[0],0.5*dc[4],0.5*dc[8]};
    std::array<double, 3> dsn{};
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            dsn[i] += law.normalStiffness[3*i+j] * den[j];
    const Matrix ds{dsn[0],law.shear[2]*dc[1],law.shear[1]*dc[2],
        law.shear[2]*dc[1],dsn[1],law.shear[0]*dc[5],
        law.shear[1]*dc[2],law.shear[0]*dc[5],dsn[2]};
    out.tangent = add(multiply(h, second), multiply(f, ds));
    return out;
}

double expressionValue(const MaterialProgram& material, const Matrix& f) {
    std::vector<double> values;
    values.reserve(material.expressions.nodes.size());
    for (const auto& expression : material.expressions.nodes) {
        const auto arg = [&](const unsigned i) { return values.at(expression.arguments[i]); };
        double value = 0.0;
        switch (expression.kind) {
        case ExprKind::constant: value = expression.constant; break;
        case ExprKind::parameter: value = material.parameters.at(expression.index).defaultValue; break;
        case ExprKind::deformation: value = f.at(expression.index); break;
        case ExprKind::add: value = arg(0) + arg(1); break;
        case ExprKind::subtract: value = arg(0) - arg(1); break;
        case ExprKind::multiply: value = arg(0) * arg(1); break;
        case ExprKind::divide: value = arg(0) / arg(1); break;
        case ExprKind::negate: value = -arg(0); break;
        case ExprKind::integerPower: value = std::pow(arg(0), expression.integer); break;
        default: throw std::runtime_error("unexpected AST operation in fixed StVK material");
        }
        require(std::isfinite(value), "nonfinite authored energy expression");
        values.push_back(value);
    }
    return values.at(material.energyRoot);
}

double bytecode(const ScalarBytecode& program, const MaterialProgram& material,
                const Matrix& f, const Matrix& h) {
    std::vector<double> stack;
    const auto pop = [&]() {
        require(!stack.empty(), "constitutive bytecode stack underflow");
        const double value = stack.back();
        stack.pop_back();
        return value;
    };
    for (const auto& instruction : program.instructions) {
        double value = 0.0;
        switch (instruction.opcode) {
        case NM_EXPR_CONSTANT: value = instruction.immediate.x; break;
        case NM_EXPR_PARAMETER: value = material.parameters.at(instruction.index).defaultValue; break;
        case NM_EXPR_F: value = f.at(instruction.index); break;
        case NM_EXPR_DF: value = h.at(instruction.index); break;
        case NM_EXPR_ADD: { const double b=pop(); value=pop()+b; break; }
        case NM_EXPR_SUBTRACT: { const double b=pop(); value=pop()-b; break; }
        case NM_EXPR_MULTIPLY: { const double b=pop(); value=pop()*b; break; }
        case NM_EXPR_DIVIDE: { const double b=pop(); value=pop()/b; break; }
        case NM_EXPR_NEGATE: value = -pop(); break;
        case NM_EXPR_POW_INTEGER: value = std::pow(pop(), instruction.integer); break;
        default: throw std::runtime_error("unexpected bytecode operation in fixed StVK material");
        }
        require(std::isfinite(value), "nonfinite constitutive bytecode");
        stack.push_back(value);
    }
    require(stack.size() == 1u, "constitutive bytecode did not reduce to one scalar");
    return stack.front();
}

double parameter(const MaterialProgram& material, const std::string& name) {
    for (const auto& p : material.parameters) if (p.name == name) return p.defaultValue;
    throw std::runtime_error("missing fixed material parameter " + name);
}

void checkFixedMaterial(const MaterialProgram& material, const Law& law) {
    require(!material.parameters.empty(), material.name + " has no parameters");
    for (const auto& p : material.parameters)
        require(!p.identifiable && p.lower == p.defaultValue && p.upper == p.defaultValue,
            material.name + " parameter is not fixed/non-identifiable: " + p.name);
    require(material.internalState.empty() && material.dissipationRoot == NM_INVALID_INDEX,
        material.name + " silently acquired state or dissipation");
    require(material.hint == ConstitutiveHint::generic,
        material.name + " is not the authored generic energy law");
    require(std::abs(parameter(material,"density") - law.density) < 1e-12,
        material.name + " density assumption/source mapping changed");
}

struct Compiled {
    MaterialProgram material;
    ConstitutiveProgram program;
};

Compiled compile(const std::filesystem::path& path, const Law& law) {
    auto parsed = parseMatterFile(path);
    require(parsed.succeeded(), "parse failed for " + path.string() + ": " + diagnostics(parsed.diagnostics));
    checkFixedMaterial(parsed.material, law);
    const auto native = detail::compileConstitutive(parsed.material, NM_EXPRESSION_STACK_CAPACITY);
    require(native.succeeded(), "native constitutive compile failed for " + path.string() + ": " + diagnostics(native.diagnostics));
    return {std::move(parsed.material), native.program};
}

Matrix nativeStress(const Compiled& m, const Matrix& f) {
    Matrix p{};
    const Matrix zero{};
    for (unsigned i = 0; i < 9; ++i)
        p[i] = bytecode(m.program.stress[i], m.material, f, zero);
    return p;
}

Matrix nativeTangent(const Compiled& m, const Matrix& f, const Matrix& h) {
    Matrix dp{};
    for (unsigned i = 0; i < 9; ++i)
        dp[i] = bytecode(m.program.tangentVector[i], m.material, f, h);
    return dp;
}

void checkConstitutive(const Compiled& material, const Law& law) {
    const Matrix zero{};
    const Matrix q = multiply(rotation(0.41,2), rotation(-0.23,1));
    const Matrix stretch{1.08,0.03,-0.01, 0.02,0.96,0.04, -0.01,0.02,1.025};
    const std::array<Matrix,4> states{identity, stretch, multiply(q,stretch), multiply(q,identity)};
    double maxEnergyError = 0.0, maxStressError = 0.0, maxTangentError = 0.0;
    double maxTangentFdError = 0.0, maxEnergyGradientError = 0.0;
    for (std::size_t state = 0; state < states.size(); ++state) {
        const Matrix& f = states[state];
        const Result expected = reference(law, f, zero);
        const double energy = expressionValue(material.material, f);
        maxEnergyError = std::max(maxEnergyError,
            std::abs(energy-expected.energy)/std::max({1.0,std::abs(energy),std::abs(expected.energy)}));
        require(maxEnergyError < 2e-11,
            material.material.name + " authored energy differs from independent tensor energy: "+number(maxEnergyError));
        const Matrix stress = nativeStress(material,f);
        maxStressError = std::max(maxStressError,relativeError(stress,expected.stress));
        require(maxStressError < 1e-5,
            material.material.name + " compiled stress differs from independent tensor stress: "+number(maxStressError));
        Matrix h{.17,-.23,.11,.31,-.19,.07,-.13,.29,.21};
        const Result tangentExpected = reference(law,f,h);
        const Matrix tangent = nativeTangent(material,f,h);
        maxTangentError = std::max(maxTangentError,relativeError(tangent,tangentExpected.tangent));
        require(maxTangentError < 1e-5,
            material.material.name + " compiled tangent differs from independent tensor tangent: "+number(maxTangentError));

        constexpr double step = 2e-6;
        Matrix plus=f,minus=f;
        for(unsigned i=0;i<9;++i){plus[i]+=step*h[i];minus[i]-=step*h[i];}
        Matrix fd{};
        const Matrix pPlus=nativeStress(material,plus),pMinus=nativeStress(material,minus);
        for(unsigned i=0;i<9;++i)fd[i]=(pPlus[i]-pMinus[i])/(2*step);
        maxTangentFdError=std::max(maxTangentFdError,relativeError(fd,tangent));
        require(maxTangentFdError < 3e-5,
            material.material.name + " compiled stress finite difference disagrees with compiled tangent: "+number(maxTangentFdError));

        if (state == 1u) {
            constexpr double energyStep=2e-7;
            for(unsigned i=0;i<9;++i) {
                Matrix fp=f,fm=f;fp[i]+=energyStep;fm[i]-=energyStep;
                const double fdEnergy=(expressionValue(material.material,fp)-expressionValue(material.material,fm))/(2*energyStep);
                maxEnergyGradientError=std::max(maxEnergyGradientError,
                    std::abs(fdEnergy-stress[i])/std::max({1.0,std::abs(fdEnergy),std::abs(stress[i])}));
            }
        }
    }
    require(maxEnergyGradientError < 3e-5,
        material.material.name + " finite-difference energy gradient disagrees with compiled stress: "+number(maxEnergyGradientError));

    const Matrix rotationOnly=states[3];
    const Matrix zeroStress=nativeStress(material,rotationOnly);
    require(std::abs(expressionValue(material.material,rotationOnly)) < 1e-4 && maxAbs(zeroStress) < 1e-2,
        material.material.name + " has nonzero energy/stress under rigid rotation");
    const Result base=reference(law,stretch,zero);
    const Matrix rotatedF=multiply(q,stretch);
    const Matrix rotatedP=nativeStress(material,rotatedF);
    require(relativeError(rotatedP,multiply(q,base.stress)) < 1e-5,
        material.material.name + " violates spatial-rotation stress objectivity");
    const Matrix h{.17,-.23,.11,.31,-.19,.07,-.13,.29,.21};
    const Matrix rotatedH=multiply(q,h);
    require(relativeError(nativeTangent(material,rotatedF,rotatedH),
                multiply(q,reference(law,stretch,h).tangent)) < 1e-5,
        material.material.name + " violates spatial-rotation tangent objectivity");

    const Matrix atIdentity=nativeStress(material,identity);
    require(std::abs(expressionValue(material.material,identity)) < 1e-10 && maxAbs(atIdentity) < 1e-3,
        material.material.name + " has nonzero reference energy/stress");
    std::cout << material.material.name << " energy_err=" << maxEnergyError
              << " stress_err=" << maxStressError << " tangent_err=" << maxTangentError
              << " tangent_fd_err=" << maxTangentFdError
              << " energy_gradient_fd_err=" << maxEnergyGradientError << '\n';
}

void checkUniaxialFreeSides(const Compiled& material, const Law& law) {
    for (unsigned axis=0; axis<3; ++axis) {
        constexpr double axialGreenStrain=1e-5;
        std::array<double,3> green{};
        green[axis]=axialGreenStrain;
        const double axialStress=axialGreenStrain/law.compliance[3*axis+axis];
        for (unsigned j=0;j<3;++j) if(j!=axis)
            green[j]=law.compliance[3*j+axis]*axialStress;
        Matrix f{};
        for(unsigned i=0;i<3;++i)f[3*i+i]=std::sqrt(1.0+2.0*green[i]);
        const Matrix p=nativeStress(material,f);
        const double recoveredStress=p[3*axis+axis]/f[3*axis+axis];
        const double modulus=recoveredStress/green[axis];
        const double expected=1.0/law.compliance[3*axis+axis];
        require(std::abs(modulus-expected)/expected<2e-5,
            material.material.name + " free-sided axial modulus mismatch axis="+std::to_string(axis));
        for(unsigned j=0;j<3;++j)if(j!=axis)
            require(std::abs(p[3*j+j]/f[3*j+j]) < 30.0,
                material.material.name + " free lateral stress is not zero axis="+std::to_string(axis));
    }

    const Matrix hM{0,1,0,0,0,0,0,0,0};
    const Matrix hT{0,0,1,0,0,0,0,0,0};
    const Matrix hC{0,0,0,0,0,1,0,0,0};
    const double gCM=nativeTangent(material,identity,hM)[1];
    const double gMT=nativeTangent(material,identity,hT)[2];
    const double gCT=nativeTangent(material,identity,hC)[5];
    require(std::abs(gCM-law.shear[2])/law.shear[2]<2e-5 &&
            std::abs(gMT-law.shear[1])/law.shear[1]<2e-5 &&
            std::abs(gCT-law.shear[0])/law.shear[0]<2e-5,
        material.material.name + " identity shear tangent does not match authored engineering moduli");
}

void checkPaperCoefficients(const Compiled& material,
                            const OrthotropicConstants& source) {
    const auto closeMPa=[&](const char* name,const double expected) {
        const double actual=parameter(material.material,name);
        require(std::abs(actual-expected*1e6)/std::max(1.0,expected*1e6)<2e-12,
            material.material.name+" coefficient is not inverse of source compliance: "+name+
            " actual_Pa="+number(actual)+" expected_Pa="+number(expected*1e6));
    };
    closeMPa("c_MM",source.normalStiffness[0]/1e6);
    closeMPa("c_CC",source.normalStiffness[4]/1e6);
    closeMPa("c_TT",source.normalStiffness[8]/1e6);
    closeMPa("c_MC",source.normalStiffness[1]/1e6);
    closeMPa("c_MT",source.normalStiffness[2]/1e6);
    closeMPa("c_CT",source.normalStiffness[5]/1e6);
    closeMPa("g_CT",source.shear[0]/1e6);
    closeMPa("g_MT",source.shear[1]/1e6);
    closeMPa("g_CM",source.shear[2]/1e6);
    const double nuTM=-source.compliance[2]/source.compliance[8];
    const double nuCT=-source.compliance[5]/source.compliance[4];
    const double nuMC=-source.compliance[1]/source.compliance[0];
    require(std::abs(nuTM-source.nuMT*source.young[2]/source.young[0])<1e-14 &&
            std::abs(nuCT-source.nuTC*source.young[1]/source.young[2])<1e-14 &&
            std::abs(nuMC-source.nuCM*source.young[0]/source.young[1])<1e-14,
        material.material.name+" reciprocal Poisson ratios were not derived through symmetric compliance");
    std::cout << material.material.name << " reciprocal_nu_TM=" << nuTM
              << " nu_CT=" << nuCT << " nu_MC=" << nuMC << '\n';
}

void checkRegionalNativeCompile(const Compiled& liner, const Compiled& medium) {
    WorldSource source;
    source.materials={liner.material,medium.material};
    source.gravity={0,0,0};
    source.frameTimestep=1e-7;
    ObjectSource object;
    object.name="synthetic_cardboard_material_admission";
    object.representation=Representation::fem;
    object.mixedFEM=false;
    object.deformableContact=false;
    object.deformableSelfContact=false;
    object.materialIndex=0;
    object.femNodes={{{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,.01}},{{0,0,-.01}}};
    object.tetrahedra={{{0,1,2,3}},{{0,2,1,4}}};
    object.femMaterialIndices={0,1};
    object.femMaterialSourceIdentity={0x1a2b3c4d5e6f7788ull,0x99aabbccddeeff00ull,
        0x1029384756abcdefull,0x8fedcba654321001ull};
    source.objects.push_back(object);
    CompileOptions options;
    options.maximumRateExponent=0;
    options.emitSpecializedMetal=false;
    const auto cooked=compileWorld(source,options);
    require(cooked.succeeded(),"native regional FEM compile failed: "+diagnostics(cooked.diagnostics));
    require(cooked.world.fem.tetrahedra.size()==2u &&
            cooked.world.fem.tetrahedra[0].identity.x==0u &&
            cooked.world.fem.tetrahedra[1].identity.x==1u,
        "native regional compile changed the per-element liner/medium ownership");
    const auto package=std::filesystem::temp_directory_path()/(
        "numi-cardboard-material-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".nmatterpack");
    std::string error;
    require(writePackage(cooked,package,&error),"native package write failed: "+error);
    CompiledWorld decoded;
    require(readPackage(package,decoded,nullptr,&error),"native package read failed: "+error);
    std::filesystem::remove(package);
    require(decoded.fingerprint==cooked.world.fingerprint &&
            decoded.physicsFingerprint==cooked.world.physicsFingerprint &&
            decoded.fem.tetrahedra.size()==2u &&
            decoded.fem.tetrahedra[0].identity.x==0u &&
            decoded.fem.tetrahedra[1].identity.x==1u,
        "native regional material assignment changed across package round-trip");
    std::cout << "regional_native_compile=pass package_roundtrip=pass physical_steps=0 "
              << "anatomical_or_cardboard_qualification=false\n";
}

} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==4,"usage: cardboard-material-check LINER.nmatter MEDIUM_ORTHOTROPIC.nmatter MEDIUM_RCT.nmatter");
        const OrthotropicConstants linerSource=linerConstants();
        const OrthotropicConstants mediumSource=mediumOrthotropicConstants();
        const Law linerLawValue=orthotropicLaw(linerSource);
        const Law mediumLawValue=orthotropicLaw(mediumSource);
        const Law rctLawValue=mediumRctLaw();
        const auto liner=compile(argv[1],linerLawValue);
        const auto medium=compile(argv[2],mediumLawValue);
        const auto rct=compile(argv[3],rctLawValue);
        require(liner.material.staticFriction==medium.material.staticFriction &&
                liner.material.dynamicFriction==medium.material.dynamicFriction &&
                liner.material.restitution==medium.material.restitution &&
                liner.material.adhesion==medium.material.adhesion &&
                liner.material.staticFriction==rct.material.staticFriction &&
                liner.material.dynamicFriction==rct.material.dynamicFriction &&
                liner.material.restitution==rct.material.restitution &&
                liner.material.adhesion==rct.material.adhesion,
            "regional material interface response differs between liner and medium");
        require(liner.material.staticFriction==0.0 && liner.material.dynamicFriction==0.0 &&
                liner.material.restitution==0.0 && liner.material.adhesion==0.0,
            "shared frictionless, uncalibrated regional interface assumption changed");
        checkPaperCoefficients(liner,linerSource);
        checkPaperCoefficients(medium,mediumSource);
        checkConstitutive(liner,linerLawValue);
        checkConstitutive(medium,mediumLawValue);
        checkConstitutive(rct,rctLawValue);
        checkUniaxialFreeSides(liner,linerLawValue);
        checkUniaxialFreeSides(medium,mediumLawValue);
        checkUniaxialFreeSides(rct,rctLawValue);
        checkRegionalNativeCompile(liner,medium);
        checkRegionalNativeCompile(liner,rct);
        std::cout << "PASS cardboard_material_check checks=" << checks
                  << " liner_and_orthotropic_medium_source_constants=true "
                  << " rct_medium_nu_assumption=true density_from_source_geometry=true "
                  << "yield_and_plasticity_implemented=false physical_steps=0\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "FAIL cardboard_material_check checks=" << checks
                  << " error=" << error.what() << '\n';
        return 1;
    }
}
