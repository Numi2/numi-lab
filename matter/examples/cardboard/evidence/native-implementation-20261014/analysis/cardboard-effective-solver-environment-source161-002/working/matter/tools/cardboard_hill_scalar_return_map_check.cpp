#include "cardboard_hill_scalar_return_map.hpp"
#include "cardboard_paper_reference.hpp"
#include "cardboard_solver_reference.hpp"
#include "numi/matter/hill_compensated_stress.h"
#include "numi/matter/hill_compensated_system.h"
#include "numi/matter/hill_scalar_initializer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
namespace hill=numi::cardboard::hill_scalar;
namespace paper=numi_cardboard_paper;
namespace solver=numi_cardboard_solver;

struct CapturedParameters {
    std::string name;
    std::array<float,20> parameterTable{};
    hill::Parameters<float> fp32;
    paper::Parameters fp64;
    paper::Parameters fp64ExactInput;
};

struct CapturedRow {
    std::string key;
    std::string material;
    paper::Matrix F{};
    paper::State accepted{};
    paper::State capturedCandidate{};
};

struct Metrics {
    unsigned count=0;
    unsigned scalarFailures=0;
    unsigned yielded=0;
    unsigned elastic=0;
    double maxReferenceResidual=0.0;
    double maxCapturedCandidateResidual=0.0;
    double maxStateError=0.0;
    double maxScalarResidual=0.0;
    double maxFBResidual=0.0;
    double maxStressSolveResidual=0.0;
    double maxFlowResidual=0.0;
    double maxFP32SourceResidual=0.0;
    double maxFP32TrialStressVsFP64Source=0.0;
    double maxRecordedOracleDifference=0.0;
    double maxExactInputOracleResidual=0.0;
    double maxExactInputFP64MapResidual=0.0;
    double maxExactInputFP64MapVsOracle=0.0;
    std::string worstReferenceKey;
    std::string worstCandidateKey;
    std::string worstStateKey;
};

struct SweepMetrics {
    unsigned count=0;
    double minTrialQ=std::numeric_limits<double>::infinity();
    double maxTrialQ=0.0;
    double maxLambdaIncrement=0.0;
    double maxFP32SourceResidual=0.0;
    double maxFP32ToFP64StateDifference=0.0;
    double maxFP64SourceResidual=0.0;
    double maxFP32FischerBurmeisterResidual=0.0;
};

struct CompensatedMetrics {
    double maxOrdinaryStressError=0.0;
    double maxCompensatedStressError=0.0;
    double sumSquaredOrdinaryStressError=0.0;
    double sumSquaredCompensatedStressError=0.0;
    unsigned components=0;
    double maxTangentError=0.0;
    double sumSquaredTangentError=0.0;
    double maxDirectionalError=0.0;
    double maxTangentActionVsDirect=0.0;
    unsigned tangentComponents=0;
    unsigned directionalComponents=0;
};

struct ElasticInitializerMetrics {
    unsigned totalRows=0;
    unsigned qAtOrBelowOne=0;
    unsigned qAboveOne=0;
    unsigned floatQFalseElastic=0;
    unsigned nearYieldGuardRows=0;
    unsigned totalScalarIterations=0;
    unsigned maxScalarIterations=0;
    unsigned elasticScalarIterations=0;
    unsigned elasticMaxScalarIterations=0;
    unsigned elasticCandidateZeroFloatFB=0;
    double maxElasticCandidateFloatFB=0.0;
    double maxElasticCandidateExactFB=0.0;
    double maxElasticFullMapLambdaIncrement=0.0;
    double maxElasticFullMapPlasticIncrement=0.0;
    double maxFloatVsExactQ=0.0;
    double minElasticGap=std::numeric_limits<double>::infinity();
};

struct ImplicitSystemMetrics {
    double maxEquivalentStressError=0.0;
    double sumSquaredEquivalentStressError=0.0;
    double maxRelativeJacobianError=0.0;
    double maxResidualError=0.0;
    double maxJacobianError=0.0;
    double maxDeformationActionError=0.0;
    double sumSquaredResidualError=0.0;
    double sumSquaredJacobianError=0.0;
    double sumSquaredDeformationActionError=0.0;
    unsigned residualComponents=0;
    unsigned jacobianComponents=0;
    unsigned deformationActionComponents=0;
    unsigned equivalentStressComponents=0;
};

using SourceStates=std::unordered_map<std::string,paper::State>;

void require(bool condition,const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<CapturedParameters> readParameters(const std::string& path) {
    std::ifstream input(path);
    require(input.good(),"cannot open captured material parameter table: "+path);
    std::vector<CapturedParameters> result;
    std::string line;
    while (std::getline(input,line)) {
        if (line.empty()) continue;
        std::istringstream row(line);
        CapturedParameters p;
        std::array<double,20> v{};
        require(static_cast<bool>(row>>p.name),"bad captured parameter row");
        for (double& value:v) require(static_cast<bool>(row>>value),
            "captured material row does not have exactly 20 parameter values: "+p.name);
        for (double value:v) require(static_cast<double>(static_cast<float>(value))==value,
            "captured material table is not exact widened-FP32 input: "+p.name);
        for (unsigned i=0;i<20;++i) p.parameterTable[i]=static_cast<float>(v[i]);
        std::string extra;
        require(!(row>>extra),"captured material row has trailing fields: "+p.name);
        p.fp32.stiffness={static_cast<float>(v[1]),static_cast<float>(v[2]),
            static_cast<float>(v[3]),static_cast<float>(v[4]),static_cast<float>(v[5]),
            static_cast<float>(v[6])};
        p.fp32.shear={static_cast<float>(v[7]),static_cast<float>(v[8]),static_cast<float>(v[9])};
        p.fp32.sigmaReference=static_cast<float>(v[10]);
        p.fp32.hill={static_cast<float>(v[11]),static_cast<float>(v[12]),
            static_cast<float>(v[13]),static_cast<float>(v[14]),static_cast<float>(v[15]),
            static_cast<float>(v[16])};
        p.fp32.qFloor=static_cast<float>(v[17]);
        p.fp32.fischerBurmeisterEpsilon=static_cast<float>(v[18]);

        const auto capturedFloat=[&](unsigned i) {
            return static_cast<double>(static_cast<float>(v[i]));
        };
        p.fp64.name=p.name=="liner"?paper::liner.name:paper::medium.name;
        p.fp64.density=capturedFloat(0);
        p.fp64.c={capturedFloat(1)/1.0e6,capturedFloat(2)/1.0e6,capturedFloat(3)/1.0e6,
            capturedFloat(4)/1.0e6,capturedFloat(5)/1.0e6,capturedFloat(6)/1.0e6};
        p.fp64.shear={capturedFloat(7)/1.0e6,capturedFloat(8)/1.0e6,capturedFloat(9)/1.0e6};
        p.fp64.compressiveYield={0,0,0};
        p.fp64.tau12=0; p.fp64.tau13=0; p.fp64.tau23=0;
        p.fp64.sigmaReference=capturedFloat(10)/1.0e6;
        p.fp64.hill={capturedFloat(11),capturedFloat(12),capturedFloat(13),capturedFloat(14),
            capturedFloat(15),capturedFloat(16)};
        p.fp64.qFloor=capturedFloat(17);
        p.fp64.fischerBurmeisterEpsilon=capturedFloat(18);
        p.fp64ExactInput=p.fp64;
        p.fp64ExactInput.density=v[0];
        p.fp64ExactInput.c={v[1]/1.0e6,v[2]/1.0e6,v[3]/1.0e6,
            v[4]/1.0e6,v[5]/1.0e6,v[6]/1.0e6};
        p.fp64ExactInput.shear={v[7]/1.0e6,v[8]/1.0e6,v[9]/1.0e6};
        p.fp64ExactInput.sigmaReference=v[10]/1.0e6;
        p.fp64ExactInput.hill={v[11],v[12],v[13],v[14],v[15],v[16]};
        p.fp64ExactInput.qFloor=v[17];
        p.fp64ExactInput.fischerBurmeisterEpsilon=v[18];
        result.push_back(p);
    }
    require(result.size()==2,"expected exactly the captured liner and medium parameter rows");
    return result;
}

CapturedRow parseInputRow(const std::string& line) {
    std::istringstream row(line);
    CapturedRow result;
    require(static_cast<bool>(row>>result.key>>result.material),"bad replay row prefix");
    for (double& value:result.F) {
        require(static_cast<bool>(row>>value),"replay row missing F components: "+result.key);
        require(static_cast<double>(static_cast<float>(value))==value,
            "F component is not exact widened-FP32 input: "+result.key);
    }
    // Captured direction H is deliberately consumed but does not enter the
    // constitutive projection at this accepted material state.
    for (unsigned i=0;i<9;++i) {
        double unused=0;
        require(static_cast<bool>(row>>unused),"replay row missing direction components: "+result.key);
        require(static_cast<double>(static_cast<float>(unused))==unused,
            "direction component is not exact widened-FP32 input: "+result.key);
    }
    for (double& value:result.accepted) {
        require(static_cast<bool>(row>>value),"replay row missing accepted 13-state history: "+result.key);
        require(static_cast<double>(static_cast<float>(value))==value,
            "accepted history is not exact widened-FP32 input: "+result.key);
    }
    for (double& value:result.capturedCandidate) {
        require(static_cast<bool>(row>>value),"replay row missing captured candidate state: "+result.key);
        require(static_cast<double>(static_cast<float>(value))==value,
            "candidate state is not exact widened-FP32 input: "+result.key);
    }
    std::string extra;
    require(!(row>>extra),"replay row has trailing fields: "+result.key);
    return result;
}

SourceStates readSourceStates(const std::string& path) {
    std::ifstream input(path);
    require(input.good(),"cannot open independent FP64 source replay: "+path);
    SourceStates result;
    std::string line;
    while (std::getline(input,line)) {
        if (line.empty()) continue;
        const auto keyStart=line.find("\"key\":\"");
        const auto stateStart=line.find("\"oracle_state\":[");
        require(keyStart!=std::string::npos&&stateStart!=std::string::npos,
            "source replay row missing key/oracle_state");
        require(line.find("\"input_contract\":\"exact_widened_fp32\"")!=std::string::npos,
            "source replay does not declare exact_widened_fp32 input contract");
        const auto keyValueStart=keyStart+7;
        const auto keyEnd=line.find('"',keyValueStart);
        require(keyEnd!=std::string::npos,"unterminated source replay key");
        const std::string key=line.substr(keyValueStart,keyEnd-keyValueStart);
        const auto valuesStart=stateStart+std::string("\"oracle_state\":[").size();
        const char* cursor=line.c_str()+valuesStart;
        paper::State state{};
        for (double& value:state) {
            while (*cursor==' '||*cursor=='\t') ++cursor;
            char* end=nullptr;
            value=std::strtod(cursor,&end);
            require(end!=cursor&&std::isfinite(value),"invalid source replay state at "+key);
            cursor=end;
            while (*cursor==' '||*cursor=='\t') ++cursor;
            if (&value!=&state.back()) {
                require(*cursor==',',"source replay state has fewer than 13 values at "+key);
                ++cursor;
            }
        }
        require(result.emplace(key,state).second,"duplicate source replay key: "+key);
    }
    require(result.size()==14400,"independent source replay did not contain 14,400 states");
    return result;
}

const CapturedParameters& findParameters(const std::vector<CapturedParameters>& values,
                                         const std::string& material) {
    for (const auto& value:values) if (value.name==material) return value;
    throw std::runtime_error("unknown replay material "+material);
}

template <class T>
double maxStateDifference(const paper::State& a,const hill::State13<T>& b) {
    double result=0.0;
    for (unsigned i=0;i<13;++i) result=std::max(result,std::abs(a[i]-static_cast<double>(b[i])));
    return result;
}

double maxResidual(const paper::State& state,const paper::Parameters& p,
                   const paper::Matrix& F,const paper::State& accepted) {
    return paper::residualNorm(paper::residual(p,F,accepted,state));
}

void requirePortableBitMatch(const CapturedParameters& parameters,
    const hill::State13<float>& accepted,const hill::Vector6<float>& trialStress,
    const hill::State13<float>& expected,const std::string& caseName) {
    float candidate[13];
    for (float& value:candidate) value=-12345.0f;
    const int status=nmHillScalarInitializeCandidate(parameters.parameterTable.data(),
        accepted.data(),trialStress.data(),candidate);
    require(status==NM_HILL_SCALAR_INITIALIZER_SUCCESS,
        "portable shared initializer failed "+caseName+" status="+std::to_string(status));
    for (unsigned i=0;i<13;++i) {
        if (std::bit_cast<std::uint32_t>(candidate[i])!=
            std::bit_cast<std::uint32_t>(expected[i])) {
            std::ostringstream message;
            hill::Vector6<float> portableStress{},templateStress{};
            for (unsigned j=0;j<6;++j) {
                portableStress[j]=candidate[6+j];
                templateStress[j]=expected[6+j];
            }
            const float portableQ=hill::hillEquivalent(parameters.fp32,portableStress);
            const float templateQ=hill::hillEquivalent(parameters.fp32,templateStress);
            message<<"portable shared initializer differs bitwise from float template at "
                <<caseName<<" state["<<i<<"] portable="<<std::setprecision(9)<<candidate[i]
                <<" template="<<expected[i]<<" portable_bits=0x"<<std::hex
                <<std::bit_cast<std::uint32_t>(candidate[i])<<" template_bits=0x"
                <<std::bit_cast<std::uint32_t>(expected[i])<<std::dec
                <<" q(port/template)="<<portableQ<<"/"<<templateQ
                <<" dlambda(port/template)="<<(candidate[12]-accepted[12])<<"/"
                <<(expected[12]-accepted[12])
                <<" stressdiff="<<std::setprecision(9)<<candidate[6+i%6]-expected[6+i%6];
            require(false,message.str());
        }
    }
}

std::array<double,54> exactSourceTangent(const paper::Parameters& p,
    const std::array<float,9>& F) {
    std::array<double,54> tangent{};
    const double inverseReference=1.0/p.sigmaReference;
    for (unsigned r=0;r<3;++r) {
        const unsigned i0=3*r,i1=i0+1,i2=i0+2;
        tangent[0*9+i0]=p.c[0]*F[i0]*inverseReference;
        tangent[0*9+i1]=p.c[3]*F[i1]*inverseReference;
        tangent[0*9+i2]=p.c[4]*F[i2]*inverseReference;
        tangent[1*9+i0]=p.c[3]*F[i0]*inverseReference;
        tangent[1*9+i1]=p.c[1]*F[i1]*inverseReference;
        tangent[1*9+i2]=p.c[5]*F[i2]*inverseReference;
        tangent[2*9+i0]=p.c[4]*F[i0]*inverseReference;
        tangent[2*9+i1]=p.c[5]*F[i1]*inverseReference;
        tangent[2*9+i2]=p.c[2]*F[i2]*inverseReference;
        tangent[3*9+i1]=p.shear[2]*F[i2]*inverseReference;
        tangent[3*9+i2]=p.shear[2]*F[i1]*inverseReference;
        tangent[4*9+i0]=p.shear[1]*F[i2]*inverseReference;
        tangent[4*9+i2]=p.shear[1]*F[i0]*inverseReference;
        tangent[5*9+i0]=p.shear[0]*F[i1]*inverseReference;
        tangent[5*9+i1]=p.shear[0]*F[i0]*inverseReference;
    }
    return tangent;
}

void checkPortableFailureSemantics(const CapturedParameters& parameters) {
    std::array<float,13> accepted{};
    std::array<float,6> trial{};
    float candidate[13];
    const auto checkFailure=[&](std::array<float,20> table,
                                std::array<float,6> stress,
                                int expectedStatus,const char* label) {
        for (unsigned i=0;i<13;++i) candidate[i]=1234.5f+float(i);
        float before[13];
        std::memcpy(before,candidate,sizeof(before));
        const int status=nmHillScalarInitializeCandidate(table.data(),accepted.data(),
            stress.data(),candidate);
        require(status==expectedStatus,
            std::string("portable initializer returned wrong ")+label+" status="+
            std::to_string(status));
        require(std::memcmp(before,candidate,sizeof(before))==0,
            std::string("portable initializer wrote output on ")+label+" failure");
    };
    auto invalidParameters=parameters.parameterTable;
    invalidParameters[10]=0.0f;
    checkFailure(invalidParameters,trial,NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT,
        "invalid-input");
    auto nonfiniteStress=trial;
    nonfiniteStress[0]=std::numeric_limits<float>::infinity();
    checkFailure(parameters.parameterTable,nonfiniteStress,
        NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT,"nonfinite-input");
    auto overflowingStress=trial;
    overflowingStress[0]=std::numeric_limits<float>::max();
    checkFailure(parameters.parameterTable,overflowingStress,
        NM_HILL_SCALAR_INITIALIZER_NONFINITE,"overflow");
}

void checkCompensatedFailureSemantics(const CapturedParameters& parameters) {
    std::array<float,13> accepted{};
    std::array<float,9> F{1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f};
    std::array<float,9> direction{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    std::array<float,20> invalidParameters=parameters.parameterTable;
    invalidParameters[10]=0.0f;
    float stress[6],stressBefore[6],tangent[54],tangentBefore[54],direct[6],directBefore[6];
    float residual[13],residualBefore[13],jacobian[169],jacobianBefore[169];
    float action[13],actionBefore[13];
    for (unsigned i=0;i<6;++i) stress[i]=stressBefore[i]=100.0f+float(i);
    for (unsigned i=0;i<54;++i) tangent[i]=tangentBefore[i]=200.0f+float(i);
    for (unsigned i=0;i<6;++i) direct[i]=directBefore[i]=300.0f+float(i);
    for (unsigned i=0;i<13;++i) {
        residual[i]=residualBefore[i]=400.0f+float(i);
        action[i]=actionBefore[i]=500.0f+float(i);
    }
    for (unsigned i=0;i<169;++i) jacobian[i]=jacobianBefore[i]=600.0f+float(i);
    require(nmHillCompensatedTrialStress(invalidParameters.data(),F.data(),accepted.data(),stress)
            ==NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT,
        "compensated stress did not reject invalid sigmaReference");
    require(std::memcmp(stress,stressBefore,sizeof(stress))==0,
        "compensated stress wrote output on invalid input");
    require(nmHillCompensatedTrialStressAndTangent(invalidParameters.data(),F.data(),accepted.data(),
            stress,tangent)==NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT,
        "compensated stress/tangent did not reject invalid sigmaReference");
    require(std::memcmp(stress,stressBefore,sizeof(stress))==0&&
            std::memcmp(tangent,tangentBefore,sizeof(tangent))==0,
        "compensated stress/tangent wrote output on invalid input");
    require(nmHillCompensatedTrialStressDirectionalDerivative(invalidParameters.data(),F.data(),
            direction.data(),direct)==NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT,
        "compensated directional derivative did not reject invalid sigmaReference");
    require(std::memcmp(direct,directBefore,sizeof(direct))==0,
        "compensated directional derivative wrote output on invalid input");
    require(nmHillCompensatedHillImplicitSystem(invalidParameters.data(),F.data(),
            accepted.data(),accepted.data(),direction.data(),residual,jacobian,action)
            ==NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT,
        "compensated Hill implicit system did not reject invalid sigmaReference");
    require(std::memcmp(residual,residualBefore,sizeof(residual))==0&&
            std::memcmp(jacobian,jacobianBefore,sizeof(jacobian))==0&&
            std::memcmp(action,actionBefore,sizeof(action))==0,
        "compensated Hill implicit system wrote output on invalid input");

    auto nonfiniteF=F;
    nonfiniteF[0]=std::numeric_limits<float>::infinity();
    require(nmHillCompensatedTrialStress(parameters.parameterTable.data(),nonfiniteF.data(),
            accepted.data(),stress)==NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT,
        "compensated stress did not reject nonfinite F");
    require(std::memcmp(stress,stressBefore,sizeof(stress))==0,
        "compensated stress wrote output on nonfinite F");
}

float maxFP32SourceResidual(const hill::Parameters<float>& p,
    const std::array<float,9>& F,const hill::State13<float>& accepted,
    const hill::State13<float>& candidate) {
    float result=0.0f;
    const auto elasticStress=hill::elasticTrialStress(p,F,candidate);
    for (unsigned i=0;i<6;++i)
        result=std::max(result,std::abs(candidate[6+i]-elasticStress[i]));
    const float q=hill::hillEquivalent(p,
        {candidate[6],candidate[7],candidate[8],candidate[9],candidate[10],candidate[11]});
    const float deltaLambda=candidate[12]-accepted[12];
    const float Fh=p.hill[0],Gh=p.hill[1],Hh=p.hill[2];
    const std::array<float,6> numerator{
        Gh*(candidate[6]-candidate[8])+Hh*(candidate[6]-candidate[7]),
        Fh*(candidate[7]-candidate[8])+Hh*(candidate[7]-candidate[6]),
        Fh*(candidate[8]-candidate[7])+Gh*(candidate[8]-candidate[6]),
        p.hill[3]*candidate[9],p.hill[4]*candidate[10],p.hill[5]*candidate[11]};
    for (unsigned i=0;i<6;++i) {
        const float residual=candidate[i]-accepted[i]-deltaLambda*numerator[i]/q;
        result=std::max(result,std::abs(residual));
    }
    const float gap=1.0f-q;
    const float fb=std::sqrt(deltaLambda*deltaLambda+gap*gap+
        p.fischerBurmeisterEpsilon*p.fischerBurmeisterEpsilon)-deltaLambda-gap;
    result=std::max(result,std::abs(fb));
    return result;
}

void evaluateCapturedRows(const std::string& path,
    const std::string& referencePath,const std::vector<CapturedParameters>& parameters,Metrics& total,
    std::array<Metrics,2>& byMaterial,CompensatedMetrics& compensated,
    ElasticInitializerMetrics& elasticInitializer,
    ImplicitSystemMetrics& implicitSystem) {
    std::ifstream input(path);
    require(input.good(),"cannot open replay input: "+path);
    const SourceStates sourceStates=readSourceStates(referencePath);
    std::string line;
    while (std::getline(input,line)) {
        if (line.empty()) continue;
        const CapturedRow row=parseInputRow(line);
        const CapturedParameters& p=findParameters(parameters,row.material);
        std::array<float,9> F32{};
        hill::State13<float> accepted32{};
        for (unsigned i=0;i<9;++i) F32[i]=static_cast<float>(row.F[i]);
        for (unsigned i=0;i<13;++i) accepted32[i]=static_cast<float>(row.accepted[i]);
        const auto trial=hill::elasticTrialStress(p.fp32,F32,accepted32);
        const auto mapped=hill::returnMap(p.fp32,trial,accepted32);
        require(mapped.status==hill::Status::success,
            "scalar return map failed for captured row "+row.key+" status="+
            std::to_string(static_cast<unsigned>(mapped.status)));
        requirePortableBitMatch(p,accepted32,trial,mapped.state,"captured row "+row.key);

        const auto referenceIt=sourceStates.find(row.key);
        require(referenceIt!=sourceStates.end(),"source replay lacks captured key "+row.key);
        const paper::State& reference=referenceIt->second;
        paper::Matrix FCaptured{};
        paper::State acceptedCaptured{};
        for (unsigned i=0;i<9;++i) FCaptured[i]=static_cast<double>(F32[i]);
        for (unsigned i=0;i<13;++i) acceptedCaptured[i]=static_cast<double>(accepted32[i]);
        const auto trial64=paper::normalizedSecondPiola(p.fp64,FCaptured,
            paper::plasticStrain(acceptedCaptured));
        hill::Vector6<float> compensatedTrial{};
        float compensatedTangent[54];
        const int compensatedStatus=nmHillCompensatedTrialStressAndTangent(
            p.parameterTable.data(),F32.data(),accepted32.data(),
            compensatedTrial.data(),compensatedTangent);
        require(compensatedStatus==NM_HILL_SCALAR_INITIALIZER_SUCCESS,
            "compensated stress/tangent failed at "+row.key+" status="+
            std::to_string(compensatedStatus));
        for (unsigned i=0;i<6;++i) {
            const double ordinaryError=std::abs(static_cast<double>(trial[i])-trial64[i]);
            const double compensatedError=
                std::abs(static_cast<double>(compensatedTrial[i])-trial64[i]);
            compensated.maxOrdinaryStressError=std::max(
                compensated.maxOrdinaryStressError,ordinaryError);
            compensated.maxCompensatedStressError=std::max(
                compensated.maxCompensatedStressError,compensatedError);
            compensated.sumSquaredOrdinaryStressError+=ordinaryError*ordinaryError;
            compensated.sumSquaredCompensatedStressError+=compensatedError*compensatedError;
            ++compensated.components;
        }
        const auto sourceTangent=exactSourceTangent(p.fp64ExactInput,F32);
        std::array<float,9> dF{};
        for (unsigned i=0;i<9;++i) {
            const int integerValue=int((total.count*17+i*13)%23)-11;
            dF[i]=float(integerValue)*0.0078125f;
        }
        std::array<float,13> candidate32{};
        paper::State candidate64{};
        for (unsigned i=0;i<13;++i) {
            candidate32[i]=static_cast<float>(row.capturedCandidate[i]);
            candidate64[i]=static_cast<double>(candidate32[i]);
        }
        float systemResidual[13],systemJacobian[169],deformationAction[13];
        const int systemStatus=nmHillCompensatedHillImplicitSystem(
            p.parameterTable.data(),F32.data(),accepted32.data(),candidate32.data(),
            dF.data(),systemResidual,systemJacobian,deformationAction);
        require(systemStatus==NM_HILL_SCALAR_INITIALIZER_SUCCESS,
            "compensated Hill implicit system failed at "+row.key+
            " status="+std::to_string(systemStatus));
        const auto exactResidual=paper::residual(p.fp64ExactInput,row.F,
            row.accepted,candidate64);
        const auto exactJacobian=solver::implicitJacobian(
            p.fp64ExactInput,row.accepted,candidate64);
        const double residualScale=static_cast<double>(p.parameterTable[19]);
        const auto exactFixedStressTangent=exactSourceTangent(p.fp64ExactInput,F32);
        float candidateStress[6];
        for (unsigned i=0;i<6;++i) candidateStress[i]=candidate32[6+i];
        NMHillCompensatedFloatPair equivalentPair;
        const int equivalentStatus=nmHillCompensatedHillEquivalent(
            p.parameterTable.data(),candidateStress,&equivalentPair);
        require(equivalentStatus==NM_HILL_SCALAR_INITIALIZER_SUCCESS,
            "compensated Hill norm failed at "+row.key+
            " status="+std::to_string(equivalentStatus));
        paper::Vector6 candidateStress64{};
        for (unsigned i=0;i<6;++i) candidateStress64[i]=candidate64[6+i];
        const double equivalentError=std::abs(
            static_cast<double>(equivalentPair.high+equivalentPair.low)-
            paper::hillEquivalent(p.fp64ExactInput,candidateStress64));
        implicitSystem.maxEquivalentStressError=std::max(
            implicitSystem.maxEquivalentStressError,equivalentError);
        implicitSystem.sumSquaredEquivalentStressError+=equivalentError*equivalentError;
        ++implicitSystem.equivalentStressComponents;
        for (unsigned i=0;i<13;++i) {
            const double expected=residualScale*exactResidual[i];
            const double error=std::abs(static_cast<double>(systemResidual[i])-expected);
            implicitSystem.maxResidualError=std::max(implicitSystem.maxResidualError,error);
            implicitSystem.sumSquaredResidualError+=error*error;
            ++implicitSystem.residualComponents;
            double expectedAction=0.0;
            if (i>=6&&i<12) {
                const unsigned stressComponent=i-6;
                for (unsigned j=0;j<9;++j)
                    expectedAction-=residualScale*
                        exactFixedStressTangent[stressComponent*9+j]*dF[j];
            }
            const double actionError=std::abs(
                static_cast<double>(deformationAction[i])-expectedAction);
            implicitSystem.maxDeformationActionError=std::max(
                implicitSystem.maxDeformationActionError,actionError);
            implicitSystem.sumSquaredDeformationActionError+=actionError*actionError;
            ++implicitSystem.deformationActionComponents;
        }
        for (unsigned i=0;i<169;++i) {
            const double expected=residualScale*exactJacobian[i];
            const double error=std::abs(static_cast<double>(systemJacobian[i])-expected);
            implicitSystem.maxJacobianError=std::max(implicitSystem.maxJacobianError,error);
            const double relativeError=std::abs(expected)>1.0e-30?error/std::abs(expected):error;
            implicitSystem.maxRelativeJacobianError=std::max(
                implicitSystem.maxRelativeJacobianError,relativeError);
            implicitSystem.sumSquaredJacobianError+=error*error;
            ++implicitSystem.jacobianComponents;
        }
        float directDirectional[6];
        const int directionalStatus=nmHillCompensatedTrialStressDirectionalDerivative(
            p.parameterTable.data(),F32.data(),dF.data(),directDirectional);
        require(directionalStatus==NM_HILL_SCALAR_INITIALIZER_SUCCESS,
            "compensated stress directional derivative failed at "+row.key+
            " status="+std::to_string(directionalStatus));
        for (unsigned i=0;i<54;++i) {
            const double error=std::abs(static_cast<double>(compensatedTangent[i])-sourceTangent[i]);
            compensated.maxTangentError=std::max(compensated.maxTangentError,error);
            compensated.sumSquaredTangentError+=error*error;
            ++compensated.tangentComponents;
        }
        for (unsigned component=0;component<6;++component) {
            double referenceDirectional=0.0;
            float tangentAction=0.0f;
            for (unsigned i=0;i<9;++i) {
                referenceDirectional+=sourceTangent[component*9+i]*dF[i];
                tangentAction+=compensatedTangent[component*9+i]*dF[i];
            }
            compensated.maxDirectionalError=std::max(compensated.maxDirectionalError,
                std::abs(static_cast<double>(directDirectional[component])-referenceDirectional));
            compensated.maxTangentActionVsDirect=std::max(
                compensated.maxTangentActionVsDirect,
                std::abs(static_cast<double>(tangentAction)-directDirectional[component]));
            ++compensated.directionalComponents;
        }
        double trialStressDifference=0.0;
        for (unsigned i=0;i<6;++i)
            trialStressDifference=std::max(trialStressDifference,
                std::abs(static_cast<double>(trial[i])-trial64[i]));
        ++elasticInitializer.totalRows;
        elasticInitializer.totalScalarIterations+=mapped.iterations;
        elasticInitializer.maxScalarIterations=std::max(
            elasticInitializer.maxScalarIterations,mapped.iterations);
        hill::Parameters<double> qParameters;
        qParameters.hill=p.fp64ExactInput.hill;
        qParameters.qFloor=p.fp64ExactInput.qFloor;
        hill::Vector6<double> compensatedTrial64{};
        for (unsigned i=0;i<6;++i)
            compensatedTrial64[i]=static_cast<double>(compensatedTrial[i]);
        const double compensatedQ64=hill::hillEquivalent(qParameters,compensatedTrial64);
        const float compensatedQ32=hill::hillEquivalent(p.fp32,compensatedTrial);
        elasticInitializer.maxFloatVsExactQ=std::max(elasticInitializer.maxFloatVsExactQ,
            std::abs(static_cast<double>(compensatedQ32)-compensatedQ64));
        if (compensatedQ32<=1.0f) {
            ++elasticInitializer.qAtOrBelowOne;
            if (compensatedQ64>1.0) ++elasticInitializer.floatQFalseElastic;
            const double gap64=1.0-compensatedQ64;
            const double exactCandidateFB=std::hypot(gap64,
                static_cast<double>(p.parameterTable[18]))-gap64;
            elasticInitializer.maxElasticCandidateExactFB=std::max(
                elasticInitializer.maxElasticCandidateExactFB,exactCandidateFB);
            const float gap32=1.0f-compensatedQ32;
            const float fb32=std::sqrt(gap32*gap32+
                p.parameterTable[18]*p.parameterTable[18])-gap32;
            elasticInitializer.maxElasticCandidateFloatFB=std::max(
                elasticInitializer.maxElasticCandidateFloatFB,static_cast<double>(fb32));
            if (fb32==0.0f) ++elasticInitializer.elasticCandidateZeroFloatFB;
            elasticInitializer.minElasticGap=std::min(elasticInitializer.minElasticGap,gap64);
            if (compensatedQ32>=0.9999f) ++elasticInitializer.nearYieldGuardRows;
            const auto fullElasticMap=hill::returnMap(p.fp32,compensatedTrial,accepted32);
            require(fullElasticMap.status==hill::Status::success,
                "full scalar map failed on q<=1 compensated-stress case "+row.key);
            elasticInitializer.elasticScalarIterations+=fullElasticMap.iterations;
            elasticInitializer.elasticMaxScalarIterations=std::max(
                elasticInitializer.elasticMaxScalarIterations,fullElasticMap.iterations);
            elasticInitializer.maxElasticFullMapLambdaIncrement=std::max(
                elasticInitializer.maxElasticFullMapLambdaIncrement,
                static_cast<double>(fullElasticMap.lambdaIncrement));
            for (unsigned i=0;i<6;++i)
                elasticInitializer.maxElasticFullMapPlasticIncrement=std::max(
                    elasticInitializer.maxElasticFullMapPlasticIncrement,
                    std::abs(static_cast<double>(fullElasticMap.state[i]-accepted32[i])));
        } else {
            ++elasticInitializer.qAboveOne;
        }
        hill::Parameters<double> p64;
        for (unsigned i=0;i<6;++i) p64.stiffness[i]=p.fp64.c[i]*1.0e6;
        for (unsigned i=0;i<3;++i) p64.shear[i]=p.fp64.shear[i]*1.0e6;
        p64.sigmaReference=p.fp64.sigmaReference*1.0e6;
        for (unsigned i=0;i<6;++i) p64.hill[i]=p.fp64.hill[i];
        p64.qFloor=p.fp64.qFloor;
        p64.fischerBurmeisterEpsilon=p.fp64.fischerBurmeisterEpsilon;
        const auto referenceMap=hill::returnMap(p64,trial64,acceptedCaptured);
        require(referenceMap.status==hill::Status::success,
            "FP64 scalar reference failed captured row "+row.key);
        paper::State referenceState{};
        for (unsigned i=0;i<13;++i) referenceState[i]=referenceMap.state[i];
        const double sourceReferenceResidual=maxResidual(referenceState,p.fp64,FCaptured,acceptedCaptured);
        require(sourceReferenceResidual<1.0e-10,
            "FP64 source-equation reference residual too high at "+row.key);
        paper::State mappedState{};
        for (unsigned i=0;i<13;++i) mappedState[i]=static_cast<double>(mapped.state[i]);
        const double referenceResidual=maxResidual(mappedState,p.fp64,FCaptured,acceptedCaptured);
        paper::State candidateCaptured{};
        for (unsigned i=0;i<13;++i)
            candidateCaptured[i]=static_cast<double>(static_cast<float>(row.capturedCandidate[i]));
        const double candidateResidual=maxResidual(candidateCaptured,p.fp64,FCaptured,acceptedCaptured);
        const double stateError=maxStateDifference(referenceState,mapped.state);
        const double recordedOracleDifference=maxStateDifference(reference,mapped.state);
        const double exactInputOracleResidual=maxResidual(reference,p.fp64ExactInput,row.F,row.accepted);
        const auto exactInputTrial=paper::normalizedSecondPiola(p.fp64ExactInput,row.F,
            paper::plasticStrain(row.accepted));
        hill::Parameters<double> exactInputMapParameters;
        for (unsigned i=0;i<6;++i)
            exactInputMapParameters.stiffness[i]=p.fp64ExactInput.c[i]*1.0e6;
        for (unsigned i=0;i<3;++i)
            exactInputMapParameters.shear[i]=p.fp64ExactInput.shear[i]*1.0e6;
        exactInputMapParameters.sigmaReference=p.fp64ExactInput.sigmaReference*1.0e6;
        for (unsigned i=0;i<6;++i)
            exactInputMapParameters.hill[i]=p.fp64ExactInput.hill[i];
        exactInputMapParameters.qFloor=p.fp64ExactInput.qFloor;
        exactInputMapParameters.fischerBurmeisterEpsilon=p.fp64ExactInput.fischerBurmeisterEpsilon;
        const auto exactInputFP64Map=hill::returnMap(
            exactInputMapParameters,exactInputTrial,row.accepted);
        require(exactInputFP64Map.status==hill::Status::success,
            "exact-input FP64 scalar reference failed captured row "+row.key);
        paper::State exactInputFP64MapState{};
        for (unsigned i=0;i<13;++i) exactInputFP64MapState[i]=exactInputFP64Map.state[i];
        const double exactInputFP64MapResidual=maxResidual(
            exactInputFP64MapState,p.fp64ExactInput,row.F,row.accepted);
        const double exactInputFP64MapVsOracle=maxStateDifference(reference,exactInputFP64Map.state);
        const float trialQ=hill::hillEquivalent(p.fp32,trial);
        const float fp32EquationResidual=maxFP32SourceResidual(
            p.fp32,F32,accepted32,mapped.state);
        const unsigned materialIndex=row.material=="liner"?0u:1u;
        for (Metrics* m:{&total,&byMaterial[materialIndex]}) {
            ++m->count;
            if (trialQ>1.0f) ++m->yielded;
            else ++m->elastic;
            if (referenceResidual>m->maxReferenceResidual) {
                m->maxReferenceResidual=referenceResidual;
                m->worstReferenceKey=row.key;
            }
            if (candidateResidual>m->maxCapturedCandidateResidual) {
                m->maxCapturedCandidateResidual=candidateResidual;
                m->worstCandidateKey=row.key;
            }
            if (stateError>m->maxStateError) {
                m->maxStateError=stateError;
                m->worstStateKey=row.key;
            }
            m->maxScalarResidual=std::max(m->maxScalarResidual,
                std::abs(static_cast<double>(mapped.scalarResidual)));
            m->maxFBResidual=std::max(m->maxFBResidual,
                std::abs(static_cast<double>(mapped.fischerBurmeisterResidual)));
            m->maxStressSolveResidual=std::max(m->maxStressSolveResidual,
                static_cast<double>(mapped.stressResidual));
            m->maxFlowResidual=std::max(m->maxFlowResidual,
                static_cast<double>(mapped.flowResidual));
            m->maxFP32SourceResidual=std::max(m->maxFP32SourceResidual,
                static_cast<double>(fp32EquationResidual));
            m->maxFP32TrialStressVsFP64Source=std::max(m->maxFP32TrialStressVsFP64Source,
                trialStressDifference);
            m->maxRecordedOracleDifference=std::max(m->maxRecordedOracleDifference,
                recordedOracleDifference);
            m->maxExactInputOracleResidual=std::max(m->maxExactInputOracleResidual,
                exactInputOracleResidual);
            m->maxExactInputFP64MapResidual=std::max(m->maxExactInputFP64MapResidual,
                exactInputFP64MapResidual);
            m->maxExactInputFP64MapVsOracle=std::max(m->maxExactInputFP64MapVsOracle,
                exactInputFP64MapVsOracle);
        }
    }
    require(total.count==14400,"replay input was not the expected 14,400-row capture");
    require(byMaterial[0].count>0&&byMaterial[1].count>0,
        "capture did not include both materials");
    require(total.maxScalarResidual<1.0e-9,
        "captured scalar roots exceed the documented scalar residual bound");
    require(total.maxStressSolveResidual<2.0e-8,
        "captured stress solves exceed the documented linear residual bound");
    require(total.maxFlowResidual<2.0e-8,
        "captured associated-flow updates exceed the documented residual bound");
    require(total.maxFP32SourceResidual<2.0e-7,
        "captured maps exceed the FP32 source-equation residual bound");
    require(total.maxExactInputOracleResidual<1.0e-9,
        "recorded exact-input oracle exceeds source-equation residual bound");
    require(total.maxExactInputFP64MapResidual<1.0e-12,
        "exact-input FP64 scalar map exceeds source-equation residual bound");
    require(total.maxExactInputFP64MapVsOracle<1.0e-8,
        "exact-input FP64 scalar map diverges from recorded oracle state");
    require(compensated.components==14400u*6u&&
        compensated.tangentComponents==14400u*54u&&
        compensated.directionalComponents==14400u*6u,
        "compensated stress/tangent coverage is incomplete");
    require(elasticInitializer.totalRows==14400u&&
        elasticInitializer.qAtOrBelowOne+elasticInitializer.qAboveOne==14400u,
        "elastic-branch initializer audit did not cover every captured row");
    require(implicitSystem.residualComponents==14400u*13u&&
        implicitSystem.jacobianComponents==14400u*169u&&
        implicitSystem.deformationActionComponents==14400u*13u&&
        implicitSystem.equivalentStressComponents==14400u,
        "compensated implicit-system coverage is incomplete");
    require(compensated.maxTangentError<2.0e-5&&compensated.maxDirectionalError<3.0e-6&&
        compensated.maxTangentActionVsDirect<1.0e-5,
        "compensated analytical derivative exceeds exact-source FP64 error bounds");
}

paper::Matrix deformationForTrial(const paper::Parameters& p,const paper::State& accepted,
                                  double targetUniaxialStress) {
    // Find diagonal Green strain whose source elastic law gives the requested
    // normalized pure MD stress; this keeps the branch sweep source-equivalent.
    const double a=p.c[0]/p.sigmaReference,b=p.c[3]/p.sigmaReference;
    const double c=p.c[4]/p.sigmaReference,d=p.c[1]/p.sigmaReference;
    const double e=p.c[5]/p.sigmaReference,f=p.c[2]/p.sigmaReference;
    const double determinant=a*(d*f-e*e)-b*(b*f-c*e)+c*(b*e-c*d);
    require(std::abs(determinant)>1.0e-15,"singular normal stiffness in branch sweep");
    // Cramer's rule for C_normal * strain = {targetStress,0,0}.
    const double e11=targetUniaxialStress*(d*f-e*e)/determinant;
    const double e22=targetUniaxialStress*(c*e-b*f)/determinant;
    const double e33=targetUniaxialStress*(b*e-c*d)/determinant;
    const double c11=1.0+2.0*(e11+accepted[0]);
    const double c22=1.0+2.0*(e22+accepted[1]);
    const double c33=1.0+2.0*(e33+accepted[2]);
    require(c11>0&&c22>0&&c33>0,"branch sweep strain makes nonpositive F^T F");
    return {std::sqrt(c11),0,0, 0,std::sqrt(c22),0, 0,0,std::sqrt(c33)};
}

SweepMetrics checkBranchSweeps(const std::vector<CapturedParameters>& parameters) {
    constexpr std::array<double,11> levels{
        0.0,0.1,0.5,0.9,0.99,0.999999,1.0,1.000001,1.01,2.0,10.0};
    SweepMetrics metrics;
    for (const auto& captured:parameters) {
        double previousTrialQ=-std::numeric_limits<double>::infinity();
        double previousLambdaIncrement=-std::numeric_limits<double>::infinity();
        for (double target:levels) {
            paper::State accepted{};
            accepted[0]=1.0e-4; accepted[1]=-2.0e-5; accepted[2]=3.0e-6;
            accepted[12]=0.125; // sentinel history: returned state may only add d-lambda
            const paper::Matrix F=deformationForTrial(captured.fp64,accepted,-target);
            std::array<float,9> F32{};
            hill::State13<float> accepted32{};
            for (unsigned i=0;i<9;++i) F32[i]=static_cast<float>(F[i]);
            for (unsigned i=0;i<13;++i) accepted32[i]=static_cast<float>(accepted[i]);
            const auto trial32=hill::elasticTrialStress(captured.fp32,F32,accepted32);
            const auto map32=hill::returnMap(captured.fp32,trial32,accepted32);
            require(map32.status==hill::Status::success,
                "FP32 branch sweep failed material="+captured.name+
                " trial_q="+std::to_string(target));
            requirePortableBitMatch(captured,accepted32,trial32,map32.state,
                "branch sweep material="+captured.name+" trial_q="+std::to_string(target));
            const double trialQ32=hill::hillEquivalent(captured.fp32,trial32);
            require(trialQ32+2.0e-6>=previousTrialQ,
                "branch sweep trial q reversed as the applied level increased");
            require(static_cast<double>(map32.lambdaIncrement)+2.0e-6>=previousLambdaIncrement,
                "branch sweep multiplier increment decreased as the applied level increased");
            require(map32.lambdaIncrement>=0.0f,
                "smoothed complementarity returned a negative multiplier increment");
            previousTrialQ=trialQ32;
            previousLambdaIncrement=map32.lambdaIncrement;
            hill::Parameters<double> p64;
            for (unsigned i=0;i<6;++i) p64.stiffness[i]=captured.fp64.c[i]*1.0e6;
            for (unsigned i=0;i<3;++i) p64.shear[i]=captured.fp64.shear[i]*1.0e6;
            p64.sigmaReference=captured.fp64.sigmaReference*1.0e6;
            for (unsigned i=0;i<6;++i) p64.hill[i]=captured.fp64.hill[i];
            p64.qFloor=captured.fp64.qFloor;
            p64.fischerBurmeisterEpsilon=captured.fp64.fischerBurmeisterEpsilon;
            const auto trial64=hill::elasticTrialStress(p64,F,accepted);
            const auto map64=hill::returnMap(p64,trial64,accepted);
            require(map64.status==hill::Status::success,
                "FP64 scalar branch sweep failed material="+captured.name+
                " trial_q="+std::to_string(target));
            paper::State state64{};
            for (unsigned i=0;i<13;++i) state64[i]=map64.state[i];
            const double sourceResidual=paper::residualNorm(
                paper::residual(captured.fp64,F,accepted,state64));
            require(sourceResidual<1.0e-10,
                "FP64 scalar branch sweep fails independent source equations at trial_q="+
                std::to_string(target));
            require(std::abs(static_cast<double>(map32.state[12])-state64[12])<2.0e-5,
                "FP32 multiplier diverged from FP64 source-equation root at trial_q="+
                std::to_string(target));
            require(map32.state[12]>=accepted32[12],
                "accumulated multiplier regressed in branch sweep");
            for (unsigned i=0;i<6;++i)
                require(std::abs(static_cast<double>(map32.state[i])-state64[i])<2.0e-5,
                    "FP32 plastic history diverged from FP64 source-equation root at trial_q="+
                    std::to_string(target));
            require(std::abs(static_cast<double>(map32.fischerBurmeisterResidual))<1.0e-6,
                "FP32 smoothed FB residual exceeds representable branch-sweep envelope material="+
                captured.name+" trial_q="+std::to_string(target)+" residual="+
                std::to_string(map32.fischerBurmeisterResidual));
            double stateDifference=0.0;
            for (unsigned i=0;i<13;++i)
                stateDifference=std::max(stateDifference,
                    std::abs(static_cast<double>(map32.state[i])-state64[i]));
            ++metrics.count;
            metrics.minTrialQ=std::min(metrics.minTrialQ,trialQ32);
            metrics.maxTrialQ=std::max(metrics.maxTrialQ,trialQ32);
            metrics.maxLambdaIncrement=std::max(metrics.maxLambdaIncrement,
                static_cast<double>(map32.lambdaIncrement));
            metrics.maxFP32SourceResidual=std::max(metrics.maxFP32SourceResidual,
                static_cast<double>(maxFP32SourceResidual(captured.fp32,F32,accepted32,map32.state)));
            metrics.maxFP32ToFP64StateDifference=std::max(
                metrics.maxFP32ToFP64StateDifference,stateDifference);
            metrics.maxFP64SourceResidual=std::max(metrics.maxFP64SourceResidual,sourceResidual);
            metrics.maxFP32FischerBurmeisterResidual=std::max(
                metrics.maxFP32FischerBurmeisterResidual,
                std::abs(static_cast<double>(map32.fischerBurmeisterResidual)));
        }
    }
    require(metrics.count==22,"branch sweep did not cover 11 levels for both materials");
    require(metrics.maxFP32SourceResidual<1.0e-5,
        "branch sweep FP32 source-equation residual exceeded the diagnostic bound");
    require(metrics.maxFP32FischerBurmeisterResidual<2.0e-6,
        "branch sweep FP32 smoothed complementarity residual exceeded the diagnostic bound");
    return metrics;
}

void printMetric(const char* label,const Metrics& m) {
    std::cout<<"  \""<<label<<"\": {\"records\":"<<m.count
        <<",\"elastic_trial_count\":"<<m.elastic
        <<",\"yielded_trial_count\":"<<m.yielded
        <<",\"max_fp64_source_residual_on_fp32_map\":"<<m.maxReferenceResidual
        <<",\"max_fp64_source_residual_on_captured_candidate\":"<<m.maxCapturedCandidateResidual
        <<",\"max_fp32_to_fp64_state_error\":"<<m.maxStateError
        <<",\"max_abs_scalar_residual\":"<<m.maxScalarResidual
        <<",\"max_abs_smoothed_fb_residual\":"<<m.maxFBResidual
        <<",\"max_stress_linear_solve_residual\":"<<m.maxStressSolveResidual
        <<",\"max_flow_update_residual\":"<<m.maxFlowResidual
        <<",\"max_fp32_source_equation_residual\":"<<m.maxFP32SourceResidual
        <<",\"max_fp32_trial_stress_vs_fp64_source\":"<<m.maxFP32TrialStressVsFP64Source
        <<",\"max_fp32_map_vs_recorded_exact_input_oracle_state\":"<<m.maxRecordedOracleDifference
        <<",\"max_recorded_exact_input_oracle_source_residual\":"<<m.maxExactInputOracleResidual
        <<",\"max_exact_input_fp64_scalar_map_source_residual\":"<<m.maxExactInputFP64MapResidual
        <<",\"max_exact_input_fp64_scalar_map_vs_recorded_oracle_state\":"<<m.maxExactInputFP64MapVsOracle
        <<",\"worst_reference_key\":\""<<m.worstReferenceKey
        <<"\",\"worst_captured_candidate_key\":\""<<m.worstCandidateKey
        <<"\",\"worst_state_error_key\":\""<<m.worstStateKey<<"\"}";
}

} // namespace

int main(int argc,char** argv) try {
    if (argc!=4) {
        std::cerr<<"usage: cardboard-hill-scalar-return-map-check replay-input.txt captured-material-parameters.txt replay-output.jsonl\n";
        return 2;
    }
    std::cout<<std::setprecision(12);
    const auto parameters=readParameters(argv[2]);
    checkPortableFailureSemantics(parameters.front());
    checkCompensatedFailureSemantics(parameters.front());
    Metrics total;
    std::array<Metrics,2> byMaterial{};
    CompensatedMetrics compensated;
    ElasticInitializerMetrics elasticInitializer;
    ImplicitSystemMetrics implicitSystem;
    evaluateCapturedRows(argv[1],argv[3],parameters,total,byMaterial,compensated,
        elasticInitializer,implicitSystem);
    const SweepMetrics sweep=checkBranchSweeps(parameters);
    std::cout<<"{\"schema\":\"numi.cardboard.hill-scalar-return-map-check.v1\",\n"
        <<" \"input_contract\":\"exact_widened_fp32\",\n"
        <<" \"parameters_path\":\""<<argv[2]<<"\",\n"
        <<" \"replay_path\":\""<<argv[1]<<"\",\n"
        <<" \"source_reference_path\":\""<<argv[3]<<"\",\n"
        <<" \"captured_rows\":14400,\n \"branch_sweep\": {\"cases\":"<<sweep.count
        <<",\"portable_bitwise_match_cases\":"<<(14400+sweep.count)
        <<",\"failure_semantics_checked\":true"
        <<",\"min_trial_q\":"<<sweep.minTrialQ<<",\"max_trial_q\":"<<sweep.maxTrialQ
        <<",\"max_multiplier_increment\":"<<sweep.maxLambdaIncrement
        <<",\"max_fp32_source_residual\":"<<sweep.maxFP32SourceResidual
        <<",\"max_fp32_to_fp64_state_difference\":"<<sweep.maxFP32ToFP64StateDifference
        <<",\"max_fp64_source_residual\":"<<sweep.maxFP64SourceResidual
        <<",\"max_fp32_fb_residual\":"<<sweep.maxFP32FischerBurmeisterResidual<<"},\n"
        <<" \"elastic_initializer_fast_path_audit\": {\"rows\":"<<elasticInitializer.totalRows
        <<",\"q_le_one\":"<<elasticInitializer.qAtOrBelowOne
        <<",\"q_gt_one\":"<<elasticInitializer.qAboveOne
        <<",\"float_q_false_elastic_rows\":"<<elasticInitializer.floatQFalseElastic
        <<",\"q_le_one_within_1e-4_of_yield\":"<<elasticInitializer.nearYieldGuardRows
        <<",\"scalar_bisection_iterations_total\":"<<elasticInitializer.totalScalarIterations
        <<",\"scalar_bisection_iterations_max\":"<<elasticInitializer.maxScalarIterations
        <<",\"q_le_one_full_map_iterations_total\":"<<elasticInitializer.elasticScalarIterations
        <<",\"q_le_one_full_map_iterations_max\":"<<elasticInitializer.elasticMaxScalarIterations
        <<",\"candidate_exact_fb_max_for_q_le_one\":"
        <<elasticInitializer.maxElasticCandidateExactFB
        <<",\"candidate_float_fb_max_for_q_le_one\":"
        <<elasticInitializer.maxElasticCandidateFloatFB
        <<",\"candidate_float_fb_zero_rows\":"<<elasticInitializer.elasticCandidateZeroFloatFB
        <<",\"full_map_lambda_increment_max_for_q_le_one\":"
        <<elasticInitializer.maxElasticFullMapLambdaIncrement
        <<",\"full_map_plastic_history_increment_max_for_q_le_one\":"
        <<elasticInitializer.maxElasticFullMapPlasticIncrement
        <<",\"max_fp32_q_vs_exact_q\":"<<elasticInitializer.maxFloatVsExactQ
        <<",\"minimum_exact_elastic_gap\":"<<elasticInitializer.minElasticGap<<"},\n"
        <<" \"compensated_implicit_system\": {\"residual_components\":"
        <<implicitSystem.residualComponents
        <<",\"equivalent_stress_components\":"<<implicitSystem.equivalentStressComponents
        <<",\"equivalent_stress_max_abs_error_vs_fp64\":"
        <<implicitSystem.maxEquivalentStressError
        <<",\"equivalent_stress_rms_error_vs_fp64\":"
        <<std::sqrt(implicitSystem.sumSquaredEquivalentStressError/
            implicitSystem.equivalentStressComponents)
        <<",\"residual_max_abs_error_vs_fp64\":"<<implicitSystem.maxResidualError
        <<",\"residual_rms_error_vs_fp64\":"
        <<std::sqrt(implicitSystem.sumSquaredResidualError/implicitSystem.residualComponents)
        <<",\"jacobian_components\":"<<implicitSystem.jacobianComponents
        <<",\"jacobian_max_abs_error_vs_fp64\":"<<implicitSystem.maxJacobianError
        <<",\"jacobian_max_relative_error_vs_fp64\":"
        <<implicitSystem.maxRelativeJacobianError
        <<",\"jacobian_rms_error_vs_fp64\":"
        <<std::sqrt(implicitSystem.sumSquaredJacobianError/implicitSystem.jacobianComponents)
        <<",\"deformation_action_components\":"
        <<implicitSystem.deformationActionComponents
        <<",\"deformation_action_max_abs_error_vs_fp64\":"
        <<implicitSystem.maxDeformationActionError
        <<",\"deformation_action_rms_error_vs_fp64\":"
        <<std::sqrt(implicitSystem.sumSquaredDeformationActionError/
            implicitSystem.deformationActionComponents)<<"},\n"
        <<" \"compensated_stress_eval\": {\"components\":"<<compensated.components
        <<",\"ordinary_max_abs_error_vs_fp64\":"<<compensated.maxOrdinaryStressError
        <<",\"compensated_max_abs_error_vs_fp64\":"<<compensated.maxCompensatedStressError
        <<",\"ordinary_rms_error_vs_fp64\":"
        <<std::sqrt(compensated.sumSquaredOrdinaryStressError/compensated.components)
        <<",\"compensated_rms_error_vs_fp64\":"
        <<std::sqrt(compensated.sumSquaredCompensatedStressError/compensated.components)
        <<",\"max_error_reduction_factor\":"
        <<compensated.maxOrdinaryStressError/compensated.maxCompensatedStressError
        <<",\"rms_error_reduction_factor\":"
        <<std::sqrt(compensated.sumSquaredOrdinaryStressError/
            compensated.sumSquaredCompensatedStressError)<<"},\n"
        <<" \"analytical_tangent\": {\"components\":"<<compensated.tangentComponents
        <<",\"max_abs_error_vs_fp64_formula\":"<<compensated.maxTangentError
        <<",\"rms_abs_error_vs_fp64_formula\":"
        <<std::sqrt(compensated.sumSquaredTangentError/compensated.tangentComponents)
        <<",\"directional_components\":"<<compensated.directionalComponents
        <<",\"max_abs_directional_error_vs_fp64_formula\":"
        <<compensated.maxDirectionalError
        <<",\"max_tangent_action_vs_direct_directional\":"
        <<compensated.maxTangentActionVsDirect<<"},\n"
        <<" \"total\": {\n";
    printMetric("all",total);
    std::cout<<",\n";
    printMetric("liner",byMaterial[0]);
    std::cout<<",\n";
    printMetric("medium",byMaterial[1]);
    std::cout<<"}\n}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr<<"cardboard scalar return-map check failed: "<<error.what()<<"\n";
    return 1;
}
