#include "numi/matter/detail.hpp"
#include "cardboard_paper_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace numi::matter;
namespace paper=numi_cardboard_paper;
unsigned checks=0;
constexpr double kNumericalResidualScale=10.0;

void require(bool condition,const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

std::string number(double value) {
    std::ostringstream out; out<<std::setprecision(12)<<value; return out.str();
}

std::string diagnostics(const std::vector<Diagnostic>& values) {
    std::string result;
    for (const auto& value:values) result+=value.message+"; ";
    return result;
}

double evaluate(const ScalarBytecode& bytecode,const MaterialProgram& material,
                const paper::Matrix& f,const paper::Matrix& h,
                const paper::State& accepted,const paper::State& candidate) {
    std::vector<double> stack;
    const auto pop=[&]() {
        require(!stack.empty(),"paper bytecode stack underflow");
        const double value=stack.back(); stack.pop_back(); return value;
    };
    for (const NMExpressionInstructionGPU& instruction:bytecode.instructions) {
        double value=0.0;
        switch (instruction.opcode) {
        case NM_EXPR_CONSTANT: value=instruction.immediate.x; break;
        case NM_EXPR_PARAMETER:
            value=static_cast<float>(material.parameters.at(instruction.index).defaultValue); break;
        case NM_EXPR_F: value=f.at(instruction.index); break;
        case NM_EXPR_DF: value=h.at(instruction.index); break;
        case NM_EXPR_STATE: value=accepted.at(instruction.index); break;
        case NM_EXPR_NEXT_STATE: value=candidate.at(instruction.index); break;
        case NM_EXPR_ADD: { const double b=pop(); value=pop()+b; break; }
        case NM_EXPR_SUBTRACT: { const double b=pop(); value=pop()-b; break; }
        case NM_EXPR_MULTIPLY: { const double b=pop(); value=pop()*b; break; }
        case NM_EXPR_DIVIDE: {
            const double b=pop(); const double a=pop();
            require(std::abs(b)>1.0e-12,"paper bytecode division violates production geometry floor");
            value=a/b; break;
        }
        case NM_EXPR_NEGATE: value=-pop(); break;
        case NM_EXPR_SQRT: value=std::sqrt(pop()); break;
        case NM_EXPR_ABS: value=std::abs(pop()); break;
        case NM_EXPR_LOG: value=std::log(pop()); break;
        case NM_EXPR_EXP: value=std::exp(pop()); break;
        case NM_EXPR_POW_INTEGER: {
            const double x=pop();
            require(instruction.integer>=0 || std::abs(x)>1.0e-12,
                "paper bytecode negative power violates production geometry floor");
            value=std::pow(x,instruction.integer); break;
        }
        case NM_EXPR_MIN: { const double b=pop(); value=std::min(pop(),b); break; }
        case NM_EXPR_MAX: { const double b=pop(); value=std::max(pop(),b); break; }
        case NM_EXPR_EXPM1_MINUS_X: { const double x=pop(); value=std::expm1(x)-x; break; }
        default: throw std::runtime_error("unsupported opcode in paper constitutive bytecode interpreter: "+
            std::to_string(instruction.opcode));
        }
        require(std::isfinite(value),"nonfinite authored paper bytecode result");
        stack.push_back(value);
    }
    require(stack.size()==1,"paper bytecode failed to reduce to one scalar");
    return stack.front();
}

float evaluateFP32(const ScalarBytecode& bytecode,const MaterialProgram& material,
                   const paper::Matrix& f,const paper::Matrix& h,
                   const paper::State& accepted,const paper::State& candidate,
                   float& minimumDivision) {
    std::vector<float> stack;
    const auto pop=[&]() {
        require(!stack.empty(),"paper FP32 bytecode stack underflow");
        const float value=stack.back(); stack.pop_back(); return value;
    };
    for(const NMExpressionInstructionGPU& instruction:bytecode.instructions) {
        float value=0.0f;
        switch(instruction.opcode) {
        case NM_EXPR_CONSTANT:value=instruction.immediate.x;break;
        case NM_EXPR_PARAMETER:value=static_cast<float>(material.parameters.at(instruction.index).defaultValue);break;
        case NM_EXPR_F:value=static_cast<float>(f.at(instruction.index));break;
        case NM_EXPR_DF:value=static_cast<float>(h.at(instruction.index));break;
        case NM_EXPR_STATE:value=static_cast<float>(accepted.at(instruction.index));break;
        case NM_EXPR_NEXT_STATE:value=static_cast<float>(candidate.at(instruction.index));break;
        case NM_EXPR_ADD:{const float b=pop();value=pop()+b;break;}
        case NM_EXPR_SUBTRACT:{const float b=pop();value=pop()-b;break;}
        case NM_EXPR_MULTIPLY:{const float b=pop();value=pop()*b;break;}
        case NM_EXPR_DIVIDE:{const float b=pop();const float a=pop();minimumDivision=std::min(minimumDivision,std::abs(b));
            require(std::abs(b)>1.0e-12f,"paper FP32 bytecode division violates production geometry floor");value=a/b;break;}
        case NM_EXPR_NEGATE:value=-pop();break;
        case NM_EXPR_SQRT:value=std::sqrt(pop());break;
        case NM_EXPR_ABS:value=std::abs(pop());break;
        case NM_EXPR_LOG:value=std::log(pop());break;
        case NM_EXPR_EXP:value=std::exp(pop());break;
        case NM_EXPR_POW_INTEGER:{
            const float x=pop();const int exponent=instruction.integer;
            require(exponent>=0||std::abs(x)>1.0e-12f,"paper FP32 negative power violates production geometry floor");
            float result=1.0f,factor=x;unsigned power=static_cast<unsigned>(exponent<0?-exponent:exponent);
            while(power!=0u){if((power&1u)!=0u)result*=factor;factor*=factor;power>>=1u;}
            value=exponent<0?1.0f/result:result;break;
        }
        case NM_EXPR_MIN:{const float b=pop();value=std::min(pop(),b);break;}
        case NM_EXPR_MAX:{const float b=pop();value=std::max(pop(),b);break;}
        case NM_EXPR_EXPM1_MINUS_X:{const float x=pop();value=std::exp(x)-1.0f-x;break;}
        default:throw std::runtime_error("unsupported opcode in paper FP32 bytecode interpreter: "+
            std::to_string(instruction.opcode));
        }
        require(std::isfinite(value),"nonfinite authored paper FP32 bytecode result");
        stack.push_back(value);
    }
    require(stack.size()==1,"paper FP32 bytecode failed to reduce to one scalar");
    return stack.front();
}

std::array<float,13> compiledFP32Residuals(const ConstitutiveProgram& program,
    const paper::Matrix& f,const paper::State& accepted,const paper::State& candidate,
    float& minimumDivision) {
    std::array<float,13> result{}; const paper::Matrix zero{};
    for(unsigned i=0;i<13;++i) result[i]=evaluateFP32(program.implicitResiduals.at(i),
        program.material,f,zero,accepted,candidate,minimumDivision);
    return result;
}

std::array<float,169> compiledFP32Jacobian(const ConstitutiveProgram& program,
    const paper::Matrix& f,const paper::State& accepted,const paper::State& candidate,
    float& minimumDivision) {
    std::array<float,169> result{}; const paper::Matrix zero{};
    for(unsigned i=0;i<169;++i) result[i]=evaluateFP32(program.implicitJacobians.at(i),
        program.material,f,zero,accepted,candidate,minimumDivision);
    return result;
}

bool productionLocalSolve(std::array<float,169> matrix,std::array<float,13> rhs,
                          float& minimumPivot) {
    for(unsigned column=0;column<13;++column) {
        unsigned pivot=column;float magnitude=std::abs(matrix[13*column+column]);
        for(unsigned row=column+1;row<13;++row) {
            const float candidate=std::abs(matrix[13*row+column]);
            if(candidate>magnitude){pivot=row;magnitude=candidate;}
        }
        minimumPivot=std::min(minimumPivot,magnitude);
        if(!(magnitude>1.0e-8f)||!std::isfinite(magnitude))return false;
        if(pivot!=column) {
            for(unsigned entry=column;entry<13;++entry)
                std::swap(matrix[13*column+entry],matrix[13*pivot+entry]);
            std::swap(rhs[column],rhs[pivot]);
        }
        const float diagonal=matrix[13*column+column];
        for(unsigned row=column+1;row<13;++row) {
            const float factor=matrix[13*row+column]/diagonal;
            matrix[13*row+column]=0.0f;
            for(unsigned entry=column+1;entry<13;++entry)
                matrix[13*row+entry]-=factor*matrix[13*column+entry];
            rhs[row]-=factor*rhs[column];
        }
    }
    for(int row=12;row>=0;--row) {
        float value=rhs[static_cast<unsigned>(row)];
        for(unsigned column=static_cast<unsigned>(row)+1;column<13;++column)
            value-=matrix[13*static_cast<unsigned>(row)+column]*rhs[column];
        rhs[static_cast<unsigned>(row)]=value/matrix[13*static_cast<unsigned>(row)+static_cast<unsigned>(row)];
        if(!std::isfinite(rhs[static_cast<unsigned>(row)]))return false;
    }
    return true;
}

struct ProductionTrace {
    bool valid=false;
    std::string failure;
    std::string failureDetail;
    unsigned convergedIteration=0u;
    unsigned firstAcceptedBacktrack=0u;
    float firstAcceptedStep=0.0f;
    float finalResidual=0.0f;
    float finalTolerance=0.0f;
};

ProductionTrace traceProductionProjection(const ConstitutiveProgram& program,
    const paper::Matrix& f,const paper::State& accepted) {
    ProductionTrace trace;
    const paper::Matrix zero{};
    std::array<float,13> candidate{};
    const auto evaluateRow=[&](const ScalarBytecode& bytecode,
                               const std::array<float,13>& state,
                               const std::string& label,float& result) {
        paper::State candidateDouble{};
        for(unsigned i=0;i<13;++i) candidateDouble[i]=state[i];
        try {
            float minimumDivision=std::numeric_limits<float>::infinity();
            result=evaluateFP32(bytecode,program.material,f,zero,accepted,
                candidateDouble,minimumDivision);
        } catch(const std::exception& error) {
            trace.failure=label+": "+error.what();
            trace.failureDetail=trace.failure;
            return false;
        }
        if(!std::isfinite(result)) {
            trace.failure=label+": nonfinite result";
            trace.failureDetail=trace.failure;
            return false;
        }
        return true;
    };
    for(unsigned state=0;state<13;++state) {
        float seed=0.0f,minimumDivision=std::numeric_limits<float>::infinity();
        paper::State candidateDouble{};
        try {
            seed=evaluateFP32(program.stateUpdates.at(state),program.material,f,zero,
                accepted,candidateDouble,minimumDivision);
        } catch(const std::exception& error) {
            trace.failure="stateUpdate["+std::to_string(state)+"]: "+error.what();
            trace.failureDetail=trace.failure;
            return trace;
        }
        if(!std::isfinite(seed)) {
            trace.failure="stateUpdate["+std::to_string(state)+"] produced nonfinite seed";
            trace.failureDetail=trace.failure;
            return trace;
        }
        candidate[state]=seed;
    }
    for(unsigned iteration=0;iteration<16;++iteration) {
        std::array<float,13> residual{};
        std::array<float,169> jacobian{};
        float residualNorm=0.0f,stateNorm=0.0f;
        for(unsigned row=0;row<13;++row) {
            if(!evaluateRow(program.implicitResiduals.at(row),candidate,
                    "residual["+std::to_string(row)+"]",residual[row])) return trace;
            residualNorm=std::max(residualNorm,std::abs(residual[row]));
            stateNorm=std::max(stateNorm,std::abs(candidate[row]));
            for(unsigned column=0;column<13;++column) {
                if(!evaluateRow(program.implicitJacobians.at(13*row+column),candidate,
                        "jacobian["+std::to_string(row)+","+std::to_string(column)+"]",
                        jacobian[13*row+column])) return trace;
            }
        }
        const float tolerance=2.0e-5f*(1.0f+stateNorm);
        if(residualNorm<=tolerance) {
            trace.valid=true;
            trace.convergedIteration=iteration;
            trace.finalResidual=residualNorm;
            trace.finalTolerance=tolerance;
            return trace;
        }
        std::array<float,13> correction{};
        for(unsigned row=0;row<13;++row) correction[row]=-residual[row];
        bool singular=false;
        unsigned failedColumn=0;
        float failedPivot=0.0f;
        float minimumPivot=std::numeric_limits<float>::infinity();
        unsigned minimumPivotColumn=0;
        for(unsigned column=0;column<13;++column) {
            unsigned pivot=column;
            float pivotMagnitude=std::abs(jacobian[13*column+column]);
            for(unsigned row=column+1;row<13;++row) {
                const float magnitude=std::abs(jacobian[13*row+column]);
                if(magnitude>pivotMagnitude) { pivot=row; pivotMagnitude=magnitude; }
            }
            if(pivotMagnitude<minimumPivot) {
                minimumPivot=pivotMagnitude;
                minimumPivotColumn=column;
            }
            if(!(pivotMagnitude>1.0e-8f)||!std::isfinite(pivotMagnitude)) {
                singular=true;failedColumn=column;failedPivot=pivotMagnitude;break;
            }
            if(pivot!=column) {
                for(unsigned entry=column;entry<13;++entry)
                    std::swap(jacobian[13*column+entry],jacobian[13*pivot+entry]);
                std::swap(correction[column],correction[pivot]);
            }
            const float diagonal=jacobian[13*column+column];
            for(unsigned row=column+1;row<13;++row) {
                const float factor=jacobian[13*row+column]/diagonal;
                jacobian[13*row+column]=0.0f;
                for(unsigned entry=column+1;entry<13;++entry)
                    jacobian[13*row+entry]-=factor*jacobian[13*column+entry];
                correction[row]-=factor*correction[column];
            }
        }
        if(singular) {
            trace.failure="iteration "+std::to_string(iteration)+
                " local Jacobian pivot rejected at column "+std::to_string(failedColumn)+
                " magnitude="+number(failedPivot);
            trace.failureDetail=trace.failure;
            return trace;
        }
        for(int row=12;row>=0;--row) {
            float value=correction[static_cast<unsigned>(row)];
            for(unsigned column=static_cast<unsigned>(row)+1;column<13;++column)
                value-=jacobian[13*static_cast<unsigned>(row)+column]*correction[column];
            correction[static_cast<unsigned>(row)]=value/
                jacobian[13*static_cast<unsigned>(row)+static_cast<unsigned>(row)];
            if(!std::isfinite(correction[static_cast<unsigned>(row)])) {
                trace.failure="iteration "+std::to_string(iteration)+
                    " local solve produced nonfinite correction row "+std::to_string(row);
                trace.failureDetail=trace.failure;
                return trace;
            }
        }
        std::ostringstream systemLog;
        systemLog<<"iteration="<<iteration<<" min_pivot="<<number(minimumPivot)
            <<" pivot_column="<<minimumPivotColumn<<" R=[";
        for(unsigned i=0;i<13;++i) systemLog<<(i?",":"")<<number(residual[i]);
        systemLog<<"] delta=[";
        for(unsigned i=0;i<13;++i) systemLog<<(i?",":"")<<number(correction[i]);
        systemLog<<"]";
        std::ostringstream stepLog;
        stepLog<<"iteration="<<iteration<<" residual="<<number(residualNorm)
            <<" tolerance="<<number(tolerance)<<" step=";
        float step=1.0f;
        bool acceptedStep=false;
        // Mirror production generic projection's current bounded 16-trial
        // backtracking budget. Material residuals/tolerances are unchanged.
        for(unsigned backtrack=0;backtrack<16;++backtrack) {
            std::array<float,13> trial=candidate;
            for(unsigned state=0;state<13;++state)
                trial[state]=candidate[state]+step*correction[state];
            float trialNorm=0.0f;
            unsigned worstRow=0;
            float worstValue=0.0f;
            bool valid=true;
            std::string invalid;
            for(unsigned row=0;row<13;++row) {
                float value=0.0f;
                if(!evaluateRow(program.implicitResiduals.at(row),trial,
                        "backtrack residual["+std::to_string(row)+"]",value)) {
                    valid=false;invalid=trace.failure;trace.failure.clear();break;
                }
                if(std::abs(value)>trialNorm) {
                    trialNorm=std::abs(value);
                    worstRow=row;
                    worstValue=value;
                }
            }
            const float armijoLimit=(1.0f-1.0e-4f*step)*residualNorm;
            if(valid && trialNorm<=armijoLimit) {
                stepLog<<number(step)<<" accepted trial="<<number(trialNorm)
                    <<" row="<<worstRow<<" value="<<number(worstValue)
                    <<" armijo="<<number(armijoLimit);
                if(iteration==0u) {
                    trace.firstAcceptedBacktrack=backtrack+1u;
                    trace.firstAcceptedStep=step;
                }
                candidate=trial;acceptedStep=true;break;
            }
            stepLog<<number(step)<<":";
            if(!valid) stepLog<<invalid;
            else stepLog<<"trial="<<number(trialNorm)<<" row="<<worstRow
                <<" value="<<number(worstValue)<<">armijo="<<number(armijoLimit);
            stepLog<<" ";
            step*=0.5f;
        }
        if(!acceptedStep) {
            trace.failure="iteration "+std::to_string(iteration)+
                " rejected all 16 production line-search trials";
            trace.failureDetail=systemLog.str()+"; "+stepLog.str();
            return trace;
        }
    }
    std::array<float,13> finalResidual{};
    for(unsigned row=0;row<13;++row) {
        if(!evaluateRow(program.implicitResiduals.at(row),candidate,
                "final residual["+std::to_string(row)+"]",finalResidual[row])) return trace;
    }
    float residualNorm=0.0f,stateNorm=0.0f;
    for(unsigned i=0;i<13;++i) {
        residualNorm=std::max(residualNorm,std::abs(finalResidual[i]));
        stateNorm=std::max(stateNorm,std::abs(candidate[i]));
    }
    const float tolerance=2.0e-5f*(1.0f+stateNorm);
    trace.valid=residualNorm<=tolerance;
    trace.finalResidual=residualNorm;
    trace.finalTolerance=tolerance;
    trace.convergedIteration=16u;
    if(!trace.valid) {
        trace.failure="local Newton exhausted 16 iterations residual="+
            number(residualNorm)+" tolerance="+number(tolerance);
        trace.failureDetail=trace.failure;
    }
    return trace;
}

std::array<double,13> compiledResiduals(const ConstitutiveProgram& program,
    const paper::Matrix& f,const paper::State& accepted,const paper::State& candidate) {
    std::array<double,13> result{};
    const paper::Matrix zero{};
    for (unsigned i=0;i<13;++i)
        result[i]=evaluate(program.implicitResiduals.at(i),program.material,f,zero,accepted,candidate);
    return result;
}

std::array<double,169> compiledJacobian(const ConstitutiveProgram& program,
    const paper::Matrix& f,const paper::State& accepted,const paper::State& candidate) {
    std::array<double,169> result{};
    const paper::Matrix zero{};
    for (unsigned i=0;i<169;++i)
        result[i]=evaluate(program.implicitJacobians.at(i),program.material,f,zero,accepted,candidate);
    return result;
}

double maxAbs(const std::array<double,13>& values) {
    double result=0.0; for(double value:values)result=std::max(result,std::abs(value)); return result;
}

paper::Matrix compiledAlgorithmicTangent(const ConstitutiveProgram& program,
    const paper::Matrix& f,const paper::Matrix& h,const paper::State& accepted,
    const paper::State& candidate) {
    std::array<double,169> jac=compiledJacobian(program,f,accepted,candidate);
    std::array<double,13> rhs{},stateDirection{};
    const paper::Matrix zero{};
    for(unsigned state=0;state<13;++state) {
        rhs[state]=-evaluate(program.implicitDeformationDirections.at(state),
            program.material,f,h,accepted,candidate);
    }
    require(paper::solveLinear(jac,rhs,stateDirection),"compiled implicit material Jacobian is singular");
    paper::Matrix tangent{};
    for(unsigned component=0;component<9;++component) {
        tangent[component]=evaluate(program.tangentVector.at(component),
            program.material,f,h,accepted,candidate);
        for(unsigned state=0;state<13;++state) {
            const double derivative=evaluate(program.stressStateDerivatives.at(component*13+state),
                program.material,f,zero,accepted,candidate);
            tangent[component]+=derivative*stateDirection[state];
        }
    }
    return tangent;
}

void checkCompiledStatePrograms(const ConstitutiveProgram& program,paper::MaterialKind kind) {
    const paper::State initial{};
    const paper::Matrix identity=paper::identity();
    const auto identityResidual=compiledResiduals(program,identity,initial,initial);
    const double identityPhysicalResidual=maxAbs(identityResidual)/kNumericalResidualScale;
    require(identityPhysicalResidual<2.0e-7,
        "compiled local material residual, normalized to physical equations, is not finite/equilibrated at identity: "+
        number(identityPhysicalResidual));
    const auto identityJacobian=compiledJacobian(program,identity,initial,initial);
    for(double value:identityJacobian) require(std::isfinite(value),
        "compiled local material Jacobian is nonfinite at identity");
    std::array<double,169> probe=identityJacobian;
    std::array<double,13> rhs{},solution{}; rhs[0]=1.0;
    require(paper::solveLinear(probe,rhs,solution),"compiled identity local Jacobian is singular");
    float minimumDivision=std::numeric_limits<float>::infinity();
    const auto fp32Residual=compiledFP32Residuals(program,identity,initial,initial,minimumDivision);
    float fp32ResidualNorm=0.0f;
    for(float value:fp32Residual)fp32ResidualNorm=std::max(fp32ResidualNorm,std::abs(value));
    require(fp32ResidualNorm<=2.0e-5f*(1.0f+2.0e-6f),
        "production-scaled FP32 identity residual misses the local convergence tolerance: "+
        number(fp32ResidualNorm));
    require(fp32ResidualNorm/static_cast<float>(kNumericalResidualScale)<2.0e-6f,
        "normalized FP32 identity residual did not meet the tighter constitutive accuracy target");
    auto fp32Jacobian=compiledFP32Jacobian(program,identity,initial,initial,minimumDivision);
    float minimumPivot=std::numeric_limits<float>::infinity();
    require(productionLocalSolve(fp32Jacobian,{},minimumPivot),
        "production FP32 local solver rejects the identity material Jacobian");

    paper::Matrix loadedF=identity; loadedF[0]=0.994;
    const auto loaded=paper::project(kind,loadedF,initial);
    const auto loadedResidual=compiledResiduals(program,loadedF,initial,loaded.state);
    const auto oracleLoadedResidual=paper::residual(paper::parameters(kind),loadedF,initial,loaded.state);
    double normalizedResidualDifference=0.0;
    for(unsigned i=0;i<13;++i)
        normalizedResidualDifference=std::max(normalizedResidualDifference,
            std::abs(loadedResidual[i]/kNumericalResidualScale-oracleLoadedResidual[i]));
    require(normalizedResidualDifference<1.0e-6,
        "compiled residual divided by its numerical scale disagrees with the independent physical return map: "+
        number(normalizedResidualDifference));
    const auto loadedFP32Residual=compiledFP32Residuals(program,loadedF,initial,loaded.state,minimumDivision);
    float loadedFP32Norm=0.0f;
    for(float value:loadedFP32Residual)loadedFP32Norm=std::max(loadedFP32Norm,std::abs(value));
    require(loadedFP32Norm/static_cast<float>(kNumericalResidualScale)<1.0e-5f,
        "normalized FP32 residual at the independent FP64 returned state is unexpectedly large: "+
        number(loadedFP32Norm/static_cast<float>(kNumericalResidualScale)));
    const paper::Matrix stressDirection{1,0,0,0,0,0,0,0,0};
    paper::Matrix compiledTangent=compiledAlgorithmicTangent(
        program,loadedF,stressDirection,initial,loaded.state);
    paper::Matrix compiledStress{};
    for(unsigned component=0;component<9;++component) {
        compiledStress[component]=evaluate(program.stress.at(component),program.material,
            loadedF,{},initial,loaded.state)*1.0e-6;
        compiledTangent[component]*=1.0e-6;
    }
    const paper::Matrix referenceStress=paper::firstPiola(paper::parameters(kind),loadedF,
        paper::Vector6{loaded.state[0],loaded.state[1],loaded.state[2],
            loaded.state[3],loaded.state[4],loaded.state[5]});
    double stressDifference=0.0,stressScale=1.0;
    for(unsigned i=0;i<9;++i) {
        stressDifference=std::max(stressDifference,std::abs(compiledStress[i]-referenceStress[i]));
        stressScale=std::max({stressScale,std::abs(compiledStress[i]),std::abs(referenceStress[i])});
    }
    require(stressDifference/stressScale<3.0e-5,
        "compiled candidate stress differs from independent FP64 reference: "+
        number(stressDifference/stressScale));
    const paper::Matrix fdTangent=paper::directionalTangent(
        kind,loadedF,stressDirection,initial,1.0e-5);
    double tangentDifference=0.0,tangentScale=1.0;
    for(unsigned i=0;i<9;++i) {
        tangentDifference=std::max(tangentDifference,std::abs(compiledTangent[i]-fdTangent[i]));
        tangentScale=std::max({tangentScale,std::abs(compiledTangent[i]),std::abs(fdTangent[i])});
    }
    require(tangentDifference/tangentScale<3.0e-3,
        "compiled implicit algorithmic tangent differs from independent FD reference: "+
        number(tangentDifference/tangentScale));
    std::cout<<program.material.name<<" scaled_identity_residual_fp32="<<fp32ResidualNorm
        <<" physical_identity_residual_fp32="<<fp32ResidualNorm/static_cast<float>(kNumericalResidualScale)
        <<" local_min_pivot_fp32="<<minimumPivot<<" min_divisor_fp32="<<minimumDivision
        <<" scaled_loaded_residual_fp32="<<loadedFP32Norm
        <<" physical_loaded_residual_fp32="<<loadedFP32Norm/static_cast<float>(kNumericalResidualScale)<<"\n";
}

void checkProductionMixedProjectionTrace(const MaterialProgram& material) {
    const auto compiled=detail::compileConstitutive(material,NM_EXPRESSION_STACK_CAPACITY);
    require(compiled.succeeded(),"cannot compile mixed-projection FP32 trace: "+diagnostics(compiled.diagnostics));
    const paper::Matrix direction{.013,.007,.004,-.002,-.009,.003,.002,-.004,.006};
    paper::Matrix center=paper::identity();
    center[0]=material.name.find("liner")!=std::string::npos?.994:.9975;
    constexpr float epsilon=1.0e-3f;
    const paper::MaterialKind kind=material.name.find("liner")!=std::string::npos
        ?paper::MaterialKind::liner:paper::MaterialKind::medium;
    for(const int sign:{-1,1}) {
        paper::Matrix f=center;
        for(unsigned i=0;i<9;++i)
            f[i]=static_cast<float>(center[i]+double(sign)*double(epsilon)*direction[i]);
        const paper::State initial{};
        const auto oracle=paper::project(kind,f,initial);
        const auto compiledResidual=compiledResiduals(compiled.program,f,initial,oracle.state);
        const auto oracleResidual=paper::residual(paper::parameters(kind),f,initial,oracle.state);
        double normalizedResidualDifference=0.0;
        for(unsigned i=0;i<13;++i)
            normalizedResidualDifference=std::max(normalizedResidualDifference,
                std::abs(compiledResidual[i]/kNumericalResidualScale-oracleResidual[i]));
        require(normalizedResidualDifference<1.0e-6,
            "compiled mixed residual, divided by numerical scale, disagrees with the independent physical law: "+
            material.name+" "+(sign<0?"minus":"plus")+
            " error="+number(normalizedResidualDifference));
        const auto trace=traceProductionProjection(compiled.program,f,paper::State{});
        std::cout<<"production_fp32_projection_trace material="<<material.name
            <<" perturbation="<<(sign<0?"minus":"plus")
            <<" valid="<<(trace.valid?"true":"false")
            <<" converged_iteration="<<trace.convergedIteration
            <<" residual="<<number(trace.finalResidual)
            <<" physical_residual="<<number(trace.finalResidual/kNumericalResidualScale)
            <<" oracle_law_residual_difference="<<number(normalizedResidualDifference)
            <<" tolerance="<<number(trace.finalTolerance)
            <<" first_backtrack="<<trace.firstAcceptedBacktrack
            <<" first_step="<<number(trace.firstAcceptedStep)
            <<" failure="<<(trace.failure.empty()?"none":trace.failure)<<"\n";
        if(!trace.valid) std::cout<<"  failed_iteration: "<<trace.failureDetail<<"\n";
        require(trace.valid,"compiled FP32 local projection does not match production result: "+
            material.name+" "+(sign<0?"minus":"plus")+" "+trace.failure);
        require(trace.finalResidual<=trace.finalTolerance,
            "production-scaled local Newton trace returned without meeting its convergence threshold");
        require(trace.finalResidual/kNumericalResidualScale<2.0e-6f,
            "mixed FP32 production trace did not achieve the tighter normalized physical residual: "+
            material.name+" "+(sign<0?"minus":"plus"));
    }
}

WorldSource oneTetWorld(const MaterialProgram& material) {
    WorldSource world;
    world.materials={material};
    world.gravity={0.0,0.0,0.0};
    world.environmentCount=1u;
    world.frameTimestep=1.0e-4;
    ObjectSource object;
    object.name="local_material_budget_fixture";
    object.representation=Representation::fem;
    object.deformableContact=false;
    object.deformableSelfContact=false;
    object.mixedFEM=false;
    object.characteristicLength=0.01;
    object.femNodes={{{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,.01}}};
    object.tetrahedra={{{0,1,2,3}}};
    world.objects={object};
    return world;
}

void checkLocalMaterialBudget(const MaterialProgram& material) {
    const WorldSource source=oneTetWorld(material);
    CompileOptions defaultOptions;
    defaultOptions.maximumRateExponent=0u;
    defaultOptions.emitSpecializedMetal=false;
    const auto defaultCook=compileWorld(source,defaultOptions);
    require(defaultCook.succeeded(),"default local material budget cook failed: "+diagnostics(defaultCook.diagnostics));
    require(defaultCook.world.materials.at(0).localNewtonIterations==8u &&
        defaultCook.world.constitutive.at(0).gpu.localNewtonIterations==8u,
        "omitted local material budget did not retain the historical default of eight");

    CompileOptions extendedOptions=defaultOptions;
    extendedOptions.localMaterialNewtonIterations=16u;
    const auto extendedCook=compileWorld(source,extendedOptions);
    require(extendedCook.succeeded(),"16-iteration local material budget cook failed: "+diagnostics(extendedCook.diagnostics));
    require(extendedCook.world.materials.at(0).localNewtonIterations==16u &&
        extendedCook.world.constitutive.at(0).gpu.localNewtonIterations==16u,
        "selected local material budget was not copied into both cooked material views");
    require(defaultCook.world.fingerprint!=extendedCook.world.fingerprint,
        "compiled fingerprint did not bind the local material Newton budget");
    require(defaultCook.world.physicsFingerprint!=extendedCook.world.physicsFingerprint,
        "physics fingerprint did not bind the local material Newton budget");

    for (const std::uint32_t invalid:{0u,17u}) {
        CompileOptions invalidOptions=defaultOptions;
        invalidOptions.localMaterialNewtonIterations=invalid;
        const auto rejected=compileWorld(source,invalidOptions);
        require(!rejected.succeeded(),"invalid local material Newton budget was accepted");
        require(diagnostics(rejected.diagnostics).find("local material Newton iteration budget")!=std::string::npos,
            "invalid local material Newton budget rejection lacks targeted diagnostic");
    }
}

void checkParameter(const MaterialProgram& material,const std::string& name,double expected) {
    double actual=0.0;
    for (const auto& p:material.parameters) if (p.name==name) {
        actual=p.defaultValue;
        require(!p.identifiable,"paper parameter is identifiable: "+name);
        require(p.lower==p.upper,"paper parameter has a fit interval: "+name);
        require(std::abs(actual-expected)<=2.0e-12*std::max(1.0,std::abs(expected)),
            "parameter differs from source-derived oracle: "+name+" actual="+number(actual)+
            " expected="+number(expected));
        return;
    }
    throw std::runtime_error("missing authored material parameter "+name);
}

void checkHillSurface(const paper::Parameters& p) {
    for (unsigned axis=0;axis<3;++axis) {
        paper::Vector6 stress{};
        stress[axis]=-p.compressiveYield[axis]/p.sigmaReference;
        const double q=paper::hillEquivalent(p,stress);
        require(std::abs(q-1.0)<2.0e-8,"Table 3 compression threshold mismatch: "+
            std::string(p.name)+" axis="+std::to_string(axis)+" q="+number(q));
    }
    // Eq. 2 maps L -> tau23, M -> tau31, N -> tau12.
    const std::array<std::pair<unsigned,double>,3> shear{{{3,p.tau23},{4,p.tau13},{5,p.tau12}}};
    for (const auto [component,yield]:shear) {
        paper::Vector6 stress{};
        stress[component]=yield/p.sigmaReference;
        const double q=paper::hillEquivalent(p,stress);
        require(std::abs(q-1.0)<2.0e-8,"Table 3 shear threshold/index mismatch: "+
            std::string(p.name)+" component="+std::to_string(component)+" q="+number(q));
    }
    for (const paper::Vector6 stress:std::array<paper::Vector6,2>{
             paper::Vector6{-.7,.2,.5,.11,-.03,.06},
             paper::Vector6{.4,-.8,.4,-.09,.05,.02}}) {
        const paper::Vector6 flow=paper::hillFlow(p,stress);
        const double work=stress[0]*flow[0]+stress[1]*flow[1]+stress[2]*flow[2]
            +2.0*(stress[3]*flow[3]+stress[4]*flow[4]+stress[5]*flow[5]);
        const double q=paper::hillEquivalent(p,stress);
        require(work>=-1.0e-12 && std::abs(work-q)<2.0e-9*std::max(1.0,q),
            "associated Hill flow fails positive Euler plastic-work identity: "+std::string(p.name));
    }
}

void checkAuthoredSource(const MaterialProgram& m,const paper::Parameters& p) {
    require(m.name==p.name,"material name differs from oracle identity");
    require(m.internalState.size()==13,"Hill material must retain 13 generic local states");
    require(m.stateImplicitRoots.size()==13,"paper-law state root vector is incomplete");
    require(m.stateUpdateRoots.size()==13,"state update vector layout is incomplete");
    for (unsigned i=0;i<13;++i) {
        require(m.stateImplicitRoots[i]!=NM_INVALID_INDEX,
            "implicit equation absent for state "+m.internalState[i].name);
    }
    for (unsigned i=0;i<6;++i)
        require(m.stateUpdateRoots[i]==NM_INVALID_INDEX,
            "plastic strain must not have an explicit Newton seed");
    for (unsigned i=6;i<12;++i) {
        require(m.stateUpdateRoots[i]!=NM_INVALID_INDEX,
            "implicit stress is missing its explicit Newton seed");
        require(m.internalState[i].name.starts_with("sbar"),
            "Newton-seeded state is not a normalized stress");
    }
    require(m.stateUpdateRoots[12]==NM_INVALID_INDEX,
        "accumulated multiplier must remain in generic implicit return map");
    checkParameter(m,"density",p.density);
    checkParameter(m,"c11",p.c[0]*1.0e6); checkParameter(m,"c22",p.c[1]*1.0e6);
    checkParameter(m,"c33",p.c[2]*1.0e6); checkParameter(m,"c12",p.c[3]*1.0e6);
    checkParameter(m,"c13",p.c[4]*1.0e6); checkParameter(m,"c23",p.c[5]*1.0e6);
    checkParameter(m,"g12",p.shear[0]*1.0e6); checkParameter(m,"g13",p.shear[1]*1.0e6);
    checkParameter(m,"g23",p.shear[2]*1.0e6);
    checkParameter(m,"sigma_ref",p.sigmaReference*1.0e6);
    checkParameter(m,"hill_F",p.hill[0]); checkParameter(m,"hill_G",p.hill[1]);
    checkParameter(m,"hill_H",p.hill[2]); checkParameter(m,"hill_L",p.hill[3]);
    checkParameter(m,"hill_M",p.hill[4]); checkParameter(m,"hill_N",p.hill[5]);
    checkParameter(m,"q_floor",p.qFloor);
    checkParameter(m,"fb_epsilon",p.fischerBurmeisterEpsilon);
    checkParameter(m,"residual_scale",kNumericalResidualScale);
    checkHillSurface(p);
    const auto compiled=detail::compileConstitutive(m,NM_EXPRESSION_STACK_CAPACITY);
    require(compiled.succeeded(),"generic Hill implicit material did not compile: "+diagnostics(compiled.diagnostics));
    require(compiled.program.gpu.constitutiveKind==NM_CONSTITUTIVE_BYTECODE &&
            compiled.program.gpu.projectionKind==NM_MATERIAL_PROJECTION_GENERIC,
        "orthotropic Hill material was routed into a specialized isotropic projection");
    require(compiled.program.implicitResiduals.size()==13 &&
            compiled.program.implicitJacobians.size()==169 &&
            compiled.program.stressStateDerivatives.size()==117,
        "compiled implicit Hill residual/Jacobian/tangent layout is incomplete");
    checkCompiledStatePrograms(compiled.program,
        p.name==paper::liner.name?paper::MaterialKind::liner:paper::MaterialKind::medium);
}

struct CaseResult {
    paper::Result loaded;
    paper::Result unloaded;
    double tangentRefinementError=0.0;
};

CaseResult checkReturnMapping(paper::MaterialKind kind) {
    paper::Matrix f=paper::identity();
    const double compression=kind==paper::MaterialKind::liner?0.994:0.9975;
    f[0]=compression; // Source-threshold overrun, chosen below reverse-yield on full release.
    const paper::State initial{};
    const auto loaded=paper::project(kind,f,initial);
    const auto& p=paper::parameters(kind);
    require(loaded.state[12]>1.0e-7,"published-strength compression produced no plastic flow: "+std::string(p.name));
    require(loaded.plasticWorkMPa>=-1.0e-9,"plastic work became negative in compression: "+std::string(p.name));
    require(std::abs(loaded.equivalentStress-1.0)<2.0e-3,
        "returned stress misses compression-calibrated Hill surface: "+std::string(p.name)+
        " q="+number(loaded.equivalentStress));
    paper::Matrix identity=paper::identity();
    const auto unloaded=paper::project(kind,identity,loaded.state);
    double retainedError=0.0;
    for (unsigned i=0;i<6;++i) retainedError=std::max(retainedError,
        std::abs(unloaded.state[i]-loaded.state[i]));
    require(unloaded.state[12]>=loaded.state[12]-1.0e-11,
        "accumulated plastic multiplier decreased on unload: "+std::string(p.name));
    require(retainedError<1.0e-4,"unload erased retained plastic Green strain: "+std::string(p.name)+
        " error="+number(retainedError));
    const double unloadIncrement=unloaded.state[12]-loaded.state[12];
    require(unloadIncrement<1.0e-4,"unload generated material plastic flow: "+std::string(p.name)+
        " increment="+number(unloadIncrement));

    const paper::Matrix direction{1,0,0,0,0,0,0,0,0};
    const paper::Matrix coarse=paper::directionalTangent(kind,f,direction,initial,2.0e-5);
    const paper::Matrix fine=paper::directionalTangent(kind,f,direction,initial,1.0e-5);
    double difference=0.0,scale=1.0;
    for (unsigned i=0;i<9;++i) {
        difference=std::max(difference,std::abs(coarse[i]-fine[i]));
        scale=std::max({scale,std::abs(coarse[i]),std::abs(fine[i])});
    }
    const double refinement=difference/scale;
    require(refinement<2.0e-3,"directional stress tangent failed finite-difference refinement: "+
        std::string(p.name)+" relative="+number(refinement));
    return {loaded,unloaded,refinement};
}

MaterialProgram parse(const std::filesystem::path& path) {
    const auto result=parseMatterFile(path);
    if (!result.succeeded()) throw std::runtime_error("parse failure for "+path.string()+": "+
        diagnostics(result.diagnostics));
    return result.material;
}

void checkDualRootInitialization() {
    const std::string prefix="material dual_root { parameter density : kg/m^3 = 1; parameter p : Pa = 1; state x : one = 0; model generic; energy = p; ";
    const std::string suffix=" }";
    const auto updateFirst=parseMatter(prefix+
        "update x = 2; implicit x = next(x)-x-1;"+suffix);
    require(updateFirst.succeeded(),"update then implicit must parse: "+diagnostics(updateFirst.diagnostics));
    const auto implicitFirst=parseMatter(prefix+
        "implicit x = next(x)-x-1; update x = 2;"+suffix);
    require(implicitFirst.succeeded(),"implicit then update must parse: "+diagnostics(implicitFirst.diagnostics));
    for (const ParseResult* parsed:std::array<const ParseResult*,2>{&updateFirst,&implicitFirst}) {
        require(parsed->material.stateUpdateRoots.size()==1u &&
                parsed->material.stateImplicitRoots.size()==1u &&
                parsed->material.stateUpdateRoots[0]!=NM_INVALID_INDEX &&
                parsed->material.stateImplicitRoots[0]!=NM_INVALID_INDEX,
            "dual-root declaration lost its Newton initializer or physical residual");
    }

    const auto compiled=detail::compileConstitutive(
        updateFirst.material,NM_EXPRESSION_STACK_CAPACITY);
    require(compiled.succeeded(),"dual-root generic state did not compile: "+diagnostics(compiled.diagnostics));
    require(compiled.program.implicitResiduals.size()==1u,
        "dual-root material did not retain its implicit equation");
    paper::State accepted{},candidate{};
    candidate[0]=3.0;
    const double residualValue=evaluate(compiled.program.implicitResiduals[0],
        compiled.program.material,paper::identity(),{},accepted,candidate);
    require(std::abs(residualValue-2.0)<1.0e-12,
        "Newton initializer replaced the authored implicit residual");

    const auto duplicateUpdate=parseMatter(prefix+
        "update x = 2; update x = 3; implicit x = next(x)-x-1;"+suffix);
    require(!duplicateUpdate.succeeded() &&
            diagnostics(duplicateUpdate.diagnostics).find("duplicate update for state 'x'")!=std::string::npos,
        "duplicate explicit initializer was not rejected");
    const auto duplicateImplicit=parseMatter(prefix+
        "update x = 2; implicit x = next(x)-x-1; implicit x = next(x)-x-2;"+suffix);
    require(!duplicateImplicit.succeeded() &&
            diagnostics(duplicateImplicit.diagnostics).find("duplicate implicit residual for state 'x'")!=std::string::npos,
        "duplicate implicit equation was not rejected");
}

void print(const paper::Parameters& p,const CaseResult& result) {
    std::cout<<"{\"material\":\""<<p.name<<"\",\"case\":\"md_compression_then_unload\","
        <<"\"F11\":"<<number(p.name==paper::liner.name?0.994:0.9975)<<","
        <<"\"iterations\":"<<result.loaded.iterations<<",\"hill_q_loaded\":"<<number(result.loaded.equivalentStress)
        <<",\"lambda_loaded\":"<<number(result.loaded.state[12])
        <<",\"plastic_work_mpa\":"<<number(result.loaded.plasticWorkMPa)
        <<",\"plastic_ep11\":"<<number(result.loaded.state[0])
        <<",\"plastic_ep22\":"<<number(result.loaded.state[1])
        <<",\"plastic_ep33\":"<<number(result.loaded.state[2])
        <<",\"unload_lambda_increment\":"<<number(result.unloaded.state[12]-result.loaded.state[12])
        <<",\"tangent_fd_refinement_relative_error\":"<<number(result.tangentRefinementError)<<"}\n";
}
}

int main(int argc,char** argv) {
    try {
        const auto root=std::filesystem::path(__FILE__).parent_path().parent_path();
        const auto linerPath=argc>1?std::filesystem::path(argv[1]):root/"materials/hajali2009_liner_hill_ideal.nmatter";
        const auto mediumPath=argc>2?std::filesystem::path(argv[2]):root/"materials/hajali2009_medium_hill_ideal.nmatter";
        require(argc<=3,"expected optional liner and medium material paths");
        const MaterialProgram linerSource=parse(linerPath);
        const MaterialProgram mediumSource=parse(mediumPath);
        checkDualRootInitialization();
        checkAuthoredSource(linerSource,paper::liner);
        checkAuthoredSource(mediumSource,paper::medium);
        checkProductionMixedProjectionTrace(linerSource);
        checkProductionMixedProjectionTrace(mediumSource);
        checkLocalMaterialBudget(linerSource);
        print(paper::liner,checkReturnMapping(paper::MaterialKind::liner));
        print(paper::medium,checkReturnMapping(paper::MaterialKind::medium));
        std::cout<<"checks="<<checks<<" status=pass scope=independent_fp64_oracle_and_matter_compiler_only\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"cardboard_paper_material_check: "<<error.what()<<"\n";
        return 1;
    }
}
