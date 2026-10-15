#include "numi/matter/detail.hpp"
#include "cardboard_paper_reference.hpp"
#include "cardboard_paper_projection_scaling.hpp"
#include "cardboard_paper_material_selection.hpp"
#include "cardboard_asymmetric_c1_material_oracle.hpp"
#include "numi/matter/paper_compensated_bytecode.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
using namespace numi::matter;
namespace paper=numi_cardboard_paper;
namespace c1oracle=numi_cardboard_asymmetric_c1_oracle;
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

template<class State>
double evaluate(const ScalarBytecode& bytecode,const MaterialProgram& material,
                const paper::Matrix& f,const paper::Matrix& h,
                const State& accepted,const State& candidate) {
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
        case NM_EXPR_SELECT_NONNEGATIVE: { const double negative=pop(),positive=pop(),condition=pop(); value=condition>=0?positive:negative; break; }
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
        case NM_EXPR_SELECT_NONNEGATIVE:{const float negative=pop(),positive=pop(),condition=pop();value=condition>=0?positive:negative;break;}
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

bool productionLocalSolve(std::array<float,169> matrix,std::array<float,13>& rhs,
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

void checkParameter(const MaterialProgram& material,const std::string& name,
                    double expected);

double parameterValue(const MaterialProgram& material,const std::string& name) {
    const auto found=std::find_if(material.parameters.begin(),material.parameters.end(),
        [&](const Parameter& value) { return value.name==name; });
    require(found!=material.parameters.end(),"material is missing parameter "+name);
    return found->defaultValue;
}

struct CompiledPoint64 {
    std::array<double,15> state{};
    paper::Matrix firstPiolaMPa{};
};

std::array<double,15> compiledResiduals15(const ConstitutiveProgram& program,
    const paper::Matrix& f,const std::array<double,15>& accepted,
    const std::array<double,15>& candidate) {
    std::array<double,15> result{};
    const paper::Matrix zero{};
    for(unsigned i=0;i<result.size();++i)
        result[i]=evaluate(program.implicitResiduals.at(i),program.material,
                           f,zero,accepted,candidate);
    return result;
}

std::array<double,225> compiledJacobian15(const ConstitutiveProgram& program,
    const paper::Matrix& f,const std::array<double,15>& accepted,
    const std::array<double,15>& candidate) {
    std::array<double,225> result{};
    const paper::Matrix zero{};
    for(unsigned i=0;i<result.size();++i)
        result[i]=evaluate(program.implicitJacobians.at(i),program.material,
                           f,zero,accepted,candidate);
    return result;
}

double maxAbs(const std::array<double,15>& values) {
    double result=0.0;
    for(double value:values) result=std::max(result,std::abs(value));
    return result;
}

bool solveLinear15(std::array<double,225> matrix,std::array<double,15> rhs,
                   std::array<double,15>& solution) {
    constexpr unsigned n=15u;
    for(unsigned column=0;column<n;++column) {
        unsigned pivot=column;
        double magnitude=std::abs(matrix[column*n+column]);
        for(unsigned row=column+1u;row<n;++row) {
            const double candidate=std::abs(matrix[row*n+column]);
            if(candidate>magnitude) { magnitude=candidate; pivot=row; }
        }
        if(!std::isfinite(magnitude)||magnitude<1.0e-14) return false;
        if(pivot!=column) {
            for(unsigned c=column;c<n;++c)
                std::swap(matrix[column*n+c],matrix[pivot*n+c]);
            std::swap(rhs[column],rhs[pivot]);
        }
        const double diagonal=matrix[column*n+column];
        for(unsigned row=column+1u;row<n;++row) {
            const double factor=matrix[row*n+column]/diagonal;
            for(unsigned c=column+1u;c<n;++c)
                matrix[row*n+c]-=factor*matrix[column*n+c];
            rhs[row]-=factor*rhs[column];
        }
    }
    for(int row=static_cast<int>(n)-1;row>=0;--row) {
        double value=rhs[static_cast<unsigned>(row)];
        for(unsigned c=static_cast<unsigned>(row)+1u;c<n;++c)
            value-=matrix[static_cast<unsigned>(row)*n+c]*solution[c];
        solution[static_cast<unsigned>(row)]=
            value/matrix[static_cast<unsigned>(row)*n+static_cast<unsigned>(row)];
        if(!std::isfinite(solution[static_cast<unsigned>(row)])) return false;
    }
    return true;
}

CompiledPoint64 projectCompiled64(const ConstitutiveProgram& program,
    const paper::Matrix& f,const std::array<double,15>& accepted) {
    constexpr unsigned kStates=15u;
    std::array<double,kStates> candidate{};
    const std::array<double,kStates> zeroState{};
    const paper::Matrix zero{};
    for(unsigned state=0;state<kStates;++state)
        candidate[state]=evaluate(program.stateUpdates.at(state),program.material,
            f,zero,accepted,zeroState);
    const auto residualAt=[&](const std::array<double,kStates>& value) {
        return compiledResiduals15(program,f,accepted,value);
    };
    bool converged=false;
    for(unsigned iteration=0;iteration<256u;++iteration) {
        const auto residual=residualAt(candidate);
        const double norm=maxAbs(residual);
        if(norm<2.0e-13) { converged=true; break; }
        auto jacobian=compiledJacobian15(program,f,accepted,candidate);
        std::array<double,kStates> rhs{},correction{};
        for(unsigned row=0;row<kStates;++row) rhs[row]=-residual[row];
        require(solveLinear15(jacobian,rhs,correction),
            "bilinear source-fit local Newton Jacobian is singular");
        double step=1.0;
        bool acceptedStep=false;
        double lastTrialNorm=std::numeric_limits<double>::infinity();
        for(unsigned backtrack=0;backtrack<24u;++backtrack) {
            std::array<double,kStates> trial=candidate;
            for(unsigned state=0;state<kStates;++state)
                trial[state]+=step*correction[state];
            const double trialNorm=maxAbs(residualAt(trial));
            lastTrialNorm=trialNorm;
            if(std::isfinite(trialNorm) && trialNorm<norm) {
                candidate=trial;
                acceptedStep=true;
                break;
            }
            step*=0.5;
        }
        if(!acceptedStep) {
            std::ostringstream detail;
            detail<<"bilinear source-fit local Newton could not decrease residual at F11="
                  <<number(f[0])<<" F22="<<number(f[4])<<" F33="<<number(f[8])
                  <<" norm="<<number(norm)<<" candidate=";
            for(double value:candidate) detail<<number(value)<<',';
            detail<<" residuals=";
            for(double value:residual) detail<<number(value)<<',';
            detail<<" last_step="<<number(step)<<" last_trial_norm="<<number(lastTrialNorm);
            throw std::runtime_error(detail.str());
        }
    }
    const double finalNorm=maxAbs(residualAt(candidate));
    if(!converged) converged=finalNorm<2.0e-13;
    if(!converged) {
        std::ostringstream detail;
        detail<<"bilinear source-fit local projection did not converge; residual="
              <<number(finalNorm)<<" at F11="<<number(f[0])<<" F22="<<number(f[4])
              <<" F33="<<number(f[8])<<" candidate=";
        for(double value:candidate) detail<<number(value)<<',';
        detail<<" residuals=";
        for(double value:residualAt(candidate)) detail<<number(value)<<',';
        throw std::runtime_error(detail.str());
    }
    CompiledPoint64 result;
    result.state=candidate;
    for(unsigned component=0;component<9u;++component)
        result.firstPiolaMPa[component]=evaluate(program.stress.at(component),
            program.material,f,zero,accepted,candidate)*1.0e-6;
    return result;
}

struct UniaxialPoint {
    double stressMPa=0.0;
    std::array<double,15> state{};
    double lateralResidualMPa=0.0;
};

std::array<double,9> compiledAlgorithmicTangent15(
    const ConstitutiveProgram& program,const paper::Matrix& f,
    const paper::Matrix& h,const std::array<double,15>& accepted,
    const std::array<double,15>& candidate);

UniaxialPoint stressFreeUniaxialPoint(const ConstitutiveProgram& program,
    const unsigned axis,const double engineeringStrain) {
    require(axis<2u,"source-fit uniaxial test only admits MD/CD axes");
    paper::Matrix f=paper::identity();
    if(axis==0u) f[0]=1.0+engineeringStrain;
    else f[4]=1.0+engineeringStrain;
    std::array<unsigned,2> lateralComponents{};
    std::array<unsigned,2> lateralStretches{};
    unsigned stressComponent=0u;
    if(axis==0u) {
        lateralComponents={4u,8u}; lateralStretches={4u,8u}; stressComponent=0u;
        f[4]=1.0-0.18*engineeringStrain;
        f[8]=1.0-0.01*engineeringStrain;
    } else {
        lateralComponents={0u,8u}; lateralStretches={0u,8u}; stressComponent=4u;
        f[0]=1.0-0.18*engineeringStrain;
        f[8]=1.0-0.01*engineeringStrain;
    }
    const std::array<double,15> accepted{};
    CompiledPoint64 point;
    double lateralNorm=std::numeric_limits<double>::infinity();
    for(unsigned iteration=0;iteration<24u;++iteration) {
        point=projectCompiled64(program,f,accepted);
        const std::array<double,2> residual{
            point.firstPiolaMPa[lateralComponents[0]],
            point.firstPiolaMPa[lateralComponents[1]]};
        lateralNorm=std::max(std::abs(residual[0]),std::abs(residual[1]));
        if(lateralNorm<2.0e-7) break;
        // Stay within the C1 compression transition when this helper is used
        // by the asymmetric candidate. The registered normalized-stress band
        // is 5e-6; a strain perturbation of 2e-6 crosses that entire branch
        // for these orthotropic moduli and produces a biased lateral Jacobian.
        std::array<double,4> jacobian{};
        for(unsigned column=0;column<2u;++column) {
            paper::Matrix direction{};
            direction[lateralStretches[column]]=1.0;
            const auto tangent=compiledAlgorithmicTangent15(
                program,f,direction,accepted,point.state);
            for(unsigned row=0;row<2u;++row)
                jacobian[2u*row+column]=tangent[lateralComponents[row]];
        }
        const double determinant=jacobian[0]*jacobian[3]-jacobian[1]*jacobian[2];
        require(std::isfinite(determinant)&&std::abs(determinant)>1.0e-10,
            "source-fit stress-free lateral tangent is singular");
        const std::array<double,2> correction{
            (-jacobian[3]*residual[0]+jacobian[1]*residual[1])/determinant,
            (jacobian[2]*residual[0]-jacobian[0]*residual[1])/determinant};
        const auto oldLateral=lateralNorm;
        double step=1.0;
        bool acceptedStep=false;
        for(unsigned backtrack=0;backtrack<16u;++backtrack) {
            paper::Matrix trialF=f;
            for(unsigned column=0;column<2u;++column)
                trialF[lateralStretches[column]]+=step*correction[column];
            const auto trial=projectCompiled64(program,trialF,accepted);
            const double trialNorm=std::max(
                std::abs(trial.firstPiolaMPa[lateralComponents[0]]),
                std::abs(trial.firstPiolaMPa[lateralComponents[1]]));
            if(trialNorm<oldLateral) {
                f=trialF;
                acceptedStep=true;
                break;
            }
            step*=0.5;
        }
        require(acceptedStep,"source-fit stress-free lateral solve stalled");
    }
    point=projectCompiled64(program,f,accepted);
    lateralNorm=std::max(std::abs(point.firstPiolaMPa[lateralComponents[0]]),
                         std::abs(point.firstPiolaMPa[lateralComponents[1]]));
    require(lateralNorm<2.0e-6,
        "source-fit uniaxial response did not reach stress-free transverse conditions: "+
        number(lateralNorm));
    return {point.firstPiolaMPa[stressComponent],point.state,lateralNorm};
}

struct SourceFitCurve {
    std::string component;
    std::string axis;
    double yieldStrain=0.0;
    double yieldStressMPa=0.0;
    double endpointStrain=0.0;
    double endpointStressMPa=0.0;
    double elasticModulusMPa=0.0;
    double tangentMPa=0.0;
    double tangentLowMPa=0.0;
    double tangentHighMPa=0.0;
    double plasticHardeningMPa=0.0;
    double plasticHardeningLowMPa=0.0;
    double plasticHardeningHighMPa=0.0;
};

struct MarkerCenter {
    std::string component;
    std::string axis;
    std::string pointId;
    double strain=0.0;
    double stressMPa=0.0;
    double strainHalfwidth=0.0;
    double stressHalfwidthMPa=0.0;
};

std::vector<std::string> splitCsvRow(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted=false;
    for(std::size_t index=0;index<line.size();++index) {
        const char character=line[index];
        if(character=='"') {
            if(quoted && index+1u<line.size() && line[index+1u]=='"') {
                field.push_back('"');
                ++index;
            } else {
                quoted=!quoted;
            }
        } else if(character==',' && !quoted) {
            fields.push_back(std::move(field));
            field.clear();
        } else {
            field.push_back(character);
        }
    }
    require(!quoted,"source fit CSV contains an unclosed quoted field");
    fields.push_back(std::move(field));
    return fields;
}

std::vector<SourceFitCurve> readSourceFitCurves(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(input.good(),"cannot open registered Haj-Ali source-fit nodes: "+path.string());
    std::string line;
    require(static_cast<bool>(std::getline(input,line)),"source-fit CSV is empty");
    const auto header=splitCsvRow(line);
    const auto column=[&](const std::string& name) {
        const auto found=std::find(header.begin(),header.end(),name);
        require(found!=header.end(),"source-fit CSV lacks column "+name);
        return static_cast<std::size_t>(std::distance(header.begin(),found));
    };
    const std::size_t componentColumn=column("component");
    const std::size_t axisColumn=column("axis");
    const std::size_t roleColumn=column("point_role");
    const std::size_t strainColumn=column("strain_mm_per_mm");
    const std::size_t stressColumn=column("stress_MPa");
    const std::size_t elasticColumn=column("table2_elastic_modulus_MPa");
    const std::size_t tangentColumn=column("postyield_tangent_MPa");
    const std::size_t tangentLowColumn=column("postyield_tangent_low_MPa");
    const std::size_t tangentHighColumn=column("postyield_tangent_high_MPa");
    const std::size_t hardeningColumn=column("uniaxial_plastic_hardening_modulus_MPa");
    const std::size_t hardeningLowColumn=column("uniaxial_plastic_hardening_modulus_low_MPa");
    const std::size_t hardeningHighColumn=column("uniaxial_plastic_hardening_modulus_high_MPa");
    std::map<std::string,SourceFitCurve> curves;
    while(std::getline(input,line)) {
        if(line.empty()) continue;
        const auto fields=splitCsvRow(line);
        require(fields.size()==header.size(),"source-fit CSV row has an unexpected field count");
        const std::string component=fields[componentColumn];
        const std::string axis=fields[axisColumn];
        auto& curve=curves[component+"/"+axis];
        curve.component=component;
        curve.axis=axis;
        const auto value=[&](const std::size_t index) { return std::stod(fields[index]); };
        curve.elasticModulusMPa=value(elasticColumn);
        curve.tangentMPa=value(tangentColumn);
        curve.tangentLowMPa=value(tangentLowColumn);
        curve.tangentHighMPa=value(tangentHighColumn);
        curve.plasticHardeningMPa=value(hardeningColumn);
        curve.plasticHardeningLowMPa=value(hardeningLowColumn);
        curve.plasticHardeningHighMPa=value(hardeningHighColumn);
        if(fields[roleColumn]=="published_Table_1_tensile_yield_anchor") {
            curve.yieldStrain=value(strainColumn);
            curve.yieldStressMPa=value(stressColumn);
        } else if(fields[roleColumn]=="digitized_bilinear_fit_endpoint") {
            curve.endpointStrain=value(strainColumn);
            curve.endpointStressMPa=value(stressColumn);
        }
    }
    require(curves.size()==4u,"registered source-fit CSV must contain the four liner/medium MD/CD curves");
    std::vector<SourceFitCurve> result;
    for(auto& [key,curve]:curves) {
        (void)key;
        require(curve.endpointStrain>curve.yieldStrain &&
                    curve.endpointStressMPa>curve.yieldStressMPa &&
                    curve.elasticModulusMPa>curve.tangentHighMPa,
                "source-fit series is missing ordered yield/endpoint values");
        const double derived=curve.elasticModulusMPa*curve.tangentMPa/
            (curve.elasticModulusMPa-curve.tangentMPa);
        require(std::abs(derived-curve.plasticHardeningMPa)<1.0e-8*std::max(1.0,derived),
                "registered uniaxial H does not follow H=E*Et/(E-Et)");
        result.push_back(curve);
    }
    return result;
}

std::vector<MarkerCenter> readMarkerCenters(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(input.good(),"cannot open hash-registered Haj-Ali marker centers: "+path.string());
    std::string line;
    require(static_cast<bool>(std::getline(input,line)),"Haj-Ali marker-center CSV is empty");
    const auto header=splitCsvRow(line);
    const auto column=[&](const std::string& name) {
        const auto found=std::find(header.begin(),header.end(),name);
        require(found!=header.end(),"marker-center CSV lacks column "+name);
        return static_cast<std::size_t>(std::distance(header.begin(),found));
    };
    const auto sourceColumn=column("source_id");
    const auto componentColumn=column("component");
    const auto axisColumn=column("axis");
    const auto statusColumn=column("record_status");
    const auto idColumn=column("point_id");
    const auto strainColumn=column("strain_mm_per_mm");
    const auto stressColumn=column("stress_MPa");
    const auto strainWidthColumn=column("strain_halfwidth_mm_per_mm");
    const auto stressWidthColumn=column("stress_halfwidth_MPa");
    const auto renderHashColumn=column("source_render_sha256");
    const auto classColumn=column("observation_class");
    constexpr std::string_view expectedRenderHash=
        "bffb53d8685c5990b2fa43f120ac0382493e893db2284b4f8c24adff6a48a6d5";
    constexpr std::string_view expectedClass=
        "digitized_plot_marker_center_estimate_not_raw_measurement";
    std::vector<MarkerCenter> result;
    std::map<std::string,unsigned> counts;
    while(std::getline(input,line)) {
        if(line.empty())continue;
        const auto fields=splitCsvRow(line);
        require(fields.size()==header.size(),"marker-center CSV row has an unexpected field count");
        if(fields[statusColumn]!="selected_marker_center")continue;
        require(fields[sourceColumn]=="haj_ali_2009"&&fields[renderHashColumn]==expectedRenderHash&&
                fields[classColumn]==expectedClass,
            "marker center lost its source/render/estimate identity");
        MarkerCenter point;
        point.component=fields[componentColumn];
        point.axis=fields[axisColumn];
        point.pointId=fields[idColumn];
        point.strain=std::stod(fields[strainColumn]);
        point.stressMPa=std::stod(fields[stressColumn]);
        point.strainHalfwidth=std::stod(fields[strainWidthColumn]);
        point.stressHalfwidthMPa=std::stod(fields[stressWidthColumn]);
        require(!point.pointId.empty()&&std::isfinite(point.strain)&&
                std::isfinite(point.stressMPa)&&point.strain>=0.0&&
                point.strainHalfwidth>0.0&&point.stressHalfwidthMPa>0.0,
            "marker center has invalid coordinates or uncertainty");
        const std::string key=point.component+"/"+point.axis;
        ++counts[key];
        result.push_back(std::move(point));
    }
    require(result.size()==278u,
        "marker calibration must use the 278 selected centers, excluding unresolved overplot regions");
    require(counts["liner/MD"]==56u&&counts["liner/CD"]==64u&&
            counts["medium_NSSC/MD"]==59u&&counts["medium_NSSC/CD"]==99u,
        "marker calibration center counts differ from the four reviewed plot panels");
    std::cout<<"marker_input source_render_sha256="<<expectedRenderHash
        <<" marker_csv_sha256=7a2dd7a351586971566fe34ca144b511d0b2014ab13f4cd7c9de0e87c2cf39b9"
        <<" selected_centers="<<result.size()
        <<" classification=calibration_only_not_raw_measurements_or_validation\n";
    return result;
}

using AsymmetricTransferRecords =
    std::map<std::tuple<std::string,std::string,std::string>,double>;

AsymmetricTransferRecords readAsymmetricTransferRecords(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    require(input.good(),"cannot open versioned C1 asymmetric-transfer input record");
    std::string line;
    require(static_cast<bool>(std::getline(input,line)),"C1 transfer CSV is empty");
    const auto header=splitCsvRow(line);
    const auto column=[&](const std::string& name) {
        const auto found=std::find(header.begin(),header.end(),name);
        require(found!=header.end(),"C1 transfer CSV lacks column "+name);
        return static_cast<std::size_t>(std::distance(header.begin(),found));
    };
    const auto modelColumn=column("model");
    const auto axisColumn=column("axis");
    const auto quantityColumn=column("quantity");
    const auto valueColumn=column("value");
    const auto unitColumn=column("unit");
    AsymmetricTransferRecords result;
    unsigned dataRows=0u;
    while(std::getline(input,line)) {
        if(line.empty())continue;
        const auto fields=splitCsvRow(line);
        require(fields.size()==header.size(),"C1 transfer CSV row width changed");
        const auto key=std::make_tuple(fields[modelColumn],fields[axisColumn],
                                       fields[quantityColumn]);
        const double value=std::stod(fields[valueColumn]);
        require(std::isfinite(value),"C1 transfer CSV contains a nonfinite number");
        require(result.emplace(key,value).second,"C1 transfer CSV has a duplicate model parameter");
        if(fields[quantityColumn]=="compression_smoothing_width")
            require(fields[unitColumn]=="one","C1 smoothing width must be dimensionless");
        ++dataRows;
    }
    require(dataRows==30u,"C1 transfer CSV row count changed without a version update");
    return result;
}

void checkBilinearFitMaterial(const MaterialProgram& fit,
    const MaterialProgram& ideal,const paper::MaterialKind kind,
    const std::vector<SourceFitCurve>& curves) {
    const bool liner=kind==paper::MaterialKind::liner;
    const std::string expectedName=liner
        ?"hajali2009_liner_hill_bilinear_fit"
        :"hajali2009_medium_hill_bilinear_fit";
    require(fit.name==expectedName,"bilinear fit asset identity differs from registered role");
    require(fit.internalState.size()==15u && fit.stateImplicitRoots.size()==15u &&
                fit.stateUpdateRoots.size()==15u,
            "bilinear fit asset must append two directional hardening histories to the Hill program");
    for(const char* name:{"density","c11","c22","c33","c12","c13","c23",
                           "g12","g13","g23","sigma_ref","hill_F","hill_G",
                           "hill_H","hill_L","hill_M","hill_N","q_floor",
                           "fb_epsilon","residual_scale"})
        checkParameter(fit,name,parameterValue(ideal,name));
    checkParameter(fit,"hardening_weight_scale",1.0e-3);
    const std::string component=liner?"liner":"medium_NSSC";
    const auto findCurve=[&](const std::string& axis)->const SourceFitCurve& {
        const auto found=std::find_if(curves.begin(),curves.end(),[&](const auto& curve) {
            return curve.component==component && curve.axis==axis;
        });
        require(found!=curves.end(),"registered source-fit CSV lacks "+component+" "+axis);
        return *found;
    };
    for(const auto axis:{"MD","CD"}) {
        const auto& curve=findCurve(axis);
        const std::string hardeningName=std::string("hardening_")+(
            std::string_view(axis)=="MD"?"md":"cd");
        checkParameter(fit,hardeningName,curve.plasticHardeningMPa*1.0e6);
        require(curve.plasticHardeningLowMPa>0.0 &&
                    curve.plasticHardeningMPa>=curve.plasticHardeningLowMPa &&
                    curve.plasticHardeningMPa<=curve.plasticHardeningHighMPa,
                "digitized hardening point estimate is outside its reported pixel interval");
    }
    const auto compiled=detail::compileConstitutive(fit,NM_EXPRESSION_STACK_CAPACITY);
    require(compiled.succeeded(),"generic source-fit Hill material did not compile: "+
        diagnostics(compiled.diagnostics));
    require(compiled.program.implicitResiduals.size()==15u &&
                compiled.program.implicitJacobians.size()==225u &&
                compiled.program.stressStateDerivatives.size()==135u,
            "source-fit material must use the generic 15-state projection/tangent program");

    for(const auto axis:{"MD","CD"}) {
        const auto& curve=findCurve(axis);
        const double span=curve.endpointStrain-curve.yieldStrain;
        const double strainA=curve.yieldStrain+0.30*span;
        const double strainB=curve.yieldStrain+0.70*span;
        const auto a=stressFreeUniaxialPoint(compiled.program,
            std::string_view(axis)=="MD"?0u:1u,strainA);
        const auto b=stressFreeUniaxialPoint(compiled.program,
            std::string_view(axis)=="MD"?0u:1u,strainB);
        require(a.state[12]>1.0e-7 && b.state[12]>a.state[12],
            "source-fit uniaxial test did not traverse positive plastic flow");
        const double observed=(b.stressMPa-a.stressMPa)/(strainB-strainA);
        const double relative=std::abs(observed-curve.tangentMPa)/curve.tangentMPa;
        const double intervalPadding=0.005*curve.tangentMPa;
        const bool insideDigitizedInterval=
            observed>=curve.tangentLowMPa-intervalPadding &&
            observed<=curve.tangentHighMPa+intervalPadding;
        std::cout<<fit.name<<" axis="<<axis<<" observed_total_tangent_MPa="
            <<number(observed)<<" source_fit_Et_MPa="<<number(curve.tangentMPa)
            <<" source_fit_pixel_interval_MPa=["<<number(curve.tangentLowMPa)<<','
            <<number(curve.tangentHighMPa)<<"] relative_error="<<number(relative)
            <<" transverse_stress_max_MPa="<<number(std::max(a.lateralResidualMPa,
                                                               b.lateralResidualMPa))<<"\n";
        require(insideDigitizedInterval,
            "compiled source-fit law's observed uniaxial total-stress slope misses the digitized tangent interval: "+
            fit.name+" "+axis+" error="+number(relative));
    }
}

std::array<double,9> compiledAlgorithmicTangent15(
    const ConstitutiveProgram& program,const paper::Matrix& f,
    const paper::Matrix& h,const std::array<double,15>& accepted,
    const std::array<double,15>& candidate) {
    std::array<double,225> jacobian=compiledJacobian15(program,f,accepted,candidate);
    std::array<double,15> rhs{},stateDirection{};
    const paper::Matrix zero{};
    for(unsigned state=0;state<15u;++state)
        rhs[state]=-evaluate(program.implicitDeformationDirections.at(state),
            program.material,f,h,accepted,candidate);
    require(solveLinear15(jacobian,rhs,stateDirection),
        "asymmetric candidate implicit material Jacobian is singular");
    std::array<double,9> tangent{};
    for(unsigned component=0;component<9u;++component) {
        tangent[component]=evaluate(program.tangentVector.at(component),
            program.material,f,h,accepted,candidate);
        for(unsigned state=0;state<15u;++state)
            tangent[component]+=evaluate(
                program.stressStateDerivatives.at(component*15u+state),
                program.material,f,zero,accepted,candidate)*stateDirection[state];
    }
    // The authored source law stores stress and tangent coefficients in Pa;
    // CompiledPoint64 exposes first Piola stress in MPa for the fit comparison.
    for(double& value:tangent)value*=1.0e-6;
    return tangent;
}

void setParameterValue(MaterialProgram& material,const std::string& name,const double value) {
    const auto found=std::find_if(material.parameters.begin(),material.parameters.end(),
        [&](const Parameter& parameter){return parameter.name==name;});
    require(found!=material.parameters.end(),"missing transfer parameter "+name);
    require(!found->identifiable && found->lower==found->upper,
        "source-transfer parameter must remain fixed and nonidentifiable: "+name);
    found->defaultValue=value;
}

double sourceFitTangentForAxis(const ConstitutiveProgram& program,
    const SourceFitCurve& curve,const unsigned axis) {
    require(axis<2u,"source-fit tangent only supports MD/CD");
    const double span=curve.endpointStrain-curve.yieldStrain;
    const auto a=stressFreeUniaxialPoint(program,axis,curve.yieldStrain+0.30*span);
    const auto b=stressFreeUniaxialPoint(program,axis,curve.yieldStrain+0.70*span);
    return (b.stressMPa-a.stressMPa)/(0.40*span);
}

std::array<double,2> sourceFitTangents(const ConstitutiveProgram& program,
    const SourceFitCurve& md,const SourceFitCurve& cd) {
    return {sourceFitTangentForAxis(program,md,0u),
            sourceFitTangentForAxis(program,cd,1u)};
}

struct AsymmetricTransferFit {
    std::array<double,2> scales{};
    std::array<double,2> slopes{};
};

AsymmetricTransferFit fitAsymmetricTransferScales(const ConstitutiveProgram& baseline,
    const SourceFitCurve& md,const SourceFitCurve& cd,const double initialMd,
    const double initialCd) {
    const std::array<double,2> target{md.tangentMPa,cd.tangentMPa};
    const std::array<const SourceFitCurve*,2> curves{&md,&cd};
    const std::array<const char*,2> names{
        "fit_transfer_scale_md","fit_transfer_scale_cd"};
    std::array<double,2> scales{initialMd,initialCd};
    const auto measured=[&](const std::array<double,2>& values) {
        ConstitutiveProgram trial=baseline;
        setParameterValue(trial.material,names[0],values[0]);
        setParameterValue(trial.material,names[1],values[1]);
        return sourceFitTangents(trial,md,cd);
    };
    for(unsigned coordinatePass=0u;coordinatePass<4u;++coordinatePass) {
        for(unsigned axis=0u;axis<2u;++axis) {
            const auto at=[&](double value) {
                ConstitutiveProgram trial=baseline;
                setParameterValue(trial.material,names[0],scales[0]);
                setParameterValue(trial.material,names[1],scales[1]);
                setParameterValue(trial.material,names[axis],value);
                return sourceFitTangentForAxis(trial,*curves[axis],axis);
            };
            double x0=scales[axis];
            double y0=at(x0);
            if(std::abs(y0-target[axis])/target[axis]<2.0e-6)continue;
            double x1=y0>target[axis]?x0*0.8:x0*1.2;
            double y1=at(x1);
            for(unsigned expand=0u;expand<16u&&
                (y0-target[axis])*(y1-target[axis])>0.0;++expand) {
                x1=y0>target[axis]?x1*0.5:x1*2.0;
                require(x1>0.0&&x1<128.0,
                    "asymmetric transfer fit could not bracket a positive scale");
                y1=at(x1);
            }
            require((y0-target[axis])*(y1-target[axis])<=0.0,
                "asymmetric hardening transfer tangent could not be bracketed");
            for(unsigned iteration=0u;iteration<24u;++iteration) {
                double x2=x1-(y1-target[axis])*(x1-x0)/(y1-y0);
                const double low=std::min(x0,x1),high=std::max(x0,x1);
                if(!std::isfinite(x2)||x2<=low+0.1*(high-low)||x2>=high-0.1*(high-low))
                    x2=0.5*(x0+x1);
                const double y2=at(x2);
                if(std::abs(y2-target[axis])/target[axis]<2.0e-6) {
                    x0=x2;y0=y2;x1=x2;y1=y2;break;
                }
                if((y0-target[axis])*(y2-target[axis])<=0.0) {x1=x2;y1=y2;}
                else {x0=x2;y0=y2;}
            }
            scales[axis]=std::abs(y0-target[axis])<std::abs(y1-target[axis])?x0:x1;
        }
        const auto slopes=measured(scales);
        if(std::abs(slopes[0]-target[0])/target[0]<2.0e-6&&
           std::abs(slopes[1]-target[1])/target[1]<2.0e-6)
            return {scales,slopes};
    }
    const auto final=measured(scales);
    require(std::abs(final[0]-target[0])/target[0]<5.0e-4&&
            std::abs(final[1]-target[1])/target[1]<5.0e-4,
        "asymmetric hardening transfer fit did not converge to both source-fit tangents");
    return {scales,final};
}

void reportMarkerCurveResiduals(const ConstitutiveProgram& program,
    const std::vector<MarkerCenter>& markers,const std::string& component) {
    struct Statistics {
        unsigned count=0u;
        unsigned withinVerticalReadout=0u;
        double squaredError=0.0;
        double absoluteError=0.0;
        double maximumAbsoluteError=0.0;
    };
    std::map<std::string,Statistics> statistics;
    for(const auto& marker:markers) {
        if(marker.component!=component)continue;
        const unsigned axis=marker.axis=="MD"?0u:(marker.axis=="CD"?1u:2u);
        require(axis<2u,"marker calibration contains a non-MD/CD test axis");
        const auto model=stressFreeUniaxialPoint(program,axis,marker.strain);
        const double residual=model.stressMPa-marker.stressMPa;
        auto& aggregate=statistics[marker.axis];
        ++aggregate.count;
        aggregate.squaredError+=residual*residual;
        aggregate.absoluteError+=std::abs(residual);
        aggregate.maximumAbsoluteError=std::max(aggregate.maximumAbsoluteError,std::abs(residual));
        if(std::abs(residual)<=marker.stressHalfwidthMPa)++aggregate.withinVerticalReadout;
    }
    for(const auto& [axis,aggregate]:statistics) {
        require(aggregate.count>0u,"marker calibration has an empty source curve");
        std::cout<<program.material.name<<" marker_curve_calibration component="<<component
            <<" axis="<<axis<<" n="<<aggregate.count
            <<" rmse_MPa="<<number(std::sqrt(aggregate.squaredError/aggregate.count))
            <<" mae_MPa="<<number(aggregate.absoluteError/aggregate.count)
            <<" max_abs_residual_MPa="<<number(aggregate.maximumAbsoluteError)
            <<" within_vertical_pixel_readout="<<aggregate.withinVerticalReadout
            <<"/"<<aggregate.count
            <<" x_uncertainty_not_propagated=true classification=calibration_only_not_validation\n";
    }
}

void checkAsymmetricTransferCandidate(const MaterialProgram& candidate,
    const MaterialProgram& ideal,const paper::MaterialKind kind,
    const std::vector<SourceFitCurve>& curves,const std::vector<MarkerCenter>& markers,
    const AsymmetricTransferRecords& transferRecords,const bool fullCalibration) {
    const bool liner=kind==paper::MaterialKind::liner;
    const bool c1Regularized=candidate.name.find("_asymmetric_c1_transfer")!=std::string::npos;
    const std::string expectedName=liner
        ?(c1Regularized?"hajali2009_liner_hill_asymmetric_c1_transfer":
                        "hajali2009_liner_hill_asymmetric_transfer")
        :(c1Regularized?"hajali2009_medium_hill_asymmetric_c1_transfer":
                        "hajali2009_medium_hill_asymmetric_transfer");
    require(candidate.name==expectedName,
        "asymmetric transfer candidate has unexpected material identity");
    require(candidate.internalState.size()==15u &&
                candidate.stateImplicitRoots.size()==15u &&
                candidate.stateUpdateRoots.size()==15u,
        "asymmetric transfer candidate must preserve the 15-state tensile-hardening model");
    for(const char* name:{"density","c11","c22","c33","c12","c13","c23",
                           "g12","g13","g23","sigma_ref","hill_L","hill_M",
                           "hill_N","q_floor",
                           "fb_epsilon","residual_scale"})
        checkParameter(candidate,name,parameterValue(ideal,name));
    const double f=parameterValue(candidate,"hill_F");
    const double g=parameterValue(candidate,"hill_G");
    const double h=parameterValue(candidate,"hill_H");
    require(f+g>0.0 && f+h>0.0 && g+h>0.0,
        "tensile Hill normal quadratic is not positive semidefinite");
    const double sigmaRefMPa=parameterValue(candidate,"sigma_ref")/1.0e6;
    const double qFloor=parameterValue(candidate,"q_floor");
    const double compressionWidth=c1Regularized
        ?parameterValue(candidate,"compression_smoothing_width"):0.0;
    if(c1Regularized)
        require(compressionWidth>0.0 && compressionWidth<=1.0e-4,
            "C1 compression positive-part width must be an explicit small numerical regularizer");
    const std::array<double,3> tensile=liner
        ?std::array<double,3>{38.2,18.4,12.55}
        :std::array<double,3>{15.65,8.06,5.38};
    const std::array<double,3> compression=liner
        ?std::array<double,3>{22.92,11.02,7.52}
        :std::array<double,3>{9.391,4.836,3.23};
    const std::array<double,3> tensilePairSum{g+h,f+h,f+g};
    const std::string modelName=candidate.name;
    const auto record=[&](const std::string& axis,const std::string& quantity) {
        const auto found=transferRecords.find({modelName,axis,quantity});
        require(found!=transferRecords.end(),
            "versioned C1 source-data record is missing "+modelName+"/"+axis+"/"+quantity);
        return found->second;
    };
    if(c1Regularized) {
        const double declaredWidth=record("all","compression_smoothing_width");
        require(std::abs(compressionWidth-declaredWidth)<1.0e-15,
            "C1 material smoothing width differs from its versioned source-data record");
        require(std::abs(parameterValue(candidate,"fit_transfer_scale_md")-
                         record("MD","fit_transfer_scale"))<2.0e-12 &&
                std::abs(parameterValue(candidate,"fit_transfer_scale_cd")-
                         record("CD","fit_transfer_scale"))<2.0e-12,
            "C1 material transfer scales differ from their versioned source-data record");
    }
    const std::array<double,3> reLu{parameterValue(candidate,"compression_k11"),
        parameterValue(candidate,"compression_k22"),
        parameterValue(candidate,"compression_k33")};
    for(unsigned axis=0;axis<3u;++axis) {
        require(reLu[axis]>=0.0,"compression ReLU coefficient must be nonnegative");
        const double computed=sigmaRefMPa*(1.0/compression[axis]-1.0/tensile[axis]);
        require(std::abs(reLu[axis]-computed)<2.0e-12,
            "compression ReLU coefficient differs from the declared Table 3 transfer inputs axis="+
            std::to_string(axis)+" actual="+number(reLu[axis])+" expected="+number(computed));
        const double expectedPair=std::pow(sigmaRefMPa/tensile[axis],2);
        require(std::abs(tensilePairSum[axis]-expectedPair)<2.0e-12,
            "asymmetric Hill quadratic is not derived from the source tensile strengths axis="+
            std::to_string(axis)+" actual="+number(tensilePairSum[axis])+" expected="+
            number(expectedPair));
        const double stretchCoeff=axis==0u?g+h:(axis==1u?f+h:f+g);
        const double stressTension=tensile[axis]/sigmaRefMPa;
        const double stressCompression=-compression[axis]/sigmaRefMPa;
        const auto positivePart=[&](const double value) {
            if(!c1Regularized) return std::max(-value,0.0);
            if(value>=compressionWidth) return 0.0;
            if(value<=-compressionWidth) return -value;
            const double delta=compressionWidth;
            return (delta-value)*(delta-value)/(4.0*delta);
        };
        const auto qFor=[&](const double axisStress) {
            double q=std::sqrt(stretchCoeff*axisStress*axisStress+qFloor*qFloor);
            for(unsigned component=0;component<3u;++component)
                q+=reLu[component]*positivePart(component==axis?axisStress:0.0);
            return q;
        };
        const double qTension=qFor(stressTension);
        const double qCompression=qFor(stressCompression);
        const double regularizerShift=0.25*compressionWidth*
            (reLu[(axis+1u)%3u]+reLu[(axis+2u)%3u]);
        std::cout<<candidate.name<<" axis="<<axis
            <<" q_at_Table3_tension="<<number(qTension)
            <<" q_at_Table3_compression="<<number(qCompression)
            <<" C1_zero-stress_shift="<<number(regularizerShift)<<"\n";
        const double radiusTolerance=c1Regularized?1.0e-5:1.0e-10;
        require(std::abs(qTension-1.0)<radiusTolerance &&
                    std::abs(qCompression-1.0)<radiusTolerance,
            "asymmetric transfer uniaxial radii miss Table 3 strengths beyond the declared regularization shift");
    }
    const auto compiled=detail::compileConstitutive(candidate,NM_EXPRESSION_STACK_CAPACITY);
    require(compiled.succeeded(),"generic asymmetric 15-state material did not compile: "+
        diagnostics(compiled.diagnostics));
    require(compiled.program.implicitResiduals.size()==15u &&
                compiled.program.implicitJacobians.size()==225u &&
                compiled.program.implicitDeformationDirections.size()==15u &&
                compiled.program.stressStateDerivatives.size()==135u,
        "asymmetric transfer did not retain its full implicit stress/state tangent chain");

    // Keep the native checker backed by a second, equation-level FP64 oracle.
    // This intentionally does not call the compiler's bytecode evaluator: it
    // independently differentiates the declared 15-state StVK/Hill/Huber
    // system, then compares projection, stress and all nine tangent columns.
    // It is a CPU implementation cross-check, not native Metal evidence.
    const std::array<std::pair<paper::Matrix,const char*>,5> oracleCases{{
        {paper::identity(),"rest"},
        {paper::Matrix{1.035,0,0,0,0.995,0,0,0,0.999},"positive_MD"},
        {paper::Matrix{0.965,0,0,0,1.004,0,0,0,1.001},"negative_MD"},
        {paper::Matrix{1.0+2.0e-7,0,0,0,1.0-1.0e-7,0,0,0,1.0},"near_zero"},
        {paper::Matrix{1.012,0.0012,0,0,0.997,0,0,0,1.003},"mixed"}}};
    const c1oracle::State zeroAccepted{};
    for(const auto& [fOracle,caseName]:oracleCases) {
        const auto independent=c1oracle::project(compiled.program,fOracle,zeroAccepted);
        const auto bytecode=projectCompiled64(compiled.program,fOracle,zeroAccepted);
        double maxStateError=0.0,maxStressErrorPa=0.0,maxStressScalePa=1.0;
        unsigned worstState=0u;
        for(unsigned state=0;state<15u;++state) {
            const double error=std::abs(independent.state[state]-bytecode.state[state]);
            if(error>maxStateError){maxStateError=error;worstState=state;}
        }
        for(unsigned componentIndex=0;componentIndex<9u;++componentIndex) {
            const double referencePa=bytecode.firstPiolaMPa[componentIndex]*1.0e6;
            maxStressErrorPa=std::max(maxStressErrorPa,
                std::abs(independent.firstPiolaPa[componentIndex]-referencePa));
            maxStressScalePa=std::max(maxStressScalePa,std::abs(referencePa));
        }
        std::cout<<candidate.name<<" independent_oracle_case="<<caseName
            <<" state_inf_error="<<number(maxStateError)
            <<" worst_state="<<worstState
            <<" manual_value="<<number(independent.state[worstState])
            <<" bytecode_value="<<number(bytecode.state[worstState])
            <<" stress_relative_error="<<number(maxStressErrorPa/maxStressScalePa)
            <<" classification=cpu_oracle_crosscheck_not_native\n";
        require(maxStateError<1.0e-6 && maxStressErrorPa/maxStressScalePa<1.0e-6,
            "manual C1 oracle disagrees with compiled FP64 law: "+candidate.name+
            " case="+caseName);
        double worstTangentRelative=0.0;
        for(unsigned column=0;column<9u;++column) {
            paper::Matrix direction{};direction[column]=1.0;
            const auto independentTangent=c1oracle::tangent(
                compiled.program,fOracle,direction,zeroAccepted,independent.state);
            const auto bytecodeTangent=compiledAlgorithmicTangent15(
                compiled.program,fOracle,direction,zeroAccepted,bytecode.state);
            double error=0.0,scale=1.0;
            for(unsigned row=0;row<9u;++row) {
                const double bytecodePa=bytecodeTangent[row]*1.0e6;
                error=std::max(error,std::abs(independentTangent[row]-bytecodePa));
                scale=std::max(scale,std::abs(bytecodePa));
            }
            worstTangentRelative=std::max(worstTangentRelative,error/scale);
        }
        std::cout<<candidate.name<<" independent_oracle_case="<<caseName
            <<" all9_tangent_columns_max_relative_error="<<number(worstTangentRelative)
            <<" classification=cpu_oracle_crosscheck_not_native\n";
        require(worstTangentRelative<2.0e-5,
            "manual C1 tangent disagrees with compiled FP64 law: "+candidate.name+
            " case="+caseName);
    }

    const std::string component=liner?"liner":"medium_NSSC";
    const auto findCurve=[&](const std::string& axis)->const SourceFitCurve& {
        const auto found=std::find_if(curves.begin(),curves.end(),[&](const auto& curve) {
            return curve.component==component && curve.axis==axis;
        });
        require(found!=curves.end(),"registered source-fit CSV lacks "+component+" "+axis);
        return *found;
    };
    const auto& mdCurve=findCurve("MD");
    const auto& cdCurve=findCurve("CD");
    checkParameter(candidate,"hardening_md",mdCurve.plasticHardeningMPa*1.0e6);
    checkParameter(candidate,"hardening_cd",cdCurve.plasticHardeningMPa*1.0e6);
    const double authoredMd=parameterValue(candidate,"fit_transfer_scale_md");
    const double authoredCd=parameterValue(candidate,"fit_transfer_scale_cd");
    std::array<double,2> authoredSlopes{};
    if(fullCalibration) {
        const auto fitted=fitAsymmetricTransferScales(compiled.program,mdCurve,cdCurve,
            tensilePairSum[0],tensilePairSum[1]);
        std::cout<<candidate.name<<" selected_width_normalized_stress="
            <<number(compressionWidth)<<" fit_transfer_scale_md="<<number(fitted.scales[0])
            <<" fit_transfer_scale_cd="<<number(fitted.scales[1])
            <<" calibration_tangents_MPa=["<<number(fitted.slopes[0])<<','
            <<number(fitted.slopes[1])<<"] source_Et_MPa=["
            <<number(mdCurve.tangentMPa)<<','<<number(cdCurve.tangentMPa)<<"]\n";
        require(std::abs(authoredMd-fitted.scales[0])<2.0e-7&&
                std::abs(authoredCd-fitted.scales[1])<2.0e-7,
            "authored transfer scales do not reproduce the CPU source-tangent fit");
        authoredSlopes=fitted.slopes;
        reportMarkerCurveResiduals(compiled.program,markers,component);

        // Width is a numerical modeling assumption; recompute each tangent
        // using the same strict local projection certificate.
        for(const double width:{5.0e-6,2.5e-6,1.25e-6}) {
            ConstitutiveProgram widthProgram=compiled.program;
            setParameterValue(widthProgram.material,"fit_transfer_scale_md",authoredMd);
            setParameterValue(widthProgram.material,"fit_transfer_scale_cd",authoredCd);
            setParameterValue(widthProgram.material,"compression_smoothing_width",width);
            const auto slopes=sourceFitTangents(widthProgram,mdCurve,cdCurve);
            std::cout<<candidate.name<<" width_sensitivity width="<<number(width)
                <<" tangent_MPa=["<<number(slopes[0])<<','<<number(slopes[1])<<"]"
                <<" delta_from_selected_MPa=["<<number(slopes[0]-authoredSlopes[0])<<','
                <<number(slopes[1]-authoredSlopes[1])<<"] local_certificate_max=2e-11\n";
            require(std::isfinite(slopes[0])&&std::isfinite(slopes[1]),
                "C1 smoothing width sensitivity produced a nonfinite tangent");
        }
        require(fitted.slopes[0]>=mdCurve.tangentLowMPa&&
                fitted.slopes[0]<=mdCurve.tangentHighMPa&&
                fitted.slopes[1]>=cdCurve.tangentLowMPa&&
                fitted.slopes[1]<=cdCurve.tangentHighMPa,
            "chosen C1 width fit does not fall inside the digitized tangent intervals");
    } else {
        authoredSlopes=sourceFitTangents(compiled.program,mdCurve,cdCurve);
        std::cout<<candidate.name<<" authored_transfer_slopes_MPa=["
            <<number(authoredSlopes[0])<<','<<number(authoredSlopes[1])
            <<"] source_Et_MPa=["<<number(mdCurve.tangentMPa)<<','
            <<number(cdCurve.tangentMPa)<<"] source_data_check=pass\n";
        require(authoredSlopes[0]>=mdCurve.tangentLowMPa&&
                authoredSlopes[0]<=mdCurve.tangentHighMPa&&
                authoredSlopes[1]>=cdCurve.tangentLowMPa&&
                authoredSlopes[1]<=cdCurve.tangentHighMPa,
            "authored C1 transfer scales miss the source-fit tangent intervals");
    }

    // Compare the complete implicit algorithmic tangent against branch-preserving
    // stress differences on tensile and compressive multiaxial states.
    const std::array<std::pair<paper::Matrix,const char*>,2> probes{{
        {paper::Matrix{1.04,0,0,0,1.01,0,0,0,1.005},"tensile"},
        {paper::Matrix{0.97,0,0,0,0.99,0,0,0,0.995},"compressive"}}};
    for(const auto& [fProbe,label]:probes) {
        const std::array<double,15> accepted{};
        const auto center=projectCompiled64(compiled.program,fProbe,accepted);
        paper::Matrix direction{}; direction[0]=1.0;
        const auto tangent=compiledAlgorithmicTangent15(
            compiled.program,fProbe,direction,accepted,center.state);
        constexpr double step=2.0e-7;
        paper::Matrix plusF=fProbe,minusF=fProbe;
        plusF[0]+=step;minusF[0]-=step;
        const auto plus=projectCompiled64(compiled.program,plusF,accepted);
        const auto minus=projectCompiled64(compiled.program,minusF,accepted);
        double maxError=0.0,maxScale=1.0;
        unsigned worstComponent=0u;
        double worstFiniteDifference=0.0,worstTangent=0.0;
        for(unsigned componentIndex=0;componentIndex<9u;++componentIndex) {
            const double finiteDifference=(plus.firstPiolaMPa[componentIndex]-
                minus.firstPiolaMPa[componentIndex])/(2.0*step);
            if(std::abs(finiteDifference-tangent[componentIndex])>maxError) {
                maxError=std::abs(finiteDifference-tangent[componentIndex]);
                worstComponent=componentIndex;
                worstFiniteDifference=finiteDifference;
                worstTangent=tangent[componentIndex];
            }
            maxScale=std::max({maxScale,std::abs(finiteDifference),
                               std::abs(tangent[componentIndex])});
        }
        const double relative=maxError/maxScale;
        std::cout<<candidate.name<<" tangent_branch="<<label
                 <<" relative_fd_error="<<number(relative)
                 <<" worst_component="<<worstComponent
                 <<" fd="<<number(worstFiniteDifference)
                 <<" tangent="<<number(worstTangent)<<"\n";
        require(relative<1.0e-3,
            "asymmetric transfer complete implicit tangent misses branch-preserving finite differences: "+
            candidate.name+" branch="+label+" relative="+number(relative));
    }
}

void checkMaterialSelectionDefaults(const std::filesystem::path& materialDirectory) {
    using numi::cardboard::paper_materials::selectPaths;
    const auto ordinary=selectPaths(materialDirectory,"literature2009",false);
    require(ordinary.liner.filename()=="hajali2009_liner_hill_bilinear_fit.nmatter" &&
                ordinary.medium.filename()=="hajali2009_medium_hill_bilinear_fit.nmatter",
            "ordinary literature2009 default does not select source-fit bilinear materials");
    const auto overlay=selectPaths(materialDirectory,"literature2009",true);
    require(overlay.liner.filename()=="hajali2009_liner_hill_ideal.nmatter" &&
                overlay.medium.filename()=="hajali2009_medium_hill_ideal.nmatter",
            "retained-matrix overlay base must remain the ideal-Hill assets");
    const std::filesystem::path explicitLiner="custom/liner.nmatter";
    const auto explicitPaths=selectPaths(materialDirectory,"literature2009",false,
                                         explicitLiner,{});
    require(explicitPaths.liner==explicitLiner &&
                explicitPaths.medium.filename()=="hajali2009_medium_hill_bilinear_fit.nmatter",
            "explicit material path was overridden while filling an ordinary default");
    const auto explicitOverlay=selectPaths(materialDirectory,"literature2009",true,
                                           explicitLiner,{});
    require(explicitOverlay.liner==explicitLiner &&
                explicitOverlay.medium.filename()=="hajali2009_medium_hill_ideal.nmatter",
            "explicit material path was overridden while selecting an overlay base");
    require(numi::cardboard::paper_materials::isBilinearFitIdentity(
                "hajali2009_liner_hill_bilinear_fit") &&
                numi::cardboard::paper_materials::isBilinearFitIdentity(
                "hajali2009_medium_hill_bilinear_fit") &&
                !numi::cardboard::paper_materials::isBilinearFitIdentity(
                "hajali2009_liner_hill_ideal"),
            "bilinear/ideal overlay compatibility identity guard is incorrect");
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
    std::array<float,13> identityDirection{};
    require(productionLocalSolve(fp32Jacobian,identityDirection,minimumPivot),
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

void checkCapturedElasticTangent(const ConstitutiveProgram& program) {
    // Actual root 19 / Newton 3 / environment 0 / tetrahedron 838, captured
    // from a9d801c with the literature2009 recipe. It is well inside the
    // elastic branch (q=.605), so branch crossing cannot explain the original
    // 69% tangent discrepancy. These are diagnostic inputs, not material data.
    const paper::Matrix f{1.00022197,9.71238023e-6,-6.64472827e-5,
        7.03046771e-6,.999993622,1.93879659e-5,
        .00015213844,-1.21867174e-6,1.00023377};
    const paper::Matrix h{.104916036,-.0808128715,13.2877254,
        .426670283,.0377386734,-1.49040747,
        -17.2101955,.0203557257,-1.88280761};
    const paper::State accepted{};
    const auto projected=paper::project(paper::MaterialKind::liner,f,accepted);
    require(projected.equivalentStress<.7,"captured regression must remain elastic");
    float floor=std::numeric_limits<float>::infinity();
    auto jac=compiledFP32Jacobian(program,f,accepted,projected.state,floor);
    std::array<float,13> direction{};
    for(unsigned i=0;i<13;++i) direction[i]=-evaluateFP32(program.implicitDeformationDirections[i],
        program.material,f,h,accepted,projected.state,floor);
    float pivot=std::numeric_limits<float>::infinity();
    require(productionLocalSolve(jac,direction,pivot),"captured elastic material solve failed");
    const auto reference=compiledAlgorithmicTangent(program,f,h,accepted,projected.state);
    paper::Matrix full{}; const paper::Matrix zero{};
    for(unsigned c=0;c<9;++c) {
        float value=evaluateFP32(program.tangentVector[c],program.material,f,h,accepted,projected.state,floor);
        float chain=0;
        for(unsigned s=0;s<13;++s) chain+=evaluateFP32(program.stressStateDerivatives[c*13+s],
            program.material,f,zero,accepted,projected.state,floor)*direction[s];
        full[c]=value+chain;
    }
    double error=0,scale=0;
    for(unsigned c=0;c<9;++c) {
        error=std::max(error,std::abs(full[c]-reference[c]));
        scale=std::max(scale,std::abs(reference[c]));
    }
    require(error/scale<=1.0e-3,"captured elastic FP32 tangent fails 0.1% gate: "+number(error/scale));
    require(std::abs(direction[12])<1.0e-7,"captured elastic tangent creates a spurious plastic multiplier");
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

void checkScalarInitializerAdmission(const MaterialProgram& material) {
    auto initialized=material;
    initialized.paperHillScalarInitializer=true;
    CompileOptions options;
    options.maximumRateExponent=0u;
    options.emitSpecializedMetal=false;
    const auto original=compileWorld(oneTetWorld(material),options);
    const auto candidate=compileWorld(oneTetWorld(initialized),options);
    require(original.succeeded()&&candidate.succeeded(),"scalar initializer admission failed");
    require((candidate.world.materials.at(0).flags&NM_MATERIAL_PAPER_HILL_SCALAR_INITIALIZER)!=0u,
        "scalar initializer policy was lost during compilation");
    require(candidate.world.fingerprint!=original.world.fingerprint&&
        candidate.world.physicsFingerprint!=original.world.physicsFingerprint,
        "scalar initializer policy was not fingerprinted");
    require(candidate.world.scalarPrograms.size()==original.world.scalarPrograms.size()&&
        candidate.world.instructions.size()==original.world.instructions.size(),
        "initializer changed authored equation programs");
    require(std::memcmp(candidate.world.instructions.data(),original.world.instructions.data(),
        original.world.instructions.size()*sizeof(NMExpressionInstructionGPU))==0,
        "initializer changed authored physical equations");
    auto invalid=initialized;
    invalid.parameters.front().name="unsupported_layout";
    require(!detail::compileConstitutive(invalid,NM_EXPRESSION_STACK_CAPACITY).succeeded(),
        "initializer admitted an unsupported parameter layout");
    invalid=initialized;
    invalid.stateImplicitRoots.pop_back();
    require(!detail::compileConstitutive(invalid,NM_EXPRESSION_STACK_CAPACITY).succeeded(),
        "initializer admitted an incomplete implicit system");
    auto accurate=material;
    accurate.paperCompensatedArithmetic=true;
    const auto accurateCook=compileWorld(oneTetWorld(accurate),options);
    require(accurateCook.succeeded(),"compensated arithmetic admission failed");
    require(accurateCook.world.physicsFingerprint!=original.world.physicsFingerprint,
        "compensated arithmetic policy was not fingerprinted");
    for(const auto& program:accurateCook.world.scalarPrograms)
        require((program.flags&2u)!=0u,"compensated policy was lost from a scalar program");
    auto tight=accurate;
    tight.paperHillAccurateProjection=true;
    const auto tightCook=compileWorld(oneTetWorld(tight),options);
    require(tightCook.succeeded() &&
        tightCook.world.physicsFingerprint!=accurateCook.world.physicsFingerprint,
        "accurate Hill projection was not admitted with a distinct execution identity");
    require(tightCook.world.instructions.size()==accurateCook.world.instructions.size() &&
        std::memcmp(tightCook.world.instructions.data(),accurateCook.world.instructions.data(),
            tightCook.world.instructions.size()*sizeof(NMExpressionInstructionGPU))==0,
        "accurate Hill projection changed the authored law");
    auto tightInvalid=tight;
    tightInvalid.paperCompensatedArithmetic=false;
    require(!detail::compileConstitutive(tightInvalid,NM_EXPRESSION_STACK_CAPACITY).succeeded(),
        "accurate Hill projection admitted unresolved scalar arithmetic");
    const std::array<float,9> identityF{1,0,0,0,1,0,0,0,1},zero{};
    const auto& world=accurateCook.world;
    std::vector<float> parameters;
    for(const auto& parameter:world.parameters)parameters.push_back(parameter.valueAndBounds.x);
    for(unsigned i=0;i<9u;++i) {
        bool valid=true;
        const float stress=nmPaperCompensatedBytecode<0u>(world.scalarPrograms[world.materials[0].stressProgramOffset+i],
            world.instructions.data(),parameters.data(),world.stateInitials.data(),world.stateInitials.data(),
            identityF.data(),zero.data(),zero.data(),1e-4f,293.15f,valid);
        require(valid&&std::abs(stress)<1e-5f,"compensated bytecode violated stress-free reference");
    }
    // A deformation smaller than one FP32 ULP at identity must still produce
    // the authored elastic trial state. This is the failure seen in native
    // residual sweeps; computing the bytecode accurately after discarding the
    // deformation's low word cannot repair it.
    std::array<float,9> lowF{};
    lowF[0]=1e-8f;
    for(unsigned component=0u;component<3u;++component) {
        bool valid=true;
        const auto scalar=world.scalarPrograms[world.materials[0].stateUpdateProgramOffset+6u+component];
        const float actual=nmPaperCompensatedBytecode<0u>(scalar,world.instructions.data(),
            parameters.data(),world.stateInitials.data(),world.stateInitials.data(),
            identityF.data(),zero.data(),zero.data(),1e-4f,293.15f,valid,nullptr,lowF.data());
        const unsigned stiffnessIndex=component==0u?1u:(component==1u?4u:5u);
        const double delta=double(lowF[0]);
        const double expected=double(parameters[stiffnessIndex])*(delta+0.5*delta*delta)/double(parameters[10]);
        require(valid&&std::abs(double(actual)-expected)<1e-12,
            "two-word deformation was lost before the authored trial state");
        valid=true;
        const float discarded=nmPaperCompensatedBytecode<0u>(scalar,world.instructions.data(),
            parameters.data(),world.stateInitials.data(),world.stateInitials.data(),
            identityF.data(),zero.data(),zero.data(),1e-4f,293.15f,valid);
        require(valid&&discarded==0.0f,"high-only precision-loss control did not exercise zero strain");
    }
}

void checkCompensatedStarch(const MaterialProgram& source) {
    auto candidate=source;
    candidate.paperCompensatedArithmetic=true;
    CompileOptions options;
    options.maximumRateExponent=0u;
    options.emitSpecializedMetal=false;
    const auto original=compileWorld(oneTetWorld(source),options);
    const auto accurate=compileWorld(oneTetWorld(candidate),options);
    require(original.succeeded()&&accurate.succeeded(),
        "compensated finite-starch admission failed: "+diagnostics(accurate.diagnostics));
    require(accurate.world.instructions.size()==original.world.instructions.size()&&
        std::memcmp(accurate.world.instructions.data(),original.world.instructions.data(),
            original.world.instructions.size()*sizeof(NMExpressionInstructionGPU))==0,
        "compensated starch arithmetic changed authored instructions");
    const auto& world=accurate.world;
    const auto& material=world.materials[0];
    std::vector<float> parameters;
    for(const auto& p:world.parameters)parameters.push_back(p.valueAndBounds.x);
    const double mu=parameters[1],lambda=parameters[2];
    for(double delta:{-0.15,-1e-6,-1e-8,0.0,1e-8,1e-6,0.01}) {
        std::array<float,9> f{1,0,0,0,1,0,0,0,1},low{},h{},zero{};
        f[0]=float(1+delta);low[0]=float((1+delta)-double(f[0]));h[0]=1;
        const double strain=(double(f[0])-1)+double(low[0]);
        const double stretch=1+strain,logJ=std::log1p(strain);
        const std::array<double,9> stress{
            mu*strain*(2+strain)/stretch+lambda*logJ/stretch,0,0,
            0,lambda*logJ,0,0,0,lambda*logJ};
        const std::array<double,9> tangent{
            mu+(mu+lambda*(1-logJ))/(stretch*stretch),0,0,
            0,lambda/stretch,0,0,0,lambda/stretch};
        for(unsigned component=0;component<9u;++component) {
            bool valid=true;
            const float actual=nmPaperCompensatedBytecode<0u>(
                world.scalarPrograms[material.stressProgramOffset+component],world.instructions.data(),
                parameters.data(),static_cast<const float*>(nullptr),static_cast<const float*>(nullptr),
                f.data(),h.data(),zero.data(),1e-4f,293.15f,
                valid,nullptr,low.data());
            require(valid&&std::abs(double(actual)-stress[component])<
                2e-7*std::max(1.0,std::abs(stress[component])),
                "two-word starch stress differs from independent Neo-Hookean response delta="+
                    number(delta)+" component="+std::to_string(component)+" actual="+number(actual)+
                    " expected="+number(stress[component]));
            valid=true;
            const float actualT=nmPaperCompensatedBytecode<0u>(
                world.scalarPrograms[material.tangentProgramOffset+component],world.instructions.data(),
                parameters.data(),static_cast<const float*>(nullptr),static_cast<const float*>(nullptr),
                f.data(),h.data(),zero.data(),1e-4f,293.15f,
                valid,nullptr,low.data());
            require(valid&&std::abs(double(actualT)-tangent[component])<
                2e-7*std::max(1.0,std::abs(tangent[component])),
                "two-word starch derivative differs from independent Neo-Hookean response delta="+
                    number(delta)+" component="+std::to_string(component)+" actual="+number(actualT)+
                    " expected="+number(tangent[component]));
        }
    }
    candidate.paperHillScalarInitializer=true;
    require(!detail::compileConstitutive(candidate,NM_EXPRESSION_STACK_CAPACITY).succeeded(),
        "Hill initializer admitted stateless starch");
    candidate.paperHillScalarInitializer=false;
    candidate.paperHillAccurateProjection=true;
    require(!detail::compileConstitutive(candidate,NM_EXPRESSION_STACK_CAPACITY).succeeded(),
        "accurate Hill projection admitted stateless starch");
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
    if(p.name==paper::liner.name) checkCapturedElasticTangent(compiled.program);
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

void checkSelectNonnegativeLanguageAndDerivative() {
    const std::string prefix=
        "material select_branch { parameter density : kg/m^3 = 1; "
        "parameter p : Pa = 1; state x : one = 0; model generic; "
        "energy = 0.5*p*pow(F(0,0)-1,2); update x = x; ";
    const std::string suffix=" }";
    const auto parsed=parseMatter(prefix+
        "implicit x = next(x)-select_nonnegative(next(x),next(x),0);"+suffix);
    require(parsed.succeeded(),"select_nonnegative should parse and preserve arm dimensions: "+
        diagnostics(parsed.diagnostics));
    const auto compiled=detail::compileConstitutive(parsed.material,NM_EXPRESSION_STACK_CAPACITY);
    require(compiled.succeeded(),"select_nonnegative should compile with an implicit derivative: "+
        diagnostics(compiled.diagnostics));
    const paper::Matrix identity=paper::identity(),zero{};
    const std::array<double,1> accepted{};
    const auto derivative=[&](double value) {
        const std::array<double,1> candidate{value};
        return evaluate(compiled.program.implicitJacobians.at(0),compiled.program.material,
                        identity,zero,accepted,candidate);
    };
    require(std::abs(derivative(0.25))<1.0e-12,
        "select_nonnegative positive arm derivative is incorrect");
    require(std::abs(derivative(-0.25)-1.0)<1.0e-12,
        "select_nonnegative negative arm derivative is incorrect");
    require(std::abs(derivative(0.0))<1.0e-12,
        "select_nonnegative zero tie must select the nonnegative arm derivative");
    const auto badDimensions=parseMatter(prefix+
        "implicit x = next(x)-select_nonnegative(next(x),next(x),p);"+suffix);
    require(!badDimensions.succeeded() &&
                diagnostics(badDimensions.diagnostics).find("arms require identical dimensions")!=std::string::npos,
        "select_nonnegative must reject branch arms with different dimensions");
}

void checkComplementarityBranches() {
    const auto parsed=parseMatter("material fb_branches { parameter density : kg/m^3 = 1; "
        "parameter p : Pa = 1; parameter eps : one = 0.00000001; state x : one = 0; "
        "model generic; energy = p*pow(F(0,0)-1,2); update x = x; "
        "implicit x = sqrt(pow(next(x),2)+pow(1-F(0,0),2)+pow(eps,2))-next(x)-(1-F(0,0)); }");
    require(parsed.succeeded(),"complementarity branch fixture did not parse: "+diagnostics(parsed.diagnostics));
    const auto compiled=detail::compileConstitutive(parsed.material,NM_EXPRESSION_STACK_CAPACITY);
    require(compiled.succeeded(),"complementarity branch fixture did not compile");
    for(const auto pair:std::array<std::array<double,2>,7>{{
        {0,.4},{.4,0},{-.4,.3},{.3,-.4},{1e-8,2e-8},{.2,.3},{-.2,-.3}}}) {
        paper::Matrix f=paper::identity(),h{};f[0]=1-pair[1];h[0]=1;
        paper::State old{},next{};next[0]=pair[0];
        const double a=pair[0],b=1-f[0],eps=static_cast<float>(1e-8);
        const double rho=std::sqrt(a*a+b*b+eps*eps);
        const auto coefficient=[&](double x,double y) {
            return x>=0?-(y*y+eps*eps)/(rho*(rho+x)):x/rho-1;
        };
        const double expectedX=coefficient(a,b),expectedF=-coefficient(b,a);
        const double actualX=evaluate(compiled.program.implicitJacobians.at(0),
            compiled.program.material,f,h,old,next);
        const double actualF=evaluate(compiled.program.implicitDeformationDirections.at(0),
            compiled.program.material,f,h,old,next);
        require(std::abs(actualX-expectedX)<1e-12 && std::abs(actualF-expectedF)<1e-12,
            "stable complementarity derivative disagrees with its analytic branch");
        float floor=std::numeric_limits<float>::infinity();
        const double fp32X=evaluateFP32(compiled.program.implicitJacobians.at(0),
            compiled.program.material,f,h,old,next,floor);
        const double fp32F=evaluateFP32(compiled.program.implicitDeformationDirections.at(0),
            compiled.program.material,f,h,old,next,floor);
        // At the tiny two-sided transition, the F input itself is rounded;
        // compare the native derivative at those exact representable inputs.
        const double af=static_cast<float>(a),bf=1-static_cast<float>(f[0]);
        const double rf=std::sqrt(af*af+bf*bf+eps*eps);
        const auto cf=[&](double x,double y){return x>=0?-(y*y+eps*eps)/(rf*(rf+x)):x/rf-1;};
        require(std::abs(fp32X-cf(af,bf))<1e-6 && std::abs(fp32F+cf(bf,af))<1e-6,
            "FP32 complementarity derivative loses a resolved analytic branch");
    }
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

void replayCaptured(const numi::matter::MaterialProgram& liner,
                    const numi::matter::MaterialProgram& medium) {
    using namespace numi::matter;
    const auto l=detail::compileConstitutive(liner,NM_EXPRESSION_STACK_CAPACITY);
    const auto m=detail::compileConstitutive(medium,NM_EXPRESSION_STACK_CAPACITY);
    require(l.succeeded() && m.succeeded(),"captured constitutive compilation failed");
    const std::array<ConstitutiveProgram,2> programs{l.program,m.program};
    std::string line;
    std::cout<<std::setprecision(17);
    while(std::getline(std::cin,line)) {
        if(line.empty() || line[0]=='#') continue;
        std::istringstream input(line);
        std::string key,kind;
        paper::Matrix f{},h{};
        paper::State old{},candidate{};
        input>>key>>kind;
        if(kind!="liner" && kind!="medium") throw std::runtime_error("unknown captured kind");
        if(key.empty() || !std::all_of(key.begin(),key.end(),[](unsigned char c){
            return std::isalnum(c)||c=='_'||c=='-'||c=='.';}))
            throw std::runtime_error("invalid captured key");
        const auto read=[&](auto& values){for(auto& v:values)if(!(input>>v)||!std::isfinite(v))throw std::runtime_error("invalid captured values");};
        read(f);read(h);read(old);read(candidate);
        std::string extra;if(input>>extra)throw std::runtime_error("extra captured values");
        const auto& p=programs[kind=="liner"?0:1];
        float floor=std::numeric_limits<float>::infinity();
        auto jac=compiledFP32Jacobian(p,f,old,candidate,floor);
        std::array<float,13> rhs{},direction{};
        for(unsigned i=0;i<13;++i) rhs[i]=-evaluateFP32(p.implicitDeformationDirections[i],p.material,f,h,old,candidate,floor);
        direction=rhs;
        float pivot=std::numeric_limits<float>::infinity();
        if(!productionLocalSolve(jac,direction,pivot)) throw std::runtime_error("captured local solve failed");
        paper::Matrix fixed{},chain{},full{};
        const paper::Matrix zero{};
        for(unsigned c=0;c<9;++c) {
            fixed[c]=evaluateFP32(p.tangentVector[c],p.material,f,h,old,candidate,floor);
            float contribution=0;
            for(unsigned s=0;s<13;++s) contribution+=evaluateFP32(p.stressStateDerivatives[c*13+s],p.material,f,zero,old,candidate,floor)*direction[s];
            chain[c]=contribution;full[c]=static_cast<float>(static_cast<float>(fixed[c])+contribution);
        }
        const auto fp64=compiledAlgorithmicTangent(p,f,h,old,candidate);
        auto emit=[&](const auto& values){std::cout<<'[';for(unsigned i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}std::cout<<']';};
        std::cout<<"{\"key\":\""<<key<<"\",\"fixed_candidate_tangent_fp32_Pa_per_s\":";emit(fixed);
        std::cout<<",\"state_chain_tangent_fp32_Pa_per_s\":";emit(chain);
        std::cout<<",\"full_tangent_fp32_Pa_per_s\":";emit(full);
        std::cout<<",\"full_tangent_fp64_Pa_per_s\":";emit(fp64);
        std::cout<<",\"projected_state_direction_fp32\":";emit(direction);
        std::cout<<",\"fb_jacobian_row_fp32\":[";
        for(unsigned s=0;s<13;++s){if(s)std::cout<<',';std::cout<<jac[12*13+s];}
        std::cout<<"],\"minimum_pivot\":"<<pivot<<",\"minimum_divisor\":"<<floor<<"}\n";
    }
}

int main(int argc,char** argv) {
    try {
        const auto root=std::filesystem::path(__FILE__).parent_path().parent_path();
        if(argc==2&&std::string(argv[1])=="--asymmetric-c1-calibration") {
            const auto fitCurves=readSourceFitCurves(
                root/"examples/cardboard/source-data/haj_ali_2009_bilinear_tension_fit_nodes.csv");
            const auto markerCenters=readMarkerCenters(
                root/"examples/cardboard/source-data/haj_ali_2009_experimental_marker_centers_v1.csv");
            const auto transferRecords=readAsymmetricTransferRecords(
                root/"examples/cardboard/source-data/haj_ali_2009_asymmetric_c1_transfer_v1.csv");
            checkAsymmetricTransferCandidate(
                parse(root/"materials/hajali2009_liner_hill_asymmetric_c1_transfer.nmatter"),
                parse(root/"materials/hajali2009_liner_hill_ideal.nmatter"),
                paper::MaterialKind::liner,fitCurves,markerCenters,transferRecords,true);
            checkAsymmetricTransferCandidate(
                parse(root/"materials/hajali2009_medium_hill_asymmetric_c1_transfer.nmatter"),
                parse(root/"materials/hajali2009_medium_hill_ideal.nmatter"),
                paper::MaterialKind::medium,fitCurves,markerCenters,transferRecords,true);
            std::cout<<"checks="<<checks
                <<" status=pass classification=source_curve_calibration_only_not_validation\n";
            return 0;
        }
        if((argc==4 || (argc==5&&std::string(argv[4])=="--scaled-residuals")) && std::string(argv[1])=="--replay-captured") {
            auto liner=parse(argv[2]),medium=parse(argv[3]);
            if(argc==5) {
                liner=numi::cardboard::scalePaperProjectionResiduals(std::move(liner));
                medium=numi::cardboard::scalePaperProjectionResiduals(std::move(medium));
            }
            replayCaptured(liner,medium);return 0;
        }
        const auto linerPath=argc>1?std::filesystem::path(argv[1]):root/"materials/hajali2009_liner_hill_ideal.nmatter";
        const auto mediumPath=argc>2?std::filesystem::path(argv[2]):root/"materials/hajali2009_medium_hill_ideal.nmatter";
        require(argc<=3,"expected optional liner and medium material paths");
        const MaterialProgram linerSource=parse(linerPath);
        const MaterialProgram mediumSource=parse(mediumPath);
        checkSelectNonnegativeLanguageAndDerivative();
        checkMaterialSelectionDefaults(root/"materials");
        const auto fitCurves=readSourceFitCurves(
            root/"examples/cardboard/source-data/haj_ali_2009_bilinear_tension_fit_nodes.csv");
        const auto transferRecords=readAsymmetricTransferRecords(
            root/"examples/cardboard/source-data/haj_ali_2009_asymmetric_c1_transfer_v1.csv");
        const auto linerFit=parse(root/"materials/hajali2009_liner_hill_bilinear_fit.nmatter");
        const auto mediumFit=parse(root/"materials/hajali2009_medium_hill_bilinear_fit.nmatter");
        checkBilinearFitMaterial(linerFit,parse(root/"materials/hajali2009_liner_hill_ideal.nmatter"),
            paper::MaterialKind::liner,fitCurves);
        checkBilinearFitMaterial(mediumFit,parse(root/"materials/hajali2009_medium_hill_ideal.nmatter"),
            paper::MaterialKind::medium,fitCurves);
        checkAsymmetricTransferCandidate(
            parse(root/"materials/hajali2009_liner_hill_asymmetric_c1_transfer.nmatter"),
            parse(root/"materials/hajali2009_liner_hill_ideal.nmatter"),
            paper::MaterialKind::liner,fitCurves,{},transferRecords,false);
        checkAsymmetricTransferCandidate(
            parse(root/"materials/hajali2009_medium_hill_asymmetric_c1_transfer.nmatter"),
            parse(root/"materials/hajali2009_medium_hill_ideal.nmatter"),
            paper::MaterialKind::medium,fitCurves,{},transferRecords,false);
        checkDualRootInitialization();
        checkComplementarityBranches();
        checkAuthoredSource(linerSource,paper::liner);
        checkAuthoredSource(mediumSource,paper::medium);
        checkProductionMixedProjectionTrace(linerSource);
        checkProductionMixedProjectionTrace(mediumSource);
        checkLocalMaterialBudget(linerSource);
        checkScalarInitializerAdmission(linerSource);
        checkScalarInitializerAdmission(mediumSource);
        checkCompensatedStarch(parse(linerPath.parent_path()/"starch2007_finite_glue_elastic.nmatter"));
        print(paper::liner,checkReturnMapping(paper::MaterialKind::liner));
        print(paper::medium,checkReturnMapping(paper::MaterialKind::medium));
        std::cout<<"checks="<<checks<<" status=pass scope=independent_fp64_oracle_and_matter_compiler_only\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"cardboard_paper_material_check: "<<error.what()<<"\n";
        return 1;
    }
}
