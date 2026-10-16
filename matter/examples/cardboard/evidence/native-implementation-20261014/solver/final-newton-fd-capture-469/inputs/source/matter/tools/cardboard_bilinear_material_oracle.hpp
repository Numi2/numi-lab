#pragma once

// Independent FP64 interpreter/projector for the compiled 15-state
// Haj-Ali tensile-fit programs. It evaluates the compiler's scalar bytecode
// with double intermediates and solves the full authored implicit system;
// it does not call the production Metal projection or stress/tangent path.

#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace numi_cardboard_bilinear_oracle {

constexpr unsigned kStateCount = 15u;
using Matrix = std::array<double, 9>;
using State = std::array<double, kStateCount>;
using FloatState = std::array<float, kStateCount>;
using Residual = std::array<double, kStateCount>;
using Jacobian = std::array<double, kStateCount * kStateCount>;

struct Point {
    State state{};
    Matrix firstPiolaPa{};
    double residualInfinity = 0.0;
    unsigned iterations = 0u;
};

struct Float32RestDiagnostic {
    std::array<float, kStateCount> candidate{};
    std::array<float, kStateCount> residual{};
    std::array<float, kStateCount> pivotMagnitudes{};
    float residualInfinity = 0.0f;
    float minimumDivision = std::numeric_limits<float>::infinity();
    float minimumUpdateDivision = std::numeric_limits<float>::infinity();
    float minimumResidualDivision = std::numeric_limits<float>::infinity();
    float minimumJacobianDivision = std::numeric_limits<float>::infinity();
    unsigned minimumJacobianDivisionRow = 0u;
    unsigned minimumJacobianDivisionColumn = 0u;
    float minimumPivot = std::numeric_limits<float>::infinity();
    int firstRejectedPivot = -1;
    bool divisionDomainValid = true;
};

// CPU-only reproduction of the ordinary FP32 bytecode inputs at F=I. This
// intentionally does not invoke Metal; it reports whether the authored rest
// candidate reaches the same 1e-8 pivot gate before a GPU run is requested.
inline float evaluateFloat32(const numi::matter::ScalarBytecode& code,
                             const numi::matter::MaterialProgram& material,
                             const Matrix& f, const FloatState& accepted,
                             const FloatState& candidate, float& minimumDivision,
                             bool& divisionDomainValid) {
    std::vector<float> stack;
    std::array<float, NM_EXPRESSION_CACHE_CAPACITY> cache{};
    const auto pop = [&]() {
        if (stack.empty()) throw std::runtime_error("FP32 bytecode stack underflow");
        const float value = stack.back();
        stack.pop_back();
        return value;
    };
    for (const auto& instruction : code.instructions) {
        float value = 0.0f;
        switch (instruction.opcode) {
        case NM_EXPR_CONSTANT: value = instruction.immediate.x; break;
        case NM_EXPR_PARAMETER:
            value = static_cast<float>(material.parameters.at(instruction.index).defaultValue);
            break;
        case NM_EXPR_F: value = static_cast<float>(f.at(instruction.index)); break;
        case NM_EXPR_DF: value = 0.0f; break;
        case NM_EXPR_RATE: value = 0.0f; break;
        case NM_EXPR_STATE: value = static_cast<float>(accepted.at(instruction.index)); break;
        case NM_EXPR_NEXT_STATE: value = static_cast<float>(candidate.at(instruction.index)); break;
        case NM_EXPR_DT: value = 1.0e-4f; break;
        case NM_EXPR_TEMPERATURE: value = 293.15f; break;
        case NM_EXPR_ADD: { const float b=pop(); value=pop()+b; break; }
        case NM_EXPR_SUBTRACT: { const float b=pop(); value=pop()-b; break; }
        case NM_EXPR_MULTIPLY: { const float b=pop(); value=pop()*b; break; }
        case NM_EXPR_DIVIDE: {
            const float b=pop(), a=pop();
            minimumDivision=std::min(minimumDivision,std::abs(b));
            if (!(std::abs(b)>1.0e-12f)) {
                divisionDomainValid=false;
                value=std::numeric_limits<float>::quiet_NaN();
            } else value=a/b;
            break;
        }
        case NM_EXPR_NEGATE: value=-pop(); break;
        case NM_EXPR_LOG: value=std::log(pop()); break;
        case NM_EXPR_EXP: value=std::exp(pop()); break;
        case NM_EXPR_EXPM1_MINUS_X: { const float x=pop(); value=std::expm1(x)-x; break; }
        case NM_EXPR_SQRT: { const float x=pop(); value=x<0.0f?std::numeric_limits<float>::quiet_NaN():std::sqrt(x); break; }
        case NM_EXPR_ABS: value=std::abs(pop()); break;
        case NM_EXPR_MIN: { const float b=pop(); value=std::min(pop(),b); break; }
        case NM_EXPR_MAX: { const float b=pop(); value=std::max(pop(),b); break; }
        case NM_EXPR_POW_INTEGER: {
            const float x=pop();
            if (instruction.integer<0 && !(std::abs(x)>1.0e-12f)) {
                divisionDomainValid=false;
                value=std::numeric_limits<float>::quiet_NaN();
            } else value=std::pow(x,instruction.integer);
            break;
        }
        case NM_EXPR_CLAMP: {
            const float upper=pop(),lower=pop(),x=pop();
            value=upper<lower?std::numeric_limits<float>::quiet_NaN():std::clamp(x,lower,upper);
            break;
        }
        case NM_EXPR_SELECT_NONNEGATIVE: {
            const float negative=pop(), nonnegative=pop(), condition=pop();
            value=condition>=0.0f?nonnegative:negative;
            break;
        }
        case NM_EXPR_CACHE_STORE:
            if (instruction.index>=cache.size() || stack.empty())
                throw std::runtime_error("invalid FP32 expression-cache store");
            cache[instruction.index]=stack.back();
            continue;
        case NM_EXPR_CACHE_LOAD:
            if (instruction.index>=cache.size())
                throw std::runtime_error("invalid FP32 expression-cache load");
            value=cache[instruction.index];
            break;
        default:
            throw std::runtime_error("unsupported FP32 bytecode opcode: "+
                                     std::to_string(instruction.opcode));
        }
        stack.push_back(value);
    }
    if (stack.size()!=1u) throw std::runtime_error("FP32 bytecode did not reduce to one value");
    return stack.back();
}

inline Float32RestDiagnostic diagnoseFloat32Rest(
    const numi::matter::ConstitutiveProgram& program) {
    if (program.stateUpdates.size()!=kStateCount ||
        program.implicitResiduals.size()!=kStateCount ||
        program.implicitJacobians.size()!=kStateCount*kStateCount)
        throw std::runtime_error("FP32 rest diagnostic requires the full 15-state system");
    Float32RestDiagnostic result;
    FloatState accepted{};
    for (unsigned i=0;i<kStateCount;++i)
        accepted[i]=static_cast<float>(program.material.internalState.at(i).initialValue);
    const Matrix identity{1,0,0,0,1,0,0,0,1};
    const FloatState zero{};
    bool domainsValid=true;
    for (unsigned i=0;i<kStateCount;++i) {
        float localMinimum=std::numeric_limits<float>::infinity();
        result.candidate[i]=evaluateFloat32(program.stateUpdates[i],program.material,
            identity,accepted,zero,localMinimum,domainsValid);
        result.minimumUpdateDivision=std::min(result.minimumUpdateDivision,localMinimum);
    }
    for (unsigned i=0;i<kStateCount;++i) {
        float localMinimum=std::numeric_limits<float>::infinity();
        result.residual[i]=evaluateFloat32(program.implicitResiduals[i],program.material,
            identity,accepted,result.candidate,localMinimum,domainsValid);
        result.minimumResidualDivision=std::min(result.minimumResidualDivision,localMinimum);
        result.residualInfinity=std::max(result.residualInfinity,std::abs(result.residual[i]));
    }
    std::array<float,kStateCount*kStateCount> matrix{};
    for (unsigned i=0;i<matrix.size();++i) {
        float localMinimum=std::numeric_limits<float>::infinity();
        matrix[i]=evaluateFloat32(program.implicitJacobians[i],program.material,
            identity,accepted,result.candidate,localMinimum,domainsValid);
        if (localMinimum<result.minimumJacobianDivision) {
            result.minimumJacobianDivision=localMinimum;
            result.minimumJacobianDivisionRow=static_cast<unsigned>(i)/kStateCount;
            result.minimumJacobianDivisionColumn=static_cast<unsigned>(i)%kStateCount;
        }
    }
    result.minimumDivision=std::min({result.minimumUpdateDivision,
        result.minimumResidualDivision,result.minimumJacobianDivision});
    result.divisionDomainValid=domainsValid;
    for (unsigned column=0;column<kStateCount;++column) {
        unsigned pivot=column;
        float magnitude=std::abs(matrix[kStateCount*column+column]);
        for (unsigned row=column+1;row<kStateCount;++row) {
            const float candidate=std::abs(matrix[kStateCount*row+column]);
            if (candidate>magnitude) { magnitude=candidate; pivot=row; }
        }
        result.pivotMagnitudes[column]=magnitude;
        result.minimumPivot=std::min(result.minimumPivot,magnitude);
        if (!(magnitude>1.0e-8f) && result.firstRejectedPivot<0)
            result.firstRejectedPivot=static_cast<int>(column);
        if (!std::isfinite(magnitude)) {
            result.firstRejectedPivot=static_cast<int>(column);
            break;
        }
        if (pivot!=column) for (unsigned entry=column;entry<kStateCount;++entry)
            std::swap(matrix[kStateCount*column+entry],matrix[kStateCount*pivot+entry]);
        const float diagonal=matrix[kStateCount*column+column];
        for (unsigned row=column+1;row<kStateCount;++row) {
            const float factor=matrix[kStateCount*row+column]/diagonal;
            for (unsigned entry=column+1;entry<kStateCount;++entry)
                matrix[kStateCount*row+entry]-=factor*matrix[kStateCount*column+entry];
        }
    }
    return result;
}

inline double evaluate(const numi::matter::ScalarBytecode& code,
                       const numi::matter::MaterialProgram& material,
                       const Matrix& f, const Matrix& h,
                       const State& accepted, const State& candidate,
                       const double dt = 1.0e-4,
                       const double temperature = 293.15) {
    std::vector<double> stack;
    std::array<double, NM_EXPRESSION_CACHE_CAPACITY> cache{};
    const auto pop = [&]() {
        if (stack.empty()) throw std::runtime_error("FP64 bytecode stack underflow");
        const double value = stack.back();
        stack.pop_back();
        return value;
    };
    const auto push = [&](const double value) {
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite FP64 bytecode value");
        stack.push_back(value);
    };
    for (const auto& instruction : code.instructions) {
        double value = 0.0;
        switch (instruction.opcode) {
        case NM_EXPR_CONSTANT: value = instruction.immediate.x; break;
        case NM_EXPR_PARAMETER:
            value = static_cast<float>(material.parameters.at(instruction.index).defaultValue);
            break;
        case NM_EXPR_F: value = f.at(instruction.index); break;
        case NM_EXPR_DF: value = h.at(instruction.index); break;
        case NM_EXPR_RATE: value = 0.0; break;
        case NM_EXPR_STATE: value = accepted.at(instruction.index); break;
        case NM_EXPR_NEXT_STATE: value = candidate.at(instruction.index); break;
        case NM_EXPR_DT: value = dt; break;
        case NM_EXPR_TEMPERATURE: value = temperature; break;
        case NM_EXPR_ADD: { const double b=pop(); value=pop()+b; break; }
        case NM_EXPR_SUBTRACT: { const double b=pop(); value=pop()-b; break; }
        case NM_EXPR_MULTIPLY: { const double b=pop(); value=pop()*b; break; }
        case NM_EXPR_DIVIDE: {
            const double b=pop(), a=pop();
            if (!(std::abs(b)>1.0e-12)) throw std::runtime_error("FP64 bytecode division floor");
            value=a/b;
            break;
        }
        case NM_EXPR_NEGATE: value=-pop(); break;
        case NM_EXPR_LOG: { const double x=pop(); if (!(x>0.0)) throw std::runtime_error("log domain"); value=std::log(x); break; }
        case NM_EXPR_EXP: value=std::exp(pop()); break;
        case NM_EXPR_EXPM1_MINUS_X: { const double x=pop(); value=std::expm1(x)-x; break; }
        case NM_EXPR_SQRT: { const double x=pop(); if (x<0.0) throw std::runtime_error("sqrt domain"); value=std::sqrt(x); break; }
        case NM_EXPR_ABS: value=std::abs(pop()); break;
        case NM_EXPR_MIN: { const double b=pop(),a=pop(); value=std::min(a,b); break; }
        case NM_EXPR_MAX: { const double b=pop(),a=pop(); value=std::max(a,b); break; }
        case NM_EXPR_POW_INTEGER: {
            const double x=pop();
            if (instruction.integer<0 && !(std::abs(x)>1.0e-12))
                throw std::runtime_error("negative power floor");
            value=std::pow(x,instruction.integer);
            break;
        }
        case NM_EXPR_CLAMP: {
            const double upper=pop(),lower=pop(),x=pop();
            if (upper<lower) throw std::runtime_error("invalid clamp bounds");
            value=std::clamp(x,lower,upper);
            break;
        }
        case NM_EXPR_SELECT_NONNEGATIVE: {
            const double negative=pop(), nonnegative=pop(), condition=pop();
            value=condition>=0.0 ? nonnegative : negative;
            break;
        }
        case NM_EXPR_CACHE_STORE:
            if (instruction.index>=cache.size() || stack.empty())
                throw std::runtime_error("invalid expression-cache store");
            cache[instruction.index]=stack.back();
            continue;
        case NM_EXPR_CACHE_LOAD:
            if (instruction.index>=cache.size()) throw std::runtime_error("invalid expression-cache load");
            value=cache[instruction.index];
            break;
        default:
            throw std::runtime_error("unsupported bytecode opcode in bilinear oracle: "+
                                     std::to_string(instruction.opcode));
        }
        push(value);
    }
    if (stack.size()!=1u) throw std::runtime_error("FP64 bytecode did not reduce to one value");
    return stack.back();
}

inline double normInfinity(const Residual& value) {
    double norm=0.0;
    for (const double x : value) norm=std::max(norm,std::abs(x));
    return norm;
}

inline Residual residual(const numi::matter::ConstitutiveProgram& program,
                         const Matrix& f, const State& accepted,
                         const State& candidate) {
    if (program.implicitResiduals.size()!=kStateCount)
        throw std::runtime_error("bilinear oracle requires fifteen implicit states");
    Residual result{};
    const Matrix zero{};
    for (unsigned i=0;i<kStateCount;++i)
        result[i]=evaluate(program.implicitResiduals[i],program.material,f,zero,
                           accepted,candidate);
    return result;
}

inline Jacobian jacobian(const numi::matter::ConstitutiveProgram& program,
                         const Matrix& f, const State& accepted,
                         const State& candidate) {
    if (program.implicitJacobians.size()!=kStateCount*kStateCount)
        throw std::runtime_error("bilinear oracle requires a complete 15x15 local Jacobian");
    Jacobian result{};
    const Matrix zero{};
    for (unsigned i=0;i<result.size();++i)
        result[i]=evaluate(program.implicitJacobians[i],program.material,f,zero,
                           accepted,candidate);
    return result;
}

inline bool solve(Jacobian matrix, State rhs, State& solution) {
    constexpr unsigned n=kStateCount;
    for (unsigned column=0;column<n;++column) {
        unsigned pivot=column;
        double magnitude=std::abs(matrix[n*column+column]);
        for (unsigned row=column+1;row<n;++row) {
            const double candidate=std::abs(matrix[n*row+column]);
            if (candidate>magnitude) { magnitude=candidate; pivot=row; }
        }
        if (!std::isfinite(magnitude) || magnitude<1.0e-14) return false;
        if (pivot!=column) {
            for (unsigned j=column;j<n;++j) std::swap(matrix[n*column+j],matrix[n*pivot+j]);
            std::swap(rhs[column],rhs[pivot]);
        }
        const double diagonal=matrix[n*column+column];
        for (unsigned row=column+1;row<n;++row) {
            const double factor=matrix[n*row+column]/diagonal;
            for (unsigned j=column+1;j<n;++j) matrix[n*row+j]-=factor*matrix[n*column+j];
            rhs[row]-=factor*rhs[column];
        }
    }
    for (int row=static_cast<int>(n)-1;row>=0;--row) {
        double value=rhs[static_cast<unsigned>(row)];
        for (unsigned column=static_cast<unsigned>(row)+1;column<n;++column)
            value-=matrix[n*static_cast<unsigned>(row)+column]*solution[column];
        solution[static_cast<unsigned>(row)]=
            value/matrix[n*static_cast<unsigned>(row)+static_cast<unsigned>(row)];
        if (!std::isfinite(solution[static_cast<unsigned>(row)])) return false;
    }
    return true;
}

inline Point project(const numi::matter::ConstitutiveProgram& program,
                     const Matrix& f, const State& accepted) {
    if (program.stateUpdates.size()!=kStateCount)
        throw std::runtime_error("bilinear oracle requires fifteen state initializers");
    State candidate{};
    const State zeroState{};
    const Matrix zero{};
    for (unsigned i=0;i<kStateCount;++i)
        candidate[i]=evaluate(program.stateUpdates[i],program.material,f,zero,
                              accepted,zeroState);

    constexpr double tolerance=2.0e-11;
    Point result;
    bool converged=false;
    for (unsigned iteration=0;iteration<256u;++iteration) {
        const Residual current=residual(program,f,accepted,candidate);
        const double currentNorm=normInfinity(current);
        if (currentNorm<tolerance) { result.iterations=iteration; converged=true; break; }
        Jacobian local=jacobian(program,f,accepted,candidate);
        State rhs{},correction{};
        for (unsigned row=0;row<kStateCount;++row) rhs[row]=-current[row];
        if (!solve(local,rhs,correction)) throw std::runtime_error("FP64 15-state Jacobian is singular");
        double alpha=1.0;
        bool acceptedStep=false;
        for (unsigned backtrack=0;backtrack<24u;++backtrack) {
            State trial=candidate;
            for (unsigned state=0;state<kStateCount;++state) trial[state]+=alpha*correction[state];
            const double trialNorm=normInfinity(residual(program,f,accepted,trial));
            if (std::isfinite(trialNorm) && trialNorm<currentNorm) {
                candidate=trial;
                acceptedStep=true;
                break;
            }
            alpha*=0.5;
        }
        if (!acceptedStep) throw std::runtime_error("FP64 15-state local projection line search stalled");
    }
    result.residualInfinity=normInfinity(residual(program,f,accepted,candidate));
    if (!converged && result.residualInfinity<tolerance) { result.iterations=256u; converged=true; }
    if (!converged) throw std::runtime_error("FP64 15-state local projection did not converge");
    result.state=candidate;
    for (unsigned component=0;component<9;++component)
        result.firstPiolaPa[component]=evaluate(program.stress[component],program.material,
                                               f,zero,accepted,candidate);
    return result;
}

inline Matrix tangent(const numi::matter::ConstitutiveProgram& program,
                     const Matrix& f, const Matrix& h,
                     const State& accepted, const State& candidate) {
    const Jacobian local=jacobian(program,f,accepted,candidate);
    State rhs{},stateDirection{};
    for (unsigned state=0;state<kStateCount;++state)
        rhs[state]=-evaluate(program.implicitDeformationDirections.at(state),
                             program.material,f,h,accepted,candidate);
    if (!solve(local,rhs,stateDirection)) throw std::runtime_error("FP64 tangent state solve is singular");
    Matrix result{};
    const Matrix zero{};
    for (unsigned component=0;component<9;++component) {
        result[component]=evaluate(program.tangentVector[component],program.material,
                                  f,h,accepted,candidate);
        for (unsigned state=0;state<kStateCount;++state)
            result[component]+=evaluate(program.stressStateDerivatives.at(
                    component*kStateCount+state),program.material,f,zero,
                    accepted,candidate)*stateDirection[state];
    }
    return result;
}

} // namespace numi_cardboard_bilinear_oracle
