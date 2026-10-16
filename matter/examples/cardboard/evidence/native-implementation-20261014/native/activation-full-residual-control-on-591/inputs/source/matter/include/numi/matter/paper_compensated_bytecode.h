#pragma once
#include "numi/matter/shared.h"
#include "metalrobo/compensated_geometry_gpu.h"

#ifndef __METAL_VERSION__
#include <cmath>
#include <limits>
#endif

// Ordered two-word arithmetic for the existing scalar instructions. This
// changes rounding only: all authored residual, Jacobian and stress programs
// still run, including their original domains and acceptance thresholds.
#ifdef __METAL_VERSION__
#define NM_PAPER_DEVICE device
#define NM_PAPER_THREAD thread
#else
#define NM_PAPER_DEVICE
#define NM_PAPER_THREAD
#endif

// Candidate precision is local to an implicit projection. Accepted history is
// still the authored FP32 state, fixed for every Newton/Armijo trial. Keeping
// the correction's low word avoids treating storage quantization as a failed
// local solve; the stress and complete derivative see the same candidate.
struct NMPaperCandidateState {
    NM_PAPER_THREAD const float* high;
    NM_PAPER_THREAD const float* low;
    inline float operator[](unsigned index) const { return high[index]; }
    inline bool operator!() const { return high == nullptr; }
    inline bool operator==(NM_PAPER_THREAD const float* other) const {
        return high == other;
    }
};
template<class Pointer>
inline MRCompensatedScalar nmPaperStateWords(Pointer state,unsigned index) {
    return {state[index],0.0f};
}
inline MRCompensatedScalar nmPaperStateWords(NMPaperCandidateState state,unsigned index) {
    return {state.high[index],state.low ? state.low[index] : 0.0f};
}

// Two-word pivoted elimination for the local implicit tangent solve. It keeps
// the same pivot floor and equations as solveLocalMaterialSystem; only the
// arithmetic precision of the compensated-material path changes.
inline bool nmPaperSolveCompensatedSystem(
    NM_PAPER_THREAD MRCompensatedScalar* matrix,
    NM_PAPER_THREAD MRCompensatedScalar* rightHandSide,
    const unsigned count) {
    if (count==0u||count>NM_MAX_MATERIAL_STATE) return false;
    for (unsigned column=0u;column<count;++column) {
        unsigned pivot=column;
        auto diagonal=matrix[column*count+column];
        float pivotMagnitude=abs(diagonal.high+diagonal.low);
        for (unsigned row=column+1u;row<count;++row) {
            const auto value=matrix[row*count+column];
            const float magnitude=abs(value.high+value.low);
            if (magnitude>pivotMagnitude) { pivot=row; pivotMagnitude=magnitude; }
        }
        if (!(pivotMagnitude>1.0e-8f)||!mrCompensatedFinite(pivotMagnitude)) return false;
        if (pivot!=column) {
            for (unsigned entry=column;entry<count;++entry) {
                const auto value=matrix[column*count+entry];
                matrix[column*count+entry]=matrix[pivot*count+entry];
                matrix[pivot*count+entry]=value;
            }
            const auto value=rightHandSide[column];
            rightHandSide[column]=rightHandSide[pivot];
            rightHandSide[pivot]=value;
        }
        diagonal=matrix[column*count+column];
        for (unsigned row=column+1u;row<count;++row) {
            const auto factor=mrCompensatedDivide(matrix[row*count+column],diagonal);
            matrix[row*count+column]={0.0f,0.0f};
            for (unsigned entry=column+1u;entry<count;++entry) {
                matrix[row*count+entry]=mrCompensatedAdd(matrix[row*count+entry],
                    mrCompensatedNegate(mrCompensatedMultiply(
                        factor,matrix[column*count+entry])));
            }
            rightHandSide[row]=mrCompensatedAdd(rightHandSide[row],
                mrCompensatedNegate(mrCompensatedMultiply(factor,rightHandSide[column])));
        }
    }
    for (int row=int(count)-1;row>=0;--row) {
        MRCompensatedScalar value=rightHandSide[unsigned(row)];
        for (unsigned column=unsigned(row)+1u;column<count;++column)
            value=mrCompensatedAdd(value,mrCompensatedNegate(mrCompensatedMultiply(
                matrix[unsigned(row)*count+column],rightHandSide[column])));
        rightHandSide[unsigned(row)]=mrCompensatedDivide(
            value,matrix[unsigned(row)*count+unsigned(row)]);
        if (!mrCompensatedFinite(rightHandSide[unsigned(row)].high)||
            !mrCompensatedFinite(rightHandSide[unsigned(row)].low)) return false;
    }
    return true;
}

// Range-reduced log with two-word arithmetic. The positive argument and
// authored domain remain unchanged. The atanh series has |y|<=1/3 after
// reduction. Twelve terms bound truncation below two-word rounding after
// centering the mantissa on one; small strains terminate at that accuracy.
inline MRCompensatedScalar nmCompensatedLog(MRCompensatedScalar argument) {
    int exponent=0;
#ifdef __METAL_VERSION__
    const float mantissa=frexp(argument.high,exponent);
    const float scaledLow=ldexp(argument.low,-exponent);
#else
    const float mantissa=std::frexp(argument.high,&exponent);
    const float scaledLow=std::ldexp(argument.low,-exponent);
#endif
    MRCompensatedScalar m{mantissa,scaledLow};
    // Center the reduced argument on one. In particular log(1+tiny strain)
    // must not subtract two rounded approximations to log(2).
    if(mantissa<0.7071067811865475f) {
        m=mrCompensatedMultiply(m,{2.0f,0.0f});
        --exponent;
    }
    const auto y=mrCompensatedDivide(mrCompensatedAdd(m,{-1.0f,0.0f}),
        mrCompensatedAdd(m,{1.0f,0.0f}));
    const auto square=mrCompensatedMultiply(y,y);
    auto term=y;
    MRCompensatedScalar sum{0.0f,0.0f};
    for(unsigned k=0u;k<12u;++k) {
        sum=mrCompensatedAdd(sum,mrCompensatedDivide(term,{float(2u*k+1u),0.0f}));
        term=mrCompensatedMultiply(term,square);
        if(abs(term.high)<=abs(sum.high)*1.0e-16f) break;
    }
    const MRCompensatedScalar logTwo{0.693147182464599609375f,-1.904654323148236e-9f};
    return mrCompensatedAdd(mrCompensatedMultiply({2.0f,0.0f},sum),
        mrCompensatedMultiply({float(exponent),0.0f},logTwo));
}

// Range-reduced exponential using the same ordered two-word operations as the
// rest of the scalar VM. The bounded residual is evaluated by a Taylor series;
// unsupported fiber-specific exponential bytecode is still rejected at compile
// time rather than silently falling back to single precision.
inline MRCompensatedScalar nmCompensatedExp(MRCompensatedScalar argument) {
    const float estimate = argument.high + argument.low;
    if (!mrCompensatedFinite(estimate) || abs(estimate) > 100.0f) {
#ifdef __METAL_VERSION__
        return {as_type<float>(0x7fc00000u), 0.0f};
#else
        return {std::numeric_limits<float>::quiet_NaN(), 0.0f};
#endif
    }
    const float log2Estimate = estimate * 1.4426950408889634f;
#ifdef __METAL_VERSION__
    const int exponent = log2Estimate >= 0.0f
        ? int(floor(log2Estimate + 0.5f))
        : int(ceil(log2Estimate - 0.5f));
#else
    const int exponent = log2Estimate >= 0.0f
        ? int(std::floor(log2Estimate + 0.5f))
        : int(std::ceil(log2Estimate - 0.5f));
#endif
    const MRCompensatedScalar logTwo{
        0.693147182464599609375f, -1.904654323148236e-9f};
    const auto reduced = mrCompensatedAdd(argument,
        mrCompensatedNegate(mrCompensatedMultiply(
            {float(exponent), 0.0f}, logTwo)));
    MRCompensatedScalar sum{1.0f, 0.0f};
    MRCompensatedScalar term{1.0f, 0.0f};
    for (unsigned order = 1u; order <= 14u; ++order) {
        term = mrCompensatedDivide(
            mrCompensatedMultiply(term, reduced),
            {float(order), 0.0f});
        sum = mrCompensatedAdd(sum, term);
    }
#ifdef __METAL_VERSION__
    const MRCompensatedScalar result{
        ldexp(sum.high, exponent), ldexp(sum.low, exponent)};
#else
    const MRCompensatedScalar result{
        std::ldexp(sum.high, exponent), std::ldexp(sum.low, exponent)};
#endif
    return result;
}

inline MRCompensatedScalar nmCompensatedExpm1MinusX(
    MRCompensatedScalar argument) {
    if (abs(argument.high + argument.low) < 0.25f) {
        MRCompensatedScalar term = mrCompensatedDivide(
            mrCompensatedMultiply(argument, argument), {2.0f, 0.0f});
        MRCompensatedScalar sum = term;
        for (unsigned order = 3u; order <= 20u; ++order) {
            term = mrCompensatedDivide(
                mrCompensatedMultiply(term, argument),
                {float(order), 0.0f});
            sum = mrCompensatedAdd(sum, term);
        }
        return sum;
    }
    return mrCompensatedAdd(
        mrCompensatedAdd(nmCompensatedExp(argument), {-1.0f, 0.0f}),
        mrCompensatedNegate(argument));
}

template<unsigned CacheCapacity,class StatePointer,class CandidatePointer>
inline float nmPaperCompensatedBytecode(
    const NMScalarProgramGPU program,
    NM_PAPER_DEVICE const NMExpressionInstructionGPU* instructions,
    NM_PAPER_DEVICE const float* parameters,
    StatePointer accepted, CandidatePointer candidate,
    NM_PAPER_THREAD const float* f,NM_PAPER_THREAD const float* direction,
    NM_PAPER_THREAD const float* rate,float timestep,float temperature,
    NM_PAPER_THREAD bool& valid,NM_PAPER_THREAD bool* nonfinite=nullptr,
    NM_PAPER_THREAD const float* fLow=nullptr,
    NM_PAPER_THREAD MRCompensatedScalar* resultWords=nullptr) {
    if (resultWords) *resultWords={0.0f,0.0f};
    if(program.maximumStack>NM_EXPRESSION_STACK_CAPACITY ||
       !(timestep>=0.0f)||!mrCompensatedFinite(timestep)||!mrCompensatedFinite(temperature)) {
        valid=false;return 0.0f;
    }
    MRCompensatedScalar stack[NM_EXPRESSION_STACK_CAPACITY];
    MRCompensatedScalar cache[CacheCapacity>0u?CacheCapacity:1u];
    unsigned depth=0;
    const auto push=[&](MRCompensatedScalar x) {
        if(depth>=NM_EXPRESSION_STACK_CAPACITY||!mrCompensatedFinite(x.high)||!mrCompensatedFinite(x.low)) {
            if(nonfinite&&(!mrCompensatedFinite(x.high)||!mrCompensatedFinite(x.low))) *nonfinite=true;
            valid=false;return;
        }
        stack[depth++]=x;
    };
    const auto pop=[&]() -> MRCompensatedScalar {
        if(depth==0) {valid=false;return {0,0};}
        return stack[--depth];
    };
    const auto less=[](MRCompensatedScalar a,MRCompensatedScalar b) {
        const auto d=mrCompensatedAdd(a,mrCompensatedNegate(b));
        return d.high<0.0f||(d.high==0.0f&&d.low<0.0f);
    };
    for(unsigned i=0;i<program.instructionCount;++i) {
        const auto instruction=instructions[program.firstInstruction+i];
        switch(instruction.opcode) {
        case NM_EXPR_CONSTANT:push({instruction.immediate.x,0});break;
        case NM_EXPR_PARAMETER:push({parameters[instruction.index],0});break;
        case NM_EXPR_F:push({f[instruction.index],fLow ? fLow[instruction.index] : 0.0f});break;
        case NM_EXPR_DF:push({direction[instruction.index],0});break;
        case NM_EXPR_RATE:push({rate[instruction.index],0});break;
        case NM_EXPR_DT:push({timestep,0});break;
        case NM_EXPR_TEMPERATURE:push({temperature,0});break;
        case NM_EXPR_STATE:
            if(!accepted||instruction.index>=NM_MAX_MATERIAL_STATE)valid=false;
            else push(nmPaperStateWords(accepted,instruction.index));break;
        case NM_EXPR_NEXT_STATE:
            if(!candidate||instruction.index>=NM_MAX_MATERIAL_STATE)valid=false;
            else push(nmPaperStateWords(candidate,instruction.index));break;
        case NM_EXPR_ADD: {const auto b=pop(),a=pop();push(mrCompensatedAdd(a,b));break;}
        case NM_EXPR_SUBTRACT: {const auto b=pop(),a=pop();push(mrCompensatedAdd(a,mrCompensatedNegate(b)));break;}
        case NM_EXPR_MULTIPLY: {const auto b=pop(),a=pop();push(mrCompensatedMultiply(a,b));break;}
        case NM_EXPR_DIVIDE: {
            const auto b=pop(),a=pop();
            const float denominator=b.high+b.low;
            if(!(denominator>1e-12f||denominator<-1e-12f))valid=false;
            else push(mrCompensatedDivide(a,b));break;
        }
        case NM_EXPR_NEGATE:push(mrCompensatedNegate(pop()));break;
        case NM_EXPR_SQRT: {
            const auto a=pop();
            if(less(a,{0,0}))valid=false;
            else push(a.high==0.0f&&a.low==0.0f?MRCompensatedScalar{0,0}:mrCompensatedSqrt(a));break;
        }
        case NM_EXPR_LOG: {
            const auto a=pop();
            if(!(a.high>0.0f))valid=false;
            else push(nmCompensatedLog(a));break;
        }
        case NM_EXPR_EXP:
            push(nmCompensatedExp(pop()));break;
        case NM_EXPR_EXPM1_MINUS_X:
            push(nmCompensatedExpm1MinusX(pop()));break;
        case NM_EXPR_ABS: {const auto a=pop();push(less(a,{0,0})?mrCompensatedNegate(a):a);break;}
        case NM_EXPR_MIN: {const auto b=pop(),a=pop();push(less(a,b)?a:b);break;}
        case NM_EXPR_MAX: {const auto b=pop(),a=pop();push(less(a,b)?b:a);break;}
        case NM_EXPR_CLAMP: {
            const auto upper=pop(),lower=pop(),a=pop();
            if(less(upper,lower))valid=false;
            else push(less(a,lower)?lower:(less(upper,a)?upper:a));break;
        }
        case NM_EXPR_POW_INTEGER: {
            const auto a=pop();const int exponent=instruction.integer;
            const float value=a.high+a.low;
            if(exponent<0&&!(value>1e-12f||value<-1e-12f)) {valid=false;break;}
            MRCompensatedScalar result{1,0},factor=a;
            unsigned power=unsigned(exponent<0?-exponent:exponent);
            while(power) {
                if(power&1u)result=mrCompensatedMultiply(result,factor);
                factor=mrCompensatedMultiply(factor,factor);power>>=1u;
            }
            push(exponent<0?mrCompensatedDivide({1,0},result):result);break;
        }
        case NM_EXPR_SELECT_NONNEGATIVE: {
            const auto negative=pop(),nonnegativeValue=pop(),condition=pop();
            push(less(condition,{0,0})?negative:nonnegativeValue);break;
        }
        case NM_EXPR_CACHE_STORE:
            if(!(program.flags&1u)||!depth||instruction.index>=CacheCapacity)valid=false;
            else cache[instruction.index]=stack[depth-1u];break;
        case NM_EXPR_CACHE_LOAD:
            if(!(program.flags&1u)||instruction.index>=CacheCapacity)valid=false;
            else push(cache[instruction.index]);break;
        default:valid=false;break; // Compiler rejects unsupported transcendentals.
        }
        if(!valid)return 0.0f;
    }
    if(depth!=1u) {valid=false;return 0.0f;}
    const float result=stack[0].high+stack[0].low;
    if(!mrCompensatedFinite(result)) {valid=false;if(nonfinite)*nonfinite=true;return 0.0f;}
    if (resultWords) *resultWords=stack[0];
    return result;
}
#undef NM_PAPER_DEVICE
#undef NM_PAPER_THREAD
