#include "numi/matter/fem_compensated_deformation.h"
#include "numi/matter/paper_compensated_bytecode.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

int main() {
    try {
        std::mt19937 generator(27u);
        std::uniform_real_distribution<float> random(-1.0f,1.0f);
        double largest=0.0, lost=0.0;
        unsigned checks=0u;
        for(unsigned sample=0;sample<4000u;++sample) {
            float u[12],inverse[9],frame[9],high[9],low[9];
            const float scale=std::pow(10.0f,-12.0f+float(sample%8u));
            for(float& v:u)v=scale*random(generator);
            for(float& v:inverse)v=1000.0f*random(generator);
            const float angle=0.37f*float(sample%7u);
            const float c=std::cos(angle),s=std::sin(angle);
            const float q[9]={c,-s,0,s,c,0,0,0,1};
            std::copy(q,q+9,frame);
            nmFEMReferenceDeformationWords(u,inverse,frame,high,low);
            double world[9];
            for(unsigned r=0;r<3u;++r)for(unsigned col=0;col<3u;++col) {
                double v=r==col?1.0:0.0;
                for(unsigned k=0;k<3u;++k)
                    v+=(double(u[3u*(k+1u)+r])-double(u[r]))*double(inverse[3u*k+col]);
                world[3u*r+col]=v;
            }
            for(unsigned r=0;r<3u;++r)for(unsigned col=0;col<3u;++col) {
                double expected=0.0;
                for(unsigned k=0;k<3u;++k)expected+=world[3u*r+k]*double(frame[3u*k+col]);
                const auto index=3u*r+col;
                const double actual=double(high[index])+double(low[index]);
                largest=std::max(largest,std::abs(actual-expected));
                lost=std::max(lost,std::abs(double(high[index])-expected));
                if(std::abs(actual-expected)>2e-13*std::max(1.0,std::abs(expected)))
                    throw std::runtime_error("two-word deformation does not resolve FP64 source kinematics");
                ++checks;
            }
        }
        // A sub-ULP extension must enter the material energy as a nonzero
        // strain; a plain FP32 I+grad(u) gives exactly identity here.
        float u[12]={0,0,0,1e-11f,0,0,0,0,0,0,0,0};
        const float identity[9]={1,0,0,0,1,0,0,0,1};
        const float inverse[9]={1000,0,0,0,1000,0,0,0,1000};
        float high[9],low[9];
        nmFEMReferenceDeformationWords(u,inverse,identity,high,low);
        if(high[0]!=1.0f||!(low[0]>9.9e-9f&&low[0]<1.01e-8f))
            throw std::runtime_error("small extension was quantized away");
        double logarithmError=0.0;
        // The old FP32 candidate cannot represent this correction, whereas
        // the residual must still resolve it below the local solve tolerance.
        float accepted[1]={0.0f},candidateHigh[1]={-0.272619992f};
        float candidateLow[1]={3.47050047e-8f};
        NMExpressionInstructionGPU instructions[3]{};
        instructions[0].opcode=NM_EXPR_NEXT_STATE;
        instructions[1].opcode=NM_EXPR_CONSTANT;
        instructions[1].immediate.x=candidateHigh[0];
        instructions[2].opcode=NM_EXPR_SUBTRACT;
        NMScalarProgramGPU program{};
        program.instructionCount=3u;program.maximumStack=2u;
        bool valid=true;
        const float resolved=nmPaperCompensatedBytecode<0u>(program,
            instructions,accepted,accepted,
            NMPaperCandidateState{candidateHigh,candidateLow},
            high,high,high,1.0f,0.0f,valid);
        if(!valid || resolved!=candidateLow[0])
            throw std::runtime_error("implicit candidate low word was discarded");
        valid=true;
        const float rounded=nmPaperCompensatedBytecode<0u>(program,
            instructions,accepted,accepted,candidateHigh,
            high,high,high,1.0f,0.0f,valid);
        if(!valid || rounded!=0.0f)
            throw std::runtime_error("legacy candidate representation changed");
        checks+=2u;
        for(int exponent=-120;exponent<=120;++exponent) {
            for(float perturbation:{-1e-8f,0.0f,1e-8f}) {
                const float a=std::ldexp(1.0f,exponent);
                const MRCompensatedScalar input{a,a*perturbation};
                const auto result=nmCompensatedLog(input);
                const double expected=std::log(double(input.high)+double(input.low));
                const double error=std::abs(double(result.high)+double(result.low)-expected);
                logarithmError=std::max(logarithmError,error);
                if(error>2e-12) throw std::runtime_error("two-word logarithm differs from FP64");
                ++checks;
            }
        }
        std::cout<<"PASS compensated deformation "<<checks
                 <<" FP64 comparisons max_abs="<<largest
                 <<" projected_high_only_max_abs="<<lost
                 <<" log_max_abs="<<logarithmError<<"\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"FAIL compensated deformation: "<<error.what()<<"\n";
        return 1;
    }
}
