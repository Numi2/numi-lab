#include "numi/matter/detail.hpp"
#include "numi/matter/fiber_exp_linear.h"
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Matrix = std::array<double, 9>;
constexpr Matrix identity{1,0,0,0,1,0,0,0,1};
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
struct Tissue {
    const char* name; double c3, c4, c5, limit;
    double c1=1.95e6, bulk=146.41e6, stretch=1.016, fiberScale=1;
};
constexpr std::array<Tissue,9> tissues{{
    {"ACL",13900,116.22,535039000,1.046,1.95e6,146.41e6,1.016},
    {"PCL",119600,87.178,431063000,1.035,3.25e6,243.9e6,1},
    {"MCL",570000,48,467100000,1.063,1.44e6,793.65e6,1.034},
    {"LCL",570000,48,467100000,1.063,1.44e6,793.65e6,1.027},
    {"PTL",65000,115.89,777560000,1.042,2.75e6,206.61e6,1},
    {"QAT",65000,115.89,777560000,1.042,2.75e6,206.61e6,1},
    {"MNS-L",119700,150,400000000,1.019,4.61e6,92.16e6,1},
    {"MNS-M",119700,150,400000000,1.019,4.61e6,92.16e6,1},
    // All four source cartilage materials share these c1/c2/k values.
    {"cartilage",13900,116.22,535039000,1.046,2.54e6,100e6,1,0}}};
double sourceStress(const Tissue& t, double x) {
    if (x < 1) return 0;
    const double q = x < t.limit ? t.c3 * std::expm1(t.c4*(x-1)) :
        t.c3 * std::expm1(t.c4*(t.limit-1)) + t.c5*(x-t.limit);
    return q/x;
}
// Independent composite Simpson integration of the published q/lambda law.
double sourceEnergy(const Tissue& t, double x) {
    if (x <= 1) return 0;
    const double top=std::min(x,t.limit), h=(top-1)/4096;
    double sum=sourceStress(t,1)+sourceStress(t,top);
    for (unsigned i=1;i<4096;++i) sum+=(i%2?4:2)*sourceStress(t,1+i*h);
    const double q=t.c3*std::expm1(t.c4*(t.limit-1));
    return sum*h/3 + (x>top ? t.c5*(x-top)+(q-t.c5*top)*std::log(x/top) : 0);
}

// Independent tensor form: Cauchy stress from deviatoric B and the current
// fibre dyad, then pull back to first Piola. This does not call the material
// expression graph, its symbolic derivatives, or the fibre primitive.
// It checks the c2=0 source equations, not a run of FEBio 2.9.1.
Matrix sourcePiola(const Tissue& t,const Matrix& f,const std::array<double,3>& a,double* energy=nullptr) {
    const double j=f[0]*(f[4]*f[8]-f[5]*f[7])-f[1]*(f[3]*f[8]-f[5]*f[6])+
        f[2]*(f[3]*f[7]-f[4]*f[6]);
    require(j>0,"oracle inverted deformation");
    // Cofactor = J F^-T.
    Matrix cof{f[4]*f[8]-f[5]*f[7],f[5]*f[6]-f[3]*f[8],f[3]*f[7]-f[4]*f[6],
        f[2]*f[7]-f[1]*f[8],f[0]*f[8]-f[2]*f[6],f[1]*f[6]-f[0]*f[7],
        f[1]*f[5]-f[2]*f[4],f[2]*f[3]-f[0]*f[5],f[0]*f[4]-f[1]*f[3]};
    Matrix fp{},elastic{},b{},sigma{},p{};
    std::array<double,3> current{};
    const double transverse=1/std::sqrt(t.stretch);
    for(unsigned i=0;i<3;++i) for(unsigned k=0;k<3;++k)
        fp[3*i+k]=(i==k?transverse:0)+(t.stretch-transverse)*a[i]*a[k];
    for(unsigned i=0;i<3;++i) for(unsigned k=0;k<3;++k) for(unsigned l=0;l<3;++l)
        elastic[3*i+k]+=f[3*i+l]*fp[3*l+k];
    double norm=0;
    for(unsigned i=0;i<3;++i){
        for(unsigned k=0;k<3;++k)current[i]+=elastic[3*i+k]*a[k];
        norm+=current[i]*current[i];
    }
    norm=std::sqrt(norm);
    const double lambda=norm*std::pow(j,-1.0/3.0);
    const double q=lambda*sourceStress(t,lambda)*t.fiberScale;
    double trace=0;
    for(unsigned i=0;i<3;++i)for(unsigned k=0;k<3;++k){
        for(unsigned l=0;l<3;++l)b[3*i+k]+=elastic[3*i+l]*elastic[3*k+l];
        b[3*i+k]*=std::pow(j,-2.0/3.0);
        if(i==k)trace+=b[3*i+k];
    }
    if(energy)*energy=t.c1*(trace-3)+.5*t.bulk*std::pow(std::log(j),2)+t.fiberScale*sourceEnergy(t,lambda);
    for(unsigned i=0;i<3;++i)for(unsigned k=0;k<3;++k){
        sigma[3*i+k]=(2*t.c1*(b[3*i+k]-(i==k?trace/3:0))+
            q*(current[i]*current[k]/(norm*norm)-(i==k?1.0/3:0)))/j+
            (i==k?t.bulk*std::log(j)/j:0);
    }
    for(unsigned i=0;i<3;++i)for(unsigned k=0;k<3;++k)for(unsigned l=0;l<3;++l)
        p[3*i+k]+=sigma[3*i+l]*cof[3*l+k];
    return p;
}
double evaluate(const numi::matter::ScalarBytecode& program,
                const numi::matter::MaterialProgram& material,
                const Matrix& f, const Matrix& df) {
    std::vector<double> stack;
    const auto pop=[&] { require(!stack.empty(),"stack underflow");
        double v=stack.back(); stack.pop_back(); return v; };
    for (const auto& op:program.instructions) {
        double v=0;
        switch(op.opcode) {
        case NM_EXPR_CONSTANT: v=op.immediate.x; break;
        case NM_EXPR_PARAMETER: v=material.parameters.at(op.index).defaultValue; break;
        case NM_EXPR_F: v=f.at(op.index); break;
        case NM_EXPR_DF: v=df.at(op.index); break;
        case NM_EXPR_RATE: v=df.at(op.index); break;
        case NM_EXPR_ADD: {double b=pop();v=pop()+b;break;}
        case NM_EXPR_SUBTRACT: {double b=pop();v=pop()-b;break;}
        case NM_EXPR_MULTIPLY: {double b=pop();v=pop()*b;break;}
        case NM_EXPR_DIVIDE: {double b=pop();v=pop()/b;break;}
        case NM_EXPR_NEGATE: v=-pop();break;
        case NM_EXPR_LOG: v=std::log(pop());break;
        case NM_EXPR_EXP: v=std::exp(pop());break;
        case NM_EXPR_SQRT: v=std::sqrt(pop());break;
        case NM_EXPR_POW_INTEGER:v=std::pow(pop(),op.integer);break;
        case NM_EXPR_FIBER_EXP_LINEAR: {
            double m=pop(),c5=pop(),c4=pop(),c3=pop(),x=pop(); bool valid=true;
            v=numi_matter_fiber::evaluate<double>(x,c3,c4,c5,m,op.integer,valid);
            require(valid,"fibre domain rejected"); break;
        }
        default: throw std::runtime_error("unhandled material opcode");
        }
        require(std::isfinite(v),"nonfinite scalar");stack.push_back(v);
    }
    require(stack.size()==1,"incomplete scalar");return stack[0];
}
void set(numi::matter::MaterialProgram& m,const char* name,double value) {
    auto it=std::find_if(m.parameters.begin(),m.parameters.end(),
        [&](const auto& p){return p.name==name;});
    require(it!=m.parameters.end(),std::string("missing parameter ")+name);
    it->defaultValue=value;
}

void gpuCheck(id<MTLDevice> device,id<MTLComputePipelineState> pipeline,
              const std::vector<numi::matter::ScalarBytecode>& bytecode,
              const numi::matter::MaterialProgram& material,
              const std::vector<std::array<float,18>>& rows,bool tensorNorm=false) {
    std::vector<NMScalarProgramGPU> programs;
    std::vector<NMExpressionInstructionGPU> instructions;
    std::vector<float> parameters;
    for(const auto& p:material.parameters) parameters.push_back(p.defaultValue);
    auto quantized=material;
    for(std::size_t i=0;i<parameters.size();++i) quantized.parameters[i].defaultValue=parameters[i];
    for(const auto& code:bytecode) {
        programs.push_back({static_cast<nm_u32>(instructions.size()),
            static_cast<nm_u32>(code.instructions.size()),code.maximumStack,0});
        instructions.insert(instructions.end(),code.instructions.begin(),code.instructions.end());
    }
    const auto buffer=[&](const void* p,std::size_t n){
        return [device newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> pb=buffer(programs.data(),programs.size()*sizeof(programs[0]));
    id<MTLBuffer> ib=buffer(instructions.data(),instructions.size()*sizeof(instructions[0]));
    id<MTLBuffer> ab=buffer(parameters.data(),parameters.size()*sizeof(float));
    id<MTLBuffer> rb=buffer(rows.data(),rows.size()*sizeof(rows[0]));
    const uint32_t count=static_cast<uint32_t>(programs.size());
    const std::size_t total=rows.size()*count;
    id<MTLBuffer> output=[device newBufferWithLength:total*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    require(pb&&ib&&ab&&rb&&output&&encoder,"GPU check allocation failed");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:pb offset:0 atIndex:0];[encoder setBuffer:ib offset:0 atIndex:1];
    [encoder setBuffer:ab offset:0 atIndex:2];[encoder setBuffer:rb offset:0 atIndex:3];
    [encoder setBuffer:output offset:0 atIndex:4];
    [encoder setBytes:&count length:sizeof(count) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(total,1,1)
        threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"GPU material command failed");
    const float* actual=static_cast<const float*>(output.contents);
    double worst=0,worstTensor=0;
    for(std::size_t row=0;row<rows.size();++row) {
        Matrix f{},df{}; for(unsigned i=0;i<9;++i){f[i]=rows[row][i];df[i]=rows[row][9+i];}
        double squaredError=0,squaredExpected=0;
        for(unsigned p=0;p<count;++p){
            double expected=evaluate(bytecode[p],quantized,f,df);
            double error=std::abs(actual[row*count+p]-expected)/std::max(1.0e5,std::abs(expected));
            require(std::isfinite(actual[row*count+p])&&(tensorNorm||error<3e-4),
                "Metal scalar differs from FP64 row="+std::to_string(row)+" program="+std::to_string(p)+" relative="+std::to_string(error));
            worst=std::max(worst,error);
            squaredError+=std::pow(actual[row*count+p]-expected,2);
            squaredExpected+=expected*expected;
            // Stress and directional tangent are separate 3x3 tensors.
            // A component can cancel to zero while the tensor is O(bulk).
            // Keep the component metric visible, and assess the new tensor
            // cases with Frobenius error at 1e-5 (not a component-relative
            // quotient that is singular at zero). Original scalar gates stay.
            if(tensorNorm&&(p+1)%9==0){
                const double normError=std::sqrt(squaredError)/std::max(1e5,std::sqrt(squaredExpected));
                require(normError<1e-5,"Metal tensor differs from FP64: "+std::to_string(normError));
                worstTensor=std::max(worstTensor,normError);
                squaredError=0;squaredExpected=0;
            }
        }
    }
    std::cout<<" metal_scalar_checks="<<total<<" maximum_component_scaled_error="<<worst;
    if(tensorNorm)std::cout<<" maximum_tensor_relative_error="<<worstTensor;
}
} // namespace

int main(int argc,char** argv) {
 @autoreleasepool {try {
    const bool cpuOnly=argc==2&&std::string(argv[1])=="--cpu-only";
    require(argc==1||cpuOnly,"usage: fiber check [--cpu-only]");
    id<MTLDevice> device=nil;id<MTLComputePipelineState> pipeline=nil;
    if(!cpuOnly){
        device=MTLCreateSystemDefaultDevice();NSError* error=nil;
        require(device!=nil,"Metal device unavailable");
        std::cout<<"device="<<[device.name UTF8String]<<"\n";
        id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_FIBER_CHECK_METALLIB] error:&error];
        require(library!=nil,"material check metallib unavailable");
        pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"fiber_exp_linear_check"] error:&error];
        require(pipeline!=nil,"material check pipeline unavailable");
    }
    auto parsed=numi::matter::parseMatterFile(NUMI_FIBER_SOURCE_MATERIAL);
    require(parsed.succeeded(),"source fibre material parse failed");
    const std::string invalidShape=R"(
      material bad { parameter c : Pa = 1 in [0,10]; model generic;
      energy = fiber_exp_linear(F(0,0),c*F(1,1),10,c,1.05);
      valid = J(); supports fem; }
    )";
    require(!numi::matter::parseMatter(invalidShape).succeeded(),
            "deformation-dependent fibre shape silently admitted");
    const std::string invalidUnits=R"(
      material bad { parameter c : Pa = 1 in [0,10]; model generic;
      energy = fiber_exp_linear(F(0,0),1,10,c,1.05);
      valid = J(); supports fem; }
    )";
    require(!numi::matter::parseMatter(invalidUnits).succeeded(),
            "dimensionally invalid fibre primitive admitted");
    // Inspect the whole admitted toe envelope, including maximum exponent.
    for(double limit:{1.001,1.025,1.063,1.25}) for(double exponent:{.001,1.,8.,16.}) {
        Tissue envelope{"certificate",1.,exponent/(limit-1),1.,limit};
        for(double fraction:{.001,.25,.5,1.}) {
            const double x=1+fraction*(limit-1);bool valid=true;
            const double e=numi_matter_fiber::evaluate<double>(x,1.,envelope.c4,1.,limit,0,valid);
            require(valid&&std::abs(e-sourceEnergy(envelope,x))<2e-10*std::max(1.,std::abs(e)),
                    "toe quadrature envelope mismatch");
        }
    }
    for(const auto& t:tissues){
        for(double x:{0.9,1.0,1.0+1e-7,1.01,t.limit-1e-6,t.limit,t.limit+1e-6,1.15}){
            bool valid=true;
            const double e=numi_matter_fiber::evaluate<double>(x,t.c3,t.c4,t.c5,t.limit,0,valid);
            const double p=numi_matter_fiber::evaluate<double>(x,t.c3,t.c4,t.c5,t.limit,1,valid);
            const double h=numi_matter_fiber::evaluate<double>(x,t.c3,t.c4,t.c5,t.limit,2,valid);
            require(valid&&std::abs(p-sourceStress(t,x))<1e-7*std::max(1.0,std::abs(p)),"source stress mismatch");
            require(std::abs(e-sourceEnergy(t,x))<1e-8*std::max(1.0,std::abs(e)),"source energy quadrature mismatch");
            if(x!=1.0&&x!=t.limit){
                const double step=1e-8;
                const double fd=(sourceStress(t,x+step)-sourceStress(t,x-step))/(2*step);
                require(std::abs(h-fd)<3e-7*std::max(1.0,std::abs(h)),"source tangent mismatch");
            }
        }
        auto material=parsed.material;
        set(material,"c3",t.c3);set(material,"c4",t.c4);set(material,"c5",t.c5);set(material,"lambda_max",t.limit);
        set(material,"initial_stretch",t.stretch);
        set(material,"c1",t.c1);set(material,"bulk",t.bulk);set(material,"fiber_scale",t.fiberScale);
        auto compiled=numi::matter::detail::compileConstitutive(material,NM_EXPRESSION_STACK_CAPACITY);
        require(compiled.succeeded(),"source fibre symbolic compile failed");
        // Explicit primitive programs also exercise order-zero energy on
        // Metal; production constitutive stress uses orders one and two.
        std::vector<numi::matter::ScalarBytecode> primitiveCodes;
        for(int order=0;order<3;++order) {
            numi::matter::ScalarBytecode code;code.maximumStack=5;
            NMExpressionInstructionGPU f{};f.opcode=NM_EXPR_F;f.index=0;code.instructions.push_back(f);
            for(const char* name:{"c3","c4","c5","lambda_max"}) {
                NMExpressionInstructionGPU parameter{};parameter.opcode=NM_EXPR_PARAMETER;
                auto it=std::find_if(material.parameters.begin(),material.parameters.end(),
                    [&](const auto& value){return value.name==name;});
                parameter.index=static_cast<nm_u32>(it-material.parameters.begin());
                code.instructions.push_back(parameter);
            }
            NMExpressionInstructionGPU op{};op.opcode=NM_EXPR_FIBER_EXP_LINEAR;op.integer=order;
            code.instructions.push_back(op);primitiveCodes.push_back(code);
        }
        if(!cpuOnly) {
            std::vector<std::array<float,18>> primitiveRows;
            const float seam=static_cast<float>(t.limit);
            for(float x:{.9f,1.f,std::nextafter(1.f,2.f),1.01f,
                         std::nextafter(seam,1.f),seam,std::nextafter(seam,2.f),1.15f}) {
                std::array<float,18> row{};row[0]=x;primitiveRows.push_back(row);
            }
            std::cout<<"primitive="<<t.name;gpuCheck(device,pipeline,primitiveCodes,material,primitiveRows);std::cout<<"\n";
        }
        std::vector<numi::matter::ScalarBytecode> codes;
        codes.insert(codes.end(),compiled.program.stress.begin(),compiled.program.stress.end());
        codes.insert(codes.end(),compiled.program.tangentVector.begin(),compiled.program.tangentVector.end());
        std::vector<std::array<float,18>> rows;
        for(double axial:{0.97,1.01,1.08}) for(unsigned column=0;column<9;++column){
            Matrix f=identity,df{}; f[0]=axial;f[1]=0.003;f[5]=-0.002;df[column]=1;
            Matrix plus=f,minus=f; plus[column]+=1e-6;minus[column]-=1e-6;
            for(unsigned row=0;row<9;++row){
                const double h=evaluate(compiled.program.tangentVector[row],material,f,df);
                const double fd=(evaluate(compiled.program.stress[row],material,plus,{})-
                    evaluate(compiled.program.stress[row],material,minus,{}))/2e-6;
                require(std::abs(h-fd)<3e-6*std::max(1e6,std::abs(h)),"full source material stress/tangent mismatch");
            }
            std::array<float,18> values{};for(unsigned i=0;i<9;++i){values[i]=f[i];values[9+i]=df[i];}rows.push_back(values);
        }
        std::cout<<"tissue="<<t.name<<" source_fp64=passed";
        if(!cpuOnly)gpuCheck(device,pipeline,codes,material,rows);
        std::cout<<"\n";
        // Oblique fibres and shear expose missing transverse prestrain,
        // mistaken matrix prestretch, volumetric pressure, and transpose bugs.
        for(const std::array<double,3>& axis:std::array<std::array<double,3>,2>{{{1,0,0},{.36,-.48,.8}}}){
            set(material,"fiber_x",axis[0]);set(material,"fiber_y",axis[1]);set(material,"fiber_z",axis[2]);
            std::vector<std::array<float,18>> tensorRows;
            for(double scale:{.91,1.0,1.09}) for(unsigned column=0;column<9;++column){
                Matrix f{scale,.032,-.011,-.024,1.015,.027,.015,-.01,.995},df{};
                df[column]=1;
                Matrix plus=f,minus=f;plus[column]+=1e-6;minus[column]-=1e-6;
                double expectedEnergy=0;
                const Matrix expected=sourcePiola(t,f,axis,&expectedEnergy);
                const double energy=evaluate(compiled.program.energy,material,f,{});
                require(std::abs(energy-expectedEnergy)<3e-6*std::max(1e4,std::abs(expectedEnergy)),
                        std::string(t.name)+" full stored energy differs from tensor source oracle");
                const Matrix upper=sourcePiola(t,plus,axis),lower=sourcePiola(t,minus,axis);
                for(unsigned row=0;row<9;++row){
                    const double p=evaluate(compiled.program.stress[row],material,f,{});
                    const double h=evaluate(compiled.program.tangentVector[row],material,f,df);
                    const double tangent=(upper[row]-lower[row])/2e-6;
                    require(std::abs(p-expected[row])<3e-6*std::max(1e6,std::abs(expected[row])),
                            std::string(t.name)+" tensor source stress mismatch");
                    require(std::abs(h-tangent)<1e-5*std::max(1e6,std::abs(tangent)),
                            std::string(t.name)+" tensor source tangent mismatch");
                    require(evaluate(compiled.program.viscousStress[row],material,f,df)==0,
                            "source baseline introduces numerical dissipation");
                }
                std::array<float,18> values{};for(unsigned i=0;i<9;++i){values[i]=f[i];values[9+i]=df[i];}
                tensorRows.push_back(values);
            }
            std::cout<<"tensor="<<t.name<<" independent_cauchy_oracle=passed";
            if(!cpuOnly)gpuCheck(device,pipeline,codes,material,tensorRows,true);
            std::cout<<"\n";
            if(!cpuOnly){
                std::cout<<"stored_energy="<<t.name;
                gpuCheck(device,pipeline,{compiled.program.energy},material,tensorRows);
                std::cout<<"\n";
            }
        }
    }
    bool valid=true;
    (void)numi_matter_fiber::evaluate<double>(1.01,1.,300.,1.,1.25,0,valid);
    require(!valid,"out-of-certificate exponent accepted");
    std::cout<<"fiber_exp_linear=passed continuum_equilibrium=unqualified calibration=not_performed\n";
    return 0;
 }catch(const std::exception& e){std::cerr<<"fiber_exp_linear=failed "<<e.what()<<"\n";return 1;}}
}
