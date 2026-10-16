#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>

#include "numi/matter/matter.hpp"
#include "numi/matter/detail.hpp"
#include "numi/matter/shared.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace numi::matter;
using Matrix = std::array<double,9>;
using State = std::array<double,15>;
constexpr double kRelativeGate=1.0e-3;
constexpr char kMediumSHA[]="76f2bef552822708c36b5acc9f92bd44e7bb92c578e5e46adb64a1a2e82e47f9";
constexpr char kKernelName[]="cardboard_bilinear15_captured_point";

void need(bool value,const std::string& message) {
    if(!value) throw std::runtime_error(message);
}
std::string sha256(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);
    need(input.good(),"cannot read input for SHA256: "+path.string());
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)),{});
    need(bytes.size()<=static_cast<std::size_t>(UINT32_MAX),"SHA256 input too large");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    need(CC_SHA256(bytes.data(),static_cast<CC_LONG>(bytes.size()),digest)!=nullptr,
         "CommonCrypto SHA256 failed");
    static constexpr char hex[]="0123456789abcdef";
    std::string out; out.reserve(64u);
    for(unsigned char b:digest){out.push_back(hex[b>>4]);out.push_back(hex[b&0xf]);}
    return out;
}
id readJSON(const std::filesystem::path& path) {
    NSData* data=[NSData dataWithContentsOfFile:
        [NSString stringWithUTF8String:path.string().c_str()]];
    need(data!=nil,"cannot read captured-point input JSON");
    NSError* error=nil;
    id value=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    need(value!=nil,"cannot parse captured-point input JSON");
    return value;
}
NSDictionary* dict(id value,const std::string& label) {
    NSDictionary* result=[value isKindOfClass:[NSDictionary class]]?(NSDictionary*)value:nil;
    need(result!=nil,label+" must be a JSON object");
    return result;
}
id field(NSDictionary* object,const char* key) {
    id value=[object objectForKey:[NSString stringWithUTF8String:key]];
    need(value!=nil,std::string("missing JSON field ")+key);
    return value;
}
std::string stringValue(id value,const std::string& label) {
    need([value isKindOfClass:[NSString class]],label+" must be a string");
    return [(NSString*)value UTF8String];
}
std::string jsonString(id value,const std::string& label) {
    NSError* error=nil;
    NSData* data=[NSJSONSerialization dataWithJSONObject:value options:0 error:&error];
    need(data!=nil,label+" cannot be serialized as JSON");
    NSString* text=[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    need(text!=nil,label+" is not UTF-8 JSON");
    return [text UTF8String];
}
double numberValue(id value,const std::string& label) {
    need([value isKindOfClass:[NSNumber class]],label+" must be numeric");
    const double result=[(NSNumber*)value doubleValue];
    need(std::isfinite(result),label+" is nonfinite");
    return result;
}
std::vector<double> vectorValue(id value,const std::size_t expected,const std::string& label) {
    need([value isKindOfClass:[NSArray class]],label+" must be an array");
    NSArray* array=(NSArray*)value;
    need(array.count==expected,label+" has the wrong length");
    std::vector<double> result; result.reserve(expected);
    for(id entry in array) result.push_back(numberValue(entry,label));
    return result;
}
template<std::size_t N>
std::array<double,N> arrayValue(id value,const std::string& label) {
    const auto v=vectorValue(value,N,label);
    std::array<double,N> out{};
    std::copy(v.begin(),v.end(),out.begin());
    return out;
}
std::string escape(const std::string& value) {
    std::string out;
    for(char c:value){
        if(c=='\\'||c=='"'){out.push_back('\\');out.push_back(c);}
        else if(c=='\n')out+="\\n";
        else if(c=='\r')out+="\\r";
        else if(static_cast<unsigned char>(c)<0x20u){static constexpr char hex[]="0123456789abcdef";out+="\\u00";out.push_back(hex[(static_cast<unsigned char>(c)>>4)&0xf]);out.push_back(hex[static_cast<unsigned char>(c)&0xf]);}
        else out.push_back(c);
    }
    return out;
}

struct Case {
    std::string id, captureLabel, referenceSource, branchIndicator;
    std::uint32_t tet=0u;
    std::vector<double> parameters;
    State accepted{},capturedState{},referenceState{};
    Matrix fHigh{},fLow{},direction{},fDot{},rateDirection{};
    Matrix capturedP{},capturedTangent{},referenceP{},referenceTangent{};
    double timestep=0.0,temperature=0.0,multiplierIncrement=0.0;
    ::id branchDetails=nil;
    ::id captureProjection=nil;
};
Case readCase(NSDictionary* value) {
    Case c;
    c.id=stringValue(field(value,"id"),"id");
    c.captureLabel=stringValue(field(value,"capture_label"),"capture_label");
    c.referenceSource=stringValue(field(value,"reference_source"),"reference_source");
    c.branchIndicator=stringValue(field(value,"branch_indicator"),"branch_indicator");
    c.tet=static_cast<std::uint32_t>(numberValue(field(value,"tet_id"),"tet_id"));
    c.parameters=vectorValue(field(value,"parameters"),28u,"parameters");
    c.accepted=arrayValue<15>(field(value,"accepted_state"),"accepted_state");
    c.capturedState=arrayValue<15>(field(value,"captured_default_state"),"captured_default_state");
    c.referenceState=arrayValue<15>(field(value,"reference_state"),"reference_state");
    c.fHigh=arrayValue<9>(field(value,"F_high"),"F_high");
    c.fLow=arrayValue<9>(field(value,"F_low"),"F_low");
    c.direction=arrayValue<9>(field(value,"deformation_direction"),"deformation_direction");
    c.fDot=arrayValue<9>(field(value,"Fdot"),"Fdot");
    c.rateDirection=arrayValue<9>(field(value,"rate_direction"),"rate_direction");
    c.capturedP=arrayValue<9>(field(value,"captured_default_stress_Pa"),"captured_default_stress_Pa");
    c.capturedTangent=arrayValue<9>(field(value,"captured_default_tangent_Pa_per_s"),"captured_default_tangent_Pa_per_s");
    c.referenceP=arrayValue<9>(field(value,"reference_stress_Pa"),"reference_stress_Pa");
    c.referenceTangent=arrayValue<9>(field(value,"reference_tangent_Pa_per_s"),"reference_tangent_Pa_per_s");
    c.timestep=numberValue(field(value,"dt_seconds"),"dt_seconds");
    c.temperature=numberValue(field(value,"temperature_K"),"temperature_K");
    c.multiplierIncrement=numberValue(field(value,"multiplier_increment_from_reference"),"multiplier_increment_from_reference");
    c.branchDetails=field(value,"branch_details_from_reference");
    c.captureProjection=field(value,"captured_projection_report");
    need(c.timestep>0.0 && c.temperature>0.0,"invalid captured time/temperature");
    return c;
}

CompiledWorld compileMedium(const std::filesystem::path& path,const bool accurate) {
    auto parsed=parseMatterFile(path.string());
    need(parsed.succeeded(),"source material parse failed");
    MaterialProgram material=parsed.material;
    need(material.name=="hajali2009_medium_hill_bilinear_fit" &&
         material.internalState.size()==15u && material.parameters.size()==28u,
         "material is not the bound generic 15-state medium program");
    material.paperCompensatedArithmetic=true;
    material.genericCompensatedImplicitProjection=true;
    material.genericBilinear15AccurateProjection=accurate;
    WorldSource source;
    source.materials={material}; source.gravity={0.0,0.0,0.0};
    source.environmentCount=1u; source.frameTimestep=1.0e-4;
    ObjectSource object;
    object.name="captured_bilinear15_native_material_point";
    object.representation=Representation::fem;
    object.deformableContact=false; object.deformableSelfContact=false; object.mixedFEM=false;
    object.femNodes={{0.0,0.0,0.0},{0.01,0.0,0.0},{0.0,0.01,0.0},{0.0,0.0,0.01}};
    object.tetrahedra={{0u,1u,2u,3u}}; object.characteristicLength=0.01;
    source.objects={object};
    CompileOptions options;
    options.maximumRateExponent=0u; options.emitSpecializedMetal=false;
    options.localMaterialNewtonIterations=32u;
    auto result=compileWorld(source,options);
    need(result.succeeded(),"captured point material world failed compilation");
    need(result.world.materials.size()==1u && result.world.constitutive.size()==1u,
         "compiled captured point world has wrong material count");
    const auto& gpu=result.world.materials[0];
    const auto expectedFlags=NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC|
        NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION;
    need((gpu.flags&expectedFlags)==expectedFlags,
         "compiled world omitted production compensated source15 flags");
    need(((gpu.flags&NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION)!=0u)==accurate,
         "compiled accurate-projection flag does not match the requested policy");
    need(gpu.localNewtonIterations==(accurate?16u:32u),
         "compiled source15 local budget differs from explicit 32/default or opt-in 16 contract");
    return result.world;
}
void verifyCapturedParameters(const CompiledWorld& world,const Case& c) {
    need(world.parameters.size()==c.parameters.size(),"compiled parameter count differs from capture");
    for(std::size_t i=0;i<world.parameters.size();++i) {
        const float compiled=world.parameters[i].valueAndBounds.x;
        const float captured=static_cast<float>(c.parameters[i]);
        need(std::bit_cast<std::uint32_t>(compiled)==std::bit_cast<std::uint32_t>(captured),
             "compiled medium parameter bits differ from sealed capture at index "+std::to_string(i));
    }
}

template<class T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device,const std::vector<T>& values) {
    T zero{};
    const void* bytes=values.empty()?static_cast<const void*>(&zero):static_cast<const void*>(values.data());
    return [device newBufferWithBytes:bytes length:sizeof(T)*std::max<std::size_t>(1u,values.size())
        options:MTLResourceStorageModeShared];
}
struct Response {
    bool valid=false,projectionValid=false;
    Matrix p{},tangent{};
    State state{};
    std::array<std::uint32_t,9> pBits{},tangentBits{};
    std::array<std::uint32_t,15> stateHighBits{},stateLowBits{};
    std::uint32_t validBits=0u,projectionValidBits=0u;
    std::uint32_t residualBits=0u,stateNormBits=0u,iterationsBits=0u,flagsBits=0u,failureBits=0u;
    double residual=0.0,stateNorm=0.0,iterationsRaw=0.0,flagsRaw=0.0,failureRaw=0.0;
    std::uint32_t iterations=0u,flags=0u,failure=0u;
};
struct MetalInstrument {
    id<MTLDevice> device=nil; id<MTLCommandQueue> queue=nil;
    id<MTLComputePipelineState> pipeline=nil;
    MetalInstrument() {
        device=MTLCreateSystemDefaultDevice(); need(device!=nil,"Metal device unavailable");
        const std::string name=[[device name] UTF8String];
        need(name.find("Apple")!=std::string::npos && name.find("Paravirtual")==std::string::npos,
             "physical Apple GPU required; no native projection was run");
        queue=[device newCommandQueue]; need(queue!=nil,"Metal command queue unavailable");
        NSError* error=nil;
        id<MTLLibrary> library=[device newLibraryWithURL:
            [NSURL fileURLWithPath:@NUMI_CARDBOARD_PAPER_METALLIB] error:&error];
        need(library!=nil,"production paper metallib missing");
        id<MTLFunction> function=[library newFunctionWithName:@"cardboard_bilinear15_captured_point"];
        need(function!=nil,"captured-point production evaluator kernel missing");
        pipeline=[device newComputePipelineStateWithFunction:function error:&error];
        need(pipeline!=nil,"could not create captured-point pipeline");
        std::cout<<"{\"record\":\"instrument\",\"device\":\""<<escape(name)
                 <<"\",\"kernel\":\""<<kKernelName<<"\",\"physical_steps\":0}\n";
    }
    Response evaluate(const CompiledWorld& world,const Case& c,const bool accurate) {
        std::vector<float> inputs; inputs.reserve(48u);
        for(double x:c.fHigh) inputs.push_back(static_cast<float>(x));
        for(double x:c.direction) inputs.push_back(static_cast<float>(x));
        for(double x:c.fLow) inputs.push_back(static_cast<float>(x));
        for(double x:c.fDot) inputs.push_back(static_cast<float>(x));
        for(double x:c.rateDirection) inputs.push_back(static_cast<float>(x));
        inputs.push_back(static_cast<float>(c.timestep));
        inputs.push_back(static_cast<float>(c.temperature));
        inputs.push_back(accurate?1.0f:0.0f);
        std::vector<float> accepted; accepted.reserve(15u);
        for(double x:c.accepted) accepted.push_back(static_cast<float>(x));
        verifyCapturedParameters(world,c);
        std::vector<float> parameters; parameters.reserve(world.parameters.size());
        for(const auto& parameter:world.parameters) parameters.push_back(parameter.valueAndBounds.x);
        std::vector<float> outputZero(25u+2u*NM_MAX_MATERIAL_STATE,0.0f);
        std::array<id<MTLBuffer>,7> buffers{
            makeBuffer(device,std::vector<NMMaterialGPU>{world.materials.at(0)}),
            makeBuffer(device,world.scalarPrograms),makeBuffer(device,world.instructions),
            makeBuffer(device,parameters),makeBuffer(device,accepted),
            makeBuffer(device,inputs),makeBuffer(device,outputZero)};
        need(std::all_of(buffers.begin(),buffers.end(),[](id<MTLBuffer> b){return b!=nil;}),
             "failed allocating captured point buffers");
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        need(encoder!=nil,"captured point command encoder unavailable");
        [encoder setComputePipelineState:pipeline];
        for(NSUInteger i=0;i<buffers.size();++i)[encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        need(command.status==MTLCommandBufferStatusCompleted,
             command.error?std::string([[command.error description] UTF8String]):"Metal command failed");
        const auto* out=static_cast<const float*>(buffers[6].contents);
        Response response;
        response.validBits=std::bit_cast<std::uint32_t>(out[0]);
        response.valid=std::isfinite(out[0])&&out[0]==1.0f;
        for(unsigned i=0;i<9u;++i) {
            response.pBits[i]=std::bit_cast<std::uint32_t>(out[1u+i]);
            response.tangentBits[i]=std::bit_cast<std::uint32_t>(out[10u+i]);
            response.p[i]=double(out[1u+i]);
            response.tangent[i]=double(out[10u+i]);
        }
        for(unsigned i=0;i<15u;++i) {
            const float high=out[19u+i],low=out[25u+NM_MAX_MATERIAL_STATE+i];
            response.stateHighBits[i]=std::bit_cast<std::uint32_t>(high);
            response.stateLowBits[i]=std::bit_cast<std::uint32_t>(low);
            response.state[i]=double(high)+double(low);
        }
        const float projectionValid=out[19u+NM_MAX_MATERIAL_STATE];
        response.projectionValidBits=std::bit_cast<std::uint32_t>(projectionValid);
        response.projectionValid=std::isfinite(projectionValid)&&projectionValid==1.0f;
        const float residual=out[20u+NM_MAX_MATERIAL_STATE];
        const float stateNorm=out[21u+NM_MAX_MATERIAL_STATE];
        const float iterations=out[22u+NM_MAX_MATERIAL_STATE];
        const float flags=out[23u+NM_MAX_MATERIAL_STATE];
        const float failure=out[24u+NM_MAX_MATERIAL_STATE];
        response.residualBits=std::bit_cast<std::uint32_t>(residual);
        response.stateNormBits=std::bit_cast<std::uint32_t>(stateNorm);
        response.iterationsBits=std::bit_cast<std::uint32_t>(iterations);
        response.flagsBits=std::bit_cast<std::uint32_t>(flags);
        response.failureBits=std::bit_cast<std::uint32_t>(failure);
        response.residual=double(residual); response.stateNorm=double(stateNorm);
        response.iterationsRaw=double(iterations); response.flagsRaw=double(flags);
        response.failureRaw=double(failure);
        if(std::isfinite(iterations)&&iterations>=0.0f&&iterations<=float(UINT32_MAX))
            response.iterations=static_cast<std::uint32_t>(iterations);
        if(std::isfinite(flags)&&flags>=0.0f&&flags<=float(UINT32_MAX))
            response.flags=static_cast<std::uint32_t>(flags);
        if(std::isfinite(failure)&&failure>=0.0f&&failure<=float(UINT32_MAX))
            response.failure=static_cast<std::uint32_t>(failure);
        return response;
    }

};
struct Metric { bool finite=false; double norm=0.0,referenceNorm=0.0,relative=0.0,maxAbs=0.0; };
template<std::size_t N>
Metric metric(const std::array<double,N>& a,const std::array<double,N>& b) {
    double d2=0.0,b2=0.0,maxAbs=0.0;
    for(std::size_t i=0;i<N;++i) {
        if(!std::isfinite(a[i])||!std::isfinite(b[i]))
            return {false,NAN,NAN,NAN,NAN};
        const double d=a[i]-b[i]; d2+=d*d; b2+=b[i]*b[i]; maxAbs=std::max(maxAbs,std::abs(d));
    }
    const double dn=std::sqrt(d2),bn=std::sqrt(b2);
    if(!std::isfinite(dn)||!std::isfinite(bn)) return {false,NAN,NAN,NAN,NAN};
    return {true,dn,bn,bn>0.0?dn/bn:NAN,maxAbs};
}
template<std::size_t N>
double repeatDifference(const std::array<double,N>& a,const std::array<double,N>& b) {
    double result=0.0;
    for(std::size_t i=0;i<N;++i) {
        if(!std::isfinite(a[i])||!std::isfinite(b[i])) return NAN;
        result=std::max(result,std::abs(a[i]-b[i]));
    }
    return result;
}
void jsonNumber(std::ostream& out,const double value) {
    if(std::isfinite(value)) out<<value; else out<<"null";
}
template<std::size_t N>
void jsonArray(std::ostream& out,const std::array<double,N>& values) {
    out<<'[';
    for(std::size_t i=0;i<N;++i){if(i)out<<',';jsonNumber(out,values[i]);}
    out<<']';
}
template<std::size_t N>
void jsonBits(std::ostream& out,const std::array<std::uint32_t,N>& values) {
    out<<'[';
    for(std::size_t i=0;i<N;++i) {
        if(i)out<<',';
        std::ostringstream word; word<<"0x"<<std::hex<<std::setw(8)<<std::setfill('0')<<values[i];
        out<<'"'<<word.str()<<'"';
    }
    out<<']';
}
void jsonBitWord(std::ostream& out,const std::uint32_t bits,const bool present) {
    if(!present){out<<"null";return;}
    std::ostringstream word; word<<"0x"<<std::hex<<std::setw(8)<<std::setfill('0')<<bits;
    out<<'"'<<word.str()<<'"';
}
template<std::size_t N>
void jsonMaybeArray(std::ostream& out,const std::array<double,N>& values,const bool present) {
    if(present)jsonArray(out,values);else out<<"null";
}
template<std::size_t N>
void jsonMaybeBits(std::ostream& out,const std::array<std::uint32_t,N>& values,const bool present) {
    if(present)jsonBits(out,values);else out<<"null";
}
std::string metricStatus(const Metric& m,const bool evaluated=true) {
    if(!evaluated) return "not_evaluated";
    if(!m.finite) return "fail_nonfinite";
    if(m.referenceNorm==0.0) return "inconclusive_zero_reference";
    if(!std::isfinite(m.relative)) return "fail_nonfinite_metric";
    return m.relative<=kRelativeGate?"pass_relative_gate":"fail_relative_gate";
}
struct Attempt { bool evaluated=false; Response response{}; std::string error; };
Attempt safeEvaluate(MetalInstrument& metal,const CompiledWorld& world,const Case& c,const bool accurate) {
    try { return {true,metal.evaluate(world,c,accurate),{}}; }
    catch(const std::exception& error) { return {false,{},error.what()}; }
}
std::string policyStatus(const Attempt& attempt,const Metric& p,const Metric& t,const Metric& state) {
    if(!attempt.evaluated) return "fail_instrument";
    if(!attempt.response.valid||!attempt.response.projectionValid) return "fail_projection";
    const std::array<std::string,3> statuses{metricStatus(p),metricStatus(t),metricStatus(state)};
    for(const auto& status:statuses) if(status.rfind("fail_",0u)==0u) return "fail_comparison";
    for(const auto& status:statuses) if(status.rfind("inconclusive_",0u)==0u) return "inconclusive_zero_reference";
    return "pass";
}
void jsonAttempt(std::ostream& out,const Attempt& attempt,const Metric& p,const Metric& t,
                 const Metric& state,const double repeatP,const double repeatT,const double repeatState,
                 const char* comparisonTarget) {
    const Response& r=attempt.response;
    out<<"{\"status\":\""<<policyStatus(attempt,p,t,state)<<"\",\"comparison_target\":\""<<comparisonTarget<<"\",\"evaluated\":"
       <<(attempt.evaluated?"true":"false")<<",\"error\":";
    if(attempt.error.empty()) out<<"null"; else out<<'"'<<escape(attempt.error)<<'"';
    out<<",\"projection_valid\":";
    if(attempt.evaluated)out<<(r.projectionValid?"true":"false");else out<<"null";
    out<<",\"projection_valid_f32_bits\":";jsonBitWord(out,r.projectionValidBits,attempt.evaluated);
    out<<",\"valid\":";
    if(attempt.evaluated)out<<(r.valid?"true":"false");else out<<"null";
    out<<",\"valid_f32_bits\":";jsonBitWord(out,r.validBits,attempt.evaluated);
    out<<",\"iterations\":"; if(attempt.evaluated)jsonNumber(out,r.iterationsRaw);else out<<"null";
    out<<",\"iterations_f32_bits\":";jsonBitWord(out,r.iterationsBits,attempt.evaluated);
    out<<",\"flags\":"; if(attempt.evaluated)jsonNumber(out,r.flagsRaw);else out<<"null";
    out<<",\"flags_f32_bits\":";jsonBitWord(out,r.flagsBits,attempt.evaluated);
    out<<",\"failure\":"; if(attempt.evaluated)jsonNumber(out,r.failureRaw);else out<<"null";
    out<<",\"failure_f32_bits\":";jsonBitWord(out,r.failureBits,attempt.evaluated);
    out<<",\"projection_residual\":"; if(attempt.evaluated)jsonNumber(out,r.residual);else out<<"null";
    out<<",\"projection_residual_f32_bits\":";jsonBitWord(out,r.residualBits,attempt.evaluated);
    out<<",\"projection_state_norm\":"; if(attempt.evaluated)jsonNumber(out,r.stateNorm);else out<<"null";
    out<<",\"projection_state_norm_f32_bits\":";jsonBitWord(out,r.stateNormBits,attempt.evaluated);
    out<<",\"stress_relative_l2\":";
    jsonNumber(out,p.relative); out<<",\"stress_abs_l2\":"; jsonNumber(out,p.norm);
    out<<",\"stress_target_norm_l2\":"; jsonNumber(out,p.referenceNorm);
    out<<",\"stress_comparison\":\""<<metricStatus(p,attempt.evaluated)<<"\",\"tangent_relative_l2\":";
    jsonNumber(out,t.relative); out<<",\"tangent_abs_l2\":"; jsonNumber(out,t.norm);
    out<<",\"tangent_target_norm_l2\":"; jsonNumber(out,t.referenceNorm);
    out<<",\"tangent_comparison\":\""<<metricStatus(t,attempt.evaluated)<<"\",\"state_relative_l2\":";
    jsonNumber(out,state.relative); out<<",\"state_abs_l2\":"; jsonNumber(out,state.norm);
    out<<",\"state_target_norm_l2\":"; jsonNumber(out,state.referenceNorm);
    out<<",\"state_comparison\":\""<<metricStatus(state,attempt.evaluated)<<"\""
       <<",\"repeat_difference_max_abs\":{\"stress\":";
    jsonNumber(out,repeatP); out<<",\"tangent\":"; jsonNumber(out,repeatT);
    out<<",\"state\":"; jsonNumber(out,repeatState);
    out<<"},\"repeat_difference_interpretation\":\"identical-run variation only; not a representability or accuracy bound\""
       <<",\"stress\":"; jsonMaybeArray(out,r.p,attempt.evaluated);
    out<<",\"stress_f32_bits\":"; jsonMaybeBits(out,r.pBits,attempt.evaluated);
    out<<",\"tangent\":"; jsonMaybeArray(out,r.tangent,attempt.evaluated);
    out<<",\"tangent_f32_bits\":"; jsonMaybeBits(out,r.tangentBits,attempt.evaluated);
    out<<",\"state\":"; jsonMaybeArray(out,r.state,attempt.evaluated);
    out<<",\"state_high_f32_bits\":"; jsonMaybeBits(out,r.stateHighBits,attempt.evaluated);
    out<<",\"state_low_f32_bits\":"; jsonMaybeBits(out,r.stateLowBits,attempt.evaluated);
    out<<'}';
}
std::string runCase(MetalInstrument& metal,const CompiledWorld& defaultWorld,
                    const CompiledWorld& accurateWorld,const Case& c) {
    const Attempt offA=safeEvaluate(metal,defaultWorld,c,false);
    const Attempt offB=safeEvaluate(metal,defaultWorld,c,false);
    const Attempt onA=safeEvaluate(metal,accurateWorld,c,true);
    const Attempt onB=safeEvaluate(metal,accurateWorld,c,true);
    const Metric unavailable{false,NAN,NAN,NAN,NAN};
    const Metric offP=offA.evaluated?metric(offA.response.p,c.capturedP):unavailable;
    const Metric offT=offA.evaluated?metric(offA.response.tangent,c.capturedTangent):unavailable;
    const Metric offS=offA.evaluated?metric(offA.response.state,c.capturedState):unavailable;
    const Metric onP=onA.evaluated?metric(onA.response.p,c.referenceP):unavailable;
    const Metric onT=onA.evaluated?metric(onA.response.tangent,c.referenceTangent):unavailable;
    const Metric onS=onA.evaluated?metric(onA.response.state,c.referenceState):unavailable;
    const double offRepeatP=offA.evaluated&&offB.evaluated?repeatDifference(offA.response.p,offB.response.p):NAN;
    const double offRepeatT=offA.evaluated&&offB.evaluated?repeatDifference(offA.response.tangent,offB.response.tangent):NAN;
    const double offRepeatS=offA.evaluated&&offB.evaluated?repeatDifference(offA.response.state,offB.response.state):NAN;
    const double onRepeatP=onA.evaluated&&onB.evaluated?repeatDifference(onA.response.p,onB.response.p):NAN;
    const double onRepeatT=onA.evaluated&&onB.evaluated?repeatDifference(onA.response.tangent,onB.response.tangent):NAN;
    const double onRepeatS=onA.evaluated&&onB.evaluated?repeatDifference(onA.response.state,onB.response.state):NAN;
    const std::string offStatus=policyStatus(offA,offP,offT,offS);
    const std::string onStatus=policyStatus(onA,onP,onT,onS);
    const bool casePass=offStatus=="pass"&&onStatus=="pass";
    const bool inconclusive=offStatus.rfind("inconclusive_",0u)==0u||onStatus.rfind("inconclusive_",0u)==0u;
    const std::string caseStatus=casePass?"pass":(inconclusive&&offStatus.rfind("fail_",0u)!=0u&&onStatus.rfind("fail_",0u)!=0u
        ?"inconclusive":"fail");
    std::ostringstream row; row<<std::setprecision(17)
      <<"{\"record\":\"case\",\"id\":\""<<escape(c.id)<<"\",\"capture\":\""<<escape(c.captureLabel)
      <<"\",\"tet_id\":"<<c.tet<<",\"branch_indicator\":\""<<escape(c.branchIndicator)
      <<"\",\"reference_source\":\""<<escape(c.referenceSource)<<"\",\"multiplier_increment\":"<<c.multiplierIncrement
      <<",\"reference_branch_details\":"<<jsonString(c.branchDetails,"branch details")
      <<",\"captured_projection_report\":"<<jsonString(c.captureProjection,"projection report")
      <<",\"compiled_local_iterations_default\":"<<defaultWorld.materials[0].localNewtonIterations
      <<",\"compiled_local_iterations_optin\":"<<accurateWorld.materials[0].localNewtonIterations
      <<",\"default\":";
    jsonAttempt(row,offA,offP,offT,offS,offRepeatP,offRepeatT,offRepeatS,"captured_production_point");
    row<<",\"optin\":";
    jsonAttempt(row,onA,onP,onT,onS,onRepeatP,onRepeatT,onRepeatS,"independent_fp64_reference");
    row<<",\"status\":\""<<caseStatus<<"\"}\n";
    std::cout<<row.str();
    return caseStatus;
}

} // namespace

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            const bool cpuAdmission=argc==4&&std::string(argv[1])=="--cpu-admission";
            need(cpuAdmission||argc==3,
                 "usage: captured-point-check [--cpu-admission] INPUTS.json MEDIUM_ASSET.nmatter");
            const std::filesystem::path inputsPath=argv[cpuAdmission?2:1];
            const std::filesystem::path materialPath=argv[cpuAdmission?3:2];
            const std::string sourceSHA=sha256(materialPath);
            need(sourceSHA==kMediumSHA,"medium source material hash differs from captured asset");
            NSDictionary* document=dict(readJSON(inputsPath),"input document");
            need(stringValue(field(document,"schema"),"schema")==
                 "numi.cardboard.bilinear15-native-captured-point-inputs.v1",
                 "unsupported captured-point input schema");
            NSDictionary* source=dict(field(document,"source_material"),"source_material");
            need(stringValue(field(source,"sha256"),"source material sha") == sourceSHA &&
                 stringValue(field(source,"name"),"source material name")==
                    "hajali2009_medium_hill_bilinear_fit",
                 "input document source material binding mismatch");
            NSArray* cases=(NSArray*)field(document,"cases");
            need(cases.count==16u,"expected exactly 8 captured N0 and 8 captured N63 test points");
            const CompiledWorld defaultWorld=compileMedium(materialPath,false);
            const CompiledWorld accurateWorld=compileMedium(materialPath,true);
            std::vector<Case> parsed; parsed.reserve(cases.count);
            for(id value in cases) parsed.push_back(readCase(dict(value,"case")));
            for(const Case& c:parsed) {
                verifyCapturedParameters(defaultWorld,c);
                verifyCapturedParameters(accurateWorld,c);
            }
            if(cpuAdmission) {
                std::cout<<"{\"record\":\"cpu_admission\",\"status\":\"pass\","
                  <<"\"input_sha256\":\""<<sha256(inputsPath)<<"\",\"source_material_sha256\":\""
                  <<sourceSHA<<"\",\"case_count\":"<<parsed.size()
                  <<",\"default_local_iterations\":"<<defaultWorld.materials[0].localNewtonIterations
                  <<",\"optin_local_iterations\":"<<accurateWorld.materials[0].localNewtonIterations
                  <<",\"native_metal_execution\":false}\n";
                return 0;
            }
            MetalInstrument metal;
            std::cout<<std::setprecision(17)
              <<"{\"record\":\"header\",\"schema\":\"numi.cardboard.bilinear15-native-captured-point-result.v2\","
              <<"\"base_snapshot_manifest_sha256\":\""<<NUMI_CARDBOARD_BASE_SNAPSHOT_SHA256<<"\","
              <<"\"source_material_sha256\":\""<<sourceSHA<<"\",\"input_sha256\":\""<<sha256(inputsPath)
              <<"\",\"physical_steps\":0,\"physical_validation\":false,\"relative_gate\":"<<kRelativeGate
              <<",\"repeat_difference_interpretation\":\"identical-run variation only; not a representability or accuracy bound\","
              <<"\"near_zero_policy\":\"inconclusive when the reference norm is exactly zero; no absolute pass is inferred from repeated evaluations\","
              <<"\"default_policy\":\"both captured source flags enabled, accurate local projection flag disabled, explicit material iteration option 32\","
              <<"\"optin_policy\":\"same captured source flags, accurate 15-state policy enabled, compiled local budget 16\"}\n";
            std::size_t passedCount=0u,failedCount=0u,inconclusiveCount=0u;
            for(const Case& c:parsed) {
                try {
                    const std::string status=runCase(metal,defaultWorld,accurateWorld,c);
                    if(status=="pass") ++passedCount;
                    else if(status=="inconclusive") ++inconclusiveCount;
                    else ++failedCount;
                } catch(const std::exception& error) {
                    ++failedCount;
                    std::cout<<"{\"record\":\"case\",\"id\":\""<<escape(c.id)
                             <<"\",\"capture\":\""<<escape(c.captureLabel)
                             <<"\",\"tet_id\":"<<c.tet<<",\"status\":\"fail_case_exception\",\"error\":\""
                             <<escape(error.what())<<"\"}\n";
                }
            }
            const bool allPassed=failedCount==0u&&inconclusiveCount==0u;
            std::cout<<"{\"record\":\"end\",\"cases\":"<<parsed.size()
                     <<",\"passed_count\":"<<passedCount<<",\"failed_count\":"<<failedCount
                     <<",\"inconclusive_count\":"<<inconclusiveCount<<",\"status\":\""
                     <<(allPassed?"pass":(failedCount?"fail":"inconclusive"))
                     <<"\",\"scope\":\"pointwise native projection/stress/tangent only\"}\n";
            return allPassed?0:2;
        } catch(const std::exception& error) {
            std::cerr<<"FAIL captured bilinear-15 native point replay: "<<error.what()<<'\n';
            return 1;
        }
    }
}
