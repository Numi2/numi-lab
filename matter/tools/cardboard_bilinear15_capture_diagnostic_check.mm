#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include "numi/matter/matter.hpp"
#include "numi/matter/detail.hpp"
#include "numi/matter/shared.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
using namespace numi::matter;
constexpr std::uint32_t kInputStride=47u, kOutputStride=65u, kStateCount=15u;
constexpr std::uint32_t kTraceSlots=5992u;
constexpr char kCaptureSchema[]="numi.cardboard.fem-newton-capture.v1";
constexpr char kLinerSHA[]="a2720b90f2004f2eccf180063e7ecc4e0acdda93f700c4fd357914fc24d2255a";
constexpr char kMediumSHA[]="76f2bef552822708c36b5acc9f92bd44e7bb92c578e5e46adb64a1a2e82e47f9";
constexpr char kLinerName[]="hajali2009_liner_hill_bilinear_fit";
constexpr char kMediumName[]="hajali2009_medium_hill_bilinear_fit";
#ifndef NUMI_CARDBOARD_BASE_SNAPSHOT_SHA256
#error "The checker must be bound to an exact frozen source manifest"
#endif
constexpr char kBaseSnapshotSHA[]=NUMI_CARDBOARD_BASE_SNAPSHOT_SHA256;

void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
std::string sha256(const std::filesystem::path& path){
    std::ifstream in(path,std::ios::binary);need(in.good(),"cannot read for SHA256: "+path.string());
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),{});
    need(bytes.size()<=static_cast<std::size_t>(UINT32_MAX),"SHA256 input exceeds CommonCrypto bound");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    need(CC_SHA256(bytes.data(),static_cast<CC_LONG>(bytes.size()),digest)!=nullptr,"SHA256 failed");
    static constexpr char hex[]="0123456789abcdef";std::string out;out.reserve(64u);
    for(unsigned char b:digest){out.push_back(hex[b>>4]);out.push_back(hex[b&15]);}return out;
}
NSDictionary* parseObject(const std::string& line,const std::string& label){
    NSData* data=[NSData dataWithBytes:line.data() length:line.size()];NSError* error=nil;
    id value=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    NSDictionary* result=[value isKindOfClass:[NSDictionary class]]?(NSDictionary*)value:nil;
    need(result!=nil,label+" is not a JSON object");return result;
}
id field(NSDictionary* d,const char* key){id v=[d objectForKey:[NSString stringWithUTF8String:key]];need(v!=nil,std::string("missing field ")+key);return v;}
double number(id v,const std::string& label){
    need([v isKindOfClass:[NSNumber class]],label+" is not numeric");double x=[(NSNumber*)v doubleValue];
    need(std::isfinite(x),label+" is nonfinite");return x;
}
std::string stringValue(id v,const std::string& label){
    need([v isKindOfClass:[NSString class]],label+" is not a string");return [(NSString*)v UTF8String];
}
template<std::size_t N> std::array<float,N> flatArray(id v,const std::string& label){
    need([v isKindOfClass:[NSArray class]],label+" is not an array");NSArray* a=(NSArray*)v;
    std::array<float,N> out{};std::size_t p=0;
    for(id row in a){
        if([row isKindOfClass:[NSArray class]]){
            for(id cell in (NSArray*)row){need(p<N,label+" too long");out[p++]=static_cast<float>(number(cell,label));}
        }else{need(p<N,label+" too long");out[p++]=static_cast<float>(number(row,label));}
    }
    need(p==N,label+" has wrong flattened length");return out;
}
std::vector<float> flatVector(id v,const std::size_t expected,const std::string& label){
    need([v isKindOfClass:[NSArray class]],label+" is not an array");NSArray* a=(NSArray*)v;
    need(a.count==expected,label+" has wrong length");std::vector<float> out;out.reserve(expected);
    for(id x in a)out.push_back(static_cast<float>(number(x,label)));return out;
}
struct Record{
    std::uint32_t env=0,tet=0,material=0;
    std::string materialName;
    std::array<float,kInputStride> input{};
    std::array<float,kStateCount> accepted{};
};
struct Capture{
    NSDictionary* header=nil;
    std::vector<Record> rows;
    std::vector<float> parameterRow;
};
Capture readCapture(const std::filesystem::path& path){
    std::ifstream in(path);need(in.good(),"cannot open capture JSONL");
    Capture result;std::string line;bool first=true;std::set<std::pair<unsigned,unsigned>> unique;
    std::array<std::size_t,2> materialCounts{};
    while(std::getline(in,line)){
        if(line.empty())continue;
        NSDictionary* d=parseObject(line,first?"capture header":"capture row");
        if(first){
            result.header=d;first=false;
            NSArray* mats=(NSArray*)field(d,"materials");
            need(mats.count>=2u,"capture header omits paper material bindings");
            for(std::size_t i=0;i<2u;++i){
                NSDictionary* md=(NSDictionary*)mats[i];
                need(stringValue(field(md,"name"),"header material name")== (i==0u?kLinerName:kMediumName),
                     "capture header paper material order differs");
                need(number(field(md,"index"),"header material index")==double(i)&&
                     number(field(md,"parameter_offset"),"header parameter offset")==double(28u*i),
                     "capture header material parameter binding differs");
            }
            need(stringValue(field(d,"schema"),"capture schema")==kCaptureSchema,"unexpected capture schema");
            need(number(field(d,"root"),"capture root")==0.0&&number(field(d,"newton_iteration"),"capture Newton") == 0.0,
                 "capture is not root0/Newton0");
            need(number(field(d,"environment_count"),"environment count")==2.0&&
                 number(field(d,"tetrahedron_count_per_environment"),"tet count")==1920.0,
                 "capture dimensions differ from pinned 2x1920 source");
            need(stringValue(field(d,"center_residual_source"),"residual source")=="pre_fgmres_private_copy",
                 "capture residual source is not the production pre-FGMRES copy");
            need(stringValue(field(d,"F_local_word_encoding"),"F encoding").find("high_float32_plus_low_float32")!=std::string::npos,
                 "capture does not bind compensated F input words");
            NSArray* envParams=(NSArray*)field(d,"parameters_by_environment");
            need(envParams.count==2u,"capture parameter environment count mismatch");
            auto p0=flatVector(envParams[0],59u,"env0 parameters");
            auto p1=flatVector(envParams[1],59u,"env1 parameters");
            for(std::size_t i=0;i<p0.size();++i)
                need(std::bit_cast<std::uint32_t>(p0[i])==std::bit_cast<std::uint32_t>(p1[i]),
                     "capture environments have differing material parameter words");
            result.parameterRow=std::move(p0);continue;
        }
        if(stringValue(field(d,"record"),"record type")!="tetrahedron")continue;
        const auto mat=static_cast<std::uint32_t>(number(field(d,"material_index"),"material index"));
        if(mat>1u)continue; // Glue rows are explicitly outside this paper projection diagnostic.
        Record row;
        row.env=static_cast<std::uint32_t>(number(field(d,"environment"),"environment"));
        row.tet=static_cast<std::uint32_t>(number(field(d,"tet_id"),"tet id"));
        row.material=mat;row.materialName=stringValue(field(d,"material_name"),"material name");
        need(row.env<2u&&row.tet<1920u,"paper row index outside capture shape");
        need(row.materialName==(mat==0u?kLinerName:kMediumName),"material index/name mismatch in capture");
        need(number(field(d,"material_state_count"),"paper state count")==15.0&&
             number(field(d,"material_parameter_count"),"paper parameter count")==28.0,
             "captured paper row is not the exact 15-state/28-parameter source-fit system");
        need(unique.emplace(row.env,row.tet).second,"duplicate env/tet paper row");
        ++materialCounts[row.material];
        NSDictionary* c=(NSDictionary*)field(d,"center");
        auto F=flatArray<9>(field(c,"F_local"),"F_local");
        auto Flo=flatArray<9>(field(c,"F_local_low"),"F_local_low");
        auto Fdot=flatArray<9>(field(c,"Fdot_local_per_s"),"Fdot_local_per_s");
        auto dF=flatArray<9>(field(c,"F_direction_local_per_s"),"F_direction_local_per_s");
        auto dFdot=flatArray<9>(field(c,"rate_direction_local_per_s2"),"rate_direction_local_per_s2");
        auto state=flatVector(field(c,"accepted_state"),15u,"accepted state");
        std::copy(F.begin(),F.end(),row.input.begin());
        std::copy(Flo.begin(),Flo.end(),row.input.begin()+9);
        std::copy(Fdot.begin(),Fdot.end(),row.input.begin()+18);
        std::copy(dF.begin(),dF.end(),row.input.begin()+27);
        std::copy(dFdot.begin(),dFdot.end(),row.input.begin()+36);
        NSArray* dtTemp=(NSArray*)field(c,"dt_seconds_bulk_modulus_kind_frame_active");
        need(dtTemp.count==4u,"capture dt/temp metadata shape mismatch");
        row.input[45]=static_cast<float>(number(dtTemp[0],"captured dt"));
        row.input[46]=static_cast<float>(number(dtTemp[1],"captured temperature"));
        need(row.input[45]>0.0f&&row.input[46]>0.0f,"nonpositive captured dt or temperature");
        std::copy(state.begin(),state.end(),row.accepted.begin());
        result.rows.push_back(std::move(row));
    }
    need(!first,"capture file is empty");
    need(result.rows.size()==3456u,"expected exactly 3456 paper rows (glue excluded)");
    need(unique.size()==3456u&&materialCounts[0]==2304u&&materialCounts[1]==1152u,
         "paper row identity/material distribution is incomplete");
    need(std::any_of(result.rows.begin(),result.rows.end(),[](const Record& r){return r.env==0u&&r.tet==1528u&&r.material==0u;}),
         "required first-failure liner tet env0/tet1528 is missing");
    return result;
}
CompiledWorld compileSources(const std::filesystem::path& linerPath,const std::filesystem::path& mediumPath,const bool accurate){
    auto l=parseMatterFile(linerPath.string()),m=parseMatterFile(mediumPath.string());
    need(l.succeeded()&&m.succeeded(),"paper source material parse failed");
    need(l.material.name==kLinerName&&m.material.name==kMediumName,"paper material source identity mismatch");
    need(l.material.internalState.size()==15u&&m.material.internalState.size()==15u&&
         l.material.parameters.size()==28u&&m.material.parameters.size()==28u,
         "source assets do not match captured 15-state/28-parameter bindings");
    for(auto* material:{&l.material,&m.material}){
        material->paperCompensatedArithmetic=true;
        material->genericCompensatedImplicitProjection=true;
        material->genericBilinear15AccurateProjection=accurate;
    }
    WorldSource source;source.materials={l.material,m.material};source.gravity={0.0,0.0,0.0};
    source.environmentCount=1u;source.frameTimestep=1.0e-4;
    ObjectSource object;object.name="readonly_bilinear15_capture_diagnostic";
    object.representation=Representation::fem;object.deformableContact=false;
    object.deformableSelfContact=false;object.mixedFEM=false;
    object.femNodes={{0,0,0},{.01,0,0},{0,.01,0},{0,0,.01}};
    object.tetrahedra={{0u,1u,2u,3u}};object.characteristicLength=.01;
    source.objects={object};
    CompileOptions options;options.maximumRateExponent=0u;options.emitSpecializedMetal=false;
    options.localMaterialNewtonIterations=32u;
    auto compiled=compileWorld(source,options);
    need(compiled.succeeded(),"two-material captured world failed CPU compilation");
    need(compiled.world.materials.size()==2u&&compiled.world.parameters.size()==56u,
         "compiled paper world has unexpected material/parameter layout");
    const std::uint32_t required=NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC|
        NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION;
    for(std::size_t i=0;i<2u;++i){
        const auto& gpu=compiled.world.materials[i];
        need(gpu.stateCount==15u&&(gpu.flags&required)==required,
             "compiled paper material omitted state count or actual source flags");
        need(((gpu.flags&NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION)!=0u)==accurate,
             "compiled accurate projection selector differs from requested policy");
        need(gpu.localNewtonIterations==(accurate?16u:32u),
             "compiled local iteration budget differs from actual source policy");
    }
    return compiled.world;
}
void verifyParameters(const CompiledWorld& world,const Capture& capture){
    need(world.parameters.size()==56u,"compiled paper parameter count changed");
    for(std::size_t i=0;i<56u;++i){
        const float compiled=world.parameters[i].valueAndBounds.x;
        need(std::bit_cast<std::uint32_t>(compiled)==std::bit_cast<std::uint32_t>(capture.parameterRow[i]),
             "compiled parameter word differs from capture header at "+std::to_string(i));
    }
}
template<class T> id<MTLBuffer> buffer(id<MTLDevice> d,const std::vector<T>& x){
    T zero{};const void* p=x.empty()?static_cast<const void*>(&zero):static_cast<const void*>(x.data());
    return [d newBufferWithBytes:p length:sizeof(T)*std::max<std::size_t>(1u,x.size()) options:MTLResourceStorageModeShared];
}
std::string escaped(const std::string& s){std::ostringstream o;for(unsigned char c:s){if(c=='\\'||c=='"')o<<'\\'<<char(c);else if(c=='\n')o<<"\\n";else if(c=='\r')o<<"\\r";else if(c<0x20)o<<' ';else o<<char(c);}return o.str();}
void numberJSON(std::ostream& o,double v){if(std::isfinite(v))o<<std::setprecision(9)<<v;else o<<"null";}
void floatArrayJSON(std::ostream& o,const float* v,std::size_t n){o<<'[';for(std::size_t i=0;i<n;++i){if(i)o<<',';numberJSON(o,v[i]);}o<<']';}
void bitsArrayJSON(std::ostream& o,const float* v,std::size_t n){o<<'[';for(std::size_t i=0;i<n;++i){if(i)o<<',';o<<'"'<<"0x"<<std::hex<<std::setw(8)<<std::setfill('0')<<std::bit_cast<std::uint32_t>(v[i])<<std::dec<<std::setfill(' ')<<'"';}o<<']';}
void inputRecordJSON(std::ostream& o,const Record& r){
    o<<"{\"environment\":"<<r.env<<",\"tet_id\":"<<r.tet<<",\"material_index\":"<<r.material
     <<",\"material_name\":\""<<r.materialName<<"\",\"input_f32_bits\":";
    bitsArrayJSON(o,r.input.data(),r.input.size());o<<",\"accepted_state_f32_bits\":";
    bitsArrayJSON(o,r.accepted.data(),r.accepted.size());o<<'}';
}
struct Metal {
    id<MTLDevice> device=nil;id<MTLCommandQueue> queue=nil;id<MTLLibrary> library=nil;
    id<MTLComputePipelineState> batch=nil,trace=nil;
    Metal(){
        device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device unavailable");
        const std::string name=[[device name] UTF8String];
        need(name.find("Apple")!=std::string::npos&&name.find("Paravirtual")==std::string::npos,
             "physical Apple GPU required; no native execution on virtual device");
        queue=[device newCommandQueue];need(queue!=nil,"Metal queue unavailable");NSError* e=nil;
        library=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_CARDBOARD_PAPER_METALLIB] error:&e];
        need(library!=nil,"paper diagnostic metallib unavailable");
        auto make=[&](NSString* n){id<MTLFunction> f=[library newFunctionWithName:n];need(f!=nil,"missing production diagnostic function");id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:f error:&e];need(p!=nil,"pipeline creation failed");return p;};
        batch=make(@"cardboard_bilinear15_capture_batch");trace=make(@"cardboard_bilinear15_capture_trace");
    }
    id<MTLBuffer> compileBuffer(const std::vector<NMMaterialGPU>& x){return buffer(device,x);}
    void dispatch(id<MTLComputePipelineState> pipeline,const std::vector<id<MTLBuffer>>& bs,
                  std::uint32_t count,const std::uint32_t scalarBuffer=UINT32_MAX){
        id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"Metal command buffer unavailable");
        id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];need(enc!=nil,"Metal encoder unavailable");
        [enc setComputePipelineState:pipeline];for(NSUInteger i=0;i<bs.size();++i)[enc setBuffer:bs[i] offset:0 atIndex:i];
        if(scalarBuffer!=UINT32_MAX)[enc setBytes:&count length:sizeof(count) atIndex:scalarBuffer];
        const NSUInteger width=std::max<NSUInteger>(1u,std::min<NSUInteger>(pipeline.threadExecutionWidth,count));
        [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];
        need(cb.status==MTLCommandBufferStatusCompleted,cb.error?std::string([[cb.error description] UTF8String]):"Metal dispatch failed");
    }
};
std::vector<float> materialParameters(const CompiledWorld& w){std::vector<float> p;p.reserve(w.parameters.size());for(const auto& x:w.parameters)p.push_back(x.valueAndBounds.x);return p;}
std::vector<float> acceptedRows(const std::vector<Record>& rows){std::vector<float> x;x.reserve(rows.size()*kStateCount);for(const auto& r:rows)x.insert(x.end(),r.accepted.begin(),r.accepted.end());return x;}
std::vector<float> inputRows(const std::vector<Record>& rows){std::vector<float> x;x.reserve(rows.size()*kInputStride);for(const auto& r:rows)x.insert(x.end(),r.input.begin(),r.input.end());return x;}
std::vector<std::uint32_t> materialRows(const std::vector<Record>& rows){std::vector<std::uint32_t>x;x.reserve(rows.size());for(const auto&r:rows)x.push_back(r.material);return x;}
void writeBatch(const std::filesystem::path& path,const std::string& policy,const CompiledWorld& world,
                const std::vector<Record>& rows,const std::vector<float>& out,const std::string& captureHash){
    std::ofstream f(path);need(f.good(),"cannot write batch output");
    f<<"{\"record\":\"header\",\"schema\":\"numi.cardboard.bilinear15-capture-diagnostic.v1\",\"policy\":\""<<policy
     <<"\",\"base_snapshot_manifest_sha256\":\""<<kBaseSnapshotSHA<<"\",\"capture_sha256\":\""<<captureHash<<"\",\"record_count\":"<<rows.size()
     <<",\"compiled_local_iterations_liner\":"<<world.materials[0].localNewtonIterations
     <<",\"compiled_local_iterations_medium\":"<<world.materials[1].localNewtonIterations
     <<",\"physical_validation\":false,\"interpretation\":\"direct stress and tangent projection status comparison only; not a material calibration or full solver qualification\"}\n";
    std::size_t stressOK=0,tangentOK=0,projectOK=0;
    for(std::size_t i=0;i<rows.size();++i){const auto* p=out.data()+i*kOutputStride;stressOK+=p[0]==1.0f;tangentOK+=p[1]==1.0f;projectOK+=p[2]==1.0f;
        double diff2=0.0,ref2=0.0,maxDiff=0.0;for(unsigned j=0;j<9u;++j){double d=double(p[8+j])-double(p[17+j]);diff2+=d*d;ref2+=double(p[8+j])*double(p[8+j]);maxDiff=std::max(maxDiff,std::abs(d));}
        f<<"{\"record\":\"point\",\"environment\":"<<rows[i].env<<",\"tet_id\":"<<rows[i].tet
         <<",\"material_index\":"<<rows[i].material<<",\"material_name\":\""<<rows[i].materialName
         <<"\",\"stress_valid\":"<<(p[0]==1.0f?"true":"false")<<",\"tangent_valid\":"<<(p[1]==1.0f?"true":"false")
         <<",\"tangent_projection_converged\":"<<(p[2]==1.0f?"true":"false")<<",\"projection_residual\":";numberJSON(f,p[3]);
        f<<",\"projection_state_norm\":";numberJSON(f,p[4]);f<<",\"projection_iterations\":";numberJSON(f,p[5]);
        f<<",\"projection_flags\":";numberJSON(f,p[6]);f<<",\"projection_failure\":";numberJSON(f,p[7]);
        f<<",\"direct_stress_pa\":";floatArrayJSON(f,p+8,9);f<<",\"tangent_path_stress_pa\":";floatArrayJSON(f,p+17,9);
        f<<",\"tangent_pa_per_s\":";floatArrayJSON(f,p+26,9);f<<",\"projected_state_high\":";floatArrayJSON(f,p+35,15);
        f<<",\"projected_state_low\":";floatArrayJSON(f,p+50,15);f<<",\"direct_vs_tangent_stress_abs_l2\":";numberJSON(f,std::sqrt(diff2));
        f<<",\"direct_vs_tangent_stress_relative_l2\":";numberJSON(f,ref2>0?std::sqrt(diff2/ref2):NAN);
        f<<",\"direct_vs_tangent_stress_max_abs\":";numberJSON(f,maxDiff);f<<",\"output_f32_bits\":";
        bitsArrayJSON(f,p,kOutputStride);f<<"}\n";
    }
    f<<"{\"record\":\"end\",\"rows\":"<<rows.size()<<",\"direct_stress_valid_count\":"<<stressOK
     <<",\"tangent_valid_count\":"<<tangentOK<<",\"tangent_projection_converged_count\":"<<projectOK
     <<",\"status\":\"diagnostic_complete\",\"not_qualification\":true}\n";
}
void writeTrace(const std::filesystem::path& path,const std::string& policy,const CompiledWorld& world,
                const Record& row,const float* out,const std::string& captureHash){
    std::ofstream f(path);need(f.good(),"cannot write selected projection trace");
    f<<"{\"record\":\"header\",\"schema\":\"numi.cardboard.bilinear15-selected-projection-trace.v1\",\"policy\":\""<<policy
     <<"\",\"base_snapshot_manifest_sha256\":\""<<kBaseSnapshotSHA<<"\",\"capture_sha256\":\""<<captureHash<<"\",\"compiled_local_iterations\":"<<world.materials[row.material].localNewtonIterations
     <<",\"point\":";inputRecordJSON(f,row);f<<",\"summary\":{\"direct_project_valid\":"<<(out[0]==1?"true":"false")
     <<",\"direct_projection_residual\":";numberJSON(f,out[1]);f<<",\"direct_state_norm\":";numberJSON(f,out[2]);
    f<<",\"direct_iterations\":";numberJSON(f,out[3]);f<<",\"direct_flags\":";numberJSON(f,out[4]);f<<",\"direct_failure\":";numberJSON(f,out[5]);
    f<<",\"stress_valid\":"<<(out[6]==1?"true":"false")<<",\"tangent_valid\":"<<(out[7]==1?"true":"false")
     <<",\"tangent_projection_converged\":"<<(out[8]==1?"true":"false")<<",\"tangent_projection_residual\":";numberJSON(f,out[9]);
    f<<",\"tangent_iterations\":";numberJSON(f,out[11]);f<<",\"tangent_flags\":";numberJSON(f,out[12]);f<<",\"tangent_failure\":";numberJSON(f,out[13]);
    f<<",\"stress_direct\":";floatArrayJSON(f,out+14,9);f<<",\"stress_tangent_path\":";floatArrayJSON(f,out+23,9);
    f<<",\"tangent\":";floatArrayJSON(f,out+32,9);f<<",\"projected_state_high\":";floatArrayJSON(f,out+41,15);
    f<<",\"projected_state_low\":";floatArrayJSON(f,out+56,15);f<<"},\"summary_f32_bits\":";
    bitsArrayJSON(f,out,73u);
    const float traceStatus=out[80u];
    const char* traceDisposition=traceStatus==-1.0f?"unsupported_by_existing_16_iteration_trace_capacity":"trace_recorded";
    f<<",\"trace_status_code\":";numberJSON(f,traceStatus);
    f<<",\"trace_disposition\":\""<<traceDisposition<<"\",\"packing_sentinels\":[\"0x3e200000\",\"0x3ea00000\"],\"trace_layout\":{\"header_slots\":8,\"iteration_slots\":374,\"record_slot5\":\"scaled_correction_norm; -1 when the legacy solve path has no scaled metric\",\"max_iterations\":16,\"slots\":"<<kTraceSlots<<"},\"trace_f32\":";
    floatArrayJSON(f,out+73,kTraceSlots);f<<",\"trace_f32_bits\":";bitsArrayJSON(f,out+73,kTraceSlots);f<<"}\n";
}
void runBatch(Metal& metal,const CompiledWorld& world,const std::string& policy,const std::vector<Record>& rows,
              const std::filesystem::path& outdir,const std::string& captureHash){
    auto params=materialParameters(world);auto acc=acceptedRows(rows);auto inputs=inputRows(rows);auto ids=materialRows(rows);
    std::vector<float> out(rows.size()*kOutputStride,0.0f);
    std::vector<id<MTLBuffer>> bs{buffer(metal.device,world.materials),buffer(metal.device,world.scalarPrograms),
       buffer(metal.device,world.instructions),buffer(metal.device,params),buffer(metal.device,acc),
       buffer(metal.device,inputs),buffer(metal.device,ids),buffer(metal.device,out)};
    need(std::all_of(bs.begin(),bs.end(),[](id<MTLBuffer> b){return b!=nil;}),"batch Metal buffer allocation failed");
    metal.dispatch(metal.batch,bs,static_cast<std::uint32_t>(rows.size()),8u);
    const auto* data=static_cast<const float*>(bs[7].contents);std::copy(data,data+out.size(),out.begin());
    writeBatch(outdir/("paper-points-"+policy+".jsonl"),policy,world,rows,out,captureHash);
}
void runTrace(Metal& metal,const CompiledWorld& world,const std::string& policy,const Record& row,
              const std::filesystem::path& outdir,const std::string& captureHash){
    auto params=materialParameters(world);std::vector<float> acc(row.accepted.begin(),row.accepted.end());
    std::vector<float> inputs(row.input.begin(),row.input.end());std::vector<float> out(73u+kTraceSlots,0.0f);
    std::uint32_t mat=row.material;
    std::vector<id<MTLBuffer>> bs{buffer(metal.device,world.materials),buffer(metal.device,world.scalarPrograms),
       buffer(metal.device,world.instructions),buffer(metal.device,params),buffer(metal.device,acc),
       buffer(metal.device,inputs),buffer(metal.device,out)};
    need(std::all_of(bs.begin(),bs.end(),[](id<MTLBuffer> b){return b!=nil;}),"trace Metal buffer allocation failed");
    id<MTLCommandBuffer> cb=[metal.queue commandBuffer];need(cb!=nil,"trace command buffer unavailable");
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];need(enc!=nil,"trace encoder unavailable");
    [enc setComputePipelineState:metal.trace];for(NSUInteger i=0;i<bs.size();++i)[enc setBuffer:bs[i] offset:0 atIndex:i];
    [enc setBytes:&mat length:sizeof(mat) atIndex:7];
    [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"trace Metal dispatch failed");
    const auto* data=static_cast<const float*>(bs[6].contents);
    need(std::bit_cast<std::uint32_t>(data[71])==0x3e200000u&&
         std::bit_cast<std::uint32_t>(data[72])==0x3ea00000u,
         "selected trace high/low state packing sentinel was overwritten");
    writeTrace(outdir/("tet1528-env0-"+policy+"-trace.json"),policy,world,row,data,captureHash);
}
// Reserve evidence exclusively before dispatch; failures retain partial evidence.
struct ExclusiveSweepOutput {
    int descriptor = -1;
    explicit ExclusiveSweepOutput(const std::filesystem::path& path) {
        descriptor=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
        need(descriptor>=0,"cannot exclusively create controlled sweep output: "+path.string());
    }
    ~ExclusiveSweepOutput(){if(descriptor>=0)::close(descriptor);}
    ExclusiveSweepOutput(const ExclusiveSweepOutput&)=delete;
    ExclusiveSweepOutput& operator=(const ExclusiveSweepOutput&)=delete;
    void writeAll(const std::string& text) {
        need(text.size()<=16u*1024u*1024u,"controlled sweep output exceeds bounded writer");
        std::size_t written=0u;
        while(written<text.size()) {
            const auto count=::write(descriptor,text.data()+written,text.size()-written);
            if(count<0&&errno==EINTR)continue;
            need(count>0,"controlled sweep write failed; partial evidence retained");
            written+=static_cast<std::size_t>(count);
        }
        const int saved=descriptor;descriptor=-1;
        need(::close(saved)==0,"controlled sweep close failed; evidence retained");
    }
};
// Instrument-only controlled directions around immutable captured centers.
// Output records identify authored perturbations separately from capture data.
struct DerivativeSweepRecord {
    Record row;
    const char* kind = "center_repeat";
    int basis = -1;
    int sign = 0;
    unsigned repeat = 0;
    double epsilon = 0.0;
};
std::vector<DerivativeSweepRecord> derivativeSweepRecords(const Capture& capture) {
    constexpr std::array<unsigned,8> selected = {567u,568u,569u,1527u,1528u,1529u,1634u,1638u};
    constexpr std::array<double,10> epsilons = {1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12};
    std::vector<DerivativeSweepRecord> records;
    for (const auto tet : selected) {
        const auto found = std::find_if(capture.rows.begin(),capture.rows.end(),[&](const Record& r) {
            return r.env == 0u && r.tet == tet;
        });
        need(found != capture.rows.end(),"controlled derivative sweep point missing");
        const Record center = *found;
        for (unsigned repeat=0u;repeat<3u;++repeat) {
            DerivativeSweepRecord record;record.row=center;record.repeat=repeat;
            records.push_back(record);
        }
        for (unsigned basis=0u;basis<9u;++basis) {
            Record directed=center;
            for (unsigned j=0u;j<9u;++j) {
                directed.input[27u+j]=j==basis?1.0f:0.0f;
                directed.input[36u+j]=j==basis?1.0f/center.input[45u]:0.0f;
            }
            DerivativeSweepRecord action;action.row=directed;
            action.kind="basis_action";action.basis=static_cast<int>(basis);
            records.push_back(action);
            for (const double epsilon : epsilons) for (const int sign : {-1,1}) {
                DerivativeSweepRecord sample=action;
                sample.kind="basis_perturbation";sample.sign=sign;sample.epsilon=epsilon;
                const double targetF=double(center.input[basis])+double(center.input[9u+basis])+
                    double(sign)*epsilon;
                const float high=static_cast<float>(targetF);
                sample.row.input[basis]=high;
                // Form the residual before adding the order-one high word.
                // This preserves sub-FP64-ULP perturbations when the high
                // word stays unchanged; every realized input is still logged.
                const double residualF=(double(center.input[basis])-double(high))+
                    (double(center.input[9u+basis])+double(sign)*epsilon);
                sample.row.input[9u+basis]=static_cast<float>(residualF);
                // Preserve the captured FEM kinematic F/Fdot perturbation
                // relationship; output retains every actually rounded word.
                const double targetRate=double(center.input[18u+basis])+
                    double(sign)*epsilon/double(center.input[45u]);
                sample.row.input[18u+basis]=static_cast<float>(targetRate);
                need(std::isfinite(sample.row.input[basis])&&
                     std::isfinite(sample.row.input[9u+basis])&&
                     std::isfinite(sample.row.input[18u+basis]),"nonfinite controlled perturbation");
                records.push_back(sample);
            }
        }
    }
    need(records.size()==8u*(3u+9u*(1u+2u*epsilons.size())),"controlled sweep cardinality mismatch");
    return records;
}
void runDerivativeSweep(Metal& metal,const CompiledWorld& world,const std::string& policy,
                        const Capture& capture,const std::filesystem::path& outdir,
                        const std::string& captureHash) {
    ExclusiveSweepOutput evidence(outdir/("paper-derivative-sweep-"+policy+".jsonl"));
    const auto samples=derivativeSweepRecords(capture);
    std::vector<Record> rows;rows.reserve(samples.size());
    for(const auto& sample:samples)rows.push_back(sample.row);
    auto params=materialParameters(world);auto accepted=acceptedRows(rows);
    auto inputs=inputRows(rows);auto ids=materialRows(rows);
    std::vector<float> output(rows.size()*kOutputStride,0.0f);
    std::vector<id<MTLBuffer>> buffers{buffer(metal.device,world.materials),
        buffer(metal.device,world.scalarPrograms),buffer(metal.device,world.instructions),
        buffer(metal.device,params),buffer(metal.device,accepted),buffer(metal.device,inputs),
        buffer(metal.device,ids),buffer(metal.device,output)};
    need(std::all_of(buffers.begin(),buffers.end(),[](id<MTLBuffer> b){return b!=nil;}),
         "controlled derivative sweep allocation failed");
    metal.dispatch(metal.batch,buffers,static_cast<std::uint32_t>(rows.size()),8u);
    const auto* result=static_cast<const float*>(buffers[7].contents);
    std::copy(result,result+output.size(),output.begin());
    std::ostringstream file;
    file<<"{\"record\":\"header\",\"schema\":\"numi.cardboard.controlled-material-derivative-sweep.v1\","
        <<"\"policy\":\""<<policy<<"\",\"compiled_local_iterations_liner\":"<<world.materials[0].localNewtonIterations
        <<",\"compiled_local_iterations_medium\":"<<world.materials[1].localNewtonIterations
        <<",\"capture_sha256\":\""<<captureHash
        <<"\",\"base_snapshot_manifest_sha256\":\""<<kBaseSnapshotSHA
        <<"\",\"rows\":"<<samples.size()<<",\"selected_points\":8,\"basis_order\":\"row_major_F_11_12_13_21_22_23_31_32_33\","
        <<"\"epsilon_unit\":\"s\",\"basis_direction_unit\":\"per_s\","
        <<"\"rate_direction_rule\":\"Hdot=H/dt; actual rounded F and Fdot words recorded\","
        <<"\"center_repeat_meaning\":\"same-batch nondeterminism observation only; not an absolute arithmetic noise floor\","
        <<"\"original_capture_modified\":false,\"physical_steps\":0,\"physical_validation\":false}\n";
    for(std::size_t i=0u;i<samples.size();++i) {
        const auto& sample=samples[i];const auto* value=output.data()+i*kOutputStride;
        file<<"{\"record\":\"controlled_sample\",\"kind\":\""<<sample.kind
            <<"\",\"basis\":"<<sample.basis<<",\"sign\":"<<sample.sign
            <<",\"repeat\":"<<sample.repeat<<",\"epsilon_s\":"<<std::setprecision(17)<<sample.epsilon
            <<",\"point\":";inputRecordJSON(file,sample.row);
        file<<",\"output_f32_bits\":";bitsArrayJSON(file,value,kOutputStride);
        file<<",\"stress_valid\":"<<(value[0]==1.0f?"true":"false")
            <<",\"tangent_valid\":"<<(value[1]==1.0f?"true":"false")
            <<",\"projection_valid\":"<<(value[2]==1.0f?"true":"false")
            <<",\"projection_failure\":";numberJSON(file,value[7]);file<<"}\n";
    }
    file<<"{\"record\":\"end\",\"rows\":"<<samples.size()
        <<",\"status\":\"diagnostic_complete\",\"not_qualification\":true}\n";
    need(file.good(),"controlled derivative sweep formatting failed");
    evidence.writeAll(file.str());
}

}

int main(int argc,char** argv){@autoreleasepool{try{
    const bool cpu=argc==7&&std::string(argv[1])=="--cpu-admission"&&std::string(argv[2])=="--expected-capture-sha256";
    const bool nativeSelected=argc==9&&std::string(argv[1])=="--native"&&std::string(argv[2])=="--selected-only"&&std::string(argv[3])=="--expected-capture-sha256";
    const bool nativeAll=argc==9&&std::string(argv[1])=="--native"&&std::string(argv[2])=="--all-points"&&std::string(argv[3])=="--expected-capture-sha256";
    const bool nativeSweep=argc==9&&std::string(argv[1])=="--native"&&std::string(argv[2])=="--derivative-sweep"&&std::string(argv[3])=="--expected-capture-sha256";
    const bool native=nativeSelected||nativeAll||nativeSweep;
    need(cpu||native,"usage: capture-diagnostic --cpu-admission --expected-capture-sha256 SHA CAPTURE LINER.nmatter MEDIUM.nmatter | --native --selected-only|--all-points|--derivative-sweep --expected-capture-sha256 SHA CAPTURE LINER.nmatter MEDIUM.nmatter OUTPUT_DIR");
    const std::string expectedCaptureSHA=argv[cpu?3:4];
    need(expectedCaptureSHA.size()==64u&&std::all_of(expectedCaptureSHA.begin(),expectedCaptureSHA.end(),[](unsigned char c){return std::isxdigit(c)!=0;}),
         "expected capture SHA must be exactly 64 hexadecimal characters");
    const auto capturePath=std::filesystem::path(argv[cpu?4:5]);
    const auto linerPath=std::filesystem::path(argv[cpu?5:6]);const auto mediumPath=std::filesystem::path(argv[cpu?6:7]);
    const auto capSHA=sha256(capturePath),linerSHA=sha256(linerPath),mediumSHA=sha256(mediumPath);
    need(capSHA==expectedCaptureSHA,"capture SHA differs from explicit source-bound CLI binding");
    need(linerSHA==kLinerSHA&&mediumSHA==kMediumSHA,"source paper material SHA mismatch");
    const Capture capture=readCapture(capturePath);
    const CompiledWorld defaults=compileSources(linerPath,mediumPath,false);
    const CompiledWorld accurate=compileSources(linerPath,mediumPath,true);
    verifyParameters(defaults,capture);verifyParameters(accurate,capture);
    if(cpu){
        need(derivativeSweepRecords(capture).size()==960u,"controlled sweep CPU admission failed");
        std::cout<<"{\"record\":\"cpu_admission\",\"schema\":\"numi.cardboard.bilinear15-capture-diagnostic.v1\",\"status\":\"pass\",\"capture_sha256\":\""<<capSHA
         <<"\",\"base_snapshot_manifest_sha256\":\""<<kBaseSnapshotSHA
         <<"\",\"liner_sha256\":\""<<linerSHA<<"\",\"medium_sha256\":\""<<mediumSHA
         <<"\",\"paper_row_count\":"<<capture.rows.size()<<",\"selected_liner_tet1528\":true,\"default_iterations_liner\":"<<defaults.materials[0].localNewtonIterations
         <<",\"default_iterations_medium\":"<<defaults.materials[1].localNewtonIterations<<",\"accurate_iterations_liner\":"<<accurate.materials[0].localNewtonIterations
         <<",\"accurate_iterations_medium\":"<<accurate.materials[1].localNewtonIterations<<",\"physical_native_execution\":false}\n";return 0;
    }
    const std::filesystem::path outdir(argv[8]);std::filesystem::create_directories(outdir);
    Metal metal;
    if(nativeSweep){
        runDerivativeSweep(metal,defaults,"default",capture,outdir,capSHA);
        runDerivativeSweep(metal,accurate,"accurate",capture,outdir,capSHA);
        std::cout<<"{\"record\":\"end\",\"status\":\"controlled_derivative_diagnostic_complete\",\"physical_steps\":0,\"physical_validation\":false}\n";
        return 0;
    }
    std::cout<<"{\"record\":\"native_instrument\",\"device\":\""<<escaped([[metal.device name] UTF8String])
      <<"\",\"base_snapshot_manifest_sha256\":\""<<kBaseSnapshotSHA<<"\",\"capture_sha256\":\""<<capSHA
      <<"\",\"execution_scope\":\""<<(nativeAll?"selected_then_all_points":"selected_point_only")
      <<"\",\"paper_rows\":"<<capture.rows.size()<<",\"steps\":0,\"physical_validation\":false}\n";
    const auto it=std::find_if(capture.rows.begin(),capture.rows.end(),[](const Record&r){return r.env==0u&&r.tet==1528u&&r.material==0u;});
    need(it!=capture.rows.end(),"selected liner tet1528 disappeared");
    // Run the selected failure point first so a full 3,456-row replay cannot
    // delay the immediate native projection diagnosis.
    runTrace(metal,defaults,"default",*it,outdir,capSHA);
    runTrace(metal,accurate,"accurate",*it,outdir,capSHA);
    if(nativeAll){
        runBatch(metal,defaults,"default",capture.rows,outdir,capSHA);
        runBatch(metal,accurate,"accurate",capture.rows,outdir,capSHA);
    }
    std::cout<<"{\"record\":\"end\",\"status\":\"diagnostic_complete\",\"paper_rows_per_policy\":"<<capture.rows.size()
      <<",\"execution_scope\":\""<<(nativeAll?"selected_then_all_points":"selected_point_only")
      <<"\",\"selected_point\":\"env0/tet1528/liner\",\"physical_steps\":0,\"physical_validation\":false}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL source-bound bilinear15 capture diagnostic: "<<e.what()<<'\n';return 1;}}}
