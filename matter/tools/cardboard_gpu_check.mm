#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"

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
using Matrix = std::array<double,9>;
using Vec = std::array<double,3>;
constexpr Matrix identity{1,0,0,0,1,0,0,0,1};
constexpr double edge=0.0078125;
unsigned checks=0;

void need(bool ok,const std::string& reason) {
    ++checks;
    if(!ok)throw std::runtime_error(reason);
}
std::string number(double x) {
    std::ostringstream out;out<<std::setprecision(17)<<x;return out.str();
}
std::string diagnostics(const std::vector<Diagnostic>& values) {
    std::string out;for(const auto& value:values)out+=value.message+"; ";return out;
}
Matrix multiply(const Matrix& a,const Matrix& b) {
    Matrix c{};
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)
        for(unsigned k=0;k<3;++k)c[3*i+j]+=a[3*i+k]*b[3*k+j];
    return c;
}
Matrix transpose(const Matrix& a) {
    Matrix b{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)b[3*i+j]=a[3*j+i];return b;
}
Matrix add(const Matrix& a,const Matrix& b) {
    Matrix c{};for(unsigned i=0;i<9;++i)c[i]=a[i]+b[i];return c;
}
double vectorMax(const std::array<Vec,4>& a) {
    double value=0;for(const auto& row:a)for(double x:row)value=std::max(value,std::abs(x));return value;
}
Matrix rotation(double angle,unsigned axis) {
    Matrix r=identity;const unsigned i=(axis+1u)%3u,j=(axis+2u)%3u;
    r[3*i+i]=r[3*j+j]=std::cos(angle);r[3*i+j]=-std::sin(angle);r[3*j+i]=std::sin(angle);return r;
}
Vec transform(const Matrix& a,const Vec& b) {
    Vec c{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)c[i]+=a[3*i+j]*b[j];return c;
}
Vec xyz(const nm_float4& v) { return {v.x,v.y,v.z}; }
Vec sub(const Vec& a,const Vec& b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
Matrix columns(const Vec& a,const Vec& b,const Vec& c) {
    return {a[0],b[0],c[0],a[1],b[1],c[1],a[2],b[2],c[2]};
}
template<class T> id<MTLBuffer> buffer(id<MTLDevice> device,const std::vector<T>& values) {
    const T zero{};const void* data=values.empty()?static_cast<const void*>(&zero):values.data();
    return [device newBufferWithBytes:data length:std::max(std::size_t(1),values.size())*sizeof(T)
        options:MTLResourceStorageModeShared];
}

struct Device {
    id<MTLDevice> device=nil;id<MTLCommandQueue> queue=nil;id<MTLLibrary> library=nil;
    id<MTLComputePipelineState> force=nil,op=nil;
    Device() {
        device=MTLCreateSystemDefaultDevice();need(device!=nil,"Apple Metal is unavailable");
        need([[device name] rangeOfString:@"Apple"].location!=NSNotFound &&
             [[device name] rangeOfString:@"Paravirtual"].location==NSNotFound,
             "a physical Apple Metal device is required");
        queue=[device newCommandQueue];NSError* error=nil;
        library=[device newLibraryWithURL:[NSURL fileURLWithPath:
            [NSString stringWithUTF8String:NUMI_MATTER_METALLIB]] error:&error];
        need(library!=nil,"production Matter metallib unavailable");
        force=make(@"numi_matter_metal::nm_fem_internal_forces");
        op=make(@"numi_matter_metal::nm_fem_apply_operator_elements");
    }
    id<MTLComputePipelineState> make(NSString* name) {
        NSError* error=nil;id<MTLFunction> function=[library newFunctionWithName:name];
        need(function!=nil,"production FEM function missing");
        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
        need(pipeline!=nil,"production FEM pipeline creation failed");return pipeline;
    }
};

struct Law {
    Matrix c{}; // normal Green-strain stiffness in M,C,T order, SI
    std::array<double,3> g{}; // CT, MT, CM engineering shear modulus, SI
    Matrix compliance{};
};

double get(const MaterialProgram& material,const std::string& name) {
    for(const auto& p:material.parameters)if(p.name==name)return p.defaultValue;
    throw std::runtime_error("missing source parameter "+name);
}
bool has(const MaterialProgram& material,const std::string& name) {
    for(const auto& p:material.parameters)if(p.name==name)return true;return false;
}
Law lawFrom(const MaterialProgram& material) {
    Law law;
    if(has(material,"c_MM")) {
        law.c={get(material,"c_MM"),get(material,"c_MC"),get(material,"c_MT"),
            get(material,"c_MC"),get(material,"c_CC"),get(material,"c_CT"),
            get(material,"c_MT"),get(material,"c_CT"),get(material,"c_TT")};
        law.g={get(material,"g_CT"),get(material,"g_MT"),get(material,"g_CM")};
        // Invert the small symmetric normal block for the free-sided load state.
        const double a=law.c[0],b=law.c[1],c=law.c[2],d=law.c[4],e=law.c[5],f=law.c[8];
        const double det=a*(d*f-e*e)-b*(b*f-c*e)+c*(b*e-c*d);
        need(det>0.0,"orthotropic stiffness is not positive definite");
        law.compliance={(d*f-e*e)/det,(c*e-b*f)/det,(b*e-c*d)/det,
            (c*e-b*f)/det,(a*f-c*c)/det,(b*c-a*e)/det,
            (b*e-c*d)/det,(b*c-a*e)/det,(a*d-b*b)/det};
    } else {
        const double lambda=get(material,"lambda"),mu=get(material,"mu");
        law.c={lambda+2*mu,lambda,lambda,lambda,lambda+2*mu,lambda,
            lambda,lambda,lambda+2*mu};
        law.g={mu,mu,mu};
        const double young=mu*(3*lambda+2*mu)/(lambda+mu);
        const double nu=lambda/(2*(lambda+mu));
        law.compliance={1/young,-nu/young,-nu/young,-nu/young,1/young,-nu/young,
            -nu/young,-nu/young,1/young};
    }
    return law;
}

struct Result { Matrix stress{},tangent{}; };
Result stvk(const Law& law,const Matrix& f,const Matrix& h) {
    // Independent tensor oracle: second Piola stress S=C:E, first Piola P=F*S,
    // and DP[H]=H*S+F*C:sym(F^T H). It does not use Matter bytecode or AD.
    const Matrix c= multiply(transpose(f),f);
    const std::array<double,3> en{.5*(c[0]-1),.5*(c[4]-1),.5*(c[8]-1)};
    std::array<double,3> sn{};
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)sn[i]+=law.c[3*i+j]*en[j];
    const Matrix second{sn[0],law.g[2]*c[1],law.g[1]*c[2],
        law.g[2]*c[1],sn[1],law.g[0]*c[5],law.g[1]*c[2],law.g[0]*c[5],sn[2]};
    const Matrix dc=add(multiply(transpose(f),h),multiply(transpose(h),f));
    const std::array<double,3> den{.5*dc[0],.5*dc[4],.5*dc[8]};
    std::array<double,3> dsn{};
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)dsn[i]+=law.c[3*i+j]*den[j];
    const Matrix ds{dsn[0],law.g[2]*dc[1],law.g[1]*dc[2],
        law.g[2]*dc[1],dsn[1],law.g[0]*dc[5],law.g[1]*dc[2],law.g[0]*dc[5],dsn[2]};
    return {multiply(f,second),add(multiply(h,second),multiply(f,ds))};
}
Matrix quaternionFrame(const nm_float4& raw) {
    const double inv=1/std::sqrt(double(raw.x)*raw.x+double(raw.y)*raw.y+
        double(raw.z)*raw.z+double(raw.w)*raw.w);
    const double x=raw.x*inv,y=raw.y*inv,z=raw.z*inv,w=raw.w*inv;
    return {1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)};
}
Result framed(const Law& law,const Matrix& f,const Matrix& h,const Matrix& q) {
    auto r=stvk(law,multiply(f,q),multiply(h,q));
    r.stress=multiply(r.stress,transpose(q));
    r.tangent=multiply(r.tangent,transpose(q));
    return r;
}

CompiledWorld cook(const MaterialProgram& material) {
    WorldSource source;source.environmentCount=1;source.frameTimestep=1e-8;
    source.gravity={0,0,0};source.materials={material};
    ObjectSource object;object.name="synthetic_rotated_cardboard_constitutive_tet";
    object.representation=Representation::fem;object.mixedFEM=false;
    object.deformableContact=false;object.deformableSelfContact=false;
    object.characteristicLength=edge;
    object.femNodes={{{0,0,0}},{{edge,0,0}},{{0,2*edge,0}},{{0,0,4*edge}}};
    object.tetrahedra={{{0,1,2,3}}};
    const double half=.155;
    object.femMaterialFrameRotations={{0,0,std::sin(half),std::cos(half)}};
    object.femMaterialFrameSourceIdentity={0x47acb1d27e33f690ull,0x912a7d41584f3c02ull,
        0xdf60c9a71b438e52ull,0x2581e4bc7d9a063full};
    source.objects={object};
    const auto result=compileWorld(source,{.maximumRateExponent=0,.emitSpecializedMetal=false});
    need(result.succeeded(),"native FEM cook failed: "+diagnostics(result.diagnostics));
    return result.world;
}

struct Inputs {
    std::vector<NMFEMNodeStateGPU> nodes;
    std::vector<NMFEMFieldStateGPU> fields;
    std::vector<nm_float4> direction;
    std::vector<float> state;
};
Inputs makeInputs(const CompiledWorld& world,const Matrix& f,const Matrix& h) {
    Inputs input;input.nodes=world.fem.nodes;input.fields=world.fem.fields;
    input.direction.resize(world.dispatch.femNodeCount);
    input.state.resize(world.dispatch.tetrahedronCount*world.dispatch.materialStateStride);
    for(unsigned n=0;n<world.dispatch.femNodeCount;++n) {
        const Vec rest=xyz(world.fem.nodes[n].positionAndMass);
        const Vec current=transform(f,rest),direction=transform(h,rest);
        input.nodes[n].positionAndMass.x=float(current[0]);
        input.nodes[n].positionAndMass.y=float(current[1]);
        input.nodes[n].positionAndMass.z=float(current[2]);
        input.nodes[n].velocityAndInverseMass.x=0;
        input.nodes[n].velocityAndInverseMass.y=0;
        input.nodes[n].velocityAndInverseMass.z=0;
        input.direction[n]={float(direction[0]),float(direction[1]),float(direction[2]),0};
    }
    for(unsigned i=0;i<world.stateInitials.size();++i)input.state[i]=world.stateInitials[i];
    return input;
}
Matrix deformation(const CompiledWorld& world,const Inputs& input) {
    const auto& tet=world.fem.tetrahedra[0];
    const auto at=[&](const unsigned node){return xyz(input.nodes[node].positionAndMass);};
    const Matrix inv{tet.inverseRestRow0.x,tet.inverseRestRow0.y,tet.inverseRestRow0.z,
        tet.inverseRestRow1.x,tet.inverseRestRow1.y,tet.inverseRestRow1.z,
        tet.inverseRestRow2.x,tet.inverseRestRow2.y,tet.inverseRestRow2.z};
    return multiply(columns(sub(at(tet.nodes.y),at(tet.nodes.x)),
        sub(at(tet.nodes.z),at(tet.nodes.x)),sub(at(tet.nodes.w),at(tet.nodes.x))),inv);
}
Matrix deformationDirection(const CompiledWorld& world,const Inputs& input) {
    const auto& tet=world.fem.tetrahedra[0];
    const auto at=[&](const unsigned node){return xyz(input.direction[node]);};
    const Matrix inv{tet.inverseRestRow0.x,tet.inverseRestRow0.y,tet.inverseRestRow0.z,
        tet.inverseRestRow1.x,tet.inverseRestRow1.y,tet.inverseRestRow1.z,
        tet.inverseRestRow2.x,tet.inverseRestRow2.y,tet.inverseRestRow2.z};
    return multiply(columns(sub(at(tet.nodes.y),at(tet.nodes.x)),
        sub(at(tet.nodes.z),at(tet.nodes.x)),sub(at(tet.nodes.w),at(tet.nodes.x))),inv);
}

std::vector<NMFEMElementVectorGPU> dispatch(Device& device,const CompiledWorld& world,
                                           const Inputs& input,const bool tangent) {
    NMMicrostepGPU micro{};micro.flags=NM_MICROSTEP_FGMRES_OPERATOR;
    const float dt=world.dispatch.gravityAndTimestep.w;
    micro.time={dt,1.0f/dt,0,0};
    std::vector<float> parameters;for(const auto& p:world.parameters)parameters.push_back(p.valueAndBounds.x);
    auto objects=buffer(device.device,world.objects),materials=buffer(device.device,world.materials);
    auto programs=buffer(device.device,world.scalarPrograms),instructions=buffer(device.device,world.instructions);
    auto params=buffer(device.device,parameters),nodes=buffer(device.device,input.nodes);
    auto tets=buffer(device.device,world.fem.tetrahedra),state=buffer(device.device,input.state);
    auto schedulers=buffer(device.device,world.schedulers),adaptive=buffer(device.device,world.adaptive);
    auto mixed=buffer(device.device,world.mixedMaterials),fields=buffer(device.device,input.fields);
    auto learned=buffer(device.device,world.learnedMaterials),layers=buffer(device.device,world.learnedLayers);
    auto weights=buffer(device.device,world.learnedWeights),direction=buffer(device.device,input.direction);
    auto solver=buffer(device.device,std::vector<NMFGMRESStateGPU>(1));
    auto statuses=buffer(device.device,std::vector<NMMatterStatusGPU>(1));
    auto output=buffer(device.device,std::vector<NMFEMElementVectorGPU>(1));
    id<MTLCommandBuffer> command=[device.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:tangent?device.op:device.force];
    [encoder setBytes:&world.dispatch length:sizeof(world.dispatch) atIndex:0];
    [encoder setBytes:&micro length:sizeof(micro) atIndex:1];
    [encoder setBuffer:objects offset:0 atIndex:2];[encoder setBuffer:materials offset:0 atIndex:3];
    [encoder setBuffer:programs offset:0 atIndex:4];[encoder setBuffer:instructions offset:0 atIndex:5];
    [encoder setBuffer:params offset:0 atIndex:6];[encoder setBuffer:nodes offset:0 atIndex:7];
    [encoder setBuffer:tets offset:0 atIndex:8];[encoder setBuffer:state offset:0 atIndex:9];
    if(tangent) {
        [encoder setBuffer:direction offset:0 atIndex:10];[encoder setBuffer:schedulers offset:0 atIndex:11];
        [encoder setBuffer:adaptive offset:0 atIndex:12];[encoder setBuffer:output offset:0 atIndex:13];
        [encoder setBuffer:statuses offset:0 atIndex:14];[encoder setBuffer:mixed offset:0 atIndex:16];
        [encoder setBuffer:fields offset:0 atIndex:17];[encoder setBuffer:learned offset:0 atIndex:18];
        [encoder setBuffer:layers offset:0 atIndex:19];[encoder setBuffer:weights offset:0 atIndex:20];
        [encoder setBytes:&world.mixedSolver length:sizeof(world.mixedSolver) atIndex:21];
        [encoder setBuffer:solver offset:0 atIndex:22];
    } else {
        [encoder setBuffer:schedulers offset:0 atIndex:10];[encoder setBuffer:adaptive offset:0 atIndex:11];
        [encoder setBuffer:output offset:0 atIndex:12];[encoder setBuffer:statuses offset:0 atIndex:13];
        [encoder setBuffer:mixed offset:0 atIndex:14];[encoder setBuffer:fields offset:0 atIndex:15];
        [encoder setBuffer:learned offset:0 atIndex:16];[encoder setBuffer:layers offset:0 atIndex:17];
        [encoder setBuffer:weights offset:0 atIndex:18];
    }
    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    need(command.status==MTLCommandBufferStatusCompleted,"production FEM dispatch failed");
    const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);
    need(status[0].code==NM_STATUS_SUCCESS,"production FEM status="+std::to_string(status[0].code));
    const auto* value=static_cast<const NMFEMElementVectorGPU*>(output.contents);
    return {value,value+1};
}

std::array<Vec,4> elementForces(const NMTetrahedronGPU& tet,const Matrix& p) {
    const Matrix inverseRest{tet.inverseRestRow0.x,tet.inverseRestRow0.y,tet.inverseRestRow0.z,
        tet.inverseRestRow1.x,tet.inverseRestRow1.y,tet.inverseRestRow1.z,
        tet.inverseRestRow2.x,tet.inverseRestRow2.y,tet.inverseRestRow2.z};
    const Matrix h=multiply(p,transpose(inverseRest));
    const double volume=double(tet.inverseRestRow0.w);
    std::array<Vec,4> out{};
    for(unsigned node=1;node<4;++node)for(unsigned row=0;row<3;++row) {
        out[node][row]=-volume*h[3*row+node-1];
        out[0][row]-=out[node][row];
    }
    return out;
}
std::array<Vec,4> observed(const NMFEMElementVectorGPU& value) {
    return {xyz(value.node0),xyz(value.node1),xyz(value.node2),xyz(value.node3)};
}
double relativeForceError(const std::array<Vec,4>& a,const std::array<Vec,4>& b) {
    double error=0,scale=0;
    for(unsigned n=0;n<4;++n)for(unsigned i=0;i<3;++i) {
        error=std::max(error,std::abs(a[n][i]-b[n][i]));scale=std::max(scale,std::abs(b[n][i]));
    }
    return error/std::max(1e-6,scale);
}
double relativeForceDifference(const std::array<Vec,4>& a,const std::array<Vec,4>& b) {
    double error=0,scale=0;
    for(unsigned n=0;n<4;++n)for(unsigned i=0;i<3;++i) {
        error=std::max(error,std::abs(a[n][i]-b[n][i]));scale=std::max(scale,std::abs(a[n][i]));
    }
    return error/std::max(1e-6,scale);
}
double absoluteForceDifference(const std::array<Vec,4>& a,const std::array<Vec,4>& b) {
    double error=0;
    for(unsigned n=0;n<4;++n)for(unsigned i=0;i<3;++i)
        error=std::max(error,std::abs(a[n][i]-b[n][i]));
    return error;
}
std::array<Vec,4> expectedForces(const CompiledWorld& world,const Law& law,
                                 const Inputs& input,const bool tangent) {
    const auto& tet=world.fem.tetrahedra[0];
    const Matrix q=quaternionFrame(tet.materialFrameRotation);
    const Matrix f=deformation(world,input),h=deformationDirection(world,input);
    const auto result=framed(law,f,h,q);
    return elementForces(tet,tangent?result.tangent:result.stress);
}

void checkOne(Device& device,const std::filesystem::path& path,const double sourceEMax) {
    auto parsed=parseMatterFile(path);
    need(parsed.succeeded(),"material parse failed: "+path.string()+" "+diagnostics(parsed.diagnostics));
    const MaterialProgram material=parsed.material;
    const Law law=lawFrom(material);
    const CompiledWorld world=cook(material);
    const auto& tet=world.fem.tetrahedra[0];
    const Matrix q=quaternionFrame(tet.materialFrameRotation);
    const Matrix bodyRotation=multiply(rotation(.37,1),rotation(-.21,0));
    Matrix oneAxis{};
    constexpr double axialGreen=.002;
    oneAxis[0]=axialGreen;
    oneAxis[4]=law.compliance[3]/law.compliance[0]*axialGreen;
    oneAxis[8]=law.compliance[6]/law.compliance[0]*axialGreen;
    Matrix stretch=identity;
    stretch[0]=std::sqrt(1+2*oneAxis[0]);
    stretch[4]=std::sqrt(1+2*oneAxis[4]);
    stretch[8]=std::sqrt(1+2*oneAxis[8]);
    const Matrix loaded=multiply(multiply(multiply(bodyRotation,q),stretch),transpose(q));
    const Matrix h{.19,-.13,.07,.11,-.17,.04,-.05,.09,.14};

    const auto noLoad=makeInputs(world,identity,Matrix{});
    const auto noLoadForces=observed(dispatch(device,world,noLoad,false)[0]);
    const auto identityExpected=expectedForces(world,law,noLoad,false);
    const double identityError=absoluteForceDifference(noLoadForces,identityExpected);
    const auto rotated=makeInputs(world,bodyRotation,Matrix{});
    const auto rotatedForces=observed(dispatch(device,world,rotated,false)[0]);
    const auto rotatedExpected=expectedForces(world,law,rotated,false);
    const double rotatedError=absoluteForceDifference(rotatedForces,rotatedExpected);
    const double rotatedFloor=vectorMax(rotatedForces);
    const double floorBound=64.0*std::numeric_limits<float>::epsilon()*sourceEMax*
        double(tet.inverseRestRow0.w)*128.0;
    need(vectorMax(noLoadForces)<=floorBound,
        material.name+" identity no-load force exceeds the conservative FP32 floor bound: "+number(vectorMax(noLoadForces)));
    need(rotatedFloor<=floorBound,
        material.name+" rotated no-load force exceeds the conservative FP32 floor bound: "+number(rotatedFloor));

    const auto loadedInput=makeInputs(world,loaded,h);
    const auto loadedForce=dispatch(device,world,loadedInput,false)[0];
    const auto loadedTangent=dispatch(device,world,loadedInput,true)[0];
    const auto actual=observed(loadedForce);
    const auto expected=expectedForces(world,law,loadedInput,false);
    const double loadedOracleError=relativeForceError(actual,expected);
    const auto actualTangent=observed(loadedTangent);
    const auto expectedTangent=expectedForces(world,law,loadedInput,true);
    const double tangentOracleError=relativeForceError(actualTangent,expectedTangent);
    constexpr double epsilon=1.0/2048.0;
    Matrix plusF=loaded,minusF=loaded;
    for(unsigned i=0;i<9;++i){plusF[i]+=epsilon*h[i];minusF[i]-=epsilon*h[i];}
    const auto plus=observed(dispatch(device,world,makeInputs(world,plusF,Matrix{}),false)[0]);
    const auto minus=observed(dispatch(device,world,makeInputs(world,minusF,Matrix{}),false)[0]);
    std::array<Vec,4> fd{};
    for(unsigned n=0;n<4;++n)for(unsigned i=0;i<3;++i)fd[n][i]=(plus[n][i]-minus[n][i])/(2*epsilon);
    const double tangentFdError=relativeForceDifference(fd,actualTangent);
    const double loadedMagnitude=vectorMax(actual);
    need(loadedMagnitude>100.0*floorBound,
        material.name+" loaded force is not well separated from rotated FP32 no-load floor");
    need(loadedOracleError<2e-4,
        material.name+" production force differs from independent StVK oracle: "+number(loadedOracleError));
    need(tangentOracleError<5e-4,
        material.name+" production tangent differs from independent StVK oracle: "+number(tangentOracleError));
    need(tangentFdError<4e-3,
        material.name+" production tangent differs from force finite difference: "+number(tangentFdError));

    std::cout<<material.name<<" identity_no_load_force_N="<<vectorMax(noLoadForces)
        <<" rigid_rotation_no_load_force_N="<<rotatedFloor
        <<" fp32_force_floor_bound_N="<<floorBound
        <<" rigid_rotation_oracle_error_N="<<rotatedError
        <<" identity_oracle_error_N="<<identityError
        <<" loaded_force_N="<<loadedMagnitude
        <<" force_oracle_error="<<loadedOracleError
        <<" tangent_oracle_error="<<tangentOracleError
        <<" tangent_fd_error="<<tangentFdError<<'\n';
}

} // namespace

int main(int argc,char** argv) {@autoreleasepool {
    try {
        need(argc==3,"usage: cardboard-gpu-check LINER.nmatter MEDIUM_ORTHOTROPIC.nmatter");
        Device device;
        std::cout<<"cardboard_gpu_device="<<[[device.device name] UTF8String]
                 <<" abi="<<NM_MATTER_ABI_VERSION<<'\n';
        checkOne(device,argv[1],5.7e9);
        checkOne(device,argv[2],5.3e9);
        std::cout<<"PASS cardboard_gpu_check checks="<<checks
            <<" production_kernels=FEM_internal_forces+apply_operator physical_steps=0 "
            <<"crease_plasticity_and_cardboard_qualification=false\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"FAIL cardboard_gpu_check checks="<<checks<<" error="<<error.what()<<'\n';
        return 1;
    }
}}
