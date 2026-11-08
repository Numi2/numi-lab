#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace metalrobo;
namespace {
void need(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
MRBodyPropertiesGPU body(unsigned owner,unsigned parent,unsigned joint,float inertia){
    MRBodyPropertiesGPU b{};b.articulationIndex=owner;b.parentBody=parent;b.inboundJoint=joint;b.motionType=MR_MOTION_DYNAMIC;
    b.massAndInverseMass={.004f,250,0,0};b.inertiaRow0={inertia,0,0,0};b.inertiaRow1={0,inertia,0,0};b.inertiaRow2={0,0,inertia,0};
    b.inverseInertiaRow0={1/inertia,0,0,0};b.inverseInertiaRow1={0,1/inertia,0,0};b.inverseInertiaRow2={0,0,1/inertia,0};b.dampingAndSpeedLimits={0,0,100000,100000};return b;
}
EngineModel mechanism(unsigned variant){
    auto m=makeFreeSphereEngineModel();m.name="bounded_drive_native_constitutive_probe";
    m.articulations.clear();m.joints.clear();m.dofs.clear();m.bodies.clear();m.shapes.clear();
    const unsigned n=variant==0?1:variant==4?40:2;const bool independent=variant==2||variant==4;
    m.defaultQ.assign(n,0);m.defaultV.assign(n,0);
    for(unsigned owner=0;owner<(independent?n:1);++owner){
        MRArticulationGPU a{};a.rootBody=independent?2*owner:0;a.firstBody=a.rootBody;a.rootType=MR_ROOT_FIXED;
        a.firstJoint=independent?owner:0;a.bodyCount=independent?2:n+1;a.jointCount=independent?1:n;
        a.nq=a.nv=independent?1:n;a.qOffset=a.vOffset=independent?owner:0;m.articulations.push_back(a);
        m.bodies.push_back(body(owner,MR_INVALID_INDEX,MR_INVALID_INDEX,.01f));
        for(unsigned local=0;local<a.jointCount;++local){
            const unsigned i=a.firstJoint+local;const unsigned parent=independent?2*owner:local;
            m.bodies.push_back(body(owner,parent,i,i==0?4e-6f:2.5e-6f));
            MRJointDescriptorGPU j{};j.parentBody=parent;j.childBody=parent+1;
            j.jointType=MR_JOINT_REVOLUTE;j.nq=j.nv=1;j.qOffset=j.vOffset=i;
            j.axis0={0,0,1,0};j.parentRotation=j.childRotation={0,0,0,1};m.joints.push_back(j);
            MRDofPropertiesGPU d{};d.articulationIndex=owner;d.jointIndex=d.qIndex=d.vIndex=i;
            d.flags=MR_DOF_FLAG_ACTUATED|MR_DOF_FLAG_DRIVE|MR_DOF_FLAG_EFFORT_LIMIT;
            d.limits={0,0,0,i==0?.01f:.006f};d.drive={i==0?.30f:.17f,i==0?.015f:.008f,i==0?2e-6f:1e-6f,variant==3?.0008f:0};m.dofs.push_back(d);
        }
    }
    m.world.bodyCount=m.bodies.size();m.world.articulationCount=m.articulations.size();m.world.jointCount=m.joints.size();
    m.world.shapeCount=0;m.world.nq=m.world.nv=n;m.world.gravityAndTimestep={0,0,0,.001f};
    std::string reason;need(m.valid(&reason),"Invalid native mechanism: "+reason);return m;
}
std::vector<double> mass(const EngineModel& m){
    const unsigned n=m.world.nv;std::vector<double> out(n*n,0),q(m.world.nq,0);
    ArticulatedDynamicsConfig c;c.gravity={0,0,0};c.applyBodyDamping=false;
    for(unsigned owner=0;owner<m.articulations.size();++owner){const auto& a=m.articulations[owner];
        std::vector<double> local(a.nv*a.nv);auto r=computeArticulatedMassMatrix(m,owner,std::span(q).subspan(a.qOffset,a.nq),local,c);need(r.succeeded(),"FP64 mass rejected");
        for(unsigned i=0;i<a.nv;++i)for(unsigned j=0;j<a.nv;++j)out[(a.vOffset+i)*n+a.vOffset+j]=local[i*a.nv+j];
    }return out;
}
std::vector<double> solve(std::vector<double> a,std::vector<double> b){
    const unsigned n=b.size();need(n<=2,"Reference supports one/two DoFs only");
    if(n==1){need(a[0]>0,"Nonpositive reference mass");return {b[0]/a[0]};}
    const double det=a[0]*a[3]-a[1]*a[2];need(det>0,"Non-SPD reference mass");
    return {(a[3]*b[0]-a[1]*b[1])/det,(a[0]*b[1]-a[2]*b[0])/det};
}
double soft(double x,double f){return std::copysign(std::max(std::abs(x)-f,0.),x);}
std::vector<double> reference(const EngineModel& m,const std::vector<double>& M,const double h,const std::vector<float>& q,const std::vector<float>& v,const std::vector<float>& target,const bool bounded){
    const unsigned n=m.world.nv;std::vector<double>b(n),c(n),cap(n),f(n),rhs(n);
    for(unsigned i=0;i<n;++i){const auto&d=m.dofs[i];b[i]=double(d.drive.x)*(double(target[i])-q[i]-h*v[i])-double(d.drive.y)*v[i];c[i]=h*(d.drive.y+h*d.drive.x);cap[i]=d.limits.w;
        if(std::abs(v[i])>1e-4)b[i]-=std::copysign(double(d.drive.w),double(v[i]));else f[i]=d.drive.w;
    }
    if(!bounded){auto A=M;for(unsigned i=0;i<n;++i){A[i*n+i]+=c[i];rhs[i]=std::clamp(soft(b[i],f[i]),-cap[i],cap[i]);}return solve(A,rhs);}
    // Independently enumerate the five motor regions: negative saturation,
    // negative linear, stationary dead zone, positive linear, positive saturation.
    // This is a small FP64 reference, never a physical state update authority.
    unsigned combinations=1;for(unsigned i=0;i<n;++i)combinations*=5;
    for(unsigned pattern=0;pattern<combinations;++pattern){unsigned digits=pattern;auto A=M;
        for(unsigned i=0;i<n;++i){const int region=int(digits%5)-2;digits/=5;
            rhs[i]=region==-2?-cap[i]:region==2?cap[i]:region==0?0:b[i]-std::copysign(f[i],double(region));
            if(region==-1||region==1)A[i*n+i]+=c[i];
        }
        const auto a=solve(A,rhs);bool valid=true;
        for(unsigned i=0;i<n;++i){double tau=0;for(unsigned j=0;j<n;++j)tau+=M[i*n+j]*a[j];
            valid&=std::abs(tau-std::clamp(soft(b[i]-c[i]*a[i],f[i]),-cap[i],cap[i]))<1e-11;
        }if(valid)return a;
    }throw std::runtime_error("No valid FP64 motor active set");
}
const std::array<float,8> q0{0,.1f,0,0,-.03f,.07f,-.02f,.14f},v0{0,0,.002f,0,-.1f,-.4f,.6f,-.7f},t0{0,.10005f,0,.4f,-.05f,.04f,-.33f,-.29f};
const std::array<float,8> q1{.013f,-.027f,.041f,-.063f,.029f,-.017f,.081f,-.11f},v1{.051f,-.083f,.137f,-.211f,0,0,-.31f,.43f},t1{-.37f,.089f,.04103f,.063f,.02901f,-.01701f,.22f,-.32f};
void calibrate(){double massError=0,kktError=0;unsigned cases=0;
    for(unsigned variant=0;variant<4;++variant){const auto m=mechanism(variant);const auto M=mass(m);const unsigned n=m.world.nv;
        CompiledWorld world;const auto compiled=compileMetalWorld(m,0,world);need(compiled.succeeded() && world.nv()==n && world.eligiblePairCount()==0 && world.sceneBodyCount()==0,"Native host instrument calibration failed: "+compiled.message);
        const double i1=m.bodies[1].inertiaRow2.z,a1=m.dofs[0].drive.z;
        std::vector<double> expected{ i1+a1 };
        if(n==2){const double i2=m.bodies.back().inertiaRow2.z,a2=m.dofs[1].drive.z;expected=variant==2?std::vector<double>{i1+a1,0,0,i2+a2}:std::vector<double>{i1+i2+a1,i2,i2,i2+a2};}
        for(unsigned i=0;i<M.size();++i)massError=std::max(massError,std::abs(M[i]-expected[i]));
        for(unsigned lane=0;lane<8;++lane)for(unsigned cadence:{16u,32u,64u,128u}){
            std::vector<float>q{q0[lane]},v{v0[lane]},t{t0[lane]};if(n==2){q.push_back(q1[lane]);v.push_back(v1[lane]);t.push_back(t1[lane]);}
            const double h=float(.02f/cadence);const auto a=reference(m,M,h,q,v,t,true);
            for(unsigned i=0;i<n;++i){const auto&d=m.dofs[i];double torque=0;for(unsigned j=0;j<n;++j)torque+=M[i*n+j]*a[j];
                double raw=d.drive.x*(double(t[i])-q[i]-h*(v[i]+h*a[i]))-d.drive.y*(v[i]+h*a[i]);
                raw=std::abs(v[i])>1e-4?raw-std::copysign(double(d.drive.w),double(v[i])):soft(raw,d.drive.w);
                kktError=std::max(kktError,std::abs(torque-std::clamp(raw,-double(d.limits.w),double(d.limits.w))));
            }++cases;
        }
    }
    need(massError<1e-14 && kktError<1e-9,"Reference calibration failed");
    std::cout<<std::setprecision(17)<<"{\"mass_error\":"<<massError<<",\"constitutive_error\":"<<kktError<<",\"conditions\":"<<cases<<"}\n";
}
void execute(const char* shader,const char* output){
    MetalWorldConfig cfg;cfg.metallibPath=shader;MetalWorldContext context(cfg);
    std::ofstream out(output);need(bool(out),"Cannot create new native evidence");
    out<<std::setprecision(17)<<"variant\tbounded\thybrid\tcadence\tlane\tdof\th\tq\tv\ttarget\tfinal_q\tfinal_v\treference_v\tmotor_torque\treference_torque\taccepted\taba_implicit\tcontact_mass_implicit\n";
    unsigned accepted=0;
    for(unsigned variant=0;variant<4;++variant){const auto m=mechanism(variant);const auto M=mass(m);const unsigned n=m.world.nv;
        CompiledWorld world;const auto c=compileMetalWorld(m,0,world);need(c.succeeded(),"Native world compilation failed: "+c.message);
        std::vector<float>q(8*n),v(8*n),target(8*n);
        for(unsigned lane=0;lane<8;++lane){q[lane*n]=q0[lane];v[lane*n]=v0[lane];target[lane*n]=t0[lane];if(n==2){q[lane*n+1]=q1[lane];v[lane*n+1]=v1[lane];target[lane*n+1]=t1[lane];}}
        const MetalWorldBatch batch{.environmentCount=8,.controlStepCount=1,.initialQ=q,.initialV=v,.efforts=target};
        for(unsigned bounded=0;bounded<2;++bounded)for(unsigned hybrid=0;hybrid<2;++hybrid)for(unsigned cadence:{16u,32u,64u,128u}){
            MetalWorldStepConfig s;s.timestepSeconds=.02f/cadence;s.physicsSubsteps=1;s.applyBodyDamping=false;
            s.solverMode=MetalWorldSolverMode::temporalCone;s.ccdMode=hybrid?MetalWorldCCDMode::hybrid:MetalWorldCCDMode::disabled;
            s.actuationMode=bounded?MetalWorldActuationMode::boundedPositionDrive:MetalWorldActuationMode::implicitPositionDrive;
            MetalWorldResult result;const auto r=context.run(world,batch,s,result);
            need(r.succeeded() && r.dispatched && r.published && r.failedStepCount==0 && r.successfulStepCount==8,"Native motor invocation failed: "+r.message+" GPU="+std::to_string(r.firstGPUStatusCode)+" env="+std::to_string(r.firstFailingEnvironment));accepted+=8;std::cout<<"device="<<r.deviceName<<" variant="<<variant<<" bounded="<<bounded<<" hybrid="<<hybrid<<" cadence="<<cadence<<" gpu_ms="<<r.gpuElapsedMilliseconds<<"\n";
            for(unsigned lane=0;lane<8;++lane){std::vector<float>lq(q.begin()+lane*n,q.begin()+(lane+1)*n),lv(v.begin()+lane*n,v.begin()+(lane+1)*n),lt(target.begin()+lane*n,target.begin()+(lane+1)*n);
                const auto a=reference(m,M,s.timestepSeconds,lq,lv,lt,bounded);
                for(unsigned i=0;i<n;++i){double tau=0,refTau=0;for(unsigned j=0;j<n;++j){tau+=M[i*n+j]*(result.finalV[lane*n+j]-lv[j])/s.timestepSeconds;refTau+=M[i*n+j]*a[j];}
                    out<<variant<<'\t'<<bounded<<'\t'<<hybrid<<'\t'<<cadence<<'\t'<<lane<<'\t'<<i<<'\t'<<s.timestepSeconds<<'\t'<<lq[i]<<'\t'<<lv[i]<<'\t'<<lt[i]<<'\t'<<result.finalQ[lane*n+i]<<'\t'<<result.finalV[lane*n+i]<<'\t'<<lv[i]+s.timestepSeconds*a[i]<<'\t'<<tau<<'\t'<<refTau<<'\t'<<8<<'\t'<<((r.layout.abaDispatch.flags&MR_ABA_IMPLICIT_DRIVES)!=0)<<'\t'<<((r.layout.factorDispatches.front().flags&MR_ARTICULATED_OPERATOR_IMPLICIT_DRIVES)!=0)<<'\n';
                }
            }
        }
    }
    // Exercise every declared generalized-velocity slot, including columns
    // 32..39 which share SIMD lanes with the first eight response columns.
    const auto wide=mechanism(4);CompiledWorld wideWorld;
    const auto wc=compileMetalWorld(wide,0,wideWorld);need(wc.succeeded(),"40-DoF world compile failed: "+wc.message);
    std::vector<float>q(40,0),v(40,0),target(40);
    for(unsigned i=0;i<40;++i)target[i]=i%2?.4f:-.4f;
    const MetalWorldBatch wideBatch{.environmentCount=1,.controlStepCount=1,.initialQ=q,.initialV=v,.efforts=target};
    for(unsigned hybrid=0;hybrid<2;++hybrid){MetalWorldStepConfig s;s.timestepSeconds=.02f/32;s.physicsSubsteps=1;s.applyBodyDamping=false;s.solverMode=MetalWorldSolverMode::temporalCone;s.ccdMode=hybrid?MetalWorldCCDMode::hybrid:MetalWorldCCDMode::disabled;s.actuationMode=MetalWorldActuationMode::boundedPositionDrive;
        MetalWorldResult result;const auto r=context.run(wideWorld,wideBatch,s,result);need(r.succeeded()&&r.successfulStepCount==1&&!r.failedStepCount,"40-DoF native prediction failed: "+r.message);++accepted;
        for(unsigned i=0;i<40;++i){const auto&d=wide.dofs[i];const double inertia=double(wide.bodies[2*i+1].inertiaRow2.z)+d.drive.z;const double h=s.timestepSeconds;
            const double rv=std::clamp(h*d.drive.x*target[i]/(inertia+h*d.drive.y+h*h*d.drive.x),-h*d.limits.w/inertia,h*d.limits.w/inertia);
            out<<4<<'\t'<<1<<'\t'<<hybrid<<'\t'<<32<<'\t'<<0<<'\t'<<i<<'\t'<<h<<'\t'<<0<<'\t'<<0<<'\t'<<target[i]<<'\t'<<result.finalQ[i]<<'\t'<<result.finalV[i]<<'\t'<<rv<<'\t'<<inertia*result.finalV[i]/h<<'\t'<<inertia*rv/h<<'\t'<<1<<'\t'<<((r.layout.abaDispatch.flags&MR_ABA_IMPLICIT_DRIVES)!=0)<<'\t'<<((r.layout.factorDispatches.front().flags&MR_ARTICULATED_OPERATOR_IMPLICIT_DRIVES)!=0)<<'\n';
        }
    }
    need(accepted==514&&bool(out),"Incomplete native motor study");std::cout<<"514 accepted native motor microsteps completed\n";
}
}
int main(int argc,char**argv){try{if(argc==2&&std::string(argv[1])=="--calibrate"){calibrate();return 0;}need(argc==3,"Usage: bounded-drive-probe --calibrate OR METALLIB NEW_TSV");execute(argv[1],argv[2]);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
