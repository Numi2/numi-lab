#define main saturation_audit_main
#include "probe.cpp"
#undef main

// The bounded motor is a scalar canonical IR row, not a replacement host
// integrator. The native world owns the mass operator, impulse solve and q/v.
MRConstraintIRRowGPU motorRow(float k,float d,float error,float limit,float h){
    MRConstraintIRRowGPU row{};const float denom=std::fma(h,k,d);
    require(std::isfinite(denom)&&denom>0&&limit>=0&&h>0,"Invalid bounded motor constitutive input");
    row.direction={1,0,0,0};row.targetVelocity=-k*error/denom;
    row.dissipation=1/denom;row.impulseLower=-h*limit;row.impulseUpper=h*limit;
    return row;
}
int main(int argc,char**argv){try{
    require(argc==3,"Usage: bounded-probe METALLIB NEW_TSV");
    const auto base=model();const auto&d=base.dofs[0];
    const std::array<float,16>q{0,.1f,0,0,0,0,0,.1f,.03f,-.03f,-.07f,.07f,.02f,-.02f,-.14f,.14f},
      v{0,0,.002f,0,0,-.3f,.3f,.02f,.1f,-.1f,.4f,-.4f,-.6f,.6f,.7f,-.7f},
      target{0,.10005f,0,.4f,-.4f,.4f,-.4f,.1f,.05f,-.05f,-.04f,.04f,.33f,-.33f,.29f,-.29f};
    MetalWorldConfig cfg;cfg.metallibPath=argv[1];MetalWorldContext context(cfg);
    std::ofstream out(argv[2]);require(bool(out),"Cannot create bounded drive evidence");
    out<<std::setprecision(17)<<"path\tsubstep_equivalent\tlane\th\tq\tv\ttarget\tmass\tk\td\tlimit\tfinal_q\tfinal_v\tmotor_impulse\tstop_impulse\taccepted\n";
    unsigned accepted=0;
    for(unsigned stop=0;stop<2;++stop)for(unsigned path=1;path<=2;++path)for(unsigned cadence:{16u,32u,64u,128u})for(unsigned lane=0;lane<q.size();++lane){
        if(stop&&lane!=3&&lane!=4)continue;
        auto m=base;const float h=.02f/cadence;
        const auto motor=motorRow(d.drive.x,d.drive.y,q[lane]-target[lane],d.limits.w,h);
        const auto motorEndpoint=makeConstraintIRGeneralizedEndpoint(0,MR_INVALID_INDEX,0,0,1);
        auto stopRow=MRConstraintIRRowGPU{};stopRow.direction={1,0,0,0};stopRow.impulseLower=0;stopRow.impulseUpper=MR_CONSTRAINT_IR_UNBOUNDED;stopRow.flags=MR_CONSTRAINT_IR_ROW_UNILATERAL;
        const auto stopEndpoint=makeConstraintIRGeneralizedEndpoint(0,MR_INVALID_INDEX,0,0,-1);
        const std::array<float,1> warm{0};
        const std::array<ConstraintIRSourceBlock,2>blocks{{
          {.key={{1,0,0,0}},.type=MR_CONSTRAINT_BILATERAL,.endpoints=std::span(&motorEndpoint,1),.rows=std::span(&motor,1),.warmImpulses=warm},
          {.key={{2,0,0,0}},.type=MR_CONSTRAINT_BILATERAL,.endpoints=std::span(&stopEndpoint,1),.rows=std::span(&stopRow,1),.warmImpulses=warm}
        }};
        auto ir=compileConstraintIR(std::span(blocks).first(stop?2:1));require(ir.succeeded(),"Bounded motor IR invalid: "+ir.diagnostics.message);m.constraintProgram=std::move(ir.ir);
        CompiledWorld world;const auto compiled=compileMetalWorld(m,0,world);require(compiled.succeeded(),"Bounded motor world invalid: "+compiled.message);
        const std::array<float,1> qi{q[lane]},vi{v[lane]},zero{0};
        const MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=qi,.initialV=vi,.efforts=zero};
        MetalWorldStepConfig step;step.timestepSeconds=h;step.physicsSubsteps=1;step.applyBodyDamping=false;
        step.solverMode=MetalWorldSolverMode::temporalCone;step.ccdMode=path==2?MetalWorldCCDMode::hybrid:MetalWorldCCDMode::disabled;
        step.actuationMode=MetalWorldActuationMode::effort;step.captureContactEvidence=true;
        MetalWorldResult result;const auto r=context.run(world,batch,step,result);
        require(r.succeeded()&&!r.failedStepCount&&r.successfulStepCount==1,"Bounded native step rejected: "+r.message);
        require(result.contactEvidence.contacts.size()>=1+(stop?1:0),"Bounded motor impulses not published");
        const auto& contacts=result.contactEvidence.contacts;
        out<<path+2*stop<<'\t'<<cadence<<'\t'<<lane<<'\t'<<h<<'\t'<<q[lane]<<'\t'<<v[lane]<<'\t'<<target[lane]<<'\t'<<double(m.bodies[1].inertiaRow2.z)+d.drive.z<<'\t'<<d.drive.x<<'\t'<<d.drive.y<<'\t'<<d.limits.w<<'\t'<<result.finalQ[0]<<'\t'<<result.finalV[0]<<'\t'<<contacts[0].impulses.x<<'\t'<<(stop?contacts[1].impulses.x:0)<<'\t'<<r.successfulStepCount<<'\n';++accepted;
    }
    require(accepted==144&&bool(out),"Incomplete bounded motor evidence");std::cout<<"144 bounded native motor/stop one-microstep observations completed\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
