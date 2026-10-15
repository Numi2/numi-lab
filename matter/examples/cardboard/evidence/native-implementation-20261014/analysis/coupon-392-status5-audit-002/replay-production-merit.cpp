#define main cardboard_paper_material_legacy_main
#include "/Users/home/numi-cardboard-20261006/matter/tools/cardboard_paper_material_check.cpp"
#undef main
#include "numi/matter/compiler.hpp"
#include "numi/matter/fem_compensated_deformation.h"
#include <bit>
#include <fstream>

namespace {
constexpr unsigned KCache=NM_EXPRESSION_CACHE_CAPACITY;
float evalProgram(unsigned index,const std::vector<NMScalarProgramGPU>& programs,
    const std::vector<NMExpressionInstructionGPU>& instructions,const float* params,
    const float* accepted,const NMPaperCandidateState& candidate,const float* f,const float* low,
    const float* rate,float dt,float temp,bool& ok) {
    const auto& p=programs.at(index);
    if((p.flags&2u)==0u) throw std::runtime_error("expected compensated compiled program");
    if(p.flags&1u) return nmPaperCompensatedBytecode<KCache>(p,instructions.data(),params,
        accepted,candidate,f,nullptr,rate,dt,temp,ok,nullptr,low);
    return nmPaperCompensatedBytecode<0u>(p,instructions.data(),params,
        accepted,candidate,f,nullptr,rate,dt,temp,ok,nullptr,low);
}
bool solveLocal(float* a,float* b,unsigned n) {
    for(unsigned c=0;c<n;++c){unsigned pivot=c;float pm=std::abs(a[c*n+c]);
        for(unsigned r=c+1;r<n;++r)if(std::abs(a[r*n+c])>pm){pivot=r;pm=std::abs(a[r*n+c]);}
        if(!(pm>1e-8f)||!std::isfinite(pm))return false;
        if(pivot!=c){for(unsigned j=c;j<n;++j)std::swap(a[c*n+j],a[pivot*n+j]);std::swap(b[c],b[pivot]);}
        float d=a[c*n+c];for(unsigned r=c+1;r<n;++r){float f=a[r*n+c]/d;a[r*n+c]=0;
            for(unsigned j=c+1;j<n;++j)a[r*n+j]-=f*a[c*n+j];b[r]-=f*b[c];}}
    for(int r=int(n)-1;r>=0;--r){float v=b[r];for(unsigned c=unsigned(r)+1;c<n;++c)v-=a[unsigned(r)*n+c]*b[c];
        b[r]=v/a[unsigned(r)*n+unsigned(r)];if(!std::isfinite(b[r]))return false;}return true;
}
struct Projection {bool valid=false;std::string failure;unsigned iter=0;float r=0,tol=0,corr=0;std::array<float,32> hi{},lo{};};
Projection projectProduction(const NMMaterialGPU& m,const std::vector<NMScalarProgramGPU>& programs,
    const std::vector<NMExpressionInstructionGPU>& instructions,const std::vector<float>& params,
    const std::array<float,9>& f,const std::array<float,9>& low,const std::array<float,9>& rate,
    float dt,float temp) {
    Projection out;unsigned n=m.stateCount;std::array<float,32> old{};
    NMPaperCandidateState state{out.hi.data(),out.lo.data()};
    for(unsigned s=0;s<n;++s){bool ok=true;out.hi[s]=evalProgram(m.stateUpdateProgramOffset+s,programs,instructions,
        params.data()+m.parameterOffset,old.data(),state,f.data(),low.data(),rate.data(),dt,temp,ok);
        if(!ok){out.failure="state_update_"+std::to_string(s);return out;}}
    const bool scaled=(m.flags&NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION)!=0u;
    const float tol=(m.flags&(NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC|NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION))?2e-8f:2e-5f;
    for(unsigned it=0;it<m.localNewtonIterations;++it){std::array<float,32> residual{};std::array<float,1024> jac{};float rn=0,sn=0;
        for(unsigned row=0;row<n;++row){bool ok=true;residual[row]=evalProgram(m.implicitResidualProgramOffset+row,programs,instructions,
            params.data()+m.parameterOffset,old.data(),state,f.data(),low.data(),rate.data(),dt,temp,ok);
            if(!ok){out.failure="residual_invalid_"+std::to_string(row);return out;}
            rn=std::max(rn,std::abs(residual[row]));sn=std::max(sn,std::abs(out.hi[row]));
            for(unsigned col=0;col<n;++col){jac[row*n+col]=evalProgram(m.implicitJacobianProgramOffset+row*n+col,
                programs,instructions,params.data()+m.parameterOffset,old.data(),state,f.data(),low.data(),rate.data(),dt,temp,ok);
                if(!ok){out.failure="jacobian_invalid_"+std::to_string(row)+"_"+std::to_string(col);return out;}}}
        out.iter=it+1;out.r=rn;out.tol=tol*(1+sn);
        if(rn<=out.tol&&!scaled){out.valid=true;return out;}
        std::array<float,32> scales{};std::array<float,32> rows{};float scaledRn=0.0f;
        if(scaled){float floor=params.at(m.parameterOffset+26u);
            for(unsigned c=0;c<n;++c)scales[c]=std::max({std::abs(old[c]),std::abs(out.hi[c]),floor});
            for(unsigned r=0;r<n;++r){float q=0;for(unsigned c=0;c<n;++c)q+=std::abs(jac[r*n+c])*scales[c];
                if(!(q>0)||!std::isfinite(q)){out.failure="scaled_row_"+std::to_string(r);return out;}rows[r]=q;
                scaledRn=std::max(scaledRn,std::abs(residual[r])/q);}
            for(unsigned r=0;r<n;++r){for(unsigned c=0;c<n;++c)jac[r*n+c]=jac[r*n+c]*scales[c]/rows[r];residual[r]=-residual[r]/rows[r];}}
        else for(unsigned r=0;r<n;++r)residual[r]=-residual[r];
        if(!solveLocal(jac.data(),residual.data(),n)){out.failure="local_solve_failed";return out;}
        float cn=0;if(scaled)for(unsigned c=0;c<n;++c){cn=std::max(cn,std::abs(residual[c]));residual[c]*=scales[c];}
        out.corr=cn;if(rn<=out.tol&&scaled&&cn<=2e-8f){out.valid=true;return out;}
        float alpha=1;bool accepted=false;
        for(unsigned bt=0;bt<16;++bt){auto th=out.hi,tl=out.lo;
            for(unsigned s=0;s<n;++s){auto q=mrCompensatedAdd({out.hi[s],out.lo[s]},{alpha*residual[s],0});th[s]=q.high;tl[s]=q.low;}
            NMPaperCandidateState trial{th.data(),tl.data()};float tr=0,trialScaled=0;bool good=true;
            for(unsigned row=0;row<n;++row){bool ok=true;float v=evalProgram(m.implicitResidualProgramOffset+row,programs,instructions,
                params.data()+m.parameterOffset,old.data(),trial,f.data(),low.data(),rate.data(),dt,temp,ok);good=good&&ok;tr=std::max(tr,std::abs(v));
                if(scaled)trialScaled=std::max(trialScaled,std::abs(v)/rows[row]);}
            const bool decrease=scaled
                ? (trialScaled<scaledRn&&trialScaled<=(1-1e-4f*alpha)*scaledRn)
                : (tr<=(1-1e-4f*alpha)*rn);
            if(good&&decrease){out.hi=th;out.lo=tl;accepted=true;break;}alpha*=0.5f;}
        if(!accepted){out.failure="armijo_no_trial_iter_"+std::to_string(it);return out;}}
    // Source-392's status-5 point records the residual of the last accepted
    // trial state, whereas the loop's final iteration diagnostic above is the
    // residual before that last update. Re-evaluate the terminal candidate so
    // the CPU report compares the same state meaning as native401.
    NMPaperCandidateState terminal{out.hi.data(),out.lo.data()};
    float terminalResidual=0.0f,terminalStateNorm=0.0f;
    for(unsigned stateIndex=0u;stateIndex<n;++stateIndex)
        terminalStateNorm=std::max(terminalStateNorm,std::abs(out.hi[stateIndex]));
    for(unsigned row=0u;row<n;++row){bool ok=true;const float value=evalProgram(
        m.implicitResidualProgramOffset+row,programs,instructions,
        params.data()+m.parameterOffset,old.data(),terminal,f.data(),low.data(),rate.data(),dt,temp,ok);
        if(!ok){out.failure="terminal_residual_invalid_"+std::to_string(row);return out;}
        terminalResidual=std::max(terminalResidual,std::abs(value));}
    out.r=terminalResidual;out.tol=tol*(1+terminalStateNorm);
    out.failure="iteration_budget_exhausted";return out;
}
}

int main(int argc,char**argv) {
    try {
        if(argc!=4) throw std::runtime_error("package candidate_nodes medium_asset");
        using namespace numi::matter;
        CompiledWorld world; std::string error;
        if(!readMatterPackage(argv[1],world,nullptr,&error)) throw std::runtime_error(error);
        std::ifstream input(argv[2],std::ios::binary);
        std::vector<NMFEMNodeStateGPU> nodes(world.dispatch.environmentCount*world.dispatch.femNodeCount);
        input.read(reinterpret_cast<char*>(nodes.data()),static_cast<std::streamsize>(nodes.size()*sizeof(NMFEMNodeStateGPU)));
        if(!input) throw std::runtime_error("candidate node read failed");
        constexpr unsigned ti=858u;
        const auto& tet=world.fem.tetrahedra.at(ti);
        const std::array<unsigned,4> ids{tet.nodes.x,tet.nodes.y,tet.nodes.z,tet.nodes.w};
        float u[12],inv[9],frame[9]; std::array<float,9> fhi{},flo{};
        for(unsigned n=0;n<4;++n) {
            const auto value=nodes.at(ids[n]).referenceDisplacementAndMode;
            u[3*n]=value.x;u[3*n+1]=value.y;u[3*n+2]=value.z;
        }
        const float* rows[3]={&tet.inverseRestRow0.x,&tet.inverseRestRow1.x,&tet.inverseRestRow2.x};
        for(unsigned r=0;r<3;++r)for(unsigned c=0;c<3;++c)inv[3*r+c]=rows[r][c];
        const auto q=tet.materialFrameRotation;
        const float invNorm=1.0f/std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
        const float x=q.x*invNorm,y=q.y*invNorm,z=q.z*invNorm,w=q.w*invNorm;
        const float R[9]={1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
            2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
            2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)};
        const bool framed=(world.objects.at(0).flags&NM_OBJECT_FEM_MATERIAL_FRAME)!=0u;
        for(unsigned i=0;i<9;++i)frame[i]=framed?R[i]:((i/3)==(i%3)?1.0f:0.0f);
        nmFEMReferenceDeformationWords(u,inv,frame,fhi.data(),flo.data());
        numi_cardboard_paper::Matrix F{};
        for(unsigned i=0;i<9;++i)F[i]=double(fhi[i])+double(flo[i]);
        const auto parsed=parseMatterFile(argv[3]);
        if(!parsed.succeeded())throw std::runtime_error("medium source parse failed");
        const auto compiled=detail::compileConstitutive(parsed.material,NM_EXPRESSION_STACK_CAPACITY);
        if(!compiled.succeeded())throw std::runtime_error("medium source compile failed");
        std::array<double,15> accepted{};
        const auto projected=projectCompiled64(compiled.program,F,accepted);
        const auto residual=compiledResiduals15(compiled.program,F,accepted,projected.state);
        std::array<float,9> rate{}; std::array<std::array<float,3>,4> vel{};
        for(unsigned n=0;n<4;++n){auto v=nodes.at(ids[n]).velocityAndInverseMass;vel[n]={v.x,v.y,v.z};}
        float dv[9]{};for(unsigned c=0;c<3;++c)for(unsigned r=0;r<3;++r)dv[3*r+c]=vel[c+1][r]-vel[0][r];
        float wr[9]{};for(unsigned r=0;r<3;++r)for(unsigned c=0;c<3;++c)for(unsigned k=0;k<3;++k)wr[3*r+c]+=dv[3*r+k]*inv[3*k+c];
        for(unsigned r=0;r<3;++r)for(unsigned c=0;c<3;++c)for(unsigned k=0;k<3;++k)rate[3*r+c]+=wr[3*r+k]*frame[3*k+c];
        std::vector<float> pvals;for(const auto& p:world.parameters)pvals.push_back(p.valueAndBounds.x);
        const auto prod=projectProduction(world.materials.at(1),world.scalarPrograms,world.instructions,pvals,fhi,flo,rate,1.0e-4f,world.materials.at(1).bulk.y);
        float prodStressMax=0;std::array<float,9> prodStress{};float acceptedFloat[32]{};if(prod.valid){NMPaperCandidateState ps{prod.hi.data(),prod.lo.data()};bool good=true;
            for(unsigned c=0;c<9;++c)prodStress[c]=evalProgram(world.materials.at(1).stressProgramOffset+c,world.scalarPrograms,world.instructions,
                pvals.data()+world.materials.at(1).parameterOffset,acceptedFloat,ps,fhi.data(),flo.data(),rate.data(),1e-4f,world.materials.at(1).bulk.y,good);
            for(float v:prodStress)prodStressMax=std::max(prodStressMax,std::abs(v));if(!good)prodStressMax=std::numeric_limits<float>::quiet_NaN();}

        double rinf=0.0, sinf=0.0;
        for(double value:residual)rinf=std::max(rinf,std::abs(value));
        for(double value:projected.firstPiolaMPa)sinf=std::max(sinf,std::abs(value));
        std::cout<<std::setprecision(17)
            <<"status5_cpu_replay=completed\n"
            <<"physics_fingerprint=0x"<<std::hex<<world.physicsFingerprint<<std::dec<<"\n"
            <<"tet=858 nodes="<<ids[0]<<","<<ids[1]<<","<<ids[2]<<","<<ids[3]
            <<" material="<<tet.identity.x<<" flags="<<tet.identity.w<<"\n"
            <<"material_flags="<<world.materials.at(1).flags
            <<" localNewtonIterations="<<world.materials.at(1).localNewtonIterations
            <<" validity="<<world.materials.at(1).validity.x<<","<<world.materials.at(1).validity.y<<","<<world.materials.at(1).validity.z<<"\n"
            <<"F_high=";for(float value:fhi)std::cout<<value<<",";
        std::cout<<"\nF_low=";for(float value:flo)std::cout<<value<<",";
        std::cout<<"\nF_sum_fp64=";for(double value:F)std::cout<<value<<",";
        std::cout<<"\naccepted_history=";for(double value:accepted)std::cout<<value<<",";
        std::cout<<"\nFP64_projected_state=";for(double value:projected.state)std::cout<<value<<",";
        std::cout<<"\nFP64_first_P_MPa=";for(double value:projected.firstPiolaMPa)std::cout<<value<<",";
        std::cout<<"\nFP64_residual_inf="<<rinf<<" FP64_max_abs_P_MPa="<<sinf
            <<" updates="<<world.materials.at(1).stateCount<<"\nrate_local=";
        for(float value:rate)std::cout<<value<<",";
        std::cout<<"\nproduction_projection_valid="<<prod.valid<<" iterations="<<prod.iter<<" residual="<<prod.r
            <<" tolerance="<<prod.tol<<" scaled_correction="<<prod.corr<<" failure="<<prod.failure<<"\nproduction_state=";
        for(unsigned i=0;i<15;++i)std::cout<<prod.hi[i]<<"+"<<prod.lo[i]<<",";
        std::cout<<"\nproduction_stress_max_Pa="<<prodStressMax<<" stress=";
        for(float value:prodStress)std::cout<<value<<",";
        std::cout<<"\n";
        return 0;
    } catch(const std::exception&e) {std::cerr<<"FAIL: "<<e.what()<<"\n";return 2;}
}
