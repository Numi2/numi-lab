#define main cardboard_paper_material_legacy_main
#include "/Users/home/cardboard-implementation-20261014/snapshots/source-valid-summary-tet858-121/matter/tools/cardboard_paper_material_check.cpp"
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
std::array<float,32> exactAccepted{};
struct Projection {bool valid=false;std::string failure;unsigned iter=0;float r=0,tol=0,corr=0;std::array<float,32> hi{},lo{};};
Projection projectProduction(const NMMaterialGPU& m,const std::vector<NMScalarProgramGPU>& programs,
    const std::vector<NMExpressionInstructionGPU>& instructions,const std::vector<float>& params,
    const std::array<float,9>& f,const std::array<float,9>& low,const std::array<float,9>& rate,
    float dt,float temp) {
    const int meritMode=std::getenv("CARDBOARD_MERIT_MODE")?std::atoi(std::getenv("CARDBOARD_MERIT_MODE")):1;Projection out;unsigned n=m.stateCount;std::array<float,32> old=exactAccepted;
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
        float scaledNorm=0;std::array<float,32> scales{};std::array<float,32> rows{};
        if(scaled){float floor=params.at(m.parameterOffset+26u);
            for(unsigned c=0;c<n;++c)scales[c]=std::max({std::abs(old[c]),std::abs(out.hi[c]),floor});
            for(unsigned r=0;r<n;++r){float q=0;for(unsigned c=0;c<n;++c)q+=std::abs(jac[r*n+c])*scales[c];
                if(!(q>0)||!std::isfinite(q)){out.failure="scaled_row_"+std::to_string(r);return out;}rows[r]=q;scaledNorm=std::max(scaledNorm,std::abs(residual[r])/q);}
            for(unsigned r=0;r<n;++r){for(unsigned c=0;c<n;++c)jac[r*n+c]=jac[r*n+c]*scales[c]/rows[r];residual[r]=-residual[r]/rows[r];}}
        else for(unsigned r=0;r<n;++r)residual[r]=-residual[r];
        if(!solveLocal(jac.data(),residual.data(),n)){out.failure="local_solve_failed";return out;}
        float cn=0;if(scaled)for(unsigned c=0;c<n;++c){cn=std::max(cn,std::abs(residual[c]));residual[c]*=scales[c];}
        out.corr=cn;if(rn<=out.tol&&scaled&&cn<=2e-8f){out.valid=true;return out;}
        std::cerr<<std::setprecision(17)<<"ITER "<<it<<" raw "<<rn<<" bound "<<out.tol<<" corr "<<cn<<" scaled "<<scaledNorm<<"\n";float alpha=1;bool accepted=false;
        for(unsigned bt=0;bt<16;++bt){auto th=out.hi,tl=out.lo;
            for(unsigned s=0;s<n;++s){auto q=mrCompensatedAdd({out.hi[s],out.lo[s]},{alpha*residual[s],0});th[s]=q.high;tl[s]=q.low;}
            NMPaperCandidateState trial{th.data(),tl.data()};float tr=0,ts=0;bool good=true;
            for(unsigned row=0;row<n;++row){bool ok=true;float v=evalProgram(m.implicitResidualProgramOffset+row,programs,instructions,
                params.data()+m.parameterOffset,old.data(),trial,f.data(),low.data(),rate.data(),dt,temp,ok);good=good&&ok;tr=std::max(tr,std::abs(v));if(scaled)ts=std::max(ts,std::abs(v)/rows[row]);}
            const bool rawAccept=tr<=(1-1e-4f*alpha)*rn;const bool scaledAccept=ts<scaledNorm&&ts<=(1-1e-4f*alpha)*scaledNorm;const bool rawGuard=rn<=out.tol?tr<=out.tol:rawAccept;const bool accept=good&&(meritMode==0?rawAccept:(meritMode==1?scaledAccept:(meritMode==2?(scaledAccept&&rawGuard):(rn>out.tol?rawAccept:scaledAccept))));std::cerr<<"TRIAL "<<it<<" "<<bt<<" alpha "<<alpha<<" raw "<<tr<<" scaled "<<ts<<" rawAccept "<<rawAccept<<" scaledAccept "<<scaledAccept<<" rawGuard "<<rawGuard<<" accepted "<<accept<<"\n";if(accept){out.hi=th;out.lo=tl;accepted=true;break;}alpha*=0.5f;}
        if(!accepted){out.failure="armijo_no_trial_iter_"+std::to_string(it);return out;}}
    out.failure="iteration_budget_exhausted";return out;
}
}


int main(int argc,char**argv) {
 try {
  if(argc!=3)throw std::runtime_error("package exact44-float-input");
  numi::matter::CompiledWorld w;std::string error;
  if(!numi::matter::readMatterPackage(argv[1],w,nullptr,&error))throw std::runtime_error(error);
  std::array<float,44> x{};std::ifstream in(argv[2],std::ios::binary);in.read(reinterpret_cast<char*>(x.data()),sizeof(x));if(!in)throw std::runtime_error("input");
  std::array<float,9> f{},lo{},rate{};for(unsigned i=0;i<9;++i){f[i]=x[i];lo[i]=x[9+i];rate[i]=x[18+i];}for(unsigned i=0;i<15;++i)exactAccepted[i]=x[27+i];
  std::vector<float> p;for(const auto& q:w.parameters)p.push_back(q.valueAndBounds.x);
  const auto& m=w.materials.at(1);const auto q=projectProduction(m,w.scalarPrograms,w.instructions,p,f,lo,rate,x[42],x[43]);
  NMPaperCandidateState state{q.hi.data(),q.lo.data()};float finalResidual=0;bool ok=true;
  for(unsigned i=0;i<15;++i)finalResidual=std::max(finalResidual,std::abs(evalProgram(m.implicitResidualProgramOffset+i,w.scalarPrograms,w.instructions,p.data()+m.parameterOffset,exactAccepted.data(),state,f.data(),lo.data(),rate.data(),x[42],x[43],ok)));
  std::cout<<std::setprecision(17)<<"valid="<<q.valid<<" iterations="<<q.iter<<" pre_update_residual="<<q.r<<" final_residual="<<finalResidual<<" bound="<<q.tol<<" scaled_correction="<<q.corr<<" failure="<<q.failure<<"\nstate=";
  for(unsigned i=0;i<15;++i)std::cout<<q.hi[i]<<"+"<<q.lo[i]<<",";std::cout<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}
}
