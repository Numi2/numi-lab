from pathlib import Path
import hashlib,json,subprocess,os,struct
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');S=B/'snapshots/source-valid-summary-tet858-121'
O=B/'analysis/exact-material-merit-prototype-003';O.mkdir(exist_ok=False)
s=(B/'analysis/material-two-stage-merit-prototype-002/replay.cpp').read_text().split('int main(int argc,char**argv)')[0]
s=s.replace('struct Projection {','std::array<float,32> exactAccepted{};\nstruct Projection {')
s=s.replace('std::array<float,32> old{};', 'std::array<float,32> old=exactAccepted;')
s+=r'''
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
'''
(O/'replay.cpp').write_text(s)
cmd=['clang++','-std=c++20','-O2','-ffp-contract=off','-I'+str(S/'matter/include'),'-I'+str(S/'include'),'-I'+str(S/'matter/tools'),str(O/'replay.cpp'),str(R/'build-cardboard-provenance-121/matter/libnumi_matter_compiler.a'),'-o',str(O/'replay')]
(O/'compile-command.json').write_text(json.dumps(cmd,indent=2)+'\n');q=subprocess.run(cmd,capture_output=True,text=True);(O/'compile.log').write_text(q.stdout+q.stderr);assert q.returncode==0,q.stderr
f674=B/'native/dimensionless-material674-point-390/output/tet674-env0-source380_exact_compiled_budget32-trace.json';j=json.loads(f674.read_text())['point'];ib=j['input_f32_bits'];a=j['accepted_state_f32_bits'];u674=[int(v,16) for v in ib[:27]+a+ib[45:47]]
f858=B/'native/source392-tet858-point-401/output/point.json';j=json.loads(f858.read_text());raw=j['raw_160_f32_words'];u858=[int(v,16) for v in raw[3:30]]+[0]*15+[struct.unpack('<I',struct.pack('<f',j[k]))[0] for k in ('dt_s','temperature_K')]
results=[]
for ti,u,source in [(674,u674,f674),(858,u858,f858)]:
 assert len(u)==44;f=O/f'tet{ti}-exact-input.bin';f.write_bytes(struct.pack('<44I',*u))
 for mode in (0,1,3):
  env=os.environ.copy();env['CARDBOARD_MERIT_MODE']=str(mode)
  args=[str(O/'replay'),str(B/'solver/dimensionless-certified-coupon-392/output/compiled.nmatterpack'),str(f)]
  q=subprocess.run(args,capture_output=True,text=True,env=env);stem=f'tet{ti}-mode{mode}'
  (O/(stem+'.stdout.log')).write_text(q.stdout);(O/(stem+'.stderr.log')).write_text(q.stderr)
  results.append(dict(tet=ti,mode=mode,command=args,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),exit_code=q.returncode,result=q.stdout.splitlines()[0] if q.stdout else None))
(O/'results.json').write_text(json.dumps(results,indent=2)+'\n');(O/'prepare.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(results,indent=2))
