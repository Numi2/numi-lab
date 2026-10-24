#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using V3=std::array<double,3>; using M3=std::array<std::array<double,3>,3>;
static void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
static double numberAfter(const std::string&s,size_t p){p=s.find(':',p);need(p!=s.npos,"number colon");p++;while(p<s.size()&&(s[p]==' '||s[p]=='\n'||s[p]=='\t'))p++;size_t e=p;while(e<s.size()&&(s[e]=='-'||s[e]=='+'||s[e]=='.'||(s[e]>='0'&&s[e]<='9')||s[e]=='e'||s[e]=='E'))e++;return std::stod(s.substr(p,e-p));}
static std::vector<double> arrayAfter(const std::string&s,size_t p){p=s.find('[',p);need(p!=s.npos,"array start");size_t e=s.find(']',p);need(e!=s.npos,"array end");std::string t=s.substr(p+1,e-p-1);for(char&c:t)if(c==',')c=' ';std::istringstream in(t);std::vector<double> v;double x;while(in>>x)v.push_back(x);return v;}
static M3 rotq(const double*q){double x=q[0],y=q[1],z=q[2],w=q[3];return {{{1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)},{2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)},{2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)}}};}
static M3 mm(const M3&a,const M3&b){M3 c{};for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)c[i][j]+=a[i][k]*b[k][j];return c;}
static V3 mv(const M3&a,const V3&x){V3 y{};for(int i=0;i<3;i++)for(int j=0;j<3;j++)y[i]+=a[i][j]*x[j];return y;}
struct Pose{V3 t;std::array<double,4>q;}; struct Bind{uint32_t body;V3 t;std::array<double,4>q;double scale;};
int main(int argc,char**argv){try{
 need(argc==5,"usage payload pose.json input.csv output.csv");std::ifstream bin(argv[1],std::ios::binary);need(bool(bin),"payload open");bin.seekg(0,std::ios::end);size_t sz=(size_t)bin.tellg();bin.seekg(0);std::vector<char> raw(sz);bin.read(raw.data(),sz);need(sz>60,"payload size");uint32_t nb,nv,ni;std::memcpy(&nb,raw.data()+12,4);std::memcpy(&nv,raw.data()+16,4);std::memcpy(&ni,raw.data()+20,4);need(nb==86&&nv==61950&&ni==326961,"payload dimensions");
 std::ifstream jf(argv[2]);need(bool(jf),"pose open");std::string js((std::istreambuf_iterator<char>(jf)),{});size_t p=js.find("\"source_poses\"");need(p!=js.npos,"source_poses missing");std::map<uint32_t,Pose> poses;while((p=js.find("\"core_body\"",p))!=js.npos){uint32_t id=(uint32_t)numberAfter(js,p);size_t tp=js.find("\"position\"",p),qp=js.find("\"quaternion_xyzw\"",tp);need(tp!=js.npos&&qp!=js.npos,"pose fields");auto t=arrayAfter(js,tp),q=arrayAfter(js,qp);need(t.size()==3&&q.size()==4,"pose array sizes");poses[id]={{t[0],t[1],t[2]+0.35},{q[0],q[1],q[2],q[3]}};p=qp+18;}
 std::vector<Bind> binds(nb);size_t bo=60;for(uint32_t i=0;i<nb;i++){const char*x=raw.data()+bo+i*36;Bind b{};std::memcpy(&b.body,x,4);float f[8];std::memcpy(f,x+4,32);b.t={f[0],f[1],f[2]};for(int k=0;k<4;k++)b.q[k]=f[k+3];b.scale=f[7];need(poses.count(b.body),"binding body pose absent");binds[i]=b;}
 std::vector<M3>A(nb);std::vector<V3>shift(nb);for(uint32_t i=0;i<nb;i++){auto&b=binds[i];M3 r=rotq(poses[b.body].q.data()),br=rotq(b.q.data());for(auto&row:br)for(double&v:row)v*=b.scale;A[i]=mm(r,br);V3 rt=mv(r,b.t);shift[i]={poses[b.body].t[0]+rt[0],poses[b.body].t[1]+rt[1],poses[b.body].t[2]+rt[2]};}
 std::ifstream in(argv[3]);need(bool(in),"csv open");std::ofstream out(argv[4]);need(bool(out),"output open");std::string line;need(bool(std::getline(in,line)),"csv header");out<<line<<'\n'<<std::setprecision(17);size_t count=0;while(std::getline(in,line)){std::vector<std::string> f;std::stringstream ss(line);std::string x;while(std::getline(ss,x,','))f.push_back(x);need(f.size()==20,"CSV columns");uint32_t id=(uint32_t)std::stoul(f[0]);need(id<nv,"CSV vertex range");size_t vo=60+nb*36+(size_t)id*56,wo=60+nb*36+(size_t)nv*56+(size_t)ni*4+(size_t)id*nb*4;float xyz[3],w[86];std::memcpy(xyz,raw.data()+vo,12);std::memcpy(w,raw.data()+wo,sizeof(w));V3 pos{xyz[0],xyz[1],xyz[2]},world{};for(uint32_t k=0;k<nb;k++){V3 y=mv(A[k],pos);for(int j=0;j<3;j++)world[j]+=double(w[k])*(y[j]+shift[k][j]);}std::ostringstream ssout;ssout<<std::setprecision(17)<<world[0];f[2]=ssout.str();ssout.str("");ssout.clear();ssout<<std::setprecision(17)<<world[1];f[3]=ssout.str();ssout.str("");ssout.clear();ssout<<std::setprecision(17)<<world[2];f[4]=ssout.str();for(size_t j=0;j<f.size();j++){if(j)out<<',';out<<f[j];}out<<'\n';count++;}need(count==nv,"CSV row count");std::cerr<<"wrote_rows="<<count<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
