#include "metalrobo/HumanBehaviorProgram.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

namespace metalrobo {
namespace {
void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct Reader {
    std::span<const std::byte> data; std::size_t offset=0;
    template<class T> T read() {
        static_assert(std::endian::native == std::endian::little);
        need(sizeof(T)<=data.size()-offset,"truncated behavior program");
        T result; std::memcpy(&result,data.data()+offset,sizeof(T)); offset+=sizeof(T); return result;
    }
};
std::array<double,3> rotateInverse(const std::array<double,4>& q, const std::array<double,3>& a) {
    const std::array<double,3> v{-q[0],-q[1],-q[2]};
    const std::array<double,3> t{2*(v[1]*a[2]-v[2]*a[1]),2*(v[2]*a[0]-v[0]*a[2]),2*(v[0]*a[1]-v[1]*a[0])};
    return {a[0]+q[3]*t[0]+v[1]*t[2]-v[2]*t[1],a[1]+q[3]*t[1]+v[2]*t[0]-v[0]*t[2],a[2]+q[3]*t[2]+v[0]*t[1]-v[1]*t[0]};
}
void finite(double x) { need(std::isfinite(x),"nonfinite behavior parameter"); }
void unit(const std::array<double,3>& v) { double n=0; for (double x:v) { finite(x); n+=x*x; } need(std::abs(n-1)<=1e-10,"behavior axis must be unit"); }
void pair(double x, float& high, float& low) {
    finite(x); high=static_cast<float>(x); low=static_cast<float>(x-static_cast<double>(high));
    need(std::isfinite(high)&&std::isfinite(low),"behavior parameter exceeds GPU precision range");
    need(x==0.0 || high!=0.0f || low!=0.0f,"behavior parameter underflows GPU representation");
}
void vectorPair(const std::array<double,3>& v, mr_float4& high, mr_float4& low) {
    pair(v[0],high.x,low.x); pair(v[1],high.y,low.y); pair(v[2],high.z,low.z); high.w=low.w=0;
}
std::array<double,3> triple(const std::array<double,16>& a,std::size_t offset) { return {a[offset],a[offset+1],a[offset+2]}; }
std::uint64_t fingerprint(const HumanBehaviorProgramSource& s,const HumanBehaviorCompileBinding& b) {
    std::uint64_t h=14695981039346656037ull;
    auto add=[&](const auto& x){ for(auto v:std::as_bytes(std::span{&x,1})) { h^=std::to_integer<std::uint8_t>(v); h*=1099511628211ull; } };
    add(s.bodyCount);add(s.nq);add(s.nv);add(s.task);add(s.timestepNanoseconds);
    for(const auto& hash:s.sourceSHA256) for(auto x:hash) add(x);
    for(const auto& row:s.bodies) { add(row.sourceBodyRecordIndex);add(row.coreBodyIndex);for(auto x:row.sourceOriginInOriginalCOMFrame)add(x);for(auto x:row.sourceInertialQuaternionBody)add(x); }
    for(auto x:s.numerical)add(x); for(const auto& row:s.forbiddenBodies)for(auto x:row)add(x);
    for(const auto& row:b.cookedCOMOffset)for(auto x:row)add(x);
    need(h!=0,"zero behavior fingerprint"); return h;
}
}

bool decodeHumanBehaviorProgram(std::span<const std::byte> bytes,HumanBehaviorProgramSource& output,std::string& error) {
    try {
        need(bytes.size()>=248,"truncated behavior header"); Reader r{bytes};
        const auto magic=r.read<std::array<char,8>>(); need(magic==std::array<char,8>{'N','H','B','H','V','1',0,0},"invalid behavior program magic");
        need(r.read<std::uint32_t>()==1 && r.read<std::uint32_t>()==248,"unsupported behavior program ABI");
        need(r.read<std::uint32_t>()==bytes.size(),"behavior program byte count mismatch");
        HumanBehaviorProgramSource s; s.bodyCount=r.read<std::uint32_t>();s.nq=r.read<std::uint32_t>();s.nv=r.read<std::uint32_t>();
        const auto forbiddenCount=r.read<std::uint32_t>();s.task=r.read<std::uint32_t>();s.timestepNanoseconds=r.read<std::uint64_t>();
        need(r.read<std::uint64_t>()==0,"nonzero behavior reserved field");
        need(forbiddenCount>0&&forbiddenCount<=s.bodyCount&&s.bodyCount<=4096&&bytes.size()==568ull+8ull*forbiddenCount,"invalid behavior program counts");
        for(auto& hash:s.sourceSHA256)hash=r.read<std::array<std::uint8_t,32>>();
        for(auto& row:s.bodies) { row.sourceBodyRecordIndex=r.read<std::uint32_t>();row.coreBodyIndex=r.read<std::uint32_t>();for(auto& x:row.sourceOriginInOriginalCOMFrame)x=r.read<double>();for(auto& x:row.sourceInertialQuaternionBody)x=r.read<double>(); }
        for(auto& x:s.numerical)x=r.read<double>();
        s.forbiddenBodies.resize(forbiddenCount);for(auto& row:s.forbiddenBodies)for(auto& x:row)x=r.read<std::uint32_t>();
        need(r.offset==bytes.size(),"trailing behavior program bytes"); output=std::move(s);error.clear();return true;
    } catch(const std::exception& e) { error=e.what();return false; }
}

bool compileHumanBehaviorProgram(const HumanBehaviorProgramSource& s,const HumanBehaviorCompileBinding& b,CompiledHumanBehaviorProgram& output,std::string& error) {
    try {
        need(s.bodyCount>0&&s.bodyCount<=4096&&s.nq>0&&s.nv>0&&s.nv<=4096&&s.task<=2,"invalid behavior program shape");
        need(s.bodyCount==b.bodyCount&&s.nq==b.nq&&s.nv==b.nv,"behavior/native shape mismatch");
        need(s.timestepNanoseconds>0&&s.timestepNanoseconds<=std::numeric_limits<std::int64_t>::max()&&s.timestepNanoseconds==b.timestepNanoseconds,"behavior/native exact clock mismatch");
        need(s.sourceSHA256[0]==b.sourceArchiveSHA256&&s.sourceSHA256[1]==b.rigidSHA256,"behavior/native source identity mismatch");
        for(const auto& h:s.sourceSHA256)need(std::any_of(h.begin(),h.end(),[](auto x){return x!=0;}),"missing behavior source digest");
        need(b.cookedCOMOffset.size()==s.bodyCount&&!b.sourceToCore.empty(),"missing actual cooked body frame binding");
        std::set<std::uint32_t> sourceIDs,coreIDs;
        for(const auto& row:b.sourceToCore)need(row[0]<b.sourceToCore.size()&&row[1]<s.bodyCount&&sourceIDs.insert(row[0]).second&&coreIDs.insert(row[1]).second,"duplicate native source mapping");
        const auto registered=[&](std::uint32_t source,std::uint32_t core){return core<s.bodyCount&&std::find(b.sourceToCore.begin(),b.sourceToCore.end(),std::array<std::uint32_t,2>{source,core})!=b.sourceToCore.end();};
        for(const auto& row:b.cookedCOMOffset)for(double x:row)finite(x);
        for(const auto& row:s.bodies) {
            need(registered(row.sourceBodyRecordIndex,row.coreBodyIndex),"unresolved behavior source semantic");
            for(double x:row.sourceOriginInOriginalCOMFrame)finite(x);
            double n=0;for(double x:row.sourceInertialQuaternionBody){finite(x);n+=x*x;}need(std::abs(n-1)<=1e-10,"invalid source inertial rotation");
        }
        need(!s.forbiddenBodies.empty()&&s.forbiddenBodies.size()<=s.bodyCount,"missing forbidden semantic registrations");
        std::set<std::uint32_t> forbidden;
        for(const auto& row:s.forbiddenBodies)need(registered(row[0],row[1])&&forbidden.insert(row[1]).second,"duplicate or unresolved forbidden semantic");
        for(double x:s.numerical)finite(x);
        auto up=triple(s.numerical,3),forward=triple(s.numerical,6),trunk=triple(s.numerical,9);unit(up);unit(forward);unit(trunk);
        double dot=0;for(unsigned i=0;i<3;++i)dot+=up[i]*forward[i];need(std::abs(dot)<=1e-10,"nonorthogonal behavior world axes");
        need(s.numerical[12]>=0&&s.numerical[13]>=0&&s.numerical[13]<=3.14159265358979323846&&s.numerical[14]>=0&&s.numerical[15]>=0,"invalid behavior physical criterion");
        need(s.task==2||s.numerical[15]==0,"nonwalking task has target speed");
        CompiledHumanBehaviorProgram result;auto& p=result.gpu;
        p.abiVersion=MR_HUMAN_BEHAVIOR_ABI_VERSION;p.task=s.task;p.bodyCount=s.bodyCount;p.dofCount=s.nv;
        p.rootBody=s.bodies[0].coreBodyIndex;p.trunkBody=s.bodies[1].coreBodyIndex;p.velocityBody=s.bodies[2].coreBodyIndex;
        p.forbiddenBodyCount=static_cast<std::uint32_t>(forbidden.size());p.timestepNanoseconds=s.timestepNanoseconds;
        p.fingerprint=fingerprint(s,b);vectorPair(triple(s.numerical,0),p.worldOriginHigh,p.worldOriginLow);
        auto origin=s.bodies[0].sourceOriginInOriginalCOMFrame;for(unsigned i=0;i<3;++i)origin[i]-=b.cookedCOMOffset[p.rootBody][i];
        vectorPair(origin,p.rootOriginHigh,p.rootOriginLow);
        trunk=rotateInverse(s.bodies[1].sourceInertialQuaternionBody,trunk);unit(trunk);
        p.worldUp={static_cast<float>(up[0]),static_cast<float>(up[1]),static_cast<float>(up[2]),0};
        p.worldForward={static_cast<float>(forward[0]),static_cast<float>(forward[1]),static_cast<float>(forward[2]),0};
        p.trunkUp={static_cast<float>(trunk[0]),static_cast<float>(trunk[1]),static_cast<float>(trunk[2]),0};
        pair(s.numerical[12],p.limitsHigh.x,p.limitsLow.x);pair(s.numerical[13],p.limitsHigh.y,p.limitsLow.y);
        pair(s.numerical[14],p.limitsHigh.z,p.limitsLow.z);pair(s.numerical[15],p.limitsHigh.w,p.limitsLow.w);
        result.sourceSHA256=s.sourceSHA256;result.forbiddenBodies.assign(forbidden.begin(),forbidden.end());
        output=std::move(result);error.clear();return true;
    } catch(const std::exception& e) { error=e.what();return false; }
}
} // namespace metalrobo
