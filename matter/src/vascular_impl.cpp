#include "numi/matter/vascular.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string_view>

namespace numi::matter::detail {
namespace {
bool finite(const double x) { return std::isfinite(x) && std::isfinite(static_cast<float>(x)) && (x == 0.0 || static_cast<float>(x) != 0.0f); }
bool positive(const double x) { return finite(x) && x > 0.0; }
bool tolerance(const double x) { return positive(x) && x <= 1.0; }
bool finite4(const nm_float4 v) { return finite(v.x) && finite(v.y) && finite(v.z) && finite(v.w); }
bool utf8(std::string_view s) {
    if (s.empty()) return false;
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i++]);
        if (c == 0 || c < 0x20 || c == 0x7f) return false;
        if (c < 0x80) continue;
        unsigned remaining = 0, code = 0, minimum = 0;
        if (c >= 0xc2 && c <= 0xdf) { remaining=1; code=c&31; minimum=0x80; }
        else if (c >= 0xe0 && c <= 0xef) { remaining=2; code=c&15; minimum=0x800; }
        else if (c >= 0xf0 && c <= 0xf4) { remaining=3; code=c&7; minimum=0x10000; }
        else return false;
        if (remaining > s.size()-i) return false;
        while (remaining--) { const auto b=static_cast<unsigned char>(s[i++]); if ((b&0xc0)!=0x80) return false; code=(code<<6)|(b&63); }
        if (code<minimum || code>0x10ffff || (code>=0xd800 && code<=0xdfff)) return false;
    }
    return true;
}
nm_float4 f4(double x=0, double y=0, double z=0, double w=0) { return {float(x),float(y),float(z),float(w)}; }
template<class T> std::vector<std::size_t> order(const std::vector<T>& v) {
    std::vector<std::size_t> r(v.size()); std::iota(r.begin(),r.end(),0u);
    std::sort(r.begin(),r.end(),[&](auto a,auto b){return v[a].stableIdentifier<v[b].stableIdentifier;}); return r;
}
template<class T> bool validIds(const std::vector<T>& v) {
    std::set<std::uint32_t> ids; for (const auto& x:v) if (x.stableIdentifier==0 || !ids.insert(x.stableIdentifier).second) return false; return true;
}
void incidence(const std::vector<std::vector<std::uint32_t>>& rows, std::vector<std::uint32_t>& values, std::vector<NMVascularRangeGPU>& ranges) {
    for (const auto& row:rows) { ranges.push_back({std::uint32_t(values.size()),std::uint32_t(row.size()),0u,0u}); values.insert(values.end(),row.begin(),row.end()); }
}
bool zero4(const nm_float4 v) { return v.x==0 && v.y==0 && v.z==0 && v.w==0; }
int clockExponent(float timestep) { return std::ilogb(timestep)-23-16; }
bool periodTicks(double period, int exponent, std::uint64_t& ticks) {
    const double value=std::ldexp(period,-exponent);
    if (!std::isfinite(value) || value<1 || value>=0x1p63 || value!=std::floor(value)) return false;
    ticks=static_cast<std::uint64_t>(value); return std::ldexp(double(ticks),exponent)==period;
}
bool rationalPeriod(const VascularCompartmentSource& source, int exponent,
    std::uint64_t& numerator, std::uint64_t& multiplier) {
    const bool rational=source.periodNumeratorSeconds!=0 || source.periodDenominator!=0;
    if (!rational) {
        multiplier=1;
        return periodTicks(source.periodSeconds,exponent,numerator);
    }
    if(source.periodSeconds!=0 || !source.periodNumeratorSeconds || !source.periodDenominator ||
       std::gcd(source.periodNumeratorSeconds,source.periodDenominator)!=1u)return false;
    numerator=source.periodNumeratorSeconds;multiplier=source.periodDenominator;
    int shift=-exponent;
    while(shift>0 && (multiplier&1u)==0u){multiplier>>=1u;--shift;}
    while(shift<0 && (numerator&1u)==0u){numerator>>=1u;++shift;}
    constexpr std::uint64_t maximumNumerator=0x7fffffffffffffffull;
    if(shift>=63 || shift<=-64)return false;
    if(shift>0){if(numerator>(maximumNumerator>>shift))return false;numerator<<=shift;}
    else if(shift<0){if(multiplier>(std::numeric_limits<std::uint64_t>::max()>>-shift))return false;multiplier<<=-shift;}
    return numerator>0 && numerator<=maximumNumerator && multiplier>0;
}
bool validCompartment(const NMVascularCompartmentGPU& x, const NMVascularUnknownGPU& u) {
    if(x.identity.z>1 || x.identity.w>5 || !finite4(x.compliance) || !finite4(x.elastance) ||
       !finite4(x.waveform) || !finite4(x.pressureParameters))return false;
    if(x.identity.w==5) return x.identity.z==0 && positive(u.initialAndScaling.x) &&
        x.compliance.x==0 && x.compliance.y==0 && x.compliance.z==0 &&
        zero4(x.elastance) && zero4(x.waveform) && zero4(x.pressureParameters) &&
        x.periodTicks==0 && x.periodMultiplier==0;
    if(x.identity.z==0 && (!positive(x.compliance.x) || !positive(u.initialAndScaling.x)))return false;
    if(x.pressureParameters.y!=0 || x.pressureParameters.z!=0 || x.pressureParameters.w!=0)return false;
    if(x.identity.w==0 || x.identity.w==3) {
        if(!positive(x.compliance.z) || !positive(1.0/double(x.compliance.z)) || !zero4(x.elastance) ||
           x.periodTicks || x.periodMultiplier)return false;
        if(x.identity.w==0) {
            if(!zero4(x.waveform) || !zero4(x.pressureParameters))return false;
            return finite(double(x.compliance.w)+x.compliance.y+(double(u.initialAndScaling.x)-x.compliance.x)/x.compliance.z);
        }
        if(x.identity.z!=0 || !positive(x.pressureParameters.x) ||
           x.waveform.x!=float(std::acos(-1.0)) || x.waveform.y!=0 || x.waveform.z!=0 || x.waveform.w!=0)return false;
        const float normalizedInitial=u.initialAndScaling.x/u.initialAndScaling.y;
        const float restoredInitial=normalizedInitial*u.initialAndScaling.y;
        const float displacement=restoredInitial-x.compliance.x;
        if(!(restoredInitial>0) || !std::isfinite(restoredInitial) ||
           !(std::abs(u.initialAndScaling.x-x.compliance.x)<x.pressureParameters.x) ||
           !(std::abs(displacement)<x.pressureParameters.x))return false;
        const float angle=(0.5f*x.waveform.x)*(displacement/x.pressureParameters.x);
        const float tangent=std::tan(angle);
        const float inverseCompliance=(1.0f+tangent*tangent)/x.compliance.z;
        const float pressure=x.compliance.w+x.compliance.y+
            (2.0f*x.pressureParameters.x/x.waveform.x/x.compliance.z)*tangent;
        return std::cos(angle)>0 && positive(inverseCompliance) && finite(pressure);
    }
    if(x.identity.z!=0 || x.compliance.z!=0 || !zero4(x.pressureParameters) ||
       !positive(x.elastance.x) || x.elastance.y<x.elastance.x ||
       !x.periodTicks || x.periodTicks>=0x8000000000000000ULL || !x.periodMultiplier ||
       std::gcd(x.periodTicks,x.periodMultiplier)!=1u || x.waveform.x<3 || x.waveform.x>4 ||
       x.waveform.z!=0 || x.waveform.w!=0)return false;
    const auto start=x.elastance.z,end=x.elastance.w;
    if(!(start>0 && start<1 && end>0 && end<1))return false;
    if(x.identity.w==2 ? start+end<1 : end<=start)return false;
    if(x.identity.w==4 ? (x.waveform.y<0 || x.waveform.y+end>1) : x.waveform.y!=0)return false;
    for(double e:{double(x.elastance.x),double(x.elastance.y)})
        if(!finite(double(x.compliance.w)+x.compliance.y+e*(double(u.initialAndScaling.x)-x.compliance.x)))return false;
    return true;
}
bool validConnection(const NMVascularConnectionGPU& x, const NMVascularUnknownGPU& u) {
    if(x.identity.w>4 || !finite4(x.physical) || !finite4(x.directional))return false;
    if(x.identity.w==4) {
        const double r = u.initialAndScaling.x < 0 ? x.directional.x : x.physical.x;
        return x.physical.x>=0 && positive(x.directional.x) && x.directional.x>=x.physical.x &&
            x.physical.y==0 && x.physical.z==0 && x.physical.w==0 &&
            x.directional.y==0 && x.directional.z==0 && x.directional.w==0 &&
            positive(double(u.initialAndScaling.z)/u.initialAndScaling.y) &&
            finite(r*u.initialAndScaling.x);
    }
    if(!zero4(x.directional))return false;
    if(x.identity.w==0)return positive(x.physical.x) && x.physical.y>=0 && x.physical.z==0 && x.physical.w==0;
    if(u.initialAndScaling.x<0 || !positive(double(u.initialAndScaling.z)/u.initialAndScaling.y))return false;
    if(x.identity.w>=2)return positive(x.physical.x) && x.physical.y==0 && x.physical.z==0 &&
        (x.identity.w==3 || x.physical.w==0) && finite(double(x.physical.x)*u.initialAndScaling.x);
    if(x.physical.x!=0 || x.physical.y!=0 || !positive(x.physical.z) || x.physical.w!=0)return false;
    const double ratio=double(u.initialAndScaling.x)/x.physical.z;
    return finite(ratio*ratio);
}
bool allZero(const NMVascularIdentityGPU& x) { for (unsigned i=0;i<4;++i) if (x.content[i]||x.source[i]||x.authored[i]) return false; return true; }
// Direct material-boundary embedding: no geometry-dependent tolerance weld,
// artificial caps, source-volume offsets, or additional fluid mechanical mass.
using CavityPoint = std::array<long double,3>;
using CavityFaceKey = std::array<std::uint32_t,3>;
CavityPoint cavitySubtract(const CavityPoint& a,const CavityPoint& b) {
    return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};
}
CavityPoint cavityCross(const CavityPoint& a,const CavityPoint& b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
long double cavityDot(const CavityPoint& a,const CavityPoint& b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
CavityFaceKey cavityKey(CavityFaceKey key) {std::sort(key.begin(),key.end());return key;}
bool validateCavities(const CompiledWorld& world,std::string* error) {
    const auto& c=world.vascular;
    const auto fail=[&](const char* reason){if(error&&error->empty())*error=std::string("vascular cavity: ")+reason;return false;};
    const auto B=c.cavities.size(),C=c.compartments.size();
    if(B==0) {
        if(!c.cavityFaces.empty()||!c.cavityNodeIncidence.empty()||!c.cavityNodeRanges.empty())return fail("unowned cavity arrays");
        for(std::size_t i=0;i<C;++i)if(c.compartmentCavity[i]!=NM_INVALID_INDEX||c.compartments[i].identity.w==5)return fail("unbound deforming compartment");
        return true;
    }
    if(c.cavityFaces.size()>std::numeric_limits<std::uint32_t>::max()/3u)return fail("face incidence arena overflow");
    const auto point=[&](std::uint32_t node)->CavityPoint {
        const auto& p=world.fem.nodes[node].positionAndMass;
        return {p.x,p.y,p.z};
    };
    std::vector<std::uint32_t> owners(C,NM_INVALID_INDEX);
    std::vector<std::vector<std::uint32_t>> nodeFaces(world.fem.nodes.size());
    std::map<std::uint32_t,std::map<CavityFaceKey,std::vector<std::uint32_t>>> objectFaces;
    std::set<CavityFaceKey> ownedFaces;
    std::size_t nextFace=0;
    for(std::size_t i=0;i<B;++i) {
        const auto& cavity=c.cavities[i];const auto compartment=cavity.identity.y,objectIndex=cavity.identity.z;
        if(!cavity.identity.x||(i&&cavity.identity.x<=c.cavities[i-1].identity.x)||
           compartment>=C||objectIndex>=world.objects.size()||owners[compartment]!=NM_INVALID_INDEX||
           c.compartments[compartment].identity.w!=5||cavity.identity.w!=c.layout.cavities.z+i||
           cavity.faces.x!=nextFace||cavity.faces.y<4||cavity.faces.z||cavity.faces.w||
           cavity.faces.y>c.cavityFaces.size()-nextFace)return fail("invalid identity, face range or pressure ownership");
        bool source=false,mechanics=false;
        for(unsigned j=0;j<4;++j){source|=cavity.sourceIdentity[j]!=0;mechanics|=cavity.mechanicalIdentity[j]!=0;}
        if(!source||!mechanics)return fail("missing geometry or mechanical provenance");
        const auto& object=world.objects[objectIndex];
        if(object.representation!=NM_REPRESENTATION_FEM||
           (object.flags&(NM_OBJECT_ADAPTIVE|NM_OBJECT_MUTABLE_TOPOLOGY))||
           std::uint64_t(object.stateOffset)+object.stateCount>world.fem.nodes.size()||
           std::uint64_t(object.elementOffset)+object.elementCount>world.fem.tetrahedra.size())
            return fail("boundary requires an immutable nonadaptive real FEM object");
        if(object.schedulerIndex>=world.schedulers.size())return fail("cavity wall scheduler is absent");
        const auto& scheduler=world.schedulers[object.schedulerIndex];
        const auto exponent=world.dispatch.maximumRateExponent;
        if(object.solver.x!=exponent||object.solver.y!=exponent||
           scheduler.baseExponent!=exponent||scheduler.activeExponent!=exponent||scheduler.requestedExponent!=exponent)
            return fail("wall and hydraulic microtick rates disagree");
        if(!objectFaces.contains(objectIndex)) {
            auto& boundary=objectFaces[objectIndex];
            for(std::uint32_t k=0;k<object.elementCount;++k) {
                const auto& tet=world.fem.tetrahedra[object.elementOffset+k];
                if((tet.identity.w&NM_OBJECT_ACTIVE)==0)continue;
                const std::array<std::uint32_t,4> nodes{tet.nodes.x,tet.nodes.y,tet.nodes.z,tet.nodes.w};
                for(auto node:nodes)if(node<object.stateOffset||std::uint64_t(node)>=std::uint64_t(object.stateOffset)+object.stateCount)
                    return fail("wall tetrahedron escapes its object");
                for(unsigned opposite=0;opposite<4;++opposite) {
                    CavityFaceKey key{};unsigned slot=0;
                    for(unsigned j=0;j<4;++j)if(j!=opposite)key[slot++]=nodes[j];
                    boundary[cavityKey(key)].push_back(nodes[opposite]);
                }
            }
        }
        const auto& boundary=objectFaces.at(objectIndex);
        std::map<std::array<std::uint32_t,2>,std::vector<std::pair<std::uint32_t,int>>> edges;
        std::map<std::uint32_t,std::vector<std::array<std::uint32_t,2>>> vertexLinks;
        std::vector<std::vector<std::uint32_t>> adjacent(cavity.faces.y);
        long double volume=0;CavityPoint origin{};
        for(std::uint32_t j=0;j<cavity.faces.y;++j) {
            const auto faceIndex=std::uint32_t(nextFace+j);const auto& face=c.cavityFaces[faceIndex];
            const std::array<std::uint32_t,3> nodes{face.nodes.x,face.nodes.y,face.nodes.z};
            if(face.nodes.w||face.identity.x!=i||!face.identity.y||
               (j&&face.identity.y<=c.cavityFaces[faceIndex-1].identity.y)||
               face.identity.z!=std::uint32_t(VascularCavityFaceRole::materialWall)||face.identity.w)
                return fail("invalid face identity or nonmaterial interface role");
            for(auto node:nodes)if(node<object.stateOffset||std::uint64_t(node)>=std::uint64_t(object.stateOffset)+object.stateCount)
                return fail("cavity face escapes wall FEM node ownership");
            const auto key=cavityKey(nodes);const auto found=boundary.find(key);
            if(nodes[0]==nodes[1]||nodes[1]==nodes[2]||nodes[2]==nodes[0]||!ownedFaces.insert(key).second||
               found==boundary.end()||found->second.size()!=1)return fail("duplicate face or face is not an exposed material boundary");
            const auto a=point(nodes[0]),b=point(nodes[1]),d=point(nodes[2]);
            const auto normal=cavityCross(cavitySubtract(b,a),cavitySubtract(d,a));
            const auto side=cavityDot(normal,cavitySubtract(point(found->second.front()),a));
            if(!std::isfinite(side)||!(side>0))return fail("degenerate or outer-wall face; cavity normal must point into material");
            if(j==0)origin=a;
            volume+=cavityDot(cavitySubtract(a,origin),cavityCross(cavitySubtract(b,origin),cavitySubtract(d,origin)))/6;
            for(unsigned k=0;k<3;++k) {
                const auto n=nodes[k],m=nodes[(k+1)%3];
                const std::array<std::uint32_t,2> edge{std::min(n,m),std::max(n,m)};
                edges[edge].push_back({j,n<m?1:-1});
                vertexLinks[n].push_back({nodes[(k+1)%3],nodes[(k+2)%3]});
                nodeFaces[n].push_back(faceIndex);
            }
        }
        for(const auto& [edge,incidence]:edges) {
            (void)edge;
            if(incidence.size()!=2||incidence[0].second+incidence[1].second!=0)return fail("boundary is open, nonmanifold or inconsistently oriented");
            adjacent[incidence[0].first].push_back(incidence[1].first);
            adjacent[incidence[1].first].push_back(incidence[0].first);
        }
        const auto connected=[](const auto& graph,std::uint32_t start) {
            std::set<std::uint32_t> seen;std::vector<std::uint32_t> pending{start};
            while(!pending.empty()) {const auto n=pending.back();pending.pop_back();if(!seen.insert(n).second)continue;
                for(auto other:graph.at(n))pending.push_back(other);}
            return seen.size()==graph.size();
        };
        if(!connected(adjacent,0u))return fail("cavity boundary has disconnected components");
        for(const auto& [node,links]:vertexLinks) {
            (void)node;std::map<std::uint32_t,std::vector<std::uint32_t>> graph;
            for(const auto& link:links){graph[link[0]].push_back(link[1]);graph[link[1]].push_back(link[0]);}
            for(const auto& [vertex,neighbours]:graph){(void)vertex;if(neighbours.size()!=2)return fail("cavity vertex link is not a manifold cycle");}
            if(!connected(graph,graph.begin()->first))return fail("cavity vertex link has pinched components");
        }
        const auto pressure=c.unknowns[cavity.identity.w].initialAndScaling;
        const auto volumeRow=c.unknowns[compartment].initialAndScaling;
        const float restoredPressure=(pressure.x/pressure.y)*pressure.y;
        if(!std::isfinite(restoredPressure)||!finite(double(restoredPressure)-c.compartments[compartment].compliance.w))
            return fail("initial absolute or transmural pressure is not representable");
        const float initialVolume=(volumeRow.x/volumeRow.y)*volumeRow.y;
        const long double discrepancy=std::abs(volume-static_cast<long double>(initialVolume));
        if(!std::isfinite(volume)||!(volume>0)||!std::isfinite(initialVolume)||!(initialVolume>0)||
           pressure.z!=volumeRow.y||
           discrepancy>static_cast<long double>(pressure.w)*pressure.z||
           discrepancy>64*std::numeric_limits<float>::epsilon()*volume)
            return fail("initial absolute volume does not match the cooked enclosed cavity without offsets");
        owners[compartment]=std::uint32_t(i);nextFace+=cavity.faces.y;
    }
    if(nextFace!=c.cavityFaces.size()||owners!=c.compartmentCavity)return fail("unowned faces or forged compartment-to-cavity mapping");
    for(std::size_t i=0;i<C;++i)if((owners[i]!=NM_INVALID_INDEX)!=(c.compartments[i].identity.w==5))return fail("deforming compartment has missing cavity geometry");
    if(c.cavityNodeRanges.size()!=nodeFaces.size())return fail("cavity node incidence coverage mismatch");
    std::size_t next=0;
    for(std::size_t i=0;i<nodeFaces.size();++i) {
        const auto& range=c.cavityNodeRanges[i];
        if(range.first!=next||range.count!=nodeFaces[i].size()||range.reserved0||range.reserved1||
           range.count>c.cavityNodeIncidence.size()-next)return fail("invalid cavity node incidence range");
        for(auto face:nodeFaces[i])if(c.cavityNodeIncidence[next++]!=face)return fail("cavity node incidence is not canonical");
    }
    return next==c.cavityNodeIncidence.size()||fail("unowned cavity node incidence");
}

}

bool compileVascular(const WorldSource& source, CompiledWorld& world, std::vector<Diagnostic>& diagnostics) {
    const auto& v=source.vascular; auto& c=world.vascular; c={};
    const auto fail=[&](std::string message){diagnostics.push_back({Diagnostic::Severity::error,0u,0u,"vascular: "+message});return false;};
    const std::uint64_t C=v.compartments.size(),E=v.connections.size(),S=v.species.size(),T=v.tissues.size(),X=v.exchanges.size(),B=v.cavities.size();
    if (C==0) {
        if (E||S||T||X||B || std::any_of(v.contentIdentity.begin(),v.contentIdentity.end(),[](auto x){return x!=0;}) || std::any_of(v.sourceIdentity.begin(),v.sourceIdentity.end(),[](auto x){return x!=0;}) || std::any_of(v.authoredIdentity.begin(),v.authoredIdentity.end(),[](auto x){return x!=0;})) return fail("empty network has dangling authored data");
        return true;
    }
    for (const auto& object : source.objects)
        if (object.mutationPolicy.enabled || !object.mutationCommands.empty())
            return fail("vascular V1 does not support topology mutation or binding remapping");
    const auto limit=std::numeric_limits<std::uint32_t>::max();
    if (C>limit || E>limit/2u || S>limit || T>limit || X>limit || B>limit || C+E>limit || C+T>limit || (S && C+T>(limit-C-E)/S)) return fail("network exceeds 32-bit arena capacity");
    const std::uint64_t pressureBase=C+E+(C+T)*S;
    const std::uint64_t N=pressureBase+B;
    if (N>limit || source.environmentCount==0 || N>limit/(std::uint64_t(source.environmentCount)*(NM_MIXED_FGMRES_RESTART+1u))) return fail("network exceeds GPU Krylov address capacity");
    if (!validIds(v.compartments)||!validIds(v.connections)||!validIds(v.species)||!validIds(v.tissues)||!validIds(v.exchanges)||!validIds(v.cavities)) return fail("stable identifiers must be nonzero and unique within each record kind");
    if(!positive(source.frameTimestep))return fail("base timestep is not positive finite representable Float32");
    const int exponent=clockExponent(float(source.frameTimestep));
    c.layout.clock={std::bit_cast<std::uint32_t>(std::int32_t(exponent)),0u,0u,0u};
    c.layout.counts={std::uint32_t(C),std::uint32_t(E),std::uint32_t(S),std::uint32_t(T)};
    c.layout.offsets={0u,std::uint32_t(C),std::uint32_t(C+E),std::uint32_t(C+E+C*S)};
    c.layout.ranges={std::uint32_t(X),0u,std::uint32_t(N),0u};
    c.layout.cavities={std::uint32_t(B),0u,std::uint32_t(pressureBase),0u};
    c.compartmentCavity.assign(C,NM_INVALID_INDEX);
    for (unsigned i=0;i<4;++i) {c.identity.content[i]=v.contentIdentity[i];c.identity.source[i]=v.sourceIdentity[i];c.identity.authored[i]=v.authoredIdentity[i];}
    if (std::all_of(v.contentIdentity.begin(),v.contentIdentity.end(),[](auto x){return x==0;})) return fail("source payload identity is missing");
    const auto addName=[&](const std::string& name,std::uint32_t& offset) {
        if (!utf8(name) || name.size()+1>limit-c.names.size()) return false;
        offset=std::uint32_t(c.names.size());c.names.insert(c.names.end(),name.begin(),name.end());c.names.push_back(0);return true;
    };
    const auto os=order(v.species),oc=order(v.compartments),oe=order(v.connections),ot=order(v.tissues),ox=order(v.exchanges);
    std::map<std::uint32_t,std::uint32_t> si,ci,ti;
    std::set<std::string> speciesNames;
    for (auto index:os) {const auto& x=v.species[index];std::uint32_t name=0;
        if (!positive(x.amountScale)||!tolerance(x.amountResidualTolerance)||!speciesNames.insert(x.name).second||!addName(x.name,name)) return fail("invalid species name, amount scale or tolerance");
        si[x.stableIdentifier]=std::uint32_t(c.species.size());c.species.push_back({{x.stableIdentifier,name,0u,0u},f4(x.amountScale,x.amountResidualTolerance)});
    }
    c.unknowns.resize(N);
    const auto amount=[&](std::uint32_t base,const std::vector<double>& initial) {
        if (initial.size()!=S) return false;
        for (std::size_t j=0;j<S;++j) {const auto& sp=v.species[os[j]];const double m=initial[os[j]];if (!finite(m)||m<0||!finite(m/sp.amountScale)) return false;c.unknowns[base+j]={f4(m,sp.amountScale,sp.amountScale,sp.amountResidualTolerance)};} return true;
    };
    for (auto index:oc) {const auto& x=v.compartments[index];std::uint32_t name=0;const auto row=std::uint32_t(c.compartments.size());
        if (!finite(x.referenceVolume)||!finite(x.initialVolume)||!finite(x.compliance)||!finite(x.referencePressure)||!finite(x.externalPressure)||!positive(x.volumeScale)||!tolerance(x.volumeResidualTolerance)||!finite(x.initialVolume/x.volumeScale)||!addName(x.anatomicalIdentifier,name)) return fail("invalid compartment volume/storage, pressure, scale, tolerance or anatomy");
        if (!finite(x.elastanceMin)||!finite(x.elastanceMax)||!finite(x.periodSeconds)||!finite(x.activationStart)||!finite(x.activationEnd)||!finite(x.sourcePi)||!finite(x.phaseDelay)||!finite(x.maximumVolumeDisplacement))return fail("unrepresentable source waveform");
        if (x.pressureLaw==VascularPressureLaw::atanCompliance &&
            (!(x.maximumVolumeDisplacement>0) ||
             x.initialVolume<=x.referenceVolume-x.maximumVolumeDisplacement ||
             x.initialVolume>=x.referenceVolume+x.maximumVolumeDisplacement))
            return fail("authored absolute volume is outside the open atan pressure domain");
        NMVascularCompartmentGPU node{};
        node.identity={x.stableIdentifier,name,std::uint32_t(x.storageKind),std::uint32_t(x.pressureLaw)};
        node.compliance=f4(x.referenceVolume,x.referencePressure,x.compliance,x.externalPressure);
        node.elastance=f4(x.elastanceMin,x.elastanceMax,x.activationStart,x.activationEnd);
        node.waveform=f4(x.sourcePi,x.phaseDelay);
        node.pressureParameters=f4(x.maximumVolumeDisplacement);
        if(x.pressureLaw==VascularPressureLaw::ventricularElastance || x.pressureLaw==VascularPressureLaw::atrialElastance || x.pressureLaw==VascularPressureLaw::cosinePulseElastance) {
            if(!rationalPeriod(x,exponent,node.periodTicks,node.periodMultiplier))return fail("source period cannot be represented exactly by bounded native rational clock");
        } else if(x.periodSeconds!=0 || x.periodNumeratorSeconds || x.periodDenominator)return fail("untimed pressure law has unexpected period");
        c.unknowns[row]={f4(x.initialVolume,x.volumeScale,x.volumeScale,x.volumeResidualTolerance)};
        if(!validCompartment(node,c.unknowns[row]))return fail("unsupported storage or cardiac pressure law");
        if(x.storageKind==VascularStorageKind::storageDisplacement && (S||T||X))return fail("hydraulic storage displacement lacks absolute blood volume required by species/tissue exchange");
        ci[x.stableIdentifier]=row;c.compartments.push_back(node);
        if (!amount(c.layout.offsets.z+row*S,x.initialSpeciesAmounts)) return fail("invalid blood species amounts");
    }
    std::vector<std::vector<std::uint32_t>> edges(C),bloodX(C*S),tissueX(T*S);
    for (auto index:oe) {const auto& x=v.connections[index];const auto row=std::uint32_t(c.connections.size());
        if (!ci.contains(x.fromCompartment)||!ci.contains(x.toCompartment)||x.fromCompartment==x.toCompartment||!finite(x.resistance)||!finite(x.reverseResistance)||!finite(x.inertance)||!finite(x.orificeCoefficient)||!finite(x.downstreamPressureFloor)||!finite(x.initialFlow)||!positive(x.flowScale)||!positive(x.pressureScale)||!tolerance(x.flowResidualTolerance)||!finite(x.initialFlow/x.flowScale)) return fail("invalid connection endpoint, passive law, scale or tolerance");
        if (x.flowLaw==VascularFlowLaw::directionalResistance ?
            (x.resistance<0 || !(x.reverseResistance>0) || x.reverseResistance<x.resistance) :
            x.reverseResistance!=0) return fail("invalid authored directional resistance");
        const auto a=ci[x.fromCompartment],b=ci[x.toCompartment];c.connections.push_back({{x.stableIdentifier,a,b,std::uint32_t(x.flowLaw)},f4(x.resistance,x.inertance,x.orificeCoefficient,x.downstreamPressureFloor),f4(x.reverseResistance)});
        c.unknowns[c.layout.offsets.y+row]={f4(x.initialFlow,x.flowScale,x.pressureScale,x.flowResidualTolerance)};if(!validConnection(c.connections.back(),c.unknowns[c.layout.offsets.y+row]))return fail("unsupported or invalid flow law");edges[a].push_back(row);edges[b].push_back(row);
    }
    for (auto index:ot) {const auto& x=v.tissues[index];const auto row=std::uint32_t(c.tissues.size());std::uint32_t name=0;
        if (!positive(x.volume)||x.mechanicsFeedback||!addName(x.anatomicalIdentifier,name)) return fail("invalid tissue volume/anatomy or unsupported mechanics feedback");
        if ((x.objectIndex==NM_INVALID_INDEX)!=x.femRegion.empty()) return fail("FEM tissue region requires both a real object and nodes");
        if (x.objectIndex!=NM_INVALID_INDEX && (x.objectIndex>=world.objects.size() || world.objects[x.objectIndex].representation!=NM_REPRESENTATION_FEM)) return fail("tissue binding does not name a real FEM object");
        ti[x.stableIdentifier]=row;NMVascularTissueGPU tissue{{x.stableIdentifier,name,x.objectIndex,0u},f4(x.volume),{std::uint32_t(c.tissueBindings.size()),std::uint32_t(x.femRegion.size()),0u,0u}};
        double sum=0;std::set<std::uint32_t> seen;auto region=x.femRegion;std::sort(region.begin(),region.end(),[](auto a,auto b){return a.node<b.node;});
        for (const auto& b:region) {const auto& object=world.objects[x.objectIndex];if (!positive(b.weight)||b.node>=object.stateCount||!seen.insert(b.node).second||c.tissueBindings.size()==limit) return fail("invalid or duplicate FEM regional node/weight");sum+=b.weight;c.tissueBindings.push_back({{row,object.stateOffset+b.node,0u,0u},f4(b.weight)});}
        if (!region.empty() && std::abs(sum-1.0)>1e-8) return fail("FEM regional weights must sum to one");
        c.tissues.push_back(tissue);if (!amount(c.layout.offsets.w+row*S,x.initialSpeciesAmounts)) return fail("invalid tissue species amounts");
    }
    for (auto index:ox) {const auto& x=v.exchanges[index];const auto row=std::uint32_t(c.exchanges.size());
        if (!ci.contains(x.compartment)||!ti.contains(x.tissue)||!si.contains(x.species)||!positive(x.permeabilitySurface)||!positive(x.partitionCoefficient)) return fail("invalid exchange endpoint, species, permeability or partition coefficient");
        const auto a=ci[x.compartment],b=ti[x.tissue],s=si[x.species];c.exchanges.push_back({{x.stableIdentifier,a,b,s},f4(x.permeabilitySurface,x.partitionCoefficient)});bloodX[a*S+s].push_back(row);tissueX[b*S+s].push_back(row);
    }
    for(auto index:order(v.cavities)) {
        const auto& x=v.cavities[index];
        if(!ci.contains(x.compartment) || x.objectIndex>=world.objects.size() ||
           x.objectIndex>=source.objects.size() || !finite(x.initialPressure) ||
           !positive(x.pressureScale) || !finite(x.initialPressure/x.pressureScale) ||
           !tolerance(x.geometryResidualTolerance) || !validIds(x.faces) || x.faces.empty())
            return fail("invalid cavity pressure, scale, identity, object or face IDs");
        const auto compartment=ci[x.compartment], cavity=std::uint32_t(c.cavities.size());
        if(c.compartmentCavity[compartment]!=NM_INVALID_INDEX ||
           c.compartments[compartment].identity.w!=std::uint32_t(VascularPressureLaw::deformingCavity))
            return fail("cavity pressure must replace exactly one compartment constitutive law");
        if(source.objects[x.objectIndex].adaptive || source.objects[x.objectIndex].automaticRepresentation)
            return fail("cavity boundary requires immutable nonadaptive FEM representation");
        if(x.faces.size()>limit-c.cavityFaces.size())return fail("cavity face arena overflow");
        auto& object=world.objects[x.objectIndex];
        // Geometry and hydraulic continuity share one microtick. This is a
        // cooked scheduling invariant, never a runtime host stepping loop.
        if(object.schedulerIndex>=world.schedulers.size())return fail("cavity wall scheduler is absent");
        const auto exponent=world.dispatch.maximumRateExponent;
        object.solver.x=exponent;
        auto& scheduler=world.schedulers[object.schedulerIndex];
        scheduler.baseExponent=exponent;scheduler.activeExponent=exponent;scheduler.requestedExponent=exponent;
        NMVascularCavityGPU cavityGPU{};
        cavityGPU.identity={x.stableIdentifier,compartment,x.objectIndex,std::uint32_t(pressureBase+cavity)};
        cavityGPU.faces={std::uint32_t(c.cavityFaces.size()),std::uint32_t(x.faces.size()),0u,0u};
        for(unsigned k=0;k<4;++k){cavityGPU.sourceIdentity[k]=x.sourceIdentity[k];cavityGPU.mechanicalIdentity[k]=x.mechanicalIdentity[k];}
        for(auto faceIndex:order(x.faces)) {
            const auto& face=x.faces[faceIndex];
            if(face.role!=VascularCavityFaceRole::materialWall ||
               std::any_of(face.nodes.begin(),face.nodes.end(),[&](auto n){return n>=object.stateCount;}))
                return fail("cavity face must bind material-wall FEM nodes; artificial interfaces are unsupported");
            c.cavityFaces.push_back({{object.stateOffset+face.nodes[0],object.stateOffset+face.nodes[1],object.stateOffset+face.nodes[2],0u},
                {cavity,face.stableIdentifier,std::uint32_t(face.role),0u}});
        }
        c.compartmentCavity[compartment]=cavity;
        c.unknowns[pressureBase+cavity]={f4(x.initialPressure,x.pressureScale,
            c.unknowns[compartment].initialAndScaling.y,x.geometryResidualTolerance)};
        c.cavities.push_back(cavityGPU);
    }
    c.layout.cavities.y=std::uint32_t(c.cavityFaces.size());
    if(B) {
        std::vector<std::vector<std::uint32_t>> nodeFaces(world.fem.nodes.size());
        for(std::uint32_t i=0;i<c.cavityFaces.size();++i) {
            const auto& n=c.cavityFaces[i].nodes;
            if(n.x>=nodeFaces.size()||n.y>=nodeFaces.size()||n.z>=nodeFaces.size())return fail("cavity node outside FEM arena");
            nodeFaces[n.x].push_back(i);nodeFaces[n.y].push_back(i);nodeFaces[n.z].push_back(i);
        }
        incidence(nodeFaces,c.cavityNodeIncidence,c.cavityNodeRanges);
    }
    c.layout.ranges.y=std::uint32_t(c.tissueBindings.size());
    incidence(edges,c.connectionIncidence,c.connectionRanges);incidence(bloodX,c.bloodExchangeIncidence,c.bloodExchangeRanges);incidence(tissueX,c.tissueExchangeIncidence,c.tissueExchangeRanges);
    std::string error;if (!validateVascularLayout(world,&error)) return fail(error);return true;
}

bool validateVascularLayout(const CompiledWorld& world,std::string* error) {
    const auto& c=world.vascular;const auto fail=[&](std::string_view s){if(error&&error->empty())*error="vascular: "+std::string(s);return false;};
    const std::uint64_t C=c.compartments.size(),E=c.connections.size(),S=c.species.size(),T=c.tissues.size(),X=c.exchanges.size(),B=c.cavities.size();
    const auto limit=std::numeric_limits<std::uint32_t>::max();
    if (C>limit||E>limit/2u||S>limit||T>limit||X>limit||B>limit||C+E>limit||C+T>limit||(S && C+T>(limit-C-E)/S))return fail("cooked count overflow");
    const auto pressureBase=C+E+(C+T)*S;
    const auto N=pressureBase+B;
    if(N>limit)return fail("cooked cavity pressure capacity overflow");
    if (c.layout.counts.x!=C||c.layout.counts.y!=E||c.layout.counts.z!=S||c.layout.counts.w!=T||c.layout.ranges.x!=X||c.layout.ranges.y!=c.tissueBindings.size()||c.layout.ranges.z!=N||c.layout.ranges.w!=0||c.layout.offsets.x!=0||c.layout.offsets.y!=C||c.layout.offsets.z!=C+E||c.layout.offsets.w!=C+E+C*S||c.unknowns.size()!=N||c.layout.cavities.x!=B||c.layout.cavities.y!=c.cavityFaces.size()||c.layout.cavities.z!=pressureBase||c.layout.cavities.w||c.compartmentCavity.size()!=C) return fail("layout counts or unknown offsets disagree");
    if (C==0) return (c.layout.clock.x==0&&c.layout.clock.y==0&&c.layout.clock.z==0&&c.layout.clock.w==0&&E==0&&S==0&&T==0&&X==0&&B==0&&c.cavityFaces.empty()&&c.compartmentCavity.empty()&&c.cavityNodeIncidence.empty()&&c.cavityNodeRanges.empty()&&c.names.empty()&&c.tissueBindings.empty()&&c.connectionIncidence.empty()&&c.connectionRanges.empty()&&c.bloodExchangeIncidence.empty()&&c.bloodExchangeRanges.empty()&&c.tissueExchangeIncidence.empty()&&c.tissueExchangeRanges.empty()&&allZero(c.identity))||fail("empty network has dangling state");
    if(!positive(world.dispatch.gravityAndTimestep.w))return fail("cooked base timestep is invalid");
    if(c.layout.clock.x!=std::bit_cast<std::uint32_t>(std::int32_t(clockExponent(world.dispatch.gravityAndTimestep.w)))||c.layout.clock.y||c.layout.clock.z||c.layout.clock.w)return fail("clock quantum is not canonical for the cooked timestep");
    if (world.dispatch.environmentCount==0 || N>limit/(std::uint64_t(world.dispatch.environmentCount)*(NM_MIXED_FGMRES_RESTART+1u)))return fail("Krylov address overflow");
    for (const auto& object : world.objects)
        if ((object.flags & NM_OBJECT_MUTABLE_TOPOLOGY) != 0u)
            return fail("vascular V1 rejects mutable FEM topology");
    if (!world.fem.mutationCommands.empty())
        return fail("vascular V1 rejects topology mutation commands");
    bool hasContent=false;for(auto x:c.identity.content)hasContent|=x!=0;if(!hasContent)return fail("source content identity is missing");
    std::size_t nextName=0;
    const auto name=[&](std::uint32_t offset,std::string* result=nullptr){
        if(offset!=nextName||offset>=c.names.size())return false;const auto begin=c.names.begin()+offset;const auto end=std::find(begin,c.names.end(),0);if(end==c.names.end())return false;
        const std::string value(begin,end);if(!utf8(value))return false;nextName=std::size_t(end-c.names.begin())+1;if(result)*result=value;return true;
    };
    for (std::size_t i=0;i<N;++i) {const auto v=c.unknowns[i].initialAndScaling;if(!finite4(v)||!positive(v.y)||!positive(v.z)||!tolerance(v.w)||!finite(double(v.x)/v.y))return fail("invalid normalized unknown");if((i<C && c.compartments[i].identity.z==0 && !(v.x>0))||(i>=C+E&&i<pressureBase&&v.x<0))return fail("negative amount or nonpositive volume");if((i<C||(i>=C+E&&i<pressureBase))&&v.y!=v.z)return fail("conservation row has inconsistent physical scale");}
    std::set<std::string> speciesNames;
    for(std::size_t i=0;i<S;++i) {const auto& s=c.species[i];std::string text;
        if(!s.identity.x||(i&&s.identity.x<=c.species[i-1].identity.x)||s.identity.z||s.identity.w||!name(s.identity.y,&text)||!speciesNames.insert(text).second||!finite4(s.scaling)||!positive(s.scaling.x)||!tolerance(s.scaling.y)||s.scaling.z!=0||s.scaling.w!=0)return fail("invalid species identity or scale");
        for(std::size_t j=0;j<C+T;++j){const auto u=c.unknowns[C+E+j*S+i].initialAndScaling;if(u.y!=s.scaling.x||u.z!=s.scaling.x||u.w!=s.scaling.y)return fail("species unknown scales disagree");}
    }
    for(std::size_t i=0;i<C;++i) {const auto& x=c.compartments[i];
        if(!x.identity.x||(i&&x.identity.x<=c.compartments[i-1].identity.x)||!name(x.identity.y)||!validCompartment(x,c.unknowns[i]))return fail("invalid compartment pressure/storage law");
        if(x.identity.z==1 && (S||T||X))return fail("storage displacement cannot supply blood concentration or tissue exchange");
        for (std::size_t j=0;j<S;++j) {
            const double concentration=double(c.unknowns[C+E+i*S+j].initialAndScaling.x)/c.unknowns[i].initialAndScaling.x;
            if (!finite(concentration)) return fail("initial blood concentration is not representable");
        }
    }
    std::vector<std::vector<std::uint32_t>> edges(C),bloodX(C*S),tissueX(T*S);
    for(std::size_t i=0;i<E;++i) {const auto& x=c.connections[i];if(!x.identity.x||(i&&x.identity.x<=c.connections[i-1].identity.x)||x.identity.y>=C||x.identity.z>=C||x.identity.y==x.identity.z||!validConnection(x,c.unknowns[C+i]))return fail("invalid connection flow law");edges[x.identity.y].push_back(i);edges[x.identity.z].push_back(i);}
    // Potential ideal-pressure constraints must have independent incidence
    // columns: reject undirected cycles (including parallel/antiparallel edges).
    // With positive independent compliant storage this suffices for a positive
    // definite storage Schur block. It is not a rank claim for coupled mechanics.
    std::vector<std::uint32_t> idealParent(C);std::iota(idealParent.begin(),idealParent.end(),0u);
    const auto idealRoot=[&](std::uint32_t i){while(idealParent[i]!=i){idealParent[i]=idealParent[idealParent[i]];i=idealParent[i];}return i;};
    for(const auto& edge:c.connections) if(edge.identity.w==4u && edge.physical.x==0.0f) {
        const auto a=idealRoot(edge.identity.y),b=idealRoot(edge.identity.z);
        if(a==b)return fail("zero-forward directional connections must form an undirected forest");
        idealParent[b]=a;
    }
    std::size_t nextBinding=0;
    for(std::size_t i=0;i<T;++i) {const auto& x=c.tissues[i];if(!x.identity.x||(i&&x.identity.x<=c.tissues[i-1].identity.x)||x.identity.w||!name(x.identity.y)||!finite4(x.physical)||!positive(x.physical.x)||x.physical.y!=0||x.physical.z!=0||x.physical.w!=0||x.region.x!=nextBinding||x.region.z||x.region.w||x.region.y>c.tissueBindings.size()-nextBinding)return fail("invalid fixed tissue reservoir");
        if((x.identity.z==NM_INVALID_INDEX)!=(x.region.y==0))return fail("incomplete tissue FEM binding");
        if(x.identity.z!=NM_INVALID_INDEX && (x.identity.z>=world.objects.size()||world.objects[x.identity.z].representation!=NM_REPRESENTATION_FEM))return fail("tissue does not bind a real FEM object");
        double sum=0;std::uint32_t previous=0;
        for(std::size_t j=0;j<x.region.y;++j){const auto& b=c.tissueBindings[nextBinding+j];const auto& object=world.objects[x.identity.z];if(b.identity.x!=i||b.identity.y<object.stateOffset||std::uint64_t(b.identity.y)>=std::uint64_t(object.stateOffset)+object.stateCount||(j&&b.identity.y<=previous)||b.identity.z||b.identity.w||!finite4(b.physical)||!positive(b.physical.x)||b.physical.y!=0||b.physical.z!=0||b.physical.w!=0)return fail("invalid tissue FEM incidence");previous=b.identity.y;sum+=b.physical.x;}
        if(x.region.y&&std::abs(sum-1.0)>1e-6)return fail("cooked tissue region is not normalized");nextBinding+=x.region.y;
        for (std::size_t j=0;j<S;++j) {
            const double concentration=double(c.unknowns[C+E+C*S+i*S+j].initialAndScaling.x)/x.physical.x;
            if (!finite(concentration)) return fail("initial tissue concentration is not representable");
        }
    }
    if(nextName!=c.names.size()||nextBinding!=c.tissueBindings.size())return fail("unowned names or tissue bindings");
    for(std::size_t i=0;i<X;++i){const auto& x=c.exchanges[i];if(!x.identity.x||(i&&x.identity.x<=c.exchanges[i-1].identity.x)||x.identity.y>=C||x.identity.z>=T||x.identity.w>=S||!finite4(x.physical)||!positive(x.physical.x)||!positive(x.physical.y)||x.physical.z!=0||x.physical.w!=0)return fail("invalid conservative exchange");
        const double partitionVolume=double(c.tissues[x.identity.z].physical.x)*x.physical.y;
        const double partitionConcentration=double(c.unknowns[c.layout.offsets.w+x.identity.z*S+x.identity.w].initialAndScaling.x)/partitionVolume;
        if (!positive(partitionVolume)||!finite(partitionConcentration))
            return fail("exchange partition volume or concentration is not representable");
        bloodX[x.identity.y*S+x.identity.w].push_back(i);tissueX[x.identity.z*S+x.identity.w].push_back(i);}
    const auto sameIncidence=[&](const auto& expected,const auto& values,const auto& ranges){if(ranges.size()!=expected.size())return false;std::size_t next=0;for(std::size_t i=0;i<expected.size();++i){const auto& r=ranges[i];if(r.first!=next||r.count!=expected[i].size()||r.reserved0||r.reserved1||r.count>values.size()-next)return false;for(std::size_t j=0;j<r.count;++j)if(values[next+j]!=expected[i][j])return false;next+=r.count;}return next==values.size();};
    if(!sameIncidence(edges,c.connectionIncidence,c.connectionRanges)||!sameIncidence(bloodX,c.bloodExchangeIncidence,c.bloodExchangeRanges)||!sameIncidence(tissueX,c.tissueExchangeIncidence,c.tissueExchangeRanges))return fail("incidence is missing, duplicated or out of canonical order");
    if(!validateCavities(world,error))return false;
    return true;
}
} // namespace numi::matter::detail
