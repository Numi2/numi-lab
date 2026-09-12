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
bool validCompartment(const NMVascularCompartmentGPU& x, const NMVascularUnknownGPU& u) {
    if(x.identity.z>1 || x.identity.w>2 || !finite4(x.compliance) || !finite4(x.elastance) || !finite4(x.waveform) || x.reserved0) return false;
    if(x.identity.z==0 && (!positive(x.compliance.x) || !positive(u.initialAndScaling.x))) return false;
    if(x.identity.w==0) {
        if(!positive(x.compliance.z) || !positive(1.0/double(x.compliance.z)) || !zero4(x.elastance) || !zero4(x.waveform) || x.periodTicks) return false;
        return finite(double(x.compliance.w)+x.compliance.y+(double(u.initialAndScaling.x)-x.compliance.x)/x.compliance.z);
    }
    if(x.identity.z!=0 || x.compliance.z!=0 || !positive(x.elastance.x) || x.elastance.y<x.elastance.x ||
       !x.periodTicks || x.periodTicks>=0x8000000000000000ULL || x.waveform.x<3 || x.waveform.x>4 ||
       x.waveform.y!=0 || x.waveform.z!=0 || x.waveform.w!=0) return false;
    const auto start=x.elastance.z,end=x.elastance.w;
    if(!(start>0 && start<1 && end>0 && end<1))return false;
    if(x.identity.w==1 ? end<=start : start+end<1)return false;
    for(double e:{double(x.elastance.x),double(x.elastance.y)})
        if(!finite(double(x.compliance.w)+x.compliance.y+e*(double(u.initialAndScaling.x)-x.compliance.x)))return false;
    return true;
}
bool validConnection(const NMVascularConnectionGPU& x, const NMVascularUnknownGPU& u) {
    if(x.identity.w>1 || !finite4(x.physical) || x.physical.w!=0)return false;
    if(x.identity.w==0)return positive(x.physical.x) && x.physical.y>=0 && x.physical.z==0;
    if(x.physical.x!=0 || x.physical.y!=0 || !positive(x.physical.z) || u.initialAndScaling.x<0)return false;
    const double ratio=double(u.initialAndScaling.x)/x.physical.z;
    return finite(ratio*ratio) && positive(double(u.initialAndScaling.z)/u.initialAndScaling.y);
}
bool allZero(const NMVascularIdentityGPU& x) { for (unsigned i=0;i<4;++i) if (x.content[i]||x.source[i]||x.authored[i]) return false; return true; }
}

bool compileVascular(const WorldSource& source, CompiledWorld& world, std::vector<Diagnostic>& diagnostics) {
    const auto& v=source.vascular; auto& c=world.vascular; c={};
    const auto fail=[&](std::string message){diagnostics.push_back({Diagnostic::Severity::error,0u,0u,"vascular: "+message});return false;};
    const std::uint64_t C=v.compartments.size(),E=v.connections.size(),S=v.species.size(),T=v.tissues.size(),X=v.exchanges.size();
    if (C==0) {
        if (E||S||T||X || std::any_of(v.contentIdentity.begin(),v.contentIdentity.end(),[](auto x){return x!=0;}) || std::any_of(v.sourceIdentity.begin(),v.sourceIdentity.end(),[](auto x){return x!=0;}) || std::any_of(v.authoredIdentity.begin(),v.authoredIdentity.end(),[](auto x){return x!=0;})) return fail("empty network has dangling authored data");
        return true;
    }
    for (const auto& object : source.objects)
        if (object.mutationPolicy.enabled || !object.mutationCommands.empty())
            return fail("vascular V1 does not support topology mutation or binding remapping");
    const auto limit=std::numeric_limits<std::uint32_t>::max();
    if (C>limit || E>limit/2u || S>limit || T>limit || X>limit || C+E>limit || C+T>limit || (S && C+T>(limit-C-E)/S)) return fail("network exceeds 32-bit arena capacity");
    const std::uint64_t N=C+E+(C+T)*S;
    if (N>limit || source.environmentCount==0 || N>limit/(std::uint64_t(source.environmentCount)*(NM_MIXED_FGMRES_RESTART+1u))) return fail("network exceeds GPU Krylov address capacity");
    if (!validIds(v.compartments)||!validIds(v.connections)||!validIds(v.species)||!validIds(v.tissues)||!validIds(v.exchanges)) return fail("stable identifiers must be nonzero and unique within each record kind");
    if(!positive(source.frameTimestep))return fail("base timestep is not positive finite representable Float32");
    const int exponent=clockExponent(float(source.frameTimestep));
    c.layout.clock={std::bit_cast<std::uint32_t>(std::int32_t(exponent)),0u,0u,0u};
    c.layout.counts={std::uint32_t(C),std::uint32_t(E),std::uint32_t(S),std::uint32_t(T)};
    c.layout.offsets={0u,std::uint32_t(C),std::uint32_t(C+E),std::uint32_t(C+E+C*S)};
    c.layout.ranges={std::uint32_t(X),0u,std::uint32_t(N),0u};
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
        if (!finite(x.elastanceMin)||!finite(x.elastanceMax)||!finite(x.periodSeconds)||!finite(x.activationStart)||!finite(x.activationEnd)||!finite(x.sourcePi))return fail("unrepresentable source waveform");
        NMVascularCompartmentGPU node{};
        node.identity={x.stableIdentifier,name,std::uint32_t(x.storageKind),std::uint32_t(x.pressureLaw)};
        node.compliance=f4(x.referenceVolume,x.referencePressure,x.compliance,x.externalPressure);
        node.elastance=f4(x.elastanceMin,x.elastanceMax,x.activationStart,x.activationEnd);
        node.waveform=f4(x.sourcePi);
        if(x.pressureLaw!=VascularPressureLaw::linearCompliance) {
            if(!periodTicks(x.periodSeconds,exponent,node.periodTicks))return fail("source period cannot be represented exactly by bounded native clock");
        } else if(x.periodSeconds!=0)return fail("linear compliance has unexpected period");
        c.unknowns[row]={f4(x.initialVolume,x.volumeScale,x.volumeScale,x.volumeResidualTolerance)};
        if(!validCompartment(node,c.unknowns[row]))return fail("unsupported storage or cardiac pressure law");
        if(x.storageKind==VascularStorageKind::storageDisplacement && (S||T||X))return fail("hydraulic storage displacement lacks absolute blood volume required by species/tissue exchange");
        ci[x.stableIdentifier]=row;c.compartments.push_back(node);
        if (!amount(c.layout.offsets.z+row*S,x.initialSpeciesAmounts)) return fail("invalid blood species amounts");
    }
    std::vector<std::vector<std::uint32_t>> edges(C),bloodX(C*S),tissueX(T*S);
    for (auto index:oe) {const auto& x=v.connections[index];const auto row=std::uint32_t(c.connections.size());
        if (!ci.contains(x.fromCompartment)||!ci.contains(x.toCompartment)||x.fromCompartment==x.toCompartment||!finite(x.resistance)||!finite(x.inertance)||!finite(x.orificeCoefficient)||!finite(x.initialFlow)||!positive(x.flowScale)||!positive(x.pressureScale)||!tolerance(x.flowResidualTolerance)||!finite(x.initialFlow/x.flowScale)) return fail("invalid connection endpoint, passive law, scale or tolerance");
        const auto a=ci[x.fromCompartment],b=ci[x.toCompartment];c.connections.push_back({{x.stableIdentifier,a,b,std::uint32_t(x.flowLaw)},f4(x.resistance,x.inertance,x.orificeCoefficient)});
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
    c.layout.ranges.y=std::uint32_t(c.tissueBindings.size());
    incidence(edges,c.connectionIncidence,c.connectionRanges);incidence(bloodX,c.bloodExchangeIncidence,c.bloodExchangeRanges);incidence(tissueX,c.tissueExchangeIncidence,c.tissueExchangeRanges);
    std::string error;if (!validateVascularLayout(world,&error)) return fail(error);return true;
}

bool validateVascularLayout(const CompiledWorld& world,std::string* error) {
    const auto& c=world.vascular;const auto fail=[&](std::string_view s){if(error&&error->empty())*error="vascular: "+std::string(s);return false;};
    const std::uint64_t C=c.compartments.size(),E=c.connections.size(),S=c.species.size(),T=c.tissues.size(),X=c.exchanges.size();
    const auto limit=std::numeric_limits<std::uint32_t>::max();
    if (C>limit||E>limit/2u||S>limit||T>limit||X>limit||C+E>limit||C+T>limit||(S && C+T>(limit-C-E)/S))return fail("cooked count overflow");
    const auto N=C+E+(C+T)*S;
    if (c.layout.counts.x!=C||c.layout.counts.y!=E||c.layout.counts.z!=S||c.layout.counts.w!=T||c.layout.ranges.x!=X||c.layout.ranges.y!=c.tissueBindings.size()||c.layout.ranges.z!=N||c.layout.ranges.w!=0||c.layout.offsets.x!=0||c.layout.offsets.y!=C||c.layout.offsets.z!=C+E||c.layout.offsets.w!=C+E+C*S||c.unknowns.size()!=N) return fail("layout counts or unknown offsets disagree");
    if (C==0) return (c.layout.clock.x==0&&c.layout.clock.y==0&&c.layout.clock.z==0&&c.layout.clock.w==0&&E==0&&S==0&&T==0&&X==0&&c.names.empty()&&c.tissueBindings.empty()&&c.connectionIncidence.empty()&&c.connectionRanges.empty()&&c.bloodExchangeIncidence.empty()&&c.bloodExchangeRanges.empty()&&c.tissueExchangeIncidence.empty()&&c.tissueExchangeRanges.empty()&&allZero(c.identity))||fail("empty network has dangling state");
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
    for (std::size_t i=0;i<N;++i) {const auto v=c.unknowns[i].initialAndScaling;if(!finite4(v)||!positive(v.y)||!positive(v.z)||!tolerance(v.w)||!finite(double(v.x)/v.y))return fail("invalid normalized unknown");if((i<C && c.compartments[i].identity.z==0 && !(v.x>0))||(i>=C+E&&v.x<0))return fail("negative amount or nonpositive volume");if((i<C||i>=C+E)&&v.y!=v.z)return fail("conservation row has inconsistent physical scale");}
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
    return true;
}
} // namespace numi::matter::detail
