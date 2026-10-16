#include "metalrobo/NumiHumanTissueBinding.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace metalrobo {
namespace {
using Vec = std::array<double,3>;
Vec rotate(const std::array<double,4>& q, const Vec& p) {
    const Vec t{2*(q[1]*p[2]-q[2]*p[1]),2*(q[2]*p[0]-q[0]*p[2]),2*(q[0]*p[1]-q[1]*p[0])};
    return {p[0]+q[3]*t[0]+q[1]*t[2]-q[2]*t[1],
            p[1]+q[3]*t[1]+q[2]*t[0]-q[0]*t[2],
            p[2]+q[3]*t[2]+q[0]*t[1]-q[1]*t[0]};
}
Vec local(const ArticulatedBodyKinematics& b,const Vec& p) {
    const auto& q=b.orientation;
    return rotate({-q[0],-q[1],-q[2],q[3]},
                  {p[0]-b.centerOfMassPosition[0],p[1]-b.centerOfMassPosition[1],p[2]-b.centerOfMassPosition[2]});
}
template<class T> T take(std::span<const std::byte> bytes,std::size_t& offset) {
    T value{};std::memcpy(&value,bytes.data()+offset,sizeof(T));offset+=sizeof(T);return value;
}
NumiHumanCostalCompilation fail(const std::string& message) {
    NumiHumanCostalCompilation r;r.error=message;return r;
}
} // namespace

bool decodeNumiHumanCostalBinding(
    std::span<const std::byte> bytes, std::span<const std::uint8_t,32> cartHash,
    std::span<const std::uint8_t,32> rigidHash, NumiHumanCostalBinding& result, std::string& error) {
    result={};error.clear();
    const auto reject=[&](const char* why){error=why;return false;};
    if(bytes.size()!=424u) return reject("NHTBIND1 must contain exactly fourteen region bindings");
    std::size_t at=0;
    const auto magic=take<std::array<char,8>>(bytes,at);
    const auto h=take<std::array<std::uint32_t,6>>(bytes,at);
    const auto cart=take<std::array<std::uint8_t,32>>(bytes,at);
    const auto rigid=take<std::array<std::uint8_t,32>>(bytes,at);
    NumiHumanCostalBinding candidate;
    candidate.registrationSHA256=take<std::array<std::uint8_t,32>>(bytes,at);
    if(magic!=std::array<char,8>{'N','H','T','B','I','N','D','1'} || h[0]!=1 || h[1]!=14 ||
       h[2]==0 || h[2]>4096 || h[3]==0 || h[3]>100000 || h[4]==0 || h[4]>600000 || h[5]!=0)
        return reject("invalid NHTBIND1 header");
    if(!std::equal(cart.begin(),cart.end(),cartHash.begin()) ||
       !std::equal(rigid.begin(),rigid.end(),rigidHash.begin()) ||
       std::all_of(candidate.registrationSHA256.begin(),candidate.registrationSHA256.end(),[](auto v){return v==0;}))
        return reject("NHTBIND1 source identity mismatch");
    candidate.bodyCount=h[2];candidate.nodeCount=h[3];candidate.tetrahedronCount=h[4];
    candidate.atlasToWorld=take<std::array<double,16>>(bytes,at);
    const auto& a=candidate.atlasToWorld;
    if(!std::all_of(a.begin(),a.end(),[](double v){return std::isfinite(v);}) ||
       a[12]!=0 || a[13]!=0 || a[14]!=0 || a[15]!=1)
        return reject("NHTBIND1 has invalid affine registration");
    double scale2=0;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) scale2+=a[4*i+j]*a[4*i+j]/3;
    if(scale2<0.25 || scale2>4) return reject("NHTBIND1 anatomical scale is outside its admitted domain");
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        double gram=0;for(int k=0;k<3;++k) gram+=a[4*k+i]*a[4*k+j];
        if(std::abs(gram-(i==j?scale2:0))>1e-10*scale2) return reject("NHTBIND1 contains nonuniform scale or shear");
    }
    const double determinant=a[0]*(a[5]*a[10]-a[6]*a[9])-a[1]*(a[4]*a[10]-a[6]*a[8])+a[2]*(a[4]*a[9]-a[5]*a[8]);
    if(determinant<=0) return reject("NHTBIND1 registration reverses tissue orientation");
    for(unsigned i=0;i<14;++i) {
        const auto r=take<std::array<std::uint32_t,3>>(bytes,at);
        if(r[0]>=h[2] || r[0]!=r[1] || r[0]!=r[2])
            return reject("NHTBIND1 requires an explicit partition for a split thorax");
        candidate.regions.push_back({r[0],r[1],r[2]});
    }
    result=std::move(candidate);return true;
}

NumiHumanCostalCompilation compileNumiHumanCostalTissue(
    const NumiHumanCostalCartilagePayload& cartilage,
    const NumiHumanCostalBinding& binding, const EngineModel& human,
    const numi::matter::MaterialProgram& material, const double timestepSeconds) {
    if(binding.regions.size()!=14 || cartilage.regions.size()!=14 ||
       binding.nodeCount!=cartilage.nodes.size() || binding.tetrahedronCount!=cartilage.tetrahedra.size() ||
       binding.bodyCount!=human.bodies.size() || human.articulations.size()!=1 ||
       !std::isfinite(timestepSeconds) || timestepSeconds<=0)
        return fail("costal compiler input coverage mismatch");
    std::string error;
    if(!human.valid(&error)) return fail("invalid Human model: "+error);
    if(std::any_of(human.defaultV.begin(),human.defaultV.end(),[](float v){return v!=0;}))
        return fail("costal reference compiler requires a stationary authored reset");
    const std::vector<double> q(human.defaultQ.begin(),human.defaultQ.end()),v(human.defaultV.begin(),human.defaultV.end());
    std::vector<ArticulatedBodyKinematics> poses(human.bodies.size());
    if(!computeArticulatedBodyKinematics(human,0,q,v,poses).succeeded())
        return fail("costal source kinematics failed");
    numi::matter::WorldSource source;
    source.frameTimestep=timestepSeconds;
    source.gravity={human.world.gravityAndTimestep.x,human.world.gravityAndTimestep.y,human.world.gravityAndTimestep.z};
    source.articulatedDofCapacity=160u;source.articulatedQCapacity=161u;
    source.materials.push_back(material);
    numi::matter::ObjectSource object;
    object.name="human_registered_costal_cartilage";
    object.representation=numi::matter::Representation::fem;
    object.materialIndex=0u;object.mixedFEM=false;
    // Contact qualification is independent of this mass/binding compiler.
    object.deformableContact=false;object.deformableSelfContact=false;
    const auto& a=binding.atlasToWorld;
    for(std::uint32_t index=0;index<cartilage.nodes.size();++index) {
        const auto& n=cartilage.nodes[index];
        if(n.regionIndex>=binding.regions.size()) return fail("costal node region out of range");
        Vec point{};
        for(int i=0;i<3;++i) point[i]=static_cast<float>(a[4*i]*n.restPosition[0]+a[4*i+1]*n.restPosition[1]+a[4*i+2]*n.restPosition[2]+a[4*i+3]);
        object.femNodes.push_back(point);
        const auto& b=binding.regions[n.regionIndex];
        if(b.donorBody>=poses.size() || b.sternalBody>=poses.size() || b.ribBody>=poses.size())
            return fail("costal donor or attachment outside Human body table");
        if(n.flags!=0) {
            const auto body=(n.flags&NUMI_HUMAN_COSTAL_CARTILAGE_STERNAL_ATTACHMENT)?b.sternalBody:b.ribBody;
            object.femHumanAttachments.push_back({index,body,0x43410000u+index,local(poses[body],point)});
        }
    }
    for(const auto& t:cartilage.tetrahedra) object.tetrahedra.push_back({t.node});
    source.objects.push_back(std::move(object));
    numi::matter::CompileOptions options;options.maximumRateExponent=0u;
    NumiHumanCostalCompilation result;
    result.matter=numi::matter::compileWorld(source,options);
    if(!result.matter.succeeded()) {
        std::string reason="registered costal Matter cook failed";
        for(const auto& d:result.matter.diagnostics) reason+="; "+d.message;
        return fail(reason);
    }
    const auto& world=result.matter.world;
    if(world.fem.nodes.size()!=cartilage.nodes.size()) return fail("cooked costal node ownership changed");
    std::vector<NumiHumanTissueMassNode> nodes;
    for(std::uint32_t i=0;i<world.fem.nodes.size();++i) {
        const auto& n=world.fem.nodes[i];
        const auto donor=binding.regions[cartilage.nodes[i].regionIndex].donorBody;
        nodes.push_back({i,donor,n.positionAndMass.w,
                        local(poses[donor],{n.positionAndMass.x,n.positionAndMass.y,n.positionAndMass.z})});
    }
    result.mass=compileNumiHumanTissueMassPartition(human.bodies,nodes);
    if(!result.mass.succeeded()) return fail("registered costal mass partition failed: "+result.mass.error);
    for(const auto& attachment:world.fem.humanAttachments) {
        const auto& pose=poses[attachment.identity.y];
        const auto offset=rotate(pose.orientation,{attachment.localPoint.x,attachment.localPoint.y,attachment.localPoint.z});
        const auto& n=world.fem.nodes[attachment.identity.x].positionAndMass;
        const Vec actual{n.x,n.y,n.z};
        for(int i=0;i<3;++i) result.maximumAttachmentRestErrorM=std::max(result.maximumAttachmentRestErrorM,
            std::abs(pose.centerOfMassPosition[i]+offset[i]-actual[i]));
    }
    if(result.maximumAttachmentRestErrorM>16*std::numeric_limits<float>::epsilon())
        return fail("cooked costal attachment contains hidden rest deformation");
    if(!rebaseNumiHumanTissueMassPartition(human,result.mass.partitions,result.rebasedHuman,error))
        return fail("costal rigid frame rebase failed: "+error);
    for(auto& attachment:source.objects[0].femHumanAttachments) {
        for(const auto& p:result.mass.partitions) if(p.donorBody==attachment.bodyIndex)
            for(unsigned i=0;i<3;++i) attachment.localPoint[i]-=p.remainingCOMOffsetM[i];
    }
    // Recooking binds the changed local frames into the package fingerprint.
    // World-space rest geometry and its cooked mass measure are identical.
    auto rebased=numi::matter::compileWorld(source,options);
    if(!rebased.succeeded() || rebased.world.fem.nodes.size()!=world.fem.nodes.size())
        return fail("rebased costal Matter cook failed");
    for(unsigned i=0;i<world.fem.nodes.size();++i)
        if(std::memcmp(&rebased.world.fem.nodes[i].positionAndMass,
                       &world.fem.nodes[i].positionAndMass,sizeof(nm_float4))!=0)
            return fail("COM rebase changed cooked tissue mass or rest position");
    result.matter=std::move(rebased);
    return result;
}
} // namespace metalrobo
