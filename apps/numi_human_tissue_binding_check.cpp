#include "metalrobo/NumiHumanTissueBinding.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace {
void require(bool value,const std::string& error){if(!value)throw std::runtime_error(error);}
std::vector<std::byte> read(const char* path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    require(f.good(),"cannot open compiler input");
    const auto size=f.tellg();require(size>0&&size<64*1024*1024,"invalid compiler input size");
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));f.seekg(0);
    f.read(reinterpret_cast<char*>(bytes.data()),size);require(f.good(),"truncated compiler input");return bytes;
}
std::array<std::uint8_t,32> hash(std::span<const std::byte> bytes) {
    std::array<std::uint8_t,32> h{};
    CC_SHA256(bytes.data(),static_cast<CC_LONG>(bytes.size()),h.data());return h;
}
template<class T> T take(std::span<const std::byte> bytes,std::size_t& at) {
    require(at<=bytes.size()&&sizeof(T)<=bytes.size()-at,"truncated rigid payload");
    T value{};std::memcpy(&value,bytes.data()+at,sizeof(T));at+=sizeof(T);return value;
}
template<class T> std::vector<T> records(std::span<const std::byte> bytes,std::size_t& at,std::size_t n) {
    require(n<=bytes.size()/sizeof(T),"invalid rigid record count");
    std::vector<T> result;result.reserve(n);for(std::size_t i=0;i<n;++i)result.push_back(take<T>(bytes,at));return result;
}
metalrobo::EngineModel loadRigid(std::span<const std::byte> bytes) {
    std::size_t at=0;
    const auto magic=take<std::array<char,8>>(bytes,at);
    const auto h=take<std::array<std::uint32_t,10>>(bytes,at);
    (void)take<std::array<std::uint8_t,32>>(bytes,at);
    require(magic==std::array<char,8>{'N','H','R','I','G','I','D','2'}&&h[0]==1&&h[7]==0&&h[9]==0,"invalid NHRIGID2 header");
    require(bytes.size()==224ull+160ull*h[3]+144ull*h[4]+64ull*h[6]+4ull*(h[5]+h[6])+32ull*h[2],"NHRIGID2 size drift");
    metalrobo::EngineModel m;m.name="registered_costal_mass_compiler";
    m.world=take<MRWorldGPU>(bytes,at);
    m.articulations.push_back(take<MRArticulationGPU>(bytes,at));
    m.bodies=records<MRBodyPropertiesGPU>(bytes,at,h[3]);
    m.joints=records<MRJointDescriptorGPU>(bytes,at,h[4]);
    m.dofs=records<MRDofPropertiesGPU>(bytes,at,h[6]);
    m.defaultQ=records<float>(bytes,at,h[5]);m.defaultV=records<float>(bytes,at,h[6]);
    std::string error;require(m.valid(&error),"invalid source engine model: "+error);return m;
}
template<class T> void array(const T& values) {
    std::cout<<'[';bool comma=false;for(const auto v:values){if(comma)std::cout<<',';std::cout<<v;comma=true;}std::cout<<']';
}
}
int main(int argc,char** argv) {
    try {
        require(argc>=5,"usage: numi-human-tissue-binding-check BINDING CARTILAGE RIGID REGISTRATION [--metal] [--output-world=PATH]");
        bool useMetal=false;std::string outputWorld;
        for(int i=5;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--metal"&&!useMetal)useMetal=true;
            else if(arg.starts_with("--output-world=")&&outputWorld.empty())outputWorld=arg.substr(15);
            else require(false,"unknown or duplicated compiler argument");
        }
        const auto bindingBytes=read(argv[1]),cartBytes=read(argv[2]),rigidBytes=read(argv[3]),registrationBytes=read(argv[4]);
        metalrobo::NumiHumanCostalBinding binding;std::string error;
        require(metalrobo::decodeNumiHumanCostalBinding(bindingBytes,hash(cartBytes),hash(rigidBytes),binding,error),error);
        require(binding.registrationSHA256==hash(registrationBytes),"NHTBIND1 registration bytes mismatch");
        metalrobo::NumiHumanCostalCartilagePayload cart;
        require(metalrobo::decodeNumiHumanCostalCartilagePayload(cartBytes,{},cart).succeeded(),"invalid costal payload");
        const auto human=loadRigid(rigidBytes);
        const auto material=numi::matter::parseMatterFile(NUMI_COSTAL_MATERIAL);
        require(material.succeeded(),"costal material failed to parse");
        const auto result=metalrobo::compileNumiHumanCostalTissue(cart,binding,human,material.material,1e-5);
        require(result.succeeded(),result.error);
        if(!outputWorld.empty()) require(numi::matter::writePackage(result.matter,outputWorld,&error),error);
        metalrobo::EngineModel rebased;
        require(metalrobo::rebaseNumiHumanTissueMassPartition(human,result.mass.partitions,rebased,error),error);
        // Body points represent sites, wraps, anchors and sensor origins. Test
        // the same physical points in eight configurations, including their
        // analytic Jacobians and velocities, after a non-root COM rebase.
        std::vector<std::array<double,3>> offsets(human.bodies.size());
        for(const auto& p:result.mass.partitions) offsets[p.donorBody]=p.remainingCOMOffsetM;
        std::vector<metalrobo::ArticulatedPointQuery> oldPoints,newPoints;
        for(std::uint32_t b=0;b<human.bodies.size();++b) {
            for(const auto point:std::array<std::array<double,3>,4>{{{0,0,0},{.1,0,0},{0,.1,0},{0,0,.1}}}) {
                oldPoints.push_back({b,point});
                newPoints.push_back({b,{point[0]-offsets[b][0],point[1]-offsets[b][1],point[2]-offsets[b][2]}});
            }
        }
        const auto nv=human.defaultV.size();
        double maxPointError=0,maxJacobianError=0,maxVelocityError=0;
        double maxMetalPointError=0,maxMetalJacobianError=0,maxMetalMassError=0;
        std::string deviceName;
        std::unique_ptr<metalrobo::MetalArticulatedOperatorContext> gpu;
        if(useMetal) gpu=std::make_unique<metalrobo::MetalArticulatedOperatorContext>(
            metalrobo::MetalArticulatedOperatorConfig{.pointJacobiansOnly=true});
        for(unsigned sample=0;sample<8;++sample) {
            std::vector<double> q(human.defaultQ.begin(),human.defaultQ.end());
            std::vector<double> v(human.defaultV.begin(),human.defaultV.end());
            for(const auto& dof:human.dofs) if(dof.qIndex!=MR_INVALID_INDEX)
                q[dof.qIndex]+=0.02*std::sin(double(sample)*double(dof.qIndex+1));
            for(unsigned i=0;i<v.size();++i) v[i]=0.05*std::sin(double(sample)*double(i+1));
            std::vector<metalrobo::ArticulatedPointKinematics> before(oldPoints.size()),after(oldPoints.size());
            std::vector<double> jBefore(oldPoints.size()*3*nv),jAfter(jBefore.size());
            require(metalrobo::computeArticulatedPointJacobians(human,0,q,v,oldPoints,before,jBefore).succeeded(),"source point kinematics failed");
            require(metalrobo::computeArticulatedPointJacobians(rebased,0,q,v,newPoints,after,jAfter).succeeded(),"rebased point kinematics failed");
            for(unsigned i=0;i<before.size();++i) for(unsigned a=0;a<3;++a) {
                maxPointError=std::max(maxPointError,std::abs(before[i].position[a]-after[i].position[a]));
                maxVelocityError=std::max(maxVelocityError,std::abs(before[i].linearVelocity[a]-after[i].linearVelocity[a]));
            }
            for(unsigned i=0;i<jBefore.size();++i) maxJacobianError=std::max(maxJacobianError,std::abs(jBefore[i]-jAfter[i]));
            if(gpu) {
                std::vector<float> qf(q.begin(),q.end()),vf(v.begin(),v.end());
                // CPU oracle must start from the same final FP32 inputs.
                q.assign(qf.begin(),qf.end());v.assign(vf.begin(),vf.end());
                std::vector<MRArticulatedPointImpulseGPU> points;
                std::vector<metalrobo::ArticulatedPointQuery> packedQueries;
                for(const auto& p:newPoints) {
                    MRArticulatedPointImpulseGPU point{};point.bodyIndex=p.bodyIndex;
                    point.localPoint={static_cast<float>(p.localPoint[0]),static_cast<float>(p.localPoint[1]),static_cast<float>(p.localPoint[2]),0};
                    points.push_back(point);
                    packedQueries.push_back({p.bodyIndex,{point.localPoint.x,point.localPoint.y,point.localPoint.z}});
                }
                require(metalrobo::computeArticulatedPointJacobians(rebased,0,q,v,packedQueries,after,jAfter).succeeded(),"packed FP64 point oracle failed");
                metalrobo::MetalArticulatedOperatorInput input;
                input.environmentCount=1;input.pointCount=points.size();input.q=qf;input.v=vf;input.points=points;
                metalrobo::MetalArticulatedOperatorResult actual,replay;
                const auto run=gpu->run(rebased,input,actual);
                require(run.succeeded()&&run.dispatched&&run.published&&run.failedEnvironmentCount==0,
                        "rebased Metal dynamics failed: "+run.message);
                deviceName=run.deviceName;
                require(gpu->run(rebased,input,replay).succeeded(),"rebased Metal replay failed");
                require(actual.pointJacobians==replay.pointJacobians,
                        "rebased Metal replay changed");
                require(actual.pointJacobians.size()==jAfter.size(),"rebased GPU output coverage mismatch");
                for(unsigned i=0;i<after.size();++i) {
                    const auto& p=actual.pointWorld[i].position;
                    const std::array<double,3> pos{p.x,p.y,p.z};
                    for(unsigned a=0;a<3;++a) maxMetalPointError=std::max(maxMetalPointError,std::abs(pos[a]-after[i].position[a]));
                }
                for(unsigned i=0;i<jAfter.size();++i) maxMetalJacobianError=std::max(maxMetalJacobianError,std::abs(actual.pointJacobians[i]-jAfter[i]));
            }
        }
        if(gpu) for(const auto& partition:result.mass.partitions) {
            // The dense diagnostic bucket cannot hold the 157-body Human.
            // Test each changed full inertia tensor as a free rigid body;
            // full-body coupled dynamics remains a separate runtime gate.
            auto bodyModel=metalrobo::makeFreeSphereEngineModel();
            const auto index=bodyModel.articulations[0].rootBody;
            const auto topology=bodyModel.bodies[index];
            bodyModel.bodies[index]=partition.remainingBody;
            auto& b=bodyModel.bodies[index];
            b.articulationIndex=topology.articulationIndex;b.parentBody=topology.parentBody;b.inboundJoint=topology.inboundJoint;
            std::vector<MRArticulatedPointImpulseGPU> point(1);point[0].bodyIndex=index;
            metalrobo::MetalArticulatedOperatorInput input;
            input.environmentCount=1;input.pointCount=1;input.q=bodyModel.defaultQ;input.v=bodyModel.defaultV;input.points=point;
            metalrobo::MetalArticulatedOperatorResult actual;
            const auto run=metalrobo::runMetalArticulatedOperator(bodyModel,input,actual,
                {.writeDiagnosticMassMatrix=true});
            require(run.succeeded()&&run.dispatched&&run.published&&run.failedEnvironmentCount==0,"rebased donor Metal inertia failed: "+run.message);
            const auto count=bodyModel.defaultV.size();
            std::vector<double> matrix(count*count),q(bodyModel.defaultQ.begin(),bodyModel.defaultQ.end());
            require(metalrobo::computeArticulatedMassMatrix(bodyModel,0,q,matrix).succeeded(),"rebased donor FP64 inertia failed");
            require(actual.diagnosticMassMatrix.size()==matrix.size(),"donor mass matrix output coverage mismatch");
            for(unsigned i=0;i<matrix.size();++i) maxMetalMassError=std::max(maxMetalMassError,
                std::abs(actual.diagnosticMassMatrix[i]-matrix[i])/std::max(1.0,std::abs(matrix[i])));
        }
        require(maxPointError<2e-6&&maxVelocityError<2e-6&&maxJacobianError<2e-6,"COM rebase changed physical body points");
        require(maxMetalPointError<2e-5&&maxMetalJacobianError<2e-5&&maxMetalMassError<2e-5,"rebased Metal/FP64 mismatch");
        metalrobo::EngineModel untouched=rebased;
        require(!metalrobo::rebaseNumiHumanTissueMassPartition(rebased,result.mass.partitions,untouched,error),"mass partition applied twice");
        // Admission must fail atomically for wrong identity, reflection,
        // truncation, invalid bodies, shear and trailing data.
        unsigned rejects=0;
        const auto reject=[&](std::vector<std::byte> bytes){
            metalrobo::NumiHumanCostalBinding invalid;std::string why;
            require(!metalrobo::decodeNumiHumanCostalBinding(bytes,hash(cartBytes),hash(rigidBytes),invalid,why)
                    &&invalid.regions.empty(),"invalid binding was admitted");++rejects;};
        auto bad=bindingBytes;bad[32]^=std::byte{1};reject(bad);
        bad=bindingBytes;bad[64]^=std::byte{1};reject(bad);
        bad=bindingBytes;bad.pop_back();reject(bad);
        bad=bindingBytes;bad.push_back(std::byte{});reject(bad);
        bad=bindingBytes;double mirror=-binding.atlasToWorld[0];std::memcpy(bad.data()+128,&mirror,8);reject(bad);
        bad=bindingBytes;double shear=0.25;std::memcpy(bad.data()+136,&shear,8);reject(bad);
        bad=bindingBytes;std::uint32_t wrong=binding.bodyCount;std::memcpy(bad.data()+256,&wrong,4);reject(bad);
        std::cout<<std::setprecision(17)<<"{\n\"schema\":\"numi.human.tissue-mass-compilation.v1\",\n"
                 <<"\"status\":\"compiled_requires_v5_runtime_admission\",\n"
                 <<"\"matter_world_fingerprint\":"<<result.matter.world.fingerprint<<",\n"
                 <<"\"cooked_nodes\":"<<result.matter.world.fem.nodes.size()<<",\n"
                 <<"\"cooked_tetrahedra\":"<<result.matter.world.fem.tetrahedra.size()<<",\n"
                 <<"\"attachments\":"<<result.matter.world.fem.humanAttachments.size()<<",\n"
                 <<"\"maximum_attachment_rest_error_m\":"<<result.maximumAttachmentRestErrorM<<",\n"
                 <<"\"rebase_pose_cases\":8,\n\"rebase_points_per_pose\":"<<oldPoints.size()<<",\n"
                 <<"\"maximum_rebase_point_error_m\":"<<maxPointError<<",\n"
                 <<"\"maximum_rebase_velocity_error_m_s\":"<<maxVelocityError<<",\n"
                 <<"\"maximum_rebase_jacobian_error\":"<<maxJacobianError<<",\n"
                 <<"\"metal_executed\":"<<(useMetal?"true":"false")<<",\n"
                 <<"\"metal_device\":\""<<deviceName<<"\",\n"
                 <<"\"metal_replay_cases\":"<<(useMetal?8:0)<<",\n"
                 <<"\"maximum_metal_point_error_m\":"<<maxMetalPointError<<",\n"
                 <<"\"maximum_metal_jacobian_error\":"<<maxMetalJacobianError<<",\n"
                 <<"\"maximum_metal_donor_mass_matrix_scaled_error\":"<<maxMetalMassError<<",\n"
                 <<"\"whole_body_dynamic_mass_matrix_qualified\":false,\n"
                 <<"\"negative_admission_cases\":"<<rejects<<",\n\"partitions\":[";
        bool comma=false;
        for(const auto& p:result.mass.partitions) {
            if(comma)std::cout<<',';comma=true;
            std::cout<<"{\"donor_body\":"<<p.donorBody<<",\"node_count\":"<<p.nodeCount
                     <<",\"source_mass_kg\":"<<human.bodies[p.donorBody].massAndInverseMass.x
                     <<",\"tissue_mass_kg\":"<<p.tissueMassKg
                     <<",\"remaining_mass_kg\":"<<p.remainingBody.massAndInverseMass.x
                     <<",\"remaining_com_offset_m\":";array(p.remainingCOMOffsetM);
            std::cout<<",\"tissue_first_moment_kg_m\":";array(p.tissueFirstMomentKgM);
            std::cout<<",\"tissue_second_moment_kg_m2\":";array(p.tissueSecondMomentKgM2);
            std::cout<<",\"packed_moment_relative_error\":"<<p.packedMomentRelativeError<<'}';
        }
        std::cout<<"],\n\"production_owner_fraction\":0,\n\"rib_sternum_articulation_qualified\":false\n}\n";
        return 0;
    } catch(const std::exception& e){std::cerr<<"numi-human-tissue-binding-check: "<<e.what()<<'\n';return 1;}
}
