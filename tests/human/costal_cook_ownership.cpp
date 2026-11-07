#include "metalrobo/NumiHumanTissueBinding.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
mr_float4 f4(
    const double x,
    const double y,
    const double z,
    const double w = 0.0
) {
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z),
        static_cast<float>(w),
    };
}

MRDofPropertiesGPU rootDof(const std::uint32_t localDof) {
    MRDofPropertiesGPU result{};
    result.articulationIndex = 0u;
    result.jointIndex = MR_INVALID_INDEX;
    result.qIndex = localDof < 3u
        ? localDof
        : MR_INVALID_INDEX;
    result.vIndex = localDof;
    result.localDof = localDof;
    result.flags = MR_DOF_FLAG_ROOT;
    return result;
}

MRDofPropertiesGPU passiveJointDof(
    const std::uint32_t jointIndex,
    const std::uint32_t qIndex,
    const std::uint32_t vIndex
) {
    MRDofPropertiesGPU result{};
    result.articulationIndex = 0u;
    result.jointIndex = jointIndex;
    result.qIndex = qIndex;
    result.vIndex = vIndex;
    return result;
}

MRBodyPropertiesGPU body(
    const std::uint32_t parent,
    const std::uint32_t inboundJoint,
    const double mass,
    const std::array<double, 3> centerOfMassInLink,
    const std::array<double, 3> inertia
) {
    MRBodyPropertiesGPU result{};
    result.articulationIndex = 0u;
    result.parentBody = parent;
    result.inboundJoint = inboundJoint;
    result.motionType = MR_MOTION_DYNAMIC;
    result.massAndInverseMass = f4(mass, 1.0 / mass, 0.0, 0.0);
    // Metadata remains the URDF link-frame offset. Dynamics translation and
    // joint anchors are COM-based.
    result.centerOfMass = f4(
        centerOfMassInLink[0],
        centerOfMassInLink[1],
        centerOfMassInLink[2]
    );
    result.inertiaRow0 = f4(inertia[0], 0.0, 0.0);
    result.inertiaRow1 = f4(0.0, inertia[1], 0.0);
    result.inertiaRow2 = f4(0.0, 0.0, inertia[2]);
    result.inverseInertiaRow0 = f4(1.0 / inertia[0], 0.0, 0.0);
    result.inverseInertiaRow1 = f4(0.0, 1.0 / inertia[1], 0.0);
    result.inverseInertiaRow2 = f4(0.0, 0.0, 1.0 / inertia[2]);
    result.dampingAndSpeedLimits = f4(0.0, 0.0, 1.0e6, 1.0e6);
    return result;
}

metalrobo::EngineModel makePendulumModel(const double length) {
    metalrobo::EngineModel model;
    model.name = "analytic_one_link_pendulum";
    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FLOATING;
    articulation.firstBody = 0u;
    articulation.bodyCount = 2u;
    articulation.firstJoint = 0u;
    articulation.jointCount = 1u;
    articulation.qOffset = 0u;
    articulation.nq = 8u;
    articulation.vOffset = 0u;
    articulation.nv = 7u;
    model.articulations.push_back(articulation);
    model.bodies.push_back(body(
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        3.0,
        {0.0, 0.0, 0.0},
        {0.3, 0.4, 0.5}
    ));
    model.bodies.push_back(body(
        0u,
        0u,
        1.7,
        {length, 0.0, 0.0},
        {0.08, 0.11, 0.16}
    ));

    MRJointDescriptorGPU joint{};
    joint.parentBody = 0u;
    joint.childBody = 1u;
    joint.jointType = MR_JOINT_REVOLUTE;
    joint.qOffset = 7u;
    joint.nq = 1u;
    joint.vOffset = 6u;
    joint.nv = 1u;
    joint.axis0 = f4(0.0, 0.0, 1.0);
    joint.parentAnchor = f4(0.0, 0.0, 0.0);
    // The joint is length metres in the -x direction from child COM.
    joint.childAnchor = f4(-length, 0.0, 0.0);
    joint.parentRotation = f4(0.0, 0.0, 0.0, 1.0);
    joint.childRotation = f4(0.0, 0.0, 0.0, 1.0);
    model.joints.push_back(joint);
    for (unsigned i=0; i<6; ++i) model.dofs.push_back(rootDof(i));
    model.dofs.push_back(passiveJointDof(0u, 7u, 6u));
    model.defaultQ = {0,0,0,0,0,0,1,0};
    model.defaultV.assign(7, 0.0f);
    model.world.abiVersion=MR_ENGINE_ABI_VERSION;
    model.world.bodyCount=2;model.world.articulationCount=1;model.world.jointCount=1;
    model.world.nq=8;model.world.nv=7;model.world.pairCapacity=1;model.world.contactCapacity=1;
    model.world.constraintCapacity=1;model.world.islandCapacity=1;model.world.solverType=MR_SOLVER_REFERENCE_FP64;
    model.world.frictionConeType=MR_FRICTION_CONE_ELLIPTIC;
    model.world.gravityAndTimestep=f4(0,-9.81,0,0.0001);
    model.world.solverScales=f4(1e-8,1e-9,2,1e-4);
    return model;
}


int main() {
 using namespace metalrobo;
 auto model=makePendulumModel(.1);
 std::string error;
 if (!model.valid(&error)){std::cerr << "invalid model: " << error << "\n";return 1;}
 auto material=numi::matter::parseMatter(R"MAT(material silicone {
    parameter density : kg/m^3 = 1100 in [900, 1300];
    parameter mu : kPa = 150 in [5, 1000] log identifiable;
    parameter lambda : MPa = 3 in [0.1, 30] log identifiable;

    model neo_hookean;
    energy = neo_hookean(mu, lambda);
    valid = J() - 0.05;
    supports mpm, fem, rigid;

    interface {
        static_friction = 0.80;
        dynamic_friction = 0.65;
        restitution = 0.0;
        adhesion = 0.0;
    }

    limits {
        minimum_J = 0.05;
        maximum_J = 20.0;
        maximum_stress = 1e9;
        maximum_energy_density = 1e9;
    }
}
)MAT");
 if(!material.succeeded()){std::cerr<<"material parse failed\n";return 1;}
 NumiHumanCostalBinding binding;
 binding.bodyCount=2;binding.nodeCount=4;binding.tetrahedronCount=1;
 binding.atlasToWorld={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
 binding.regions.resize(14,{MR_INVALID_INDEX,0,1});binding.nodeDonorBodies={0,0,1,1};
 NumiHumanCostalCartilagePayload cart;cart.regions.resize(14);
 cart.nodes={{{0,0,0},0,1,0},{{.01f,0,0},0,2,0},{{0,.01f,0},0,0,0},{{0,0,.01f},0,0,0}};
 cart.tetrahedra.push_back({{0,1,2,3},0});
 auto result=compileNumiHumanCostalTissue(cart,binding,model,material.material,1e-4);
 if(!result.succeeded()){std::cerr<<result.error<<"\n";return 1;}

 const auto& world=result.matter.world;
 if(world.fem.humanAttachments.size()!=2 || result.mass.partitions.size()!=2) return 2;
 if(!validateNumiHumanCostalAttachmentOwnership(cart,binding,world,error)){
     std::cerr << error << "\n";return 3;
 }
 for(const auto &a:world.fem.humanAttachments)
     if(a.identity.x>1 || a.identity.y!=a.identity.x) return 4;
 auto badWorld=world;
 badWorld.fem.humanAttachments[0].identity.y=1;
 if(validateNumiHumanCostalAttachmentOwnership(cart,binding,badWorld,error) || error.empty()) return 5;
 if(!validateNumiHumanCostalAttachmentOwnership(cart,binding,world,error)) return 6;
 auto badBinding=binding;badBinding.nodeDonorBodies[2]=2;
 const auto rejected=compileNumiHumanCostalTissue(cart,badBinding,model,material.material,1e-4);
 if(rejected.succeeded() || !rejected.mass.partitions.empty()) return 7;
 if(model.bodies.size()!=2 || model.bodies[0].massAndInverseMass.x!=3.0f ||
    model.bodies[1].massAndInverseMass.x!=1.7f) return 8;
 auto fixed=makePendulumModel(.1);
 fixed.articulations[0].rootType=MR_ROOT_FIXED;
 fixed.articulations[0].nq=1;fixed.articulations[0].nv=1;
 fixed.joints[0].qOffset=0;fixed.joints[0].vOffset=0;
 fixed.dofs={passiveJointDof(0,0,0)};fixed.defaultQ={0};fixed.defaultV={0};
 fixed.world.nq=1;fixed.world.nv=1;
 auto fixedResult=compileNumiHumanCostalTissue(cart,binding,fixed,material.material,1e-4);
 if(fixedResult.succeeded() || fixedResult.error.find("fixed-root donor")==std::string::npos) return 9;
 std::vector<double> q(result.rebasedHuman.defaultQ.begin(),result.rebasedHuman.defaultQ.end());
 std::vector<double> v(result.rebasedHuman.defaultV.begin(),result.rebasedHuman.defaultV.end());
 std::vector<ArticulatedPointQuery> points;
 for(const auto& a:world.fem.humanAttachments)
     points.push_back({a.identity.y,{a.localPoint.x,a.localPoint.y,a.localPoint.z}});
 std::vector<ArticulatedPointKinematics> positions(2);
 std::vector<double> jacobians(2*3*v.size());
 if(!computeArticulatedPointJacobians(result.rebasedHuman,0,q,v,points,positions,jacobians).succeeded()) return 10;
 double difference=0;
 for(unsigned axis=0;axis<3;++axis)
     difference=std::max(difference,std::abs(jacobians[(3+axis)*v.size()+6]-jacobians[axis*v.size()+6]));
 if(difference<0.001) {std::cerr<<"no relative motion\n";return 11;}
 std::cout<<"two donors and distinct attachment bodies passed; relative Jacobian "<<difference
          <<"; malformed ownership rejected; fixed-root rejection retained\n";
}
