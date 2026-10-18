#define NUMI_MATTER_METALLIB "/tmp/unused-Matter.metallib"
#define NUMI_HUMAN_RESPIRATION_METALLIB "/tmp/unused-Respiration.metallib"
#include "human_resting_runtime.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace numi::human;

static void expectReject(const auto& thunk) {
    bool rejected=false;
    try { thunk(); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
}

static void compileComposition(const char* networkPath,const char* bloodMaterialPath,
                              const char* femMaterialPath) {
    constexpr float dt=0.00025f;
    VascularNetworkSource network;
    std::string error;
    need(readHumanPhysiologyNetwork(networkPath,network,&error),
         "CVSim21 source parse failed: "+error);
    auto bloodParsed=parseMatterFile(bloodMaterialPath);
    need(bloodParsed.succeeded()&&
         bloodParsed.material.name=="blood_high_shear_newtonian_proxy",
         "compiler regression blood material parse failed");
    double density=0.0;
    for(const auto& parameter:bloodParsed.material.parameters)
        if(parameter.name=="density") density=parameter.defaultValue;
    need(std::isfinite(density)&&density>0.0,
         "compiler regression source density is invalid");
    auto femParsed=parseMatterFile(femMaterialPath);
    need(femParsed.succeeded()&&
         femParsed.material.name=="fem_rollback_history",
         "compiler regression 213 FEM material parse failed");

    WorldSource seed;
    seed.environmentCount=1u;
    seed.frameTimestep=static_cast<double>(dt);
    seed.gravity={0.25,-0.5,-9.81};
    seed.mixedSolver.relativeResidual=1.e-7;
    seed.mixedSolver.newtonIterations=12u;
    seed.mixedSolver.fgmresIterations=64u;
    seed.materials.push_back(std::move(femParsed.material));
    ObjectSource femObject;
    femObject.name="synthetic_fixed_base_fem_rollback_fixture";
    femObject.materialIndex=0u;
    femObject.representation=Representation::fem;
    femObject.automaticRepresentation=false;
    femObject.twoWayCoupling=false;
    femObject.adaptive=false;
    femObject.identifiable=false;
    femObject.deformableContact=false;
    femObject.deformableSelfContact=false;
    femObject.mixedFEM=false;
    femObject.characteristicLength=0.05;
    femObject.femNodes={{0,0,0},{0.05,0,0},{0,0.05,0},{0,0,0.05}};
    femObject.femFixedNodes={0,1,2};
    femObject.tetrahedra.push_back({{0,1,2,3}});
    seed.objects.push_back(std::move(femObject));

    auto source=composeRestingWorldSource(network,dt,&seed);
    ObjectSource bloodReceiver;
    bloodReceiver.name="external_blood_dormant_receiver";
    bloodReceiver.representation=Representation::mpm;
    bloodReceiver.automaticRepresentation=false;
    bloodReceiver.twoWayCoupling=false;
    bloodReceiver.adaptive=false;
    bloodReceiver.deformableContact=false;
    bloodReceiver.deformableSelfContact=false;
    bloodReceiver.characteristicLength=0.05;
    bloodReceiver.mpmGridMinimum={-0.2,-0.2,-0.2};
    bloodReceiver.mpmGridMaximum={0.2,0.2,0.2};
    for(unsigned slot=0;slot<128u;++slot) {
        ParticleSource particle;
        particle.position={0.0,0.0,0.0};
        particle.velocity={0.0,0.0,0.0};
        particle.referenceVolume=1.0e-6;
        particle.mass=density*particle.referenceVolume;
        particle.active=false;
        bloodReceiver.particles.push_back(particle);
    }
    appendRestingBloodMaterial(source,std::move(bloodParsed.material),
                               std::move(bloodReceiver));
    need(source.gravity==seed.gravity&&source.materials.size()==2u&&
         source.objects.size()==2u,
         "seed composition changed gravity or append ordering");
    const auto compiled=compileWorld(source,
        {.maximumRateExponent=NM_MAX_RATE_EXPONENT});
    for(const auto& diagnostic:compiled.diagnostics)
        if(diagnostic.severity==Diagnostic::Severity::error)
            error+=diagnostic.message+"; ";
    need(compiled.succeeded(),"combined source compiler rejected: "+error);
    need(compiled.world.objects.size()==2u&&
         compiled.world.objects[0].materialIndex==0u&&
         compiled.world.objects[1].materialIndex==1u&&
         compiled.world.constitutive.size()==2u,
         "compiled object/material offsets do not preserve seed and append order");
    need(compiled.world.dispatch.femNodeCount==4u&&
         compiled.world.dispatch.tetrahedronCount==1u&&
         compiled.world.fem.nodes.size()==4u&&
         compiled.world.fem.tetrahedra.size()==1u,
         "compiled FEM seed topology was lost or reallocated");
    const auto& tet=compiled.world.fem.tetrahedra.front();
    need(tet.nodes.x==0u&&tet.nodes.y==1u&&tet.nodes.z==2u&&tet.nodes.w==3u&&
         tet.identity.x==0u&&tet.identity.y==0u,
         "FEM seed material/object or node indices changed after append");
    need(compiled.world.objects[0].elementOffset==0u&&
         compiled.world.objects[0].elementCount==1u,
         "FEM seed object element range changed after append");
    need(compiled.world.mpm.particles.size()==128u&&
         compiled.world.vascular.compartments.size()==21u&&
         compiled.world.vascular.connections.size()==24u&&
         compiled.world.vascular.unknowns.size()==45u,
         "combined compiler output lost FEM, appended MPM, or CVSim21 capacity");
    std::cout<<"compiler_composition=PASS objects=2 materials=2 fem_material_index=0 "
                "blood_receiver_material_index=1 fem_nodes=4 fem_tets=1 "
                "mpm_particles=128 cvsim_unknowns=45 gravity_preserved=1\n";
}

int main(int argc,char** argv) {
    constexpr float dt=0.00025f;
    VascularNetworkSource network;
    network.contentIdentity[0]=0x1234u;

    const auto defaultWorld=composeRestingWorldSource(network,dt,nullptr);
    assert(defaultWorld.environmentCount==1u);
    assert(static_cast<float>(defaultWorld.frameTimestep)==dt);
    assert(defaultWorld.gravity[0]==0.0&&defaultWorld.gravity[1]==0.0&&
           defaultWorld.gravity[2]==0.0);
    assert(defaultWorld.mixedSolver.relativeResidual==1.e-7);
    assert(defaultWorld.mixedSolver.newtonIterations==12u);
    assert(defaultWorld.mixedSolver.fgmresIterations==64u);
    assert(defaultWorld.vascular.contentIdentity==network.contentIdentity);

    WorldSource seed;
    seed.environmentCount=1u;
    seed.frameTimestep=static_cast<double>(dt);
    seed.gravity={0.25,-0.5,-9.81};
    seed.mixedSolver.relativeResidual=2.e-6;
    seed.mixedSolver.newtonIterations=17u;
    seed.mixedSolver.fgmresIterations=91u;
    seed.mixedSolver.fgmresRestart=11u;
    MaterialProgram fascia; fascia.name="passive_fascia";
    seed.materials.push_back(fascia);
    ObjectSource tissue; tissue.name="fascia_region";
    tissue.representation=Representation::fem; tissue.materialIndex=0u;
    seed.objects.push_back(tissue);

    auto combined=composeRestingWorldSource(network,dt,&seed);
    assert(combined.environmentCount==seed.environmentCount);
    assert(combined.frameTimestep==seed.frameTimestep);
    assert(combined.gravity==seed.gravity);
    assert(combined.mixedSolver.relativeResidual==seed.mixedSolver.relativeResidual);
    assert(combined.mixedSolver.newtonIterations==seed.mixedSolver.newtonIterations);
    assert(combined.mixedSolver.fgmresIterations==seed.mixedSolver.fgmresIterations);
    assert(combined.mixedSolver.fgmresRestart==seed.mixedSolver.fgmresRestart);
    assert(combined.materials.size()==1u&&combined.materials[0].name=="passive_fascia");
    assert(combined.objects.size()==1u&&combined.objects[0].name=="fascia_region");
    assert(combined.vascular.contentIdentity==network.contentIdentity);
    assert(containsFEMObject(combined));

    MaterialProgram blood; blood.name="blood_high_shear_newtonian_proxy";
    ObjectSource receiver; receiver.name="external_blood_dormant_receiver";
    receiver.representation=Representation::mpm;
    appendRestingBloodMaterial(combined,blood,receiver);
    assert(combined.materials.size()==2u);
    assert(combined.materials[0].name=="passive_fascia");
    assert(combined.materials[1].name=="blood_high_shear_newtonian_proxy");
    assert(combined.objects.size()==2u);
    assert(combined.objects[0].name=="fascia_region"&&combined.objects[0].materialIndex==0u);
    assert(combined.objects[1].materialIndex==1u);
    assert(combined.gravity==seed.gravity);
    assert(seed.materials.size()==1u&&seed.objects.size()==1u);
    const std::array<double,3> expectedSeedGravity{0.25,-0.5,-9.81};
    assert(seed.gravity==expectedSeedGravity);

    auto badDt=seed; badDt.frameTimestep=0.0005;
    expectReject([&]{(void)composeRestingWorldSource(network,dt,&badDt);});
    auto badEnv=seed; badEnv.environmentCount=2u;
    expectReject([&]{(void)composeRestingWorldSource(network,dt,&badEnv);});
    auto badNetwork=seed; badNetwork.vascular.compartments.emplace_back();
    expectReject([&]{(void)composeRestingWorldSource(network,dt,&badNetwork);});
    auto duplicateBlood=seed; duplicateBlood.objects.front().name="external_blood_dormant_receiver";
    expectReject([&]{(void)composeRestingWorldSource(network,dt,&duplicateBlood);});
    expectReject([&]{appendRestingBloodMaterial(combined,blood,receiver);});

    std::cout<<"resting_world_source_composition=PASS default_compatibility=PASS "
                "seed_preservation=PASS append_only_material_index=1 "
                "conflicting_owner_rejection=PASS\n";
    if(argc==4) compileComposition(argv[1],argv[2],argv[3]);
    else if(argc!=1)
        throw std::runtime_error("expected zero args or network, blood material, and FEM material paths");
    return 0;
}
