#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "NumiHumanRestingAorticWall.hpp"
#include "metalrobo/engine_types.h"
#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace numi::human::aortic_wall;

namespace {
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
double detCols(const Vec& a,const Vec& b,const Vec& c){return dot(a,cross(b,c));}

std::pair<double,double> jacobianRange(const std::vector<Vec>& current,const ObjectSource& object){
    double lo=std::numeric_limits<double>::infinity(),hi=0.0;
    for(const auto& tet:object.tetrahedra){
        const auto n=tet.nodes;
        const Vec x0=current.at(n[0]),X0=object.femReferenceNodes.at(n[0]);
        const double dCurrent=detCols(sub(current.at(n[1]),x0),sub(current.at(n[2]),x0),sub(current.at(n[3]),x0));
        const double dReference=detCols(sub(object.femReferenceNodes.at(n[1]),X0),
                                        sub(object.femReferenceNodes.at(n[2]),X0),
                                        sub(object.femReferenceNodes.at(n[3]),X0));
        if(!(dCurrent>0.0&&dReference>0.0))throw std::runtime_error("nonpositive source/snapshot tetrahedron determinant");
        const double j=dCurrent/dReference;
        lo=std::min(lo,j);hi=std::max(hi,j);
    }
    return {lo,hi};
}
}
int main(){
    @autoreleasepool {
        try {
            SectionProof proof;
            proof.stableId=9u;
            proof.sourceBody=7u;
            proof.sourcePayloadSHA256="1a191fe2f6c4a1cb03ee290bc850b80011cc267577992ea86abb3a4cb39f4108";
            proof.planePoint={-0.03372473683003,0.05810004997253,-0.00946755672337};
            proof.planeNormal={0.0,1.0,0.0};
            proof.externalAreaM2=2.4192601236904974e-4;
#include "actual_nha_contour.inc"
            constexpr double pressurePa=11223.5;
            constexpr double targetReservoirVolumeM3=1.84183e-5;
            Config cfg;
            cfg.circumferentialSegments=12u;
            cfg.axialSegments=8u;
            MRBodyStateGPU identity{};
            identity.orientation={0.0f,0.0f,0.0f,1.0f};
            const Built built=build(proof,identity,{0.0,0.0,0.0},7u,
                                    identity,7u,targetReservoirVolumeM3,pressurePa,cfg);
            auto annular=makeAnnularMechanicalWall(built,proof,pressurePa);
            double externalForceSquared=0.0;
            for(const auto& f:annular.externalForces)
                for(unsigned k=0u;k<3u;++k)
                    externalForceSquared+=double(f[k])*double(f[k]);
            const double externalForceL2N=std::sqrt(externalForceSquared);
            require(std::isfinite(externalForceL2N)&&externalForceL2N>0.0,
                    "wall force-budget scale is not finite and positive");
            const char* budgetText=std::getenv("NUMI_AORTIC_WALL_FORCE_BUDGET_FRACTION");
            const double forceBudgetFraction=budgetText?std::strtod(budgetText,nullptr):5.0e-4;
            require(forceBudgetFraction==5.0e-4||forceBudgetFraction==2.5e-4,
                    "only the declared 0.05% and 0.025% local wall force budgets are admitted");
            // Controlled baseline: source current coordinates are also the
            // stress-free reference; this tests pressure response separately
            // from the inferred loaded-reference prestress.
            annular.object.femReferenceNodes=annular.object.femNodes;
            annular.object.femReferenceSourceIdentity={
                0x414e4e554c41522dULL,0x554e50524553532dULL,
                0x465245452d563031ULL,0x434f4e54524f4c31ULL};

            // Coordinate-invariance diagnostic: express every mechanical wall
            // position and source contour point relative to the wall-node
            // centroid. The inverse anatomical registration is the exact
            // translation by sourceWallCentroidM; no force, material, mesh,
            // timestep, or solver parameter changes in this case.
            require(!annular.object.femNodes.empty(),"wall mesh is empty before centering");
            require(annular.object.femReferenceNodes.size()==annular.object.femNodes.size(),
                    "current/reference wall node counts differ before centering");
            Vec sourceWallCentroidM{};
            for(const Vec& x:annular.object.femNodes)
                sourceWallCentroidM=add(sourceWallCentroidM,mul(x,1.0/double(annular.object.femNodes.size())));
            const Vec inverseRegistrationTranslationM=sourceWallCentroidM;
            auto translatePositions=[&](std::vector<Vec>& positions){
                for(Vec& x:positions)x=sub(x,sourceWallCentroidM);
            };
            translatePositions(annular.object.femNodes);
            translatePositions(annular.object.femReferenceNodes);
            for(Vec& x:proof.orderedExternalContour)x=sub(x,sourceWallCentroidM);
            proof.planePoint=sub(proof.planePoint,sourceWallCentroidM);
            auto positionBounds=[](const std::vector<Vec>& positions){
                std::pair<Vec,Vec> bounds;
                bounds.first={std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity()};
                bounds.second={-std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity()};
                for(const Vec& x:positions)for(unsigned k=0;k<3u;++k){
                    bounds.first[k]=std::min(bounds.first[k],x[k]);
                    bounds.second[k]=std::max(bounds.second[k],x[k]);
                }
                return bounds;
            };
            const auto translatedDoubleBounds=positionBounds(annular.object.femNodes);

            numi::matter::WorldSource source;
            source.frameTimestep=1.0/4000.0;
            const double impulseBudgetNs=source.frameTimestep*externalForceL2N*forceBudgetFraction;
            source.gravity={0.0,0.0,0.0};
            source.environmentCount=1u;
            source.mixedSolver.newtonIterations=12u;
            source.mixedSolver.fgmresIterations=64u;
            // Wall-only sensitivity experiment: convert a declared fraction of
            // external nodal force L2 into an impulse-norm certificate. This is
            // local to this source; no global solver gate is changed.
            source.mixedSolver.relativeResidual=impulseBudgetNs;
            source.materials.push_back(makeMaterial(cfg));
            annular.object.materialIndex=0u;
            source.objects.push_back(annular.object);
            numi::matter::CompileOptions options;
            options.maximumRateExponent=0u;
            const auto compile=numi::matter::compileWorld(source,options);
            if(!compile.succeeded()){
                for(const auto& d:compile.diagnostics)
                    std::cerr<<"compile_diagnostic="<<d.message<<"\n";
                throw std::runtime_error("wall-only Matter compile failed");
            }
            const auto& world=compile.world;
            const auto sourceJ=jacobianRange(annular.object.femNodes,annular.object);
            Vec storedFloatMin{std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity()};
            Vec storedFloatMax{-std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity()};
            for(const auto& node:world.fem.nodes){
                const auto& x=node.positionAndMass;
                const double p[3]={double(x.x),double(x.y),double(x.z)};
                for(unsigned k=0;k<3u;++k){
                    storedFloatMin[k]=std::min(storedFloatMin[k],p[k]);
                    storedFloatMax[k]=std::max(storedFloatMax[k],p[k]);
                }
            }
            require(world.objects.size()==1u,"wall-only world must contain exactly one object");
            require(world.fem.nodes.size()==annular.object.femNodes.size(),
                    "compiled wall FEM count differs from annular source");
            require(annular.object.femHumanAttachments.empty(),
                    "wall-only control must not retain a rigid Human attachment");
            require(!annular.object.femFixedNodes.empty(),
                    "proximal source attachments were not converted to fixed nodes");
            require(annular.externalForces.size()==world.fem.nodes.size(),
                    "annular pressure load does not cover every cooked FEM node");

            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            require(device!=nil,"no Metal device available");
            id<MTLCommandQueue> queue=[device newCommandQueue];
            require(queue!=nil,"could not create Metal command queue");
            id<MTLBuffer> worldStatusBuffer=[device
                newBufferWithLength:sizeof(MRMetalWorldStatusGPU)
                options:MTLResourceStorageModeShared];
            require(worldStatusBuffer!=nil,"could not allocate MetalWorld status buffer");
            id<MTLBuffer> externalForces=[device
                newBufferWithBytes:annular.externalForces.data()
                length:annular.externalForces.size()*sizeof(annular.externalForces[0])
                options:MTLResourceStorageModeShared];
            require(externalForces!=nil,"could not allocate FEM pressure force field");
            auto* worldStatus=static_cast<MRMetalWorldStatusGPU*>(worldStatusBuffer.contents);
            *worldStatus={};
            worldStatus->code=MR_STEP_SUCCESS;
            worldStatus->environment=0u;

            numi::matter::Runtime runtime;
            const auto initialized=runtime.initialize(world,{
                .metallib=NUMI_MATTER_METALLIB,
                .environmentCount=1u,
                .captureEvents=true,
                .captureDiagnostics=true,
                .automaticIdentification=false,
                .adaptiveTransfer=false,
            });
            require(initialized.encoded&&runtime.valid(),
                    initialized.message.empty()?"Matter Runtime initialization failed":initialized.message);
            const auto before=runtime.snapshot();
            require(before.available,"initial Matter snapshot unavailable");
            auto positionsFromSnapshot=[](const numi::matter::RuntimeStateSnapshot& snapshot){
                std::vector<Vec> positions;positions.reserve(snapshot.femNodes.size());
                for(const auto& node:snapshot.femNodes){
                    const auto& p=node.positionAndMass;
                    positions.push_back({p.x,p.y,p.z});
                }
                return positions;
            };
            const std::uint32_t requestedSteps=8u;
            const auto& initialSnapshot=before;
            require(initialSnapshot.femNodes.size()==world.fem.nodes.size(),
                    "initial Matter snapshot FEM count differs from compiled wall");
            std::vector<Vec> initialPositions=positionsFromSnapshot(initialSnapshot);
            std::vector<Vec> priorPositions=initialPositions;
            std::vector<Vec> acceptedPositions=initialPositions;
            double maxDisplacement=0.0;
            double cumulativeAcceptedExternalWorkJ=0.0;
            std::uint32_t acceptedSteps=0u;
            std::uint32_t lastStatusCode=NM_STATUS_SUCCESS;
            MTLCommandBufferStatus lastCommandStatus=MTLCommandBufferStatusNotEnqueued;
            numi::matter::RuntimeStateSnapshot after=initialSnapshot;
            for(std::uint32_t step=0u;step<requestedSteps;++step){
                // Select each controlStep for the existing read-only Newton
                // trace. Accepted Matter snapshots are read after every step;
                // first-failure file capture is disabled for this repeated run.
                const std::string rootText=std::to_string(step);
                setenv("NM_FEM_NEWTON_TRACE_ROOT",rootText.c_str(),1);

                id<MTLCommandBuffer> command=[queue commandBuffer];
                require(command!=nil,"could not create wall microstep command buffer");
                *worldStatus={};
                worldStatus->code=MR_STEP_SUCCESS;
                worldStatus->environment=0u;
                numi::matter::EncodeRequest request{};
                request.commandBuffer=(__bridge void*)command;
                request.environmentStatuses=(__bridge void*)worldStatusBuffer;
                request.femExternalForces=(__bridge void*)externalForces;
                request.femExternalForceCount=static_cast<std::uint32_t>(world.fem.nodes.size());
                request.controlStep=step;
                request.physicsSubstep=0u;
                request.physicsSubsteps=1u;
                request.timestepSeconds=runtime.timestepSeconds();
                request.runAdaptiveTransfer=false;
                request.phase=numi::matter::EncodePhase::preDynamics;
                auto encoded=runtime.encode(request);
                require(encoded.encoded,std::string("preDynamics encode failed: ")+encoded.message);
                request.phase=numi::matter::EncodePhase::postCommit;
                encoded=runtime.encode(request);
                require(encoded.encoded,std::string("postCommit encode failed: ")+encoded.message);
                [command commit];
                [command waitUntilCompleted];
                require(command.status==MTLCommandBufferStatusCompleted,
                        "wall microstep command buffer did not complete");
                lastCommandStatus=command.status;

                after=runtime.snapshot();
                require(after.available,"post-step Matter snapshot unavailable");
                require(after.femNodes.size()==before.femNodes.size(),
                        "post-step FEM snapshot length changed");
                const std::vector<Vec> stepPositions=positionsFromSnapshot(after);
                const auto stepJ=jacobianRange(stepPositions,annular.object);
                double stepMaxDisplacement=0.0;
                double stepExternalWorkJ=0.0;
                for(std::size_t i=0;i<stepPositions.size();++i){
                    const Vec delta=sub(stepPositions[i],priorPositions[i]);
                    stepMaxDisplacement=std::max(stepMaxDisplacement,norm(delta));
                    maxDisplacement=std::max(maxDisplacement,
                                             norm(sub(stepPositions[i],initialPositions[i])));
                    for(unsigned k=0;k<3u;++k)
                        stepExternalWorkJ+=double(annular.externalForces[i][k])*delta[k];
                }
                lastStatusCode=after.statuses.empty()?UINT32_MAX:after.statuses.front().code;
                const bool stepAccepted=lastStatusCode==NM_STATUS_SUCCESS;
                double stepKineticEnergyJ=0.0;
                if(stepAccepted){
                    const char* trajectoryPath=std::getenv("NUMI_AORTIC_WALL_TRAJECTORY_DIRECTORY");
                    if(trajectoryPath&&*trajectoryPath){
                        std::vector<std::array<float,6>> state;
                        state.reserve(after.femNodes.size());
                        for(const auto& node:after.femNodes){
                            const auto& x=node.positionAndMass;
                            const auto& v=node.velocityAndInverseMass;
                            state.push_back({x.x,x.y,x.z,v.x,v.y,v.z});
                        }
                        const std::string file=std::string(trajectoryPath)+"/step-"+
                            std::to_string(step)+".bin";
                        std::ofstream stream(file,std::ios::binary|std::ios::trunc);
                        require(stream.good(),"could not open accepted-state trajectory file");
                        const std::uint32_t count=static_cast<std::uint32_t>(state.size());
                        stream.write(reinterpret_cast<const char*>(&count),sizeof(count));
                        stream.write(reinterpret_cast<const char*>(state.data()),
                                     std::streamsize(state.size()*sizeof(state[0])));
                        require(stream.good(),"could not write accepted-state trajectory file");
                    }
                    for(const auto& node:after.femNodes){
                        const auto& massPosition=node.positionAndMass;
                        const auto& velocity=node.velocityAndInverseMass;
                        stepKineticEnergyJ+=0.5*double(massPosition.w)*
                            (double(velocity.x)*velocity.x+double(velocity.y)*velocity.y+
                             double(velocity.z)*velocity.z);
                    }
                    ++acceptedSteps;
                    cumulativeAcceptedExternalWorkJ+=stepExternalWorkJ;
                    acceptedPositions=stepPositions;
                    priorPositions=stepPositions;
                }
                std::cout<<std::setprecision(12)
                    <<"wall_step={\"control_step\":"<<step
                    <<",\"accepted\":"<<(stepAccepted?"true":"false")
                    <<",\"matter_status_code\":"<<lastStatusCode
                    <<",\"snapshot_min_j\":"<<stepJ.first
                    <<",\"snapshot_max_j\":"<<stepJ.second
                    <<",\"max_displacement_from_prior_m\":"<<stepMaxDisplacement
                    <<",\"frozen_load_external_work_j\":"<<stepExternalWorkJ
                    <<",\"kinetic_energy_j\":"<<stepKineticEnergyJ
                    <<",\"cumulative_accepted_external_work_j\":"<<cumulativeAcceptedExternalWorkJ
                    <<",\"fgmres_iterations\":"
                    <<(after.statuses.empty()?0u:after.statuses.front().fgmresIterations)
                    <<"}\n";
                if(!stepAccepted)break;
            }
            require(acceptedSteps>0u,"wall run did not produce an accepted microstep");
            const auto acceptedJ=jacobianRange(acceptedPositions,annular.object);
            Vec contourCentroid{};for(const auto& p:proof.orderedExternalContour)
                contourCentroid=add(contourCentroid,mul(p,1.0/double(proof.orderedExternalContour.size())));
            const Vec axis=unit(proof.planeNormal);
            double radialGrowthSum=0.0,maxRadialGrowth=0.0;
            for(std::size_t i=0;i<acceptedPositions.size();++i){
                Vec r0=sub(annular.object.femNodes[i],contourCentroid);
                Vec r1=sub(acceptedPositions[i],contourCentroid);
                r0=sub(r0,mul(axis,dot(r0,axis)));r1=sub(r1,mul(axis,dot(r1,axis)));
                const double growth=norm(r1)-norm(r0);
                radialGrowthSum+=growth;maxRadialGrowth=std::max(maxRadialGrowth,growth);
            }
            const double meanRadialGrowth=acceptedPositions.empty()?0.0:
                radialGrowthSum/double(acceptedPositions.size());
            std::cout<<std::setprecision(12)
                <<"schema=numi.aortic_wall.annular_wall_only_native.v1\n"
                <<"source_payload_sha256="<<proof.sourcePayloadSHA256<<"\n"
                <<"source_stable_id="<<proof.stableId<<"\n"
                <<"source_body="<<proof.sourceBody<<"\n"
                <<"wall_only=true\n"
                <<"reference_mode=identity_unprestressed_control\n"
                <<"coordinate_control=wall_centroid_translated_to_local_origin\n"
                <<"coordinate_translation_m="<<sourceWallCentroidM[0]<<","<<sourceWallCentroidM[1]<<","<<sourceWallCentroidM[2]<<"\n"
                <<"inverse_registration_translation_m="<<inverseRegistrationTranslationM[0]<<","<<inverseRegistrationTranslationM[1]<<","<<inverseRegistrationTranslationM[2]<<"\n"
                <<"translated_double_bounds_min_m="<<translatedDoubleBounds.first[0]<<","<<translatedDoubleBounds.first[1]<<","<<translatedDoubleBounds.first[2]<<"\n"
                <<"translated_double_bounds_max_m="<<translatedDoubleBounds.second[0]<<","<<translatedDoubleBounds.second[1]<<","<<translatedDoubleBounds.second[2]<<"\n"
                <<"compiled_float32_bounds_min_m="<<storedFloatMin[0]<<","<<storedFloatMin[1]<<","<<storedFloatMin[2]<<"\n"
                <<"compiled_float32_bounds_max_m="<<storedFloatMax[0]<<","<<storedFloatMax[1]<<","<<storedFloatMax[2]<<"\n"
                <<"fixed_anchor_indices_translated_with_node_positions="<<annular.object.femFixedNodes.size()<<"\n"
                <<"vascular_compartments="<<world.vascular.compartments.size()<<"\n"
                <<"vascular_cavities="<<world.vascular.cavities.size()<<"\n"
                <<"fem_nodes="<<world.fem.nodes.size()<<"\n"
                <<"fem_tetrahedra="<<world.fem.tetrahedra.size()<<"\n"
                <<"fixed_proximal_nodes="<<annular.object.femFixedNodes.size()<<"\n"
                <<"human_attachments="<<annular.object.femHumanAttachments.size()<<"\n"
                <<"pressure_pa="<<annular.pressurePa<<"\n"
                <<"external_nodal_force_l2_n="<<externalForceL2N<<"\n"
                <<"local_force_budget_fraction="<<forceBudgetFraction<<"\n"
                <<"derived_impulse_budget_ns="<<impulseBudgetNs<<"\n"
                <<"solver_certificate_parameter_semantics=local_wall_impulse_norm_budget_not_generic_dimensionless_relative_error\n"
                <<"sidewall_pressure_area_m2="<<annular.sidewallAreaM2<<"\n"
                <<"closed_end_resultant_n="<<annular.endForceN<<"\n"
                <<"external_net_force_n="<<norm(annular.netExternalForceN)<<"\n"
                <<"timestep_s="<<runtime.timestepSeconds()<<"\n"
                <<"solver_newton_iterations="<<source.mixedSolver.newtonIterations<<"\n"
                <<"solver_fgmres_iterations="<<source.mixedSolver.fgmresIterations<<"\n"
                <<"solver_relative_residual="<<source.mixedSolver.relativeResidual<<"\n"
                <<"source_fem_minimum_j="<<sourceJ.first<<"\n"
                <<"source_fem_maximum_j="<<sourceJ.second<<"\n"
                <<"snapshot_fem_minimum_j="<<acceptedJ.first<<"\n"
                <<"snapshot_fem_maximum_j="<<acceptedJ.second<<"\n"
                <<"command_status="<<static_cast<unsigned>(lastCommandStatus)<<"\n"
                <<"matter_status_count="<<after.statuses.size()<<"\n";
            if(!after.statuses.empty()){
                const auto& s=after.statuses.front();
                std::cout<<"matter_status_code="<<s.code<<"\n"
                    <<"matter_status_object="<<s.objectIndex<<"\n"
                    <<"matter_status_failing_index="<<s.failingIndex<<"\n"
                    <<"completed_microsteps="<<s.completedMicrosteps<<"\n"
                    <<"fgmres_iterations="<<s.fgmresIterations<<"\n"
                    <<"matter_status_diagnostics_raw_xyzw="<<s.diagnostics.x<<","<<s.diagnostics.y<<","<<s.diagnostics.z<<","<<s.diagnostics.w<<"\n";
            }
            double finalKineticEnergyJ=0.0;
            for(const auto& node:after.femNodes){
                const auto& massPosition=node.positionAndMass;
                const auto& velocity=node.velocityAndInverseMass;
                finalKineticEnergyJ+=0.5*double(massPosition.w)*
                    (double(velocity.x)*velocity.x+double(velocity.y)*velocity.y+
                     double(velocity.z)*velocity.z);
            }
            std::cout<<"max_nodal_displacement_m="<<maxDisplacement<<"\n"
                <<"final_fem_kinetic_energy_j="<<finalKineticEnergyJ<<"\n"
                <<"accepted_trajectory_dump_enabled="
                <<((std::getenv("NUMI_AORTIC_WALL_TRAJECTORY_DIRECTORY")&&
                    *std::getenv("NUMI_AORTIC_WALL_TRAJECTORY_DIRECTORY"))?"true":"false")<<"\n"
                <<"mean_radial_growth_m="<<meanRadialGrowth<<"\n"
                <<"max_radial_growth_m="<<maxRadialGrowth<<"\n"
                <<"runtime_control_step="<<after.controlStep<<"\n"
                <<"requested_consecutive_steps="<<requestedSteps<<"\n"
                <<"accepted_consecutive_steps="<<acceptedSteps<<"\n"
                <<"last_matter_status_code="<<lastStatusCode<<"\n"
                <<"cumulative_accepted_frozen_load_external_work_j="<<cumulativeAcceptedExternalWorkJ<<"\n"
                <<"accepted_simulated_time_s="<<double(acceptedSteps)*runtime.timestepSeconds()<<"\n"
                <<"internal_energy_snapshot_available=false\n"
                <<"result="<<(acceptedSteps==requestedSteps
                                ?"ACCEPTED_EIGHT_CONSECUTIVE_STEPS":"RECORDED_NATIVE_REJECTION")<<"\n";
            return 0;
        } catch(const std::exception& e){
            std::cerr<<"wall_probe_error="<<e.what()<<"\n";
            return 2;
        }
    }
}
