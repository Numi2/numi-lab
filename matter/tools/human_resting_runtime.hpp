#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/human_physiology.hpp"
#include "human_respiration_parameters.hpp"
#include "../src/metal_encoder_timing.hpp"
#include "metalrobo/engine_types.h"
#include <NumiBrainRespiratoryChemoreflexV1.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>

namespace numi::human {
using namespace numi::matter;
inline void need(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
inline id<MTLBuffer> buffer(void* value){return (__bridge id<MTLBuffer>)value;}
struct Respiration {
    id<MTLDevice> device;
    id<MTLBuffer> accepted,candidate,excitation;
    id<MTLBuffer> bodyPropertiesReference=nil;
    std::uint64_t bodyPropertiesReferenceCount=0u;
    void* bodyPropertiesReferenceCaptureCommand=nullptr;
    id<MTLComputePipelineState> predict,exchange,resolve,bloodMpmHandoff=nil;
    id<MTLComputePipelineState> commonCoordinatesSolvePipeline=nil,commonCoordinateStatusPipeline=nil;
    id<MTLBuffer> commonGeometryParameters=nil,commonGeometryBoxes=nil,commonCandidateCoordinates=nil;
    bool commonGeometryGateEnabled=false;
    std::uint32_t expectedMicrotickCount=1u;
    bool compiledFEMSubcycling=false;
    bool parallelRespiratoryMuscles=false,parallelRespiratoryGas=false;
    NMHumanRespirationParameters parameters;
    NMHumanRespirationDispatch dispatch{};
    // Probe-only fault selector. The accepted-step dispatch is copied locally
    // for each callback; this selector never mutates persistent dispatch or
    // accepted controller history and is inert unless explicitly set.
    std::uint32_t diagnosticRejectAtControlStep = NM_INVALID_INDEX;
    // Borrowed owner metadata for this synchronous Matter encode call. The
    // callback uses the same GPU buffers as the transaction pass and validates
    // the command-buffer/step identity before binding them.
    struct PhysicsLease {
        void* commandBuffer = nullptr;
        std::uint32_t controlStep = NM_INVALID_INDEX;
        void* bodyPropertiesAccepted = nullptr;
        void* bodyPropertiesCandidate = nullptr;
        void* bodyPropertiesCheckpoint = nullptr;
        void* bodyPropertiesReference = nullptr;
        void* bodyPosesAccepted = nullptr;
        void* bodyPositionLow = nullptr;
        void* pointJacobians = nullptr;
        void* generalizedVelocity = nullptr;
        void* generalizedForceArena = nullptr;
        std::uint64_t bodyPropertyCount = 0u;
        std::uint64_t bodyPoseElementCount = 0u;
        std::uint64_t bodyPositionLowElementCount = 0u;
        std::uint64_t pointJacobianElementCount = 0u;
        std::uint64_t generalizedVelocityElementCount = 0u;
        std::uint64_t generalizedForceArenaElementCount = 0u;
        std::uint32_t bodyPoseStride = 0u;
        std::uint32_t pointCount = 0u;
        std::uint32_t pointJacobianStride = 0u;
        std::uint32_t bodyJacobianPointOffset = 0u;
        std::uint32_t articulationFirstBody = 0u;
        std::uint32_t dofCount = 0u;
        std::uint32_t vStride = 0u;
        std::uint32_t generalizedForceOffset = 0u;
        std::uint32_t generalizedForceStride = 0u;
        std::uint32_t bodyCount = 0u;
        bool active = false;
    } physicsLease{};
    std::function<bool(AcceptedStepExtensionPhase,const AcceptedStepExtensionView&)> brain;
    explicit Respiration(id<MTLDevice> dev,const CompiledWorld& world,const char* config,float dt):device(dev) {
        parameters=readRespirationParameters(config,dt);
        need(world.dispatch.maximumRateExponent<=NM_MAX_RATE_EXPONENT,
             "resting continuum scheduler exponent exceeds the runtime bound");
        expectedMicrotickCount=1u<<world.dispatch.maximumRateExponent;
        compiledFEMSubcycling=world.dispatch.femNodeCount!=0u&&
            world.dispatch.tetrahedronCount!=0u;
        need(expectedMicrotickCount==1u || world.dispatch.femNodeCount!=0u ||
             world.dispatch.particleCount!=0u,
             "resting subcycling requires a compiled continuum owner");
        dispatch={};
        dispatch.environmentCount=world.dispatch.environmentCount;
        dispatch.particleCount=world.dispatch.particleCount;
        dispatch.materialStateStride=world.dispatch.materialStateStride;
        dispatch.vascularStride=45u;
        dispatch.outletReleaseStep=NM_INVALID_INDEX;
        const float batchSteps=std::ceil(0.008f/parameters.environment.w);
        need(std::isfinite(batchSteps)&&batchSteps>=1.0f&&batchSteps<=float(UINT32_MAX),
             "blood packet interval must fit a 32-bit step count");
        dispatch.outletBatchSteps=static_cast<std::uint32_t>(batchSteps);
        dispatch.bloodMpmObjectIndex=NM_INVALID_INDEX;
        const char* bloodMpmSetting=std::getenv("NUMI_HUMAN_BLOOD_MPM_ENABLED");
        need(!bloodMpmSetting||std::strcmp(bloodMpmSetting,"0")==0||
             std::strcmp(bloodMpmSetting,"1")==0,
             "NUMI_HUMAN_BLOOD_MPM_ENABLED must be exactly 0 or 1");
        if(bloodMpmSetting&&std::strcmp(bloodMpmSetting,"1")==0) {
            constexpr const char* bloodMaterialName="blood_high_shear_newtonian_proxy";
            std::uint32_t receiver=NM_INVALID_INDEX;
            std::uint32_t receiverCount=0u;
            for(std::uint32_t index=0u;index<world.objects.size();++index) {
                const auto& object=world.objects[index];
                if(object.representation!=static_cast<std::uint32_t>(Representation::mpm)||
                   object.materialIndex>=world.constitutive.size()) continue;
                if(world.constitutive[object.materialIndex].material.name==bloodMaterialName) {
                    receiver=index;
                    ++receiverCount;
                }
            }
            need(receiverCount==1u,
                 "blood MPM mode requires exactly one compiled receiver object with the source-named high-shear blood material");
            const auto& object=world.objects[receiver];
            constexpr std::uint32_t requiredReceiverFlags =
                NM_OBJECT_DISABLE_DEFORMABLE_CONTACT | NM_OBJECT_DISABLE_SELF_CONTACT;
            need((object.flags & NM_OBJECT_ADAPTIVE) == 0u &&
                 (object.flags & requiredReceiverFlags) == requiredReceiverFlags,
                 "blood MPM receiver must be nonadaptive with deformable and self-contact disabled until the coupled path is qualified");
            need(object.stateCount>0u&&object.stateCount<=4096u&&
                 object.elementCount==1u&&object.elementOffset<world.mpm.grids.size()&&
                 std::uint64_t(object.stateOffset)+object.stateCount<=world.mpm.particles.size()&&
                 world.dispatch.particleCount==world.mpm.particles.size(),
                 "blood MPM receiver range/grid is absent or exceeds the bounded dormant-slot admission");
            const auto& grid=world.mpm.grids[object.elementOffset];
            const float density=world.materials[object.materialIndex].bulk.x;
            need(std::isfinite(density)&&density>0.0f&&
                 std::isfinite(grid.metrics.x)&&grid.metrics.x>0.0f&&
                 grid.nodeMinimumAndObject.w>=0&&
                 static_cast<std::uint32_t>(grid.nodeMinimumAndObject.w)==receiver&&
                 grid.nodeDimensionsAndOffset.x>2u&&
                 grid.nodeDimensionsAndOffset.y>2u&&
                 grid.nodeDimensionsAndOffset.z>2u,
                 "blood MPM receiver grid or source density is invalid");
            double capacityVolume=0.0;
            for(std::uint32_t local=0u;local<object.stateCount;++local) {
                const auto& particle=world.mpm.particles[object.stateOffset+local];
                need(particle.identity.x==receiver&&
                     particle.identity.y==object.materialIndex&&
                     particle.identity.z==object.topologyGeneration&&
                     particle.identity.w==0u&&
                     std::isfinite(particle.positionAndMass.w)&&particle.positionAndMass.w>0.0f&&
                     std::isfinite(particle.velocityAndReferenceVolume.w)&&
                     particle.velocityAndReferenceVolume.w>0.0f&&
                     std::abs(double(particle.positionAndMass.w)-
                         double(density)*particle.velocityAndReferenceVolume.w)<=
                         std::max(1.0e-12,4.0e-6*double(particle.positionAndMass.w)),
                     "blood MPM receiver slots must be finite, source-identified, density-matched, and dormant");
                capacityVolume+=particle.velocityAndReferenceVolume.w;
            }
            need(std::isfinite(capacityVolume)&&capacityVolume>0.0,
                 "blood MPM dormant receiver has no positive volume capacity");
            dispatch.bloodMpmEnabled=1u;
            dispatch.bloodMpmFirstParticle=object.stateOffset;
            dispatch.bloodMpmParticleCount=object.stateCount;
            dispatch.bloodMpmObjectIndex=receiver;
            dispatch.bloodMpmMaterialIndex=object.materialIndex;
            dispatch.bloodMpmTopologyGeneration=object.topologyGeneration;
            dispatch.bloodMpmMaterialStateCount=world.materials[object.materialIndex].stateCount;
            dispatch.bloodMpmDensity=density;
            dispatch.bloodMpmGridCellWidth=grid.metrics.x;
            dispatch.bloodMpmGridNodeMinimum=grid.nodeMinimumAndObject;
            dispatch.bloodMpmGridNodeDimensions=grid.nodeDimensionsAndOffset;
            std::cerr<<"blood_mpm_receiver=enabled object="<<receiver
                     <<" material="<<bloodMaterialName
                     <<" slots="<<object.stateCount
                     <<" dormant_capacity_ml="<<capacityVolume*1.0e6
                     <<" cell_width_m="<<grid.metrics.x
                     <<" density_kg_m3="<<density
                     <<" stock_owner=external_blood_inventory_subset\n";
        }
        auto numericEnvironment=[](const char* name,float fallback,float low,float high) {
            const char* value=std::getenv(name);
            if(!value||!value[0]) return fallback;
            char* end=nullptr;
            const float parsed=std::strtof(value,&end);
            need(end&&*end=='\0'&&std::isfinite(parsed)&&parsed>=low&&parsed<=high,
                 std::string(name)+" must be finite and within the admitted range");
            return parsed;
        };
        const char* release=std::getenv("NUMI_HUMAN_SEPARATION_STEP");
        if(release&&release[0]) {
            char* end=nullptr;
            const unsigned long step=std::strtoul(release,&end,10);
            need(end&&*end=='\0'&&step>0&&step<=UINT32_MAX,
                 "NUMI_HUMAN_SEPARATION_STEP must be a positive 32-bit step");
            dispatch.outletReleaseStep=static_cast<std::uint32_t>(step);
        }
        const char* bleedingSetting=std::getenv("NUMI_HUMAN_BLEEDING_ENABLED");
        need(!bleedingSetting||std::strcmp(bleedingSetting,"0")==0||
             std::strcmp(bleedingSetting,"1")==0,
             "NUMI_HUMAN_BLEEDING_ENABLED must be exactly 0 or 1");
        const bool bleedingEnabled=!bleedingSetting||std::strcmp(bleedingSetting,"1")==0;
        const float apertureFraction=numericEnvironment("NUMI_HUMAN_BLEEDING_AREA_FRACTION",0.25f,0.25f,1.0f);
        const float dischargeCoefficient=numericEnvironment("NUMI_HUMAN_BLEEDING_DISCHARGE_COEFFICIENT",0.6f,0.3f,0.8f);
        NSError* error=nil;
#ifdef NUMI_HUMAN_RESPIRATION_SOURCE
        NSString* shaderSource=[NSString stringWithContentsOfFile:@(NUMI_HUMAN_RESPIRATION_SOURCE)
            encoding:NSUTF8StringEncoding error:&error];
        need(shaderSource!=nil,"respiration source: "+std::string(error?error.localizedDescription.UTF8String:"missing"));
        MTLCompileOptions* compileOptions=[MTLCompileOptions new];
        compileOptions.languageVersion=MTLLanguageVersion4_0;
        compileOptions.mathMode=MTLMathModeSafe;compileOptions.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        auto library=[device newLibraryWithSource:shaderSource options:compileOptions error:&error];
#else
        auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(NUMI_HUMAN_RESPIRATION_METALLIB)] error:&error];
#endif
        need(library!=nil,"respiration library: "+std::string(error?error.localizedDescription.UTF8String:"missing"));
        auto pipeline=[&](NSString* name){
            id<MTLFunction> f=[library newFunctionWithName:name];
            auto result=f?[device newComputePipelineStateWithFunction:f error:&error]:nil;
            need(result!=nil,std::string("respiration pipeline: ")+name.UTF8String);return result;
        };
        auto enabled=[](const char* name) {
            const char* setting=std::getenv(name);
            need(!setting||std::strcmp(setting,"0")==0||std::strcmp(setting,"1")==0,
                 std::string(name)+" must be 0 or 1");
            return setting&&std::strcmp(setting,"1")==0;
        };
        const bool subcycle=enabled("NUMI_HUMAN_GAS_TRANSPORT_SUBCYCLING");
        const bool respiratorySubcycle=enabled("NUMI_HUMAN_RESPIRATORY_SUBCYCLING");
        parallelRespiratoryMuscles=enabled("NUMI_HUMAN_PARALLEL_RESPIRATORY_MUSCLES");
        parallelRespiratoryGas=enabled("NUMI_HUMAN_PARALLEL_RESPIRATORY_GAS");
        need(!parallelRespiratoryGas||parallelRespiratoryMuscles,
             "parallel respiratory gas requires parallel respiratory muscles");
        need(!parallelRespiratoryMuscles||respiratorySubcycle,
             "parallel respiratory muscles require respiratory subcycling");
        need(!respiratorySubcycle||subcycle,
             "respiratory subcycling requires gas transport subcycling");
        {
            // Metal requires specialization even when both options are false.
            auto constants=[[MTLFunctionConstantValues alloc] init];
            [constants setConstantValue:&subcycle type:MTLDataTypeBool atIndex:40];
            [constants setConstantValue:&respiratorySubcycle type:MTLDataTypeBool atIndex:41];
            [constants setConstantValue:&parallelRespiratoryMuscles type:MTLDataTypeBool atIndex:44];
            [constants setConstantValue:&parallelRespiratoryGas type:MTLDataTypeBool atIndex:46];
            auto specialized=[&](NSString* name) {
                auto f=[library newFunctionWithName:name constantValues:constants error:&error];
                auto result=f?[device newComputePipelineStateWithFunction:f error:&error]:nil;
                need(result!=nil,std::string("respiration subcycling pipeline: ")+name.UTF8String);
                return result;
            };
            predict=specialized(@"nm_human_respiration_predict");
            exchange=specialized(@"nm_human_respiration_exchange");
            if(dispatch.bloodMpmEnabled)
                bloodMpmHandoff=pipeline(@"nm_human_blood_mpm_handoff");
            need(!parallelRespiratoryMuscles||exchange.threadExecutionWidth>=2,
                 "parallel respiratory muscles require at least two SIMD lanes");
            need(!parallelRespiratoryGas||exchange.threadExecutionWidth>=21,
                 "parallel respiratory gas requires at least 21 SIMD lanes");
        }
        std::cerr<<"resting_gas_transport_subcycling="<<(subcycle?1:0)
                 <<" respiratory_mechanics_subcycling="<<(respiratorySubcycle?1:0)
                 <<" parallel_respiratory_muscles="<<(parallelRespiratoryMuscles?1:0)
                 <<" parallel_respiratory_gas="<<(parallelRespiratoryGas?1:0)
                 <<" maximum_internal_mechanics_dt_s=0.002"
                 <<" maximum_substeps=32 donor_fraction_bound=0.1"
                 <<" external_blood=operator_split_orifice_ballistic_packets"
                 <<" source_anatomy_id=9 donor_compartment_id=7 area_fraction="<<apertureFraction
                 <<" discharge_coefficient="<<dischargeCoefficient
                 <<" density_kg_m3=1050.6 cut_step="<<dispatch.outletReleaseStep
                 <<" bleeding_enabled="<<(bleedingEnabled?1:0)
                 <<" blood_mpm_enabled="<<dispatch.bloodMpmEnabled
                 <<" blood_mpm_object="<<dispatch.bloodMpmObjectIndex
                 <<" free_surface_solver=existing_mpm_high_shear_newtonian_reduced_scope\n";
        resolve=pipeline(@"nm_human_respiration_resolve");
        auto s=initializeRespiration(parameters,world);
        s.vascularCutOutlet.aperture.x=bleedingEnabled
            ?2.41926012e-4f*apertureFraction:0.0f;
        s.vascularCutOutlet.fluid.y=dischargeCoefficient;
        accepted=[device newBufferWithLength:sizeof(s)*dispatch.environmentCount options:MTLResourceStorageModeShared];
        candidate=[device newBufferWithLength:accepted.length options:MTLResourceStorageModeShared];
        excitation=[device newBufferWithLength:sizeof(nm_float4)*dispatch.environmentCount options:MTLResourceStorageModeShared];
        need(accepted&&candidate&&excitation,"respiration allocation");
        for(unsigned e=0;e<dispatch.environmentCount;++e) static_cast<NMHumanRespirationState*>(accepted.contents)[e]=s;
        std::memcpy(candidate.contents,accepted.contents,accepted.length);
        std::memset(excitation.contents,0,excitation.length);
    }
    bool captureBodyPropertiesReference(void* commandValue,void* sourceValue,std::uint64_t count) {
        if(bodyPropertiesReference) return count==bodyPropertiesReferenceCount;
        if(!commandValue||!sourceValue||count==0u||
           count>std::numeric_limits<NSUInteger>::max()/sizeof(MRBodyPropertiesGPU)) return false;
        id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)commandValue;
        id<MTLBuffer> source=buffer(sourceValue);
        if(!cb||!source||source.device.registryID!=device.registryID||
           source.length<count*sizeof(MRBodyPropertiesGPU)) return false;
        bodyPropertiesReference=[device newBufferWithLength:
            static_cast<NSUInteger>(count)*sizeof(MRBodyPropertiesGPU)
            options:MTLResourceStorageModePrivate];
        if(!bodyPropertiesReference) return false;
        id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
        if(!blit) {bodyPropertiesReference=nil;return false;}
        [blit copyFromBuffer:source sourceOffset:0u toBuffer:bodyPropertiesReference
            destinationOffset:0u size:static_cast<NSUInteger>(count)*sizeof(MRBodyPropertiesGPU)];
        [blit endEncoding];
        bodyPropertiesReferenceCount=count;
        bodyPropertiesReferenceCaptureCommand=commandValue;
        return true;
    }
    void discardBodyPropertiesReferenceCapture(void* commandValue) noexcept {
        if(bodyPropertiesReferenceCaptureCommand==commandValue) {
            bodyPropertiesReference=nil;
            bodyPropertiesReferenceCount=0u;
            bodyPropertiesReferenceCaptureCommand=nullptr;
        }
    }
    void* bodyPropertiesReferenceHandle() const noexcept {
        return (__bridge void*)bodyPropertiesReference;
    }
    bool stagePhysicsLease(const PhysicsLease& lease) noexcept {
        if(physicsLease.active||!lease.commandBuffer||
           lease.controlStep==NM_INVALID_INDEX) return false;
        physicsLease=lease;
        physicsLease.active=true;
        return true;
    }
    void clearPhysicsLease(void* commandValue=nullptr) noexcept {
        if(physicsLease.active&&
           (!commandValue||physicsLease.commandBuffer==commandValue))
            physicsLease=PhysicsLease{};
    }
    bool encode(AcceptedStepExtensionPhase phase,const AcceptedStepExtensionView& v) {
        const PhysicsLease& lease=physicsLease;
        const bool outletStep=dispatch.outletReleaseStep!=NM_INVALID_INDEX&&
            v.controlStep>=dispatch.outletReleaseStep;
        const auto rejectLease=[&](const char* reason) {
            if(outletStep||phase==AcceptedStepExtensionPhase::candidateReady) {
                std::cerr<<"resting_outlet_lease=reject reason="<<reason
                    <<" step="<<v.controlStep
                    <<" release_step="<<dispatch.outletReleaseStep
                    <<" phase="<<static_cast<unsigned>(phase)
                    <<" command_match="<<(lease.commandBuffer==v.commandBuffer)
                    <<" lease_active="<<lease.active
                    <<" body_count="<<lease.bodyCount
                    <<" property_count="<<lease.bodyPropertyCount
                    <<" pose_stride="<<lease.bodyPoseStride
                    <<" pose_count="<<lease.bodyPoseElementCount
                    <<" low_count="<<lease.bodyPositionLowElementCount
                    <<" jac_count="<<lease.pointJacobianElementCount
                    <<" jac_stride="<<lease.pointJacobianStride
                    <<" jac_offset="<<lease.bodyJacobianPointOffset
                    <<" first_body="<<lease.articulationFirstBody
                    <<" dof="<<lease.dofCount
                    <<" velocity_stride="<<lease.vStride
                    <<" velocity_count="<<lease.generalizedVelocityElementCount
                    <<" force_offset="<<lease.generalizedForceOffset
                    <<" force_stride="<<lease.generalizedForceStride
                    <<" force_count="<<lease.generalizedForceArenaElementCount
                    <<" buffers="<<bool(lease.bodyPropertiesAccepted)
                    <<bool(lease.bodyPropertiesCandidate)
                    <<bool(lease.bodyPropertiesReference)
                    <<bool(lease.bodyPosesAccepted)
                    <<bool(lease.bodyPositionLow)
                    <<bool(lease.pointJacobians)
                    <<bool(lease.generalizedVelocity)
                    <<bool(lease.generalizedForceArena)
                    <<"\n";
            }
            return false;
        };
        // The body/Brain transaction advances once per frame. Only the
        // continuum mechanics and circuit may subcycle inside it.
        // The articulated owner consumes the recoil force after this whole
        // preDynamics encode, for the full body-frame duration.
        const bool continuumSubcycles=expectedMicrotickCount>1u;
        if((lease.active&&(lease.commandBuffer!=v.commandBuffer||
                           lease.controlStep!=v.controlStep))||
           v.microtickCount==0u||v.microtick>=v.microtickCount||
           (v.microtickCount!=expectedMicrotickCount)||
           (continuumSubcycles&&!compiledFEMSubcycling&&!dispatch.bloodMpmEnabled)||
           v.environmentCount!=dispatch.environmentCount||
           v.vascularStateStride!=45||
           v.frameTimestepSeconds!=parameters.environment.w||
           v.microstepTimestepSeconds!=v.frameTimestepSeconds/float(v.microtickCount)||
           (continuumSubcycles&&!v.vascularCheckpoint))
            return rejectLease("callback_identity_or_cadence");
        if(continuumSubcycles&&
           (phase==AcceptedStepExtensionPhase::microstepComplete||
            (phase==AcceptedStepExtensionPhase::candidateReady&&
             v.microtick+1u<v.microtickCount))) return true;
        // Only the exchange stage borrows articulated mass/force arenas.
        // Deferred frameComplete runs after the lease scope closes and uses
        // accepted/checkpoint state plus the final body status gate only.
        if(outletStep&&phase==AcceptedStepExtensionPhase::candidateReady&&!lease.active)
            return rejectLease("missing_owner_lease");
        if(phase==AcceptedStepExtensionPhase::candidateReady&&lease.active) {
            if(!lease.bodyPropertiesCandidate||!lease.bodyPropertiesAccepted||
               !lease.bodyPropertiesReference||lease.bodyPropertyCount==0u||
               lease.bodyPropertyCount>UINT32_MAX)
                return rejectLease("missing_mass_buffers_or_count");
            if(outletStep&&(!lease.bodyPosesAccepted||!lease.bodyPositionLow||
               !lease.pointJacobians||!lease.generalizedVelocity||
               !lease.generalizedForceArena||lease.bodyPropertyCount<=7u||
               lease.articulationFirstBody>7u||
               7u-lease.articulationFirstBody>=lease.bodyPoseStride||
               lease.dofCount==0u||lease.dofCount>MR_ARTICULATED_OPERATOR_KINEMATICS_MAX_DOFS||
               lease.vStride<lease.dofCount||
               lease.bodyPoseElementCount<lease.bodyPoseStride||
               lease.bodyPositionLowElementCount<lease.bodyPoseStride||
               lease.pointJacobianStride==0u||
               lease.generalizedVelocityElementCount<lease.vStride||
               lease.dofCount>lease.generalizedForceStride||
               lease.generalizedForceOffset>lease.generalizedForceArenaElementCount||
               lease.dofCount>lease.generalizedForceArenaElementCount-
                   lease.generalizedForceOffset||
               lease.pointJacobianElementCount<lease.pointJacobianStride||
               lease.bodyJacobianPointOffset+4u*(7u-lease.articulationFirstBody)+4u>
                   lease.pointJacobianStride/(3u*lease.dofCount)))
                return rejectLease("outlet_dimension_bounds");
        }
        if(dispatch.bloodMpmEnabled&&phase==AcceptedStepExtensionPhase::frameBegin) {
            const bool materialBuffersValid=dispatch.bloodMpmMaterialStateCount==0u||
                (v.particleMaterialStateDefaults&&v.particleMaterialStateAccepted&&
                 v.particleMaterialStateCandidate);
            if(!bloodMpmHandoff||!v.matterStatuses||!v.particleAccepted||
               !v.particleCandidate||!v.particleCheckpoint||
               v.environmentCount!=dispatch.environmentCount||
               v.particleCount!=dispatch.particleCount||
               v.materialStateStride!=dispatch.materialStateStride||
               v.particleCount<=dispatch.bloodMpmFirstParticle||
               dispatch.bloodMpmParticleCount>
                   v.particleCount-dispatch.bloodMpmFirstParticle||
               v.materialStateStride<dispatch.bloodMpmMaterialStateCount||
               !materialBuffersValid) return rejectLease("blood_mpm_borrowed_arena");
        }
        if(brain&&!brain(phase,v)) return false;
        NMHumanRespirationDispatch stepDispatch = dispatch;
        stepDispatch.controlStep=v.controlStep;
        if(phase==AcceptedStepExtensionPhase::candidateReady&&lease.active) {
            stepDispatch.articulationFirstBody=lease.articulationFirstBody;
            stepDispatch.bodyPoseStride=lease.bodyPoseStride;
            stepDispatch.bodyJacobianPointOffset=lease.bodyJacobianPointOffset;
            stepDispatch.pointJacobianStride=lease.pointJacobianStride;
            stepDispatch.dofCount=lease.dofCount;
            stepDispatch.vStride=lease.vStride;
            stepDispatch.generalizedForceOffset=lease.generalizedForceOffset;
            stepDispatch.generalizedForceStride=lease.generalizedForceStride;
            stepDispatch.bodyPropertyCount=static_cast<std::uint32_t>(lease.bodyPropertyCount);
        }
        if(diagnosticRejectAtControlStep != NM_INVALID_INDEX)
            stepDispatch.reject = v.controlStep == diagnosticRejectAtControlStep ? 1u : 0u;
        id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)v.commandBuffer;
        if(phase==AcceptedStepExtensionPhase::microstepComplete) return true;
        const char* stage=phase==AcceptedStepExtensionPhase::frameBegin?"respiratory_mechanics":
            phase==AcceptedStepExtensionPhase::candidateReady?"respiratory_gas_exchange":"respiratory_resolve";
        auto enc=detail::timedEncoder(cb,device,stage,v.controlStep);if(!enc)return false;
        bool dispatched = false;
        if(phase==AcceptedStepExtensionPhase::frameBegin) {
            [enc setComputePipelineState:predict];
            [enc setBytes:&parameters length:sizeof(parameters) atIndex:0];
            [enc setBytes:&stepDispatch length:sizeof(dispatch) atIndex:1];
            [enc setBuffer:accepted offset:0 atIndex:2];
            [enc setBuffer:candidate offset:0 atIndex:3];
            [enc setBuffer:excitation offset:0 atIndex:4];
            if(dispatch.bloodMpmEnabled) {
                [enc dispatchThreads:MTLSizeMake(dispatch.environmentCount,1,1)
                    threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setComputePipelineState:bloodMpmHandoff];
                [enc setBytes:&stepDispatch length:sizeof(stepDispatch) atIndex:0];
                [enc setBuffer:candidate offset:0 atIndex:1];
                [enc setBuffer:buffer(v.matterStatuses) offset:0 atIndex:2];
                [enc setBuffer:buffer(v.particleAccepted) offset:0 atIndex:3];
                [enc setBuffer:buffer(v.particleCandidate) offset:0 atIndex:4];
                [enc setBuffer:buffer(v.particleCheckpoint) offset:0 atIndex:5];
                [enc setBuffer:buffer(v.particleMaterialStateDefaults) offset:0 atIndex:6];
                [enc setBuffer:buffer(v.particleMaterialStateAccepted) offset:0 atIndex:7];
                [enc setBuffer:buffer(v.particleMaterialStateCandidate) offset:0 atIndex:8];
                [enc dispatchThreads:MTLSizeMake(dispatch.environmentCount,1,1)
                    threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                dispatched=true;
            }
        } else if(phase==AcceptedStepExtensionPhase::candidateReady) {
            if(commonGeometryGateEnabled &&
               (!commonCoordinatesSolvePipeline||!commonCoordinateStatusPipeline||
                !commonGeometryParameters||!commonGeometryBoxes||!commonCandidateCoordinates)) {
                [enc endEncoding];
                return false;
            }
            [enc setComputePipelineState:exchange];
            [enc setBytes:&parameters length:sizeof(parameters) atIndex:0];
            [enc setBytes:&stepDispatch length:sizeof(dispatch) atIndex:1];
            [enc setBuffer:accepted offset:0 atIndex:2];
            [enc setBuffer:candidate offset:0 atIndex:3];
            // First-order frame coupling: use the final circuit flow over
            // the body frame, with actual frame-start/end vascular volumes.
            // Internal gas transfers remain antisymmetric and outlet debit
            // executes once; no controller or metabolism is multiplied by
            // the continuum tick count. The temporal approximation is qualified
            // separately from inventory conservation.
            [enc setBuffer:buffer(continuumSubcycles?v.vascularCheckpoint:v.vascularAccepted) offset:0 atIndex:4];
            [enc setBuffer:buffer(v.vascularCandidate) offset:0 atIndex:5];
            [enc setBuffer:buffer(v.vascularUnknowns) offset:0 atIndex:6];
            [enc setBuffer:buffer(v.vascularConnections) offset:0 atIndex:7];
            [enc setBuffer:buffer(v.matterStatuses) offset:0 atIndex:8];
            [enc setBuffer:buffer(v.vascularCompartments) offset:0 atIndex:9];
            [enc setBuffer:buffer(v.vascularElastance) offset:0 atIndex:10];
            [enc setBuffer:buffer(lease.bodyPosesAccepted) offset:0 atIndex:11];
            [enc setBuffer:buffer(lease.bodyPropertiesCandidate) offset:0 atIndex:12];
            [enc setBuffer:buffer(lease.bodyPositionLow) offset:0 atIndex:13];
            [enc setBuffer:buffer(lease.pointJacobians) offset:0 atIndex:14];
            [enc setBuffer:buffer(lease.generalizedVelocity) offset:0 atIndex:15];
            [enc setBuffer:buffer(lease.generalizedForceArena) offset:0 atIndex:16];
            [enc setBuffer:buffer(lease.bodyPropertiesReference) offset:0 atIndex:17];
            // Exchange must execute before switching this encoder to the
            // common-coordinate solver/status pipelines. This propagates any
            // respiratory candidate failure into the Matter transaction.
            if(parallelRespiratoryMuscles)
                [enc dispatchThreadgroups:MTLSizeMake(dispatch.environmentCount,1,1)
                    threadsPerThreadgroup:MTLSizeMake(exchange.threadExecutionWidth,1,1)];
            else
                [enc dispatchThreads:MTLSizeMake(dispatch.environmentCount,1,1)
                    threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            dispatched = true;
            if(commonGeometryGateEnabled) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setComputePipelineState:commonCoordinatesSolvePipeline];
                [enc setBuffer:candidate offset:0 atIndex:0];
                [enc setBuffer:commonGeometryParameters offset:0 atIndex:1];
                [enc setBuffer:commonGeometryBoxes offset:0 atIndex:2];
                [enc setBuffer:commonCandidateCoordinates offset:0 atIndex:3];
                const NSUInteger solverWidth=commonCoordinatesSolvePipeline.threadExecutionWidth;
                if(solverWidth<7u) { [enc endEncoding]; return false; }
                [enc dispatchThreadgroups:MTLSizeMake(1,1,1)
                    threadsPerThreadgroup:MTLSizeMake(solverWidth,1,1)];
                [enc setComputePipelineState:commonCoordinateStatusPipeline];
                [enc setBuffer:commonCandidateCoordinates offset:0 atIndex:0];
                [enc setBuffer:buffer(v.matterStatuses) offset:0 atIndex:1];
                [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            }
        } else if(phase==AcceptedStepExtensionPhase::frameComplete) {
            [enc setComputePipelineState:resolve];
            [enc setBytes:&stepDispatch length:sizeof(dispatch) atIndex:0];
            [enc setBuffer:candidate offset:0 atIndex:1];
            [enc setBuffer:accepted offset:0 atIndex:2];
            [enc setBuffer:buffer(v.matterStatuses) offset:0 atIndex:3];
        } else {[enc endEncoding];return false;}
        if(!dispatched)
            [enc dispatchThreads:MTLSizeMake(dispatch.environmentCount,1,1)
                threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [enc endEncoding];return true;
    }
    static bool callback(void* ctx,AcceptedStepExtensionPhase phase,const AcceptedStepExtensionView& v) {
        return static_cast<Respiration*>(ctx)->encode(phase,v);
    }
};

struct RespiratoryBrain {
    id<MTLBuffer> input,accepted,candidate,output;
    id<MTLComputePipelineState> observe,advance,deliver,resolve;
    NBNumiRespiratoryChemoreflexParametersV1 parameters{};
    Respiration& body;
    std::uint64_t rootProgramIdentity;
    double interventionStart=-1,interventionEnd=-1;
    float interventionScale=1;
    std::uint64_t root(unsigned step)const {
        return (rootProgramIdentity^(std::uint64_t(step)*0x9e3779b97f4a7c15ull))|1ull;
    }
    RespiratoryBrain(Respiration& physiology,const CompiledWorld& world,const char* config,std::uint64_t bodySourceIdentity=0):body(physiology) {
        NSDictionary* json=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(config)] options:0 error:nil];
        NSDictionary* values=json[@"brain_parameters"];
        need([values isKindOfClass:NSDictionary.class],"reference Brain parameters missing");
        const std::array<NSString*,16> keys={@"restingMinuteVentilationLitresPerMinute",@"CO2ResponseSlopeLitresPerMinutePerMmHg",@"centralCO2Fraction",@"centralTimeConstantSeconds",@"peripheralTimeConstantSeconds",@"hypoxicReferencePaO2MillimetersMercury",@"hypoxicScalePaO2MillimetersMercury",@"hypoxicIncrementAtPaO2FortyLitresPerMinute",@"restingFrequencyBreathsPerMinute",@"restingTidalVolumeLitres",@"diaphragmExcitationGain",@"intercostalExcitationGain",@"inspirationFractionOfCycle",@"minimumMinuteVentilationLitresPerMinute",@"maximumMinuteVentilationLitresPerMinute",@"reserved"};
        for(unsigned i=0;i<keys.size();++i) {
            id value=values[keys[i]];
            need([value isKindOfClass:NSNumber.class]&&std::isfinite([value doubleValue]),"invalid Brain parameter");
            reinterpret_cast<float*>(&parameters)[i]=[value floatValue];
        }
        rootProgramIdentity=world.fingerprint;
        auto mix=[&](const void* bytes,std::size_t count) {
            auto p=static_cast<const unsigned char*>(bytes);
            for(std::size_t i=0;i<count;++i){rootProgramIdentity^=p[i];rootProgramIdentity*=1099511628211ull;}
        };
        mix(&body.parameters,sizeof(body.parameters));mix(&parameters,sizeof(parameters));
        if(body.parallelRespiratoryMuscles) {
            constexpr char mode[]="numi.respiratory.parallel-muscles.v1";
            mix(mode,sizeof(mode));
        }
        if(body.parallelRespiratoryGas) {
            constexpr char mode[]="numi.respiratory.parallel-gas.v1";
            mix(mode,sizeof(mode));
        }
        if(bodySourceIdentity)mix(&bodySourceIdentity,sizeof(bodySourceIdentity));
        #ifdef NUMI_HUMAN_RESPIRATION_SOURCE
        NSData* libraryBytes=[NSData dataWithContentsOfFile:@(NUMI_HUMAN_RESPIRATION_SOURCE)];
#else
        NSData* libraryBytes=[NSData dataWithContentsOfFile:@(NUMI_HUMAN_RESPIRATION_METALLIB)];
#endif
        mix(libraryBytes.bytes,libraryBytes.length);
        NSError* error=nil;
        #ifdef NUMI_HUMAN_RESPIRATION_SOURCE
        NSString* brainSource=[NSString stringWithContentsOfFile:@(NUMI_HUMAN_RESPIRATION_SOURCE) encoding:NSUTF8StringEncoding error:&error];
        MTLCompileOptions* brainOptions=[MTLCompileOptions new];brainOptions.languageVersion=MTLLanguageVersion4_0;brainOptions.mathMode=MTLMathModeSafe;brainOptions.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        auto library=[body.device newLibraryWithSource:brainSource options:brainOptions error:&error];
#else
        auto library=[body.device newLibraryWithURL:[NSURL fileURLWithPath:@(NUMI_HUMAN_RESPIRATION_METALLIB)] error:&error];
#endif
        auto pipeline=[&](NSString* name){auto p=[body.device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];need(p!=nil,"Brain pipeline unavailable");return p;};
        observe=pipeline(@"nm_human_respiration_brain_observe");
        advance=pipeline(@"advance_numi_respiratory_chemoreflex_v1");
        deliver=pipeline(@"nm_human_respiration_brain_deliver");
        resolve=pipeline(@"nm_human_respiration_brain_resolve");
        const auto count=body.dispatch.environmentCount;
        auto allocate=[&](std::size_t n){auto b=[body.device newBufferWithLength:n*count options:MTLResourceStorageModeShared];need(b!=nil,"Brain buffer allocation");std::memset(b.contents,0,b.length);return b;};
        input=allocate(sizeof(NBNumiRespiratoryChemoreflexInputV1));
        accepted=allocate(sizeof(NBNumiRespiratoryChemoreflexStateV1));candidate=allocate(sizeof(NBNumiRespiratoryChemoreflexStateV1));
        output=allocate(sizeof(NBNumiRespiratoryChemoreflexOutputV1));
        for(unsigned e=0;e<count;++e) {
            auto& s=static_cast<NBNumiRespiratoryChemoreflexStateV1*>(accepted.contents)[e];
            s.acceptedRootFingerprint=root(0);s.flags=NB_NUMI_RESPIRATORY_RECORD_VALID;
        }
        body.brain=[this](auto phase,const auto& v){return encode(phase,v);};
    }
    ~RespiratoryBrain(){body.brain={};}
    bool encode(AcceptedStepExtensionPhase phase,const AcceptedStepExtensionView& v) {
        if(phase!=AcceptedStepExtensionPhase::frameBegin&&phase!=AcceptedStepExtensionPhase::frameComplete)return true;
        const double dt=body.parameters.environment.w;
        NMHumanRespirationBrainDispatch d{};
        // Scheduling timestamps are rounded from Matter's exact fixed binary32
        // cadence, never accumulated in floating point by the controller.
        d.sourceTimeMicroseconds=std::llround(double(v.controlStep)*dt*1e6);
        d.targetTimeMicroseconds=std::llround(double(v.controlStep+1u)*dt*1e6);
        d.sourceRootIdentity=root(v.controlStep);d.targetRootIdentity=root(v.controlStep+1);
        d.sourceStep=v.controlStep;d.environmentCount=v.environmentCount;
        const double t=double(v.controlStep)*dt;
        d.driveScale=t>=interventionStart&&t<interventionEnd?interventionScale:1;
        auto cb=(__bridge id<MTLCommandBuffer>)v.commandBuffer;
        auto enc=detail::timedEncoder(cb,body.device,
            phase==AcceptedStepExtensionPhase::frameBegin?"brain_advance":"brain_resolve",v.controlStep);
        if(!enc)return false;
        auto launch=[&](){[enc dispatchThreads:MTLSizeMake(v.environmentCount,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];};
        if(phase==AcceptedStepExtensionPhase::frameBegin) {
            [enc setComputePipelineState:observe];[enc setBytes:&d length:sizeof(d) atIndex:0];
            [enc setBuffer:body.accepted offset:0 atIndex:1];[enc setBuffer:input offset:0 atIndex:2];launch();
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:advance];[enc setBytes:&parameters length:sizeof(parameters) atIndex:0];
            [enc setBuffer:input offset:0 atIndex:1];[enc setBuffer:accepted offset:0 atIndex:2];
            [enc setBuffer:candidate offset:0 atIndex:3];[enc setBuffer:output offset:0 atIndex:4];launch();
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:deliver];[enc setBytes:&d length:sizeof(d) atIndex:0];
            [enc setBuffer:output offset:0 atIndex:1];[enc setBuffer:body.excitation offset:0 atIndex:2];launch();
        }else {
            [enc setComputePipelineState:resolve];[enc setBytes:&d length:sizeof(d) atIndex:0];
            [enc setBuffer:candidate offset:0 atIndex:1];[enc setBuffer:accepted offset:0 atIndex:2];
            [enc setBuffer:buffer(v.matterStatuses) offset:0 atIndex:3];[enc setBuffer:body.candidate offset:0 atIndex:4];launch();
        }
        [enc endEncoding];return true;
    }
};

inline bool hasVascularPayload(const VascularNetworkSource& network) {
    for(const auto word:network.contentIdentity) if(word!=0u) return true;
    for(const auto word:network.sourceIdentity) if(word!=0u) return true;
    for(const auto word:network.authoredIdentity) if(word!=0u) return true;
    return !network.species.empty()||!network.compartments.empty()||
        !network.connections.empty()||!network.tissues.empty()||
        !network.exchanges.empty()||!network.cavities.empty();
}
inline bool containsFEMObject(const WorldSource& source) {
    for(const auto& object:source.objects)
        if(object.representation==Representation::fem) return true;
    return false;
}
inline bool containsRestingBloodReceiver(const WorldSource& source) {
    constexpr const char* receiverName="external_blood_dormant_receiver";
    constexpr const char* materialName="blood_high_shear_newtonian_proxy";
    for(const auto& object:source.objects) {
        if(object.name==receiverName) return true;
        if(object.representation==Representation::mpm&&
           object.materialIndex<source.materials.size()&&
           source.materials[object.materialIndex].name==materialName) return true;
    }
    return false;
}
inline WorldSource composeRestingWorldSource(const VascularNetworkSource& network,
                                              float timestep,
                                              const WorldSource* tissueSeed) {
    need(std::isfinite(timestep)&&timestep>0.0f,
         "resting Matter timestep must be finite and positive");
    WorldSource source;
    if(tissueSeed) {
        need(tissueSeed->environmentCount==1u,
             "resting tissue seed must contain exactly one environment");
        need(std::isfinite(tissueSeed->frameTimestep)&&tissueSeed->frameTimestep>0.0&&
             static_cast<float>(tissueSeed->frameTimestep)==timestep,
             "resting tissue seed timestep conflicts with the CVSim timestep");
        need(!hasVascularPayload(tissueSeed->vascular),
             "resting tissue seed already owns vascular state");
        need(!containsRestingBloodReceiver(*tissueSeed),
             "resting tissue seed already owns the blood MPM receiver");
        source=*tissueSeed;
    } else {
        source.environmentCount=1u;
        source.frameTimestep=timestep;
        source.gravity={0.0,0.0,0.0};
        source.mixedSolver.relativeResidual=1.e-7;
        source.mixedSolver.newtonIterations=12u;
        source.mixedSolver.fgmresIterations=64u;
    }
    source.vascular=network;
    return source;
}
inline void appendRestingBloodMaterial(WorldSource& source,
                                       MaterialProgram material,
                                       ObjectSource receiver) {
    need(source.materials.size()<std::numeric_limits<std::uint32_t>::max(),
         "resting material table exceeds 32-bit object references");
    need(material.name=="blood_high_shear_newtonian_proxy"&&
         receiver.name=="external_blood_dormant_receiver"&&
         receiver.representation==Representation::mpm,
         "resting blood append requires its declared material and MPM receiver identities");
    need(!containsRestingBloodReceiver(source),
         "resting world already owns a blood MPM receiver");
    receiver.materialIndex=static_cast<std::uint32_t>(source.materials.size());
    source.materials.push_back(std::move(material));
    source.objects.push_back(std::move(receiver));
}

struct RestingRun {
    CompiledWorld world;
    Runtime runtime;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> statuses;
    std::unique_ptr<Respiration> respiration;
    explicit RestingRun(const char* network,const char* configuration,float dt,
                        bool vascularDense45=false,
                        const WorldSource* tissueSource=nullptr) {
        VascularNetworkSource n;std::string error;
        need(readHumanPhysiologyNetwork(network,n,&error),error);
        WorldSource source;
        const char* bloodMode=std::getenv("NUMI_HUMAN_BLOOD_MPM_ENABLED");
        need(!bloodMode||std::strcmp(bloodMode,"0")==0||std::strcmp(bloodMode,"1")==0,
             "NUMI_HUMAN_BLOOD_MPM_ENABLED must be exactly 0 or 1");
        const bool bloodEnabled=bloodMode&&std::strcmp(bloodMode,"1")==0;
        source=composeRestingWorldSource(n,dt,tissueSource);
        if(bloodEnabled) {
            const char* path=std::getenv("NUMI_HUMAN_BLOOD_MPM_MATERIAL");
            need(path&&*path,"blood MPM mode requires an explicit source material path");
            auto material=parseMatterFile(path);
            need(material.succeeded()&&material.material.name=="blood_high_shear_newtonian_proxy",
                 "blood MPM source material is absent, invalid, or has the wrong identity");
            double density=0.0;
            for(const auto& parameter:material.material.parameters)
                if(parameter.name=="density") density=parameter.defaultValue;
            need(std::isfinite(density)&&density>0.0,"blood MPM source density is invalid");
            ObjectSource receiver;
            receiver.name="external_blood_dormant_receiver";
            // Assigned against the appended material table after preserving
            // every material already owned by the tissue seed.
            receiver.representation=Representation::mpm;
            receiver.automaticRepresentation=false;receiver.twoWayCoupling=false;
            receiver.adaptive=false;receiver.deformableContact=false;
            receiver.deformableSelfContact=false;
            // Bounded reference capacity, not pre-existing physical blood.
            // Partial outlet batches consume slots; exhaustion rejects rather
            // than allocating, dropping mass, or reusing live fluid particles.
            receiver.characteristicLength=0.05;
            receiver.mpmGridMinimum={-1.5,-1.5,-1.0};
            receiver.mpmGridMaximum={1.5,1.5,3.0};
            // Bounded experiments may declare a smaller world-space domain
            // without changing cell width or the physical update schedule.
            // Leaving the domain rejects; it never clips or drops fluid.
            if(const char* bounds=std::getenv("NUMI_HUMAN_BLOOD_MPM_GRID_BOUNDS_M")) {
                double values[6]{};
                const char* cursor=bounds;
                for(unsigned axis=0;axis<6u;++axis) {
                    char* end=nullptr;values[axis]=std::strtod(cursor,&end);
                    need(end!=cursor&&std::isfinite(values[axis]),
                         "blood MPM grid bounds require six finite comma-separated coordinates");
                    if(axis<5u) {
                        need(*end==',',"blood MPM grid bounds require six comma-separated coordinates");
                        cursor=end+1;
                    } else need(*end=='\0',"blood MPM grid bounds have trailing content");
                }
                for(unsigned axis=0;axis<3u;++axis) {
                    const double extent=values[axis+3u]-values[axis];
                    need(extent>=4.0*receiver.characteristicLength&&extent<=10.0,
                         "blood MPM grid extent must be between four cells and ten metres");
                }
                receiver.mpmGridMinimum={values[0],values[1],values[2]};
                receiver.mpmGridMaximum={values[3],values[4],values[5]};
            }
            const std::array<double,3> dormantGridCenter={
                0.5*(receiver.mpmGridMinimum[0]+receiver.mpmGridMaximum[0]),
                0.5*(receiver.mpmGridMinimum[1]+receiver.mpmGridMaximum[1]),
                0.5*(receiver.mpmGridMinimum[2]+receiver.mpmGridMaximum[2])};
            for(unsigned slot=0;slot<128u;++slot) {
                // Inactive receiver slots still require finite, in-domain
                // positions because the compiler validates their quadratic
                // support stencil before activation.
                ParticleSource particle;particle.position=dormantGridCenter;
                particle.velocity={0,0,0};particle.referenceVolume=1.0e-6;
                particle.mass=density*particle.referenceVolume;particle.active=false;
                receiver.particles.push_back(particle);
            }
            if(!tissueSource) source.gravity={0,0,-9.81};
            appendRestingBloodMaterial(source,std::move(material.material),std::move(receiver));
            const auto& appendedReceiver=source.objects.back();
            std::cerr<<"blood_mpm_source=reference slots=128 capacity_ml=128 cell_width_m=0.05"
                     <<" grid_min_m="<<appendedReceiver.mpmGridMinimum[0]<<","<<appendedReceiver.mpmGridMinimum[1]<<","<<appendedReceiver.mpmGridMinimum[2]
                     <<" grid_max_m="<<appendedReceiver.mpmGridMaximum[0]<<","<<appendedReceiver.mpmGridMaximum[1]<<","<<appendedReceiver.mpmGridMaximum[2]
                     <<" gravity_m_s2="<<source.gravity[0]<<","<<source.gravity[1]<<","<<source.gravity[2]
                     <<" material="<<path
                     <<" contact=disabled maximum_rate_exponent="<<NM_MAX_RATE_EXPONENT<<"\n";
        }
        const bool fullRateWorld=bloodEnabled||containsFEMObject(source);
        auto compiled=compileWorld(source,{.maximumRateExponent=fullRateWorld?NM_MAX_RATE_EXPONENT:0u});
        for(const auto& d:compiled.diagnostics)if(d.severity==Diagnostic::Severity::error)error+=d.message+"; ";
        need(compiled.succeeded(),error);world=std::move(compiled.world);
        device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device unavailable");
        need([[device name] rangeOfString:@"Apple"].location!=NSNotFound&&[[device name] rangeOfString:@"Paravirtual"].location==NSNotFound,"physical Apple GPU required");
        queue=[device newCommandQueue];statuses=[device newBufferWithLength:sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
        RuntimeConfiguration config;config.metallib=NUMI_MATTER_METALLIB;
        config.environmentCount=1;config.captureEvents=false;config.captureDiagnostics=false;config.adaptiveTransfer=false;config.enableVascularDense45=vascularDense45;
        auto init=runtime.initialize(world,config);need(init.encoded,init.message);
        respiration=std::make_unique<Respiration>(device,world,configuration,runtime.timestepSeconds());
        std::cout<<"runtime="<<init.message<<" device="<<[device name].UTF8String<<" world_fingerprint="<<world.fingerprint<<" timestep_s="<<runtime.timestepSeconds()<<'\n';
    }
    double batch(unsigned first,unsigned count,bool expectRejected=false,
                 unsigned rejectAtControlStep=NM_INVALID_INDEX) {
        *static_cast<MRMetalWorldStatusGPU*>(statuses.contents)={};
        auto cb=[queue commandBuffer];need(cb!=nil,"command buffer");
        for(unsigned step=first;step<first+count;++step) {
            // Probe-only per-frame rejection injection. The dispatch struct is
            // copied into that frame's command stream, so earlier accepted
            // frames and a later rejected frame share one command buffer.
            if(rejectAtControlStep!=NM_INVALID_INDEX)
                respiration->dispatch.reject=step==rejectAtControlStep?1u:0u;
            EncodeRequest r;r.commandBuffer=(__bridge void*)cb;r.environmentStatuses=(__bridge void*)statuses;
            r.controlStep=step;r.physicsSubsteps=1;r.timestepSeconds=runtime.timestepSeconds();r.runAdaptiveTransfer=false;
            r.acceptedStepExtensionContext=respiration.get();r.encodeAcceptedStepExtension=&Respiration::callback;
            r.phase=EncodePhase::preDynamics;auto result=runtime.encode(r);need(result.encoded,result.message);
            r.phase=EncodePhase::postCommit;result=runtime.encode(r);need(result.encoded,result.message);
        }
        [cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Metal execution failure");
        auto* state=static_cast<const NMHumanRespirationState*>(respiration->accepted.contents);
        auto* staged=static_cast<const NMHumanRespirationState*>(respiration->candidate.contents);
        if(expectRejected) {
            const unsigned expectedAccepted=rejectAtControlStep==NM_INVALID_INDEX
                ? first : rejectAtControlStep;
            need(state->status.x==expectedAccepted&&staged->status.w!=0,
                 "forced physical rejection did not preserve the last accepted prefix");
        }else if(state->status.x!=first+count) {
            const auto failed=runtime.snapshot();
            const auto status=failed.statuses.empty()?NMMatterStatusGPU{}:failed.statuses.front();
            throw std::runtime_error("rejected physical/respiratory step at accepted="+std::to_string(state->status.x)+
                " respiratory_code="+std::to_string(staged->status.w)+" candidate_step="+std::to_string(staged->status.x)+
                " matter_code="+std::to_string(status.code)+" failing_index="+std::to_string(status.failingIndex)+
                " residual="+std::to_string(status.diagnostics.z));
        }
        return cb.GPUEndTime-cb.GPUStartTime;
    }
};
inline void writeRespirationTraceHeader(std::ostream& trace) {
    trace<<"time_s,lung_volume_ml,airflow_ml_s,alveolar_pa,pleural_pa,diaphragm_mm,rib_mm,PaO2_mmhg,PaCO2_mmhg,SaO2,oxygen_balance_error_stpd_ml,co2_balance_error_stpd_ml,breaths,tidal_ml,lv_mmhg,rv_mmhg,aorta_mmhg,pulmonary_artery_mmhg,lv_ml,rv_ml,blood_ml,blood_error_ml,aortic_ejected_ml,pulmonary_ejected_ml,complete_filling_ejection_cycles,last_lv_stroke_ml,right_atrium_ml,left_atrium_ml,blood_continuity_residual_accum_ml,blood_physical_delta_accum_ml,blood_residual_minus_physical_ml,blood_endpoint_minus_physical_ml";
    trace<<",inspired_volume_accum_ml,last_inspiration_step,last_inspiration_time_s,last_inspiration_volume_accum_ml,last_complete_breath_inspired_ml";
    trace<<",respiratory_net_volume_ml,respiratory_volume_balance_ml,gas_max_donor_fraction,respiratory_gas_substeps";
    trace<<",external_blood_ml,external_blood_mass_kg,external_blood_o2_stpd_ml,external_blood_co2_stpd_ml,grounded_blood_ml,grounded_blood_mass_kg";
    trace<<",reference_positive_body_mass_kg,candidate_positive_body_mass_kg,external_mass_kg,body_mass_residual_kg,max_abs_body_mass_residual_kg";
    trace<<",algebraic_transfer_momentum_residual_kg_m_s,algebraic_transfer_angular_residual_kg_m2_s";
    trace<<",mpm_represented_blood_ml,mpm_represented_blood_mass_kg,mpm_represented_o2_stpd_ml,mpm_represented_co2_stpd_ml";
}
inline void writeRespirationTraceSample(std::ostream& trace,const NMHumanRespirationState& s,
    const NMHumanRespirationParameters& parameters) {
    trace<<double(s.status.x)*parameters.environment.w<<','<<s.mechanics.x*1e6<<','<<s.mechanics.w*1e6<<','<<s.mechanics.y<<','<<s.mechanics.z<<','
         <<s.motion.x/parameters.geometry.z*1000<<','<<s.motion.y/parameters.geometry.w*1000<<','
         <<s.observation.x<<','<<s.observation.y<<','<<s.observation.z<<','<<s.gasBudget.z*1e6<<','<<s.gasBudget.w*1e6<<','<<s.status.y<<','<<s.breath.z*1e6<<','
         <<s.cardiacPressure.x/133.322387415<<','<<s.cardiacPressure.y/133.322387415<<','<<s.cardiacPressure.z/133.322387415<<','<<s.cardiacPressure.w/133.322387415<<','
         <<s.chamberVolumes.w*1e6<<','<<s.chamberVolumes.y*1e6<<','<<s.circulation.x*1e6<<','<<s.circulation.w*1e6<<','<<s.cardiacFlow.x*1e6<<','<<s.cardiacFlow.y*1e6<<','<<s.cardiacStatus.x<<','<<s.cardiacFlow.w*1e6<<','
         <<s.chamberVolumes.x*1e6<<','<<s.chamberVolumes.z*1e6<<',';
    const double continuity=double(s.bloodBalance.continuityResidualSumM3)-
        double(s.bloodBalance.continuityResidualCompensationM3);
    const double physical=double(s.bloodBalance.physicalVolumeDeltaSumM3)-
        double(s.bloodBalance.physicalVolumeDeltaCompensationM3);
    const double endpoint=double(s.circulation.x)-double(0.00515f);
    trace<<continuity*1e6<<','<<physical*1e6<<','
         <<(continuity-physical)*1e6<<','
         <<(endpoint-physical)*1e6<<','
         <<(double(s.breath.w)-double(s.breathAccounting.y))*1e6<<','
         <<s.breathTiming.x<<','<<double(s.breathTiming.x)*parameters.environment.w<<','
         <<(double(s.breathAccounting.x)-double(s.breathAccounting.w))*1e6<<','
         <<s.breathAccounting.z*1e6<<','<<s.transportStep.x*1e6<<','
         <<s.transportStep.y*1e6<<','<<s.transportStep.z<<','<<s.transportStep.w<<','
         <<double(s.externalBlood.total.x-s.externalBlood.compensation.x)*1e6<<','
         <<double(s.externalBlood.total.y-s.externalBlood.compensation.y)<<','
         <<double(s.externalBlood.total.z-s.externalBlood.compensation.z)*1e6<<','
         <<double(s.externalBlood.total.w-s.externalBlood.compensation.w)*1e6<<','
         <<double(s.externalBlood.grounded.x-s.externalBlood.groundedCompensation.x)*1e6<<','
         <<double(s.externalBlood.grounded.y-s.externalBlood.groundedCompensation.y)<<','
         <<double(s.externalBlood.bodyMassBalance.x)<<','
         <<double(s.externalBlood.bodyMassBalance.y)<<','
         <<double(s.externalBlood.bodyMassBalance.z)<<','
         <<double(s.externalBlood.bodyMassBalance.w)<<','
         <<double(s.externalBlood.bodyMassBalanceMax.x)<<','
         <<std::sqrt(double(s.externalBlood.transferMomentumResidual.w))<<','
         <<std::sqrt(double(s.externalBlood.transferAngularResidual.w))<<','
         <<double(s.externalBlood.mpmRepresented.x-s.externalBlood.mpmRepresentedCompensation.x)*1e6<<','
         <<double(s.externalBlood.mpmRepresented.y-s.externalBlood.mpmRepresentedCompensation.y)<<','
         <<double(s.externalBlood.mpmRepresented.z-s.externalBlood.mpmRepresentedCompensation.z)*1e6<<','
         <<double(s.externalBlood.mpmRepresented.w-s.externalBlood.mpmRepresentedCompensation.w)*1e6;
}
} // namespace numi::human
