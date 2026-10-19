#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../source/matter/tools/human_resting_runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace numi::human;

constexpr float kFrameSeconds = 0.00025f;
constexpr std::uint32_t kCapacityFrames = 132u;
constexpr std::uint32_t kFloorFrames = 32u;
constexpr float kPacketVolumeM3 = 1.1e-9f;
constexpr float kOxygenFraction = 0.21f;
constexpr float kCarbonDioxideFraction = 0.05f;
constexpr float kSpinRadiansPerSecond = 0.1f;
constexpr double kCapacityGridMinimum = -0.25;
constexpr double kCapacityGridMaximum = 0.25;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void setEnv(const char* name, const std::string& value) {
    require(setenv(name, value.c_str(), 1) == 0,
            std::string("could not set ") + name);
}
std::string bounds(double xOffset) {
    return std::to_string(kCapacityGridMinimum + xOffset) + "," +
        std::to_string(kCapacityGridMinimum) + "," +
        std::to_string(kCapacityGridMinimum) + "," +
        std::to_string(kCapacityGridMaximum + xOffset) + "," +
        std::to_string(kCapacityGridMaximum) + "," +
        std::to_string(kCapacityGridMaximum);
}
void configureBlood(const char* material, bool contactEnabled, double xOffset) {
    setEnv("NUMI_HUMAN_BLOOD_MPM_ENABLED", "1");
    setEnv("NUMI_HUMAN_BLOOD_MPM_CONTACT_ENABLED",
           contactEnabled ? "1" : "0");
    setEnv("NUMI_HUMAN_BLOOD_MPM_MATERIAL", material);
    setEnv("NUMI_HUMAN_BLOOD_MPM_GRID_BOUNDS_M", bounds(xOffset));
    setEnv("NUMI_HUMAN_BLEEDING_ENABLED", "0");
    setEnv("NUMI_HUMAN_BLOOD_MPM_MAX_RATE_EXPONENT", "0");
    setEnv("NUMI_HUMAN_RESTING_TIMESTEP_SENSITIVITY", "1");
}
void kahanAdd(float value, float& sum, float& compensation) {
    const float adjusted = value - compensation;
    const float next = sum + adjusted;
    compensation = (next - sum) - adjusted;
    sum = next;
}
struct Vec3 { double x=0.0, y=0.0, z=0.0; };
Vec3 add(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
Vec3 sub(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
Vec3 mul(double s, Vec3 a) { return {s*a.x,s*a.y,s*a.z}; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
double norm(Vec3 a) { return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z); }
Vec3 particlePosition(const NMParticleStateGPU& p) {
    return {p.positionAndMass.x,p.positionAndMass.y,p.positionAndMass.z};
}
Vec3 particleMomentum(const NMParticleStateGPU& p) {
    return mul(p.positionAndMass.w,
        {p.velocityAndReferenceVolume.x,
         p.velocityAndReferenceVolume.y,
         p.velocityAndReferenceVolume.z});
}
Vec3 particleSpin(const NMParticleStateGPU& p, double h) {
    const double omegaX=0.5*(double(p.affineRow2.y)-double(p.affineRow1.z));
    const double omegaY=0.5*(double(p.affineRow0.z)-double(p.affineRow2.x));
    const double omegaZ=0.5*(double(p.affineRow1.x)-double(p.affineRow0.y));
    const double scale=0.5*double(p.positionAndMass.w)*h*h;
    return {scale*omegaX,scale*omegaY,scale*omegaZ};
}
struct Totals {
    double volume=0.0, mass=0.0, oxygen=0.0, carbonDioxide=0.0;
    Vec3 momentum{}, originAngularMomentum{}, angularMomentum{};
    std::uint32_t active=0u;
};
Totals particleTotals(const std::vector<NMParticleStateGPU>& particles,
                      std::uint32_t objectIndex, double h) {
    Totals result;
    Vec3 centerOfMass{};
    for(const auto& p:particles) {
        if(p.identity.x!=objectIndex ||
           (p.identity.w&NM_OBJECT_ACTIVE)==0u) continue;
        ++result.active;
        result.volume+=p.velocityAndReferenceVolume.w;
        result.mass+=p.positionAndMass.w;
        const Vec3 momentum=particleMomentum(p);
        result.momentum=add(result.momentum,momentum);
        centerOfMass=add(centerOfMass,mul(p.positionAndMass.w,particlePosition(p)));
        result.originAngularMomentum=add(result.originAngularMomentum,
            add(cross(particlePosition(p),momentum),particleSpin(p,h)));
    }
    if(result.mass>0.0) centerOfMass=mul(1.0/result.mass,centerOfMass);
    result.angularMomentum=sub(result.originAngularMomentum,
        cross(centerOfMass,result.momentum));
    return result;
}
void stagePacket(Respiration& respiration, float density, Vec3 position,
                 Vec3 velocity, float volume) {
    auto* state=static_cast<NMHumanRespirationState*>(respiration.accepted.contents);
    require(state!=nullptr,"respiration accepted state is unavailable");
    auto& blood=state->externalBlood;
    require(blood.pendingParcel.x==0.0f&&blood.pendingParcel.y==0.0f,
            "prior MPM packet was not consumed before staging");
    const float mass=density*volume;
    const float oxygen=volume*kOxygenFraction;
    const float carbonDioxide=volume*kCarbonDioxideFraction;
    kahanAdd(volume,blood.total.x,blood.compensation.x);
    kahanAdd(mass,blood.total.y,blood.compensation.y);
    kahanAdd(oxygen,blood.total.z,blood.compensation.z);
    kahanAdd(carbonDioxide,blood.total.w,blood.compensation.w);
    const float donorOxygen=volume*kOxygenFraction;
    const float donorCarbonDioxide=volume*kCarbonDioxideFraction;
    auto& donor=state->bloodGas[7];
    require(donor.x>=donorOxygen&&donor.y>=donorCarbonDioxide,
            "synthetic accepted packet exceeds CV donor gas inventory");
    kahanAdd(-donorOxygen,donor.x,donor.z);
    kahanAdd(-donorCarbonDioxide,donor.y,donor.w);
    require(donor.x>=0.0f&&donor.y>=0.0f,
            "synthetic packet donor gas became negative");
    const Vec3 momentum=mul(mass,velocity);
    const double h=respiration.dispatch.bloodMpmGridCellWidth;
    const Vec3 intrinsicSpin={0.0,0.0,0.5*mass*h*h*kSpinRadiansPerSecond};
    const Vec3 originAngular=add(cross(position,momentum),intrinsicSpin);
    blood.pendingParcel={volume,mass,oxygen,carbonDioxide};
    blood.pendingMomentum={float(momentum.x),float(momentum.y),float(momentum.z),0.0f};
    blood.pendingPositionMass={float(mass*position.x),float(mass*position.y),
                               float(mass*position.z),mass};
    blood.pendingAngularMomentum={float(originAngular.x),float(originAngular.y),
                                  float(originAngular.z),0.0f};
    const double speedSquared=velocity.x*velocity.x+
        velocity.y*velocity.y+velocity.z*velocity.z;
    const double spinEnergy=0.25*mass*h*h*
        kSpinRadiansPerSecond*kSpinRadiansPerSecond;
    blood.energyLedger.x=float(0.5*mass*speedSquared+spinEnergy);
}
std::uint32_t bloodObjectIndex(const CompiledWorld& world) {
    std::uint32_t result=NM_INVALID_INDEX;
    for(std::uint32_t i=0;i<world.objects.size();++i) {
        const auto& object=world.objects[i];
        if(object.materialIndex>=world.constitutive.size() ||
           world.constitutive[object.materialIndex].material.name!=
               "blood_high_shear_newtonian_proxy") continue;
        require(result==NM_INVALID_INDEX,"more than one blood receiver compiled");
        result=i;
    }
    require(result!=NM_INVALID_INDEX,"blood receiver is missing");
    return result;
}
double inventoryValue(float total, float compensation) {
    return double(total)-double(compensation);
}
bool near(double actual,double expected,double absolute,double relative) {
    return std::isfinite(actual)&&std::abs(actual-expected)<=
        std::max(absolute,relative*std::max(std::abs(actual),std::abs(expected)));
}
void checkInventory(const NMHumanRespirationState& state,
                    const Totals& particles, double expectedVolume,
                    double density, std::uint32_t expectedFrames) {
    const auto& blood=state.externalBlood;
    const double expectedMass=expectedVolume*density;
    const double expectedOxygen=expectedVolume*kOxygenFraction;
    const double expectedCO2=expectedVolume*kCarbonDioxideFraction;
    require(near(inventoryValue(blood.total.x,blood.compensation.x),
                 expectedVolume,1.0e-8,1.0e-6),"external blood volume mismatch");
    require(near(inventoryValue(blood.total.y,blood.compensation.y),
                 expectedMass,1.0e-8,1.0e-6),"external blood mass mismatch");
    require(near(inventoryValue(blood.total.z,blood.compensation.z),
                 expectedOxygen,1.0e-8,1.0e-6),"external blood O2 mismatch");
    require(near(inventoryValue(blood.total.w,blood.compensation.w),
                 expectedCO2,1.0e-8,1.0e-6),"external blood CO2 mismatch");
    require(near(inventoryValue(blood.mpmRepresented.x,
                                blood.mpmRepresentedCompensation.x),
                 expectedVolume,1.0e-8,1.0e-6),"MPM represented volume mismatch");
    require(near(inventoryValue(blood.mpmRepresented.y,
                                blood.mpmRepresentedCompensation.y),
                 expectedMass,1.0e-8,1.0e-6),"MPM represented mass mismatch");
    require(near(inventoryValue(blood.mpmRepresented.z,
                                blood.mpmRepresentedCompensation.z),
                 expectedOxygen,1.0e-8,1.0e-6),"MPM represented O2 mismatch");
    require(near(inventoryValue(blood.mpmRepresented.w,
                                blood.mpmRepresentedCompensation.w),
                 expectedCO2,1.0e-8,1.0e-6),"MPM represented CO2 mismatch");
    require(near(particles.volume,expectedVolume,1.0e-8,2.0e-5),
            "live particle reference volume mismatch");
    require(near(particles.mass,expectedMass,1.0e-8,2.0e-5),
            "live particle mass mismatch");
    require(particles.active<=128u,"receiver exceeded its fixed 128-slot storage");
    require(expectedFrames>0u,"fixture frame count is empty");
    require(inventoryValue(blood.pendingParcel.x,0.0f)==0.0,
            "successful MPM frame left pending volume");
    require(std::abs(blood.energyLedger.x)<=1.0e-8f,
            "successful MPM frame left pending kinetic energy");
    require(blood.energyLedger.y>=0.0f&&std::isfinite(blood.energyLedger.y),
            "coalescence dissipation ledger is invalid");
    require(std::isfinite(blood.energyLedger.z)&&
            std::abs(blood.energyLedger.w)<=1.0e-8f,
            "coalescence changed reserved energy-ledger lanes");
}
struct RunTotals {
    Totals particles;
    NMHumanRespirationState respiration{};
    std::uint32_t microticks=0u;
    double dt=0.0;
};
RunTotals runCapacity(const char* network,const char* config,const char* material,
                      double xOffset) {
    configureBlood(material,false,xOffset);
    WorldSource seed;
    seed.environmentCount=1u;
    seed.frameTimestep=kFrameSeconds;
    seed.gravity={0.0,0.0,0.0};
    RestingRun run(network,config,kFrameSeconds,true,&seed);
    require(run.world.dispatch.maximumRateExponent==0u,
            "capacity fixture must schedule one microtick per command buffer");
    const std::uint32_t microticks=1u<<run.world.dispatch.maximumRateExponent;
    require(microticks==1u,"capacity fixture scheduled more than one microtick");
    const auto receiver=bloodObjectIndex(run.world);
    const float density=run.world.materials[
        run.world.objects[receiver].materialIndex].bulk.x;
    const double h=run.world.mpm.grids[
        run.world.objects[receiver].elementOffset].metrics.x;
    const Vec3 spawn{xOffset,0.0,0.0};
    const Vec3 velocity{0.0,0.01,0.0};
    bool rollbackChecked=false;
    std::cerr<<"capacity_begin x_offset_m="<<xOffset<<" frames="<<kCapacityFrames<<'\n';
    for(std::uint32_t frame=0;frame<kCapacityFrames;++frame) {
        if(frame%16u==0u) std::cerr<<"capacity_progress x_offset_m="<<xOffset<<" frame="<<frame<<'\n';
        stagePacket(*run.respiration,density,spawn,velocity,kPacketVolumeM3);
        if(frame==120u) {
            const auto before=run.runtime.snapshot();
            require(before.available,"pre-rejection snapshot unavailable");
            auto* accepted=static_cast<NMHumanRespirationState*>(
                run.respiration->accepted.contents);
            const NMHumanRespirationState staged=*accepted;
            const auto rejectedTime=run.batch(frame,1u,true,frame);
            (void)rejectedTime;
            const auto rejected=run.runtime.snapshot();
            require(rejected.available,"rollback snapshot unavailable");
            require(rejected.particles.size()==before.particles.size()&&
                    std::memcmp(rejected.particles.data(),before.particles.data(),
                                before.particles.size()*sizeof(NMParticleStateGPU))==0,
                    "rejected handoff did not restore accepted MPM particle state");
            require(std::memcmp(accepted,&staged,sizeof(staged))==0,
                    "rejected handoff changed accepted staged inventory");
            run.respiration->dispatch.reject=0u;
            run.batch(frame,1u,false);
            rollbackChecked=true;
        } else {
            run.batch(frame,1u,false);
        }
    }
    require(rollbackChecked,"capacity rollback/retry was not exercised");
    const auto snapshot=run.runtime.snapshot();
    require(snapshot.available,"final capacity snapshot unavailable");
    const auto totals=particleTotals(snapshot.particles,receiver,h);
    const double expectedVolume=double(kCapacityFrames)*kPacketVolumeM3;
    const double initialCapacity=128.0e-9;
    require(expectedVolume>initialCapacity,
            "capacity fixture did not exceed the fixed reference-slot volume");
    require(totals.active<=128u&&totals.volume>initialCapacity,
            "capacity fixture did not represent more volume than its 128 dormant slots");
    const auto* state=static_cast<const NMHumanRespirationState*>(
        run.respiration->accepted.contents);
    checkInventory(*state,totals,expectedVolume,density,kCapacityFrames);
    const double expectedMass=expectedVolume*density;
    const Vec3 expectedMomentum{0.0,expectedMass*velocity.y,0.0};
    const Vec3 expectedSpin{0.0,0.0,
        0.5*expectedMass*h*h*kSpinRadiansPerSecond};
    const Vec3 expectedOriginAngular=add(
        cross(spawn,expectedMomentum),expectedSpin);
    require(norm(sub(totals.momentum,expectedMomentum))<=
                std::max(2.0e-7,1.0e-3*norm(expectedMomentum)),
            "zero-gravity MPM momentum changed during sustained injection/coalescence");
    require(norm(sub(totals.originAngularMomentum,expectedOriginAngular))<=
                std::max(2.0e-8,2.0e-2*norm(expectedOriginAngular)),
            "translated origin angular momentum changed during sustained injection/coalescence");
    require(norm(sub(totals.angularMomentum,expectedSpin))<=
                std::max(2.0e-8,2.0e-2*norm(expectedSpin)),
            "COM-relative APIC spin changed during sustained injection/coalescence");
    std::cout<<"capacity_run=PASS x_offset_m="<<xOffset
             <<" integrated_seconds="<<double(kCapacityFrames)*kFrameSeconds
             <<" frame_dt_s="<<kFrameSeconds<<" microticks_per_command_buffer="<<microticks
             <<" command_buffers="<<kCapacityFrames
             <<" injected_volume_ml="<<expectedVolume*1.0e6
             <<" live_particles="<<totals.active
             <<" live_volume_ml="<<totals.volume*1.0e6
             <<" coalescence_energy_loss_J="<<state->externalBlood.energyLedger.y
             <<" momentum_error="<<norm(sub(totals.momentum,expectedMomentum))
             <<" angular_origin_error="<<norm(sub(totals.originAngularMomentum,expectedOriginAngular))
             <<" angular_COM_error="<<norm(sub(totals.angularMomentum,expectedSpin))
             <<" rejected_step_rollback=PASS\n";
    return {totals,*state,microticks,double(kCapacityFrames)*kFrameSeconds};
}
void runFloor(const char* network,const char* config,const char* material) {
    configureBlood(material,true,0.0);
    RestingRun run(network,config,kFrameSeconds,true);
    require(run.world.dispatch.maximumRateExponent==0u,
            "floor fixture must schedule one microtick per command buffer");
    const auto receiver=bloodObjectIndex(run.world);
    const auto& object=run.world.objects[receiver];
    const double h=run.world.mpm.grids[object.elementOffset].metrics.x;
    const auto& grid=run.world.mpm.grids[object.elementOffset];
    // The compiler pads the authored world-space minimum by one grid node.
    const double minZ=(double(grid.nodeMinimumAndObject.z)+1.0)*double(h);
    const double floorZ=minZ+2.0*double(h);
    // GridDescriptor nodeMinimumAndObject.z is an integer node index, so
    // use the authored receiver bound for world-space floor and particle checks.
    // Keep the entire quadratic B-spline support above the plane initially;
    // the in-solver MPM grid-node barrier receives the descending support.
    const float density=run.world.materials[object.materialIndex].bulk.x;
    const Vec3 spawn={0.0,0.0,floorZ+1.5*h+0.001};
    const Vec3 velocity={0.0,0.0,-10.0};
    std::cerr<<"floor_begin frames="<<kFloorFrames<<" floor_z_m="<<floorZ<<'\n';
    for(std::uint32_t frame=0;frame<kFloorFrames;++frame) {
        if(frame%8u==0u) std::cerr<<"floor_progress frame="<<frame<<'\n';
        stagePacket(*run.respiration,density,spawn,velocity,1.0e-9f);
        run.batch(frame,1u,false);
    }
    const auto snapshot=run.runtime.snapshot();
    require(snapshot.available,"floor-contact snapshot unavailable");
    const auto particles=particleTotals(snapshot.particles,receiver,h);
    double minimumZ=std::numeric_limits<double>::infinity();
    for(const auto& p:snapshot.particles) {
        if(p.identity.x!=receiver || (p.identity.w&NM_OBJECT_ACTIVE)==0u) continue;
        minimumZ=std::min(minimumZ,double(p.positionAndMass.z));
    }
    require(particles.active>0u&&std::isfinite(minimumZ),
            "floor fixture has no active blood particles");
    require(minimumZ>=minZ+0.5*h,
            "blood particles left the padded MPM grid despite the fixed floor");
    double normalImpulse=0.0;
    for(const auto& sample:snapshot.contactSamples) {
        if(sample.identity.z==receiver && sample.identity.y<run.world.contact.rigidProxies.size() &&
           run.world.contact.rigidProxies[sample.identity.y].shapeKind==NM_RIGID_PLANE)
            normalImpulse+=std::max(0.0,double(sample.barrier.x));
    }
    require(normalImpulse>0.0,
            "integrated fixed-plane MPM contact produced no normal barrier impulse");
    const auto* state=static_cast<const NMHumanRespirationState*>(
        run.respiration->accepted.contents);
    const double expectedVolume=double(kFloorFrames)*1.0e-9;
    checkInventory(*state,particles,expectedVolume,density,kFloorFrames);
    std::cout<<"floor_run=PASS integrated_seconds="<<double(kFloorFrames)*kFrameSeconds
             <<" frame_dt_s="<<kFrameSeconds
             <<" microticks_per_command_buffer="<<(1u<<run.world.dispatch.maximumRateExponent)
             <<" command_buffers="<<kFloorFrames<<" floor_z_m="<<floorZ
             <<" minimum_particle_z_m="<<minimumZ
             <<" normal_barrier_impulse_Js="<<normalImpulse
             <<" live_volume_ml="<<particles.volume*1.0e6
             <<" claim_scope=fixed_grid_node_contact_no_free_surface_or_pooling\
";
}
} // namespace

int main(int argc,char** argv) {
    std::cerr.setf(std::ios::unitbuf);
    @autoreleasepool {
        try {
            require(argc==4,"usage: sustained_capacity_probe NETWORK CONFIG MATERIAL");
            const char* capacityOnly=std::getenv("NUMI_SUSTAINED_CAPACITY_ONLY");
            const char* floorOnly=std::getenv("NUMI_SUSTAINED_FLOOR_ONLY");
            require(!capacityOnly||std::strcmp(capacityOnly,"0")==0||
                    std::strcmp(capacityOnly,"1")==0,
                    "NUMI_SUSTAINED_CAPACITY_ONLY must be exactly 0 or 1");
            require(!floorOnly||std::strcmp(floorOnly,"0")==0||
                    std::strcmp(floorOnly,"1")==0,
                    "NUMI_SUSTAINED_FLOOR_ONLY must be exactly 0 or 1");
            require(!(capacityOnly&&std::strcmp(capacityOnly,"1")==0&&
                      floorOnly&&std::strcmp(floorOnly,"1")==0),
                    "capacity-only and floor-only modes are mutually exclusive");
            if(capacityOnly&&std::strcmp(capacityOnly,"1")==0) {
                runCapacity(argv[1],argv[2],argv[3],0.0);
                std::cout<<"sustained_blood_capacity=PASS "
                         <<"scope=132_synthetic_packets_existing_MPM_solver "
                         <<"physiological_bleed_rate=NOT_TESTED\n";
            } else if(floorOnly&&std::strcmp(floorOnly,"1")==0) {
                runFloor(argv[1],argv[2],argv[3]);
                std::cout<<"sustained_blood_floor=PASS "
                         <<"scope=synthetic_accepted_packets_existing_MPM_grid_barrier "
                         <<"free_surface_or_pooling=NOT_TESTED\n";
            } else {
                runCapacity(argv[1],argv[2],argv[3],0.0);
                runCapacity(argv[1],argv[2],argv[3],4.0);
                runFloor(argv[1],argv[2],argv[3]);
                std::cout<<"sustained_blood_capacity=PASS "
                         <<"scope=synthetic_accepted_external_packets_plus_existing_MPM_solver "
                         <<"physiological_bleed_rate=NOT_TESTED\n";
            }
            return 0;
        } catch(const std::exception& error) {
            std::cerr<<"sustained_blood_capacity=FAIL reason="<<error.what()<<'\n';
            return 1;
        }
    }
}

