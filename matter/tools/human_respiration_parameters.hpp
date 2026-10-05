#pragma once
#import <Foundation/Foundation.h>
#include "numi/matter/human_respiration.h"
#include "numi/matter/matter.hpp"
#include <cmath>
#include <array>
#include <stdexcept>

namespace numi::matter {
// Asset/config loading only. No CPU respiratory stepping implementation.
inline NMHumanRespirationParameters readRespirationParameters(const char* path, float dt) {
    NSData* bytes=[NSData dataWithContentsOfFile:@(path)];
    NSError* error=nil;
    NSDictionary* json=bytes?[NSJSONSerialization JSONObjectWithData:bytes options:0 error:&error]:nil;
    if(![json isKindOfClass:NSDictionary.class]) throw std::runtime_error("invalid respiration configuration");
    auto number=[&](NSString* name)->float {
        id item=json[name];
        if(![item isKindOfClass:NSNumber.class] || !std::isfinite([item doubleValue]))
            throw std::runtime_error(std::string("missing finite parameter: ")+name.UTF8String);
        return [item floatValue];
    };
    NMHumanRespirationParameters p{};
    p.lung={number(@"frc_m3"),number(@"dead_space_m3"),number(@"lung_compliance_m3_per_pa"),number(@"airway_resistance_pa_s_per_m3")};
    p.chest={number(@"diaphragm_volume_compliance_m3_per_pa"),number(@"rib_volume_compliance_m3_per_pa"),number(@"diaphragm_damping_pa_s_per_m3"),number(@"rib_damping_pa_s_per_m3")};
    p.geometry={number(@"diaphragm_inertance_pa_s2_per_m3"),number(@"rib_inertance_pa_s2_per_m3"),number(@"diaphragm_area_m2"),number(@"rib_effective_area_m2")};
    p.environment={number(@"rest_pleural_pressure_pa"),number(@"dry_atmospheric_pressure_pa"),number(@"stpd_per_btps"),dt};
    p.metabolism={number(@"inspired_oxygen_fraction"),number(@"inspired_carbon_dioxide_fraction"),number(@"oxygen_consumption_stpd_m3_per_s"),number(@"carbon_dioxide_production_stpd_m3_per_s")};
    p.oxygen={number(@"oxygen_capacity_m3_per_m3_blood"),number(@"dissolved_oxygen_m3_per_m3_per_mmhg"),number(@"oxygen_p50_mmhg"),number(@"oxygen_hill_exponent")};
    p.carbonDioxide={number(@"carbon_dioxide_content_at_40_mmhg"),number(@"carbon_dioxide_content_slope_per_mmhg"),number(@"pulmonary_equilibration_fraction"),0};
    p.tissueRows={3,8,10,12};
    p.tissueFractions={number(@"upper_body_metabolic_fraction"),number(@"renal_metabolic_fraction"),number(@"splanchnic_metabolic_fraction"),number(@"lower_body_metabolic_fraction")};
    p.topology={21,0,21,24};
    for(unsigned m=0;m<2;++m) {
        auto& muscle=p.muscles[m];
        const float optimal=number(m?@"intercostal_optimal_fibre_length_m":@"diaphragm_optimal_fibre_length_m");
        const float slack=number(m?@"intercostal_tendon_slack_m":@"diaphragm_tendon_slack_m");
        const float force=number(m?@"intercostal_maximum_force_n":@"diaphragm_maximum_force_n");
        muscle.lengthRangeAndAcceleration={0.75f*optimal+slack,1.05f*optimal+slack,1,0};
        muscle.controlRange={0,1,0,0};
        muscle.gainParameters[0]={0.75f,1.05f,force,200};
        muscle.gainParameters[1]={0.5f,1.6f,1.5f,1.3f};
        muscle.gainParameters[2]={1.2f,0,0,0};
        for(unsigned i=0;i<3;++i) muscle.biasParameters[i]=muscle.gainParameters[i];
        muscle.dynamicParameters[0]={number(@"activation_time_s"),number(@"deactivation_time_s"),0,0};
        muscle.compliantArchitecture0={optimal,slack,number(@"tendon_strain_at_unit_force"),number(@"tendon_normalized_stiffness")};
        muscle.compliantArchitecture1={number(@"tendon_force_at_toe"),0.5f,number(@"fibre_damping"),0};
    }
    if(!(dt>0&&dt<=.0021f&&p.lung.x>p.lung.y&&p.lung.y>0&&p.lung.z>0&&p.lung.w>0&&
         p.chest.x>0&&p.chest.y>0&&p.chest.z>=0&&p.chest.w>=0&&
         p.geometry.x>=0&&p.geometry.y>=0&&p.geometry.z>0&&p.geometry.w>0&&
         p.environment.y>0&&p.environment.z>0&&p.environment.z<=1&&p.oxygen.x>0&&p.oxygen.y>0&&p.oxygen.z>0&&p.oxygen.w>0&&
         p.carbonDioxide.y>0&&p.carbonDioxide.z>=0&&p.carbonDioxide.z<=1&&
         p.metabolism.x>0&&p.metabolism.x<1&&p.metabolism.y>=0&&p.metabolism.y<1&&p.metabolism.z>=0&&p.metabolism.w>=0))
        throw std::runtime_error("inadmissible respiratory reference parameters");
    if(std::abs(p.tissueFractions.x+p.tissueFractions.y+p.tissueFractions.z+p.tissueFractions.w-1)>1e-6f)
        throw std::runtime_error("metabolic partition does not sum to one");
    return p;
}

inline NMHumanRespirationState initializeRespiration(const NMHumanRespirationParameters& p, const CompiledWorld& w) {
    if(w.vascular.compartments.size()!=21||w.vascular.connections.size()!=24||w.vascular.unknowns.size()!=45)
        throw std::runtime_error("respiration requires the authored CVSim21 graph without duplicate species");
    // Require the authored source mapping before interpreting circuit indices.
    const auto& pulmonary=w.vascular.connections[21];
    if(pulmonary.identity.y!=17||pulmonary.identity.z!=18)
        throw std::runtime_error("pulmonary circuit source mapping changed");
    const std::array<const char*,21> names={"ascending_aorta","brachiocephalic_arteries","upper_body_arteries","upper_body_veins","superior_vena_cava","descending_thoracic_aorta","abdominal_aorta","renal_arteries","renal_veins","splanchnic_arteries","splanchnic_veins","lower_body_arteries","lower_body_veins","abdominal_veins","inferior_vena_cava","right_atrium","right_ventricle","pulmonary_arteries","pulmonary_veins","left_atrium","left_ventricle"};
    double totalBlood=0;
    for(unsigned row=0;row<21;++row) {
        const auto& c=w.vascular.compartments[row];
        const char* actual=reinterpret_cast<const char*>(w.vascular.names.data()+c.identity.y);
        if(std::string(actual)!=std::string("source_aggregate:CVSim21:")+names[row])
            throw std::runtime_error("source-bound respiratory circuit mapping changed");
        totalBlood+=w.vascular.unknowns[row].initialAndScaling.x;
    }
    if(std::abs(totalBlood-0.00515)>1.e-8)throw std::runtime_error("reference blood budget changed");
    NMHumanRespirationState s{};
    s.mechanics={p.lung.x,0,p.environment.x,0};
    s.chamberVolumes={w.vascular.unknowns[15].initialAndScaling.x,w.vascular.unknowns[16].initialAndScaling.x,
        w.vascular.unknowns[19].initialAndScaling.x,w.vascular.unknowns[20].initialAndScaling.x};
    s.circulation={float(totalBlood),s.chamberVolumes.w,s.chamberVolumes.y,0};
    const float arterialO2=100, venousO2=40, arterialCO2=40, venousCO2=46;
    auto oxygen=[&](float pressure){const float ratio=std::pow(pressure/p.oxygen.z,p.oxygen.w);return p.oxygen.x*ratio/(1+ratio)+p.oxygen.y*pressure;};
    auto co2=[&](float pressure){return p.carbonDioxide.x+p.carbonDioxide.y*(pressure-40);};
    for(unsigned row=0;row<21;++row) {
        const bool arterial=row==0||row==1||row==2||row==5||row==6||row==7||row==9||row==11||row>=18;
        const float volume=w.vascular.unknowns[row].initialAndScaling.x;
        s.bloodGas[row]={volume*oxygen(arterial?arterialO2:venousO2),volume*co2(arterial?arterialCO2:venousCO2),0,0};
    }
    const float dryMMHg=p.environment.y/133.322387415f;
    s.alveolarGas={(p.lung.x-p.lung.y)*p.environment.z*arterialO2/dryMMHg,(p.lung.x-p.lung.y)*p.environment.z*arterialCO2/dryMMHg,0,0};
    s.deadSpaceGas={p.lung.y*p.environment.z*arterialO2/dryMMHg,p.lung.y*p.environment.z*arterialCO2/dryMMHg,0,0};
    double oxygenTotal=s.alveolarGas.x+s.deadSpaceGas.x, co2Total=s.alveolarGas.y+s.deadSpaceGas.y;
    for(const auto& gas:s.bloodGas){oxygenTotal+=gas.x;co2Total+=gas.y;}
    s.gasBudget={float(oxygenTotal),float(co2Total),0,0};
    s.observation={100,40,(oxygen(100)-p.oxygen.y*100)/p.oxygen.x,0};
    s.breath={p.lung.x,p.lung.x,0,0};
    return s;
}
}
