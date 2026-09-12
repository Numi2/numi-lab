#pragma once
#include "shi_hose_reference_generated.hpp"
namespace numi_shi_hose_reference {
struct Observable {std::string_view name;std::size_t index;double scale;};
inline constexpr std::array<Observable,36> observables{{
{"la_volume_m3",23,1e-06},
{"la_pressure_Pa",17,133.0},
{"la_outflow_m3_per_s",21,1e-06},
{"lv_volume_m3",31,1e-06},
{"lv_pressure_Pa",19,133.0},
{"lv_outflow_m3_per_s",29,1e-06},
{"ra_volume_m3",111,1e-06},
{"ra_pressure_Pa",106,133.0},
{"ra_outflow_m3_per_s",109,1e-06},
{"rv_volume_m3",113,1e-06},
{"rv_pressure_Pa",107,133.0},
{"rv_outflow_m3_per_s",47,1e-06},
{"sas_pressure_Pa",28,133.0},
{"sas_outflow_m3_per_s",134,1e-06},
{"sat_pressure_Pa",133,133.0},
{"sat_outflow_m3_per_s",131,1e-06},
{"sar_pressure_Pa",129,133.0},
{"sar_outflow_m3_per_s",132,1e-06},
{"scp_pressure_Pa",130,133.0},
{"scp_outflow_m3_per_s",136,1e-06},
{"svn_pressure_Pa",135,133.0},
{"svn_outflow_m3_per_s",108,1e-06},
{"pas_pressure_Pa",46,133.0},
{"pas_outflow_m3_per_s",68,1e-06},
{"pat_pressure_Pa",67,133.0},
{"pat_outflow_m3_per_s",50,1e-06},
{"par_pressure_Pa",48,133.0},
{"par_outflow_m3_per_s",51,1e-06},
{"pcp_pressure_Pa",49,133.0},
{"pcp_outflow_m3_per_s",70,1e-06},
{"pvn_pressure_Pa",69,133.0},
{"pvn_outflow_m3_per_s",20,1e-06},
{"la_elastance_Pa_per_m3",0,133000000.0},
{"lv_elastance_Pa_per_m3",9,133000000.0},
{"ra_elastance_Pa_per_m3",71,133000000.0},
{"rv_elastance_Pa_per_m3",79,133000000.0},
}};
inline std::array<double,20> native_hydraulic_state(const Values& a) {
return {
a[23]*1e-6, // LA_volume
a[31]*1e-6, // LV_volume
a[28]*a[114]*1e-6, // SAS_compliance_storage
a[133]*a[115]*1e-6, // SAT_compliance_storage
a[135]*a[116]*1e-6, // SVN_compliance_storage
a[111]*1e-6, // RA_volume
a[113]*1e-6, // RV_volume
a[46]*a[53]*1e-6, // PAS_compliance_storage
a[67]*a[54]*1e-6, // PAT_compliance_storage
a[69]*a[55]*1e-6, // PVN_compliance_storage
a[21]*1e-6, // mitral
a[29]*1e-6, // aortic
a[134]*1e-6, // sas_to_sat
a[131]*1e-6, // sat_to_svn
a[108]*1e-6, // svn_to_ra
a[109]*1e-6, // tricuspid
a[47]*1e-6, // pulmonary_valve
a[68]*1e-6, // pas_to_pat
a[50]*1e-6, // pat_to_pvn
a[20]*1e-6, // pvn_to_la
};
}
}
