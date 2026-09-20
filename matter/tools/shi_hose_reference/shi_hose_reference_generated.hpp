// Generated solely from the pinned CellML1.1 source imports/MathML.
// Source: Zero dimensional (lumped parameter) modelling of native human cardiovascular dynamics.
// Original CellML by Yubing Shi, Rod Hose and PMR contributors; CC BY 3.0.
// https://models.physiomeproject.org/exposure/c49d416ae3a5132882e6ea7479ba50f5
// License: https://creativecommons.org/licenses/by/3.0/
// Numi translation changes representation only; not biological calibration.
#pragma once
#include <array>
#include <cmath>
#include <limits>
#include <string_view>
namespace numi_shi_hose_reference {
inline constexpr std::string_view revision="a679cdc2e97429fb5280af8132c119758626c1f2";
inline constexpr std::size_t state_count=14;
inline constexpr std::size_t variable_count=137;
using State=std::array<double,state_count>;
using Values=std::array<double,variable_count>;
inline constexpr std::array<std::string_view,state_count> state_names{{"LEFTHeart/LA/TempCDa/V","LEFTHeart/LV/TempCDv/Po","LEFTHeart/LV/TempCDv/V","PulLoop/ModelPul/Pi","PulLoop/Par/TempR/Qi","PulLoop/Pas/TempRLC/Po","PulLoop/Pas/TempRLC/Qo","PulLoop/Pcp/TempR/Po","RIGHTHeart/RA/TempCDa/V","RIGHTHeart/RV/TempCDv/V","SysLoop/Sar/TempR/Qi","SysLoop/Sas/TempRLC/Po","SysLoop/Sas/TempRLC/Qo","SysLoop/Scp/TempR/Po"}};
inline constexpr std::array<std::string_view,variable_count> variable_names{{"LEFTHeart/Ela/EAtrium/E","LEFTHeart/Ela/EAtrium/Emax","LEFTHeart/Ela/EAtrium/Emin","LEFTHeart/Ela/EAtrium/T","LEFTHeart/Ela/EAtrium/Tpwb","LEFTHeart/Ela/EAtrium/Tpww","LEFTHeart/Ela/EAtrium/et","LEFTHeart/Ela/EAtrium/mt","LEFTHeart/Ela/EAtrium/t","LEFTHeart/Elv/EVentricle/E","LEFTHeart/Elv/EVentricle/Emax","LEFTHeart/Elv/EVentricle/Emin","LEFTHeart/Elv/EVentricle/Ts1","LEFTHeart/Elv/EVentricle/Ts2","LEFTHeart/Elv/EVentricle/et","LEFTHeart/Elv/EVentricle/mt","LEFTHeart/LA/TempCDa/CV","LEFTHeart/LA/TempCDa/Pi","LEFTHeart/LA/TempCDa/Pini","LEFTHeart/LA/TempCDa/Po","LEFTHeart/LA/TempCDa/Qi","LEFTHeart/LA/TempCDa/Qo","LEFTHeart/LA/TempCDa/Tao","LEFTHeart/LA/TempCDa/V","LEFTHeart/LA/TempCDa/V0","LEFTHeart/LA/TempCDa/Vini","LEFTHeart/LV/TempCDv/CV","LEFTHeart/LV/TempCDv/Pini","LEFTHeart/LV/TempCDv/Po","LEFTHeart/LV/TempCDv/Qo","LEFTHeart/LV/TempCDv/Tao","LEFTHeart/LV/TempCDv/V","LEFTHeart/LV/TempCDv/V0","LEFTHeart/LV/TempCDv/Vini","LEFTHeart/ParaLHeart/ParaHeart/CVpa","LEFTHeart/ParaLHeart/ParaHeart/CVti","LEFTHeart/ParaLHeart/ParaHeart/EraMax","LEFTHeart/ParaLHeart/ParaHeart/EraMin","LEFTHeart/ParaLHeart/ParaHeart/ErvMax","LEFTHeart/ParaLHeart/ParaHeart/ErvMin","LEFTHeart/ParaLHeart/ParaHeart/PraIni","LEFTHeart/ParaLHeart/ParaHeart/PrvIni","LEFTHeart/ParaLHeart/ParaHeart/Vra0","LEFTHeart/ParaLHeart/ParaHeart/VraIni","LEFTHeart/ParaLHeart/ParaHeart/Vrv0","LEFTHeart/ParaLHeart/ParaHeart/VrvIni","PulLoop/ModelPul/Pi","PulLoop/ModelPul/Qi","PulLoop/Par/TempR/Pi","PulLoop/Par/TempR/Po","PulLoop/Par/TempR/Qi","PulLoop/Par/TempR/Qo","PulLoop/Par/TempR/R","PulLoop/ParaPul/ParaPul/Cpas","PulLoop/ParaPul/ParaPul/Cpat","PulLoop/ParaPul/ParaPul/Cpvn","PulLoop/ParaPul/ParaPul/Lpas","PulLoop/ParaPul/ParaPul/Lpat","PulLoop/ParaPul/ParaPul/P0pas","PulLoop/ParaPul/ParaPul/P0pat","PulLoop/ParaPul/ParaPul/P0pvn","PulLoop/ParaPul/ParaPul/Q0pas","PulLoop/ParaPul/ParaPul/Q0pat","PulLoop/ParaPul/ParaPul/Rpas","PulLoop/ParaPul/ParaPul/Rpat","PulLoop/ParaPul/ParaPul/Rpcp","PulLoop/ParaPul/ParaPul/Rpvn","PulLoop/Pas/TempRLC/Po","PulLoop/Pas/TempRLC/Qo","PulLoop/Pcp/TempR/Po","PulLoop/Pcp/TempR/Qo","RIGHTHeart/Era/EAtrium/E","RIGHTHeart/Era/EAtrium/Emax","RIGHTHeart/Era/EAtrium/Emin","RIGHTHeart/Era/EAtrium/T","RIGHTHeart/Era/EAtrium/Tpwb","RIGHTHeart/Era/EAtrium/Tpww","RIGHTHeart/Era/EAtrium/et","RIGHTHeart/Era/EAtrium/mt","RIGHTHeart/Erv/EVentricle/E","RIGHTHeart/Erv/EVentricle/Emax","RIGHTHeart/Erv/EVentricle/Emin","RIGHTHeart/Erv/EVentricle/Ts1","RIGHTHeart/Erv/EVentricle/Ts2","RIGHTHeart/Erv/EVentricle/et","RIGHTHeart/Erv/EVentricle/mt","RIGHTHeart/ParaRHeart/ParaHeart/CVao","RIGHTHeart/ParaRHeart/ParaHeart/CVmi","RIGHTHeart/ParaRHeart/ParaHeart/CVpa","RIGHTHeart/ParaRHeart/ParaHeart/CVti","RIGHTHeart/ParaRHeart/ParaHeart/ElaMax","RIGHTHeart/ParaRHeart/ParaHeart/ElaMin","RIGHTHeart/ParaRHeart/ParaHeart/ElvMax","RIGHTHeart/ParaRHeart/ParaHeart/ElvMin","RIGHTHeart/ParaRHeart/ParaHeart/PlaIni","RIGHTHeart/ParaRHeart/ParaHeart/PlvIni","RIGHTHeart/ParaRHeart/ParaHeart/PraIni","RIGHTHeart/ParaRHeart/ParaHeart/PrvIni","RIGHTHeart/ParaRHeart/ParaHeart/Vla0","RIGHTHeart/ParaRHeart/ParaHeart/VlaIni","RIGHTHeart/ParaRHeart/ParaHeart/Vlv0","RIGHTHeart/ParaRHeart/ParaHeart/VlvIni","RIGHTHeart/ParaRHeart/ParaHeart/Vra0","RIGHTHeart/ParaRHeart/ParaHeart/VraIni","RIGHTHeart/ParaRHeart/ParaHeart/Vrv0","RIGHTHeart/ParaRHeart/ParaHeart/VrvIni","RIGHTHeart/RA/TempCDa/Pi","RIGHTHeart/RA/TempCDa/Po","RIGHTHeart/RA/TempCDa/Qi","RIGHTHeart/RA/TempCDa/Qo","RIGHTHeart/RA/TempCDa/Tao","RIGHTHeart/RA/TempCDa/V","RIGHTHeart/RV/TempCDv/Tao","RIGHTHeart/RV/TempCDv/V","SysLoop/ParaSys/ParaSys/Csas","SysLoop/ParaSys/ParaSys/Csat","SysLoop/ParaSys/ParaSys/Csvn","SysLoop/ParaSys/ParaSys/Lsas","SysLoop/ParaSys/ParaSys/Lsat","SysLoop/ParaSys/ParaSys/P0sas","SysLoop/ParaSys/ParaSys/P0sat","SysLoop/ParaSys/ParaSys/P0svn","SysLoop/ParaSys/ParaSys/Q0sas","SysLoop/ParaSys/ParaSys/Q0sat","SysLoop/ParaSys/ParaSys/Rsar","SysLoop/ParaSys/ParaSys/Rsas","SysLoop/ParaSys/ParaSys/Rsat","SysLoop/ParaSys/ParaSys/Rscp","SysLoop/ParaSys/ParaSys/Rsvn","SysLoop/Sar/TempR/Pi","SysLoop/Sar/TempR/Po","SysLoop/Sar/TempR/Qi","SysLoop/Sar/TempR/Qo","SysLoop/Sas/TempRLC/Po","SysLoop/Sas/TempRLC/Qo","SysLoop/Scp/TempR/Po","SysLoop/Scp/TempR/Qo"}};
inline State initial_state() {return State{20.0,100.0,500.0,30.0,0.0,30.0,0.0,0.0,20.0,500.0,0.0,100.0,0.0,0.0};}
inline void evaluate(double t, const State& y, State& dy, Values* outputs=nullptr) {
Values a{};
a[8]=t;
a[1]=0.25; // LEFTHeart/Ela/EAtrium/Emax
a[2]=0.15; // LEFTHeart/Ela/EAtrium/Emin
a[3]=1.0; // LEFTHeart/Ela/EAtrium/T
a[4]=0.92; // LEFTHeart/Ela/EAtrium/Tpwb
a[5]=0.09; // LEFTHeart/Ela/EAtrium/Tpww
a[10]=2.5; // LEFTHeart/Elv/EVentricle/Emax
a[11]=0.1; // LEFTHeart/Elv/EVentricle/Emin
a[12]=0.3; // LEFTHeart/Elv/EVentricle/Ts1
a[13]=0.45; // LEFTHeart/Elv/EVentricle/Ts2
a[16]=400.0; // LEFTHeart/LA/TempCDa/CV
a[18]=1.0; // LEFTHeart/LA/TempCDa/Pini
a[24]=20.0; // LEFTHeart/LA/TempCDa/V0
a[25]=4.0; // LEFTHeart/LA/TempCDa/Vini
a[26]=350.0; // LEFTHeart/LV/TempCDv/CV
a[27]=1.0; // LEFTHeart/LV/TempCDv/Pini
a[32]=500.0; // LEFTHeart/LV/TempCDv/V0
a[33]=5.0; // LEFTHeart/LV/TempCDv/Vini
a[34]=350.0; // LEFTHeart/ParaLHeart/ParaHeart/CVpa
a[35]=400.0; // LEFTHeart/ParaLHeart/ParaHeart/CVti
a[36]=0.25; // LEFTHeart/ParaLHeart/ParaHeart/EraMax
a[37]=0.15; // LEFTHeart/ParaLHeart/ParaHeart/EraMin
a[38]=1.15; // LEFTHeart/ParaLHeart/ParaHeart/ErvMax
a[39]=0.1; // LEFTHeart/ParaLHeart/ParaHeart/ErvMin
a[40]=1.0; // LEFTHeart/ParaLHeart/ParaHeart/PraIni
a[41]=1.0; // LEFTHeart/ParaLHeart/ParaHeart/PrvIni
a[42]=20.0; // LEFTHeart/ParaLHeart/ParaHeart/Vra0
a[43]=4.0; // LEFTHeart/ParaLHeart/ParaHeart/VraIni
a[44]=500.0; // LEFTHeart/ParaLHeart/ParaHeart/Vrv0
a[45]=10.0; // LEFTHeart/ParaLHeart/ParaHeart/VrvIni
a[52]=0.05; // PulLoop/Par/TempR/R
a[53]=0.18; // PulLoop/ParaPul/ParaPul/Cpas
a[54]=3.8; // PulLoop/ParaPul/ParaPul/Cpat
a[55]=20.5; // PulLoop/ParaPul/ParaPul/Cpvn
a[56]=5.2e-05; // PulLoop/ParaPul/ParaPul/Lpas
a[57]=0.0017; // PulLoop/ParaPul/ParaPul/Lpat
a[58]=30.0; // PulLoop/ParaPul/ParaPul/P0pas
a[59]=30.0; // PulLoop/ParaPul/ParaPul/P0pat
a[60]=0.0; // PulLoop/ParaPul/ParaPul/P0pvn
a[61]=0.0; // PulLoop/ParaPul/ParaPul/Q0pas
a[62]=0.0; // PulLoop/ParaPul/ParaPul/Q0pat
a[63]=0.002; // PulLoop/ParaPul/ParaPul/Rpas
a[64]=0.01; // PulLoop/ParaPul/ParaPul/Rpat
a[65]=0.25; // PulLoop/ParaPul/ParaPul/Rpcp
a[66]=0.0006; // PulLoop/ParaPul/ParaPul/Rpvn
a[72]=0.25; // RIGHTHeart/Era/EAtrium/Emax
a[73]=0.15; // RIGHTHeart/Era/EAtrium/Emin
a[74]=1.0; // RIGHTHeart/Era/EAtrium/T
a[75]=0.92; // RIGHTHeart/Era/EAtrium/Tpwb
a[76]=0.09; // RIGHTHeart/Era/EAtrium/Tpww
a[80]=1.15; // RIGHTHeart/Erv/EVentricle/Emax
a[81]=0.1; // RIGHTHeart/Erv/EVentricle/Emin
a[82]=0.3; // RIGHTHeart/Erv/EVentricle/Ts1
a[83]=0.45; // RIGHTHeart/Erv/EVentricle/Ts2
a[86]=350.0; // RIGHTHeart/ParaRHeart/ParaHeart/CVao
a[87]=400.0; // RIGHTHeart/ParaRHeart/ParaHeart/CVmi
a[88]=350.0; // RIGHTHeart/ParaRHeart/ParaHeart/CVpa
a[89]=400.0; // RIGHTHeart/ParaRHeart/ParaHeart/CVti
a[90]=0.25; // RIGHTHeart/ParaRHeart/ParaHeart/ElaMax
a[91]=0.15; // RIGHTHeart/ParaRHeart/ParaHeart/ElaMin
a[92]=2.5; // RIGHTHeart/ParaRHeart/ParaHeart/ElvMax
a[93]=0.1; // RIGHTHeart/ParaRHeart/ParaHeart/ElvMin
a[94]=1.0; // RIGHTHeart/ParaRHeart/ParaHeart/PlaIni
a[95]=1.0; // RIGHTHeart/ParaRHeart/ParaHeart/PlvIni
a[96]=1.0; // RIGHTHeart/ParaRHeart/ParaHeart/PraIni
a[97]=1.0; // RIGHTHeart/ParaRHeart/ParaHeart/PrvIni
a[98]=20.0; // RIGHTHeart/ParaRHeart/ParaHeart/Vla0
a[99]=4.0; // RIGHTHeart/ParaRHeart/ParaHeart/VlaIni
a[100]=500.0; // RIGHTHeart/ParaRHeart/ParaHeart/Vlv0
a[101]=5.0; // RIGHTHeart/ParaRHeart/ParaHeart/VlvIni
a[102]=20.0; // RIGHTHeart/ParaRHeart/ParaHeart/Vra0
a[103]=4.0; // RIGHTHeart/ParaRHeart/ParaHeart/VraIni
a[104]=500.0; // RIGHTHeart/ParaRHeart/ParaHeart/Vrv0
a[105]=10.0; // RIGHTHeart/ParaRHeart/ParaHeart/VrvIni
a[114]=0.08; // SysLoop/ParaSys/ParaSys/Csas
a[115]=1.6; // SysLoop/ParaSys/ParaSys/Csat
a[116]=20.5; // SysLoop/ParaSys/ParaSys/Csvn
a[117]=6.2e-05; // SysLoop/ParaSys/ParaSys/Lsas
a[118]=0.0017; // SysLoop/ParaSys/ParaSys/Lsat
a[119]=100.0; // SysLoop/ParaSys/ParaSys/P0sas
a[120]=100.0; // SysLoop/ParaSys/ParaSys/P0sat
a[121]=0.0; // SysLoop/ParaSys/ParaSys/P0svn
a[122]=0.0; // SysLoop/ParaSys/ParaSys/Q0sas
a[123]=0.0; // SysLoop/ParaSys/ParaSys/Q0sat
a[124]=0.5; // SysLoop/ParaSys/ParaSys/Rsar
a[125]=0.003; // SysLoop/ParaSys/ParaSys/Rsas
a[126]=0.05; // SysLoop/ParaSys/ParaSys/Rsat
a[127]=0.52; // SysLoop/ParaSys/ParaSys/Rscp
a[128]=0.075; // SysLoop/ParaSys/ParaSys/Rsvn
a[23]=y[0]; // LEFTHeart/LA/TempCDa/V
a[28]=y[1]; // LEFTHeart/LV/TempCDv/Po
a[31]=y[2]; // LEFTHeart/LV/TempCDv/V
a[46]=y[3]; // PulLoop/ModelPul/Pi
a[50]=y[4]; // PulLoop/Par/TempR/Qi
a[67]=y[5]; // PulLoop/Pas/TempRLC/Po
a[68]=y[6]; // PulLoop/Pas/TempRLC/Qo
a[69]=y[7]; // PulLoop/Pcp/TempR/Po
a[111]=y[8]; // RIGHTHeart/RA/TempCDa/V
a[113]=y[9]; // RIGHTHeart/RV/TempCDv/V
a[131]=y[10]; // SysLoop/Sar/TempR/Qi
a[133]=y[11]; // SysLoop/Sas/TempRLC/Po
a[134]=y[12]; // SysLoop/Sas/TempRLC/Qo
a[135]=y[13]; // SysLoop/Scp/TempR/Po
a[7]=(a[8] - (a[3] * std::floor((a[8] / a[3])))); // LEFTHeart/Ela/EAtrium/mt
a[15]=(a[8] - (a[3] * std::floor((a[8] / a[3])))); // LEFTHeart/Elv/EVentricle/mt
a[51]=a[50]; // PulLoop/Par/TempR/Qo
a[78]=(a[8] - (a[74] * std::floor((a[8] / a[74])))); // RIGHTHeart/Era/EAtrium/mt
a[85]=(a[8] - (a[74] * std::floor((a[8] / a[74])))); // RIGHTHeart/Erv/EVentricle/mt
a[132]=a[131]; // SysLoop/Sar/TempR/Qo
a[6]=(((a[7] >= 0.0) && (a[7] <= (((a[4] + a[5]) * a[3]) - a[3]))) ? (1.0 - std::cos(((2.0 * 3.14159 * ((a[7] - (a[4] * a[3])) + a[3])) / (a[5] * a[3])))) : (((a[7] > (((a[4] + a[5]) * a[3]) - a[3])) && (a[7] <= (a[4] * a[3]))) ? 0.0 : (((a[7] > (a[4] * a[3])) && (a[7] <= a[3])) ? (1.0 - std::cos(((2.0 * 3.14159 * (a[7] - (a[4] * a[3]))) / (a[5] * a[3])))) : std::numeric_limits<double>::quiet_NaN()))); // LEFTHeart/Ela/EAtrium/et
a[14]=(((a[15] >= 0.0) && (a[15] <= (a[12] * a[3]))) ? (1.0 - std::cos(((3.14159 * a[15]) / (a[12] * a[3])))) : (((a[15] > (a[12] * a[3])) && (a[15] <= (a[13] * a[3]))) ? (1.0 + std::cos(((3.14159 * (a[15] - (a[12] * a[3]))) / ((a[13] - a[12]) * a[3])))) : (((a[15] > (a[13] * a[3])) && (a[15] < a[3])) ? 0.0 : std::numeric_limits<double>::quiet_NaN()))); // LEFTHeart/Elv/EVentricle/et
a[49]=(a[69] + (a[65] * a[51])); // PulLoop/Par/TempR/Po
a[70]=a[51]; // PulLoop/Pcp/TempR/Qo
a[77]=(((a[78] >= 0.0) && (a[78] <= (((a[75] + a[76]) * a[74]) - a[74]))) ? (1.0 - std::cos(((2.0 * 3.14159 * ((a[78] - (a[75] * a[74])) + a[74])) / (a[76] * a[74])))) : (((a[78] > (((a[75] + a[76]) * a[74]) - a[74])) && (a[78] <= (a[75] * a[74]))) ? 0.0 : (((a[78] > (a[75] * a[74])) && (a[78] <= a[74])) ? (1.0 - std::cos(((2.0 * 3.14159 * (a[78] - (a[75] * a[74]))) / (a[76] * a[74])))) : std::numeric_limits<double>::quiet_NaN()))); // RIGHTHeart/Era/EAtrium/et
a[84]=(((a[85] >= 0.0) && (a[85] <= (a[82] * a[74]))) ? (1.0 - std::cos(((3.14159 * a[85]) / (a[82] * a[74])))) : (((a[85] > (a[82] * a[74])) && (a[85] <= (a[83] * a[74]))) ? (1.0 + std::cos(((3.14159 * (a[85] - (a[82] * a[74]))) / ((a[83] - a[82]) * a[74])))) : (((a[85] > (a[83] * a[74])) && (a[85] < a[74])) ? 0.0 : std::numeric_limits<double>::quiet_NaN()))); // RIGHTHeart/Erv/EVentricle/et
a[130]=(a[135] + (a[127] * a[132])); // SysLoop/Sar/TempR/Po
a[136]=a[132]; // SysLoop/Scp/TempR/Qo
a[0]=(a[2] + ((a[6] * (a[1] - a[2])) / 2.0)); // LEFTHeart/Ela/EAtrium/E
a[9]=(a[11] + ((a[14] * (a[10] - a[11])) / 2.0)); // LEFTHeart/Elv/EVentricle/E
a[48]=(a[49] + (a[52] * a[50])); // PulLoop/Par/TempR/Pi
a[71]=(a[73] + ((a[77] * (a[72] - a[73])) / 2.0)); // RIGHTHeart/Era/EAtrium/E
a[79]=(a[81] + ((a[84] * (a[80] - a[81])) / 2.0)); // RIGHTHeart/Erv/EVentricle/E
a[129]=(a[130] + (a[124] * a[131])); // SysLoop/Sar/TempR/Pi
a[17]=(a[18] + (a[0] * (a[23] - a[25]))); // LEFTHeart/LA/TempCDa/Pi
a[19]=(a[27] + (a[9] * (a[31] - a[33]))); // LEFTHeart/LA/TempCDa/Po
a[106]=(a[96] + (a[71] * (a[111] - a[103]))); // RIGHTHeart/RA/TempCDa/Pi
a[107]=(a[97] + (a[79] * (a[113] - a[105]))); // RIGHTHeart/RA/TempCDa/Po
a[20]=((a[69] - a[17]) / a[66]); // LEFTHeart/LA/TempCDa/Qi
a[22]=((a[17] >= a[19]) ? 1.0 : ((a[17] < a[19]) ? 0.0 : std::numeric_limits<double>::quiet_NaN())); // LEFTHeart/LA/TempCDa/Tao
a[30]=((a[19] >= a[28]) ? 1.0 : ((a[19] < a[28]) ? 0.0 : std::numeric_limits<double>::quiet_NaN())); // LEFTHeart/LV/TempCDv/Tao
a[108]=((a[135] - a[106]) / a[128]); // RIGHTHeart/RA/TempCDa/Qi
a[110]=((a[106] >= a[107]) ? 1.0 : ((a[106] < a[107]) ? 0.0 : std::numeric_limits<double>::quiet_NaN())); // RIGHTHeart/RA/TempCDa/Tao
a[112]=((a[107] >= a[46]) ? 1.0 : ((a[107] < a[46]) ? 0.0 : std::numeric_limits<double>::quiet_NaN())); // RIGHTHeart/RV/TempCDv/Tao
a[21]=((a[17] >= a[19]) ? (a[16] * a[22] * std::pow(std::fabs((a[17] - a[19])),0.5)) : ((a[17] < a[19]) ? (-1.0 * a[16] * a[22] * std::pow(std::fabs((a[19] - a[17])),0.5)) : std::numeric_limits<double>::quiet_NaN())); // LEFTHeart/LA/TempCDa/Qo
a[29]=((a[19] >= a[28]) ? (a[26] * a[30] * std::pow(std::fabs((a[19] - a[28])),0.5)) : ((a[19] < a[28]) ? (-1.0 * a[26] * a[30] * std::pow(std::fabs((a[28] - a[19])),0.5)) : std::numeric_limits<double>::quiet_NaN())); // LEFTHeart/LV/TempCDv/Qo
a[47]=((a[107] >= a[46]) ? (a[88] * a[112] * std::pow(std::fabs((a[107] - a[46])),0.5)) : ((a[107] < a[46]) ? (-1.0 * a[88] * a[112] * std::pow(std::fabs((a[46] - a[107])),0.5)) : std::numeric_limits<double>::quiet_NaN())); // PulLoop/ModelPul/Qi
a[109]=((a[106] >= a[107]) ? (a[89] * a[110] * std::pow(std::fabs((a[106] - a[107])),0.5)) : ((a[106] < a[107]) ? (-1.0 * a[89] * a[110] * std::pow(std::fabs((a[107] - a[106])),0.5)) : std::numeric_limits<double>::quiet_NaN())); // RIGHTHeart/RA/TempCDa/Qo
dy[0]=(a[20] - a[21]); // d/dt LEFTHeart/LA/TempCDa/V
dy[1]=((a[29] - a[134]) / a[114]); // d/dt LEFTHeart/LV/TempCDv/Po
dy[2]=(a[21] - a[29]); // d/dt LEFTHeart/LV/TempCDv/V
dy[3]=((a[47] - a[68]) / a[53]); // d/dt PulLoop/ModelPul/Pi
dy[4]=(((a[67] - a[48]) - (a[64] * a[50])) / a[57]); // d/dt PulLoop/Par/TempR/Qi
dy[5]=((a[68] - a[50]) / a[54]); // d/dt PulLoop/Pas/TempRLC/Po
dy[6]=(((a[46] - a[67]) - (a[63] * a[68])) / a[56]); // d/dt PulLoop/Pas/TempRLC/Qo
dy[7]=((a[70] - a[20]) / a[55]); // d/dt PulLoop/Pcp/TempR/Po
dy[8]=(a[108] - a[109]); // d/dt RIGHTHeart/RA/TempCDa/V
dy[9]=(a[109] - a[47]); // d/dt RIGHTHeart/RV/TempCDv/V
dy[10]=(((a[133] - a[129]) - (a[126] * a[131])) / a[118]); // d/dt SysLoop/Sar/TempR/Qi
dy[11]=((a[134] - a[131]) / a[115]); // d/dt SysLoop/Sas/TempRLC/Po
dy[12]=(((a[28] - a[133]) - (a[125] * a[134])) / a[117]); // d/dt SysLoop/Sas/TempRLC/Qo
dy[13]=((a[136] - a[108]) / a[116]); // d/dt SysLoop/Scp/TempR/Po
if(outputs)*outputs=a;
}
} // namespace numi_shi_hose_reference
