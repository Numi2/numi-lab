#pragma once
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <span>
#include <string>
namespace numi::human::d37_exact_bank_export {
struct Summary { bool valid=false; std::string failure; std::size_t triangles=0u,nodes=0u,bodies=0u,banks=0u; };
template<class TR,class NR,class BR,class OF,class SR,class ROT>
Summary writeExactEmittedBankSnapshot(std::ostream& to,std::ostream& no,std::ostream& bo,
 const TR& triangles,const NR& nodes,const BR& bodies,std::uint32_t stateOffset,std::uint32_t nodeCount,
 std::span<const std::uint8_t> roles,std::span<const std::uint32_t> rows,OF offset,SR sourceRecord,ROT rotate) {
 Summary r{};
 if(!nodeCount||stateOffset>nodes.size()||nodeCount>nodes.size()-stateOffset||roles.size()!=nodeCount||
    rows.size()!=nodeCount||bodies.empty()||triangles.empty()){r.failure="extent";return r;}
 to<<"asset_triangle_ordinal,body_index,bone_stable_id,source_record_index,triangle_stable_id,"
      "v0x_m,v0y_m,v0z_m,v1x_m,v1y_m,v1z_m,v2x_m,v2y_m,v2z_m,"
      "v0x_bits_u32,v0y_bits_u32,v0z_bits_u32,v1x_bits_u32,v1y_bits_u32,v1z_bits_u32,v2x_bits_u32,v2y_bits_u32,v2z_bits_u32\n";
 to<<std::setprecision(std::numeric_limits<double>::max_digits10);
 std::uint32_t pb=UINT32_MAX,ps=UINT32_MAX;
 for(std::size_t i=0;i<triangles.size();++i){
  const auto& t=triangles[i];
  if(t.bodyIndex>=bodies.size()||t.boneStableId==UINT32_MAX||t.triangleStableId==UINT32_MAX){r.failure="triangle_identity";return r;}
  const bool nb=i==0||t.bodyIndex!=pb||t.boneStableId!=ps;
  const auto sr=sourceRecord(t.bodyIndex,t.boneStableId);
  if(sr==UINT32_MAX){r.failure="source_record_missing";return r;}
  const std::array<std::array<float,3>,3> v{{
   {{t.bodyLocalVertex0.x,t.bodyLocalVertex0.y,t.bodyLocalVertex0.z}},
   {{t.bodyLocalVertex1.x,t.bodyLocalVertex1.y,t.bodyLocalVertex1.z}},
   {{t.bodyLocalVertex2.x,t.bodyLocalVertex2.y,t.bodyLocalVertex2.z}}}};
  for(const auto& x:v)for(float f:x)if(!std::isfinite(f)){r.failure="vertex_nonfinite";return r;}
  to<<i<<','<<t.bodyIndex<<','<<t.boneStableId<<','<<sr<<','<<t.triangleStableId;
  for(const auto& x:v)for(float f:x)to<<','<<f;
  for(const auto& x:v)for(float f:x)to<<','<<std::bit_cast<std::uint32_t>(f);
  to<<'\n';if(nb)++r.banks;pb=t.bodyIndex;ps=t.boneStableId;++r.triangles;
 }
 no<<"local_node,global_node,layer,row,role_mask,x_m,y_m,z_m,mass_kg,"
      "x_bits_u32,y_bits_u32,z_bits_u32,mass_bits_u32\n";
 no<<std::setprecision(std::numeric_limits<double>::max_digits10);
 for(std::uint32_t i=0;i<nodeCount;++i){
  const auto& n=nodes[stateOffset+i];const auto mask=roles[i];
  const std::uint32_t layer=mask==1?0:mask==2?1:mask==4?2:mask==0?3:4;
  const std::uint32_t row=mask&&rows[i]!=UINT32_MAX?rows[i]:i;
  const float x=n.positionAndMass.x,y=n.positionAndMass.y,z=n.positionAndMass.z,m=n.positionAndMass.w;
  if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)||!std::isfinite(m)){r.failure="node_nonfinite";return r;}
  no<<i<<','<<stateOffset+i<<','<<layer<<','<<row<<','<<unsigned(mask)<<','<<x<<','<<y<<','<<z<<','<<m
    <<','<<std::bit_cast<std::uint32_t>(x)<<','<<std::bit_cast<std::uint32_t>(y)
    <<','<<std::bit_cast<std::uint32_t>(z)<<','<<std::bit_cast<std::uint32_t>(m)<<'\n';++r.nodes;
 }
 bo<<"body_index,position_x_m,position_y_m,position_z_m,orientation_qx,orientation_qy,orientation_qz,orientation_qw,"
     "world_to_body_m00,world_to_body_m01,world_to_body_m02,world_to_body_m10,world_to_body_m11,world_to_body_m12,"
     "world_to_body_m20,world_to_body_m21,world_to_body_m22,com_offset_x_m,com_offset_y_m,com_offset_z_m,"
     "com_offset_x_f32,com_offset_y_f32,com_offset_z_f32,position_x_bits_u32,position_y_bits_u32,"
     "position_z_bits_u32,orientation_qx_bits_u32,orientation_qy_bits_u32,orientation_qz_bits_u32,"
     "orientation_qw_bits_u32,com_offset_x_bits_f32,com_offset_y_bits_f32,com_offset_z_bits_f32\n";
 bo<<std::setprecision(std::numeric_limits<double>::max_digits10);
 for(std::uint32_t i=0;i<bodies.size();++i){
  const auto& b=bodies[i];const std::array<float,3> p{b.position.x,b.position.y,b.position.z};
  const std::array<float,4> q{b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w};
  for(float f:p)if(!std::isfinite(f)){r.failure="body_position_nonfinite";return r;}
  for(float f:q)if(!std::isfinite(f)){r.failure="body_orientation_nonfinite";return r;}
  const auto o=offset(i);for(double f:o)if(!std::isfinite(f)){r.failure="offset_nonfinite";return r;}
  const std::array<double,4> qi{-double(b.orientation.x),-double(b.orientation.y),
      -double(b.orientation.z),double(b.orientation.w)};
  std::array<double,9> m{};
  for(std::uint32_t c=0;c<3;++c){
   const std::array<double,4> e{double(c==0),double(c==1),double(c==2),0.0};
   const auto v=rotate(qi,e);m[c]=v[0];m[3+c]=v[1];m[6+c]=v[2];
  }
  bo<<i<<','<<p[0]<<','<<p[1]<<','<<p[2]<<','<<q[0]<<','<<q[1]<<','<<q[2]<<','<<q[3];
  for(double f:m){if(!std::isfinite(f)){r.failure="matrix_nonfinite";return r;}bo<<','<<f;}
  for(double f:o)bo<<','<<f;for(double f:o)bo<<','<<static_cast<float>(f);
  for(float f:p)bo<<','<<std::bit_cast<std::uint32_t>(f);
  for(float f:q)bo<<','<<std::bit_cast<std::uint32_t>(f);
  for(double f:o)bo<<','<<std::bit_cast<std::uint32_t>(static_cast<float>(f));
  bo<<'\n';++r.bodies;
 }
 if(!to.good()||!no.good()||!bo.good()){r.failure="stream_write";return r;}r.valid=true;return r;
}
} // namespace numi::human::d37_exact_bank_export
