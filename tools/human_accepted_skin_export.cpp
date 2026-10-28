#include "metalrobo/VisualPresentation.hpp"
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

static std::size_t number(std::string_view text) {
    std::size_t value=0;
    const auto [end,ec]=std::from_chars(text.data(),text.data()+text.size(),value);
    if(ec!=std::errc{}||end!=text.data()+text.size())
        throw std::runtime_error("invalid vertex interval");
    return value;
}
int main(int argc,char** argv) {
    try {
        if(argc!=5&&argc!=6) throw std::runtime_error(
            "usage: numi-human-accepted-skin-export PACK.mrvpack FIRST_VERTEX VERTEX_COUNT OUTPUT_PREFIX [WORLD_BONE_CSV]");
        // The pack is already frozen by the accepted native viewer. This tool
        // never advances or deforms physics. Frozen instance transforms are applied
        // only to recover the presentation geometry already owned by this pack.
        metalrobo::VisualAssetPackV2 pack;
        std::string error;
        if(!metalrobo::readVisualAssetPack(argv[1],pack,&error))
            throw std::runtime_error("native pack read rejected: "+error);
        const auto first=number(argv[2]),count=number(argv[3]);
        if(count<4||first>UINT32_MAX||count>UINT32_MAX||first>pack.vertices.size()||count>pack.vertices.size()-first)
            throw std::runtime_error("source skin interval exceeds native pack");
        const auto end=first+count;
        std::vector<std::array<std::uint32_t,3>> faces;
        std::vector<bool> used(count,false);
        if(pack.indices.size()%3!=0)throw std::runtime_error("pack has incomplete triangles");
        for(std::size_t i=0;i<pack.indices.size();i+=3) {
            std::array<std::uint32_t,3> f{pack.indices[i],pack.indices[i+1],pack.indices[i+2]};
            unsigned inside=0;
            for(auto v:f) {
                if(v>=pack.vertices.size())throw std::runtime_error("native index out of range");
                inside+=v>=first&&v<end;
            }
            if(inside==0)continue;
            if(inside!=3)throw std::runtime_error("source skin interval splits a native triangle");
            for(auto& v:f){v-=static_cast<std::uint32_t>(first);used[v]=true;}
            if(f[0]==f[1]||f[0]==f[2]||f[1]==f[2])
                throw std::runtime_error("source skin contains a repeated triangle corner");
            faces.push_back(f);
        }
        if(faces.empty())throw std::runtime_error("source skin interval contains no triangles");
        const auto nodes=std::filesystem::path(std::string(argv[4])+".nodes.csv");
        const auto triangles=std::filesystem::path(std::string(argv[4])+".faces.csv");
        if(std::filesystem::exists(nodes)||std::filesystem::exists(triangles))
            throw std::runtime_error("refusing to overwrite geometry evidence");
        std::ofstream n(nodes),t(triangles);
        if(!n||!t)throw std::runtime_error("cannot open geometry CSV");
        n<<"source_vertex_row,x_m,y_m,z_m,active\n"<<std::setprecision(9);
        std::size_t active=0;
        for(std::size_t i=0;i<count;++i) {
            const auto p=pack.vertices[first+i].position;
            if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))
                throw std::runtime_error("nonfinite accepted skin position");
            n<<i<<','<<p.x<<','<<p.y<<','<<p.z<<','<<used[i]<<'\n';
            active+=used[i];
        }
        t<<"source_face_row,node0,node1,node2\n";
        for(std::size_t i=0;i<faces.size();++i)
            t<<i<<','<<faces[i][0]<<','<<faces[i][1]<<','<<faces[i][2]<<'\n';
        n.flush();t.flush();
        if(!n||!t)throw std::runtime_error("geometry CSV write failed");
        if(argc==6) {
            if(std::filesystem::exists(argv[5]))
                throw std::runtime_error("refusing to overwrite accepted bone evidence");
            std::ofstream b(argv[5]);
            if(!b)throw std::runtime_error("cannot open accepted bone CSV");
            b<<"asset_triangle_ordinal,body_index,bone_stable_id,source_record_index,triangle_stable_id";
            for(unsigned corner=0;corner<3;++corner)
                for(const char axis:std::string_view("xyz"))b<<",v"<<corner<<axis<<"_m";
            for(unsigned corner=0;corner<3;++corner)
                for(const char axis:std::string_view("xyz"))b<<",v"<<corner<<axis<<"_bits_u32";
            b<<'\n'<<std::setprecision(9);
            std::size_t ordinal=0;
            for(const auto& primitive:pack.primitives) {
                if(primitive.identity.x!=51004u)continue;
                if(primitive.geometry.w>=pack.instances.size()||
                   primitive.geometry.x>pack.indices.size()||
                   primitive.geometry.y>pack.indices.size()-primitive.geometry.x||
                   primitive.geometry.y%3!=0)
                    throw std::runtime_error("invalid native bone primitive");
                const auto& instance=pack.instances[primitive.geometry.w];
                if(instance.binding.z!=MR_VISUAL_BINDING_WORLD)
                    throw std::runtime_error("bone instance has not been frozen by accepted export");
                const auto& tr=instance.translationAndScale;
                auto q=instance.orientation;
                const float norm=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
                if(!std::isfinite(norm)||std::abs(norm-1.0f)>2e-5f||
                   !std::isfinite(tr.x)||!std::isfinite(tr.y)||!std::isfinite(tr.z)||
                   !std::isfinite(tr.w)||tr.w<=0)
                    throw std::runtime_error("invalid frozen bone instance transform");
                // composeInstancePose normalizes the frozen orientation. Keep
                // this presentation-only operation in float, as in the renderer.
                q.x/=norm;q.y/=norm;q.z/=norm;q.w/=norm;
                const auto product=[](const std::array<float,4>& a,const std::array<float,4>& c) {
                    return std::array<float,4>{
                        a[3]*c[0]+a[0]*c[3]+a[1]*c[2]-a[2]*c[1],
                        a[3]*c[1]-a[0]*c[2]+a[1]*c[3]+a[2]*c[0],
                        a[3]*c[2]+a[0]*c[1]-a[1]*c[0]+a[2]*c[3],
                        a[3]*c[3]-a[0]*c[0]-a[1]*c[1]-a[2]*c[2]};
                };
                const std::array<float,4> rotation{q.x,q.y,q.z,q.w};
                const std::array<float,4> conjugate{-q.x,-q.y,-q.z,q.w};
                for(unsigned row=0;row<primitive.geometry.y/3;++row) {
                    std::array<float,9> points{};
                    for(unsigned corner=0;corner<3;++corner) {
                        const auto id=pack.indices.at(primitive.geometry.x+3*row+corner);
                        const auto point=pack.vertices.at(id).position;
                        const auto rotated=product(product(rotation,
                            {point.x*tr.w,point.y*tr.w,point.z*tr.w,0.0f}),conjugate);
                        points[3*corner]=rotated[0]+tr.x;
                        points[3*corner+1]=rotated[1]+tr.y;
                        points[3*corner+2]=rotated[2]+tr.z;
                    }
                    b<<ordinal++<<','<<instance.binding.y<<','<<primitive.identity.w
                     <<','<<primitive.identity.z<<','<<row;
                    for(const float value:points) {
                        if(!std::isfinite(value))throw std::runtime_error("nonfinite accepted bone point");
                        b<<','<<value;
                    }
                    for(const float value:points)b<<','<<std::bit_cast<std::uint32_t>(value);
                    b<<'\n';
                }
            }
            b.flush();
            if(!b||ordinal==0)throw std::runtime_error("accepted bone export is empty or failed");
        }
        std::cout<<"native_skin_geometry_export=PASS vertices="<<count
                 <<" active_vertices="<<active<<" faces="<<faces.size()
                 <<" physical_deformation=none bone_coordinates=frozen_instance_world_f32 qualification=not_claimed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n';return 2;
    }
}
