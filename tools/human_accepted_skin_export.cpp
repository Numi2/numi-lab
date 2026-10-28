#include "metalrobo/VisualPresentation.hpp"
#include <array>
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
        if(argc!=5) throw std::runtime_error(
            "usage: numi-human-accepted-skin-export PACK.mrvpack FIRST_VERTEX VERTEX_COUNT OUTPUT_PREFIX");
        // The pack is already frozen by the accepted native viewer. This tool
        // never poses geometry or substitutes a CPU deformation implementation.
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
        std::cout<<"native_skin_geometry_export=PASS vertices="<<count
                 <<" active_vertices="<<active<<" faces="<<faces.size()
                 <<" posing=none qualification=not_claimed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n';return 2;
    }
}
