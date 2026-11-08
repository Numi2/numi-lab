// Native construction export. This is mesh/material authoring, not a dynamics run.
void writeBottomSealConstruction(const SceneTopology& scene,const std::filesystem::path& dir){
    std::filesystem::create_directories(dir);
    const Grid& front=scene.grids.at(0);const Grid& back=scene.grids.at(1);
    std::map<EdgeKey,unsigned> incidence;
    for(const auto& t:scene.triangles){
        const std::array<std::uint32_t,3> ids{t.nodes.x,t.nodes.y,t.nodes.z};
        for(unsigned i=0;i<3;++i){auto a=ids[i],b=ids[(i+1)%3];++incidence[{std::min(a,b),std::max(a,b)}];}
    }
    std::set<EdgeKey> seals,sides,mouth,hinges;
    for(std::size_t u=0;u+1<front.nu;++u){
        auto a=front.nodes[gridIndex(front,u,0)],b=front.nodes[gridIndex(front,u+1,0)];
        seals.insert({std::min(a,b),std::max(a,b)});
        for(const Grid* g:{&front,&back}){
            a=g->nodes[gridIndex(*g,u,g->nv-1)];b=g->nodes[gridIndex(*g,u+1,g->nv-1)];
            mouth.insert({std::min(a,b),std::max(a,b)});
        }
    }
    for(std::size_t v=0;v+1<front.nv;++v)for(std::size_t u:{std::size_t(0),front.nu-1}){
        const auto a=front.nodes[gridIndex(front,u,v)],b=front.nodes[gridIndex(front,u,v+1)];
        if(a!=back.nodes[gridIndex(back,u,v)]||b!=back.nodes[gridIndex(back,u,v+1)])throw std::runtime_error("disconnected tube side");
        sides.insert({std::min(a,b),std::max(a,b)});
    }
    for(const auto& h:scene.hinges)hinges.insert({std::min(h.nodes.x,h.nodes.y),std::max(h.nodes.x,h.nodes.y)});
    std::map<std::uint32_t,std::vector<std::uint32_t>> boundary;
    for(const auto& [edge,count]:incidence){
        if(count<1||count>2)throw std::runtime_error("nonmanifold construction");
        if(count==1){if(!mouth.contains(edge))throw std::runtime_error("unintended shell opening");boundary[edge.a].push_back(edge.b);boundary[edge.b].push_back(edge.a);}
    }
    if(boundary.size()!=mouth.size())throw std::runtime_error("mouth vertex/edge mismatch");
    for(const auto& [n,neighbors]:boundary)if(neighbors.size()!=2)throw std::runtime_error("mouth not a loop");
    std::set<std::uint32_t> visited;std::vector<std::uint32_t> queue{boundary.begin()->first};
    for(std::size_t i=0;i<queue.size();++i){auto n=queue[i];if(!visited.insert(n).second)continue;for(auto m:boundary.at(n))if(!visited.contains(m))queue.push_back(m);}
    if(visited.size()!=boundary.size())throw std::runtime_error("multiple mouth boundaries");
    for(const auto& edge:sides)if(incidence.at(edge)!=2||!hinges.contains(edge)||seals.contains(edge))throw std::runtime_error("side is not continuous bending film");
    for(const auto& edge:seals)if(incidence.at(edge)!=2||hinges.contains(edge))throw std::runtime_error("invalid bottom line seal");
    const auto euler=static_cast<long long>(scene.rest.size())-static_cast<long long>(incidence.size())+static_cast<long long>(scene.triangles.size());
    if(euler!=1||!scene.fixedNodes.empty()||scene.skippedDegenerateTriangles)throw std::runtime_error("invalid open-top bag topology");
    const double mass=std::accumulate(scene.mass.begin(),scene.mass.end(),0.0);
    if(!std::all_of(scene.mass.begin(),scene.mass.end(),[](double m){return std::isfinite(m)&&m>0;}))throw std::runtime_error("invalid film mass");
    std::ofstream out(dir/"construction.json");out<<std::setprecision(17);
    out<<"{\"schema\":\"numi.bottom_seal_bag.construction.v1\",\"artifact_type\":\"native_construction_not_dynamics\","
       <<"\"dimensions\":{\"laid_flat_width_in\":12,\"height_in\":16,\"laid_flat_width_m\":"<<kBagWidth<<",\"height_m\":"<<kBagHeight<<"},"
       <<"\"grid\":{\"nu\":"<<front.nu<<",\"nv\":"<<front.nv<<"},"
       <<"\"material\":{\"name\":\"same exploratory clear LDPE\",\"gauge_m\":"<<kFilmGauge<<",\"density_kg_m3\":"<<kFilmDensity
       <<",\"youngs_modulus_Pa\":"<<kYoungsModulus<<",\"poisson_ratio\":"<<kPoissonRatio
       <<",\"strain_viscosity_time_s\":"<<kStrainViscosityTime<<",\"bending_viscosity_time_s\":"<<kBendViscosityTime<<"},"
       <<"\"film_mass_kg\":"<<mass<<",\"nominal_flat_film_mass_kg\":"<<2*kBagWidth*kBagHeight*kFilmGauge*kFilmDensity<<','
       <<"\"topology\":{\"nodes\":"<<scene.rest.size()<<",\"triangles\":"<<scene.triangles.size()<<",\"membrane_edges\":"<<scene.edges.size()
       <<",\"bending_hinges\":"<<scene.hinges.size()<<",\"continuous_side_fold_edges\":"<<sides.size()<<",\"bottom_heat_seal_edges\":"<<seals.size()
       <<",\"side_heat_seals\":0,\"mouth_boundary_loops\":1,\"mouth_boundary_edges\":"<<mouth.size()<<",\"euler_characteristic\":"<<euler<<",\"fixed_nodes\":0,\"coins\":0},"
       <<"\"model_limits\":[\"Construction export only; no dynamics measured\",\"Bottom heat seal is an ideal shared line; 5 mm displayed seal band is presentation only\",\"Manufactured side folds use pi rest dihedral; no calibrated plastic crease law\",\"Slightly opened geometric reference gives a small area difference from nominal flat film\",\"Material constants remain uncalibrated; no air drag\"],";
    const auto writeVec=[&](const Vec3& p){out<<'['<<p[0]<<','<<p[1]<<','<<p[2]<<']';};
    out<<"\"positions\":[";for(std::size_t i=0;i<scene.rest.size();++i){if(i)out<<',';writeVec(scene.rest[i]);}out<<"],\"node_mass_kg\":[";
    for(std::size_t i=0;i<scene.mass.size();++i){if(i)out<<',';out<<scene.mass[i];}out<<"],\"triangles\":[";
    for(std::size_t i=0;i<scene.triangles.size();++i){if(i)out<<',';const auto n=scene.triangles[i].nodes;out<<'['<<n.x<<','<<n.y<<','<<n.z<<','<<n.w<<']';}out<<"],\"hinges\":[";
    for(std::size_t i=0;i<scene.hinges.size();++i){if(i)out<<',';const auto& h=scene.hinges[i];out<<'['<<h.nodes.x<<','<<h.nodes.y<<','<<h.nodes.z<<','<<h.nodes.w<<','<<h.material.x<<']';}out<<"],";
    const auto writeEdges=[&](const char* name,const auto& set){out<<'\"'<<name<<"\":[";bool first=true;for(const auto& e:set){if(!first)out<<',';first=false;out<<'['<<e.a<<','<<e.b<<']';}out<<"],";};
    writeEdges("bottom_heat_seal",seals);writeEdges("continuous_side_folds",sides);writeEdges("open_mouth",mouth);
    out<<"\"frames\":[{\"step\":0,\"time_s\":0,\"coins\":[],";
    for(unsigned g=0;g<2;++g){if(g)out<<',';const auto& grid=scene.grids[g];out<<'\"'<<grid.name<<"\":[";for(std::size_t i=0;i<grid.nodes.size();++i){if(i)out<<',';writeVec(scene.rest[grid.nodes[i]]);}out<<']';}
    out<<"}]}\n";out.close();if(!out)throw std::runtime_error("construction export failed");
    std::cout<<"NATIVE CONSTRUCTION EXPORTED: "<<dir/"construction.json"<<" | "<<scene.rest.size()<<" nodes | "<<scene.triangles.size()<<" triangles | bottom heat seal only | "<<mass*1000<<" g\n";
}
