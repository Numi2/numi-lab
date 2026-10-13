#pragma once
// Load-time refinement bindings for passive NHTISS inspection surfaces.
// The existing accepted-pose GPU influence buffer remains the only motion owner.
inline void loadRestingTissueParentBindings(
    const std::filesystem::path& receiptPath, const std::filesystem::path& childPath,
    const RigidHeader& rigid, LoadedSoftTissues& child) {
    NSData* data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:receiptPath.c_str()]];
    require(data!=nil,"cannot read resting anatomy receipt for tissue parents");
    NSError* error=nil;
    id document=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    require([document isKindOfClass:NSDictionary.class],"invalid resting anatomy receipt");
    id extension=document[@"passive_tissue_parent_bindings"];
    if(!extension)return;
    require([extension isKindOfClass:NSDictionary.class],"invalid passive tissue parent bindings");
    auto uintValue=[](id value,const char* label)->unsigned {
        require([value isKindOfClass:NSNumber.class]&&
            CFGetTypeID((__bridge CFTypeRef)value)!=CFBooleanGetTypeID(),
            std::string(label)+" must be an integer");
        const double number=[value doubleValue];
        require(std::isfinite(number)&&number>=0&&number<=UINT32_MAX&&std::floor(number)==number,
            std::string(label)+" is outside uint32");
        return unsigned(number);
    };
    auto digest=[](const std::filesystem::path& path) {
        NSData* bytes=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
        require(bytes!=nil&&bytes.length>0&&bytes.length<=128u*1024u*1024u,
            "tissue parent payload is absent or exceeds the existing bounded mesh domain");
        return loadedKneeSHA256Hex(loadedKneeSHA256(
            static_cast<const std::uint8_t*>(bytes.bytes),bytes.length));
    };
    NSString* childSHA=extension[@"child_payload_sha256"];
    require([childSHA isKindOfClass:NSString.class]&&digest(childPath)==childSHA.UTF8String,
        "tissue parent binding child payload identity mismatch");
    NSArray* sources=extension[@"source_payloads"];
    NSArray* surfaces=extension[@"surfaces"];
    require([sources isKindOfClass:NSArray.class]&&sources.count>0&&sources.count<=8&&
        [surfaces isKindOfClass:NSArray.class]&&surfaces.count>0&&surfaces.count<=192,
        "tissue parent source/surface inventory is invalid");
    std::vector<LoadedSoftTissues> parents;
    std::vector<bool> sourceIsChild;
    for(id descriptor in sources) {
        require([descriptor isKindOfClass:NSDictionary.class],"invalid tissue parent source descriptor");
        NSString* name=descriptor[@"path"];NSString* expected=descriptor[@"sha256"];
        require([name isKindOfClass:NSString.class]&&name.length>0&&
            [expected isKindOfClass:NSString.class]&&expected.length==64,
            "tissue parent source identity is incomplete");
        std::filesystem::path path(name.UTF8String);
        if(!path.is_absolute())path=receiptPath.parent_path()/path;
        require(digest(path)==expected.UTF8String,"tissue parent source SHA-256 mismatch");
        sourceIsChild.push_back([expected isEqualToString:childSHA]);
        parents.push_back(loadSoftTissues(path,rigid));
        require(parents.back().usesRouteBodySparseWeights&&
            parents.back().header.reserved0==child.header.reserved0,
            "tissue parent source registration or ABI differs");
    }
    std::size_t termCount=0,influenceCount=0,identitySurfaceCount=0;
    std::set<unsigned> seenSurfaces;
    for(id surface in surfaces) {
        require([surface isKindOfClass:NSDictionary.class],"invalid tissue parent surface");
        const unsigned stable=uintValue(surface[@"stable_id"],"tissue parent child stable ID");
        auto row=std::find_if(child.records.begin(),child.records.end(),
            [&](const auto& r){return r.stableId==stable;});
        require(row!=child.records.end()&&seenSurfaces.insert(stable).second&&
            !child.parentInfluences.contains(stable),
            "tissue parent child row absent or duplicated");
        NSArray* vertices=surface[@"vertices"];
        require([vertices isKindOfClass:NSArray.class]&&vertices.count==row->vertexCount,
            "tissue parent map must cover every child vertex exactly once");
        bool literalIdentity=true;
        auto& output=child.parentInfluences[stable];output.resize(row->vertexCount);
        for(unsigned v=0;v<vertices.count;++v) {
            NSArray* terms=vertices[v];
            require([terms isKindOfClass:NSArray.class]&&terms.count>0&&terms.count<=128,
                "tissue parent vertex term count is invalid");
            double coefficientSum=0;
            std::set<std::array<unsigned,3>> seen;
            for(id term in terms) {
                require([term isKindOfClass:NSArray.class]&&[term count]==4,
                    "tissue parent term requires source, stable ID, vertex, coefficient");
                const unsigned source=uintValue(term[0],"tissue parent source index"),
                    parentStable=uintValue(term[1],"tissue parent stable ID"),
                    vertex=uintValue(term[2],"tissue parent vertex index");
                require(source<parents.size()&&seen.insert({source,parentStable,vertex}).second,
                    "tissue parent source index invalid or repeated raw vertex term");
                id coefficient=term[3];
                require([coefficient isKindOfClass:NSNumber.class]&&
                    CFGetTypeID((__bridge CFTypeRef)coefficient)!=CFBooleanGetTypeID(),
                    "tissue parent coefficient must be numeric");
                const double c=[coefficient doubleValue];
                require(std::isfinite(c)&&c>0&&c<=1,"tissue parent coefficient must be positive and bounded");
                coefficientSum+=c;
                literalIdentity=literalIdentity&&terms.count==1&&sourceIsChild[source]&&
                    parentStable==stable&&vertex==v&&c==1.0;
                const auto& parent=parents[source];
                const auto pr=std::find_if(parent.records.begin(),parent.records.end(),
                    [&](const auto& r){return r.stableId==parentStable;});
                require(pr!=parent.records.end()&&vertex<pr->vertexCount,
                    "tissue parent raw vertex is absent");
                const auto& pv=parent.vertices.at(pr->firstVertex+vertex);
                for(unsigned j=0;j<4;++j) {
                    if(pv.weight[j]<=0)continue;
                    const unsigned binding=pv.bindingIndex[j];
                    require(binding<pr->bindingCount,"tissue parent binding is absent");
                    const auto& b=pr->binding[binding];
                    const mr_float4 rotation{b.quaternion[0],b.quaternion[1],b.quaternion[2],b.quaternion[3]};
                    auto point=addPoint({b.translation[0],b.translation[1],b.translation[2],0},
                        scalePoint(rotatePoint(rotation,{pv.positionX,pv.positionY,pv.positionZ,0}),b.uniformScale));
                    point.w=float(c*double(pv.weight[j]));
                    require(std::isfinite(point.w)&&point.w>0,"tissue parent positive coefficient underflow");
                    const auto normal=rotatePoint(rotation,{pv.normalX,pv.normalY,pv.normalZ,0});
                    require(influenceCount<4'194'304u,
                        "tissue parent compiled influence inventory exceeds its bounded load budget");
                    output[v].push_back({point,normal,{pr->bodyIndex[binding],0,0,0}});
                    ++influenceCount;
                }
                ++termCount;
            }
            require(std::abs(coefficientSum-1)<=1e-12,"tissue parent coefficients do not form a partition of unity");
            require(!output[v].empty(),"tissue parent vertex has no active influences");
        }
        if(literalIdentity) {
            // A hash-identical payload and exact unit self-reference carry no
            // refinement. Reuse the accepted child buffers instead of rounding
            // normals through an unnecessary world/local/world conversion.
            // All descriptors, terms, bounds and budgets were still validated.
            child.parentInfluences.erase(stable);
            ++identitySurfaceCount;
        }
    }
    std::cout<<"resting_tissue_parent_surfaces="<<child.parentInfluences.size()
        <<" source_vertex_terms="<<termCount<<" compiled_positive_influences="<<influenceCount
        <<" validated_identity_surfaces="<<identitySurfaceCount<<"\n";
}
