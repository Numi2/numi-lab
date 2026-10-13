#pragma once
#include <type_traits>
// Load-time refinement bindings for passive anatomical inspection surfaces.
// Vessel attachments use the same receipt terms, NHTISS carriers and GPU influences.
// The existing accepted-pose GPU influence buffer remains the only motion owner.
template<class Child>
inline void loadRestingParentBindings(
    const std::filesystem::path& receiptPath, const std::filesystem::path& childPath,
    const RigidHeader& rigid, const Child& child, unsigned registrationFingerprint,
    NSString* surfaceKey, NSString* childDigestKey, bool vesselAttachments,
    std::map<unsigned,std::vector<std::vector<MRHumanRestingInfluence>>>& outputBindings,
    std::size_t existingInfluenceCount=0) {
    NSData* data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:receiptPath.c_str()]];
    require(data!=nil,"cannot read resting anatomy receipt for tissue parents");
    NSError* error=nil;
    id document=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    require([document isKindOfClass:NSDictionary.class],"invalid resting anatomy receipt");
    id extension=document[@"passive_tissue_parent_bindings"];
    if(!extension)return;
    require([extension isKindOfClass:NSDictionary.class],"invalid passive tissue parent bindings");
    if(!extension[surfaceKey])return;
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
    NSString* childSHA=extension[childDigestKey];
    require([childSHA isKindOfClass:NSString.class]&&digest(childPath)==childSHA.UTF8String,
        "tissue parent binding child payload identity mismatch");
    NSArray* sources=extension[@"source_payloads"];
    NSArray* surfaces=extension[surfaceKey];
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
            parents.back().header.reserved0==registrationFingerprint,
            "tissue parent source registration or ABI differs");
    }
    std::size_t termCount=0,influenceCount=existingInfluenceCount,identitySurfaceCount=0,identityVertexCount=0;
    std::set<unsigned> seenSurfaces;
    for(id surface in surfaces) {
        require([surface isKindOfClass:NSDictionary.class],"invalid tissue parent surface");
        const unsigned stable=uintValue(surface[@"stable_id"],"tissue parent child stable ID");
        auto row=std::find_if(child.records.begin(),child.records.end(),
            [&](const auto& r){return r.stableId==stable;});
        require(row!=child.records.end()&&seenSurfaces.insert(stable).second&&
            !outputBindings.contains(stable),
            "tissue parent child row absent or duplicated");
        if(vesselAttachments)require(row->layer==kTorsoAnatomyLayerVessel,
            "anatomy parent attachments are restricted to passive vessel surfaces");
        NSArray* vertices=surface[@"vertices"];
        require([vertices isKindOfClass:NSArray.class]&&vertices.count==row->vertexCount,
            "tissue parent map must cover every child vertex exactly once");
        bool literalIdentity=true;
        auto& output=outputBindings[stable];output.resize(row->vertexCount);
        for(unsigned v=0;v<vertices.count;++v) {
            NSArray* terms=vertices[v];
            require([terms isKindOfClass:NSArray.class]&&terms.count>0&&terms.count<=128,
                "tissue parent vertex term count is invalid");
            double coefficientSum=0;
            bool vertexIdentity=true;
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
                const auto& parent=parents[source];
                const auto pr=std::find_if(parent.records.begin(),parent.records.end(),
                    [&](const auto& r){return r.stableId==parentStable;});
                require(pr!=parent.records.end()&&vertex<pr->vertexCount,
                    "tissue parent raw vertex is absent");
                const auto& pv=parent.vertices.at(pr->firstVertex+vertex);
                bool sameVertex=false;
                if constexpr(std::is_same_v<Child,LoadedSoftTissues>) {
                sameVertex=terms.count==1&&parentStable==stable&&vertex==v&&c==1.0;
                if(sameVertex&&!sourceIsChild[source]) {
                    // Other vertices in a payload may have changed. Prove this
                    // vertex's exact source tuple and route bindings separately.
                    static_assert(sizeof(SoftTissueVertex)==56);
                    static_assert(sizeof(SoftTissueBodyBinding)==32);
                    const auto& cv=child.vertices.at(row->firstVertex+v);
                    sameVertex=std::memcmp(&pv,&cv,sizeof(pv))==0&&
                        pr->bindingCount==row->bindingCount;
                    for(unsigned j=0;sameVertex&&j<row->bindingCount;++j)
                        sameVertex=pr->bodyIndex[j]==row->bodyIndex[j]&&
                            std::memcmp(&pr->binding[j],&row->binding[j],sizeof(pr->binding[j]))==0;
                }
                }
                vertexIdentity=vertexIdentity&&sameVertex;
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
            if(vertexIdentity) {
                // Empty means reuse the admitted child tuple in the renderer;
                // it is assigned only after all original parent checks passed.
                output[v].clear();
                ++identityVertexCount;
            }
            literalIdentity=literalIdentity&&vertexIdentity;
        }
        if(literalIdentity) {
            // Every vertex has an exact unit self-reference with identical
            // source tuples and bindings, so the surface carries no refinement. Reuse the accepted child buffers instead of rounding
            // normals through an unnecessary world/local/world conversion.
            // All descriptors, terms, bounds and budgets were still validated.
            outputBindings.erase(stable);
            ++identitySurfaceCount;
        }
    }
    std::cout<<(vesselAttachments?"resting_anatomy_parent_surfaces=":"resting_tissue_parent_surfaces=")<<outputBindings.size()
        <<" source_vertex_terms="<<termCount<<" compiled_positive_influences="<<influenceCount
        <<" validated_identity_surfaces="<<identitySurfaceCount
        <<" validated_identity_vertices="<<identityVertexCount<<"\n";
}

inline void loadRestingTissueParentBindings(
    const std::filesystem::path& receiptPath, const std::filesystem::path& childPath,
    const RigidHeader& rigid, LoadedSoftTissues& child) {
    loadRestingParentBindings(receiptPath,childPath,rigid,child,child.header.reserved0,
        @"surfaces",@"child_payload_sha256",false,child.parentInfluences);
}
inline void loadRestingAnatomyParentBindings(
    const std::filesystem::path& receiptPath, const std::filesystem::path& childPath,
    const RigidHeader& rigid, const LoadedTorsoAnatomy& child,LoadedSoftTissues& carriers) {
    std::size_t existing=0;
    for(const auto& [stable,vertices]:carriers.parentInfluences)
        for(const auto& terms:vertices)existing+=terms.size();
    loadRestingParentBindings(receiptPath,childPath,rigid,child,child.header.registrationFingerprint,
        @"anatomy_surfaces",@"anatomy_payload_sha256",true,carriers.anatomyParentInfluences,existing);
}
