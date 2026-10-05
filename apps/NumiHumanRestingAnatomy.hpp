#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include "NumiHumanRestingVascularBindings.hpp"

// Asset admission for the functional anatomical surfaces. This only runs at
// loading; all time-dependent deformation consumes the accepted Metal state.
struct NumiHumanRestingAnatomy {
    MRHumanRestingAnatomyGPU gpu{};
    std::set<unsigned> lungs,pleura,diaphragm,intercostals,sternum;
    std::map<unsigned,unsigned> ribs;
    std::array<unsigned,4> cavities{};
    std::map<unsigned,float> enclosedVolumes;
    std::map<unsigned,std::vector<float>> cardiacFreewallWeights;
    struct CardiacWallBinding { unsigned chamber=0;mr_float4 displacementAndWeight{}; };
    std::map<unsigned,std::vector<CardiacWallBinding>> cardiacWallBindings;
    std::set<unsigned> cardiacWallSurfaces;
    struct ClosedSurface { double volume=0;mr_float4 centroid{}; };
    static double smoothstep(double low,double high,double value) {
        const double t=std::clamp((value-low)/(high-low),0.0,1.0);return t*t*(3-2*t);
    }

    struct TriangleIndex {
        struct Triangle { std::array<unsigned,3> vertex{};std::array<double,3> lower{},upper{},center{}; };
        struct Node { std::array<double,3> lower{},upper{};unsigned begin=0,count=0,left=0,right=0; };
        struct Nearest { double distanceSquared=std::numeric_limits<double>::infinity();
            std::array<double,3> point{};std::array<double,3> barycentric{};std::array<unsigned,3> vertex{}; };
        std::vector<std::array<double,3>> points;
        std::vector<Triangle> triangles;
        std::vector<unsigned> order;
        std::vector<Node> nodes;
        static std::array<double,3> add(const std::array<double,3>& a,const std::array<double,3>& b) {
            return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};
        }
        static std::array<double,3> sub(const std::array<double,3>& a,const std::array<double,3>& b) {
            return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};
        }
        static std::array<double,3> scale(const std::array<double,3>& a,double s) {
            return {a[0]*s,a[1]*s,a[2]*s};
        }
        static double dot(const std::array<double,3>& a,const std::array<double,3>& b) {
            return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
        }
        static double boxDistanceSquared(const Node& node,const std::array<double,3>& p) {
            double value=0;
            for(unsigned k=0;k<3;++k) {
                const double d=p[k]<node.lower[k]?node.lower[k]-p[k]:p[k]>node.upper[k]?p[k]-node.upper[k]:0;
                value+=d*d;
            }
            return value;
        }
        static void closestPoint(const std::array<double,3>& p,const std::array<double,3>& a,
            const std::array<double,3>& b,const std::array<double,3>& c,Nearest& result) {
            const auto ab=sub(b,a),ac=sub(c,a),ap=sub(p,a);
            const double d1=dot(ab,ap),d2=dot(ac,ap);
            std::array<double,3> q{},w{};
            if(d1<=0&&d2<=0){q=a;w={1,0,0};}
            else {
                const auto bp=sub(p,b);const double d3=dot(ab,bp),d4=dot(ac,bp);
                if(d3>=0&&d4<=d3){q=b;w={0,1,0};}
                else {
                    const double vc=d1*d4-d3*d2;
                    if(vc<=0&&d1>=0&&d3<=0){const double v=d1/(d1-d3);q=add(a,scale(ab,v));w={1-v,v,0};}
                    else {
                        const auto cp=sub(p,c);const double d5=dot(ab,cp),d6=dot(ac,cp);
                        if(d6>=0&&d5<=d6){q=c;w={0,0,1};}
                        else {
                            const double vb=d5*d2-d1*d6;
                            if(vb<=0&&d2>=0&&d6<=0){const double v=d2/(d2-d6);q=add(a,scale(ac,v));w={1-v,0,v};}
                            else {
                                const double va=d3*d6-d5*d4;
                                if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){const double v=(d4-d3)/((d4-d3)+(d5-d6));q=add(b,scale(sub(c,b),v));w={0,1-v,v};}
                                else {
                                    const double denominator=1/(va+vb+vc),v=vb*denominator,t=vc*denominator;
                                    q=add(a,add(scale(ab,v),scale(ac,t)));w={1-v-t,v,t};
                                }
                            }
                        }
                    }
                }
            }
            const auto delta=sub(p,q);const double d=dot(delta,delta);
            if(d<result.distanceSquared){result.distanceSquared=d;result.point=q;result.barycentric=w;}
        }
        unsigned build(unsigned begin,unsigned end) {
            Node node;node.begin=begin;node.count=end-begin;
            node.lower.fill(std::numeric_limits<double>::infinity());node.upper.fill(-std::numeric_limits<double>::infinity());
            std::array<double,3> centerLow=node.lower,centerHigh=node.upper;
            for(unsigned row=begin;row<end;++row) {
                const auto& triangle=triangles[order[row]];
                for(unsigned k=0;k<3;++k){node.lower[k]=std::min(node.lower[k],triangle.lower[k]);node.upper[k]=std::max(node.upper[k],triangle.upper[k]);
                    centerLow[k]=std::min(centerLow[k],triangle.center[k]);centerHigh[k]=std::max(centerHigh[k],triangle.center[k]);}
            }
            const unsigned index=unsigned(nodes.size());nodes.push_back(node);
            if(end-begin<=8)return index;
            unsigned axis=0;for(unsigned k=1;k<3;++k)if(centerHigh[k]-centerLow[k]>centerHigh[axis]-centerLow[axis])axis=k;
            const unsigned middle=begin+(end-begin)/2;
            std::nth_element(order.begin()+begin,order.begin()+middle,order.begin()+end,[&](unsigned a,unsigned b){
                const auto x=triangles[a].center[axis],y=triangles[b].center[axis];return x==y?a<b:x<y;});
            const unsigned left=build(begin,middle),right=build(middle,end);
            nodes[index].count=0;nodes[index].left=left;nodes[index].right=right;return index;
        }
        void query(unsigned nodeIndex,const std::array<double,3>& p,Nearest& nearest) const {
            const auto& node=nodes[nodeIndex];if(boxDistanceSquared(node,p)>=nearest.distanceSquared)return;
            if(node.count) {
                for(unsigned row=node.begin;row<node.begin+node.count;++row) {
                    const auto& t=triangles[order[row]];auto candidate=nearest;
                    closestPoint(p,points[t.vertex[0]],points[t.vertex[1]],points[t.vertex[2]],candidate);
                    if(candidate.distanceSquared<nearest.distanceSquared){candidate.vertex=t.vertex;nearest=candidate;}
                }
            } else {
                const double a=boxDistanceSquared(nodes[node.left],p),b=boxDistanceSquared(nodes[node.right],p);
                if(a<=b){query(node.left,p,nearest);query(node.right,p,nearest);}
                else{query(node.right,p,nearest);query(node.left,p,nearest);}
            }
        }
        TriangleIndex(const LoadedTorsoAnatomy& anatomy,const TorsoAnatomyRecord& surface) {
            points.reserve(surface.vertexCount);
            for(unsigned i=surface.firstVertex;i<surface.firstVertex+surface.vertexCount;++i){
                const auto& v=anatomy.vertices.at(i);points.push_back({v.positionX,v.positionY,v.positionZ});}
            triangles.reserve(surface.indexCount/3);
            for(unsigned i=surface.firstIndex;i<surface.firstIndex+surface.indexCount;i+=3) {
                Triangle t;for(unsigned k=0;k<3;++k)t.vertex[k]=anatomy.indices.at(i+k)-surface.firstVertex;
                for(unsigned k=0;k<3;++k){t.lower[k]=std::numeric_limits<double>::infinity();t.upper[k]=-std::numeric_limits<double>::infinity();
                    for(unsigned v:t.vertex){t.lower[k]=std::min(t.lower[k],points.at(v)[k]);t.upper[k]=std::max(t.upper[k],points.at(v)[k]);}
                    t.center[k]=(points[t.vertex[0]][k]+points[t.vertex[1]][k]+points[t.vertex[2]][k])/3;}
                triangles.push_back(t);
            }
            order.resize(triangles.size());std::iota(order.begin(),order.end(),0u);nodes.reserve(triangles.size()*2);
            require(!triangles.empty(),"cardiac closest-surface index is empty");build(0,unsigned(triangles.size()));
        }
        Nearest nearest(const std::array<double,3>& p) const { Nearest result;query(0,p,result);return result; }
    };
    static ClosedSurface closedSurface(const LoadedTorsoAnatomy& anatomy,const TorsoAnatomyRecord& s) {
        std::map<std::pair<unsigned,unsigned>,std::pair<unsigned,int>> edges;
        const auto point=[&](unsigned index) {
            require(index>=s.firstVertex&&index<s.firstVertex+s.vertexCount,"functional anatomy index outside its owner");
            const auto& v=anatomy.vertices.at(index);return std::array<double,3>{v.positionX,v.positionY,v.positionZ};
        };
        const auto origin=point(s.firstVertex);
        double volume=0;std::array<double,3> moment{};
        for(unsigned i=s.firstIndex;i<s.firstIndex+s.indexCount;i+=3) {
            std::array<unsigned,3> ids{anatomy.indices.at(i),anatomy.indices.at(i+1),anatomy.indices.at(i+2)};
            std::array<std::array<double,3>,3> x;
            for(unsigned j=0;j<3;++j) {
                x[j]=point(ids[j]);for(unsigned k=0;k<3;++k)x[j][k]-=origin[k];
                unsigned a=ids[j],b=ids[(j+1)%3];require(a!=b,"functional anatomy degenerate triangle");
                auto& e=edges[{std::min(a,b),std::max(a,b)}];++e.first;e.second+=a<b?1:-1;
            }
            const auto& a=x[0];const auto& b=x[1];const auto& c=x[2];
            const double v=(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6;
            volume+=v;for(unsigned k=0;k<3;++k)moment[k]+=v*(a[k]+b[k]+c[k])/4;
        }
        for(const auto& [key,e]:edges) {
            (void)key;require(e.first==2&&e.second==0,"functional anatomy surface is not a closed oriented manifold");
        }
        require(std::isfinite(volume)&&std::abs(volume)>1e-8,"functional anatomy has no positive enclosed volume");
        ClosedSurface result;result.volume=std::abs(volume);
        result.centroid={float(origin[0]+moment[0]/volume),float(origin[1]+moment[1]/volume),float(origin[2]+moment[2]/volume),0};
        return result;
    }
    NumiHumanRestingAnatomy(const char* receipt,const std::filesystem::path& payload,
        const LoadedTorsoAnatomy& anatomy,const LoadedBones& bones,const NMHumanRespirationParameters& physiology) {
        NSData* data=[NSData dataWithContentsOfFile:@(receipt)];NSError* error=nil;
        NSDictionary* root=data?[NSJSONSerialization JSONObjectWithData:data options:0 error:&error]:nil;
        require([root isKindOfClass:NSDictionary.class],"invalid resting anatomy receipt");
        NSDictionary* provenance=root[@"provenance"];
        NSDictionary* cardiacBinding=[provenance isKindOfClass:NSDictionary.class]?provenance[@"cardiac_geometry_binding"]:nil;
        require([cardiacBinding isKindOfClass:NSDictionary.class]&&
            [cardiacBinding[@"output_anatomy_payload_sha256"] isKindOfClass:NSString.class],
            "resting anatomy lacks the source-bound cardiac cavity ownership receipt");
        NSDictionary* bindings=root[@"functional_bindings"];
        require([bindings isKindOfClass:NSDictionary.class],"resting anatomy receipt lacks functional bindings");
        numiHumanVerifyVascularAnatomy(bindings,provenance,anatomy);
        NSString* hash=bindings[@"anatomy_payload_sha256"];
        require([hash isKindOfClass:NSString.class]&&loadedKneeSHA256Hex(loadedKneeFileSHA256(payload))==hash.UTF8String,
            "resting functional anatomy payload identity differs");
        const auto ids=[&](NSString* key) {
            NSArray* values=bindings[key];require([values isKindOfClass:NSArray.class],"missing resting anatomy ID list");
            std::vector<unsigned> out;for(id value in values) {
                require([value isKindOfClass:NSNumber.class]&&[value doubleValue]==[value unsignedIntValue]&&[value unsignedIntValue]>0,
                    "invalid resting anatomy stable ID");
                out.push_back([value unsignedIntValue]);
            }return out;
        };
        const auto lungIDs=ids(@"lung_stable_ids"),pleuraIDs=ids(@"pleura_stable_ids"),chamberIDs=ids(@"cardiac_cavity_stable_ids");
        const auto diaphragmIDs=ids(@"diaphragm_stable_ids"),intercostalIDs=ids(@"intercostal_stable_ids"),
            ribIDs=ids(@"rib_bone_stable_ids"),sternumIDs=ids(@"sternum_bone_stable_ids");
        const std::array<unsigned,4> expectedCavityIDs{{318,319,320,321}};
        const std::array<unsigned,4> expectedCVCompartments{{16,17,20,21}};
        const std::array<const char*,4> expectedChamberNames{{"right_atrium","right_ventricle","left_atrium","left_ventricle"}};
        const std::array<const char*,4> expectedSourceMembers{{"FJ2424","FJ2423","FJ2425","FJ2422"}};
        const std::array<const char*,4> expectedFMAIDs{{"FMA11359","FMA9291","FMA9465","FMA9466"}};
        require(chamberIDs.size()==expectedCavityIDs.size(),"resting anatomy requires four registered cardiac cavities");
        for(unsigned i=0;i<expectedCavityIDs.size();++i)
            require(chamberIDs[i]==expectedCavityIDs[i],"cardiac cavity stable IDs differ from the registered source order");
        NSArray* cardiacCompartments=bindings[@"cardiac_chamber_compartment_bindings"];
        require([cardiacCompartments isKindOfClass:NSArray.class]&&cardiacCompartments.count==expectedCavityIDs.size(),
            "resting anatomy requires four source-bound CVSim chamber associations");
        for(unsigned i=0;i<expectedCavityIDs.size();++i) {
            NSDictionary* row=cardiacCompartments[i];
            NSNumber* compartment=[row isKindOfClass:NSDictionary.class]?row[@"cvsim_compartment_id"]:nil;
            NSNumber* stableID=[row isKindOfClass:NSDictionary.class]?row[@"stable_id"]:nil;
            NSString* name=[row isKindOfClass:NSDictionary.class]?row[@"name"]:nil;
            NSString* sourceMember=[row isKindOfClass:NSDictionary.class]?row[@"source_member"]:nil;
            NSString* fmaID=[row isKindOfClass:NSDictionary.class]?row[@"fma_id"]:nil;
            require([compartment isKindOfClass:NSNumber.class]&&[compartment doubleValue]==expectedCVCompartments[i]&&
                [stableID isKindOfClass:NSNumber.class]&&[stableID doubleValue]==expectedCavityIDs[i]&&
                [name isKindOfClass:NSString.class]&&[name isEqualToString:[NSString stringWithUTF8String:expectedChamberNames[i]] ]&&
                [sourceMember isKindOfClass:NSString.class]&&[sourceMember isEqualToString:[NSString stringWithUTF8String:expectedSourceMembers[i]] ]&&
                [fmaID isKindOfClass:NSString.class]&&[fmaID isEqualToString:[NSString stringWithUTF8String:expectedFMAIDs[i]] ],
                "CVSim chamber ID is not bound to its registered cavity and source anatomy identity");
        }
        require(diaphragmIDs.size()==1&&intercostalIDs.size()==6&&ribIDs.size()==24&&sternumIDs.size()==2,
            "resting mechanical anatomy requires diaphragm, bilateral intercostals, 24 ribs and sternum");
        NSString* boneHash=bindings[@"bones_payload_sha256"];
        require([boneHash isKindOfClass:NSString.class]&&loadedKneeSHA256Hex(bones.payloadSha256)==boneHash.UTF8String,
            "resting respiratory bones have the wrong registered source identity");
        diaphragm.insert(diaphragmIDs.begin(),diaphragmIDs.end());
        intercostals.insert(intercostalIDs.begin(),intercostalIDs.end());sternum.insert(sternumIDs.begin(),sternumIDs.end());
        for(unsigned i=0;i<ribIDs.size();++i)ribs.emplace(ribIDs[i],i);
        require(ribs.size()==24,"resting rib bindings contain duplicate IDs");
        require(lungIDs.size()==5,"resting anatomy requires five lobes");
        lungs.insert(lungIDs.begin(),lungIDs.end());pleura.insert(pleuraIDs.begin(),pleuraIDs.end());
        require(lungs.size()==5,"resting lung stable IDs are duplicated");std::copy(chamberIDs.begin(),chamberIDs.end(),cavities.begin());
        NSNumber* body=bindings[@"torso_body_index"];
        require([body isKindOfClass:NSNumber.class]&&[body doubleValue]==[body unsignedIntValue],"invalid resting thorax body");
        gpu.bodyAndFlags={body.unsignedIntValue,1,0,0};
        NSArray* axis=bindings[@"superior_axis_body"];
        require([axis isKindOfClass:NSArray.class]&&axis.count==3,"resting anatomy requires body-local superior axis");
        mr_float4 superior{[axis[0] floatValue],[axis[1] floatValue],[axis[2] floatValue],0};
        require(std::isfinite(dotPoint(superior,superior))&&std::abs(dotPoint(superior,superior)-1)<1e-5,
            "resting anatomy superior axis is not unit length");
        NSArray* anterior=bindings[@"anterior_axis_body"];
        require([anterior isKindOfClass:NSArray.class]&&anterior.count==3,"resting anatomy requires body-local anterior axis");
        gpu.anteriorAxis={[anterior[0] floatValue],[anterior[1] floatValue],[anterior[2] floatValue],0};
        require(std::isfinite(dotPoint(gpu.anteriorAxis,gpu.anteriorAxis))&&
            std::abs(dotPoint(gpu.anteriorAxis,gpu.anteriorAxis)-1)<1e-5&&
            std::abs(dotPoint(superior,gpu.anteriorAxis))<1e-5,"resting anatomy axes are not orthonormal");
        const auto surface=[&](unsigned id)->const TorsoAnatomyRecord& {
            auto it=std::find_if(anatomy.records.begin(),anatomy.records.end(),[&](const auto& s){return s.stableId==id;});
            require(it!=anatomy.records.end()&&it->bodyIndex==gpu.bodyAndFlags.x,"functional source surface has wrong body owner");
            return *it;
        };
        mr_float4 centroid{};double totalVolume=0;float inferior=INFINITY,top=-INFINITY;
        for(unsigned id:lungs) {
            const auto& s=surface(id);require(s.layer==7,"functional lung binding is not a source lung lobe");
            const auto closed=closedSurface(anatomy,s);totalVolume+=closed.volume;
            enclosedVolumes[id]=float(closed.volume);
            centroid=addPoint(centroid,scalePoint(closed.centroid,float(closed.volume)));
            for(unsigned i=s.firstVertex;i<s.firstVertex+s.vertexCount;++i) {
                const auto& v=anatomy.vertices[i];const float height=dotPoint({v.positionX,v.positionY,v.positionZ,0},superior);
                inferior=std::min(inferior,height);top=std::max(top,height);
            }
        }
        centroid=scalePoint(centroid,1/float(totalVolume));
        gpu.lungAnchorAndVolume=addPoint(centroid,scalePoint(superior,top-dotPoint(centroid,superior)));
        gpu.lungAnchorAndVolume.w=float(totalVolume);
        gpu.superiorAxisAndHeight=superior;gpu.superiorAxisAndHeight.w=top-inferior;
        gpu.muscleAreas={physiology.geometry.z,physiology.geometry.w,0,0};
        require(totalVolume>physiology.lung.x&&top-inferior>.1f,
            "registered lung envelope cannot contain reference FRC; declared anatomy calibration is required");
        for(unsigned id:pleura)require(surface(id).layer==8,"functional pleural binding has wrong layer");
        float diaphragmLow=INFINITY,diaphragmHigh=-INFINITY;
        for(unsigned id:diaphragm) {
            const auto& s=surface(id);
            for(unsigned i=s.firstVertex;i<s.firstVertex+s.vertexCount;++i) {
                const auto& v=anatomy.vertices[i];const float h=dotPoint({v.positionX,v.positionY,v.positionZ,0},superior);
                diaphragmLow=std::min(diaphragmLow,h);diaphragmHigh=std::max(diaphragmHigh,h);
            }
        }
        require(diaphragmHigh-diaphragmLow>.01f,"source diaphragm has no dome extent");
        gpu.diaphragmHeight={diaphragmLow,diaphragmHigh,0,0};
        for(unsigned id:intercostals)(void)surface(id);
        std::set<unsigned> uniqueChambers;
        for(unsigned i=0;i<4;++i) {
            const auto& s=surface(cavities[i]);require(s.layer==9&&uniqueChambers.insert(cavities[i]).second,
                "functional cardiac cavity identity is invalid");
            const auto closed=closedSurface(anatomy,s);gpu.chamberCenterAndVolume[i]=closed.centroid;
            gpu.chamberCenterAndVolume[i].w=float(closed.volume);
            enclosedVolumes[cavities[i]]=float(closed.volume);
        }
        const auto outputAnatomyHash=loadedKneeSHA256Hex(loadedKneeFileSHA256(payload));
        require([cardiacBinding[@"output_anatomy_payload_sha256"] isKindOfClass:NSString.class]&&
            outputAnatomyHash==[cardiacBinding[@"output_anatomy_payload_sha256"] UTF8String],
            "cardiac cavity ownership receipt names another anatomy payload");
        require([cardiacBinding[@"method"] isKindOfClass:NSString.class]&&
            [cardiacBinding[@"method"] isEqualToString:@"exact_source_face_arrangement_with_RA_priority"],
            "unsupported cardiac cavity ownership convention");

        // Preserve chamber interfaces while deforming the free wall. The
        // registered RA-priority arrangement emits identical FP32 vertices
        // on both sides of its inferred source-overlap cut. The RA/RV field
        // depends only on each vertex's radial direction from that chamber's
        // source centroid, with the exact shared-interface directions fixed
        // and a 2 degree cosine taper. LA/LV have no shared cut and use unit
        // free-wall weights. The positive radial scale preserves ray order;
        // exact cycle geometry remains the admission authority.
        std::array<const TorsoAnatomyRecord*,4> cavityRecords{};
        std::array<std::unique_ptr<TriangleIndex>,4> cavityIndex;
        for(unsigned c=0;c<4;++c){
            cavityRecords[c]=&surface(cavities[c]);cavityIndex[c]=std::make_unique<TriangleIndex>(anatomy,*cavityRecords[c]);}
        auto sourcePoint=[&](const TorsoAnatomyRecord& s,unsigned localVertex) {
            const auto& v=anatomy.vertices.at(s.firstVertex+localVertex);
            return std::array<double,3>{v.positionX,v.positionY,v.positionZ};
        };
        std::set<std::array<double,3>> rightVentricleVertices,sharedInterfaceVertices;
        for(unsigned v=0;v<cavityRecords[1]->vertexCount;++v)
            rightVentricleVertices.insert(sourcePoint(*cavityRecords[1],v));
        for(unsigned v=0;v<cavityRecords[0]->vertexCount;++v) {
            const auto point=sourcePoint(*cavityRecords[0],v);
            if(rightVentricleVertices.contains(point))sharedInterfaceVertices.insert(point);
        }
        require(!sharedInterfaceVertices.empty(),"registered RA/RV partition has no exactly shared source-interface vertices");
        std::array<std::vector<std::array<double,3>>,2> sharedInterfaceDirections;
        for(unsigned c=0;c<2;++c) {
            const auto center=std::array<double,3>{gpu.chamberCenterAndVolume[c].x,
                gpu.chamberCenterAndVolume[c].y,gpu.chamberCenterAndVolume[c].z};
            for(const auto& point:sharedInterfaceVertices) {
                auto direction=TriangleIndex::sub(point,center);const double length=std::sqrt(TriangleIndex::dot(direction,direction));
                require(length>1e-9,"registered RA/RV interface coincides with a chamber centroid");
                sharedInterfaceDirections[c].push_back(TriangleIndex::scale(direction,1.0/length));
            }
        }
        const double interfaceCosine=std::cos(2.0*std::acos(-1.0)/180.0);
        for(unsigned c=0;c<4;++c) {
            const auto& s=*cavityRecords[c];auto& weights=cardiacFreewallWeights[cavities[c]];weights.resize(s.vertexCount);
            if(c>=2)std::fill(weights.begin(),weights.end(),1.0f);
            else {
                const auto center=std::array<double,3>{gpu.chamberCenterAndVolume[c].x,
                    gpu.chamberCenterAndVolume[c].y,gpu.chamberCenterAndVolume[c].z};
                for(unsigned v=0;v<s.vertexCount;++v) {
                    const auto point=sourcePoint(s,v);
                    if(sharedInterfaceVertices.contains(point)){weights[v]=0;continue;}
                    auto direction=TriangleIndex::sub(point,center);const double length=std::sqrt(TriangleIndex::dot(direction,direction));
                    require(length>1e-9,"cardiac source vertex coincides with its chamber centroid");
                    direction=TriangleIndex::scale(direction,1.0/length);
                    double nearestDirectionCosine=-1;
                    for(const auto& seamDirection:sharedInterfaceDirections[c])
                        nearestDirectionCosine=std::max(nearestDirectionCosine,TriangleIndex::dot(direction,seamDirection));
                    const double t=std::clamp((1.0-nearestDirectionCosine)/(1.0-interfaceCosine),0.0,1.0);
                    weights[v]=float(t*t*(3-2*t));
                }
            }
            const auto volumeAt=[&](double q) {
                const std::array<double,3> center{gpu.chamberCenterAndVolume[c].x,gpu.chamberCenterAndVolume[c].y,gpu.chamberCenterAndVolume[c].z};
                std::vector<std::array<double,3>> mapped(s.vertexCount);
                for(unsigned v=0;v<s.vertexCount;++v) {
                    const auto& x=anatomy.vertices.at(s.firstVertex+v);const double scale=1+q*weights[v];
                    mapped[v]={center[0]+scale*(x.positionX-center[0]),center[1]+scale*(x.positionY-center[1]),center[2]+scale*(x.positionZ-center[2])};
                }
                double volume=0;
                for(unsigned index=s.firstIndex;index<s.firstIndex+s.indexCount;index+=3) {
                    auto a=mapped.at(anatomy.indices.at(index)-s.firstVertex);
                    auto b=mapped.at(anatomy.indices.at(index+1)-s.firstVertex);
                    auto d=mapped.at(anatomy.indices.at(index+2)-s.firstVertex);
                    for(unsigned axis=0;axis<3;++axis){a[axis]-=center[axis];b[axis]-=center[axis];d[axis]-=center[axis];}
                    const double crossX=b[1]*d[2]-b[2]*d[1],crossY=b[2]*d[0]-b[0]*d[2],crossZ=b[0]*d[1]-b[1]*d[0];
                    volume+=(a[0]*crossX+a[1]*crossY+a[2]*crossZ)/6;
                }
                return std::abs(volume);
            };
            const double f0=volumeAt(0),f1=volumeAt(1),fm1=volumeAt(-1),f2=volumeAt(2);
            const double a2=(f1+fm1-2*f0)/2,s1=(f1-fm1)/2,a3=(f2-f0-4*a2-2*s1)/6,a1=s1-a3;
            require(std::isfinite(a1)&&std::isfinite(a2)&&std::isfinite(a3)&&std::abs(f0-gpu.chamberCenterAndVolume[c].w)<
                2e-5*gpu.chamberCenterAndVolume[c].w,"cardiac free-wall volume polynomial failed source registration");
            constexpr double qLow=-.999,qHigh=1.0;
            const auto volumeAtQ=[&](double q){return ((a3*q+a2)*q+a1)*q+f0;};
            const auto derivativeAtQ=[&](double q){return (3*a3*q+2*a2)*q+a1;};
            double minimumDerivative=std::min(derivativeAtQ(qLow),derivativeAtQ(qHigh));
            if(a3>0) {
                const double stationary=-a2/(3*a3);
                if(stationary>qLow&&stationary<qHigh)
                    minimumDerivative=std::min(minimumDerivative,derivativeAtQ(stationary));
            }
            const double minimumVolume=volumeAtQ(qLow);
            require(std::isfinite(minimumVolume)&&std::isfinite(volumeAtQ(qHigh))&&
                std::isfinite(minimumDerivative)&&minimumVolume>0&&minimumDerivative>1e-12,
                "cardiac free-wall volume map is not positive and monotone on the admitted q interval");
            if(c==0)require(minimumVolume<23.7e-6,
                "cardiac positive-radius branch cannot represent the observed 23.7 mL reference RA minimum");
            gpu.chamberVolumePolynomial[c]={float(f0),float(a1),float(a2),float(a3)};
        }

        // These exact BodyParts3D source records are the passive external
        // heart/myocardial geometry in the expanded scene receipt. Each
        // surface vertex follows the nearest hydraulic cavity wall through
        // the same accepted q, using that wall's free-wall interpolation and
        // a short spatial falloff. This is a presentation binding, not a
        // contractile myocardial material model.
        struct ExpectedWall { unsigned id;const char* member;const char* conceptId;const char* digest; };
        const std::array<ExpectedWall,3> expectedWalls{{
            {1,"FJ2439","FMA7088","e7c21cd1659eced056eb56e28f4fa9019ace451ea6e0333097846177f807bb5a"},
            {23,"FJ2428","FMA13884","ac3c7d6714bed8cf549c97b013546541cf67e189dad2107092a59758aa3ccc45"},
            {24,"FJ2438","FMA7088","187235f3c3612ef27abde924d01c71579c2946435e64319164f43e5ad008284d"},
        }};
        NSDictionary* sourceMap=[provenance isKindOfClass:NSDictionary.class]?provenance[@"source_id_map"]:nil;
        require([sourceMap isKindOfClass:NSDictionary.class],"expanded anatomy receipt lacks source-member identities");
        for(const auto& expected:expectedWalls) {
            NSString* key=[NSString stringWithFormat:@"%u",expected.id];NSDictionary* identity=sourceMap[key];
            NSDictionary* metadata=[identity isKindOfClass:NSDictionary.class]?identity[@"source_owner_metadata"]:nil;
            NSNumber* owner=identity[@"body_index"];NSNumber* layer=identity[@"layer"];
            NSString* member=identity[@"source_member"];NSString* digest=identity[@"source_sha256"];
            NSString* fmaConcept=[metadata isKindOfClass:NSDictionary.class]?metadata[@"concept_id"]:nil;
            require([owner isKindOfClass:NSNumber.class]&&owner.unsignedIntValue==gpu.bodyAndFlags.x&&
                [layer isKindOfClass:NSNumber.class]&&layer.unsignedIntValue==1&&
                [member isKindOfClass:NSString.class]&&[member isEqualToString:[NSString stringWithUTF8String:expected.member]]&&
                [digest isKindOfClass:NSString.class]&&[digest isEqualToString:[NSString stringWithUTF8String:expected.digest]]&&
                [fmaConcept isKindOfClass:NSString.class]&&[fmaConcept isEqualToString:[NSString stringWithUTF8String:expected.conceptId]],
                "passive heart source-member binding differs");
            cardiacWallSurfaces.insert(expected.id);
            const auto& wall=surface(expected.id);auto& mapping=cardiacWallBindings[expected.id];mapping.resize(wall.vertexCount);
            for(unsigned v=0;v<wall.vertexCount;++v) {
                const auto& source=anatomy.vertices.at(wall.firstVertex+v);const std::array<double,3> point{source.positionX,source.positionY,source.positionZ};
                TriangleIndex::Nearest closest;unsigned chamber=0;
                for(unsigned c=0;c<4;++c) {
                    const auto candidate=cavityIndex[c]->nearest(point);
                    if(candidate.distanceSquared<closest.distanceSquared){closest=candidate;chamber=c;}
                }
                const float distance=float(std::sqrt(closest.distanceSquared));
                const double falloff=1-smoothstep(.040,.080,distance);
                const auto& wallFreewall=cardiacFreewallWeights.at(cavities[chamber]);
                double freewall=0;for(unsigned k=0;k<3;++k)freewall+=closest.barycentric[k]*wallFreewall.at(closest.vertex[k]);
                const auto& center=gpu.chamberCenterAndVolume[chamber];
                mapping[v]={chamber,{float(closest.point[0]-center.x),float(closest.point[1]-center.y),
                    float(closest.point[2]-center.z),float(falloff*freewall)}};
            }
        }
    }
};
