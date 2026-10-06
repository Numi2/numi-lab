#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include "NumiHumanRestingVascularBindings.hpp"
#include "NumiHumanRestingCardiacIdentity.hpp"

// Asset admission for the functional anatomical surfaces. This only runs at
// loading; all time-dependent deformation consumes the accepted Metal state.
struct NumiHumanRestingAnatomy {
    MRHumanRestingAnatomyGPU gpu{};
    std::set<unsigned> lungs,pleura,diaphragm,intercostals,sternum;
    std::map<unsigned,unsigned> ribs;
    std::array<unsigned,4> cavities{};
    std::map<unsigned,float> enclosedVolumes;
    std::map<unsigned,float> respiratorySweptAreas;
    std::set<unsigned> passiveViscera;
    unsigned passivePelvicBody=MR_INVALID_INDEX;
    std::array<float,2> passiveTransition{}; // caudal zero and cranial unit weight, m
    struct RespiratoryBasis {
        double start=0,span=0;
        double conformingGridSpacing=0;
        std::array<double,4> rim{};
        std::array<std::array<double,4>,2> crura{};
        std::array<double,2> rimTransition{},cruralTransition{};
        // xyz are the source-space gradient; w is the scalar basal weight.
        std::array<double,4> evaluateSmooth(const std::array<double,3>& p,const std::array<double,3>& axis) const {
            const auto ellipse=[&](const auto& e,const auto& transition,bool inward) {
                const double x=(p[0]-e[0])/e[2],z=(p[2]-e[1])/e[3],r=std::sqrt(x*x+z*z);
                const double t=std::clamp((r-transition[0])/transition[1],0.0,1.0);
                const double s=t*t*(3-2*t),d=6*t*(1-t)/transition[1]*(inward?-1:1);
                return std::array<double,3>{r>0?d*x/(e[2]*r):0,r>0?d*z/(e[3]*r):0,inward?1-s:s};
            };
            const auto r=ellipse(rim,rimTransition,true),a=ellipse(crura[0],cruralTransition,false),
                b=ellipse(crura[1],cruralTransition,false);
            const double w=r[2]*a[2]*b[2];
            const double wx=r[0]*a[2]*b[2]+r[2]*a[0]*b[2]+r[2]*a[2]*b[0];
            const double wz=r[1]*a[2]*b[2]+r[2]*a[1]*b[2]+r[2]*a[2]*b[1];
            const double height=p[0]*axis[0]+p[1]*axis[1]+p[2]*axis[2];
            const double t=std::clamp((height-start)/span,0.0,1.0),g=1-t*t*(3-2*t),dg=-6*t*(1-t)/span;
            return {g*wx+w*dg*axis[0],w*dg*axis[1],g*wz+w*dg*axis[2],g*w};
        }
        std::array<double,4> evaluate(const std::array<double,3>& p,const std::array<double,3>& axis) const {
            if(conformingGridSpacing==0)return evaluateSmooth(p,axis);
            std::array<double,3> corner{},fraction{};
            for(unsigned k=0;k<3;++k) {
                const double u=p[k]/conformingGridSpacing;
                corner[k]=std::floor(u)*conformingGridSpacing;fraction[k]=u-std::floor(u);
            }
            const auto low=corner;
            std::array<unsigned,3> order{0,1,2};
            std::stable_sort(order.begin(),order.end(),[&](unsigned a,unsigned b){return fraction[a]>fraction[b];});
            std::array<double,4> result{};
            double previous=evaluateSmooth(corner,axis)[3];result[3]=previous;
            for(unsigned k:order) {
                corner[k]+=conformingGridSpacing;
                const double next=evaluateSmooth(corner,axis)[3];
                result[k]=(next-previous)/conformingGridSpacing;
                result[3]+=result[k]*(p[k]-low[k]);previous=next;
            }
            return result;
        }
    } respiratoryBasis;
    std::map<unsigned,std::vector<float>> cardiacFreewallWeights;
    std::vector<MRHumanRestingCardiacWallVertexGPU> ventricularWallMap;
    MRHumanRestingCardiacWallGPU ventricularWallGPU{};
    unsigned ventricularWallStableId=MR_INVALID_INDEX;
    struct CardiacWallBinding { unsigned chamber=0;mr_float4 displacementAndWeight{}; };
    std::map<unsigned,std::vector<CardiacWallBinding>> cardiacWallBindings;
    std::set<unsigned> cardiacWallSurfaces;
    struct ClosedSurface { double volume=0;mr_float4 centroid{}; };
    static double smoothstep(double low,double high,double value) {
        const double t=std::clamp((value-low)/(high-low),0.0,1.0);return t*t*(3-2*t);
    }
    static double basalSweptArea(const LoadedTorsoAnatomy& anatomy,const TorsoAnatomyRecord& s,
        mr_float4 superior,const RespiratoryBasis& basis) {
        // Every vertex displacement is parallel to the superior axis. Hence
        // the closed triangular-mesh volume is exactly linear in displacement;
        // all quadratic/cubic determinant terms have parallel columns.
        const auto& first=anatomy.vertices.at(s.firstVertex);
        const std::array<double,3> origin{first.positionX,first.positionY,first.positionZ};
        const std::array<double,3> axis{superior.x,superior.y,superior.z};
        const auto determinant=[](const auto& a,const auto& b,const auto& c) {
            return a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]);
        };
        double area=0;
        for(unsigned i=s.firstIndex;i<s.firstIndex+s.indexCount;i+=3) {
            std::array<std::array<double,3>,3> x,delta;
            for(unsigned j=0;j<3;++j) {
                const auto& v=anatomy.vertices.at(anatomy.indices.at(i+j));
                x[j]={v.positionX,v.positionY,v.positionZ};
                const double weight=basis.evaluate(x[j],axis)[3];
                for(unsigned k=0;k<3;++k){delta[j][k]=-axis[k]*weight;x[j][k]-=origin[k];}
            }
            area+=(determinant(delta[0],x[1],x[2])+determinant(x[0],delta[1],x[2])+determinant(x[0],x[1],delta[2]))/6;
        }
        return area;
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
        if(NSDictionary* passive=bindings[@"passive_viscera_geometry_binding"]) {
            require([passive isKindOfClass:NSDictionary.class]&&
                [passive[@"motion_model"] isEqual:@"common_respiratory_field_with_torso_pelvis_blend_v1"]&&
                [passive[@"parameter_status"] isEqual:@"inferred_reference_attachment_not_measured_subject_motion"]&&
                [passive[@"functional_role"] isEqual:@"passive_geometry_no_independent_forces_mass_or_physiology"],
                "passive visceral binding has an unsupported model or attribution");
            NSArray* values=passive[@"stable_ids"];
            require([values isKindOfClass:NSArray.class],"passive visceral binding lacks source identities");
            NSDictionary* liverIdentity=bindings[@"liver_surface_identity"];
            const bool eightPatchLiver=liverIdentity!=nil;
            if(eightPatchLiver) {
                // This is the only supported reduced liver representation:
                // eight named display patches partition one closed aggregate
                // exterior. They do not define eight independent solids or
                // segment volumes, and source ID 22 is a retired alias for 21.
                const std::string representation="single_closed_aggregate_partitioned_into_eight_passive_exterior_patches_v1";
                NSArray* liverIDs=[liverIdentity isKindOfClass:NSDictionary.class]?liverIdentity[@"stable_ids"]:nil;
                NSDictionary* aliases=[liverIdentity isKindOfClass:NSDictionary.class]?liverIdentity[@"retired_source_aliases"]:nil;
                require([liverIdentity isKindOfClass:NSDictionary.class]&&liverIdentity.count==5&&
                    [liverIdentity[@"representation"] isKindOfClass:NSString.class]&&
                    std::string([liverIdentity[@"representation"] UTF8String])==representation&&
                    [liverIDs isKindOfClass:NSArray.class]&&liverIDs.count==8&&
                    [aliases isKindOfClass:NSDictionary.class]&&aliases.count==1&&
                    [aliases[@"22"] isKindOfClass:NSNumber.class]&&[aliases[@"22"] doubleValue]==21&&
                    [aliases[@"22"] unsignedIntValue]==21&&
                    [liverIdentity[@"internal_segment_boundaries"] isEqual:@NO]&&
                    [liverIdentity[@"segment_volumes_defined"] isEqual:@NO],
                    "unsupported or incomplete eight-patch liver identity declaration");
                std::set<unsigned> declaredLiverIDs;
                for(unsigned i=0;i<8;++i) {
                    id value=liverIDs[i];
                    require([value isKindOfClass:NSNumber.class]&&[value doubleValue]==[value unsignedIntValue]&&
                        [value unsignedIntValue]==14+i&&declaredLiverIDs.insert([value unsignedIntValue]).second,
                        "eight-patch liver IDs must be the ordered exterior display patches 14 through 21");
                }
                NSDictionary* liverSourceMap=[provenance isKindOfClass:NSDictionary.class]?provenance[@"source_id_map"]:nil;
                require([liverSourceMap isKindOfClass:NSDictionary.class]&&liverSourceMap[@"22"]==nil,
                    "retired liver source ID 22 must not remain a rendered source-map record");
                const std::string patchRole="one open surface patch of a single closed aggregate shell; no independent segment volume";
                const std::string patchInterpretation="inferred display identity; no internal segment partition or segment volume";
                for(unsigned idValue:declaredLiverIDs) {
                    NSString* key=[NSString stringWithFormat:@"%u",idValue];
                    NSDictionary* source=[liverSourceMap[key] isKindOfClass:NSDictionary.class]?liverSourceMap[key]:nil;
                    NSDictionary* owner=[source[@"source_owner_metadata"] isKindOfClass:NSDictionary.class]?
                        source[@"source_owner_metadata"]:nil;
                    NSDictionary* repair=[source[@"repair"] isKindOfClass:NSDictionary.class]?source[@"repair"]:nil;
                    const auto& patch=surface(idValue);
                    require(source&&patch.bodyIndex==gpu.bodyAndFlags.x&&patch.layer==1&&
                        [source[@"body_index"] isKindOfClass:NSNumber.class]&&[source[@"body_index"] unsignedIntValue]==20&&
                        [source[@"layer"] isKindOfClass:NSNumber.class]&&[source[@"layer"] unsignedIntValue]==1&&
                        [owner[@"source_atlas"] isEqual:@"Z-Anatomy"]&&
                        [owner[@"geometry_role"] isKindOfClass:NSString.class]&&
                        std::string([owner[@"geometry_role"] UTF8String])==patchRole&&
                        [repair[@"interpretation"] isKindOfClass:NSString.class]&&
                        std::string([repair[@"interpretation"] UTF8String])==patchInterpretation&&
                        [repair[@"patch_face_count"] isKindOfClass:NSNumber.class]&&
                        [repair[@"patch_face_count"] unsignedIntValue]>0,
                        "eight-patch liver source map is not bound to the registered aggregate exterior");
                }
            }
            for(id value in values) {
                require([value isKindOfClass:NSNumber.class]&&[value doubleValue]==[value unsignedIntValue],
                    "invalid passive visceral source identity");
                require(passiveViscera.insert([value unsignedIntValue]).second,
                    "duplicate passive visceral source identity");
            }
            std::set<unsigned> expected{2,3,4,5,13,14,15,16,17,18,19,20,21};
            if(!eightPatchLiver)expected.insert(22);
            for(unsigned id=398;id<=463;++id)expected.insert(id);
            require(passiveViscera==expected,"passive visceral source identity set differs from the reference assembly");
            id lower=passive[@"pelvic_body_index"];
            require([lower isKindOfClass:NSNumber.class]&&[lower doubleValue]==[lower unsignedIntValue]&&
                [lower unsignedIntValue]!=gpu.bodyAndFlags.x,"passive visceral pelvic anchor is invalid");
            passivePelvicBody=[lower unsignedIntValue];
            float anchorMinimum=INFINITY,pelvicMaximum=-INFINITY;
            std::set<unsigned> anchors{2,3,4,5,13,14,15,16,17,18,19,20,21,461};
            if(!eightPatchLiver)anchors.insert(22);
            for(unsigned id:passiveViscera) {
                const auto& s=surface(id); // All shared interfaces use the same source coordinate frame.
                for(unsigned i=s.firstVertex;i<s.firstVertex+s.vertexCount;++i) {
                    const auto& v=anatomy.vertices[i];
                    const float h=dotPoint({v.positionX,v.positionY,v.positionZ,0},superior);
                    if(anchors.contains(id))anchorMinimum=std::min(anchorMinimum,h);
                    if(id==462||id==463)pelvicMaximum=std::max(pelvicMaximum,h);
                }
            }
            require(std::isfinite(anchorMinimum)&&std::isfinite(pelvicMaximum)&&anchorMinimum>pelvicMaximum,
                "source visceral and pelvic anchors do not define a caudal transition");
            NSArray* transition=passive[@"transition_superior_coordinates_m"];
            require([transition isKindOfClass:NSArray.class]&&transition.count==2,
                "passive visceral binding lacks source-derived transition coordinates");
            for(unsigned i=0;i<2;++i)require([transition[i] isKindOfClass:NSNumber.class]&&
                std::isfinite([transition[i] doubleValue]),"invalid passive visceral transition coordinate");
            require(std::abs([transition[0] doubleValue]-pelvicMaximum)<1e-6&&
                std::abs([transition[1] doubleValue]-anchorMinimum)<1e-6,
                "passive visceral transition differs from source geometry");
            passiveTransition={pelvicMaximum,anchorMinimum};
            std::cout<<"resting_passive_viscera=shared_torso_pelvis_field source_surfaces="<<passiveViscera.size()
                <<" caudal_m="<<pelvicMaximum<<" cranial_m="<<anchorMinimum
                <<" upper_body="<<gpu.bodyAndFlags.x<<" lower_body="<<passivePelvicBody
                <<" independent_forces=false independent_mass=false organ_physiology=false\n";
        }
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
        NSDictionary* respiratoryBinding=bindings[@"respiratory_geometry_binding"];
        require([respiratoryBinding isKindOfClass:NSDictionary.class]&&
            [respiratoryBinding[@"lung_motion_model"] isEqual:@"basal_superior_sweep_v1"],
            "resting anatomy requires the source-bound basal respiratory motion basis");
        const auto respiratoryNumber=[&](NSString* key) {
            id value=respiratoryBinding[key];
            require([value isKindOfClass:NSNumber.class]&&std::isfinite([value doubleValue]),
                "invalid respiratory geometry parameter: "+std::string(key.UTF8String));
            return [value doubleValue];
        };
        const double blendStart=respiratoryNumber(@"basal_blend_start_m");
        const double blendSpan=respiratoryNumber(@"basal_blend_span_m");
        const double declaredArea=respiratoryNumber(@"diaphragm_effective_area_m2");
        const auto parameterArray=[&](id values,auto& output) {
            require([values isKindOfClass:NSArray.class]&&[values count]==output.size(),
                "respiratory footprint parameter has wrong shape");
            for(unsigned i=0;i<output.size();++i) {
                id value=[values objectAtIndex:i];
                require([value isKindOfClass:NSNumber.class]&&std::isfinite([value doubleValue]),
                    "respiratory footprint parameter is not finite");
                output[i]=[value doubleValue];
            }
        };
        respiratoryBasis.start=blendStart;respiratoryBasis.span=blendSpan;
        id interpolation=respiratoryBinding[@"basal_weight_interpolation"];
        if(interpolation) {
            require([interpolation isEqual:@"conforming_kuhn_grid_v1"],
                "unsupported respiratory basal interpolation");
            respiratoryBasis.conformingGridSpacing=respiratoryNumber(@"conforming_grid_spacing_m");
            require(respiratoryBasis.conformingGridSpacing==1.0/64.0&&
                superior.x==0&&superior.y==1&&superior.z==0,
                "conforming respiratory cells require the declared source grid and superior axis");
        }
        parameterArray(respiratoryBinding[@"footprint_rim_ellipse_m"],respiratoryBasis.rim);
        parameterArray(respiratoryBinding[@"footprint_rim_transition"],respiratoryBasis.rimTransition);
        parameterArray(respiratoryBinding[@"footprint_crural_transition"],respiratoryBasis.cruralTransition);
        id crura=respiratoryBinding[@"footprint_crural_ellipses_m"];
        require([crura isKindOfClass:NSArray.class]&&[crura count]==2,
            "respiratory footprint requires both source crural regions");
        for(unsigned i=0;i<2;++i)parameterArray([crura objectAtIndex:i],respiratoryBasis.crura[i]);
        require(respiratoryBasis.rim[2]>0&&respiratoryBasis.rim[3]>0&&
            respiratoryBasis.crura[0][2]>0&&respiratoryBasis.crura[0][3]>0&&
            respiratoryBasis.crura[1][2]>0&&respiratoryBasis.crura[1][3]>0&&
            respiratoryBasis.rimTransition[0]>=0&&respiratoryBasis.rimTransition[1]>0&&
            respiratoryBasis.cruralTransition[0]>=0&&respiratoryBasis.cruralTransition[1]>0&&
            [respiratoryBinding[@"parameter_status"] isEqual:@"inferred_reference_registration_not_measured_subject_geometry"],
            "respiratory attachment footprint is invalid or lacks inferred-reference attribution");
        require(blendSpan>0&&blendStart>=inferior&&blendStart+blendSpan<=top,
            "basal respiratory blend is outside the registered lung envelope");
        double sweptArea=0;
        for(unsigned id:lungs) {
            const double area=basalSweptArea(anatomy,surface(id),superior,respiratoryBasis);
            require(std::isfinite(area)&&area>0,"respiratory basis does not expand a registered lobe");
            respiratorySweptAreas[id]=float(area);sweptArea+=area;
        }
        require(sweptArea>0&&std::isfinite(sweptArea)&&
            std::abs(declaredArea/sweptArea-1)<1e-6&&std::abs(double(physiology.geometry.z)/sweptArea-1)<1e-6,
            "respiratory muscle area differs from the declared source-derived swept-volume basis");
        gpu.lungBasalBlend={float(blendStart),float(blendSpan),float(sweptArea),0};
        std::cout<<"resting_lung_motion=basal_superior_sweep_v1 blend_start_m="<<blendStart
            <<" blend_span_m="<<blendSpan<<" effective_area_m2="<<sweptArea<<"\n";
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

        NSDictionary* motionParameters=cardiacBinding[@"cavity_motion_parameters"];
        NSArray* leftOffsets=[motionParameters isKindOfClass:NSDictionary.class]?motionParameters[@"left_center_offsets_m"]:nil;
        NSNumber* taper=[motionParameters isKindOfClass:NSDictionary.class]?motionParameters[@"ra_rv_taper_degrees"]:nil;
        NSNumber* raOffset=[motionParameters isKindOfClass:NSDictionary.class]?motionParameters[@"ra_center_offset_m"]:nil;
        NSNumber* interfaceWeight=[motionParameters isKindOfClass:NSDictionary.class]?motionParameters[@"ra_rv_shared_interface_weight"]:nil;
        NSNumber* leftWeight=[motionParameters isKindOfClass:NSDictionary.class]?motionParameters[@"left_chamber_freewall_weight"]:nil;
        NSString* motionModel=[motionParameters isKindOfClass:NSDictionary.class]?motionParameters[@"model"]:nil;
        NSString* raDirection=[motionParameters isKindOfClass:NSDictionary.class]?motionParameters[@"ra_center_offset_direction"]:nil;
        NSString* leftDirection=[motionParameters isKindOfClass:NSDictionary.class]?motionParameters[@"left_center_offset_direction"]:nil;
        NSString* parameterStatus=[motionParameters isKindOfClass:NSDictionary.class]?motionParameters[@"parameter_status"]:nil;
        require([motionModel isKindOfClass:NSString.class]&&[motionModel isEqualToString:@"source_centroid_radial_freewall_v2"]&&
            [taper isKindOfClass:NSNumber.class]&&std::abs(taper.doubleValue-20.0)<1e-9&&
            [raOffset isKindOfClass:NSNumber.class]&&std::abs(raOffset.doubleValue-.008)<1e-9&&
            [leftOffsets isKindOfClass:NSArray.class]&&leftOffsets.count==2&&
            [leftOffsets[0] isKindOfClass:NSNumber.class]&&[leftOffsets[1] isKindOfClass:NSNumber.class]&&
            std::abs([leftOffsets[0] doubleValue]-.012)<1e-9&&std::abs([leftOffsets[1] doubleValue]-.015)<1e-9&&
            [raDirection isKindOfClass:NSString.class]&&[raDirection isEqualToString:@"unit(source_RA_centroid - source_RV_centroid)"]&&
            [leftDirection isKindOfClass:NSString.class]&&[leftDirection isEqualToString:@"unit(source_LA_centroid - source_LV_centroid)"]&&
            [interfaceWeight isKindOfClass:NSNumber.class]&&interfaceWeight.doubleValue==0.0&&
            [leftWeight isKindOfClass:NSNumber.class]&&leftWeight.doubleValue==1.0&&
            [parameterStatus isKindOfClass:NSString.class]&&
                [parameterStatus isEqualToString:@"inferred_reference_registration_not_measured_subject_geometry"],
            "cardiac geometry receipt does not bind the admitted inferred motion map");
        const double motionTaperDegrees=taper.doubleValue;
        const double raCenterOffset=raOffset.doubleValue;
        const std::array<double,2> leftCenterOffsets{{[leftOffsets[0] doubleValue],[leftOffsets[1] doubleValue]}};

        const auto centerDifference=[&](unsigned first,unsigned second) {
            const auto& a=gpu.chamberCenterAndVolume[first];const auto& b=gpu.chamberCenterAndVolume[second];
            return std::array<double,3>{double(a.x)-b.x,double(a.y)-b.y,double(a.z)-b.z};
        };
        const auto offsetCenter=[&](unsigned chamber,const std::array<double,3>& direction,double distance) {
            const double length=std::sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2]);
            require(std::isfinite(length)&&length>1e-9,"cardiac inferred center offset direction is undefined");
            auto& center=gpu.chamberCenterAndVolume[chamber];
            center.x=float(double(center.x)+direction[0]*(distance/length));
            center.y=float(double(center.y)+direction[1]*(distance/length));
            center.z=float(double(center.z)+direction[2]*(distance/length));
        };
        offsetCenter(0,centerDifference(0,1),raCenterOffset);
        const auto leftOffsetDirection=centerDifference(2,3);
        offsetCenter(2,leftOffsetDirection,leftCenterOffsets[0]);
        offsetCenter(3,leftOffsetDirection,leftCenterOffsets[1]);

        // Preserve chamber interfaces while deforming the free wall. The
        // registered RA-priority arrangement emits identical FP32 vertices
        // on both sides of its inferred source-overlap cut. This receipt's
        // inferred reference registration offsets the RA origin 8 mm away
        // from the RV source centroid and both left origins 12/15 mm along
        // the source LA-minus-LV direction. These are geometry parameters,
        // not measured subject landmarks. The RA/RV field uses a 20 degree
        // cosine taper and pins exact shared-interface vertices; LA/LV keep
        // unit free-wall weights. Positive radial scale and exact accepted-
        // cycle geometry remain the admission authority.
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
        const double interfaceCosine=std::cos(motionTaperDegrees*std::acos(-1.0)/180.0);
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

        // Verify the passive heart source identities. Keep the tested atrial
        // reference bindings for IDs 1 and 24; ID 23 gets the closed source-
        // derived ventricular material boundary and its reduced volume map.
        struct ExpectedWall { unsigned id;const char* member;const char* conceptId;const char* digest; };
        const std::array<ExpectedWall,3> expectedWalls{{
            {1,"FJ2439","FMA9457","e7c21cd1659eced056eb56e28f4fa9019ace451ea6e0333097846177f807bb5a"},
            {23,"FJ2428","FMA13884","ac3c7d6714bed8cf549c97b013546541cf67e189dad2107092a59758aa3ccc45"},
            {24,"FJ2438","FMA9531","187235f3c3612ef27abde924d01c71579c2946435e64319164f43e5ad008284d"},
        }};
        id rawVentricularWallBinding=cardiacBinding[@"ventricular_wall_binding"];
        const bool hasVentricularWallBinding=rawVentricularWallBinding!=nil;
        NSDictionary* sourceMap=[provenance isKindOfClass:NSDictionary.class]?provenance[@"source_id_map"]:nil;
        require([sourceMap isKindOfClass:NSDictionary.class],"expanded anatomy receipt lacks source-member identities");
        for(const auto& expected:expectedWalls) {
            NSString* key=[NSString stringWithFormat:@"%u",expected.id];NSDictionary* identity=sourceMap[key];
            NSDictionary* metadata=[identity isKindOfClass:NSDictionary.class]?identity[@"source_owner_metadata"]:nil;
            NSNumber* owner=identity[@"body_index"];NSNumber* layer=identity[@"layer"];
            NSString* member=identity[@"source_member"];NSString* digest=identity[@"source_sha256"];
            NSString* fmaConcept=[metadata isKindOfClass:NSDictionary.class]?metadata[@"concept_id"]:nil;
            const char* conceptText=[fmaConcept isKindOfClass:NSString.class]?fmaConcept.UTF8String:nullptr;
            const bool conceptMatches=conceptText&&numiHumanRestingCardiacConceptAccepted(expected.id,conceptText);
            require([owner isKindOfClass:NSNumber.class]&&owner.unsignedIntValue==gpu.bodyAndFlags.x&&
                [layer isKindOfClass:NSNumber.class]&&layer.unsignedIntValue==1&&
                [member isKindOfClass:NSString.class]&&[member isEqualToString:[NSString stringWithUTF8String:expected.member]]&&
                [digest isKindOfClass:NSString.class]&&[digest isEqualToString:[NSString stringWithUTF8String:expected.digest]]&&
                conceptMatches,
                "passive heart source-member binding differs");
            if(expected.id==23&&hasVentricularWallBinding)continue;
            if(expected.id==23) {
                const auto& legacyWall=surface(expected.id);
                require(legacyWall.vertexCount==21053&&legacyWall.indexCount==112326,
                    "ID 23 has neither the exact legacy organ shell nor a declared ventricular wall binding");
            }
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

        if(hasVentricularWallBinding) {
        NSDictionary* wallBinding=[rawVentricularWallBinding isKindOfClass:NSDictionary.class]?
            (NSDictionary*)rawVentricularWallBinding:nil;
        NSDictionary* wallMap=[wallBinding isKindOfClass:NSDictionary.class]?wallBinding[@"wall_map"]:nil;
        NSDictionary* scales=[wallMap isKindOfClass:NSDictionary.class]?wallMap[@"normalization_scales"]:nil;
        NSArray* radialOrigin=wallBinding[@"radial_closure_origin_m"];
        NSArray* closureBounds=wallBinding[@"closure_bracket_m"];
        NSNumber* closureTolerance=wallBinding[@"closure_relative_tolerance"];
        NSNumber* wallStableID=wallBinding[@"stable_id"];
        NSNumber* referenceVolume=wallBinding[@"reference_material_volume_m3"];
        NSNumber* candidateVertexCount=wallBinding[@"arrangement_candidate_vertex_count"];
        NSNumber* candidateTriangleCount=wallBinding[@"arrangement_candidate_triangle_count"];
        NSArray* faceOwnerRanges=wallBinding[@"face_owner_ranges"];
        NSString* wallMethod=wallBinding[@"method"];
        NSString* wallSource=wallBinding[@"source_member"];
        NSString* wallFMA=wallBinding[@"fma_id"];
        NSString* wallSourceSHA=wallBinding[@"source_sha256"];
        NSString* wallOutputSHA=wallBinding[@"output_anatomy_payload_sha256"];
        NSString* wallMapMethod=[wallMap isKindOfClass:NSDictionary.class]?wallMap[@"method"]:nil;
        require([wallBinding isKindOfClass:NSDictionary.class]&&
            [wallMethod isKindOfClass:NSString.class]&&
            [wallMethod isEqualToString:@"exact_ID23_material_boundary_with_RV_LV_lumens_and_continuous_barycentric_wall_map_v1"]&&
            [wallStableID isKindOfClass:NSNumber.class]&&wallStableID.unsignedIntValue==23&&
            [wallSource isKindOfClass:NSString.class]&&[wallSource isEqualToString:@"FJ2428"]&&
            [wallFMA isKindOfClass:NSString.class]&&[wallFMA isEqualToString:@"FMA13884"]&&
            [wallSourceSHA isKindOfClass:NSString.class]&&[wallSourceSHA isEqualToString:@"ac3c7d6714bed8cf549c97b013546541cf67e189dad2107092a59758aa3ccc45"]&&
            [wallOutputSHA isKindOfClass:NSString.class]&&std::string(wallOutputSHA.UTF8String)==outputAnatomyHash&&
            [wallMapMethod isKindOfClass:NSString.class]&&
            [wallMapMethod isEqualToString:@"candidate_face_owner_ranges_with_pure_cavity_channel_boundary_and_natural_inverse_squared_distance_freewall_blend"]&&
            [scales isKindOfClass:NSDictionary.class]&&
            [scales[@"q_rv"] isKindOfClass:NSNumber.class]&&std::abs([scales[@"q_rv"] doubleValue]-.4)<1e-9&&
            [scales[@"q_lv"] isKindOfClass:NSNumber.class]&&std::abs([scales[@"q_lv"] doubleValue]-.2)<1e-9&&
            [scales[@"closure_m"] isKindOfClass:NSNumber.class]&&std::abs([scales[@"closure_m"] doubleValue]-.01)<1e-9&&
            [radialOrigin isKindOfClass:NSArray.class]&&radialOrigin.count==3&&
            [closureBounds isKindOfClass:NSArray.class]&&closureBounds.count==2&&
            [closureBounds[0] isKindOfClass:NSNumber.class]&&[closureBounds[1] isKindOfClass:NSNumber.class]&&
            [closureTolerance isKindOfClass:NSNumber.class]&&closureTolerance.doubleValue>0&&closureTolerance.doubleValue<1e-3&&
            [referenceVolume isKindOfClass:NSNumber.class]&&std::isfinite(referenceVolume.doubleValue)&&referenceVolume.doubleValue>0&&
            [candidateVertexCount isKindOfClass:NSNumber.class]&&candidateVertexCount.unsignedIntValue>0&&
            [candidateTriangleCount isKindOfClass:NSNumber.class]&&candidateTriangleCount.unsignedIntValue>0&&
            [faceOwnerRanges isKindOfClass:NSArray.class]&&faceOwnerRanges.count==3,
            "ventricular wall binding is missing the source-bound map, volume reference, or closure controls");
        const std::array<unsigned,3> expectedOwnerIDs{{23,319,321}};
        const std::array<const char*,3> expectedOwnerNames{{"ID23_wall","ID319_RV_lumen","ID321_LV_lumen"}};
        std::array<unsigned,3> ownerRangeFirst{},ownerRangeCount{};
        unsigned coveredTriangles=0;
        for(unsigned i=0;i<expectedOwnerIDs.size();++i) {
            NSDictionary* range=faceOwnerRanges[i];
            NSNumber* first=[range isKindOfClass:NSDictionary.class]?range[@"first_triangle"]:nil;
            NSNumber* count=[range isKindOfClass:NSDictionary.class]?range[@"triangle_count"]:nil;
            NSNumber* ownerID=[range isKindOfClass:NSDictionary.class]?range[@"stable_id"]:nil;
            NSString* ownerName=[range isKindOfClass:NSDictionary.class]?range[@"owner"]:nil;
            require([first isKindOfClass:NSNumber.class]&&[count isKindOfClass:NSNumber.class]&&
                [ownerID isKindOfClass:NSNumber.class]&&ownerID.unsignedIntValue==expectedOwnerIDs[i]&&
                [ownerName isKindOfClass:NSString.class]&&
                [ownerName isEqualToString:[NSString stringWithUTF8String:expectedOwnerNames[i]] ]&&
                first.unsignedIntValue==coveredTriangles&&count.unsignedIntValue>0,
                "ventricular face-owner ranges are not contiguous in the declared wall/RV/LV order");
            ownerRangeFirst[i]=first.unsignedIntValue;ownerRangeCount[i]=count.unsignedIntValue;
            coveredTriangles+=ownerRangeCount[i];
        }
        const auto& ventricularWall=surface(23);
        require(ventricularWall.layer==1&&ventricularWall.vertexCount==candidateVertexCount.unsignedIntValue&&
            ventricularWall.indexCount==candidateTriangleCount.unsignedIntValue*3&&
            coveredTriangles==candidateTriangleCount.unsignedIntValue,
            "stable ID 23 does not match the source-bound complete ventricular material boundary");
        ventricularWallStableId=23;
        const auto material=closedSurface(anatomy,ventricularWall);
        const double referenceMaterialVolume=referenceVolume.doubleValue;
        require(std::abs(material.volume-referenceMaterialVolume)<5e-10,
            "ventricular material shell volume differs from its source-bound candidate receipt");
        std::array<double,3> radialCenter{};
        for(unsigned k=0;k<3;++k) {
            require([radialOrigin[k] isKindOfClass:NSNumber.class],"ventricular closure origin is malformed");
            radialCenter[k]=[radialOrigin[k] doubleValue];
            require(std::isfinite(radialCenter[k]),"ventricular closure origin is nonfinite");
        }
        const double closureLow=[closureBounds[0] doubleValue],closureHigh=[closureBounds[1] doubleValue];
        require(std::isfinite(closureLow)&&std::isfinite(closureHigh)&&closureLow<closureHigh&&
            closureLow>=-.02&&closureHigh<=.02,
            "ventricular closure bracket is outside the admitted small-displacement range");
        const double closureRelativeTolerance=closureTolerance.doubleValue;
        const auto& rvIndex=*cavityIndex[1];const auto& lvIndex=*cavityIndex[3];
        const auto& rvWeights=cardiacFreewallWeights.at(cavities[1]);
        const auto& lvWeights=cardiacFreewallWeights.at(cavities[3]);
        const auto displacementAt=[&](unsigned channel,const TriangleIndex::Nearest& closest) {
            const auto& index=channel==1?rvIndex:lvIndex;
            const auto& weights=channel==1?rvWeights:lvWeights;
            const auto& center=gpu.chamberCenterAndVolume[channel];
            std::array<double,3> value{};
            for(unsigned k=0;k<3;++k) {
                const unsigned sourceVertex=closest.vertex[k];
                const auto& point=index.points.at(sourceVertex);
                const double weight=closest.barycentric[k]*weights.at(sourceVertex);
                for(unsigned axis=0;axis<3;++axis) {
                    const double c=axis==0?center.x:axis==1?center.y:center.z;
                    value[axis]+=weight*(point[axis]-c);
                }
            }
            return value;
        };
        const auto& rvVertices=rvIndex.points;const auto& lvVertices=lvIndex.points;
        std::vector<int> ownedChannel(ventricularWall.vertexCount,-1);
        for(unsigned rangeIndex=1;rangeIndex<expectedOwnerIDs.size();++rangeIndex) {
            const int channel=expectedOwnerIDs[rangeIndex]==319?1:3;
            for(unsigned triangle=ownerRangeFirst[rangeIndex];triangle<ownerRangeFirst[rangeIndex]+ownerRangeCount[rangeIndex];++triangle) {
                const unsigned faceIndex=ventricularWall.firstIndex+triangle*3;
                for(unsigned corner=0;corner<3;++corner) {
                    const unsigned global=anatomy.indices.at(faceIndex+corner);
                    require(global>=ventricularWall.firstVertex&&global<ventricularWall.firstVertex+ventricularWall.vertexCount,
                        "ventricular face-owner range points outside its registered material surface");
                    const unsigned local=global-ventricularWall.firstVertex;
                    require(ownedChannel[local]<0||ownedChannel[local]==channel,
                        "one material-boundary vertex is owned by both RV and LV cavity faces");
                    ownedChannel[local]=channel;
                }
            }
        }
        ventricularWallMap.resize(ventricularWall.vertexCount);
        for(unsigned v=0;v<ventricularWall.vertexCount;++v) {
            const auto& source=anatomy.vertices.at(ventricularWall.firstVertex+v);
            const std::array<double,3> point{source.positionX,source.positionY,source.positionZ};
            const auto rv=rvIndex.nearest(point),lv=lvIndex.nearest(point);
            const double denominator=rv.distanceSquared+lv.distanceSquared;
            require(std::isfinite(denominator)&&denominator>1e-18,
                "ventricular wall vertex is simultaneously unowned by the RV/LV blend");
            const double rvBlend=lv.distanceSquared/denominator,lvBlend=rv.distanceSquared/denominator;
            const auto dRV=displacementAt(1,rv),dLV=displacementAt(3,lv);
            double closureWeight=0;
            if(ownedChannel[v]<0) {
                double lumenVertexDistanceSquared=std::numeric_limits<double>::infinity();
                for(const auto& q:rvVertices) {
                    const double dx=point[0]-q[0],dy=point[1]-q[1],dz=point[2]-q[2];
                    lumenVertexDistanceSquared=std::min(lumenVertexDistanceSquared,dx*dx+dy*dy+dz*dz);
                }
                for(const auto& q:lvVertices) {
                    const double dx=point[0]-q[0],dy=point[1]-q[1],dz=point[2]-q[2];
                    lumenVertexDistanceSquared=std::min(lumenVertexDistanceSquared,dx*dx+dy*dy+dz*dz);
                }
                const double clearance=std::sqrt(lumenVertexDistanceSquared);
                closureWeight=smoothstep(.004,.015,clearance);
            }
            std::array<double,3> radial{point[0]-radialCenter[0],point[1]-radialCenter[1],point[2]-radialCenter[2]};
            const double radialLength=std::sqrt(TriangleIndex::dot(radial,radial));
            require(std::isfinite(radialLength)&&radialLength>1e-8,
                "ventricular material vertex coincides with the inferred closure origin");
            radial=TriangleIndex::scale(radial,closureWeight/radialLength);
            auto& map=ventricularWallMap[v];
            const double ownedRV=ownedChannel[v]==1?1.0:0.0,ownedLV=ownedChannel[v]==3?1.0:0.0;
            map.first={float(dRV[0]*(ownedChannel[v]<0?rvBlend:ownedRV)),
                float(dRV[1]*(ownedChannel[v]<0?rvBlend:ownedRV)),float(dRV[2]*(ownedChannel[v]<0?rvBlend:ownedRV)),0};
            map.second={float(dLV[0]*(ownedChannel[v]<0?lvBlend:ownedLV)),
                float(dLV[1]*(ownedChannel[v]<0?lvBlend:ownedLV)),float(dLV[2]*(ownedChannel[v]<0?lvBlend:ownedLV)),0};
            map.closure={float(radial[0]),float(radial[1]),float(radial[2]),0};
            require(std::isfinite(map.first.x)&&std::isfinite(map.first.y)&&std::isfinite(map.first.z)&&
                std::isfinite(map.second.x)&&std::isfinite(map.second.y)&&std::isfinite(map.second.z)&&
                std::isfinite(map.closure.x)&&std::isfinite(map.closure.y)&&std::isfinite(map.closure.z),
                "ventricular wall map contains nonfinite coefficients");
        }

        // Optional source-bound local coefficient conditioning. This is a
        // load-time replacement of sparse free-wall RV/LV coefficients in the
        // existing map; it adds no state, force, or per-step CPU work.
        NSDictionary* mapRefinement=[wallMap isKindOfClass:NSDictionary.class]?
            (NSDictionary*)wallMap[@"local_coefficient_refinement"]:nil;
        if(mapRefinement) {
            static_assert(sizeof(MRHumanRestingCardiacWallVertexGPU)==48,
                "cardiac wall coefficient identity expects the existing 48-byte map ABI");
            NSString* refinementSchema=mapRefinement[@"schema"];
            NSString* refinementMethod=mapRefinement[@"method"];
            NSString* refinementInterpretation=mapRefinement[@"interpretation"];
            NSString* refinementSourceSHA=mapRefinement[@"source_payload_sha256"];
            NSString* cavitySourceSHA=mapRefinement[@"cavity_source_payload_sha256"];
            NSString* cavityReceiptSHA=mapRefinement[@"cavity_source_receipt_sha256"];
            NSString* arrangementCandidateSHA=mapRefinement[@"arrangement_candidate_sha256"];
            NSString* baseMapSHA=mapRefinement[@"base_map_binary_sha256"];
            NSString* correctedMapSHA=mapRefinement[@"corrected_map_binary_sha256"];
            NSString* correctionSHA=mapRefinement[@"correction_records_sha256"];
            NSDictionary* qDomains=mapRefinement[@"q_domain"];
            NSArray* corrections=mapRefinement[@"corrections"];
            NSArray* checkedPhases=mapRefinement[@"checked_phases"];
            NSNumber* refinementVertexCount=mapRefinement[@"vertex_count"];
            NSNumber* refinementCount=mapRefinement[@"correction_vertex_count"];
            NSNumber* refinementMaxDisplacement=mapRefinement[@"max_displacement_bound_m"];
            const auto validSHA=[](NSString* value) {
                if(![value isKindOfClass:NSString.class]||value.length!=64)return false;
                for(unsigned i=0;i<64;++i) {
                    const unichar c=[value characterAtIndex:i];
                    if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
                }
                return true;
            };
            require([refinementSchema isKindOfClass:NSString.class]&&
                [refinementSchema isEqualToString:@"numi.human.cardiac.ventricular_wall_map_refinement.v1"]&&
                [refinementMethod isKindOfClass:NSString.class]&&
                [refinementMethod isEqualToString:@"joint_selected_state_affine_harmonic_map_conditioning_v1"]&&
                [refinementInterpretation isKindOfClass:NSString.class]&&
                [refinementInterpretation isEqualToString:@"inferred_reference_registration_not_measured_subject_geometry"]&&
                validSHA(refinementSourceSHA)&&std::string(refinementSourceSHA.UTF8String)==outputAnatomyHash&&
                validSHA(cavitySourceSHA)&&validSHA(cavityReceiptSHA)&&validSHA(arrangementCandidateSHA)&&
                [mapRefinement[@"candidate_arrays_match_current_ID23"] boolValue]&&
                validSHA(baseMapSHA)&&validSHA(correctedMapSHA)&&validSHA(correctionSHA)&&
                [refinementVertexCount isKindOfClass:NSNumber.class]&&
                refinementVertexCount.unsignedIntValue==ventricularWall.vertexCount&&
                [refinementCount isKindOfClass:NSNumber.class]&&[corrections isKindOfClass:NSArray.class]&&
                corrections.count==refinementCount.unsignedIntValue&&corrections.count>0&&
                [refinementMaxDisplacement isKindOfClass:NSNumber.class]&&
                std::isfinite(refinementMaxDisplacement.doubleValue)&&refinementMaxDisplacement.doubleValue<=.01&&
                [qDomains isKindOfClass:NSDictionary.class]&&
                [qDomains[@"q_rv"] isKindOfClass:NSArray.class]&&[qDomains[@"q_rv"] count]==2&&
                [qDomains[@"q_lv"] isKindOfClass:NSArray.class]&&[qDomains[@"q_lv"] count]==2&&
                [checkedPhases isKindOfClass:NSArray.class]&&checkedPhases.count>=8,
                "ventricular wall coefficient refinement metadata is malformed or not source-bound");
            for(NSString* name in @[@"q_rv",@"q_lv"]) {
                NSArray* interval=qDomains[name];
                NSNumber* low=interval[0];NSNumber* high=interval[1];
                require([low isKindOfClass:NSNumber.class]&&[high isKindOfClass:NSNumber.class]&&
                    std::isfinite(low.doubleValue)&&std::isfinite(high.doubleValue)&&low.doubleValue<high.doubleValue&&
                    low.doubleValue>=-.2&&high.doubleValue<=.2,
                    "ventricular wall refinement q-domain is outside the supported bounded range");
            }
            for(id phase in checkedPhases) {
                require([phase isKindOfClass:NSDictionary.class]&&
                    [phase[@"step"] isKindOfClass:NSNumber.class]&&
                    [phase[@"exact_self_intersection_pairs"] isKindOfClass:NSNumber.class]&&
                    [phase[@"exact_self_intersection_pairs"] unsignedLongLongValue]==0,
                    "ventricular wall refinement includes a phase without a zero-pair exact self-audit");
            }
            const auto baseMapSHAActual=loadedKneeSHA256Hex(loadedKneeSHA256(
                ventricularWallMap.data(),ventricularWallMap.size()*sizeof(MRHumanRestingCardiacWallVertexGPU)));
            require(baseMapSHAActual==baseMapSHA.UTF8String,
                "ventricular wall refinement base bytes differ from the exact native loader map");
            NSMutableData* canonicalCorrections=[NSMutableData data];
            const auto appendLE32=[&](uint32_t value) {
                const unsigned char bytes[4]={static_cast<unsigned char>(value),static_cast<unsigned char>(value>>8),
                    static_cast<unsigned char>(value>>16),static_cast<unsigned char>(value>>24)};
                [canonicalCorrections appendBytes:bytes length:sizeof(bytes)];
            };
            const auto appendLEFloat=[&](float value) {
                uint32_t bits=0;std::memcpy(&bits,&value,sizeof(bits));appendLE32(bits);
            };
            int64_t previous=-1;double maxRVDelta=0,maxLVDelta=0;
            for(id rawRow in corrections) {
                require([rawRow isKindOfClass:NSDictionary.class],"ventricular wall refinement row is not a dictionary");
                NSDictionary* row=rawRow;
                NSNumber* vertexNumber=row[@"vertex"];
                NSArray* rvValues=row[@"rv_coefficient_m"];NSArray* lvValues=row[@"lv_coefficient_m"];
                NSArray* rvDeltas=row[@"rv_delta_m"];NSArray* lvDeltas=row[@"lv_delta_m"];
                require([vertexNumber isKindOfClass:NSNumber.class]&&
                    [rvValues isKindOfClass:NSArray.class]&&rvValues.count==3&&
                    [lvValues isKindOfClass:NSArray.class]&&lvValues.count==3&&
                    [rvDeltas isKindOfClass:NSArray.class]&&rvDeltas.count==3&&
                    [lvDeltas isKindOfClass:NSArray.class]&&lvDeltas.count==3,
                    "ventricular wall refinement row has malformed coefficient vectors");
                const uint64_t vertex64=vertexNumber.unsignedLongLongValue;
                require(vertex64<ventricularWallMap.size()&&static_cast<int64_t>(vertex64)>previous&&
                    ownedChannel[vertex64]<0,
                    "ventricular wall refinement is unsorted, out of range, or changes a lumen-owned vertex");
                const unsigned vertex=static_cast<unsigned>(vertex64);previous=static_cast<int64_t>(vertex64);
                auto& map=ventricularWallMap[vertex];
                std::array<float,3> rv{},lv{},rvDelta{},lvDelta{};
                for(unsigned axis=0;axis<3;++axis) {
                    for(NSArray* vector in @[rvValues,lvValues,rvDeltas,lvDeltas]) {
                        NSNumber* component=vector[axis];
                        require([component isKindOfClass:NSNumber.class]&&std::isfinite(component.doubleValue),
                            "ventricular wall refinement includes a nonfinite coefficient");
                    }
                    rv[axis]=[rvValues[axis] floatValue];lv[axis]=[lvValues[axis] floatValue];
                    rvDelta[axis]=[rvDeltas[axis] floatValue];lvDelta[axis]=[lvDeltas[axis] floatValue];
                }
                require(std::isfinite(rv[0])&&std::isfinite(rv[1])&&std::isfinite(rv[2])&&
                    std::isfinite(lv[0])&&std::isfinite(lv[1])&&std::isfinite(lv[2])&&
                    std::isfinite(rvDelta[0])&&std::isfinite(rvDelta[1])&&std::isfinite(rvDelta[2])&&
                    std::isfinite(lvDelta[0])&&std::isfinite(lvDelta[1])&&std::isfinite(lvDelta[2]),
                    "ventricular wall refinement overflows binary32 coefficients");
                const double rvNorm=std::sqrt(double(rv[0])*rv[0]+double(rv[1])*rv[1]+double(rv[2])*rv[2]);
                const double lvNorm=std::sqrt(double(lv[0])*lv[0]+double(lv[1])*lv[1]+double(lv[2])*lv[2]);
                const double rvDeltaNorm=std::sqrt(double(rvDelta[0])*rvDelta[0]+double(rvDelta[1])*rvDelta[1]+double(rvDelta[2])*rvDelta[2]);
                const double lvDeltaNorm=std::sqrt(double(lvDelta[0])*lvDelta[0]+double(lvDelta[1])*lvDelta[1]+double(lvDelta[2])*lvDelta[2]);
                require(rvNorm<.08&&lvNorm<.08&&rvDeltaNorm<.04&&lvDeltaNorm<.04,
                    "ventricular wall refinement coefficients exceed their bounded map ranges");
                maxRVDelta=std::max(maxRVDelta,rvDeltaNorm);maxLVDelta=std::max(maxLVDelta,lvDeltaNorm);
                for(unsigned axis=0;axis<3;++axis) {
                    const float baseRV=axis==0?map.first.x:axis==1?map.first.y:map.first.z;
                    const float baseLV=axis==0?map.second.x:axis==1?map.second.y:map.second.z;
                    require(rv[axis]-baseRV==rvDelta[axis]&&lv[axis]-baseLV==lvDelta[axis],
                        "ventricular wall refinement delta does not match the loader's binary32 base coefficient");
                    if(axis==0){map.first.x=rv[axis];map.second.x=lv[axis];}
                    else if(axis==1){map.first.y=rv[axis];map.second.y=lv[axis];}
                    else {map.first.z=rv[axis];map.second.z=lv[axis];}
                }
                appendLE32(static_cast<uint32_t>(vertex64));
                for(unsigned axis=0;axis<3;++axis)appendLEFloat(rv[axis]);
                for(unsigned axis=0;axis<3;++axis)appendLEFloat(lv[axis]);
            }
            require(loadedKneeSHA256Hex(loadedKneeSHA256(canonicalCorrections.bytes,canonicalCorrections.length))==
                    correctionSHA.UTF8String,
                "ventricular wall refinement rows differ from their canonical source-bound digest");
            NSArray* qrv=qDomains[@"q_rv"];NSArray* qlv=qDomains[@"q_lv"];
            const double qrvMax=std::max(std::abs([qrv[0] doubleValue]),std::abs([qrv[1] doubleValue]));
            const double qlvMax=std::max(std::abs([qlv[0] doubleValue]),std::abs([qlv[1] doubleValue]));
            NSNumber* closureShift=mapRefinement[@"max_closure_shift_m"];
            require([closureShift isKindOfClass:NSNumber.class]&&std::isfinite(closureShift.doubleValue)&&
                closureShift.doubleValue>=0&&closureShift.doubleValue<=.02&&
                maxRVDelta*qrvMax+maxLVDelta*qlvMax+closureShift.doubleValue+1e-6<=
                    refinementMaxDisplacement.doubleValue+1e-9,
                "ventricular wall refinement displacement bound understates the accepted q-domain map change");
            const auto correctedMapSHAActual=loadedKneeSHA256Hex(loadedKneeSHA256(
                ventricularWallMap.data(),ventricularWallMap.size()*sizeof(MRHumanRestingCardiacWallVertexGPU)));
            require(correctedMapSHAActual==correctedMapSHA.UTF8String,
                "ventricular wall corrected coefficient bytes differ from the exact retained candidate map");
        }

        // The source material shell is one closed record with its RV and LV
        // endocardial boundary faces. Expand its signed volume directly from
        // all 64 determinant products per triangle; no sampled fit or frame-
        // dependent CPU correction enters the accepted simulation path.
        const std::array<std::array<unsigned,3>,20> exponents{{
            {{0,0,0}},{{1,0,0}},{{0,1,0}},
            {{2,0,0}},{{1,1,0}},{{0,2,0}},
            {{3,0,0}},{{2,1,0}},{{1,2,0}},{{0,3,0}},
            {{0,0,1}},{{1,0,1}},{{0,1,1}},{{2,0,1}},{{1,1,1}},{{0,2,1}},
            {{0,0,2}},{{1,0,2}},{{0,1,2}},{{0,0,3}}
        }};
        const auto exponentForBasis=[](unsigned basis) {
            std::array<unsigned,3> e{};if(basis==1)e[0]=1;else if(basis==2)e[1]=1;else if(basis==3)e[2]=1;return e;
        };
        const auto termIndex=[&](const std::array<unsigned,3>& exponent) {
            for(unsigned i=0;i<exponents.size();++i)if(exponents[i]==exponent)return i;
            require(false,"ventricular material-volume term exceeds cubic order");return 0u;
        };
        const auto determinant=[](const std::array<double,3>& a,const std::array<double,3>& b,const std::array<double,3>& c) {
            return a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]);
        };
        std::array<double,20> volumePolynomial{};
        for(unsigned index=ventricularWall.firstIndex;index<ventricularWall.firstIndex+ventricularWall.indexCount;index+=3) {
            std::array<std::array<std::array<double,3>,4>,3> basis{};
            for(unsigned corner=0;corner<3;++corner) {
                const unsigned global=anatomy.indices.at(index+corner),local=global-ventricularWall.firstVertex;
                const auto& point=anatomy.vertices.at(global);const auto& map=ventricularWallMap.at(local);
                basis[corner][0]={double(point.positionX)-radialCenter[0],double(point.positionY)-radialCenter[1],double(point.positionZ)-radialCenter[2]};
                basis[corner][1]={double(map.first.x)*.4,double(map.first.y)*.4,double(map.first.z)*.4};
                basis[corner][2]={double(map.second.x)*.2,double(map.second.y)*.2,double(map.second.z)*.2};
                basis[corner][3]={double(map.closure.x)*.01,double(map.closure.y)*.01,double(map.closure.z)*.01};
            }
            for(unsigned a=0;a<4;++a)for(unsigned b=0;b<4;++b)for(unsigned c=0;c<4;++c) {
                auto exponent=exponentForBasis(a),eb=exponentForBasis(b),ec=exponentForBasis(c);
                for(unsigned axis=0;axis<3;++axis)exponent[axis]+=eb[axis]+ec[axis];
                volumePolynomial[termIndex(exponent)]+=determinant(basis[0][a],basis[1][b],basis[2][c])/6.0;
            }
        }
        require(std::isfinite(volumePolynomial[0])&&std::abs(volumePolynomial[0]-material.volume)<5e-10,
            "ventricular material-volume polynomial constant differs from its exact source shell");
        for(unsigned i=0;i<volumePolynomial.size();++i) {
            mr_float4& row=ventricularWallGPU.volumePolynomial[i/4];const float value=float(volumePolynomial[i]);
            if(i%4==0)row.x=value;else if(i%4==1)row.y=value;else if(i%4==2)row.z=value;else row.w=value;
        }
        ventricularWallGPU.scalesAndVolume={.4f,.2f,.01f,float(material.volume)};
        ventricularWallGPU.closureBoundsAndTolerance={float(closureLow),float(closureHigh),
            float(closureRelativeTolerance),0};
        ventricularWallGPU.chambersAndFlags={1,3,1,ventricularWall.vertexCount};
        }
    }
};
