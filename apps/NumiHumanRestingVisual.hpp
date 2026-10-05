#pragma once
#include "NumiHumanRestingAnatomy.hpp"
#include "NumiHumanRestingSupportGeometry.hpp"
#include <charconv>
#include <filesystem>
#include <set>
#include <string_view>
// Included after the existing Human source loaders and visual-pack compiler.
// It adds a persistent presentation consumer, not another dynamics owner.
class NumiHumanRestingVisual {
    NumiHumanRestingCoupling& coupled;
    std::unique_ptr<NumiHumanRestingSupportGeometry> skinSupport;
    std::unique_ptr<metalrobo::MetalHybridRenderer> renderer;
    metalrobo::MetalWorldFamilyContext worlds;
    id<MTLBuffer> mapping, influences, anatomyParameters, surfaceAudits, volumeResults, instanceLayers, cardiacQ;
    id<MTLComputePipelineState> skinPipeline, layerPipeline, volumePipeline, skinAuditPipeline, cardiacQPipeline;
    id<MTLComputePipelineState> vertexCapturePipeline=nil;
    id<MTLBuffer> vertexCaptureBuffer=nil;
    id<MTLCommandQueue> queue;
    NumiHumanRestingWindow* window=nil;
    unsigned dimension,layer=0;
    bool complete=false;
    unsigned auditCount=0;
    double wallOrigin=0;
    std::filesystem::path initialPackPath, acceptedGeometryDirectory;
    std::string initialPackContentHash, initialPackFileSHA256;
    std::set<unsigned> requestedGeometrySteps, completedGeometrySteps;
    bool captureThisFrame=false, captureKernelEncoded=false;
    unsigned captureStep=0,captureCamera=0,captureLayer=0;
    std::uint64_t captureRootFingerprint=0,captureTransactionFingerprint=0,captureTimestampMicroseconds=0;
    std::ofstream surfaceTrace;
    static mr_float4 inverseRotation(mr_float4 q){return {-q.x,-q.y,-q.z,q.w};}
    static std::set<unsigned> geometryExportStepsFromEnvironment() {
        const char* requested=std::getenv("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS");
        if(!requested||!*requested)return {};
        std::set<unsigned> steps;std::string_view remaining(requested);
        while(!remaining.empty()) {
            const auto separator=remaining.find(',');const auto token=remaining.substr(0,separator);
            require(!token.empty(),"accepted geometry step selection contains an empty value");
            unsigned step=0;const auto parsed=std::from_chars(token.data(),token.data()+token.size(),step);
            require(parsed.ec==std::errc{}&&parsed.ptr==token.data()+token.size(),
                "accepted geometry steps must be comma-separated unsigned physical step IDs");
            require(steps.insert(step).second,"accepted geometry step selection contains a duplicate");
            require(steps.size()<=8,"accepted geometry export is limited to eight explicitly selected frames");
            if(separator==std::string_view::npos)break;
            require(separator+1<remaining.size(),"accepted geometry step selection ends with an empty value");
            remaining.remove_prefix(separator+1);
        }
        return steps;
    }
    void exportAcceptedGeometry(unsigned step,double time,std::uint64_t root,std::uint64_t transaction,
                                std::uint64_t timestamp) {
        require(vertexCaptureBuffer&&vertexCaptureBuffer.contents,"accepted geometry staging buffer is unavailable");
        metalrobo::VisualAssetPackV2 pack;std::string error;
        require(metalrobo::readVisualAssetPack(initialPackPath,pack,&error),error);
        require(pack.contentHash==initialPackContentHash&&pack.vertices.size()==renderer->layout().meshVertexCount,
            "accepted geometry base pack differs from the compiled renderer topology");
        const std::size_t vertexBytes=pack.vertices.size()*sizeof(MRVisualVertexGPUV2);
        require(vertexCaptureBuffer.length>=vertexBytes,"accepted geometry staging buffer is undersized");
        std::memcpy(pack.vertices.data(),vertexCaptureBuffer.contents,vertexBytes);
        for(const auto& vertex:pack.vertices)require(
            std::isfinite(vertex.position.x)&&std::isfinite(vertex.position.y)&&std::isfinite(vertex.position.z)&&
            std::isfinite(vertex.normalAndTangentSign.x)&&std::isfinite(vertex.normalAndTangentSign.y)&&
            std::isfinite(vertex.normalAndTangentSign.z)&&std::isfinite(vertex.tangent.x)&&
            std::isfinite(vertex.tangent.y)&&std::isfinite(vertex.tangent.z),
            "accepted geometry copy contains non-finite rendered vertices");
        for(auto& primitive:pack.primitives) {
            require(primitive.geometry.x<=pack.indices.size()&&
                primitive.geometry.y<=pack.indices.size()-primitive.geometry.x,
                "accepted geometry primitive index span is invalid");
            mr_float4 lo{INFINITY,INFINITY,INFINITY,1},hi{-INFINITY,-INFINITY,-INFINITY,1};
            for(unsigned j=primitive.geometry.x;j<primitive.geometry.x+primitive.geometry.y;++j) {
                const unsigned index=pack.indices[j];require(index<pack.vertices.size(),
                    "accepted geometry pack has an out-of-range vertex index");
                const auto p=pack.vertices[index].position;
                lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);
                hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z);
            }
            primitive.boundsMinimum=lo;primitive.boundsMaximum=hi;
        }
        std::ostringstream rootHexStream;rootHexStream<<"0x"<<std::hex<<std::setw(16)<<std::setfill('0')<<root;
        const auto rootHex=rootHexStream.str();
        pack.preprocessingProvenance+="/accepted_native_render_step_"+std::to_string(step)+"_root_"+rootHex;
        pack.contentHash=metalrobo::computeVisualAssetPackContentHash(pack);
        std::filesystem::create_directories(acceptedGeometryDirectory);
        const auto packPath=acceptedGeometryDirectory/("step-"+std::to_string(step)+".mrvpack");
        const auto receiptPath=acceptedGeometryDirectory/("step-"+std::to_string(step)+".receipt.json");
        require(!std::filesystem::exists(packPath)&&!std::filesystem::exists(receiptPath),
            "refusing to overwrite accepted geometry evidence");
        require(metalrobo::writeVisualAssetPack(pack,packPath,&error),error);
        const auto packSHA=loadedKneeSHA256Hex(loadedKneeFileSHA256(packPath));
        const auto vertexSHA=loadedKneeSHA256Hex(loadedKneeSHA256(vertexCaptureBuffer.contents,vertexBytes));
        const auto bodySHA=loadedKneeSHA256Hex(loadedKneeSHA256(coupled.presentationBodies.contents,
            coupled.presentationBodies.length));
        const auto respirationSHA=loadedKneeSHA256Hex(loadedKneeSHA256(coupled.presentationRespiration.contents,
            coupled.presentationRespiration.length));
        NSError* jsonError=nil;
        NSDictionary* receipt=@{
            @"schema":@"numi.human.accepted-render-geometry.v1",
            @"accepted_step":@(step),@"accepted_time_s":@(time),
            @"accepted_root_fingerprint":@(root),@"accepted_root_fingerprint_hex":loadedKneeNSString(rootHex),
            @"accepted_transaction_fingerprint":@(transaction),
            @"accepted_timestamp_microseconds":@(timestamp),
            @"camera_index":@(captureCamera),@"anatomy_layer_index":@(captureLayer),
            @"accepted_body_state_sha256":loadedKneeNSString(bodySHA),
            @"accepted_respiration_state_sha256":loadedKneeNSString(respirationSHA),
            @"captured_vertex_buffer_sha256":loadedKneeNSString(vertexSHA),
            @"base_pack_content_hash":loadedKneeNSString(initialPackContentHash),
            @"base_pack_file_sha256":loadedKneeNSString(initialPackFileSHA256),
            @"pack_content_hash":loadedKneeNSString(pack.contentHash),@"pack_file_sha256":loadedKneeNSString(packSHA),
            @"source_pack_path":loadedKneeNSString(initialPackPath.string()),
            @"accepted_pack_path":loadedKneeNSString(packPath.string()),
            @"vertex_count":@(pack.vertices.size()),@"index_count":@(pack.indices.size()),
            @"primitive_count":@(pack.primitives.size()),@"instance_count":@(pack.instances.size()),
            @"position_normal_tangent_source":@"accepted-state renderer mesh buffer copied on the same Metal command buffer"
        };
        NSData* json=[NSJSONSerialization dataWithJSONObject:receipt
            options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&jsonError];
        require(json!=nil,"accepted geometry receipt JSON serialization failed");
        require([json writeToURL:[NSURL fileURLWithPath:loadedKneeNSString(receiptPath.string())]
            options:NSDataWritingAtomic error:&jsonError],"accepted geometry receipt write failed");
        completedGeometrySteps.insert(step);
        const auto receiptSHA=loadedKneeSHA256Hex(loadedKneeFileSHA256(receiptPath));
        std::cout<<"accepted_geometry_export="<<packPath.string()<<" receipt="<<receiptPath.string()
            <<" accepted_root="<<rootHex
            <<" pack_sha256="<<packSHA<<" receipt_sha256="<<receiptSHA<<"\n";
    }
public:
    // Native presentation exposes accepted states at the initial frame, each
    // completed fixed-size submission (whose state ID is the final zero-based
    // step in that submission), and the exact terminal state. Validate an
    // opt-in capture request before allocating or running the simulation so
    // an unreachable ID cannot cause a late failure after a long native run.
    static void validateAcceptedGeometryExportCadence(unsigned totalAcceptedSteps,
                                                       unsigned submissionSteps) {
        const auto steps=geometryExportStepsFromEnvironment();
        if(steps.empty())return;
        require(totalAcceptedSteps>0&&submissionSteps>0,
            "accepted geometry capture requires a nonempty native horizon and submission cadence");
        for(const unsigned step:steps) {
            require(step<totalAcceptedSteps,
                "accepted geometry step is outside the requested native horizon: "+std::to_string(step));
            const bool initial=step==0;
            const bool submissionEnd=(step+1u)%submissionSteps==0u;
            const bool terminal=step+1u==totalAcceptedSteps;
            require(initial||submissionEnd||terminal,
                "accepted geometry step is not presented by the native submission cadence: "+std::to_string(step));
        }
    }

    NumiHumanRestingVisual(NumiHumanRestingCoupling& owner,metalrobo::VisualAssetPackV2 pack,
        const metalrobo::EngineModel& model,const LoadedSkin& skin,const LoadedSoftTissues* tissues,
        const std::vector<MRBodyStateGPU>& initialBodies,const std::vector<MRBodyStateGPU>& restBodies,
        const LoadedSupportContacts& support,const NumiHumanRestingAnatomy& functional,
        const std::filesystem::path& output,unsigned size,const std::string& movie,bool presentWindow=true):
        coupled(owner),dimension(size),acceptedGeometryDirectory(output/"accepted-geometry"),
        surfaceTrace(output/"resting-surface-audit.csv") {
        require(surfaceTrace.good(),"resting surface audit output unavailable");
        requestedGeometrySteps=geometryExportStepsFromEnvironment();
        require(requestedGeometrySteps.empty()||presentWindow,
            "accepted MRVPack export requires the native viewer path");
        surfaceTrace<<"time_s,step,min_skin_bed_gap_m,vertices_below_1mm,nonfinite_skin_vertices,max_functional_volume_relative_error,q_ra,q_rv,q_la,q_lv,ra_target_ml,rv_target_ml,la_target_ml,lv_target_ml,diaphragm_swept_ml,rib_swept_ml,lung_target_ml\n";
        require(initialBodies.size()*sizeof(MRBodyStateGPU)==coupled.presentationBodies.length,
            "initial native frame does not match the body owner");
        std::memcpy(coupled.presentationBodies.contents,initialBodies.data(),coupled.presentationBodies.length);
        std::memcpy(coupled.presentationRespiration.contents,coupled.physiology.respiration->accepted.contents,
            coupled.presentationRespiration.length);
        auto framing=makeCameraFraming(pack,initialBodies,model.joints,std::nullopt,std::nullopt,std::nullopt,false,nullptr);
        std::vector<MRHumanRestingVertexMap> maps(pack.vertices.size());
        std::vector<MRHumanRestingInfluence> weights;
        unsigned skinFirstVertex=MR_INVALID_INDEX;
        std::vector<mr_float4> skinRestWorld(skin.header.vertexCount);
        std::vector<MRHumanRestingSurfaceAuditGPU> audits;
        std::vector<unsigned> visibleLayers;
        // Keep the four source cavity identities visually distinguishable in
        // the native inspection layer. These are presentation colors only;
        // they do not encode oxygenation, flow, or tissue state.
        const std::array<unsigned,4> cardiacMaterials{{
            unsigned(pack.materials.size()), unsigned(pack.materials.size()+1),
            unsigned(pack.materials.size()+2), unsigned(pack.materials.size()+3)}};
        pack.materials.push_back(makeMaterial({.12f,.34f,.78f,1},{0,0,0,0},.48f)); // right atrium
        pack.materials.push_back(makeMaterial({.18f,.52f,.92f,1},{0,0,0,0},.48f)); // right ventricle
        pack.materials.push_back(makeMaterial({.88f,.22f,.18f,1},{0,0,0,0},.48f)); // left atrium
        pack.materials.push_back(makeMaterial({.72f,.08f,.10f,1},{0,0,0,0},.48f)); // left ventricle
        auto anatomyGPU=functional.gpu;
        float posteriorSkin=INFINITY,anteriorSkin=-INFINITY;
        std::array<mr_float4,24> ribLo,ribHi;
        std::set<unsigned> boundRibs;
        const auto& initialThorax=initialBodies.at(anatomyGPU.bodyAndFlags.x);
        const auto torsoLocal=[&](const MRVisualInstanceGPUV2& instance,mr_float4 point) {
            point=addPoint(instance.translationAndScale,scalePoint(rotatePoint(instance.orientation,point),instance.translationAndScale.w));
            if(instance.binding.z==MR_VISUAL_BINDING_ARTICULATED_LINK) {
                const auto& body=initialBodies.at(instance.binding.y);
                point=addPoint(body.position,rotatePoint(body.orientation,point));
            }
            return rotatePoint(inverseRotation(initialThorax.orientation),subtractPoint(point,initialThorax.position));
        };
        // Each registered rib keeps its own rigid geometry and a hinge near
        // its posterior source attachment. The single mechanical rib-volume
        // coordinate sets anterior excursion through the declared area.
        for(const auto& instance:pack.instances)if(instance.identity.x==kBoneSemantic&&functional.ribs.contains(instance.identity.w)) {
            const unsigned rib=functional.ribs.at(instance.identity.w);require(boundRibs.insert(rib).second,"duplicate functional rib");
            std::set<unsigned> ids;
            for(unsigned p=instance.geometry.x;p<instance.geometry.x+instance.geometry.y;++p) {
                const auto& primitive=pack.primitives.at(p);
                for(unsigned j=primitive.geometry.x;j<primitive.geometry.x+primitive.geometry.y;++j)ids.insert(pack.indices.at(j));
            }
            std::vector<mr_float4> points;float lo=INFINITY,hi=-INFINITY;
            ribLo[rib]={INFINITY,INFINITY,INFINITY,0};ribHi[rib]={-INFINITY,-INFINITY,-INFINITY,0};
            for(unsigned id:ids) {
                const auto p=torsoLocal(instance,pack.vertices[id].position);points.push_back(p);
                const float ap=dotPoint(p,anatomyGPU.anteriorAxis);lo=std::min(lo,ap);hi=std::max(hi,ap);
                ribLo[rib].x=std::min(ribLo[rib].x,p.x);ribLo[rib].y=std::min(ribLo[rib].y,p.y);ribLo[rib].z=std::min(ribLo[rib].z,p.z);
                ribHi[rib].x=std::max(ribHi[rib].x,p.x);ribHi[rib].y=std::max(ribHi[rib].y,p.y);ribHi[rib].z=std::max(ribHi[rib].z,p.z);
            }
            require(hi-lo>.015f,"source rib lacks anterior/posterior extent");
            mr_float4 pivot{},distal{};unsigned posteriorCount=0,anteriorCount=0;
            for(const auto& p:points) {
                const float ap=dotPoint(p,anatomyGPU.anteriorAxis);
                if(ap<lo+.05f*(hi-lo)){pivot=addPoint(pivot,p);++posteriorCount;}
                if(ap>hi-.05f*(hi-lo)){distal=addPoint(distal,p);++anteriorCount;}
            }
            pivot=scalePoint(pivot,1.0f/posteriorCount);distal=scalePoint(distal,1.0f/anteriorCount);
            const auto arm=subtractPoint(distal,pivot),ap=anatomyGPU.anteriorAxis;
            mr_float4 axis{arm.y*ap.z-arm.z*ap.y,arm.z*ap.x-arm.x*ap.z,arm.x*ap.y-arm.y*ap.x,0};
            const float lever=std::sqrt(dotPoint(axis,axis));require(lever>.005f,"source rib hinge has no excursion lever arm");
            anatomyGPU.ribAxis[rib]=scalePoint(axis,1/lever);pivot.w=1/lever;anatomyGPU.ribPivotAndGain[rib]=pivot;
        }
        require(boundRibs.size()==24,"native pack omitted a functional source rib");
        auto add=[&](unsigned vertex,unsigned body,mr_float4 p,mr_float4 n,float w) {
            if(w<=0)return;
            auto& m=maps.at(vertex);if(!m.influenceCount)m.firstInfluence=unsigned(weights.size());
            ++m.influenceCount;p.w=w;weights.push_back({p,n,{body,0,0,0}});
        };
        for(auto& instance:pack.instances) {
            std::vector<unsigned> vertices;
            for(unsigned p=instance.geometry.x;p<instance.geometry.x+instance.geometry.y;++p) {
                const auto& primitive=pack.primitives.at(p);
                for(unsigned j=primitive.geometry.x;j<primitive.geometry.x+primitive.geometry.y;++j)vertices.push_back(pack.indices.at(j));
            }
            std::sort(vertices.begin(),vertices.end());vertices.erase(std::unique(vertices.begin(),vertices.end()),vertices.end());
            require(!vertices.empty(),"empty resting anatomy instance");
            unsigned base=vertices.front();
            if(instance.identity.x==kSkinShellSemantic) {
                require(skinFirstVertex==MR_INVALID_INDEX,"resting support requires one source skin shell");
                skinFirstVertex=base;
                require(base+skin.header.vertexCount<=maps.size(),"source skin vertex range exceeds the native pack");
                // Keep exact coincident seam IDs as well as indexed render
                // vertices: contact ownership covers every source weight row.
                vertices.resize(skin.header.vertexCount);
                std::iota(vertices.begin(),vertices.end(),base);
            }
            unsigned deformation=0,chamber=0;
            if(instance.identity.x>=kOrganSurfaceSemantic) {
                if(functional.lungs.contains(instance.identity.w)||functional.pleura.contains(instance.identity.w))deformation=1;
                for(unsigned c=0;c<4;++c)if(instance.identity.w==functional.cavities[c]){deformation=2;chamber=c;}
                if(functional.cardiacWallSurfaces.contains(instance.identity.w))deformation=8;
                if(functional.diaphragm.contains(instance.identity.w))deformation=4;
                if(functional.intercostals.contains(instance.identity.w))deformation=7;
            }
            if(instance.identity.x==kBoneSemantic&&functional.ribs.contains(instance.identity.w)) {
                deformation=5;chamber=functional.ribs.at(instance.identity.w);
            }
            if(instance.identity.x==kBoneSemantic&&functional.sternum.contains(instance.identity.w))deformation=6;
            const auto semantic=instance.identity.x;
            unsigned visibility=semantic==kSkinShellSemantic?1u:
                semantic==kBoneSemantic?6u:
                (semantic==kMuscleSurfaceSemantic||semantic==kTendonSurfaceSemantic)?2u:
                semantic>=kOrganSurfaceSemantic?8u:0u;
            if(deformation==1&&functional.lungs.contains(instance.identity.w))visibility|=16u;
            if(deformation==2)visibility|=32u|64u;
            if(semantic==kVesselSurfaceSemantic||semantic==kPulmonaryArterySurfaceSemantic||semantic==kPulmonaryVeinSurfaceSemantic)
                visibility|=64u;
            if(deformation==4||deformation==7)visibility=2u|16u;
            visibleLayers.push_back(visibility);
            if(deformation==2)for(unsigned p=instance.geometry.x;p<instance.geometry.x+instance.geometry.y;++p)
                pack.primitives.at(p).geometry.z=cardiacMaterials.at(chamber);
            if(functional.enclosedVolumes.contains(instance.identity.w)&&deformation) {
                require(instance.geometry.y==1,"functional anatomy audit requires one contiguous source surface");
                const auto& p=pack.primitives.at(instance.geometry.x);
                audits.push_back({{p.geometry.x,p.geometry.y,deformation,chamber},
                    {functional.enclosedVolumes.at(instance.identity.w),0,0,0}});
            }
            const SoftTissueRecord* tissue=nullptr;
            if(tissues&&(instance.identity.x==kMuscleSurfaceSemantic||instance.identity.x==kTendonSurfaceSemantic)) {
                auto found=std::find_if(tissues->records.begin(),tissues->records.end(),[&](const auto& t){return t.stableId==instance.identity.w;});
                require(found!=tissues->records.end(),"resting tissue source identity missing");tissue=&*found;
            }
            for(unsigned v:vertices) {
                require(!maps.at(v).influenceCount,"resting anatomy vertices have multiple owners");
                maps[v].deformationKind=deformation;maps[v].chamberIndex=chamber;
                if(deformation==2) {
                    const auto& weights=functional.cardiacFreewallWeights.at(instance.identity.w);
                    require(v>=base&&v-base<weights.size(),"cardiac cavity pack order differs from source vertices");
                    maps[v].deformationWeight.x=weights.at(v-base);
                }else if(deformation==8) {
                    const auto& cardiac=functional.cardiacWallBindings.at(instance.identity.w);
                    require(v>=base&&v-base<cardiac.size(),"passive heart pack order differs from source vertices");
                    maps[v].chamberIndex=cardiac.at(v-base).chamber;
                    maps[v].deformationWeight=cardiac.at(v-base).displacementAndWeight;
                }
                if(instance.identity.x==kSkinShellSemantic) {
                    maps[v].deformationKind=3;
                    const auto& source=skin.vertices.at(v-base);
                    const unsigned count=skin.fullWeights.empty()?4u:unsigned(skin.bindings.size());
                    for(unsigned j=0;j<count;++j) {
                        const float w=skin.fullWeights.empty()?source.weight[j]:skin.fullWeights.at((v-base)*count+j);
                        if(w<=0)continue;
                        const auto& b=skin.bindings.at(skin.fullWeights.empty()?source.bindingIndex[j]:j);
                        if(b.bodyIndex==functional.gpu.bodyAndFlags.x)maps[v].deformationWeight.x+=w;
                        const mr_float4 q={b.quaternionX,b.quaternionY,b.quaternionZ,b.quaternionW};
                        const auto local=addPoint({b.translationX,b.translationY,b.translationZ,0},scalePoint(rotatePoint(q,{source.positionX,source.positionY,source.positionZ,0}),b.uniformScale));
                        const mr_float4 normal={source.normalX,source.normalY,source.normalZ,0};
                        add(v,b.bodyIndex,local,rotatePoint(skin.usesWorldRestNormals?inverseRotation(restBodies.at(b.bodyIndex).orientation):q,normal),w);
                        const auto& rest=restBodies.at(b.bodyIndex);
                        skinRestWorld[v-base]=addPoint(skinRestWorld[v-base],
                            scalePoint(addPoint(rest.position,rotatePoint(rest.orientation,local)),w));
                    }
                }else if(tissue) {
                    const auto& source=tissues->vertices.at(tissue->firstVertex+v-base);
                    for(unsigned j=0;j<4;++j) {
                        const float w=source.weight[j];if(w<=0)continue;
                        const unsigned b=source.bindingIndex[j];require(b<tissue->bindingCount,"resting tissue influence out of bounds");
                        const auto& t=tissue->binding[b];const mr_float4 q={t.quaternion[0],t.quaternion[1],t.quaternion[2],t.quaternion[3]};
                        auto p=addPoint({t.translation[0],t.translation[1],t.translation[2],0},scalePoint(rotatePoint(q,{source.positionX,source.positionY,source.positionZ,0}),t.uniformScale));
                        add(v,tissue->bodyIndex[b],p,rotatePoint(q,{source.normalX,source.normalY,source.normalZ,0}),w);
                    }
                }else {
                    const auto& vertex=pack.vertices.at(v);
                    auto p=addPoint(instance.translationAndScale,scalePoint(rotatePoint(instance.orientation,vertex.position),instance.translationAndScale.w));
                    add(v,instance.binding.z==MR_VISUAL_BINDING_ARTICULATED_LINK?instance.binding.y:MR_INVALID_INDEX,
                        p,rotatePoint(instance.orientation,vertex.normalAndTangentSign),1);
                }
            }
            // GPU skinning supplies world coordinates; identity keeps the exact
            // source semantic, stable ID and anatomical owner for inspection.
            instance.binding.z=MR_VISUAL_BINDING_WORLD;
            instance.translationAndScale={0,0,0,1};instance.orientation={0,0,0,1};
        }
        // The saved existing-format pack is itself a coherent initial pose;
        // GPU updates use these same source influences for later poses.
        for(unsigned v=0;v<maps.size();++v) {
            const auto& m=maps[v];mr_float4 p{},n{};
            if(!m.influenceCount)continue;
            for(unsigned j=0;j<m.influenceCount;++j) {
                const auto& w=weights[m.firstInfluence+j];mr_float4 point=w.positionAndWeight,normal=w.normal;
                if(w.body.x!=MR_INVALID_INDEX) {
                    const auto& b=initialBodies.at(w.body.x);
                    point=addPoint(b.position,rotatePoint(b.orientation,point));normal=rotatePoint(b.orientation,normal);
                }
                p=addPoint(p,scalePoint(point,w.positionAndWeight.w));n=addPoint(n,scalePoint(normal,w.positionAndWeight.w));
            }
            p.w=1;pack.vertices[v].position=p;
            if(maps[v].deformationKind==3) {
                const auto& torso=initialBodies.at(functional.gpu.bodyAndFlags.x);
                const auto local=rotatePoint(inverseRotation(torso.orientation),subtractPoint(p,torso.position));
                const auto& axis=functional.gpu.superiorAxisAndHeight;
                const float h=dotPoint(subtractPoint(functional.gpu.lungAnchorAndVolume,local),axis)/axis.w;
                const float upper=std::clamp(h/.2f,0.0f,1.0f),lower=std::clamp((1.6f-h)/.6f,0.0f,1.0f);
                maps[v].deformationWeight.x*=upper*upper*(3-2*upper)*lower*lower*(3-2*lower);
                if(maps[v].deformationWeight.x>.5f) {
                    const float ap=dotPoint(local,anatomyGPU.anteriorAxis);
                    posteriorSkin=std::min(posteriorSkin,ap);anteriorSkin=std::max(anteriorSkin,ap);
                }
            }
            if(maps[v].deformationKind==7||maps[v].deformationKind==4) {
                const auto local=rotatePoint(inverseRotation(initialThorax.orientation),subtractPoint(p,initialThorax.position));
                std::array<std::pair<float,unsigned>,24> distance;
                for(unsigned r=0;r<24;++r) {
                    const mr_float4 nearest{std::clamp(local.x,ribLo[r].x,ribHi[r].x),
                        std::clamp(local.y,ribLo[r].y,ribHi[r].y),std::clamp(local.z,ribLo[r].z,ribHi[r].z),0};
                    const auto delta=subtractPoint(local,nearest);distance[r]={dotPoint(delta,delta),r};
                }
                std::partial_sort(distance.begin(),distance.begin()+2,distance.end());
                maps[v].chamberIndex=distance[0].second;maps[v].deformationWeight.y=float(distance[1].second);
                const float first=std::sqrt(distance[0].first)+.0001f,second=std::sqrt(distance[1].first)+.0001f;
                maps[v].deformationWeight.z=first/(first+second);
            }
            const float length=std::sqrt(dotPoint(n,n));
            if(length>1e-6f){
                n=scalePoint(n,1/length);n.w=1;pack.vertices[v].normalAndTangentSign=n;
                const mr_float4 axis=std::abs(n.z)<.9f?mr_float4{0,0,1,0}:mr_float4{0,1,0,0};
                mr_float4 tangent{axis.y*n.z-axis.z*n.y,axis.z*n.x-axis.x*n.z,axis.x*n.y-axis.y*n.x,0};
                tangent=scalePoint(tangent,1/std::sqrt(dotPoint(tangent,tangent)));
                pack.vertices[v].tangent=tangent;
            }
        }
        require(std::isfinite(posteriorSkin)&&anteriorSkin-posteriorSkin>.05f,
            "registered skin has no usable anterior/posterior thorax extent");
        // Supine reduction: the dorsal support strip stays with its contact
        // owner. The chest expands toward the anterior surface; all five lung
        // lobes still share a single volume-preserving-interface affine map.
        anatomyGPU.lungAnchorAndVolume=addPoint(anatomyGPU.lungAnchorAndVolume,
            scalePoint(anatomyGPU.anteriorAxis,posteriorSkin-dotPoint(anatomyGPU.lungAnchorAndVolume,anatomyGPU.anteriorAxis)));
        anatomyGPU.lungAnchorAndVolume.w=functional.gpu.lungAnchorAndVolume.w;
        for(unsigned v=0;v<maps.size();++v)if(maps[v].deformationKind==3) {
            const auto& torso=initialBodies.at(anatomyGPU.bodyAndFlags.x);
            const auto p=rotatePoint(inverseRotation(torso.orientation),subtractPoint(pack.vertices[v].position,torso.position));
            const float ap=dotPoint(p,anatomyGPU.anteriorAxis),span=anteriorSkin-posteriorSkin;
            const float anteriorWeight=std::clamp((ap-posteriorSkin-.15f*span)/(.3f*span),0.0f,1.0f);
            maps[v].deformationWeight.x*=anteriorWeight*anteriorWeight*(3-2*anteriorWeight);
        }
        for(auto& primitive:pack.primitives) {
            mr_float4 lo{INFINITY,INFINITY,INFINITY,1},hi{-INFINITY,-INFINITY,-INFINITY,1};
            for(unsigned j=primitive.geometry.x;j<primitive.geometry.x+primitive.geometry.y;++j) {
                const auto p=pack.vertices[pack.indices[j]].position;
                lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);
                hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z);
            }
            primitive.boundsMinimum=lo;primitive.boundsMaximum=hi;
        }
        require(support.header.groundNormalX==0&&support.header.groundNormalY==0&&support.header.groundNormalZ==1,
                "resting visible bed must match the native +Z support plane");
        const float z=support.header.groundPointZ;
        require(z==0,"resting full-skin audit requires the authored zero-height bed plane");
        const unsigned first=unsigned(pack.vertices.size()),index=unsigned(pack.indices.size()),instance=unsigned(pack.instances.size());
        for(const auto& p:std::array<mr_float4,4>{{{framing.center.x-.65f,framing.center.y-1.15f,z,1},{framing.center.x+.65f,framing.center.y-1.15f,z,1},{framing.center.x+.65f,framing.center.y+1.15f,z,1},{framing.center.x-.65f,framing.center.y+1.15f,z,1}}})
            pack.vertices.push_back({p,{0,0,1,1},{1,0,0,0},{0,0,0,0},{1,1,1,1}});
        maps.resize(pack.vertices.size());
        pack.indices.insert(pack.indices.end(),{first,first+1,first+2,first,first+2,first+3});
        const unsigned bedMaterial=unsigned(pack.materials.size());
        pack.materials.push_back(makeMaterial({.045f,.065f,.09f,1},{0,0,0,0},.9f));
        MRVisualPrimitiveGPUV2 primitive{};primitive.geometry={index,6,bedMaterial,instance};primitive.identity={51999,1,MR_INVALID_INDEX,1};
        primitive.boundsMinimum=pack.vertices[first].position;primitive.boundsMaximum=pack.vertices[first+2].position;
        MRVisualInstanceGPUV2 bed{};bed.translationAndScale={0,0,0,1};bed.orientation={0,0,0,1};
        bed.binding={0,MR_INVALID_INDEX,MR_VISUAL_BINDING_WORLD,MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR|MR_VISUAL_INSTANCE_RECEIVES_SHADOW};
        bed.identity=primitive.identity;bed.geometry={unsigned(pack.primitives.size()),1,0,0};
        pack.primitives.push_back(primitive);pack.instances.push_back(bed);
        visibleLayers.push_back(127u);
        pack.preprocessingProvenance+="/accepted_native_body_skinning_no_truncated_weights/visible_native_contact_plane";
        pack.contentHash=metalrobo::computeVisualAssetPackContentHash(pack);
        const auto packPath=output/"resting-human.mrvpack";std::string error;
        require(metalrobo::writeVisualAssetPack(pack,packPath,&error),error);
        initialPackPath=packPath;initialPackContentHash=pack.contentHash;
        initialPackFileSHA256=loadedKneeSHA256Hex(loadedKneeFileSHA256(packPath));
        const float distance=framing.distance;
        const auto c=framing.center;
        const auto& thorax=initialBodies.at(functional.gpu.bodyAndFlags.x);
        const auto chest=addPoint(thorax.position,rotatePoint(thorax.orientation,functional.gpu.chamberCenterAndVolume[3]));
        const std::array<mr_float4,4> cameras{{{c.x,c.y-.2f*distance,c.z+distance,0},
            {c.x+distance,c.y,c.z+.2f*distance,0},{chest.x,chest.y-.12f,chest.z+.65f,0},
            {c.x+.55f*distance,c.y-.5f*distance,c.z+distance,0}}};
        const std::array<mr_float4,4> targets{c,c,chest,c};
        mr_float4 heartLocal{};
        for(const auto& center:functional.gpu.chamberCenterAndVolume)heartLocal=addPoint(heartLocal,center);
        const auto heart=addPoint(thorax.position,rotatePoint(thorax.orientation,scalePoint(heartLocal,.25f)));
        const auto heartCamera=makeCamera("heart_detail",{heart.x+.025f,heart.y-.035f,heart.z+.22f,0},heart,size);
        std::array<std::string,4> names;auto world=makeWorld(model,framing,size,names,&cameras,&targets,&heartCamera);
        metalrobo::WorldProgram program;program.id="numi_human_resting_native";metalrobo::WorldFamily family;
        auto compiled=metalrobo::compileWorldFamily(world,program,family);require(compiled.succeeded(),compiled.message);
        auto wc=worlds.compile(family,1);require(wc.succeeded(),wc.message);auto ws=worlds.sample(1,0x4e4852455354ull);require(ws.succeeded(),ws.message);
        const std::array refs{metalrobo::VisualAssetReferenceV3{packPath,pack.contentHash,0,kSkinShellSemantic,1}};
        metalrobo::VisualSceneManifestV3 manifest;
        require(metalrobo::compileVisualSceneManifestV3(world,refs,metalrobo::makeNeutralStudioEnvironmentV2(),
            makeHumanAnatomyLightRig(framing.center,cameras[3],framing.distance),manifest,&error),error);
        metalrobo::MetalHybridRendererConfig config;config.width=size;config.height=size;
        config.clearColorAndDepth={.012f,.019f,.03f,1e30f};renderer=std::make_unique<metalrobo::MetalHybridRenderer>(config);
        auto rc=renderer->compile(std::move(manifest.renderScene),metalrobo::VisualRendererProfileV1::sensorFast(),1);require(rc.succeeded(),rc.message);
        require(renderer->layout().meshVertexCount==maps.size(),"resting compiled vertex order changed");
        auto device=coupled.physiology.device;queue=[device newCommandQueue];
        mapping=[device newBufferWithBytes:maps.data() length:maps.size()*sizeof(maps.front()) options:MTLResourceStorageModeShared];
        influences=[device newBufferWithBytes:weights.data() length:weights.size()*sizeof(weights.front()) options:MTLResourceStorageModeShared];
        anatomyParameters=[device newBufferWithBytes:&anatomyGPU length:sizeof(anatomyGPU) options:MTLResourceStorageModeShared];
        auditCount=unsigned(audits.size());require(auditCount==9,"functional anatomy volume audit did not bind all nine surfaces");
        surfaceAudits=[device newBufferWithBytes:audits.data() length:audits.size()*sizeof(audits.front()) options:MTLResourceStorageModeShared];
        volumeResults=[device newBufferWithLength:(auditCount+1)*sizeof(mr_float4) options:MTLResourceStorageModeShared];
        cardiacQ=[device newBufferWithLength:4*sizeof(float) options:MTLResourceStorageModeShared];
        instanceLayers=[device newBufferWithBytes:visibleLayers.data() length:visibleLayers.size()*sizeof(unsigned) options:MTLResourceStorageModeShared];
        NSError* e=nil;auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(NUMI_HUMAN_RESPIRATION_METALLIB)] error:&e];
        require(skinFirstVertex!=MR_INVALID_INDEX,"resting support has no registered skin range");
        const auto supportQueries=makeMetalMujocoVisualQueries(model,&support);
        std::vector<MRHumanRestingSupportRegionGPU> regions;
        std::vector<unsigned> seeds;
        for(const auto& row:supportQueries.supportContacts) {
            require(row.sourceGeometryIndex>0&&row.sourceGeometryIndex<=skin.header.vertexCount,
                "resting NHCNT row has no valid source skin vertex identity");
            const unsigned seed=row.sourceGeometryIndex-1;
            require(std::find(seeds.begin(),seeds.end(),seed)==seeds.end(),"resting support seed is duplicated");
            seeds.push_back(seed);regions.push_back({row.pointQueryIndex,row.sourceGeometryIndex,0,0});
        }
        std::vector<unsigned> regionForVertex(skin.header.vertexCount);
        std::vector<unsigned> regionCounts(seeds.size());
        for(unsigned v=0;v<skin.header.vertexCount;++v) {
            float nearest=INFINITY;unsigned chosen=MR_INVALID_INDEX;
            for(unsigned r=0;r<seeds.size();++r) {
                const auto delta=subtractPoint(skinRestWorld[v],skinRestWorld[seeds[r]]);
                const float distance=dotPoint(delta,delta);
                if(distance<nearest){nearest=distance;chosen=r;}
            }
            require(chosen!=MR_INVALID_INDEX,"registered skin Voronoi assignment is invalid");
            regionForVertex[v]=chosen;++regionCounts[chosen];
        }
        HumanBrainSourceFingerprint supportIdentity;
        supportIdentity.text("numi.human.full-skin-support.v1");
        supportIdentity.integer(coupled.brain.rootProgramIdentity);
        supportIdentity.integer(skinFirstVertex*sizeof(MRHumanRestingVertexMap));
        supportIdentity.integer(skin.header.vertexCount);
        supportIdentity.bytes(maps.data()+skinFirstVertex,skin.header.vertexCount*sizeof(maps.front()));
        supportIdentity.bytes(weights.data(),weights.size()*sizeof(weights.front()));
        supportIdentity.bytes(regionForVertex.data(),regionForVertex.size()*sizeof(regionForVertex.front()));
        supportIdentity.bytes(regions.data(),regions.size()*sizeof(regions.front()));
        skinSupport=std::make_unique<NumiHumanRestingSupportGeometry>(device,lib,mapping,
            std::span<const MRHumanRestingVertexMap>(maps).subspan(skinFirstVertex,skin.header.vertexCount),
            influences,std::span<const MRHumanRestingInfluence>(weights),regionForVertex,regions,
            supportQueries.supportContacts,1u,supportIdentity.value(),
            skinFirstVertex*sizeof(MRHumanRestingVertexMap));
        std::cout<<"resting_support_geometry=full_registered_skin vertices="<<skin.header.vertexCount
            <<" regions="<<regions.size()<<" full_binding_count="<<skin.bindings.size()
            <<" region_sizes=[";
        for(unsigned r=0;r<regionCounts.size();++r){if(r)std::cout<<',';std::cout<<regionCounts[r];}
        std::cout<<"] force_owner=existing_metal_stand partition=source_rest_voronoi\n";
        skinPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_skin"] error:&e];
        cardiacQPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_cardiac_q"] error:&e];
        layerPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_layers"] error:&e];
        volumePipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_audit_volumes"] error:&e];
        skinAuditPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_audit_skin"] error:&e];
        if(!requestedGeometrySteps.empty()) {
            vertexCapturePipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_capture_vertices"] error:&e];
            vertexCaptureBuffer=[device newBufferWithLength:maps.size()*sizeof(MRVisualVertexGPUV2) options:MTLResourceStorageModeShared];
            vertexCaptureBuffer.label=@"Numi Human selected accepted render vertices";
        }
        require(mapping&&influences&&anatomyParameters&&surfaceAudits&&volumeResults&&instanceLayers&&cardiacQ&&skinPipeline&&cardiacQPipeline&&layerPipeline&&volumePipeline&&skinAuditPipeline,"resting GPU anatomy setup failed");
        require(requestedGeometrySteps.empty()||(vertexCapturePipeline&&vertexCaptureBuffer),
            "selected accepted geometry capture resources could not be created");
        if(!presentWindow)return;
        [NSApplication sharedApplication];[NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];[NSApp finishLaunching];
        window=[[NumiHumanRestingWindow alloc] initWithAdvance:[this](unsigned camera,unsigned selectedLayer){return render(camera,selectedLayer);}
            movie:[NSString stringWithUTF8String:movie.c_str()]];
        [window showWindow:nil];[NSApp activateIgnoringOtherApps:YES];wallOrigin=CACurrentMediaTime();
    }
    static bool deform(void* context,const metalrobo::MetalHybridMeshDeformationLease& lease) {
        auto& self=*static_cast<NumiHumanRestingVisual*>(context);const auto& e=*lease.encoder;
        const mr_uint4 d={lease.meshVertexCount,lease.meshInstanceCount,self.layer,self.auditCount};
        e.setPipeline(e.context,(__bridge void*)self.cardiacQPipeline);
        e.setBuffer(e.context,(__bridge void*)self.coupled.presentationRespiration,0,0);
        e.setBuffer(e.context,(__bridge void*)self.anatomyParameters,0,1);
        e.setBuffer(e.context,(__bridge void*)self.cardiacQ,0,2);
        e.dispatchThreads(e.context,4,1);
        e.setPipeline(e.context,(__bridge void*)self.skinPipeline);e.setBytes(e.context,&d,sizeof(d),0);
        e.setBuffer(e.context,(__bridge void*)self.mapping,0,1);e.setBuffer(e.context,(__bridge void*)self.influences,0,2);
        e.setBuffer(e.context,(__bridge void*)self.coupled.presentationBodies,0,3);e.setBuffer(e.context,lease.meshVertices,0,4);
        e.setBuffer(e.context,(__bridge void*)self.coupled.presentationRespiration,0,5);
        e.setBuffer(e.context,(__bridge void*)self.anatomyParameters,0,6);
        e.setBuffer(e.context,(__bridge void*)self.cardiacQ,0,7);
        e.dispatchThreads(e.context,lease.meshVertexCount,64);
        e.setPipeline(e.context,(__bridge void*)self.volumePipeline);e.setBytes(e.context,&d,sizeof(d),0);
        e.setBuffer(e.context,(__bridge void*)self.surfaceAudits,0,1);e.setBuffer(e.context,lease.meshIndices,0,2);
        e.setBuffer(e.context,lease.meshVertices,0,3);e.setBuffer(e.context,(__bridge void*)self.coupled.presentationRespiration,0,4);
        e.setBuffer(e.context,(__bridge void*)self.anatomyParameters,0,5);e.setBuffer(e.context,(__bridge void*)self.volumeResults,0,6);
        e.dispatchThreads(e.context,self.auditCount,1);
        e.setPipeline(e.context,(__bridge void*)self.skinAuditPipeline);e.setBytes(e.context,&d,sizeof(d),0);
        e.setBuffer(e.context,(__bridge void*)self.mapping,0,1);e.setBuffer(e.context,lease.meshVertices,0,2);
        e.setBuffer(e.context,(__bridge void*)self.volumeResults,0,3);e.dispatchThreads(e.context,256,256);
        e.setPipeline(e.context,(__bridge void*)self.layerPipeline);e.setBytes(e.context,&d,sizeof(d),0);
        e.setBuffer(e.context,lease.meshInstances,0,1);e.setBuffer(e.context,(__bridge void*)self.instanceLayers,0,2);
        e.dispatchThreads(e.context,lease.meshInstanceCount,64);
        if(self.captureThisFrame) {
            if(!self.vertexCapturePipeline||!self.vertexCaptureBuffer||
               lease.acceptedStateBuffer!=(__bridge void*)self.coupled.presentationRespiration||
               lease.acceptedRootFingerprint!=self.captureRootFingerprint||
               lease.acceptedTransactionFingerprint!=self.captureTransactionFingerprint||
               lease.acceptedTimestampMicroseconds!=self.captureTimestampMicroseconds)return false;
            const unsigned count=lease.meshVertexCount;
            e.setPipeline(e.context,(__bridge void*)self.vertexCapturePipeline);
            e.setBuffer(e.context,lease.meshVertices,0,0);
            e.setBuffer(e.context,(__bridge void*)self.vertexCaptureBuffer,0,1);
            e.setBytes(e.context,&count,sizeof(count),2);
            e.dispatchThreads(e.context,count,64);self.captureKernelEncoded=true;
        }
        return true;
    }
    NumiHumanRestingFrame render(unsigned camera,unsigned selectedLayer) {
        layer=selectedLayer;
        const auto& p=*static_cast<const NMHumanRespirationState*>(coupled.presentationRespiration.contents);
        const double time=p.status.x*double(coupled.physiology.runtime.timestepSeconds());
        captureStep=p.status.x;captureCamera=camera;captureLayer=selectedLayer;
        captureThisFrame=requestedGeometrySteps.contains(captureStep)&&
            !completedGeometrySteps.contains(captureStep);captureKernelEncoded=false;
        metalrobo::HybridDeviceStateBatch state;
        state.currentBodyStates=(__bridge void*)coupled.presentationBodies;state.environmentCount=1;
        state.bodyCount=unsigned(coupled.presentationBodies.length/sizeof(MRBodyStateGPU));state.frameIndex=p.status.x;
        state.sensorSequence=p.status.x;state.captureTimestampSeconds=time;
        state.acceptedRootFingerprint=coupled.brain.root(p.status.x);state.acceptedTransactionFingerprint=state.acceptedRootFingerprint;
        state.acceptedTimestampMicroseconds=std::llround(time*1e6);
        captureRootFingerprint=state.acceptedRootFingerprint;
        captureTransactionFingerprint=state.acceptedTransactionFingerprint;
        captureTimestampMicroseconds=state.acceptedTimestampMicroseconds;
        const auto layout=renderer->layout();metalrobo::MetalHybridMeshDeformationRequest request;
        request.context=this;request.encode=&deform;request.acceptedStateIsCommitted=true;
        request.acceptedStateBuffer=(__bridge void*)coupled.presentationRespiration;request.acceptedStateByteCount=coupled.presentationRespiration.length;
        request.acceptedRootFingerprint=state.acceptedRootFingerprint;request.acceptedTransactionFingerprint=state.acceptedTransactionFingerprint;
        request.acceptedTimestampMicroseconds=state.acceptedTimestampMicroseconds;
        request.expectedMeshVertexCount=layout.meshVertexCount;request.expectedMeshIndexCount=layout.meshIndexCount;
        request.expectedMeshTriangleCount=layout.meshTriangleCount;request.expectedMeshPrimitiveCount=layout.meshPrimitiveCount;request.expectedMeshInstanceCount=layout.meshInstanceCount;
        state.meshDeformation=&request;
        auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];
        auto result=renderer->encode(worlds,state,camera,(__bridge void*)enc);require(result.succeeded(),result.message);
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"resting native renderer failed");
        if(captureThisFrame) {
            require(captureKernelEncoded,"selected accepted step did not encode its geometry snapshot");
            exportAcceptedGeometry(captureStep,time,captureRootFingerprint,captureTransactionFingerprint,
                captureTimestampMicroseconds);
            captureThisFrame=false;
        }
        const auto* volumes=static_cast<const mr_float4*>(volumeResults.contents);
        const auto* cardiacCoordinates=static_cast<const float*>(cardiacQ.contents);
        float maxRelativeError=0;for(unsigned i=0;i<auditCount;++i)maxRelativeError=std::max(maxRelativeError,volumes[i].z);
        const auto skinAudit=volumes[auditCount];
        surfaceTrace<<std::setprecision(12)<<time<<','<<p.status.x<<','<<skinAudit.x<<','<<skinAudit.z<<','<<skinAudit.w<<','<<maxRelativeError
            <<','<<cardiacCoordinates[0]<<','<<cardiacCoordinates[1]<<','<<cardiacCoordinates[2]<<','<<cardiacCoordinates[3]
            <<','<<p.chamberVolumes.x*1e6<<','<<p.chamberVolumes.y*1e6<<','<<p.chamberVolumes.z*1e6<<','<<p.chamberVolumes.w*1e6
            <<','<<p.motion.x*1e6<<','<<p.motion.y*1e6<<','<<p.mechanics.x*1e6<<'\n';
        surfaceTrace.flush();
        for(unsigned i=0;i<auditCount;++i)require(volumes[i].w==0,"accepted anatomical mesh volume disagrees with its functional owner");
        require(skinAudit.w==0&&skinAudit.z==0,"accepted full skin intersects the bed beyond the 1 mm inspection tolerance");
        std::ostringstream metrics;metrics<<std::fixed<<std::setprecision(2)<<"Accepted time "<<time<<" s  |  "<<time/std::max(.001,CACurrentMediaTime()-wallOrigin)<<" x real time  |  breaths "<<p.status.y<<"  beats "<<p.cardiacStatus.x<<"\n"
            <<"Lung "<<p.mechanics.x*1e3<<" L  Airflow "<<p.mechanics.w*1e3<<" L/s  Pleural "<<p.mechanics.z/98.0665<<" cmH2O  Muscle activation "
            <<p.muscles[0].excitationAndActivation.y<<" / "<<p.muscles[1].excitationAndActivation.y<<"\n"
            <<"PaO2 "<<p.observation.x<<"  PaCO2 "<<p.observation.y<<" mmHg  SaO2 "<<p.observation.z*100<<"%  Tidal "<<p.breath.z*1e6<<" mL\n"
            <<"LV "<<p.circulation.y*1e6<<" mL / "<<p.cardiacPressure.x/133.322387415<<" mmHg  RV "<<p.circulation.z*1e6<<" mL  Stroke "<<p.cardiacFlow.w*1e6<<" mL  Blood "<<p.circulation.x*1e3<<" L  CO mean "
            <<(time>0?p.cardiacFlow.x*60e3/time:0)<<" L/min\n"
            <<"Mixed-source reference anatomy; passive structures remain inspection geometry.";
        return {(__bridge id<MTLBuffer>)renderer->nativeBuffer(metalrobo::MetalHybridRendererBuffer::rgb),dimension,dimension,metrics.str(),time,complete};
    }
    void present(bool finished=false){
        complete=finished;[window renderFrameNow];
        if(finished)for(unsigned step:requestedGeometrySteps)require(completedGeometrySteps.contains(step),
            "requested accepted geometry step was not presented: "+std::to_string(step));
    }
    metalrobo::MetalNumiHumanSupportGeometryProgram supportProgram(){return skinSupport->program();}
    ~NumiHumanRestingVisual(){[window finishRecording];}
};
