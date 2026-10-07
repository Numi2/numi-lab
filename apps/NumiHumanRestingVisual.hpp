#pragma once
#include "NumiHumanRestingAnatomy.hpp"
#include "NumiHumanRestingSupportGeometry.hpp"
#include "NumiHumanRestingSurfaceAuditDiagnostic.hpp"
#include <charconv>
#include <bit>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string_view>
// Included after the existing Human source loaders and visual-pack compiler.
// It adds a persistent presentation consumer, not another dynamics owner.
static_assert(sizeof(MRHumanRestingCardiacWallVertexGPU)==48);
static_assert(sizeof(MRHumanRestingCardiacWallGPU)==128);
static_assert(sizeof(MRHumanRestingSurfaceFailureGPU)==80);
static_assert(alignof(MRHumanRestingSurfaceFailureGPU)==16);
class NumiHumanRestingVisual {
    NumiHumanRestingCoupling& coupled;
    std::unique_ptr<NumiHumanRestingSupportGeometry> skinSupport;
    std::unique_ptr<metalrobo::MetalHybridRenderer> renderer;
    metalrobo::MetalWorldFamilyContext worlds;
    id<MTLBuffer> mapping, influences, anatomyParameters, surfaceAudits, volumeResults, instanceLayers, cardiacQ;
    id<MTLBuffer> cardiacWallMap, cardiacWallParameters, cardiacWallQ, cardiacWallNormalRanges, cardiacWallIncidentTriangles;
    id<MTLBuffer> commonFieldMapBuffer, commonFieldParameters, commonFieldBoxes, commonFieldCoordinates;
    id<MTLBuffer> commonFieldNormalRanges, commonFieldIncidentTriangles;
    id<MTLBuffer> airwayNormalRanges, airwayIncidentTriangles;
    id<MTLBuffer> meshAuditPartials,meshAuditResult;
    id<MTLComputePipelineState> meshAuditPipeline,meshAuditReducePipeline;
    std::vector<MRVisualPrimitiveGPUV2> auditedMeshPrimitives;
    static constexpr unsigned meshAuditGroupCount=64;
    id<MTLComputePipelineState> skinPipeline, layerPipeline, volumePipeline, skinAuditPipeline, cardiacQPipeline, bodyAuditPipeline;
    id<MTLComputePipelineState> cardiacWallQPipeline, cardiacWallNormalsPipeline;
    id<MTLComputePipelineState> commonCoordinatesPipeline=nil,commonCoordinateStatusPipeline=nil;
    id<MTLComputePipelineState> vertexCapturePipeline=nil;
    id<MTLBuffer> vertexCaptureBuffer=nil;
    id<MTLCommandQueue> queue;
    NumiHumanRestingWindow* window=nil;
    unsigned dimension,layer=0;
    bool complete=false;
    bool rigidHands=false;
    bool profileTiming=false;
    bool profileGpuTiming=false,gpuTimingUnavailableReported=false;
    unsigned auditCount=0;
    std::vector<unsigned> auditStableIds;
    unsigned cardiacWallVertexCount=0,cardiacWallAuditIndex=MR_INVALID_INDEX;
    bool commonCardiacGeometry=false;
    unsigned commonFieldVertexCount=0,commonFieldNormalVertexCount=0,airwayNormalVertexCount=0;
    std::array<unsigned,7> commonFieldAuditIndices{{MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX,
        MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX}};
    double wallOrigin=0;
    std::filesystem::path outputDirectory, initialPackPath, acceptedGeometryDirectory;
    std::string initialPackContentHash, initialPackFileSHA256;
    std::string cardiacWallMapSHA256,cardiacWallParametersSHA256,cardiacWallBundleSHA256,cardiacWallIdentityReceiptSHA256;
    std::string commonFieldMapSHA256,commonFieldParametersSHA256,commonFieldPolynomialSHA256,
        commonFieldBoxesSHA256,commonFieldBundleSHA256,commonFieldIdentityReceiptSHA256,commonFieldSourcePayloadSHA256;
    NSDictionary* initialAnatomicalRegistration=nil;
    std::set<unsigned> registrationBodyIndices;
    std::set<unsigned> requestedGeometrySteps, completedGeometrySteps;
    bool captureThisFrame=false, captureKernelEncoded=false;
    unsigned captureStep=0,captureCamera=0,captureLayer=0;
    std::uint64_t captureRootFingerprint=0,captureTransactionFingerprint=0,captureTimestampMicroseconds=0;
    std::ofstream surfaceTrace;
    static mr_float4 inverseRotation(mr_float4 q){return {-q.x,-q.y,-q.z,q.w};}
    static NSDictionary* bodyPose(unsigned index,const MRBodyStateGPU& body) {
        return @{@"body_index":@(index),
            @"position_m":@[@(body.position.x),@(body.position.y),@(body.position.z)],
            @"quaternion_xyzw":@[@(body.orientation.x),@(body.orientation.y),@(body.orientation.z),@(body.orientation.w)]};
    }
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
    void writeCommonFieldIdentity(const std::filesystem::path& output,
                                  const NumiHumanRestingAnatomy& functional) {
        require(functional.commonCardiacGeometry,"common-field identity requested for legacy anatomy");
        require(sizeof(float)==4&&std::numeric_limits<float>::is_iec559&&
            std::endian::native==std::endian::little,
            "common-field capture requires little-endian IEEE-754 binary32");
        const auto mapPath=output/"common-field-map-f32.bin";
        const auto parametersPath=output/"common-field-parameters-f32.bin";
        const auto boxesPath=output/"common-field-domain-boxes-f32.bin";
        const auto polynomialPath=output/"common-field-polynomials-f32.bin";
        const auto receiptPath=output/"common-field-map-identity.json";
        require(std::filesystem::is_directory(output)&&!std::filesystem::exists(mapPath)&&
            !std::filesystem::exists(parametersPath)&&!std::filesystem::exists(boxesPath)&&
            !std::filesystem::exists(polynomialPath)&&!std::filesystem::exists(receiptPath),
            "common-field identity output is unavailable or already exists");
        const auto mapBytes=functional.commonFieldMap.size()*sizeof(MRHumanRestingCommonFieldVertexGPU);
        const auto parameterBytes=sizeof(functional.commonFieldGPU);
        const auto boxesBytes=functional.commonFieldBoxes.size()*sizeof(MRHumanRestingCommonCoordinateBoxGPU);
        require(commonFieldMapBuffer.contents&&commonFieldParameters.contents&&commonFieldBoxes.contents&&
            commonFieldMapBuffer.length==mapBytes&&commonFieldParameters.length==parameterBytes&&commonFieldBoxes.length==boxesBytes&&
            std::memcmp(commonFieldMapBuffer.contents,functional.commonFieldMap.data(),mapBytes)==0&&
            std::memcmp(commonFieldParameters.contents,&functional.commonFieldGPU,parameterBytes)==0&&
            std::memcmp(commonFieldBoxes.contents,functional.commonFieldBoxes.data(),boxesBytes)==0,
            "uploaded common-field buffers differ from the admitted anatomy map");
        const auto digest=[](const void* bytes,std::size_t size) {
            return loadedKneeSHA256Hex(loadedKneeSHA256(bytes,size));
        };
        const auto polynomialBytes=sizeof(functional.commonFieldGPU.volumePolynomial);
        commonFieldMapSHA256=digest(commonFieldMapBuffer.contents,mapBytes);
        commonFieldParametersSHA256=digest(commonFieldParameters.contents,parameterBytes);
        commonFieldPolynomialSHA256=digest(commonFieldParameters.contents,polynomialBytes);
        commonFieldBoxesSHA256=digest(commonFieldBoxes.contents,boxesBytes);
        require(commonFieldPolynomialSHA256==functional.commonFieldPolynomialSHA256&&
            commonFieldMapSHA256==functional.commonFieldMapSHA256&&commonFieldBoxesSHA256==functional.commonFieldBoxesSHA256,
            "loaded common-field sidecar hashes differ from exact uploaded bytes");
        NSMutableData* bundle=[NSMutableData dataWithBytes:commonFieldMapBuffer.contents length:mapBytes];
        [bundle appendBytes:commonFieldParameters.contents length:parameterBytes];
        [bundle appendBytes:commonFieldBoxes.contents length:boxesBytes];
        commonFieldBundleSHA256=digest(bundle.bytes,bundle.length);
        const auto writeBytes=[&](const std::filesystem::path& path,const void* bytes,std::size_t size,
                                  const std::string& expected) {
            NSData* data=[NSData dataWithBytes:bytes length:size];NSError* error=nil;
            require([data writeToURL:[NSURL fileURLWithPath:loadedKneeNSString(path.string())]
                options:NSDataWritingAtomic error:&error],"common-field coefficient identity write failed");
            require(loadedKneeSHA256Hex(loadedKneeFileSHA256(path))==expected,
                "written common-field bytes differ from the uploaded buffer");
        };
        writeBytes(mapPath,commonFieldMapBuffer.contents,mapBytes,commonFieldMapSHA256);
        writeBytes(parametersPath,commonFieldParameters.contents,parameterBytes,commonFieldParametersSHA256);
        writeBytes(boxesPath,commonFieldBoxes.contents,boxesBytes,commonFieldBoxesSHA256);
        writeBytes(polynomialPath,commonFieldParameters.contents,polynomialBytes,commonFieldPolynomialSHA256);
        NSMutableArray* ranges=[NSMutableArray array];
        for(unsigned id:functional.commonFieldStableIDs) {
            const auto range=functional.commonFieldRanges.at(id);
            [ranges addObject:@{@"stable_id":@(id),@"first_vertex":@(range.first),@"vertex_count":@(range.second)}];
        }
        NSMutableArray* passiveRanges=[NSMutableArray array];
        for(const auto& range:functional.commonFieldPassiveAttachments) {
            NSMutableDictionary* row=[@{@"stable_id":@(range.stableId),@"semantic":@(range.semantic),
                @"body_index":@(range.bodyIndex),@"first_vertex":@(range.firstVertex),
                @"vertex_count":@(range.vertexCount)} mutableCopy];
            if(range.inferiorVenaCava)row[@"original_body_index"]=@(range.originalBodyIndex);
            if(range.inferiorVenaCava)row[@"attachment"]=@{
                @"anchor_body_index":@(range.anchorBodyIndex),@"torso_body_index":@(range.bodyIndex),
                @"source_superior_axis":@(range.sourceSuperiorAxis),
                @"transition_lower_m":@(range.transitionLower),@"transition_upper_m":@(range.transitionUpper),
                @"rule":@"quintic_smoothstep"};
            [passiveRanges addObject:row];
        }
        NSError* error=nil;
        NSDictionary* receipt=@{
            @"schema":@"numi.human.cardiac_common_field_loaded_buffers.v1",
            @"geometry_mode":@"common_seven_coordinate_v1",
            @"source_anatomy_payload_sha256":loadedKneeNSString(functional.commonFieldAnatomyPayloadSHA256),
            @"coordinate_order":@[@"RA",@"RV",@"LA",@"LV",@"RA-material",@"ventricular-material",@"LA-material"],
            @"volume_stable_ids":@[@318,@319,@320,@321,@1,@23,@24],
            @"vertex_ranges":ranges,
            @"passive_attachment_ranges":passiveRanges,
            @"map":@{@"path":loadedKneeNSString(mapPath.filename().string()),@"sha256":loadedKneeNSString(commonFieldMapSHA256),
                @"record_count":@(functional.commonFieldMap.size()),@"record_stride_bytes":@112},
            @"polynomials":@{@"path":loadedKneeNSString(polynomialPath.filename().string()),@"sha256":loadedKneeNSString(commonFieldPolynomialSHA256),
                @"volume_count":@7,@"term_count":@120,@"coefficient_type":@"float32_le",
                @"normalization":@"source_reference_volume",
                @"monomial_order":@"lexicographic_e0_to_e6_total_degree_le_3"},
            @"parameters":@{@"path":loadedKneeNSString(parametersPath.filename().string()),@"sha256":loadedKneeNSString(commonFieldParametersSHA256),
                @"record_bytes":@(parameterBytes)},
            @"domain_boxes":@{@"path":loadedKneeNSString(boxesPath.filename().string()),@"sha256":loadedKneeNSString(commonFieldBoxesSHA256),
                @"count":@(functional.commonFieldBoxes.size()),@"record_stride_bytes":@64},
            @"map_record_layout":@"seven float4 displacements, xyz metres per unit dimensionless coordinate, W exact +0",
            @"domain_record_layout":@"lower float4[2] followed by upper float4[2], padding lanes exact +0",
            @"uploaded_buffer_bundle_sha256":loadedKneeNSString(commonFieldBundleSHA256)
        };
        NSData* json=[NSJSONSerialization dataWithJSONObject:receipt
            options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
        require(json!=nil,"common-field identity receipt serialization failed");
        require([json writeToURL:[NSURL fileURLWithPath:loadedKneeNSString(receiptPath.string())]
            options:NSDataWritingAtomic error:&error],"common-field identity receipt write failed");
        commonFieldIdentityReceiptSHA256=loadedKneeSHA256Hex(loadedKneeFileSHA256(receiptPath));
        std::cout<<"common_field_map_sha256="<<commonFieldMapSHA256
            <<" common_field_polynomial_sha256="<<commonFieldPolynomialSHA256
            <<" common_field_boxes_sha256="<<commonFieldBoxesSHA256
            <<" common_field_identity_receipt_sha256="<<commonFieldIdentityReceiptSHA256<<"\n";
    }
    // Reuse the retained load-time map identity format. Capture only when the
    // caller opts into accepted geometry evidence; no extra per-step readback.
    void writeCardiacWallMapIdentity(const std::filesystem::path& output,
                                    const NumiHumanRestingAnatomy& functional) {
        if(functional.ventricularWallMap.empty())return;
        require(sizeof(float)==4&&std::numeric_limits<float>::is_iec559&&
            std::endian::native==std::endian::little,
            "cardiac coefficient capture requires little-endian IEEE-754 binary32");
        const auto mapPath=output/"ventricular-wall-map-f32.bin";
        const auto parametersPath=output/"ventricular-wall-parameters-f32.bin";
        const auto receiptPath=output/"ventricular-wall-map-identity.json";
        require(std::filesystem::is_directory(output)&&!std::filesystem::exists(mapPath)&&
            !std::filesystem::exists(parametersPath)&&!std::filesystem::exists(receiptPath),
            "cardiac coefficient identity output is unavailable or already exists");
        const auto mapBytes=functional.ventricularWallMap.size()*sizeof(MRHumanRestingCardiacWallVertexGPU);
        const auto parameterBytes=sizeof(functional.ventricularWallGPU);
        require(cardiacWallMap.contents&&cardiacWallParameters.contents&&cardiacWallMap.length==mapBytes&&
            cardiacWallParameters.length==parameterBytes&&
            std::memcmp(cardiacWallMap.contents,functional.ventricularWallMap.data(),mapBytes)==0&&
            std::memcmp(cardiacWallParameters.contents,&functional.ventricularWallGPU,parameterBytes)==0,
            "uploaded cardiac coefficient buffers differ from the admitted anatomy map");
        const auto digest=[](const void* bytes,std::size_t size) {
            return loadedKneeSHA256Hex(loadedKneeSHA256(bytes,size));
        };
        cardiacWallMapSHA256=digest(cardiacWallMap.contents,mapBytes);
        cardiacWallParametersSHA256=digest(cardiacWallParameters.contents,parameterBytes);
        NSMutableData* bundle=[NSMutableData dataWithBytes:cardiacWallMap.contents length:mapBytes];
        [bundle appendBytes:cardiacWallParameters.contents length:parameterBytes];
        cardiacWallBundleSHA256=digest(bundle.bytes,bundle.length);
        const auto writeBytes=[&](const std::filesystem::path& path,const void* bytes,std::size_t size,
                                  const std::string& expected) {
            NSData* data=[NSData dataWithBytes:bytes length:size];NSError* error=nil;
            require([data writeToURL:[NSURL fileURLWithPath:loadedKneeNSString(path.string())]
                options:NSDataWritingAtomic error:&error],"cardiac coefficient identity write failed");
            require(loadedKneeSHA256Hex(loadedKneeFileSHA256(path))==expected,
                "written cardiac coefficient bytes differ from the uploaded buffer");
        };
        writeBytes(mapPath,cardiacWallMap.contents,mapBytes,cardiacWallMapSHA256);
        writeBytes(parametersPath,cardiacWallParameters.contents,parameterBytes,cardiacWallParametersSHA256);
        const auto& w=functional.ventricularWallGPU;
        NSMutableArray* polynomial=[NSMutableArray array];
        for(unsigned i=0;i<20;++i) {
            const auto& row=w.volumePolynomial[i/4];
            const std::array<float,4> values{{row.x,row.y,row.z,row.w}};
            [polynomial addObject:@(values[i%4])];
        }
        NSDictionary* receipt=@{
            @"schema":@"numi.human.cardiac.ventricular_wall_map_identity.v1",@"stable_id":@23,
            @"source_anatomy_payload_sha256":loadedKneeNSString(functional.ventricularWallAnatomyPayloadSHA256),
            @"coefficient_abi":@"MRHumanRestingCardiacWallVertexGPU:48-byte IEEE-754 binary32 records; MRHumanRestingCardiacWallGPU:128 bytes",
            @"byte_order":@"little-endian native uploaded bytes",
            @"record_layout":@"first.xyz=RV m/qRV; second.xyz=LV m/qLV; closure.xyz=dimensionless displacement per metre of closure; all w lanes reserved zero",
            @"map_file":@"ventricular-wall-map-f32.bin",@"map_bytes":@(mapBytes),
            @"map_sha256":loadedKneeNSString(cardiacWallMapSHA256),
            @"parameters_file":@"ventricular-wall-parameters-f32.bin",@"parameters_bytes":@(parameterBytes),
            @"parameters_sha256":loadedKneeNSString(cardiacWallParametersSHA256),
            @"bundle_sha256":loadedKneeNSString(cardiacWallBundleSHA256),
            @"bundle_order":@"map bytes followed by parameter bytes",
            @"map_vertex_count":@(functional.ventricularWallMap.size()),
            @"source_material_volume_ml":@(functional.ventricularWallSourceMaterialVolumeM3*1e6),
            @"reference_material_volume_ml":@(functional.ventricularWallReferenceMaterialVolumeM3*1e6),
            @"gpu_q_scales":@[@(w.scalesAndVolume.x),@(w.scalesAndVolume.y),@(w.scalesAndVolume.z)],
            @"closure_bounds_and_relative_tolerance":@[@(w.closureBoundsAndTolerance.x),
                @(w.closureBoundsAndTolerance.y),@(w.closureBoundsAndTolerance.z)],
            @"gpu_volume_polynomial_m3":polynomial,
            @"capture_scope":@"Exact immutable Metal input buffers immediately after native asset upload; no simulation step or independent physiology update"
        };
        NSError* error=nil;
        NSData* json=[NSJSONSerialization dataWithJSONObject:receipt
            options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
        require(json!=nil,"cardiac coefficient identity JSON serialization failed");
        cardiacWallIdentityReceiptSHA256=digest(json.bytes,json.length);
        writeBytes(receiptPath,json.bytes,json.length,cardiacWallIdentityReceiptSHA256);
        std::cout<<"cardiac_wall_map_sha256="<<cardiacWallMapSHA256
            <<" cardiac_wall_parameters_sha256="<<cardiacWallParametersSHA256
            <<" cardiac_wall_identity_receipt="<<receiptPath.string()
            <<" cardiac_wall_identity_receipt_sha256="<<cardiacWallIdentityReceiptSHA256<<"\n";
    }
    void exportAcceptedGeometry(unsigned step,double time,std::uint64_t root,std::uint64_t transaction,
                                std::uint64_t timestamp,NSDictionary* surfaceAuditOutcome) {
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
        // Opt-in geometry capture already hashes these completed presentation
        // buffers. Expose the few anatomical registration poses from the same
        // bytes, so offline interface audits need no fitted body transforms.
        NSMutableArray* registeredPoses=[NSMutableArray array];
        const auto* presentedBodies=static_cast<const MRBodyStateGPU*>(coupled.presentationBodies.contents);
        for(unsigned index:registrationBodyIndices)
            [registeredPoses addObject:bodyPose(index,presentedBodies[index])];
        NSError* jsonError=nil;
        NSDictionary* receipt=@{
            @"schema":@"numi.human.accepted-render-geometry.v1",
            @"accepted_step":@(step),@"accepted_time_s":@(time),
            @"accepted_root_fingerprint":@(root),@"accepted_root_fingerprint_hex":loadedKneeNSString(rootHex),
            @"accepted_transaction_fingerprint":@(transaction),
            @"accepted_timestamp_microseconds":@(timestamp),
            @"camera_index":@(captureCamera),@"anatomy_layer_index":@(captureLayer),
            @"accepted_body_state_sha256":loadedKneeNSString(bodySHA),
            @"accepted_registered_body_poses":registeredPoses,
            @"initial_anatomical_registration":initialAnatomicalRegistration?initialAnatomicalRegistration:@{},
            @"physical_endpoint":@"accepted",@"surface_audit_endpoint":@"passed",
            @"surface_audit":surfaceAuditOutcome,
            @"accepted_respiration_state_sha256":loadedKneeNSString(respirationSHA),
            @"ventricular_wall_map_sha256":loadedKneeNSString(cardiacWallMapSHA256),
            @"ventricular_wall_parameters_sha256":loadedKneeNSString(cardiacWallParametersSHA256),
            @"ventricular_wall_coefficient_bundle_sha256":loadedKneeNSString(cardiacWallBundleSHA256),
            @"ventricular_wall_map_identity_receipt_sha256":loadedKneeNSString(cardiacWallIdentityReceiptSHA256),
            @"cardiac_geometry_mode":commonCardiacGeometry?@"common_seven_coordinate_v1":@"legacy_ventricular_wall_v2",
            @"common_field_source_anatomy_payload_sha256":loadedKneeNSString(commonFieldSourcePayloadSHA256),
            @"common_field_map_sha256":loadedKneeNSString(commonFieldMapSHA256),
            @"common_field_parameters_sha256":loadedKneeNSString(commonFieldParametersSHA256),
            @"common_field_polynomial_sha256":loadedKneeNSString(commonFieldPolynomialSHA256),
            @"common_field_domain_boxes_sha256":loadedKneeNSString(commonFieldBoxesSHA256),
            @"common_field_uploaded_buffer_bundle_sha256":loadedKneeNSString(commonFieldBundleSHA256),
            @"common_field_identity_receipt_sha256":loadedKneeNSString(commonFieldIdentityReceiptSHA256),
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
    struct RejectedGeometryDiagnostic {
        std::string path,fileSHA256,contentHash;
    };
    RejectedGeometryDiagnostic exportRejectedGeometryDiagnostic(unsigned step,double time,std::uint64_t root,
        std::uint64_t transaction,std::uint64_t timestamp,unsigned stableId,unsigned status,float relativeError) {
        require(vertexCaptureBuffer&&vertexCaptureBuffer.contents&&captureKernelEncoded,
            "requested rejected geometry snapshot was not captured on the render command buffer");
        metalrobo::VisualAssetPackV2 pack;std::string error;
        require(metalrobo::readVisualAssetPack(initialPackPath,pack,&error),error);
        require(pack.contentHash==initialPackContentHash&&pack.vertices.size()==renderer->layout().meshVertexCount,
            "rejected geometry base pack differs from the compiled renderer topology");
        const std::size_t vertexBytes=pack.vertices.size()*sizeof(MRVisualVertexGPUV2);
        require(vertexCaptureBuffer.length>=vertexBytes,"rejected geometry staging buffer is undersized");
        std::memcpy(pack.vertices.data(),vertexCaptureBuffer.contents,vertexBytes);
        for(const auto& vertex:pack.vertices)require(
            std::isfinite(vertex.position.x)&&std::isfinite(vertex.position.y)&&std::isfinite(vertex.position.z)&&
            std::isfinite(vertex.normalAndTangentSign.x)&&std::isfinite(vertex.normalAndTangentSign.y)&&
            std::isfinite(vertex.normalAndTangentSign.z)&&std::isfinite(vertex.tangent.x)&&
            std::isfinite(vertex.tangent.y)&&std::isfinite(vertex.tangent.z),
            "rejected geometry diagnostic contains non-finite rendered vertices");
        for(auto& primitive:pack.primitives) {
            require(primitive.geometry.x<=pack.indices.size()&&
                primitive.geometry.y<=pack.indices.size()-primitive.geometry.x,
                "rejected geometry primitive index span is invalid");
            mr_float4 lo{INFINITY,INFINITY,INFINITY,1},hi{-INFINITY,-INFINITY,-INFINITY,1};
            for(unsigned j=primitive.geometry.x;j<primitive.geometry.x+primitive.geometry.y;++j) {
                const unsigned index=pack.indices[j];require(index<pack.vertices.size(),
                    "rejected geometry pack has an out-of-range vertex index");
                const auto point=pack.vertices[index].position;
                lo.x=std::min(lo.x,point.x);lo.y=std::min(lo.y,point.y);lo.z=std::min(lo.z,point.z);
                hi.x=std::max(hi.x,point.x);hi.y=std::max(hi.y,point.y);hi.z=std::max(hi.z,point.z);
            }
            primitive.boundsMinimum=lo;primitive.boundsMaximum=hi;
        }
        std::ostringstream rootHexStream;rootHexStream<<"0x"<<std::hex<<std::setw(16)<<std::setfill('0')<<root;
        pack.preprocessingProvenance+="/rejected_surface_audit_render_step_"+std::to_string(step)+"_root_"+rootHexStream.str();
        pack.contentHash=metalrobo::computeVisualAssetPackContentHash(pack);
        const auto directory=outputDirectory/"rejected-surface-audit";
        std::filesystem::create_directories(directory);
        const auto packPath=directory/("step-"+std::to_string(step)+".mrvpack");
        require(!std::filesystem::exists(packPath),"refusing to overwrite rejected render diagnostic geometry");
        require(metalrobo::writeVisualAssetPack(pack,packPath,&error),error);
        RejectedGeometryDiagnostic result{packPath.string(),
            loadedKneeSHA256Hex(loadedKneeFileSHA256(packPath)),pack.contentHash};
        std::cout<<"rejected_surface_audit_geometry="<<result.path
            <<" stable_id="<<stableId<<" status="<<status<<" relative_volume_error="<<relativeError
            <<" accepted_step="<<step<<" accepted_time_s="<<time
            <<" root="<<rootHexStream.str()<<" pack_sha256="<<result.fileSHA256
            <<" content_hash="<<result.contentHash<<" transaction="<<transaction
            <<" timestamp_us="<<timestamp<<"\n";
        return result;
    }
    static id surfaceAuditJSONNumber(float value) {
        if(std::isfinite(value))return @(static_cast<double>(value));
        if(std::isnan(value))return @"NaN";
        return std::signbit(value)?@"-Infinity":@"+Infinity";
    }
    void writeSurfaceAuditFailureReceipt(unsigned step,double time,std::uint64_t root,
        std::uint64_t transaction,std::uint64_t timestamp,unsigned auditIndex,unsigned stableId,
        unsigned status,float relativeError,unsigned indexBufferStart,
        const MRHumanRestingSurfaceFailureGPU& firstFailure,bool captureRequested,
        const RejectedGeometryDiagnostic& diagnostic,unsigned semantic=0,
        const char* auditScope="functional_volume") {
        NSMutableDictionary* triangle=[NSMutableDictionary dictionary];
        const bool haveTriangle=firstFailure.surfaceTriangleKind.x==auditIndex&&
            firstFailure.surfaceTriangleKind.z!=MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE;
        if(haveTriangle) {
            NSMutableArray* positions=[NSMutableArray array];NSMutableArray* positionBits=[NSMutableArray array];
            for(unsigned k=0;k<3;++k) {
                const auto& p=firstFailure.renderedPositions[k];
                [positions addObject:@[surfaceAuditJSONNumber(p.x),surfaceAuditJSONNumber(p.y),surfaceAuditJSONNumber(p.z)]];
                [positionBits addObject:@[loadedKneeNSString(numiHumanRestingSurfaceAudit::float32BitsHex(p.x)),
                    loadedKneeNSString(numiHumanRestingSurfaceAudit::float32BitsHex(p.y)),
                    loadedKneeNSString(numiHumanRestingSurfaceAudit::float32BitsHex(p.z))]];
            }
            triangle[@"triangle_index_in_surface"]=@(firstFailure.surfaceTriangleKind.y);
            triangle[@"index_buffer_offset"]=@(indexBufferStart+3u*firstFailure.surfaceTriangleKind.y);
            triangle[@"area_failure_kind"]=loadedKneeNSString(
                numiHumanRestingSurfaceAudit::triangleFailureName(firstFailure.surfaceTriangleKind.z));
            triangle[@"mesh_vertex_indices"]=@[@(firstFailure.vertexIndices.x),@(firstFailure.vertexIndices.y),
                @(firstFailure.vertexIndices.z)];
            triangle[@"rendered_positions_m"]=positions;
            triangle[@"rendered_positions_f32_bits_hex"]=positionBits;
        }
        const auto fingerprintHex=[](std::uint64_t value) {
            std::ostringstream out;out<<std::hex<<std::setw(16)<<std::setfill('0')<<value;return out.str();
        };
        NSDictionary* receipt=@{
            @"schema":@"numi.human.accepted_surface_audit_failure.v1",
            @"physical_endpoint":@"accepted",@"surface_audit_endpoint":@"rejected",
            @"rejection_stage":@"post_accept_surface_presentation",
            @"accepted_step":@(step),@"accepted_time_s":@(time),
            @"accepted_root_fingerprint":@(root),@"accepted_root_fingerprint_hex":loadedKneeNSString(fingerprintHex(root)),
            @"accepted_transaction_fingerprint":@(transaction),
            @"accepted_transaction_fingerprint_hex":loadedKneeNSString(fingerprintHex(transaction)),
            @"accepted_timestamp_microseconds":@(timestamp),
            @"source_pack_content_hash":loadedKneeNSString(initialPackContentHash),
            @"source_pack_file_sha256":loadedKneeNSString(initialPackFileSHA256),
            @"source_pack_path":loadedKneeNSString(initialPackPath.string()),
            @"surface_stable_id":@(stableId),@"surface_audit_index":@(auditIndex),
            @"surface_semantic":@(semantic),@"surface_audit_scope":loadedKneeNSString(auditScope),
            @"surface_status":@(status),@"volume_relative_error":surfaceAuditJSONNumber(relativeError),
            @"volume_relative_error_applicable":@(std::strcmp(auditScope,"functional_volume")==0),
            @"volume_owner_mismatch":@((status&1u)!=0),
            @"invalid_triangle_present":@((status&2u)!=0),
            @"first_invalid_triangle_available":@(haveTriangle),
            @"first_invalid_triangle":triangle,
            @"accepted_geometry_capture_requested":@(captureRequested),
            @"accepted_geometry_capture_written":@NO,
            @"accepted_geometry_capture_skipped_reason":captureRequested?@"surface_audit_rejected":@"not_requested",
            @"rejected_geometry_diagnostic_written":@(!diagnostic.path.empty()),
            @"rejected_geometry_diagnostic_mrvpack_path":loadedKneeNSString(diagnostic.path),
            @"rejected_geometry_diagnostic_mrvpack_sha256":loadedKneeNSString(diagnostic.fileSHA256),
            @"rejected_geometry_diagnostic_content_hash":loadedKneeNSString(diagnostic.contentHash)
        };
        const auto path=outputDirectory/("surface-audit-failure-step-"+std::to_string(step)+".json");
        require(!std::filesystem::exists(path),"refusing to overwrite accepted surface-audit failure receipt");
        NSData* json=[NSJSONSerialization dataWithJSONObject:receipt
            options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
        require(json!=nil,"accepted surface-audit failure receipt serialization failed");
        NSError* error=nil;
        require([json writeToURL:[NSURL fileURLWithPath:loadedKneeNSString(path.string())]
            options:NSDataWritingAtomic error:&error],"accepted surface-audit failure receipt write failed");
        std::cout<<"accepted_surface_audit_failure_receipt="<<path.string()
            <<" physical_endpoint=accepted surface_audit_endpoint=rejected\n";
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

    struct ViewerTimedEncoderContext {
        __strong id<MTLCommandBuffer> command = nil;
        __strong id<MTLComputeCommandEncoder> encoder = nil;
        __strong id<MTLCounterSampleBuffer> samples = nil;
        __strong id<MTLFence> fence = nil;
        __strong id<MTLHeap> heap = nil;
        __strong NSString* label = nil;
        unsigned stage = 0u;
    };

    static id<MTLCounterSampleBuffer> makeViewerGpuTimingSamples(id<MTLDevice> device) {
        if(!device||![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])return nil;
        for(id<MTLCounterSet> set in device.counterSets) {
            if(![set.name isEqualToString:MTLCommonCounterSetTimestamp])continue;
            MTLCounterSampleBufferDescriptor* descriptor=[MTLCounterSampleBufferDescriptor new];
            descriptor.counterSet=set;descriptor.storageMode=MTLStorageModeShared;descriptor.sampleCount=8u;
            return [device newCounterSampleBufferWithDescriptor:descriptor error:nil];
        }
        return nil;
    }

    static bool beginViewerTimedEncoder(ViewerTimedEncoderContext& context) {
        if(context.command==nil||context.samples==nil||context.fence==nil||context.stage>=4u)return false;
        MTLComputePassDescriptor* pass=[MTLComputePassDescriptor computePassDescriptor];
        pass.sampleBufferAttachments[0].sampleBuffer=context.samples;
        pass.sampleBufferAttachments[0].startOfEncoderSampleIndex=context.stage*2u;
        pass.sampleBufferAttachments[0].endOfEncoderSampleIndex=context.stage*2u+1u;
        context.encoder=[context.command computeCommandEncoderWithDescriptor:pass];
        if(context.encoder==nil)return false;
        if(context.stage>0u)[context.encoder waitForFence:context.fence];
        context.encoder.label=context.label;
        if(context.heap!=nil)[context.encoder useHeap:context.heap];
        return true;
    }

    static ViewerTimedEncoderContext& viewerTimedContext(void* opaque) {
        return *static_cast<ViewerTimedEncoderContext*>(opaque);
    }
    static void viewerTimedSetLabel(void* opaque,const char* label) {
        auto& context=viewerTimedContext(opaque);
        context.label=[NSString stringWithUTF8String:label];
        context.encoder.label=context.label;
    }
    static void viewerTimedUseHeap(void* opaque,void* heap) {
        auto& context=viewerTimedContext(opaque);
        context.heap=(__bridge id<MTLHeap>)heap;
        [context.encoder useHeap:context.heap];
    }
    static void viewerTimedSetPipeline(void* opaque,void* pipeline) {
        [viewerTimedContext(opaque).encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)pipeline];
    }
    static void viewerTimedSetBuffer(void* opaque,void* buffer,std::size_t offset,std::uint32_t index) {
        [viewerTimedContext(opaque).encoder setBuffer:(__bridge id<MTLBuffer>)buffer offset:offset atIndex:index];
    }
    static void viewerTimedSetBytes(void* opaque,const void* bytes,std::size_t length,std::uint32_t index) {
        [viewerTimedContext(opaque).encoder setBytes:bytes length:length atIndex:index];
    }
    static void viewerTimedDispatchThreads(void* opaque,std::size_t count,std::size_t perGroup) {
        [viewerTimedContext(opaque).encoder dispatchThreads:MTLSizeMake(count,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(perGroup,1u,1u)];
    }
    static void viewerTimedDispatchThreadgroups(void* opaque,std::size_t count,std::size_t perGroup) {
        [viewerTimedContext(opaque).encoder dispatchThreadgroups:MTLSizeMake(count,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(perGroup,1u,1u)];
    }
    static void viewerTimedDispatchIndirect(void* opaque,void* arguments,std::size_t offset,std::size_t perGroup) {
        [viewerTimedContext(opaque).encoder dispatchThreadgroupsWithIndirectBuffer:(__bridge id<MTLBuffer>)arguments
            indirectBufferOffset:offset threadsPerThreadgroup:MTLSizeMake(perGroup,1u,1u)];
    }
    static bool viewerTimedSplitEncoder(void* opaque) {
        auto& context=viewerTimedContext(opaque);
        if(context.encoder==nil||context.fence==nil||context.stage>=3u)return false;
        [context.encoder updateFence:context.fence];
        [context.encoder endEncoding];context.encoder=nil;++context.stage;
        switch(context.stage) {
            case 1u:context.label=@"Numi Human functional volume audit";break;
            case 2u:context.label=@"Numi Human whole mesh and secondary audits";break;
            case 3u:context.label=@"Numi Human mesh renderer";break;
            default:return false;
        }
        return beginViewerTimedEncoder(context);
    }
    static metalrobo::MetalHybridComputeEncoderCallbacks viewerTimedCallbacks(
        ViewerTimedEncoderContext& context) {
        metalrobo::MetalHybridComputeEncoderCallbacks callbacks;
        callbacks.context=&context;
        callbacks.setLabel=&viewerTimedSetLabel;
        callbacks.useHeap=&viewerTimedUseHeap;
        callbacks.setPipeline=&viewerTimedSetPipeline;
        callbacks.setBuffer=&viewerTimedSetBuffer;
        callbacks.setBytes=&viewerTimedSetBytes;
        callbacks.dispatchThreads=&viewerTimedDispatchThreads;
        callbacks.dispatchThreadgroups=&viewerTimedDispatchThreadgroups;
        callbacks.dispatchThreadgroupsIndirect=&viewerTimedDispatchIndirect;
        callbacks.splitCommandEncoder=&viewerTimedSplitEncoder;
        return callbacks;
    }

    static void reportViewerGpuTiming(id<MTLCounterSampleBuffer> samples,unsigned step) {
        if(!samples)return;
        NSData* data=[samples resolveCounterRange:NSMakeRange(0u,8u)];
        if(data.length!=8u*sizeof(MTLCounterResultTimestamp)) {
            std::fprintf(stderr,"resting_viewer_gpu_timing sampling=unavailable step=%u\n",step);return;
        }
        const auto* stamps=static_cast<const MTLCounterResultTimestamp*>(data.bytes);
        for(unsigned i=0;i<8u;++i)if(stamps[i].timestamp==0u||stamps[i].timestamp==MTLCounterErrorValue) {
            std::fprintf(stderr,"resting_viewer_gpu_timing sampling=unavailable step=%u\n",step);return;
        }
        for(unsigned i=1;i<8u;++i)if(stamps[i].timestamp<stamps[i-1u].timestamp) {
            std::fprintf(stderr,"resting_viewer_gpu_timing sampling=unavailable step=%u\n",step);return;
        }
        const auto deformation=stamps[1].timestamp-stamps[0].timestamp;
        const auto volumeAudit=stamps[3].timestamp-stamps[2].timestamp;
        const auto meshAudit=stamps[5].timestamp-stamps[4].timestamp;
        const auto renderer=stamps[7].timestamp-stamps[6].timestamp;
        const auto gaps=(stamps[2].timestamp-stamps[1].timestamp)+
            (stamps[4].timestamp-stamps[3].timestamp)+
            (stamps[6].timestamp-stamps[5].timestamp);
        std::fprintf(stderr,
            "resting_viewer_gpu_timing step=%u deformation_ns=%llu functional_volume_audit_ns=%llu whole_mesh_audit_ns=%llu renderer_ns=%llu encoder_gaps_ns=%llu total_ns=%llu\n",
            step,static_cast<unsigned long long>(deformation),
            static_cast<unsigned long long>(volumeAudit),static_cast<unsigned long long>(meshAudit),
            static_cast<unsigned long long>(renderer),static_cast<unsigned long long>(gaps),
            static_cast<unsigned long long>(stamps[7].timestamp-stamps[0].timestamp));
    }

    NumiHumanRestingVisual(NumiHumanRestingCoupling& owner,metalrobo::VisualAssetPackV2 pack,
        const metalrobo::EngineModel& model,const LoadedSkin& skin,const LoadedSoftTissues* tissues,
        const std::vector<MRBodyStateGPU>& initialBodies,const std::vector<MRBodyStateGPU>& restBodies,
        const LoadedSupportContacts& support,const NumiHumanRestingAnatomy& functional,
        const std::filesystem::path& output,unsigned size,const std::string& movie,bool presentWindow=true):
        coupled(owner),dimension(size),outputDirectory(output),acceptedGeometryDirectory(output/"accepted-geometry"),
        surfaceTrace(output/"resting-surface-audit.csv") {
        require(surfaceTrace.good(),"resting surface audit output unavailable");
        const char* profileSetting=std::getenv("NUMI_HUMAN_TRAINING_PROFILE");
        require(!profileSetting||!profileSetting[0]||std::strcmp(profileSetting,"0")==0||
            std::strcmp(profileSetting,"1")==0,"NUMI_HUMAN_TRAINING_PROFILE must be 0 or 1");
        profileTiming=profileSetting&&std::strcmp(profileSetting,"1")==0;
        const char* gpuTimingSetting=std::getenv("NUMI_HUMAN_GPU_TIMING");
        require(!gpuTimingSetting||!gpuTimingSetting[0]||std::strcmp(gpuTimingSetting,"0")==0||
            std::strcmp(gpuTimingSetting,"1")==0,"NUMI_HUMAN_GPU_TIMING must be 0 or 1");
        profileGpuTiming=gpuTimingSetting&&std::strcmp(gpuTimingSetting,"1")==0;
        commonCardiacGeometry=functional.commonCardiacGeometry;
        commonFieldVertexCount=unsigned(functional.commonFieldMap.size());
        commonFieldSourcePayloadSHA256=functional.commonFieldAnatomyPayloadSHA256;
        requestedGeometrySteps=geometryExportStepsFromEnvironment();
        require(requestedGeometrySteps.empty()||presentWindow,
            "accepted MRVPack export requires the native viewer path");
        surfaceTrace<<"time_s,step,min_skin_bed_gap_m,vertices_below_1mm,nonfinite_skin_vertices,max_functional_volume_relative_error,q_ra,q_rv,q_la,q_lv,ra_target_ml,rv_target_ml,la_target_ml,lv_target_ml,diaphragm_swept_ml,rib_swept_ml,lung_target_ml,body_com_x_m,body_com_y_m,body_com_z_m,represented_body_mass_kg,ventricular_wall_bound,ventricular_material_ml,ventricular_material_target_ml,ventricular_closure_mm,ventricular_material_status,functional_geometry_status,geometry_mode,common_coordinate_RA,common_coordinate_RV,common_coordinate_LA,common_coordinate_LV,common_coordinate_RA_material,common_coordinate_ventricular_material,common_coordinate_LA_material,common_coordinate_solver_status,common_coordinate_solver_iterations,common_coordinate_domain_box,common_coordinate_normalized_residual,ventricular_closure_mm_applicable,mesh_zero_area_triangles,mesh_nonfinite_area_triangles,mesh_triangles_checked\n";
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
        std::vector<mr_uint4> wallNormalRanges;
        std::vector<unsigned> wallIncidentTriangles;
        std::vector<mr_uint4> airwayPointNormalRanges;
        std::vector<unsigned> airwayPointIncidentTriangles;
        std::vector<unsigned> visibleLayers;
        std::set<unsigned> commonAttachmentsSeen;
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
        // Source-registration evidence is computed once at load. These bounds
        // and poses describe the authoritative neutral native assembly; they
        // neither change attachment ownership nor advance a simulation state.
        const std::set<unsigned> upperVisceralAnchors{2,3,4,5,13,14,15,16,17,18,19,20,21,22,461};
        const std::set<unsigned> pelvicVisceralAnchors{462,463};
        NSMutableArray* registrationRows=[NSMutableArray array];
        registrationBodyIndices.insert(anatomyGPU.bodyAndFlags.x);
        if(!functional.passiveViscera.empty())registrationBodyIndices.insert(functional.passivePelvicBody);
        for(const auto& instance:pack.instances)if(instance.identity.x==kOrganSurfaceSemantic&&
            (upperVisceralAnchors.contains(instance.identity.w)||pelvicVisceralAnchors.contains(instance.identity.w))) {
            require(instance.binding.z==MR_VISUAL_BINDING_ARTICULATED_LINK&&instance.binding.y<initialBodies.size(),
                "passive source anatomy lacks an initial native body transform");
            registrationBodyIndices.insert(instance.binding.y);
            mr_float4 lo{INFINITY,INFINITY,INFINITY,0},hi{-INFINITY,-INFINITY,-INFINITY,0};
            for(unsigned p=instance.geometry.x;p<instance.geometry.x+instance.geometry.y;++p) {
                const auto& primitive=pack.primitives.at(p);
                for(unsigned j=primitive.geometry.x;j<primitive.geometry.x+primitive.geometry.y;++j) {
                    const auto v=torsoLocal(instance,pack.vertices.at(pack.indices.at(j)).position);
                    lo.x=std::min(lo.x,v.x);lo.y=std::min(lo.y,v.y);lo.z=std::min(lo.z,v.z);
                    hi.x=std::max(hi.x,v.x);hi.y=std::max(hi.y,v.y);hi.z=std::max(hi.z,v.z);
                }
            }
            [registrationRows addObject:@{@"stable_id":@(instance.identity.w),@"semantic":@(instance.identity.x),
                @"source_body_index":@(instance.binding.y),
                @"role":upperVisceralAnchors.contains(instance.identity.w)?@"upper_visceral_anchor":@"pelvic_anchor",
                @"torso_local_min_m":@[@(lo.x),@(lo.y),@(lo.z)],
                @"torso_local_max_m":@[@(hi.x),@(hi.y),@(hi.z)]}];
        }
        NSMutableArray* initialPoses=[NSMutableArray array];
        for(unsigned index:registrationBodyIndices)[initialPoses addObject:bodyPose(index,initialBodies.at(index))];
        initialAnatomicalRegistration=@{@"torso_body_index":@(anatomyGPU.bodyAndFlags.x),
            @"coordinate_source":@"source pack transformed by native initialBodies before physical stepping",
            @"body_poses":initialPoses,@"surfaces":registrationRows};
        NSData* registrationJSON=[NSJSONSerialization dataWithJSONObject:initialAnatomicalRegistration
            options:NSJSONWritingSortedKeys error:nil];
        require(registrationJSON!=nil,"initial anatomical registration serialization failed");
        std::cout<<"resting_initial_anatomical_registration="
            <<[[[NSString alloc] initWithData:registrationJSON encoding:NSUTF8StringEncoding] UTF8String]<<"\n";
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
        // ID31 is a bronchopulmonary segment surface that meets ID32 at a
        // source-derived contact neighborhood. Keep the first 2 mm fixed to
        // that registered neighbor, then blend smoothly into the same GPU
        // respiratory map used by the lung surfaces over the next 8 mm.
        // The distances are computed once at scene load in their shared
        // articulated body-20 source frame; no per-step CPU work is added.
        const MRVisualInstanceGPUV2* airway31Instance=nullptr;
        const MRVisualInstanceGPUV2* airway32Instance=nullptr;
        for(const auto& candidate:pack.instances)if(candidate.identity.x==kAirwaySurfaceSemantic) {
            if(candidate.identity.w==31) {
                require(!airway31Instance,"airway ID31 has multiple visual instances");
                airway31Instance=&candidate;
            }else if(candidate.identity.w==32) {
                require(!airway32Instance,"airway ID32 has multiple visual instances");
                airway32Instance=&candidate;
            }
        }
        std::vector<float> airway31RespiratoryWeight(pack.vertices.size(),1.0f);
        if(airway31Instance||airway32Instance) {
            require(airway31Instance&&airway32Instance,
                "source-derived ID31 respiratory attachment requires both airway surfaces 31 and 32");
            for(const auto* airway:{airway31Instance,airway32Instance})
                require(airway->binding.z==MR_VISUAL_BINDING_ARTICULATED_LINK&&
                    airway->binding.y==anatomyGPU.bodyAndFlags.x,
                    "ID31/ID32 source attachment surfaces must share the articulated torso owner");
            auto airwaySourcePoint=[&](const MRVisualInstanceGPUV2& instance,unsigned vertexIndex) {
                const auto& vertex=pack.vertices.at(vertexIndex);
                return addPoint(instance.translationAndScale,
                    scalePoint(rotatePoint(instance.orientation,vertex.position),instance.translationAndScale.w));
            };
            std::vector<std::array<mr_float4,3>> airway32Triangles;
            std::set<unsigned> airway31VertexIndices;
            for(unsigned p=airway32Instance->geometry.x;
                p<airway32Instance->geometry.x+airway32Instance->geometry.y;++p) {
                const auto& primitive=pack.primitives.at(p);
                require(primitive.geometry.y%3==0,"airway ID32 surface indices are not triangles");
                for(unsigned j=primitive.geometry.x;j<primitive.geometry.x+primitive.geometry.y;j+=3) {
                    std::array<mr_float4,3> triangle{};
                    for(unsigned k=0;k<3;++k)
                        triangle[k]=airwaySourcePoint(*airway32Instance,pack.indices.at(j+k));
                    airway32Triangles.push_back(triangle);
                }
            }
            for(unsigned p=airway31Instance->geometry.x;
                p<airway31Instance->geometry.x+airway31Instance->geometry.y;++p) {
                const auto& primitive=pack.primitives.at(p);
                for(unsigned j=primitive.geometry.x;j<primitive.geometry.x+primitive.geometry.y;++j)
                    airway31VertexIndices.insert(pack.indices.at(j));
            }
            require(!airway32Triangles.empty()&&!airway31VertexIndices.empty(),
                "source-derived airway attachment has no triangles or ID31 vertices");
            std::vector<std::byte> airwayMaskDigestBytes;
            std::vector<std::tuple<unsigned,unsigned,float,float,mr_float4,mr_float4>> airwayMaskRows;
            unsigned airwayHeldCount=0,airwayBlendCount=0,airwayFullCount=0;
            float airwayMinDistance=INFINITY,airwayMaxDistance=0.0f;
            for(unsigned vertexIndex:airway31VertexIndices) {
                const auto point=airwaySourcePoint(*airway31Instance,vertexIndex);
                float bestSquared=INFINITY;unsigned bestTriangle=MR_INVALID_INDEX;mr_float4 bestPoint{};
                for(unsigned triangleIndex=0;triangleIndex<airway32Triangles.size();++triangleIndex) {
                    const auto& triangle=airway32Triangles[triangleIndex];
                    const auto nearest=closestPointOnTriangle(point,triangle[0],triangle[1],triangle[2]);
                    const auto delta=subtractPoint(point,nearest);
                    const float distanceSquared=dotPoint(delta,delta);
                    if(distanceSquared<bestSquared) {
                        bestSquared=distanceSquared;bestTriangle=triangleIndex;bestPoint=nearest;
                    }
                }
                require(std::isfinite(bestSquared)&&bestTriangle!=MR_INVALID_INDEX,
                    "ID31 source vertex has no finite nearest ID32 triangle");
                const float distance=std::sqrt(std::max(0.0f,bestSquared));
                const float t=std::clamp((distance-.002f)/.008f,0.0f,1.0f);
                const float smooth=t*t*(3.0f-2.0f*t);
                airway31RespiratoryWeight[vertexIndex]=smooth;
                airwayMinDistance=std::min(airwayMinDistance,distance);
                airwayMaxDistance=std::max(airwayMaxDistance,distance);
                if(distance<=.002f)++airwayHeldCount;
                else if(distance<.010f)++airwayBlendCount;
                else ++airwayFullCount;
                airwayMaskRows.emplace_back(vertexIndex,bestTriangle,distance,smooth,point,bestPoint);
                appendLoadedKneeScalar(airwayMaskDigestBytes,vertexIndex);
                appendLoadedKneeScalar(airwayMaskDigestBytes,distance);
                appendLoadedKneeScalar(airwayMaskDigestBytes,smooth);
                appendLoadedKneeScalar(airwayMaskDigestBytes,bestTriangle);
                for(const float value:{point.x,point.y,point.z,bestPoint.x,bestPoint.y,bestPoint.z})
                    appendLoadedKneeScalar(airwayMaskDigestBytes,value);
            }
            const auto airwayMaskSHA=loadedKneeSHA256Hex(
                loadedKneeSHA256(airwayMaskDigestBytes.data(),airwayMaskDigestBytes.size()));
            const auto airwayMapPath=output/"resting-airway-anchor-map.json";
            require(!std::filesystem::exists(airwayMapPath),
                "refusing to overwrite a prior source-derived airway anchor receipt");
            std::ofstream airwayMapFile(airwayMapPath);
            require(airwayMapFile.good(),"source-derived airway anchor receipt is not writable");
            airwayMapFile<<std::setprecision(9)
                <<"{\n  \"schema\": \"resting-airway-anchor-map.v1\",\n"
                <<"  \"source_pack_content_hash\": \""<<pack.contentHash<<"\",\n"
                <<"  \"semantic\": "<<kAirwaySurfaceSemantic<<", \"stable_id\": 31, \"neighbor_stable_id\": 32,\n"
                <<"  \"body_index\": "<<anatomyGPU.bodyAndFlags.x<<", \"coordinate_frame\": \"shared articulated body-20 source frame\",\n"
                <<"  \"mask_rule\": \"0 for distance <= 0.002 m; smoothstep((d-0.002)/0.008) for 0.002 < d < 0.010 m; 1 for d >= 0.010 m\",\n"
                <<"  \"id32_triangle_count\": "<<airway32Triangles.size()<<", \"id31_vertex_count\": "<<airwayMaskRows.size()<<",\n"
                <<"  \"held_at_or_below_2mm\": "<<airwayHeldCount<<", \"blended_2_to_10mm\": "<<airwayBlendCount
                <<", \"full_map_at_or_above_10mm\": "<<airwayFullCount<<",\n"
                <<"  \"min_distance_m\": "<<airwayMinDistance<<", \"max_distance_m\": "<<airwayMaxDistance
                <<", \"mask_inputs_sha256\": \""<<airwayMaskSHA<<"\",\n  \"vertices\": [\n";
            for(std::size_t row=0;row<airwayMaskRows.size();++row) {
                const auto& [index,triangleIndex,distance,weight,point,nearest]=airwayMaskRows[row];
                airwayMapFile<<"    {\"pack_vertex\": "<<index<<", \"nearest_id32_triangle\": "<<triangleIndex
                    <<", \"distance_m\": "<<distance<<", \"respiratory_weight\": "<<weight
                    <<", \"source_point_m\": ["
                    <<point.x<<", "<<point.y<<", "<<point.z<<"], \"nearest_id32_point_m\": ["
                    <<nearest.x<<", "<<nearest.y<<", "<<nearest.z<<"]}"
                    <<(row+1==airwayMaskRows.size()?"\n":",\n");
            }
            airwayMapFile<<"  ]\n}\n";
            require(airwayMapFile.good(),"source-derived airway anchor receipt write failed");
            std::cout<<"resting_airway_anchor stable_id=31 neighbor_stable_id=32 body="
                <<anatomyGPU.bodyAndFlags.x<<" vertices="<<airwayMaskRows.size()
                <<" triangles="<<airway32Triangles.size()<<" held_le_2mm="<<airwayHeldCount
                <<" blend_2_10mm="<<airwayBlendCount<<" full_ge_10mm="<<airwayFullCount
                <<" distance_m=["<<airwayMinDistance<<","<<airwayMaxDistance<<"]"
                <<" map_sha256="<<airwayMaskSHA<<" receipt="<<airwayMapPath.string()<<"\n";
        }
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
            if(instance.identity.x==kAirwaySurfaceSemantic&&instance.identity.w==31)deformation=11;
            if(instance.identity.x>=kOrganSurfaceSemantic) {
                if(functional.lungs.contains(instance.identity.w)||functional.pleura.contains(instance.identity.w))deformation=1;
                for(unsigned c=0;c<4;++c)if(instance.identity.w==functional.cavities[c]){deformation=2;chamber=c;}
                if(functional.cardiacWallSurfaces.contains(instance.identity.w))deformation=8;
                if(functional.diaphragm.contains(instance.identity.w))deformation=4;
                if(functional.intercostals.contains(instance.identity.w))deformation=7;
            }
            if(instance.identity.x==kOrganSurfaceSemantic&&functional.passiveViscera.contains(instance.identity.w))
                deformation=9;
            if(instance.identity.x==kOrganSurfaceSemantic&&instance.identity.w==functional.ventricularWallStableId)
                deformation=10;
            if(instance.identity.x==kBoneSemantic&&functional.ribs.contains(instance.identity.w)) {
                deformation=5;chamber=functional.ribs.at(instance.identity.w);
            }
            if(instance.identity.x==kBoneSemantic&&functional.sternum.contains(instance.identity.w))deformation=6;
            unsigned commonChannel=MR_INVALID_INDEX,commonAttachmentIndex=MR_INVALID_INDEX;
            if(commonCardiacGeometry)for(unsigned c=0;c<functional.commonFieldStableIDs.size();++c) {
                const unsigned expectedSemantic=c<4?kCavityReferenceSemantic:kOrganSurfaceSemantic;
                if(instance.identity.w==functional.commonFieldStableIDs[c]&&
                    instance.identity.x==expectedSemantic&&
                    instance.binding.z==MR_VISUAL_BINDING_ARTICULATED_LINK&&
                    instance.binding.y==anatomyGPU.bodyAndFlags.x) {
                    commonChannel=c;break;
                }
            }
            if(commonCardiacGeometry&&commonChannel==MR_INVALID_INDEX)
                for(unsigned a=0;a<functional.commonFieldPassiveAttachments.size();++a) {
                    const auto& range=functional.commonFieldPassiveAttachments[a];
                    if(instance.identity.w==range.stableId&&instance.identity.x==range.semantic) {
                        commonAttachmentIndex=a;break;
                    }
                }
            if(commonChannel!=MR_INVALID_INDEX){deformation=12;chamber=commonChannel;}
            else if(commonAttachmentIndex!=MR_INVALID_INDEX){deformation=12;chamber=7+commonAttachmentIndex;}
            const auto semantic=instance.identity.x;
            unsigned visibility=semantic==kSkinShellSemantic?1u:
                semantic==kBoneSemantic?6u:
                (semantic==kMuscleSurfaceSemantic||semantic==kTendonSurfaceSemantic)?2u:
                semantic>=kOrganSurfaceSemantic?8u:0u;
            if(deformation==1&&functional.lungs.contains(instance.identity.w))visibility|=16u;
            if(deformation==2||(deformation==12&&commonChannel<4))visibility|=32u|64u;
            if(deformation==12&&commonChannel>=4&&commonChannel<7)visibility|=32u;
            if(deformation==8||deformation==10)visibility|=32u;
            if(semantic==kVesselSurfaceSemantic||semantic==kPulmonaryArterySurfaceSemantic||semantic==kPulmonaryVeinSurfaceSemantic)
                visibility|=64u;
            if(deformation==4||deformation==7)visibility=2u|16u;
            visibleLayers.push_back(visibility);
            if(deformation==2||(deformation==12&&commonChannel<4))for(unsigned p=instance.geometry.x;p<instance.geometry.x+instance.geometry.y;++p)
                pack.primitives.at(p).geometry.z=cardiacMaterials.at(chamber);
            if(deformation==11&&instance.identity.w==31) {
                require(airwayNormalVertexCount==0&&instance.geometry.y==1&&vertices.size()>0&&
                    vertices.back()-base+1==vertices.size(),
                    "ID31 normal reconstruction requires one contiguous source surface");
                const auto& primitive=pack.primitives.at(instance.geometry.x);
                require(primitive.geometry.y%3==0,"ID31 normal source has a partial triangle");
                std::vector<std::vector<unsigned>> incident(vertices.size());
                for(unsigned j=primitive.geometry.x;j<primitive.geometry.x+primitive.geometry.y;j+=3)
                    for(unsigned k=0;k<3;++k) {
                        const unsigned v=pack.indices.at(j+k);
                        require(v>=base&&v-base<vertices.size(),"ID31 normal triangle escapes its source vertices");
                        incident[v-base].push_back(j);
                    }
                for(unsigned local=0;local<vertices.size();++local) {
                    require(!incident[local].empty(),"ID31 respiratory surface has an unreferenced source vertex");
                    airwayPointNormalRanges.push_back({base+local,unsigned(airwayPointIncidentTriangles.size()),
                        unsigned(incident[local].size()),0});
                    airwayPointIncidentTriangles.insert(airwayPointIncidentTriangles.end(),
                        incident[local].begin(),incident[local].end());
                }
                airwayNormalVertexCount=unsigned(vertices.size());
            }
            if(deformation==12) {
                const bool volumeOwner=commonChannel<7;
                require(commonCardiacGeometry&&(volumeOwner||commonAttachmentIndex<functional.commonFieldPassiveAttachments.size())&&
                    instance.geometry.y==1,
                    "common field surface has an invalid cardiac channel or passive attachment range");
                const auto range=functional.commonFieldRanges.at(instance.identity.w);
                std::cout<<"common_cardiac_source_binding stable_id="<<instance.identity.w
                    <<" map_first="<<range.first<<" map_count="<<range.second
                    <<" indexed_unique_count="<<vertices.size()<<" source_base="<<base
                    <<" indexed_last="<<vertices.back()
                    <<" binding_kind="<<instance.binding.z<<" binding_body="<<instance.binding.y
                    <<" expected_body="<<anatomyGPU.bodyAndFlags.x
                    <<" primitive_count="<<instance.geometry.y<<"\n";
                require(range.second==vertices.size()&&vertices.back()-base+1==range.second&&
                    instance.binding.z==MR_VISUAL_BINDING_ARTICULATED_LINK&&
                    instance.binding.y==anatomyGPU.bodyAndFlags.x&&
                    instance.translationAndScale.x==0&&instance.translationAndScale.y==0&&
                    instance.translationAndScale.z==0&&instance.translationAndScale.w==1&&
                    instance.orientation.x==0&&instance.orientation.y==0&&instance.orientation.z==0&&instance.orientation.w==1,
                    "common field surfaces must preserve the contiguous registered torso source frame");
                require(range.first<=functional.commonFieldMap.size()&&range.second<=functional.commonFieldMap.size()-range.first,
                    "common field surface range escapes its coefficient map");
                const auto& primitive=pack.primitives.at(instance.geometry.x);
                require(primitive.geometry.y%3==0,"common field surface contains a partial triangle");
                if(volumeOwner) {
                    require(commonFieldAuditIndices[chamber]==MR_INVALID_INDEX,
                        "common cardiac volume channel is duplicated in the renderer pack");
                    const float reference=chamber<4?
                        reinterpret_cast<const float*>(functional.commonFieldGPU.sourceReferenceVolumes)[chamber]:
                        reinterpret_cast<const float*>(&functional.commonFieldGPU.materialTargetVolumes)[chamber-4];
                    commonFieldAuditIndices[chamber]=unsigned(audits.size());
                    audits.push_back({{primitive.geometry.x,primitive.geometry.y,deformation,chamber},{reference,0,0,0}});
                    auditStableIds.push_back(instance.identity.w);
                } else {
                    const auto& attachment=functional.commonFieldPassiveAttachments.at(commonAttachmentIndex);
                    require(instance.identity.w==attachment.stableId&&instance.identity.x==attachment.semantic&&
                        attachment.bodyIndex==anatomyGPU.bodyAndFlags.x&&commonAttachmentsSeen.insert(attachment.stableId).second,
                        "common passive attachment does not match its unique typed source range");
                }
                std::vector<std::vector<unsigned>> incident(range.second);
                for(unsigned j=primitive.geometry.x;j<primitive.geometry.x+primitive.geometry.y;j+=3)
                    for(unsigned k=0;k<3;++k) {
                        const unsigned v=pack.indices.at(j+k);
                        require(v>=base&&v-base<range.second,"common cardiac surface triangle vertex is outside its source range");
                        incident[v-base].push_back(j);
                    }
                for(unsigned local=0;local<range.second;++local) {
                    require(!incident[local].empty(),"common cardiac map has a source vertex unused by its triangles");
                    wallNormalRanges.push_back({base+local,unsigned(wallIncidentTriangles.size()),unsigned(incident[local].size()),0});
                    wallIncidentTriangles.insert(wallIncidentTriangles.end(),incident[local].begin(),incident[local].end());
                }
                commonFieldNormalVertexCount+=range.second;
            }
            if(deformation==10) {
                require(cardiacWallVertexCount==0&&instance.geometry.y==1,
                    "ventricular material wall requires one source surface with one owner");
                require(instance.binding.z==MR_VISUAL_BINDING_ARTICULATED_LINK&&
                    instance.binding.y==anatomyGPU.bodyAndFlags.x&&
                    instance.translationAndScale.x==0&&instance.translationAndScale.y==0&&
                    instance.translationAndScale.z==0&&instance.translationAndScale.w==1&&
                    instance.orientation.x==0&&instance.orientation.y==0&&instance.orientation.z==0&&instance.orientation.w==1,
                    "ventricular wall coefficients require the registered torso source frame");
                cardiacWallVertexCount=unsigned(functional.ventricularWallMap.size());
                require(cardiacWallVertexCount>0&&vertices.size()==cardiacWallVertexCount&&
                    vertices.back()-base+1==cardiacWallVertexCount,
                    "ventricular wall pack vertex order differs from its source coefficient map");
                const auto& p=pack.primitives.at(instance.geometry.x);
                require(p.geometry.y%3==0,"ventricular material boundary must contain triangles");
                cardiacWallAuditIndex=unsigned(audits.size());
                audits.push_back({{p.geometry.x,p.geometry.y,deformation,0},
                    {functional.ventricularWallGPU.scalesAndVolume.w,0,0,0}});
                auditStableIds.push_back(instance.identity.w);
                std::vector<std::vector<unsigned>> incident(cardiacWallVertexCount);
                for(unsigned j=p.geometry.x;j<p.geometry.x+p.geometry.y;j+=3)
                    for(unsigned k=0;k<3;++k) {
                        const unsigned v=pack.indices.at(j+k);
                        require(v>=base&&v-base<cardiacWallVertexCount,"ventricular material boundary index is invalid");
                        incident[v-base].push_back(j);
                    }
                for(unsigned i=0;i<cardiacWallVertexCount;++i) {
                    require(!incident[i].empty(),"ventricular material wall has an unreferenced vertex");
                    wallNormalRanges.push_back({base+i,unsigned(wallIncidentTriangles.size()),unsigned(incident[i].size()),0});
                    wallIncidentTriangles.insert(wallIncidentTriangles.end(),incident[i].begin(),incident[i].end());
                }
            }else if(deformation!=12&&functional.enclosedVolumes.contains(instance.identity.w)&&deformation) {
                require(instance.geometry.y==1,"functional anatomy audit requires one contiguous source surface");
                const auto& p=pack.primitives.at(instance.geometry.x);
                audits.push_back({{p.geometry.x,p.geometry.y,deformation,chamber},
                    {functional.enclosedVolumes.at(instance.identity.w),
                     deformation==1?functional.respiratorySweptAreas.at(instance.identity.w):0,0,0}});
                auditStableIds.push_back(instance.identity.w);
            }
            const SoftTissueRecord* tissue=nullptr;
            if(tissues&&(instance.identity.x==kMuscleSurfaceSemantic||instance.identity.x==kTendonSurfaceSemantic)) {
                auto found=std::find_if(tissues->records.begin(),tissues->records.end(),[&](const auto& t){return t.stableId==instance.identity.w;});
                require(found!=tissues->records.end(),"resting tissue source identity missing");tissue=&*found;
                // appendSoftTissueGeometry preserves the complete source vertex
                // allocation, including unused leading vertices. The first
                // referenced mesh vertex therefore need not be source vertex 0.
                // Recover that allocation base from the exact source index
                // correspondence before reading deformation bindings.
                require(instance.geometry.y==1,"resting tissue requires one source primitive");
                const auto& primitive=pack.primitives.at(instance.geometry.x);
                require(primitive.geometry.y==tissue->indexCount&&tissue->indexCount>0,
                    "resting tissue source index count differs from its native primitive");
                const unsigned sourceFirst=tissues->indices.at(tissue->firstIndex);
                require(sourceFirst>=tissue->firstVertex&&
                    sourceFirst-tissue->firstVertex<tissue->vertexCount,
                    "resting tissue first source index is outside its vertex allocation");
                const unsigned sourceOffset=sourceFirst-tissue->firstVertex;
                const unsigned meshFirst=pack.indices.at(primitive.geometry.x);
                require(meshFirst>=sourceOffset,"resting tissue source vertex base underflows");
                const unsigned sourceBase=meshFirst-sourceOffset;
                require(sourceBase<=pack.vertices.size()&&
                    tissue->vertexCount<=pack.vertices.size()-sourceBase,
                    "resting tissue source vertex allocation exceeds the native pack");
                for(unsigned j=0;j<tissue->indexCount;++j) {
                    const unsigned sourceIndex=tissues->indices.at(tissue->firstIndex+j);
                    require(sourceIndex>=tissue->firstVertex&&
                        sourceIndex-tissue->firstVertex<tissue->vertexCount&&
                        pack.indices.at(primitive.geometry.x+j)==
                            sourceBase+(sourceIndex-tissue->firstVertex),
                        "resting tissue native topology differs from its source vertex order");
                }
                if(sourceBase!=base)
                    std::cout<<"resting_soft_tissue_source_range stable_id="<<tissue->stableId
                        <<" pack_source_base="<<sourceBase<<" first_referenced_vertex="<<base
                        <<" unused_leading_vertices="<<(base-sourceBase)<<"\n";
                base=sourceBase;
            }
            for(unsigned v:vertices) {
                require(!maps.at(v).influenceCount,"resting anatomy vertices have multiple owners");
                maps[v].deformationKind=deformation;maps[v].chamberIndex=chamber;
                if(deformation==11) {
                    require(v<airway31RespiratoryWeight.size()&&std::isfinite(airway31RespiratoryWeight[v]),
                        "airway ID31 respiratory mask is missing a source vertex");
                    maps[v].deformationWeight.x=airway31RespiratoryWeight[v];
                }else if(deformation==10) {
                    maps[v].chamberIndex=v-base;
                }else if(deformation==12) {
                    const auto range=functional.commonFieldRanges.at(instance.identity.w);
                    maps[v].chamberIndex=range.first+(v-base);
                }else if(deformation==2) {
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
                }else if(deformation==9) {
                    const auto& vertex=pack.vertices.at(v);
                    require(instance.binding.z==MR_VISUAL_BINDING_ARTICULATED_LINK&&
                        instance.binding.y==anatomyGPU.bodyAndFlags.x,
                        "passive shared anatomy must be stored in the common torso source frame");
                    // Use the same local arithmetic as diaphragm/lobe vertices:
                    // a world/torso round trip would split coincident Float32
                    // interface points before the GPU even sees them.
                    const auto local=addPoint(instance.translationAndScale,
                        scalePoint(rotatePoint(instance.orientation,vertex.position),instance.translationAndScale.w));
                    const auto normal=rotatePoint(instance.orientation,vertex.normalAndTangentSign);
                    const float height=dotPoint(local,anatomyGPU.superiorAxisAndHeight);
                    const float h=float(NumiHumanRestingAnatomy::smoothstep(
                        functional.passiveTransition[0],functional.passiveTransition[1],height));
                    maps[v].deformationWeight.x=h;
                    const auto& pelvis=initialBodies.at(functional.passivePelvicBody);
                    const auto world=addPoint(initialThorax.position,rotatePoint(initialThorax.orientation,local));
                    const auto worldNormal=rotatePoint(initialThorax.orientation,normal);
                    const auto pelvicPoint=rotatePoint(inverseRotation(pelvis.orientation),subtractPoint(world,pelvis.position));
                    const auto pelvicNormal=rotatePoint(inverseRotation(pelvis.orientation),worldNormal);
                    add(v,anatomyGPU.bodyAndFlags.x,local,normal,h);
                    add(v,functional.passivePelvicBody,pelvicPoint,pelvicNormal,1-h);
                }else if(deformation==12&&commonAttachmentIndex!=MR_INVALID_INDEX) {
                    const auto& attachment=functional.commonFieldPassiveAttachments.at(commonAttachmentIndex);
                    require(instance.binding.z==MR_VISUAL_BINDING_ARTICULATED_LINK&&
                        instance.binding.y==anatomyGPU.bodyAndFlags.x&&
                        instance.translationAndScale.x==0&&instance.translationAndScale.y==0&&
                        instance.translationAndScale.z==0&&instance.translationAndScale.w==1&&
                        instance.orientation.x==0&&instance.orientation.y==0&&instance.orientation.z==0&&instance.orientation.w==1,
                        "passive common attachment must be registered in the exact torso source frame");
                    const auto& vertex=pack.vertices.at(v);
                    const mr_float4 local={vertex.position.x,vertex.position.y,vertex.position.z,0};
                    const mr_float4 normal={vertex.normalAndTangentSign.x,vertex.normalAndTangentSign.y,
                        vertex.normalAndTangentSign.z,0};
                    float torsoWeight=1.0f;
                    if(attachment.inferiorVenaCava) {
                        torsoWeight=numi_human_resting_common_field::quinticSmoothstep(attachment.transitionLower,attachment.transitionUpper,
                            (&local.x)[attachment.sourceSuperiorAxis]);
                        const auto& anchor=initialBodies.at(attachment.anchorBodyIndex);
                        require(attachment.originalBodyIndex==attachment.anchorBodyIndex&&
                            attachment.bodyIndex==anatomyGPU.bodyAndFlags.x&&
                            attachment.sourceSuperiorAxis==1&&std::isfinite(torsoWeight)&&torsoWeight>=0&&torsoWeight<=1,
                            "inferior vena cava torso/abdomen attachment parameters are invalid");
                        const auto world=addPoint(initialThorax.position,rotatePoint(initialThorax.orientation,local));
                        const auto worldNormal=rotatePoint(initialThorax.orientation,normal);
                        const auto anchorLocal=rotatePoint(inverseRotation(anchor.orientation),subtractPoint(world,anchor.position));
                        const auto anchorNormal=rotatePoint(inverseRotation(anchor.orientation),worldNormal);
                        add(v,anatomyGPU.bodyAndFlags.x,local,normal,torsoWeight);
                        add(v,attachment.anchorBodyIndex,anchorLocal,anchorNormal,1.0f-torsoWeight);
                    }else add(v,anatomyGPU.bodyAndFlags.x,local,normal,1.0f);
                    // W marks a typed passive attachment for identity/debugging;
                    // kind 12 still applies the complete common-field delta.
                    maps[v].deformationWeight={torsoWeight,0,0,1};
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
            const auto& soleInfluence=weights[m.firstInfluence];
            const bool exactTorsoSource=m.influenceCount==1&&
                soleInfluence.body.x==functional.gpu.bodyAndFlags.x&&
                soleInfluence.positionAndWeight.w==1.0f;
            for(unsigned j=0;j<m.influenceCount;++j) {
                const auto& w=weights[m.firstInfluence+j];mr_float4 point=w.positionAndWeight,normal=w.normal;
                if(w.body.x!=MR_INVALID_INDEX) {
                    const auto& b=initialBodies.at(w.body.x);
                    point=addPoint(b.position,rotatePoint(b.orientation,point));normal=rotatePoint(b.orientation,normal);
                }
                p=addPoint(p,scalePoint(point,w.positionAndWeight.w));n=addPoint(n,scalePoint(normal,w.positionAndWeight.w));
            }
            p.w=1;pack.vertices[v].position=p;
            if(maps[v].deformationKind==1||maps[v].deformationKind==3||maps[v].deformationKind==4||
               maps[v].deformationKind==9||maps[v].deformationKind==11) {
                const auto local=exactTorsoSource?soleInfluence.positionAndWeight:
                    rotatePoint(inverseRotation(initialThorax.orientation),subtractPoint(p,initialThorax.position));
                const auto& axis=functional.gpu.superiorAxisAndHeight;
                const auto b=functional.respiratoryBasis.evaluate({local.x,local.y,local.z},{axis.x,axis.y,axis.z});
                maps[v].respiratoryBasis={float(b[0]),float(b[1]),float(b[2]),float(b[3])};
            }
            if(maps[v].deformationKind==3) {
                const auto& torso=initialBodies.at(functional.gpu.bodyAndFlags.x);
                const auto local=exactTorsoSource?soleInfluence.positionAndWeight:
                    rotatePoint(inverseRotation(torso.orientation),subtractPoint(p,torso.position));
                const auto& axis=functional.gpu.superiorAxisAndHeight;
                const float h=dotPoint(subtractPoint(functional.gpu.lungAnchorAndVolume,local),axis)/axis.w;
                const float upper=std::clamp(h/.2f,0.0f,1.0f),lower=std::clamp((1.6f-h)/.6f,0.0f,1.0f);
                maps[v].deformationWeight.x*=upper*upper*(3-2*upper)*lower*lower*(3-2*lower);
                if(maps[v].deformationWeight.x>.5f) {
                    const float ap=dotPoint(local,anatomyGPU.anteriorAxis);
                    posteriorSkin=std::min(posteriorSkin,ap);anteriorSkin=std::max(anteriorSkin,ap);
                }
            }
            if(maps[v].deformationKind==7) {
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
        if(commonCardiacGeometry) {
            require(commonFieldVertexCount>0&&commonFieldNormalVertexCount==commonFieldVertexCount,
                "common cardiac coefficient map and rendered source ranges have different vertex totals");
            for(unsigned channel=0;channel<7;++channel)
                require(commonFieldAuditIndices[channel]!=MR_INVALID_INDEX,
                    "common cardiac renderer pack is missing a declared source surface");
            require(commonAttachmentsSeen.size()==functional.commonFieldPassiveAttachments.size(),
                "common field renderer pack is missing a declared passive attachment surface");
        }
        // Supine reduction: the dorsal support strip stays with its contact
        // owner. The chest expands toward the anterior surface; all five lung
        // lobes, pleura and diaphragm share the same basal motion field.
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
        auditedMeshPrimitives=pack.primitives;
        auto device=coupled.physiology.device;queue=[device newCommandQueue];
        meshAuditPartials=[device newBufferWithLength:meshAuditGroupCount*sizeof(mr_uint4) options:MTLResourceStorageModeShared];
        meshAuditResult=[device newBufferWithLength:sizeof(mr_uint4)+sizeof(MRHumanRestingSurfaceFailureGPU) options:MTLResourceStorageModeShared];
        mapping=[device newBufferWithBytes:maps.data() length:maps.size()*sizeof(maps.front()) options:MTLResourceStorageModeShared];
        influences=[device newBufferWithBytes:weights.data() length:weights.size()*sizeof(weights.front()) options:MTLResourceStorageModeShared];
        anatomyParameters=[device newBufferWithBytes:&anatomyGPU length:sizeof(anatomyGPU) options:MTLResourceStorageModeShared];
        auditCount=unsigned(audits.size());
        const unsigned expectedAuditCount=commonCardiacGeometry?unsigned(functional.lungs.size()+7):
            9+unsigned(cardiacWallVertexCount>0);
        require(auditCount==expectedAuditCount,
            "functional anatomy volume audit did not bind its lung, chamber and material surfaces");
        surfaceAudits=[device newBufferWithBytes:audits.data() length:audits.size()*sizeof(audits.front()) options:MTLResourceStorageModeShared];
        volumeResults=[device newBufferWithLength:(auditCount+2)*sizeof(mr_float4)+
            auditCount*sizeof(MRHumanRestingSurfaceFailureGPU) options:MTLResourceStorageModeShared];
        cardiacQ=[device newBufferWithLength:4*sizeof(float) options:MTLResourceStorageModeShared];
        const MRHumanRestingCardiacWallVertexGPU emptyWallVertex{};
        const mr_uint4 emptyWallRange{};const unsigned emptyWallIndex=0;
        const mr_float4 emptyWallQ{};
        cardiacWallMap=[device newBufferWithBytes:cardiacWallVertexCount?functional.ventricularWallMap.data():&emptyWallVertex
            length:std::max(1u,cardiacWallVertexCount)*sizeof(emptyWallVertex) options:MTLResourceStorageModeShared];
        cardiacWallParameters=[device newBufferWithBytes:&functional.ventricularWallGPU
            length:sizeof(functional.ventricularWallGPU) options:MTLResourceStorageModeShared];
        cardiacWallQ=[device newBufferWithBytes:&emptyWallQ length:sizeof(emptyWallQ) options:MTLResourceStorageModeShared];
        cardiacWallNormalRanges=[device newBufferWithBytes:wallNormalRanges.empty()?&emptyWallRange:wallNormalRanges.data()
            length:std::max(std::size_t(1),wallNormalRanges.size())*sizeof(emptyWallRange) options:MTLResourceStorageModeShared];
        cardiacWallIncidentTriangles=[device newBufferWithBytes:wallIncidentTriangles.empty()?&emptyWallIndex:wallIncidentTriangles.data()
            length:std::max(std::size_t(1),wallIncidentTriangles.size())*sizeof(unsigned) options:MTLResourceStorageModeShared];
        airwayNormalRanges=[device newBufferWithBytes:airwayPointNormalRanges.empty()?&emptyWallRange:airwayPointNormalRanges.data()
            length:std::max(std::size_t(1),airwayPointNormalRanges.size())*sizeof(emptyWallRange) options:MTLResourceStorageModeShared];
        airwayIncidentTriangles=[device newBufferWithBytes:airwayPointIncidentTriangles.empty()?&emptyWallIndex:airwayPointIncidentTriangles.data()
            length:std::max(std::size_t(1),airwayPointIncidentTriangles.size())*sizeof(unsigned) options:MTLResourceStorageModeShared];
        const MRHumanRestingCommonFieldVertexGPU emptyCommonMap{};
        const MRHumanRestingCommonCoordinateBoxGPU emptyCommonBox{};
        commonFieldMapBuffer=[device newBufferWithBytes:functional.commonFieldMap.empty()?&emptyCommonMap:functional.commonFieldMap.data()
            length:std::max(std::size_t(1),functional.commonFieldMap.size())*sizeof(emptyCommonMap) options:MTLResourceStorageModeShared];
        commonFieldParameters=[device newBufferWithBytes:&functional.commonFieldGPU length:sizeof(functional.commonFieldGPU) options:MTLResourceStorageModeShared];
        commonFieldBoxes=[device newBufferWithBytes:functional.commonFieldBoxes.empty()?&emptyCommonBox:functional.commonFieldBoxes.data()
            length:std::max(std::size_t(1),functional.commonFieldBoxes.size())*sizeof(emptyCommonBox) options:MTLResourceStorageModeShared];
        commonFieldCoordinates=coupled.presentationCommonCoordinates;
        commonFieldNormalRanges=cardiacWallNormalRanges;commonFieldIncidentTriangles=cardiacWallIncidentTriangles;
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
        cardiacWallQPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_cardiac_wall_q"] error:&e];
        cardiacWallNormalsPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_cardiac_wall_normals"] error:&e];
        commonCoordinatesPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_common_coordinates"] error:&e];
        commonCoordinateStatusPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_common_coordinate_status_gate"] error:&e];
        layerPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_layers"] error:&e];
        volumePipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_audit_volumes"] error:&e];
        skinAuditPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_audit_skin"] error:&e];
        bodyAuditPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_audit_body"] error:&e];
        meshAuditPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_audit_mesh_triangles"] error:&e];
        meshAuditReducePipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_reduce_mesh_audit"] error:&e];
        require(meshAuditPartials&&meshAuditResult&&meshAuditPipeline&&meshAuditReducePipeline,"whole-mesh GPU triangle audit setup failed");
        if(!requestedGeometrySteps.empty()) {
            vertexCapturePipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_capture_vertices"] error:&e];
            vertexCaptureBuffer=[device newBufferWithLength:maps.size()*sizeof(MRVisualVertexGPUV2) options:MTLResourceStorageModeShared];
            vertexCaptureBuffer.label=@"Numi Human selected accepted render vertices";
        }
        require(mapping&&influences&&anatomyParameters&&surfaceAudits&&volumeResults&&instanceLayers&&cardiacQ&&skinPipeline&&cardiacQPipeline&&layerPipeline&&volumePipeline&&skinAuditPipeline&&bodyAuditPipeline,"resting GPU anatomy setup failed");
        require(cardiacWallMap&&cardiacWallParameters&&cardiacWallQ&&cardiacWallNormalRanges&&cardiacWallIncidentTriangles&&
            cardiacWallQPipeline&&cardiacWallNormalsPipeline,"resting GPU cardiac material setup failed");
        require(commonFieldMapBuffer&&commonFieldParameters&&commonFieldBoxes&&commonFieldCoordinates&&
            commonFieldNormalRanges&&commonFieldIncidentTriangles,"resting common-field buffer setup failed");
        require(airwayNormalRanges&&airwayIncidentTriangles,"ID31 normal reconstruction buffers are unavailable");
        require(!commonCardiacGeometry||(commonCoordinatesPipeline&&commonCoordinateStatusPipeline&&commonFieldVertexCount>0),
            "resting common cardiac field pipelines are unavailable");
        if(commonCardiacGeometry) {
            auto& respiration=*coupled.physiology.respiration;
            require(!respiration.commonGeometryGateEnabled,"common cardiac transaction gate was already installed");
            respiration.commonGeometryGateEnabled=true;
            respiration.commonCoordinatesSolvePipeline=commonCoordinatesPipeline;
            respiration.commonCoordinateStatusPipeline=commonCoordinateStatusPipeline;
            respiration.commonGeometryParameters=commonFieldParameters;
            respiration.commonGeometryBoxes=commonFieldBoxes;
            respiration.commonCandidateCoordinates=coupled.presentationCandidateCommonCoordinates;
            // The initial accepted respiratory state needs its registered
            // geometry before any window can render. This invokes the same GPU
            // volume owner without advancing physiology or body dynamics.
            coupled.initializeInitialCommonCoordinates();
        }
        if(!requestedGeometrySteps.empty()) {
            if(commonCardiacGeometry)writeCommonFieldIdentity(output,functional);
            else writeCardiacWallMapIdentity(output,functional);
        }
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
        if(self.cardiacWallVertexCount&&!self.commonCardiacGeometry) {
            e.setPipeline(e.context,(__bridge void*)self.cardiacWallQPipeline);
            e.setBuffer(e.context,(__bridge void*)self.cardiacWallParameters,0,0);
            e.setBuffer(e.context,(__bridge void*)self.cardiacQ,0,1);
            e.setBuffer(e.context,(__bridge void*)self.cardiacWallQ,0,2);
            e.dispatchThreads(e.context,1,1);
        }
        e.setPipeline(e.context,(__bridge void*)self.skinPipeline);e.setBytes(e.context,&d,sizeof(d),0);
        e.setBuffer(e.context,(__bridge void*)self.mapping,0,1);e.setBuffer(e.context,(__bridge void*)self.influences,0,2);
        e.setBuffer(e.context,(__bridge void*)self.coupled.presentationBodies,0,3);e.setBuffer(e.context,lease.meshVertices,0,4);
        e.setBuffer(e.context,(__bridge void*)self.coupled.presentationRespiration,0,5);
        e.setBuffer(e.context,(__bridge void*)self.anatomyParameters,0,6);
        e.setBuffer(e.context,(__bridge void*)self.cardiacQ,0,7);
        e.setBuffer(e.context,(__bridge void*)self.cardiacWallMap,0,8);
        e.setBuffer(e.context,(__bridge void*)self.cardiacWallParameters,0,9);
        e.setBuffer(e.context,(__bridge void*)self.cardiacWallQ,0,10);
        e.setBuffer(e.context,(__bridge void*)self.commonFieldMapBuffer,0,11);
        e.setBuffer(e.context,(__bridge void*)self.commonFieldCoordinates,0,12);
        e.dispatchThreads(e.context,lease.meshVertexCount,64);
        const unsigned normalVertexCount=self.commonCardiacGeometry?self.commonFieldNormalVertexCount:self.cardiacWallVertexCount;
        if(normalVertexCount) {
            e.setPipeline(e.context,(__bridge void*)self.cardiacWallNormalsPipeline);
            e.setBytes(e.context,&normalVertexCount,sizeof(normalVertexCount),0);
            const auto ranges=self.commonCardiacGeometry?self.commonFieldNormalRanges:self.cardiacWallNormalRanges;
            const auto incidents=self.commonCardiacGeometry?self.commonFieldIncidentTriangles:self.cardiacWallIncidentTriangles;
            e.setBuffer(e.context,(__bridge void*)ranges,0,1);
            e.setBuffer(e.context,(__bridge void*)incidents,0,2);
            e.setBuffer(e.context,lease.meshIndices,0,3);e.setBuffer(e.context,lease.meshVertices,0,4);
            e.dispatchThreads(e.context,normalVertexCount,64);
        }
        if(self.airwayNormalVertexCount) {
            const unsigned airwayCount=self.airwayNormalVertexCount;
            e.setPipeline(e.context,(__bridge void*)self.cardiacWallNormalsPipeline);
            e.setBytes(e.context,&airwayCount,sizeof(airwayCount),0);
            e.setBuffer(e.context,(__bridge void*)self.airwayNormalRanges,0,1);
            e.setBuffer(e.context,(__bridge void*)self.airwayIncidentTriangles,0,2);
            e.setBuffer(e.context,lease.meshIndices,0,3);e.setBuffer(e.context,lease.meshVertices,0,4);
            e.dispatchThreads(e.context,airwayCount,64);
        }
        if(lease.encoder->splitCommandEncoder&&
           !lease.encoder->splitCommandEncoder(lease.encoder->context))return false;
        e.setPipeline(e.context,(__bridge void*)self.volumePipeline);e.setBytes(e.context,&d,sizeof(d),0);
        e.setBuffer(e.context,(__bridge void*)self.surfaceAudits,0,1);e.setBuffer(e.context,lease.meshIndices,0,2);
        e.setBuffer(e.context,lease.meshVertices,0,3);e.setBuffer(e.context,(__bridge void*)self.coupled.presentationRespiration,0,4);
        e.setBuffer(e.context,(__bridge void*)self.anatomyParameters,0,5);e.setBuffer(e.context,(__bridge void*)self.volumeResults,0,6);
        e.setBuffer(e.context,(__bridge void*)self.cardiacWallParameters,0,7);
        e.setBuffer(e.context,(__bridge void*)self.volumeResults,
            (self.auditCount+2)*sizeof(mr_float4),8);
        e.setBuffer(e.context,(__bridge void*)self.commonFieldParameters,0,9);
        e.setBuffer(e.context,(__bridge void*)self.commonFieldCoordinates,0,10);
        e.dispatchThreads(e.context,self.auditCount,1);
        if(lease.encoder->splitCommandEncoder&&
           !lease.encoder->splitCommandEncoder(lease.encoder->context))return false;
        // Every rendered triangle is checked in the existing presentation command.
        // The CPU receives only counts and one exact binary32 failure witness.
        const mr_uint4 meshAuditDimensions={lease.meshTriangleCount,meshAuditGroupCount,0,0};
        e.setPipeline(e.context,(__bridge void*)self.meshAuditPipeline);
        e.setBytes(e.context,&meshAuditDimensions,sizeof(meshAuditDimensions),0);
        e.setBuffer(e.context,lease.meshIndices,0,1);e.setBuffer(e.context,lease.meshVertices,0,2);
        e.setBuffer(e.context,(__bridge void*)self.meshAuditPartials,0,3);
        e.dispatchThreads(e.context,meshAuditGroupCount*256,256);
        e.setPipeline(e.context,(__bridge void*)self.meshAuditReducePipeline);
        e.setBytes(e.context,&meshAuditDimensions,sizeof(meshAuditDimensions),0);
        e.setBuffer(e.context,(__bridge void*)self.meshAuditPartials,0,1);
        e.setBuffer(e.context,lease.meshIndices,0,2);e.setBuffer(e.context,lease.meshVertices,0,3);
        e.setBuffer(e.context,(__bridge void*)self.meshAuditResult,0,4);
        e.setBuffer(e.context,(__bridge void*)self.meshAuditResult,sizeof(mr_uint4),5);
        e.dispatchThreads(e.context,1,1);
        e.setPipeline(e.context,(__bridge void*)self.skinAuditPipeline);e.setBytes(e.context,&d,sizeof(d),0);
        e.setBuffer(e.context,(__bridge void*)self.mapping,0,1);e.setBuffer(e.context,lease.meshVertices,0,2);
        e.setBuffer(e.context,(__bridge void*)self.volumeResults,0,3);e.dispatchThreads(e.context,256,256);
        const mr_uint4 bodyAuditDimensions={unsigned(self.coupled.presentationBodies.length/sizeof(MRBodyStateGPU)),self.auditCount+1,0,0};
        e.setPipeline(e.context,(__bridge void*)self.bodyAuditPipeline);
        e.setBytes(e.context,&bodyAuditDimensions,sizeof(bodyAuditDimensions),0);
        e.setBuffer(e.context,(__bridge void*)self.coupled.presentationBodies,0,1);
        e.setBuffer(e.context,(__bridge void*)self.volumeResults,0,2);e.dispatchThreads(e.context,1,1);
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
        const double renderStart=profileTiming?CACurrentMediaTime():0;
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
        const double encodeStart=profileTiming?CACurrentMediaTime():0;
        auto cb=[queue commandBuffer];
        id<MTLCounterSampleBuffer> timingSamples=profileGpuTiming
            ?makeViewerGpuTimingSamples(coupled.physiology.device):nil;
        bool timedEncode=false;
        metalrobo::MetalHybridRendererDiagnostics result;
        if(timingSamples) {
            ViewerTimedEncoderContext timing;
            timing.command=cb;timing.samples=timingSamples;timing.stage=0u;
            timing.fence=[coupled.physiology.device newFence];
            if(beginViewerTimedEncoder(timing)) {
                auto callbacks=viewerTimedCallbacks(timing);
                metalrobo::HybridDeviceObservationBuffers outputs;
                outputs.rgb=renderer->nativeBuffer(metalrobo::MetalHybridRendererBuffer::rgb);
                outputs.depth=renderer->nativeBuffer(metalrobo::MetalHybridRendererBuffer::depth);
                outputs.segmentation=renderer->nativeBuffer(metalrobo::MetalHybridRendererBuffer::segmentation);
                outputs.identities=renderer->nativeBuffer(metalrobo::MetalHybridRendererBuffer::identities);
                outputs.normals=renderer->nativeBuffer(metalrobo::MetalHybridRendererBuffer::normals);
                outputs.motion=renderer->nativeBuffer(metalrobo::MetalHybridRendererBuffer::motion);
                outputs.validity=renderer->nativeBuffer(metalrobo::MetalHybridRendererBuffer::validity);
                result=renderer->encodeGraph(worlds,state,camera,callbacks,outputs,false);
                if(timing.encoder!=nil)[timing.encoder endEncoding];
                timedEncode=true;
            } else {
                timingSamples=nil;
            }
        }
        if(!timedEncode) {
            auto enc=[cb computeCommandEncoder];
            result=renderer->encode(worlds,state,camera,(__bridge void*)enc);
            [enc endEncoding];
            if(profileGpuTiming&&!gpuTimingUnavailableReported) {
                std::fprintf(stderr,"resting_viewer_gpu_timing sampling=unavailable\n");
                gpuTimingUnavailableReported=true;
            }
        }
        require(result.succeeded(),result.message);
        const double commandStart=profileTiming?CACurrentMediaTime():0;
        [cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"resting native renderer failed");
        if(timingSamples)reportViewerGpuTiming(timingSamples,captureStep);
        const double commandEnd=profileTiming?CACurrentMediaTime():0;
        const bool geometryExportRequested=captureThisFrame;
        if(captureThisFrame)require(captureKernelEncoded,
            "selected accepted step did not encode its geometry snapshot");
        const auto* volumes=static_cast<const mr_float4*>(volumeResults.contents);
        const auto* failureRecords=reinterpret_cast<const MRHumanRestingSurfaceFailureGPU*>(
            static_cast<const unsigned char*>(volumeResults.contents)+(auditCount+2)*sizeof(mr_float4));
        const auto* cardiacCoordinates=static_cast<const float*>(cardiacQ.contents);
        const auto wallCorrection=*static_cast<const mr_float4*>(cardiacWallQ.contents);
        const auto wallAudit=commonCardiacGeometry?volumes[commonFieldAuditIndices[5]]:
            (cardiacWallVertexCount?volumes[cardiacWallAuditIndex]:mr_float4{});
        const auto commonCoordinatesValue=commonCardiacGeometry?
            *static_cast<const MRHumanRestingCommonCoordinatesGPU*>(commonFieldCoordinates.contents):
            MRHumanRestingCommonCoordinatesGPU{};
        float maxRelativeError=0;unsigned geometryStatus=0;
        for(unsigned i=0;i<auditCount;++i) {
            maxRelativeError=std::max(maxRelativeError,volumes[i].z);geometryStatus|=unsigned(volumes[i].w);
        }
        const auto meshAudit=*static_cast<const mr_uint4*>(meshAuditResult.contents);
        require(meshAudit.w==layout.meshTriangleCount,"whole-mesh GPU triangle audit is incomplete");
        const auto skinAudit=volumes[auditCount];
        const auto bodyAudit=volumes[auditCount+1];
        require(std::isfinite(bodyAudit.x)&&std::isfinite(bodyAudit.y)&&std::isfinite(bodyAudit.z)&&
            std::isfinite(bodyAudit.w)&&bodyAudit.w>0,"accepted body mass/center-of-mass diagnostic is invalid");
        surfaceTrace<<std::setprecision(12)<<time<<','<<p.status.x<<','<<skinAudit.x<<','<<skinAudit.z<<','<<skinAudit.w<<','<<maxRelativeError
            <<','<<cardiacCoordinates[0]<<','<<cardiacCoordinates[1]<<','<<cardiacCoordinates[2]<<','<<cardiacCoordinates[3]
            <<','<<p.chamberVolumes.x*1e6<<','<<p.chamberVolumes.y*1e6<<','<<p.chamberVolumes.z*1e6<<','<<p.chamberVolumes.w*1e6
            <<','<<p.motion.x*1e6<<','<<p.motion.y*1e6<<','<<p.mechanics.x*1e6
            <<','<<bodyAudit.x<<','<<bodyAudit.y<<','<<bodyAudit.z<<','<<bodyAudit.w
            <<','<<(!commonCardiacGeometry&&cardiacWallVertexCount>0)<<','<<wallAudit.x*1e6<<','<<wallAudit.y*1e6<<','
            <<(commonCardiacGeometry?std::numeric_limits<double>::quiet_NaN():wallCorrection.x*1e3)<<','<<wallAudit.w<<','<<geometryStatus
            <<','<<(commonCardiacGeometry?"common_seven_coordinate_v1":"legacy_ventricular_wall_v2")
            <<','<<commonCoordinatesValue.first.x<<','<<commonCoordinatesValue.first.y<<','
            <<commonCoordinatesValue.first.z<<','<<commonCoordinatesValue.first.w<<','
            <<commonCoordinatesValue.second.x<<','<<commonCoordinatesValue.second.y<<','
            <<commonCoordinatesValue.second.z<<','<<commonCoordinatesValue.status.x<<','
            <<commonCoordinatesValue.status.y<<','<<commonCoordinatesValue.status.z<<','
            <<commonCoordinatesValue.diagnostics.x<<','<<(!commonCardiacGeometry)
            <<','<<meshAudit.x<<','<<meshAudit.y<<','<<meshAudit.w<<'\n';
        surfaceTrace.flush();
        unsigned firstFailedAudit=MR_INVALID_INDEX;
        for(unsigned i=0;i<auditCount;++i)if(unsigned(volumes[i].w)!=0u){firstFailedAudit=i;break;}
        if(firstFailedAudit!=MR_INVALID_INDEX) {
            const unsigned status=unsigned(volumes[firstFailedAudit].w);
            const auto& failure=failureRecords[firstFailedAudit];
            const unsigned step=static_cast<unsigned>(p.status.x);
            const auto message=numiHumanRestingSurfaceAudit::describeSurfaceFailure(
                auditStableIds.at(firstFailedAudit),firstFailedAudit,
                static_cast<const MRHumanRestingSurfaceAuditGPU*>(surfaceAudits.contents)[firstFailedAudit].indicesAndOwner.x,status,volumes[firstFailedAudit].z,
                failure,initialPackContentHash);
            RejectedGeometryDiagnostic diagnostic{};
            if(captureThisFrame&&captureKernelEncoded&&skinAudit.w==0&&meshAudit.y==0)
                diagnostic=exportRejectedGeometryDiagnostic(step,time,state.acceptedRootFingerprint,
                    state.acceptedTransactionFingerprint,state.acceptedTimestampMicroseconds,
                    auditStableIds.at(firstFailedAudit),status,volumes[firstFailedAudit].z);
            writeSurfaceAuditFailureReceipt(step,time,state.acceptedRootFingerprint,
                state.acceptedTransactionFingerprint,state.acceptedTimestampMicroseconds,
                firstFailedAudit,auditStableIds.at(firstFailedAudit),status,volumes[firstFailedAudit].z,
                static_cast<const MRHumanRestingSurfaceAuditGPU*>(surfaceAudits.contents)[firstFailedAudit].indicesAndOwner.x,
                failure,captureThisFrame,diagnostic);
            std::cerr<<"resting_surface_audit_failure accepted_step="<<step<<' '<<message<<'\n';
            captureThisFrame=false;captureKernelEncoded=false;
            require(false,message);
        }
        if(meshAudit.x||meshAudit.y) {
            const float volumeErrorNotApplicable=std::numeric_limits<float>::quiet_NaN();
            const unsigned triangleIndex=meshAudit.z,indexOffset=3u*triangleIndex;
            const auto found=std::find_if(auditedMeshPrimitives.begin(),auditedMeshPrimitives.end(),
                [&](const auto& primitive){return indexOffset>=primitive.geometry.x&&
                    indexOffset-primitive.geometry.x<primitive.geometry.y;});
            require(triangleIndex<layout.meshTriangleCount&&found!=auditedMeshPrimitives.end(),
                "whole-mesh triangle failure has no source primitive");
            const unsigned primitiveIndex=unsigned(found-auditedMeshPrimitives.begin());
            auto failure=*reinterpret_cast<const MRHumanRestingSurfaceFailureGPU*>(
                static_cast<const unsigned char*>(meshAuditResult.contents)+sizeof(mr_uint4));
            failure.surfaceTriangleKind.x=primitiveIndex;
            failure.surfaceTriangleKind.y=(indexOffset-found->geometry.x)/3u;
            const unsigned step=static_cast<unsigned>(p.status.x);
            RejectedGeometryDiagnostic diagnostic{};
            if(captureThisFrame&&captureKernelEncoded&&skinAudit.w==0&&meshAudit.y==0)
                diagnostic=exportRejectedGeometryDiagnostic(step,time,state.acceptedRootFingerprint,
                    state.acceptedTransactionFingerprint,state.acceptedTimestampMicroseconds,
                    found->identity.w,2u,volumeErrorNotApplicable);
            writeSurfaceAuditFailureReceipt(step,time,state.acceptedRootFingerprint,
                state.acceptedTransactionFingerprint,state.acceptedTimestampMicroseconds,
                primitiveIndex,found->identity.w,2u,volumeErrorNotApplicable,found->geometry.x,failure,
                captureThisFrame,diagnostic,found->identity.x,"all_rendered_triangles");
            captureThisFrame=false;captureKernelEncoded=false;
            std::ostringstream message;
            message<<"whole native mesh contains invalid triangles: zero_area="<<meshAudit.x
                <<" nonfinite_area="<<meshAudit.y<<" semantic="<<found->identity.x
                <<" stable_id="<<found->identity.w<<" local_triangle="<<failure.surfaceTriangleKind.y;
            require(false,message.str());
        }
        if(captureThisFrame) {
            NSDictionary* surfaceAuditOutcome=@{
                @"schema":@"numi.human.accepted_surface_audit.v1",
                @"physical_endpoint":@"accepted",@"surface_audit_endpoint":@"passed",
                @"functional_surface_count":@(auditCount),@"functional_surface_status":@(geometryStatus),
                @"mesh_triangles_checked":@(meshAudit.w),@"mesh_zero_area_triangles":@(meshAudit.x),
                @"mesh_nonfinite_area_triangles":@(meshAudit.y)};
            exportAcceptedGeometry(captureStep,time,captureRootFingerprint,captureTransactionFingerprint,
                captureTimestampMicroseconds,surfaceAuditOutcome);
            captureThisFrame=false;
        }
        require(wallCorrection.w==0,"accepted ventricular material volume closure failed");
        require(skinAudit.w==0&&skinAudit.z==0,"accepted full skin intersects the bed beyond the 1 mm inspection tolerance");
        std::ostringstream metrics;metrics<<std::fixed<<std::setprecision(2)<<"Accepted time "<<time<<" s  |  "<<time/std::max(.001,CACurrentMediaTime()-wallOrigin)<<" x real time  |  breaths "<<p.status.y<<"  beats "<<p.cardiacStatus.x<<"\n"
            <<"Lung "<<p.mechanics.x*1e3<<" L  Airflow "<<p.mechanics.w*1e3<<" L/s  Pleural "<<p.mechanics.z/98.0665<<" cmH2O  Muscle activation "
            <<p.muscles[0].excitationAndActivation.y<<" / "<<p.muscles[1].excitationAndActivation.y<<"\n"
            <<"PaO2 "<<p.observation.x<<"  PaCO2 "<<p.observation.y<<" mmHg  SaO2 "<<p.observation.z*100<<"%  Tidal "<<p.breath.z*1e6<<" mL\n"
            <<"LV "<<p.circulation.y*1e6<<" mL / "<<p.cardiacPressure.x/133.322387415<<" mmHg  RV "<<p.circulation.z*1e6<<" mL  Stroke "<<p.cardiacFlow.w*1e6<<" mL  Blood "<<p.circulation.x*1e3<<" L  CO mean "
            <<(time>0?p.cardiacFlow.x*60e3/time:0)<<" L/min\n"
            <<"Mixed-source reference anatomy; passive structures remain inspection geometry.";
        if(rigidHands)metrics<<" Rigid digits; wrists free; hand function not simulated.";
        if(profileTiming) {
            const double renderEnd=CACurrentMediaTime();
            std::cout<<"resting_render_profile step="<<p.status.x
                <<" vertices="<<layout.meshVertexCount<<" triangles="<<layout.meshTriangleCount
                <<" geometry_export="<<geometryExportRequested
                <<" encode_wall_ms="<<(commandStart-encodeStart)*1e3
                <<" command_wall_ms="<<(commandEnd-commandStart)*1e3
                <<" gpu_ms="<<(cb.GPUEndTime-cb.GPUStartTime)*1e3
                <<" audit_export_wall_ms="<<(renderEnd-commandEnd)*1e3
                <<" total_render_wall_ms="<<(renderEnd-renderStart)*1e3<<'\n';
        }
        return {(__bridge id<MTLBuffer>)renderer->nativeBuffer(metalrobo::MetalHybridRendererBuffer::rgb),dimension,dimension,metrics.str(),time,complete};
    }
    void declareRigidHands(){rigidHands=true;}
    void present(bool finished=false){
        const double presentStart=profileTiming?CACurrentMediaTime():0;
        complete=finished;[window renderFrameNow];
        if(profileTiming) {
            const auto& state=*static_cast<const NMHumanRespirationState*>(coupled.presentationRespiration.contents);
            std::cout<<"resting_present_profile step="<<state.status.x
                <<" finished="<<finished<<" total_present_wall_ms="<<(CACurrentMediaTime()-presentStart)*1e3<<'\n';
        }
        if(finished)for(unsigned step:requestedGeometrySteps)require(completedGeometrySteps.contains(step),
            "requested accepted geometry step was not presented: "+std::to_string(step));
    }
    metalrobo::MetalNumiHumanSupportGeometryProgram supportProgram(){return skinSupport->program();}
    ~NumiHumanRestingVisual(){[window finishRecording];}
};
