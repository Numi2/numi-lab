#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/task_program_types.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cstring>
#include <fstream>
#include <vector>
#include "metalrobo/MetalHybridRenderer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

template <typename Result>
void require(const Result& result, const char* operation) {
    if (!result.succeeded()) {
        throw std::runtime_error(std::string{operation} + ": " +
                                 result.message);
    }
}

} // namespace


bool evaluate(metalrobo::MetalHybridRenderer& renderer,
              const metalrobo::HybridObservationBatch& frame,
              const char* originalPath,const char* candidatePath,
              const char* outputPath,bool dropped) {
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
    NSError* error=nil;
    std::array<id<MTLComputePipelineState>,2> pipeline;
    const char* paths[2]={originalPath,candidatePath};
    for(unsigned arm=0;arm<2;++arm) {
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:paths[arm]]] error:&error];
        id<MTLFunction> fn=[lib newFunctionWithName:@"mr_hybrid_masked_depth_history"];
        pipeline[arm]=[dev newComputePipelineStateWithFunction:fn error:&error];
        if(!pipeline[arm]) throw std::runtime_error("depth kernel pipeline");
    }
    const uint32_t envs=frame.environmentCount,w=frame.width,h=frame.height,pixels=w*h,total=envs*pixels;
    if(!envs || pixels<=256 || frame.depth.size()!=total || frame.validity.size()!=total) throw std::runtime_error("frame shape");
    auto make=[&](NSUInteger bytes){auto x=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];if(!x)throw std::runtime_error("buffer allocation");std::memset(x.contents,0,bytes);return x;};
    std::array<id<MTLBuffer>,16> b{};
    b[0]=(__bridge id<MTLBuffer>)renderer.nativeBuffer(metalrobo::MetalHybridRendererBuffer::depth);
    b[1]=(__bridge id<MTLBuffer>)renderer.nativeBuffer(metalrobo::MetalHybridRendererBuffer::identities);
    b[2]=(__bridge id<MTLBuffer>)renderer.nativeBuffer(metalrobo::MetalHybridRendererBuffer::validity);
    if(!b[0]||!b[1]||!b[2]) throw std::runtime_error("actual renderer buffers absent");
    b[3]=make(total*4);b[4]=make(total*4);b[5]=make(envs*4);b[6]=make(total*4);b[7]=make(total*4);b[8]=make(envs*sizeof(MRTaskStateGPU));
    for(unsigned i=10;i<=15;++i)b[i]=make(4096);
    auto* masks=static_cast<uint32_t*>(b[5].contents);for(unsigned i=0;i<envs;++i)masks[i]=1;
    std::vector<uint32_t> accepted;
    for(uint32_t i=0;i<total;++i)if((frame.validity[i]&MR_VISUAL_VALIDITY_GEOMETRY)!=0u)accepted.push_back(frame.identities[i].y);
    std::sort(accepted.begin(),accepted.end());accepted.erase(std::unique(accepted.begin(),accepted.end()),accepted.end());
    if(accepted.empty())throw std::runtime_error("no accepted geometry identity");
    std::memcpy(b[3].contents,accepted.data(),accepted.size()*sizeof(uint32_t));
    std::cout<<"accepted_instance_ids=";for(auto x:accepted)std::cout<<x<<',';std::cout<<'\n';
    MRHybridMaskedDepthUniformsGPU u{};u.counts={envs,h,w,(uint32_t)accepted.size()};u.layout={pixels,1,0,envs};u.history={1,1,0,1};u.frameOffsets={0,0,0,0};u.range={.1f,5.f,1.f/4.9f,0};
    MRHybridRenderUniformsGPU ru{};id<MTLCommandQueue> q=[dev newCommandQueue];
    std::ofstream file(outputPath,std::ios::binary|std::ios::app);if(!file)throw std::runtime_error("trace output");
    file.write(reinterpret_cast<const char*>(&dropped),sizeof(dropped));file.write(reinterpret_cast<const char*>(&total),sizeof(total));
    file.write(reinterpret_cast<const char*>(frame.depth.data()),total*sizeof(float));file.write(reinterpret_cast<const char*>(frame.validity.data()),total*sizeof(uint32_t));
    unsigned usable=0,truthDropped=0;
    for(uint32_t i=0;i<total;++i){usable+=(frame.validity[i]&MR_VISUAL_VALIDITY_DEPTH)!=0u;truthDropped+=((frame.validity[i]&MR_VISUAL_VALIDITY_GEOMETRY)!=0u && (frame.validity[i]&MR_VISUAL_VALIDITY_DEPTH)==0u);}
    std::cout<<"scene="<<(dropped?"dropout":"normal")<<" dimensions="<<w<<"x"<<h<<" envs="<<envs<<" depth_valid="<<usable<<" geometry_without_depth="<<truthDropped<<'\n';
    bool ok=dropped ? (truthDropped>0 && usable==0) : usable>0;
    for(unsigned flicker=0;flicker<2;++flicker) {
      u.range.w=float(flicker);
      std::vector<float> expected(total,1.0f);
      unsigned filled=0;
      for(uint32_t e=0;e<envs;++e)for(uint32_t y=0;y<h;++y)for(uint32_t x=0;x<w;++x) {
        uint32_t p=e*pixels+y*w+x;float z=frame.depth[p];
        if((frame.validity[p]&MR_VISUAL_VALIDITY_DEPTH)!=0u && z>0 && std::isfinite(z)) expected[p]=(std::clamp(z,.1f,5.f)-.1f)/4.9f;
        else if(flicker) {
          float neighbor=5.f;bool ring=false;
          for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
            int nx=int(x)+dx,ny=int(y)+dy;if((!dx&&!dy)||nx<0||ny<0||nx>=int(w)||ny>=int(h))continue;
            auto n=e*pixels+uint32_t(ny)*w+uint32_t(nx);float nz=frame.depth[n];
            if((frame.validity[n]&MR_VISUAL_VALIDITY_DEPTH)!=0u && nz>0 && std::isfinite(nz)){ring=true;neighbor=std::min(neighbor,nz);}
          }
          if(ring){expected[p]=(std::clamp(neighbor,.1f,5.f)-.1f)/4.9f;filled++;}
        }
      }
      for(unsigned arm=0;arm<2;++arm) {
        id<MTLCommandBuffer> cmd=[q commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline[arm]];
        for(NSUInteger i=0;i<9;++i)[enc setBuffer:b[i] offset:0 atIndex:i];
        [enc setBytes:&u length:sizeof(u) atIndex:9];
        for(NSUInteger i=10;i<=15;++i)[enc setBuffer:b[i] offset:0 atIndex:i];
        [enc setBytes:&ru length:sizeof(ru) atIndex:16];
        [enc dispatchThreadgroups:MTLSizeMake(envs,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];if(cmd.status!=MTLCommandBufferStatusCompleted) throw std::runtime_error("kernel command");
        auto* output=static_cast<const float*>(b[7].contents);unsigned mismatch=0;
        for(uint32_t i=0;i<total;++i)mismatch+=!std::isfinite(output[i])||std::abs(output[i]-expected[i])>2e-6f;
        file.write(reinterpret_cast<const char*>(output),total*sizeof(float));
        std::cout<<"scene="<<(dropped?"dropout":"normal")<<" flicker="<<flicker<<" arm="<<(arm?"candidate":"original")<<" mismatch="<<mismatch<<" reference_filled="<<filled<<'\n';
        if(arm)ok=ok && mismatch==0;
      }
      if(flicker && !dropped)ok=ok && filled>0;
    }
    return ok;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
        if(argc!=5) throw std::runtime_error("usage: probe FULL_LIB ORIGINAL_LIB CANDIDATE_LIB OUTPUT_BIN");
        metalrobo::WorldTemplate worldTemplate;
        require(metalrobo::compileEpisodeTwin(
                    metalrobo::makeFrankaPickPlaceEpisodeTwin(),
                    metalrobo::makeFrankaPickPlaceEngineModel(), worldTemplate),
                "episode compile");
        metalrobo::WorldFamily family;
        require(metalrobo::compileWorldFamily(
                    worldTemplate, metalrobo::makeFrankaPickPlaceWorldProgram(),
                    family),
                "family compile");
        metalrobo::MetalWorldFamilyContext worlds({.metallibPath=argv[1]});
        require(worlds.compile(family, 4u), "world compile");
        require(worlds.sample(4u, 29u), "world sample");

        metalrobo::EpisodeTwin dropoutTwin =
            metalrobo::makeFrankaPickPlaceEpisodeTwin();
        for (metalrobo::SensorSpec& sensor : dropoutTwin.sensors) {
            sensor.depthDropout = 1.0f;
        }
        metalrobo::WorldTemplate dropoutTemplate;
        require(
            metalrobo::compileEpisodeTwin(
                dropoutTwin,
                metalrobo::makeFrankaPickPlaceEngineModel(),
                dropoutTemplate
            ),
            "dropout episode compile"
        );
        metalrobo::WorldFamily dropoutFamily;
        require(
            metalrobo::compileWorldFamily(
                dropoutTemplate,
                metalrobo::makeFrankaPickPlaceWorldProgram(),
                dropoutFamily
            ),
            "dropout family compile"
        );
        metalrobo::MetalWorldFamilyContext dropoutWorlds({.metallibPath=argv[1]});
        require(
            dropoutWorlds.compile(dropoutFamily, 4u),
            "dropout world compile"
        );
        require(
            dropoutWorlds.sample(4u, 29u),
            "dropout world sample"
        );

        metalrobo::VisualRenderSceneV3 scene;
        scene.id = "camera_aligned_probe";
        scene.assetCount =
            static_cast<std::uint32_t>(worldTemplate.assets.size());
        scene.gaussians.push_back({
            {0.45f, 0.0f, 0.08f, 0.98f},
            {0.06f, 0.02f, 0.04f, 1.0f},
            {0.0f, 0.0f, 0.0f, 1.0f},
            {1.0f, 0.1f, 0.05f, 0.0f},
            {
                0u,
                MR_INVALID_INDEX,
                42u,
                MR_HYBRID_GAUSSIAN_ASSET_LOCAL,
            },
        });
        scene.environment =
            metalrobo::makeNeutralStudioEnvironmentV2();
        scene.lightRig =
            metalrobo::makeStudioKeyLightRigV1();
        scene.fingerprint =
            metalrobo::computeVisualRenderSceneV3Fingerprint(scene);
        metalrobo::MetalHybridRenderer renderer({.metallibPath=argv[1]});
        require(
            renderer.compile(
                std::move(scene),
                metalrobo::VisualRendererProfileV1::sensorFast(),
                4u
            ),
            "renderer compile"
        );
        const auto rendered = renderer.render(worlds, 4u, 0u);
        require(rendered, "render");
        metalrobo::HybridObservationBatch observations;
        require(renderer.readback(observations), "readback");

        const std::size_t center = 60u * observations.width + 80u;
        const float depth = observations.depth[center];
        const std::uint32_t semantic = observations.segmentation[center];
        const float expectedDepth = std::sqrt(
            0.35f * 0.35f +
            0.6f * 0.6f +
            0.72f * 0.72f
        );
        if (!std::isfinite(depth) ||
            std::abs(depth - expectedDepth) > 0.05f ||
            semantic != 42u ||
            observations.rgb[center].x <= observations.rgb[center].y) {
            throw std::runtime_error(
                "center pixel did not preserve Gaussian depth, "
                "semantic identity, and color");
        }

        bool allValid=evaluate(renderer,observations,argv[2],argv[3],argv[4],false);
        const auto dropoutRendered =
            renderer.render(dropoutWorlds, 4u, 0u);
        require(dropoutRendered, "dropout render");
        metalrobo::HybridObservationBatch dropoutObservations;
        require(
            renderer.readback(dropoutObservations),
            "dropout readback"
        );
        const auto truthPixel = std::find_if(
            dropoutObservations.segmentation.begin(),
            dropoutObservations.segmentation.end(),
            [](const std::uint32_t value) {
                return value == 42u;
            }
        );
        const std::size_t truthIndex =
            static_cast<std::size_t>(
                truthPixel -
                dropoutObservations.segmentation.begin()
            );
        if (truthIndex >= dropoutObservations.validity.size() ||
            (dropoutObservations.validity[truthIndex] &
             MR_VISUAL_VALIDITY_GEOMETRY) == 0u ||
            (dropoutObservations.validity[truthIndex] &
             MR_VISUAL_VALIDITY_DEPTH) != 0u ||
            dropoutObservations.identities[truthIndex].x != 42u ||
            dropoutObservations.depth[truthIndex] <
                1.0e20f) {
            throw std::runtime_error(
                "depth dropout corrupted simulation-only geometry truth"
            );
        }

        allValid=evaluate(renderer,dropoutObservations,argv[2],argv[3],argv[4],true) && allValid;
        if(!allValid) throw std::runtime_error("candidate/reference or coverage gate failed");
        std::cout << "device=\"" << rendered.deviceName << "\""
                  << " environments=" << observations.environmentCount
                  << " resolution=" << observations.width << 'x'
                  << observations.height << " center_depth=" << depth
                  << " center_semantic=" << semantic
                  << " render_ms=" << rendered.elapsedMilliseconds
                  << " private_bytes=" << renderer.layout().retainedPrivateBytes
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_hybrid_renderer_probe: " << error.what()
                  << '\n';
        return 1;
    }
  }
}
