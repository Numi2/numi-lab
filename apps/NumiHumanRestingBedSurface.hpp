#pragma once
#include "metalrobo/numi_human_resting_bed_gpu.h"
#include <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

// Reads the optional surface from the existing resting source-scene manifest.
// This is an authored fixed support environment, not participant anatomy.
struct NumiHumanRestingBedSurface {
    MRHumanRestingBedGPU gpu{};
    std::vector<float> heights;
    std::string sourceCaptureSHA256;
    static void require(bool valid, const char* message) {
        if (!valid) throw std::runtime_error(message);
    }
    static double number(id value) {
        require([value isKindOfClass:[NSNumber class]], "bed field must be numeric");
        const double x=[value doubleValue];
        require(std::isfinite(x), "bed field must be finite");
        return x;
    }
    static std::string text(id value) {
        require([value isKindOfClass:[NSString class]], "bed field must be text");
        return std::string([(NSString*)value UTF8String]);
    }
    NumiHumanRestingBedSurface(const std::filesystem::path& path,
        const std::string& skinSHA256, const std::string& supportSHA256) {
        NSData* data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
        require(data!=nil, "resting bed source-scene manifest unavailable");
        NSError* error=nil;
        id root=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        require(error==nil && [root isKindOfClass:[NSDictionary class]], "invalid resting bed source-scene JSON");
        require(text(root[@"schema"])=="numi.human.resting-supine-source-scene.v1",
            "bed must extend the existing resting source-scene manifest");
        require(text(root[@"source"][@"skin"][@"sha256"])==skinSHA256,
            "bed manifest binds a different registered skin");
        require(text(root[@"outputs"][@"support_contact"][@"sha256"])==supportSHA256,
            "bed manifest binds different support seeds");
        id h=root[@"bed"][@"heightfield"];
        require([h isKindOfClass:[NSDictionary class]], "resting bed manifest has no heightfield");
        require(text(h[@"frame"])=="fixed_world" &&
                text(h[@"triangulation"])=="00_10_01__10_11_01",
                "unsupported bed frame or grid diagonal");
        const double nx=number(h[@"nx"]),ny=number(h[@"ny"]);
        require(nx>=2 && ny>=2 && nx<=1024 && ny<=1024 &&
                std::floor(nx)==nx && std::floor(ny)==ny && nx*ny<=262144,
                "bed dimensions exceed bounded native capacity");
        id origin=h[@"origin_xy_m"],spacing=h[@"spacing_xy_m"];
        require([origin isKindOfClass:[NSArray class]] && [origin count]==2 &&
                [spacing isKindOfClass:[NSArray class]] && [spacing count]==2,
                "bed origin/spacing dimensions invalid");
        gpu.counts={unsigned(nx),unsigned(ny),1,1};
        gpu.originSpacing={float(number(origin[0])),float(number(origin[1])),
            float(number(spacing[0])),float(number(spacing[1]))};
        require(std::isfinite(gpu.originSpacing.x) && std::isfinite(gpu.originSpacing.y) &&
                gpu.originSpacing.z>=.001f && gpu.originSpacing.z<=.1f &&
                gpu.originSpacing.w>=.001f && gpu.originSpacing.w<=.1f,
                "bed spacing is not representable or outside 1-100 mm");
        for(unsigned axis=0;axis<2;++axis) {
            const unsigned count=axis?gpu.counts.y:gpu.counts.x;
            const float origin=axis?gpu.originSpacing.y:gpu.originSpacing.x;
            const float step=axis?gpu.originSpacing.w:gpu.originSpacing.z;
            float previous=origin;
            for(unsigned index=1;index<count;++index) {
                const float node=std::fma(float(index),step,origin);
                require(std::isfinite(node) && node>previous,
                    "bed grid nodes must be finite and strictly monotonic in Float32");
                previous=node;
            }
        }
        id values=h[@"heights_m"];
        require([values isKindOfClass:[NSArray class]] && [values count]==size_t(nx*ny),
                "bed height count differs from its grid");
        heights.reserve(size_t(nx*ny));
        for(id value in values) {
            const float z=float(number(value));
            require(std::isfinite(z) && z>=-.5f && z<=.5f,
                    "bed height is not representable or outside declared resting bounds");
            heights.push_back(z);
        }
        sourceCaptureSHA256=text(h[@"source_capture_sha256"]);
        require(sourceCaptureSHA256.size()==64 &&
            sourceCaptureSHA256.find_first_not_of("0123456789abcdef")==std::string::npos,
            "bed source capture identity is missing");
        // A flat rim closes onto the existing zero-height outer bed boundary.
        for(unsigned y=0;y<gpu.counts.y;++y)for(unsigned x=0;x<gpu.counts.x;++x)
            if(x==0||y==0||x+1==gpu.counts.x||y+1==gpu.counts.y)
                require(heights[y*gpu.counts.x+x]==0.0f,"bed finite boundary must meet its zero-height rim");
    }
};
