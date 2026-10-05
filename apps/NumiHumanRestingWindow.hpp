#pragma once
#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#include <functional>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Presentation boundary only. The supplied callback advances the existing
// native owners and returns their completed accepted image and measurements.
struct NumiHumanRestingFrame {
    id<MTLBuffer> rgb = nil;
    unsigned width = 0, height = 0;
    std::string measurements;
    double simulatedSeconds = 0;
    bool complete = false;
};
using NumiHumanRestingAdvance = std::function<NumiHumanRestingFrame(unsigned,unsigned)>;

@interface NumiHumanRestingWindow : NSWindowController <MTKViewDelegate, NSWindowDelegate> {
    NumiHumanRestingAdvance _advance;
    MTKView* _view;
    NSTextField* _measurements;
    NSSegmentedControl* _layers;
    NSSegmentedControl* _cameras;
    id<MTLCommandQueue> _queue;
    id<MTLComputePipelineState> _present;
    AVAssetWriter* _writer;
    AVAssetWriterInput* _writerInput;
    AVAssetWriterInputPixelBufferAdaptor* _adaptor;
    CVMetalTextureCacheRef _textureCache;
    CFTimeInterval _startTime;
    bool _finished;
    int _drawableState;
    bool _inspectionTour;
    double _inspectionPeriodSeconds;
    double _lastSimulatedSeconds;
    std::string _failure;
}
- (instancetype)initWithAdvance:(NumiHumanRestingAdvance)advance movie:(NSString*)movie;
- (void)renderFrameNow;
- (void)finishRecording;
@end

@implementation NumiHumanRestingWindow
- (instancetype)initWithAdvance:(NumiHumanRestingAdvance)advance movie:(NSString*)movie {
    NSWindow* window=[[NSWindow alloc] initWithContentRect:NSMakeRect(0,0,1280,900)
        styleMask:NSWindowStyleMaskTitled|NSWindowStyleMaskClosable|NSWindowStyleMaskMiniaturizable
        backing:NSBackingStoreBuffered defer:NO];
    self=[super initWithWindow:window]; if(!self)return nil;
    _advance=std::move(advance);_finished=false;_drawableState=-1;_startTime=CACurrentMediaTime();
    const char* tour=std::getenv("NUMI_HUMAN_RESTING_INSPECTION_TOUR");
    _inspectionTour=tour&&std::strcmp(tour,"1")==0;_lastSimulatedSeconds=0;
    _inspectionPeriodSeconds=5;
    if(const char* period=std::getenv("NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS")) {
        char* end=nullptr;_inspectionPeriodSeconds=std::strtod(period,&end);
        if(end==period||*end!='\0'||!std::isfinite(_inspectionPeriodSeconds)||_inspectionPeriodSeconds<=0)
            throw std::runtime_error("inspection period must be positive finite seconds");
    }
    window.title=@"Numi Human — supported resting reference";window.delegate=self;
    window.appearance=[NSAppearance appearanceNamed:NSAppearanceNameDarkAqua];
    window.backgroundColor=[NSColor colorWithCalibratedWhite:0.035 alpha:1];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();_queue=[device newCommandQueue];
    _view=[[MTKView alloc] initWithFrame:NSMakeRect(0,150,1280,750) device:device];
    _view.framebufferOnly=NO;_view.colorPixelFormat=MTLPixelFormatBGRA8Unorm;
    _view.preferredFramesPerSecond=30;_view.delegate=self;
    _view.paused=YES;
    [window.contentView addSubview:_view];
    _measurements=[NSTextField wrappingLabelWithString:@"Loading accepted native state…"];
    _measurements.frame=NSMakeRect(24,10,1232,100);
    _measurements.textColor=NSColor.whiteColor;
    _measurements.font=[NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightRegular];
    [window.contentView addSubview:_measurements];
    _layers=[NSSegmentedControl segmentedControlWithLabels:@[@"Skin",@"Muscles",@"Skeleton",@"Organs",@"Lungs",@"Heart",@"Vessels"]
        trackingMode:NSSegmentSwitchTrackingSelectOne target:nil action:nil];
    _layers.frame=NSMakeRect(24,116,620,26);_layers.selectedSegment=0;
    [window.contentView addSubview:_layers];
    _cameras=[NSSegmentedControl segmentedControlWithLabels:@[@"Front",@"Side",@"Chest",@"Whole body",@"Heart detail"]
        trackingMode:NSSegmentSwitchTrackingSelectOne target:nil action:nil];
    _cameras.frame=NSMakeRect(670,116,570,26);_cameras.selectedSegment=3;
    [window.contentView addSubview:_cameras];
    NSError* error=nil;
    NSString* shader=@"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void present_human(device const float4* rgb [[buffer(0)]], constant uint4& config [[buffer(1)]], texture2d<float,access::write> image [[texture(0)]], uint2 p [[thread_position_in_grid]]) { if(p.x>=image.get_width()||p.y>=image.get_height())return; uint2 size=config.xy; float2 extent=float2(image.get_width(),config.z); if(p.y>=config.z){image.write(float4(.035f,.035f,.035f,1),p);return;} float scale=min(extent.x/size.x,extent.y/size.y); float2 uv=(float2(p)-(extent-scale*float2(size))*0.5f)/scale; if(any(uv<0)||any(uv>=float2(size))){image.write(float4(.012f,.019f,.03f,1),p);return;} uint2 q=min(uint2(uv),size-1); float3 c=clamp(rgb[q.y*size.x+q.x].xyz,0.0f,1.0f); c=select(12.92f*c,1.055f*pow(c,float3(1.0f/2.4f))-0.055f,c>0.0031308f); image.write(float4(c,1),p); }";
    id<MTLLibrary> library=[device newLibraryWithSource:shader options:nil error:&error];
    _present=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"present_human"] error:&error];
    if(!_queue||!_present)throw std::runtime_error("native Human presentation pipeline unavailable");
    if(movie.length) {
        _writer=[[AVAssetWriter alloc] initWithURL:[NSURL fileURLWithPath:movie] fileType:AVFileTypeQuickTimeMovie error:&error];
        if(!_writer)throw std::runtime_error("native recording destination unavailable");
        _writerInput=[AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:@{
            AVVideoCodecKey:AVVideoCodecTypeH264,AVVideoWidthKey:@1280,AVVideoHeightKey:@900}];
        _writerInput.expectsMediaDataInRealTime=YES;
        _adaptor=[AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:_writerInput
            sourcePixelBufferAttributes:@{(NSString*)kCVPixelBufferPixelFormatTypeKey:@(kCVPixelFormatType_32BGRA),
            (NSString*)kCVPixelBufferWidthKey:@1280,(NSString*)kCVPixelBufferHeightKey:@900,
            (NSString*)kCVPixelBufferMetalCompatibilityKey:@YES}];
        [_writer addInput:_writerInput];
        if(![_writer startWriting])throw std::runtime_error("native recording could not start");
        [_writer startSessionAtSourceTime:kCMTimeZero];
        CVMetalTextureCacheCreate(kCFAllocatorDefault,nil,device,nil,&_textureCache);
    }
    [window center];return self;
}
- (void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)size { (void)view;(void)size; }
- (void)renderFrameNow {
    NSEvent* event;
    while((event=[NSApp nextEventMatchingMask:NSEventMaskAny untilDate:[NSDate distantPast]
               inMode:NSDefaultRunLoopMode dequeue:YES]))[NSApp sendEvent:event];
    if(!self.window.visible)throw std::runtime_error("native resting viewer closed by user");
    [self drawInMTKView:_view];[NSApp updateWindows];
    if(!_failure.empty())throw std::runtime_error(_failure);
}
- (void)finishRecording {
    if(_finished)return;_finished=true;_view.paused=YES;
    if(_writer) {
        [_writerInput markAsFinished];auto completion=dispatch_semaphore_create(0);
        [_writer finishWritingWithCompletionHandler:^{dispatch_semaphore_signal(completion);}];
        if(dispatch_semaphore_wait(completion,dispatch_time(DISPATCH_TIME_NOW,15*NSEC_PER_SEC))!=0||
           _writer.status!=AVAssetWriterStatusCompleted) {
            _failure="native recording did not complete";NSLog(@"Human recording failed: %@",_writer.error);
        }
    }
}
- (void)drawInMTKView:(MTKView*)view {
    if(_finished)return;
    @autoreleasepool {try {
        if(_inspectionTour) {
            // Optional presentation-only tour of the same continuing state.
            // It never changes the owner clock, forces, or intervention.
            const unsigned layer=unsigned(std::fmod(std::floor(_lastSimulatedSeconds/_inspectionPeriodSeconds),7.0));
            _layers.selectedSegment=layer;
            _cameras.selectedSegment=layer==5u?4u:layer==4u?2u:3u;
        }
        const auto frame=_advance(unsigned(_cameras.selectedSegment),unsigned(_layers.selectedSegment));
        _lastSimulatedSeconds=frame.simulatedSeconds;
        _measurements.stringValue=[NSString stringWithFormat:@"%@ / %@%s\n%s",
            [_layers labelForSegment:_layers.selectedSegment],[_cameras labelForSegment:_cameras.selectedSegment],
            _inspectionTour?" (inspection tour)":"",frame.measurements.c_str()];
        auto drawable=view.currentDrawable;
        if(!frame.rgb)throw std::runtime_error("native viewer has no accepted image");
        if(!drawable&&!_writer)throw std::runtime_error("native window drawable unavailable and no framebuffer recording was requested");
        if(_drawableState!=int(drawable!=nil)) {
            _drawableState=int(drawable!=nil);
            NSLog(@"Human native framebuffer output: %@",drawable?@"window and recording":@"recording; window drawable unavailable");
        }
        auto command=[_queue commandBuffer];auto encoder=[command computeCommandEncoder];
        auto encode=[&](id<MTLTexture> texture,unsigned pictureHeight) {
            const simd_uint4 size={frame.width,frame.height,pictureHeight,0};
            [encoder setComputePipelineState:_present];[encoder setBuffer:frame.rgb offset:0 atIndex:0];
            [encoder setBytes:&size length:sizeof(size) atIndex:1];[encoder setTexture:texture atIndex:0];
            [encoder dispatchThreads:MTLSizeMake(texture.width,texture.height,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        };
        if(drawable)encode(drawable.texture,unsigned(drawable.texture.height));
        CVPixelBufferRef pixels=nullptr;CVMetalTextureRef movieTexture=nullptr;
        if(_writer) {
            const auto deadline=CACurrentMediaTime()+1;
            while(!_writerInput.readyForMoreMediaData&&CACurrentMediaTime()<deadline)
                [NSThread sleepForTimeInterval:.002];
            if(!_writerInput.readyForMoreMediaData||!_adaptor.pixelBufferPool||
               CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault,_adaptor.pixelBufferPool,&pixels)!=kCVReturnSuccess)
                throw std::runtime_error("native recording cannot accept the next continuous frame");
            if(CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,_textureCache,pixels,nil,
                MTLPixelFormatBGRA8Unorm,1280,900,0,&movieTexture)!=kCVReturnSuccess||!movieTexture)
                throw std::runtime_error("native recording texture allocation failed");
            encode(CVMetalTextureGetTexture(movieTexture),750);
        }
        [encoder endEncoding];if(drawable)[command presentDrawable:drawable];[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("native presentation command failed");
        if(movieTexture) {
            // Composite the actual native control/measurement view into the
            // recording. This presentation-only AppKit work never updates a
            // physical state or synthesizes a physiological value.
            NSRect rect=NSMakeRect(0,0,1280,150);
            auto bitmap=[self.window.contentView bitmapImageRepForCachingDisplayInRect:rect];
            [self.window.contentView cacheDisplayInRect:rect toBitmapImageRep:bitmap];
            if(!bitmap||!bitmap.CGImage)throw std::runtime_error("native recording measurement panel unavailable");
            CVPixelBufferLockBaseAddress(pixels,0);
            auto color=CGColorSpaceCreateDeviceRGB();
            auto context=CGBitmapContextCreate(CVPixelBufferGetBaseAddress(pixels),1280,900,8,
                CVPixelBufferGetBytesPerRow(pixels),color,CGBitmapInfo(kCGBitmapByteOrder32Little)|CGBitmapInfo(kCGImageAlphaPremultipliedFirst));
            if(!context)throw std::runtime_error("native recording measurement composition failed");
            // CVPixelBuffer's CG bitmap origin is the lower edge here. The
            // AppKit cache already has an upright image; a second Y flip
            // inverted its text in the first native recording.
            CGContextDrawImage(context,CGRectMake(0,0,1280,150),bitmap.CGImage);CGContextRelease(context);
            CGColorSpaceRelease(color);CVPixelBufferUnlockBaseAddress(pixels,0);
            // The video uses elapsed wall time. The compact trace separately
            // records exact simulated time, so slow execution is never hidden.
            const CMTime timestamp=CMTimeMakeWithSeconds(CACurrentMediaTime()-_startTime,60000);
            if(![_adaptor appendPixelBuffer:pixels withPresentationTime:timestamp])
                throw std::runtime_error("native recording frame append failed");
            CFRelease(movieTexture);
        }
        if(pixels)CVPixelBufferRelease(pixels);
        if(frame.complete)[self finishRecording];
    }catch(const std::exception& error){
        _failure=error.what();
        _measurements.stringValue=[NSString stringWithFormat:@"Stopped: %s",error.what()];
        NSLog(@"Human scene stopped: %s",error.what());[self finishRecording];
    }}
}
- (void)windowWillClose:(NSNotification*)notification {
    (void)notification;[self finishRecording];[NSApp stop:nil];
}
- (void)dealloc {if(_textureCache)CFRelease(_textureCache);}
@end

inline void runNumiHumanRestingWindow(NumiHumanRestingAdvance advance,const std::string& movie) {
    [NSApplication sharedApplication];[NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    auto owner=[[NumiHumanRestingWindow alloc] initWithAdvance:std::move(advance)
        movie:[NSString stringWithUTF8String:movie.c_str()]];
    [owner showWindow:nil];[NSApp activateIgnoringOtherApps:YES];[NSApp run];
}
