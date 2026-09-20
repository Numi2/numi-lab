#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>

// Preflight library loading AND pipeline creation before expensive Human
// initialization. Successful compilation alone does not establish support
// on a hosted virtual GPU. This is capability evidence, not simulation proof.
int main(int argc, char** argv) {
    @autoreleasepool {
        NSMutableDictionary* report = [NSMutableDictionary dictionary];
        report[@"schema"] = @"numi.human.metal4-device-preflight.v1";
        report[@"os"] = NSProcessInfo.processInfo.operatingSystemVersionString;
        report[@"full_human_qualification"] = @NO;
        int code = 2;
        if (argc != 2) {
            report[@"error"] = @"usage: check-human-metal4 /path/to/probe.metallib";
        } else {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            report[@"device_available"] = @(device != nil);
            if (device == nil) {
                report[@"error"] = @"No Metal device exposed";
            } else {
                report[@"device"] = device.name;
                NSError* error = nil;
                NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
                id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
                report[@"library_loaded"] = @(library != nil);
                if (library == nil) {
                    report[@"error"] = error.localizedDescription ?: @"Metal 4 library load failed";
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:@"probe"];
                    report[@"function_found"] = @(function != nil);
                    if (function == nil) {
                        report[@"error"] = @"Probe entry point missing";
                    } else {
                        error = nil;
                        id<MTLComputePipelineState> pipeline =
                            [device newComputePipelineStateWithFunction:function error:&error];
                        report[@"pipeline_created"] = @(pipeline != nil);
                        if (pipeline == nil) {
                            report[@"error"] = error.localizedDescription ?: @"Metal 4 pipeline creation failed";
                        } else {
                            code = 0;
                        }
                    }
                }
            }
        }
        report[@"status"] = code == 0 ? @"ready_for_execution_attempt" : @"unsupported_or_invalid";
        NSData* json = [NSJSONSerialization dataWithJSONObject:report
            options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:nil];
        if (json == nil) return 3;
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);
        return code;
    }
}
