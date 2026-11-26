#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static int report(NSString *operation, NSError *error) {
    NSString *detail = error.localizedDescription ?: @"function missing from shader library";
    fprintf(stderr, "%s: %s\n", operation.UTF8String, detail.UTF8String);
    return 1;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s SHADER_LIBRARY PIPELINE_SCRIPT OUTPUT_ARCHIVE\n", argv[0]);
            return 2;
        }
        NSURL *libraryURL = [NSURL fileURLWithPath:@(argv[1])];
        NSURL *scriptURL = [NSURL fileURLWithPath:@(argv[2])];
        NSURL *outputURL = [NSURL fileURLWithPath:@(argv[3])];
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device available\n");
            return 1;
        }

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];
        if (!library) return report(@"Load Metal library", error);
        NSData *scriptData = [NSData dataWithContentsOfURL:scriptURL options:0 error:&error];
        if (!scriptData) return report(@"Load pipeline script", error);
        id script = [NSJSONSerialization JSONObjectWithData:scriptData options:0 error:&error];
        if (!script) return report(@"Read pipeline script", error);
        id pipelines = [script isKindOfClass:NSDictionary.class] ? script[@"pipelines"] : nil;
        id entries = [pipelines isKindOfClass:NSDictionary.class] ? pipelines[@"compute_pipelines"] : nil;
        if (![entries isKindOfClass:NSArray.class] || [entries count] == 0) {
            fprintf(stderr, "No compute pipeline descriptors\n");
            return 1;
        }

        id<MTLBinaryArchive> archive =
            [device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];
        if (!archive) return report(@"Create Metal archive", error);
        for (id entry in entries) {
            id function = [entry isKindOfClass:NSDictionary.class] ? entry[@"compute_function"] : nil;
            if (![function isKindOfClass:NSString.class] || ![function hasPrefix:@"alias:MetalRobo#"]) {
                fprintf(stderr, "Invalid compute function in pipeline script\n");
                return 1;
            }
            NSString *name = [function substringFromIndex:[@"alias:MetalRobo#" length]];
            MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
            descriptor.computeFunction = [library newFunctionWithName:name];
            if (!descriptor.computeFunction) return report(name, nil);
            if (![archive addComputePipelineFunctionsWithDescriptor:descriptor error:&error]) {
                return report(name, error);
            }
        }
        if (![archive serializeToURL:outputURL error:&error]) return report(@"Serialize Metal archive", error);
        fprintf(stderr, "%s: wrote %lu compute pipelines to %s\n",
                device.name.UTF8String, (unsigned long)[entries count], outputURL.path.UTF8String);
    }
    return 0;
}
