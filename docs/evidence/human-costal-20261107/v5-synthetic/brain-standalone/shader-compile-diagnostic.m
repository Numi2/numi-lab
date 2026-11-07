#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) { fprintf(stderr, "usage: shader-diagnostic source.metal\n"); return 2; }
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    if (!source) { fprintf(stderr, "source-read error=%s\n", error.description.UTF8String); return 3; }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { fprintf(stderr, "device-unavailable\n"); return 4; }
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    options.mathMode = MTLMathModeFast;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
    options.preprocessorMacros = @{@"NB_PROTECTIVE_HASH_UNROLL": @0};
    fprintf(stderr, "compile-start\n");
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
      fprintf(stderr, "compile-failed domain=%s code=%ld description=%s\n", error.domain.UTF8String, (long)error.code, error.localizedDescription.UTF8String);
      return 5;
    }
    fprintf(stderr, "compile-ok functions=%lu\n", (unsigned long)library.functionNames.count);
    if (error) fprintf(stderr, "compile-warning=%s\n", error.localizedDescription.UTF8String);
    return 0;
  }
}
