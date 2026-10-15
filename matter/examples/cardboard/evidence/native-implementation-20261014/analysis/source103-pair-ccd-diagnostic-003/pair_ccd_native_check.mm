#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr const char* kSchema = "numi.cardboard.source103_pair_ccd_inputs.v1";
constexpr const char* kKernel = "numi_matter_metal::nm_cardboard_source103_pair_ccd_diagnostic";
constexpr const char* kExpectedContactSHA =
    "427aca061ad2c6d2218b5eae0bc7620132a1ef9fff9f688390ccc96ca8a27ffc";
constexpr std::size_t kCaseCount = 30u;
constexpr std::size_t kFloat4Bytes = 16u;
constexpr std::size_t kInputRecordsPerCase = 9u;
constexpr std::size_t kSummaryRecordsPerCase = 9u;
constexpr std::size_t kTraceRecordsPerCase = 17u;

std::string sha256(const void* data, const std::size_t size) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    if (CC_SHA256(data, static_cast<CC_LONG>(size), digest) == nullptr)
        throw std::runtime_error("SHA-256 failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string out; out.reserve(64u);
    for (unsigned char x : digest) { out.push_back(hex[x >> 4u]); out.push_back(hex[x & 15u]); }
    return out;
}
std::string sha256File(NSString* path) {
    NSData* data = [NSData dataWithContentsOfFile:path];
    if (data == nil) throw std::runtime_error("could not read file for SHA-256");
    return sha256(data.bytes, data.length);
}
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
NSString* nsPath(const char* value) { return [NSString stringWithUTF8String:value]; }
NSNumber* finiteNumber(float value) {
    return std::isfinite(value) ? @(value) : (NSNumber*)[NSNull null];
}
NSString* bitString(float value) {
    std::uint32_t bits=0; std::memcpy(&bits,&value,sizeof(bits));
    return [NSString stringWithFormat:@"%08x", bits];
}
NSArray* vector3(const float* row) {
    return @[finiteNumber(row[0]), finiteNumber(row[1]), finiteNumber(row[2])];
}
NSArray* bits4(const float* row) {
    return @[bitString(row[0]),bitString(row[1]),bitString(row[2]),bitString(row[3])];
}
NSArray* values4(const float* row) {
    return @[finiteNumber(row[0]),finiteNumber(row[1]),finiteNumber(row[2]),finiteNumber(row[3])];
}
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            if (argc != 5) {
                std::cerr << "usage: pair_ccd_native_check <prepared.json> <cases.bin> <diagnostic.metallib> <output.json>\n";
                return 2;
            }
            NSString* preparedPath=nsPath(argv[1]);
            NSString* inputPath=nsPath(argv[2]);
            NSString* libraryPath=nsPath(argv[3]);
            NSString* outputPath=nsPath(argv[4]);
            NSData* preparedData=[NSData dataWithContentsOfFile:preparedPath];
            NSData* inputData=[NSData dataWithContentsOfFile:inputPath];
            require(preparedData != nil && inputData != nil, "could not read prepared manifest or cases.bin");
            NSError* jsonError=nil;
            id parsed=[NSJSONSerialization JSONObjectWithData:preparedData options:NSJSONReadingFragmentsAllowed error:&jsonError];
            require(jsonError == nil && [parsed isKindOfClass:[NSDictionary class]], "prepared manifest is not strict JSON object");
            NSDictionary* prepared=(NSDictionary*)parsed;
            require([prepared[@"schema"] isEqualToString:@(kSchema)], "prepared schema mismatch");
            require([prepared[@"source_contact_sha256"] isEqualToString:@(kExpectedContactSHA)], "prepared source-103 contact hash mismatch");
            NSDictionary* contract=prepared[@"case_contract"];
            require([contract[@"total"] unsignedIntegerValue]==kCaseCount, "prepared case count mismatch");
            require([contract[@"input_record_bytes"] unsignedIntegerValue]==kFloat4Bytes, "prepared float4 stride mismatch");
            const std::size_t expectedBytes=kCaseCount*kInputRecordsPerCase*kFloat4Bytes;
            require(inputData.length==expectedBytes, "cases.bin byte count mismatch");
            const std::string inputSHA=sha256(inputData.bytes,inputData.length);
            require([contract[@"input_sha256"] isEqualToString:[NSString stringWithUTF8String:inputSHA.c_str()]], "cases.bin SHA mismatch");
            NSArray* caseRows=prepared[@"cases"];
            require([caseRows isKindOfClass:[NSArray class]] && caseRows.count==kCaseCount, "prepared case metadata count mismatch");
            require([prepared[@"source_common_sha256"] isKindOfClass:[NSString class]], "common source binding missing");
            require([prepared[@"source_main_shader_sha256"] isKindOfClass:[NSString class]], "main shader source binding missing");

            NSError* libraryError=nil;
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            require(device != nil, "no Metal device available");
            id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:libraryPath] error:&libraryError];
            require(library != nil, libraryError.localizedDescription.UTF8String ?: "could not load diagnostic metallib");
            id<MTLFunction> function=[library newFunctionWithName:@(kKernel)];
            require(function != nil, "diagnostic kernel missing from metallib");
            id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&libraryError];
            require(pipeline != nil, libraryError.localizedDescription.UTF8String ?: "could not create diagnostic compute pipeline");

            std::vector<float> summaries(kCaseCount*kSummaryRecordsPerCase*4u,0.0f);
            std::vector<float> traces(kCaseCount*kTraceRecordsPerCase*4u,0.0f);
            id<MTLBuffer> casesBuffer=[device newBufferWithBytes:inputData.bytes length:inputData.length options:MTLResourceStorageModeShared];
            id<MTLBuffer> summaryBuffer=[device newBufferWithBytes:summaries.data() length:summaries.size()*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> traceBuffer=[device newBufferWithBytes:traces.data() length:traces.size()*sizeof(float) options:MTLResourceStorageModeShared];
            std::uint32_t count=static_cast<std::uint32_t>(kCaseCount);
            id<MTLBuffer> countBuffer=[device newBufferWithBytes:&count length:sizeof(count) options:MTLResourceStorageModeShared];
            require(casesBuffer && summaryBuffer && traceBuffer && countBuffer, "Metal buffer allocation failed");
            id<MTLCommandQueue> queue=[device newCommandQueue];
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:casesBuffer offset:0 atIndex:0];
            [encoder setBuffer:summaryBuffer offset:0 atIndex:1];
            [encoder setBuffer:traceBuffer offset:0 atIndex:2];
            [encoder setBuffer:countBuffer offset:0 atIndex:3];
            const NSUInteger group=std::max<NSUInteger>(1u,std::min<NSUInteger>(32u,pipeline.maxTotalThreadsPerThreadgroup));
            [encoder dispatchThreads:MTLSizeMake(kCaseCount,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            require(command.status==MTLCommandBufferStatusCompleted,
                command.error.localizedDescription.UTF8String ?: "Metal diagnostic dispatch failed");
            std::memcpy(summaries.data(),summaryBuffer.contents,summaries.size()*sizeof(float));
            std::memcpy(traces.data(),traceBuffer.contents,traces.size()*sizeof(float));

            NSMutableArray* results=[NSMutableArray arrayWithCapacity:kCaseCount];
            bool allClonesMatch=true;
            for (std::size_t i=0;i<kCaseCount;++i) {
                const float* s=summaries.data()+i*kSummaryRecordsPerCase*4u;
                const bool match=
                    std::memcmp(s,s+4u*4u,4u*sizeof(float))==0 &&
                    std::memcmp(s+1u*4u,s+5u*4u,4u*sizeof(float))==0 &&
                    std::memcmp(s+2u*4u,s+6u*4u,4u*sizeof(float))==0 &&
                    std::memcmp(s+3u*4u,s+7u*4u,sizeof(float))==0;
                allClonesMatch=allClonesMatch && match;
                NSMutableArray* traceRows=[NSMutableArray array];
                const float* trace=traces.data()+i*kTraceRecordsPerCase*4u;
                const std::uint32_t iterations=std::min<std::uint32_t>(17u,static_cast<std::uint32_t>(s[7u*4u+1u]));
                for (std::uint32_t j=0;j<iterations;++j) {
                    const float* row=trace+4u*j;
                    [traceRows addObject:@{@"iteration":@(j),@"time":finiteNumber(row[0]),@"distance_m":finiteNumber(row[1]),@"step":finiteNumber(row[2]),@"terminal_code":@(static_cast<std::uint32_t>(row[3]))}];
                }
                NSMutableDictionary* item=[NSMutableDictionary dictionaryWithDictionary:caseRows[i]];
                item[@"production_result"]=@{
                    @"hit":@(s[0]==1.0f),@"certified":@(s[1]==1.0f),@"toi":finiteNumber(s[2]),@"separation_m":finiteNumber(s[3]),
                    @"normal":vector3(s+4u),@"feature_parameter0":finiteNumber(s[7]),
                    @"witness_point":vector3(s+8u),@"feature_parameter1":finiteNumber(s[11]),
                    @"feature_parameter2":finiteNumber(s[12]),@"raw_float4_bits":@[
                        bits4(s),bits4(s+4u),bits4(s+8u),bits4(s+12u)]};
                item[@"trace_clone_result"]=@{
                    @"hit":@(s[16u]==1.0f),@"certified":@(s[17u]==1.0f),@"toi":finiteNumber(s[18]),@"separation_m":finiteNumber(s[19]),
                    @"normal":vector3(s+20u),@"feature_parameter0":finiteNumber(s[23]),
                    @"witness_point":vector3(s+24u),@"feature_parameter1":finiteNumber(s[27]),
                    @"feature_parameter2":finiteNumber(s[28]),@"iteration_count":@(iterations),
                    @"terminal_code":@(static_cast<std::uint32_t>(s[30u])),@"effective_thickness_m":finiteNumber(s[31]),
                    @"speed_bound_m":finiteNumber(s[32]),@"raw_float4_bits":@[
                        bits4(s+16u),bits4(s+20u),bits4(s+24u),bits4(s+28u),bits4(s+32u)]};
                item[@"production_and_trace_clone_bitwise_equal"]=@(match);
                item[@"native_conservative_advance_trace"]=traceRows;
                [results addObject:item];
            }
            const std::string outputPayloadHash=sha256(summaries.data(),summaries.size()*sizeof(float));
            const std::string traceHash=sha256(traces.data(),traces.size()*sizeof(float));
            NSDictionary* output=@{
                @"schema":@"numi.cardboard.source103_pair_ccd_native_results.v1",
                @"source_snapshot":prepared[@"source_snapshot"],
                @"source_contact_sha256":prepared[@"source_contact_sha256"],
                @"source_common_sha256":prepared[@"source_common_sha256"],
                @"source_main_shader_sha256":prepared[@"source_main_shader_sha256"],
                @"prepared_manifest_sha256":[NSString stringWithUTF8String:sha256(preparedData.bytes,preparedData.length).c_str()],
                @"cases_bin_sha256":[NSString stringWithUTF8String:inputSHA.c_str()],
                @"metallib_path":libraryPath,
                @"metallib_sha256":[NSString stringWithUTF8String:sha256File(libraryPath).c_str()],
                @"device_name":device.name,
                @"diagnostic_execution_only":@YES,
                @"physical_transactions_executed":@0,
                @"captured_input_case_count":@(kCaseCount),
                @"production_vs_trace_clone_all_bitwise_equal":@(allClonesMatch),
                @"summary_payload_sha256":[NSString stringWithUTF8String:outputPayloadHash.c_str()],
                @"trace_payload_sha256":[NSString stringWithUTF8String:traceHash.c_str()],
                @"cases":results};
            NSError* writeError=nil;
            NSData* resultData=[NSJSONSerialization dataWithJSONObject:output options:NSJSONWritingPrettyPrinted error:&writeError];
            require(resultData != nil && writeError == nil, "could not serialize strict native result JSON");
            require([resultData writeToFile:outputPath options:NSDataWritingAtomic error:&writeError], "could not write result JSON");
            std::cout << "source103_pair_ccd_native_diagnostic=COMPLETED cases=" << kCaseCount
                      << " clone_bitwise_equal=" << (allClonesMatch?"true":"false")
                      << " output=" << outputPath.UTF8String << "\n";
            return allClonesMatch ? 0 : 4;
        } catch (const std::exception& e) {
            std::cerr << "source103_pair_ccd_native_diagnostic=ERROR " << e.what() << "\n";
            return 1;
        }
    }
}
