#ifndef METALROBO_MRNX_HUMAN_BEHAVIOR_V1_H
#define METALROBO_MRNX_HUMAN_BEHAVIOR_V1_H
#include "metalrobo/mrnx_bridge_v1.h"
#ifdef __cplusplus
extern "C" {
#endif
// Optional privileged measurement channel; never part of Brain observations.
// Attach once before any physical attempt. The expected SHA256 is 64 lower-case
// hex characters. Source semantic criteria are verified against actual assets.
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_runtime_behavior_attach(
    mrnx_runtime_v1* runtime,const char* metric_program_path,const char* expected_sha256,
    uint64_t initial_committed_timestamp_nanoseconds);
// Explicit final collector. Uses the original owner queue and advances no
// physical state/time. Return required bytes INCLUDING NUL, or0 on failure.
// Output includes explicit unknown audit/contact/reset coverage. This is not
// an accepted-root behavior-trial.v1 trace or a generic TaskPack receipt.
MRNX_BRIDGE_EXPORT size_t mrnx_bridge_v1_runtime_behavior_flush_json(
    mrnx_runtime_v1* runtime,char* output,size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
