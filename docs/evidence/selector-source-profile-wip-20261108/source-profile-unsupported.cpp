#include <metalrobo/c_api.h>
#include <cstdio>
#include <cstring>
int main() {
    MRTaskRolloutConfigC profile{};
    MRRunManifestC manifest{};
    manifest.source = MR_RUN_SOURCE_FRANKA_PICK_PLACE;
    auto *handle = mr_create_task_rollout_with_policy_source(&manifest, &profile);
    const char *error = mr_last_error();
    if (handle || !error || std::strcmp(error, "policy source profile is unsupported for this owner")) {
        std::fprintf(stderr, "unsupported owner unexpectedly accepted or error mismatch: %s\n", error ? error : "(null)");
        return 1;
    }
    std::printf("unsupported owner rejected before device creation: %s\n", error);
    handle = mr_create_task_rollout_with_policy_source(nullptr, &profile);
    error = mr_last_error();
    if (handle || !error || std::strcmp(error, "run manifest is null.")) return 2;
    std::printf("null manifest rejected: %s\n", error);
    handle = mr_create_task_rollout(nullptr);
    error = mr_last_error();
    if (handle || !error || std::strcmp(error, "run manifest is null.")) return 3;
    std::printf("legacy entry point null manifest rejected: %s\n", error);
}
