# Registered test runner passed with prebuilt candidate owners

A clean CMake configure listed both `metalrobo.upload_abort.initial` and `metalrobo.upload_abort.nonempty`. On 8 November 2026 the initial runner executed 01:19:20.978423–01:19:21.106633 UTC and the nonempty runner 01:19:21.106924–01:19:21.232963 UTC; both exited 0 and pre/post identities matched. Each used the exact registered Python test runner with previously pinned candidate owner executable and shader, comparing independent control/retry output bytes. Detailed per-process status and raw-output evidence is in the earlier matrices.

The configured CMake suite was not freshly built or invoked through CTest for this result; these are runner checks using prebuilt binaries. The test does not qualify shared mode, nonempty cache mutation, excluded program families, post-commit errors, mixed sensor histories, performance, or physical accuracy.
