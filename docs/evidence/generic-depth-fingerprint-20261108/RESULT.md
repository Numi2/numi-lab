# CPU fingerprint probe aborted before qualification

The initial CPU-only Franka probe combined TaskProgram.cpp and headers from the sensing research worktree with a dylib built from a materially different historical owner. Its baseline process aborted with SIGABRT before producing a policy or fingerprint result; the macOS diagnostic identified an invalid free in RunManifest destruction. A repeat also aborted. No GPU or policy artifact was produced.

The saved hashes identify the mismatched headers, objects and binaries. This is evidence of an invalid test setup, not a defect attributed to policy compilation or a compatibility result. A separate matched-owner probe is required. The candidate patch is an isolated discriminator sketch and was not executed successfully.

