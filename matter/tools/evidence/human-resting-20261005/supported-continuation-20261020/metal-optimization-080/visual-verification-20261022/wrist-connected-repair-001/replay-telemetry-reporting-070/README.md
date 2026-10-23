# Replay telemetry reporting

The selected hand/wrist JSON field labeled an unrequested replay as failed. The deterministicReplayVerified flag defaults false and becomes true only when the existing verifyDeterminism branch executes. The overall stand summary already labels this case not_requested. This patch makes the selected-cohort field consistent: verified is bitwise, requested but unverified is failed, and unrequested is not_requested. Replay checks and physical acceptance are unchanged.

The patched native translation unit compiles with strict warnings as errors. No native simulation was repeated for this reporting-only change. Original D19 logs are retained unchanged; their replay=failed text is contradicted by the invocation and source ownership. This correction does not establish replay qualification.
