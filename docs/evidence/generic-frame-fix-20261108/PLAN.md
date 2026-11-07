# Preregistered second generic frame/depth candidate

This separate research-only shader candidate requires both FRAME and DEPTH validity for measured materialized-plane pixels and their edge neighbors, and only permits materialized edge filling into a frame-present pixel. In the direct-winner mode it retains the assumption that the renderer produced a frame; that path does not independently convey a missing-frame signal. No product source or policy fingerprint is changed.

Run three already-built immutable executables in sequence against the candidate metallib, each with an independent 120-second timeout and stopping on first failure:

1. The original generic depth validity executable from 8e836afd: baseline plus raw, identity-selected and direct-winner dropout/noise gates.
2. The mixed/full missing-frame executable from 5fdc403a: measured-only neighbor donation, frame-present fill, no-frame far, and fully absent frame.
3. The 160×120 real Franka renderer executable from b8e3d790: actual normal and 100%-dropout planes, edge flicker off and on, CPU reference, with original shader as a retained comparator.

Pin all binaries and shaders, record per-process UTC start/end and exact exits, verify hashes before/after and retain all outputs including partial/failure. No adaptive retry. These gates do not establish general renderer-profile behavior, full physics/policy transactions, physical calibration, exact replay, or direct-winner missing-frame semantics. Edge fill into frame-present depth-invalid pixels is a proposed explicit corruption contract, not an established prior promise. Fingerprint/version migration remains required before integration.

