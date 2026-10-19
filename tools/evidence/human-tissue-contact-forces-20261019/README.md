# Registered wall contact fixture evidence

This bundle records one bounded contact-only source replay plus two retained failures. It is an experimental reduced regional fixture, not whole-body or physiological qualification.

## Passing contact-only replay

The probe uses a reduced 532-node / 1,584-tet wall and a synthetic local incoming tetrahedron, with prescribed pressure of 13,300 Pa. It runs a separated control, then five accepted approach frames at 0.5 m/s and 0.25 ms per step. The accepted approach covers 1.25 ms; a pristine continuation accepts step 6 (total accepted approach horizon 1.5 ms). A fresh run from the same five-step checkpoint injects a late rejection at step 6, verifies exact rollback, then accepts the retry. The injected failed attempt is not counted as accepted simulation time.

Accepted active barrier-row counts for approach steps 1–5 were 0, 0, 50, 34, 32. The fixed-node contact-force norms were 0, 0, 0.701559001, 0.107798081, and 0.0266988924 N. The no-touch control had no active rows or contact force. The fixture explicitly makes no momentum-balance claim. The synthetic contact tetrahedron and fixed-node reaction probe do not qualify anatomical tissue interactions, physiological loading, or CVSim hydraulics.

Executed on SSH `numi`: Apple M6 Mac mini (Mac18,5), 16 GB RAM, macOS 27.0 build 26A425. DYLD verified runtime SHA-256 `8fb535de0280cb989bb8edf5d6ed6372974edf55d992651fb51543edfa2f0b0b`. Invocation time was 10.600 s, including initialization. This is not a steady-state real-time-factor measurement. No visual verification was performed.

The production change is confined to `matter/src/metal/contact.metalinc`. Every already-certified legacy CCD result is retained. Only uncertified triangle-pair narrowphase uses endpoint contact, fixed separating-plane advancement, and a bounded 64-evaluation vertex-triangle interval fallback. Actual sampled overlap provides contact geometry; a clear result requires the complete interval to be certified. Unresolved cases remain failures. Newton line-search sweep certification, contact thickness, physical mass, and state ownership are unchanged. No per-step CPU physics or new state buffer is introduced.

Exact pins:

- Probe source: `2d73e5ef1bedead9ff1938154acf8c06590557d3c8a549dc672320f4c557ab1b`
- Probe executable: `bccac1c057cc88ecb73fb9068173b4d415a2e5032f3fa653d8eab8b5f47ac581`
- Contact-only Metal source: `13af0a43bfe7b9079b8d62f655d3ec16038cbce4e58380f5549a20646a886e15`
- Loaded runtime library: `8fb535de0280cb989bb8edf5d6ed6372974edf55d992651fb51543edfa2f0b0b`
- Full argv and environment: `invocation-result.json`
- Full output: `native.stdout` and `native.stderr`

The launch argv was:

```text
/Users/n/numi-human-separation-20261018/C/aortic-wall-outer-contact-260/force-runtime-002/retry-diagnostic-001/build/aortic_wall_force_probe /Users/n/numi-human-separation-20261018/C/tissue-contact-publication-268/MatterSource-contact-only-001.metal
```

## Preserved negative evidence

`prior-e303-retry-pristine-classification.json` preserves the earlier e303 result: after the same five accepted contact frames, a pristine sixth step and the retry both failed with status 6 at the same active pair/index. The retry snapshot retained the injected-reject contact record, which is flagged as stale rather than attributed to retry status. This earlier failure is not overwritten by the contact-only passing replay.

`strict-interval270-status10-summary.json` preserves the distinct strict 1e-6 kidney/spleen result: separated controls passed, then the approach failed at frame 3 with Matter status 10, object 0, index 8, FGMRES 20, and no populated CCD witness. It is not evidence about the wall fixture.

## Native integrated smoke

The retained native human scene also accepted one 0.25 ms step with the contact-only shader and a matching guarded adapter, using unchanged runtime261. The coupled physiology CSV is byte-identical to the prior accepted 261 run. `integrated-invocation.json`, `integrated-result.json`, and `integrated-comparison.json` retain source/library/launch pins and body/physiology clock evidence. Launch took 54.929 s; reported physical-horizon wall time was 42.363 s. This one-step initialization-heavy smoke is not a sustained performance measurement or the broader all-tissue qualification. No visual verification was performed.
