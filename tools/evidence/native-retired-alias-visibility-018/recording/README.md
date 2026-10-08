# Native viewer 018 recording review

This review decodes the completed viewer-018 run at:

/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/skin-927-lung-924-viewer-018-v014-final-attempt3/native-run

The primary decode uses the pinned 942 AVFoundation forward-reader script at /Users/n/numi-human-resting-evidence-20261005/native-current-recording-review-873/movie-continuity-even-samples.swift (SHA-256 dad3a672ad34fb0f46d9c5c8a1cb6f457097ae95ccd44c1ad4f946b941d1a58c). Its first pass reads the movie in decode order and checks PTS monotonicity; its second forward-only pass extracts frames with exact PTS-sequence checks. The 25-sample script in sibling ../native-viewer018-recording-review-960-dense/decoder-25-samples.swift changes only sample count and selection-rule wording, to capture the Organs layer.

The movie SHA-256 is 46c72bc48cbb49e65bd521554d8fdd0f9f78ab5464664d22e39a746c7d06b5e8. Decode found 316 frames: one black startup frame at PTS 0 and 315 nonblank frames, matching the 315 surface-audit rows in decode order. PTS is monotonic, but intervals vary (median 0.5667 s; maximum post-startup 1.6883 s); this is not a constant-frame-rate claim. Movie PTS is not the surface audit's simulation-time coordinate.

Representative native-rendered PNGs are under selected-scene-layers/:

- wholebody.png — Whole body, PTS 187.023 s.
- viscera-organs.png — Organs, PTS 71.150 s.
- lungs.png — Lungs, PTS 94.382 s.
- heart.png — Heart detail, PTS 125.327 s.
- vessels.png — Vessels, PTS 156.363 s.

These are frame buffers decoded from the continuous native recording, not images reconstructed from exported MRVPacks. The native log records semantic=51010 stable_id=22 layer_mask=0 payload_retained=true geometry_retained=true. The render-layer route masks that alias; the payload and geometry remain in the capture data. The separate native comparison report /Users/n/numi-human-resting-evidence-20261005/native-retired-alias-geometry-identity-018/comparison.json (SHA-256 0712bd1c95c8362ebc9aee3df404f1906d77d51c6cbf8459445277069217fdf8) reports captured geometry sections identical to viewer 017 across the eight corresponding captures.

The 20-second, 2-ms native trace has 1,250 observer rows, three completed breath-counter increments (at 5.024, 10.176, and 15.504 s), and 23 reported complete filling/ejection cycles (first at 0.400 s, last at 19.264 s). Cumulative aortic and pulmonary ejection are 1,670.21 and 1,663.76 mL; root-assistance force and torque remain zero. Integrated throughput from accepted step 8 to 10,000 was 182.793 s wall for 19.984 s simulated (RTF 0.1093); the full body log reports 185.608 s wall for 20.000 s (RTF 0.1078), while run metadata's outer wall time is 191.318 s. These are native model counters and visualization evidence, not clinical or long-horizon acceptance.

review-report.json contains the machine-readable details and input hashes. SHA256SUMS.txt checks the review reports, decoder copies, and representative images.

The file review-report.pre-hash-correction.json is retained history and contains the checksum typo corrected here; use the current review-report.json.
