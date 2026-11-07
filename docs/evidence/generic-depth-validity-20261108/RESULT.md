# Result: sensor-invalid samples can become false near depth

On 2026-11-07 the pinned generic synthetic kernel process ran from 23:44:55.493852 to 23:44:55.689539 UTC and exited 7. Binary and shader checks passed before and after. Existing synthetic baseline gates passed.

For input marked FRAME|GEOMETRY with no DEPTH validity, far-cleared depth and the SensorPack's existing Gaussian observation noise, 3 of 8 pixels became less than the normalized far value in each of the unsegmented and instance-selected paths. In the independent direct-winner arm, depth-sensor dropout probability was one and the same 3 pixels became near. Complete printed outputs are retained; counts reflect this one synthetic seed and corruption setting.

The shader currently tests nonzero aggregate validity, which does not imply a usable depth measurement. The direct-winner branch additionally keeps the segmented selection after measureDepth reports failure. The behavior is established for this kernel fixture; a full-render task rollout and other renderer profiles have not been qualified. Object-track reduction also tests nonzero aggregate validity, but has not yet been tested here.

Any behavior repair must account for observation-policy fingerprint compatibility, both renderer paths, depth edges and intentional corruption, and live owner/reset contracts. This evidence does not qualify camera realism or a deployed policy.

