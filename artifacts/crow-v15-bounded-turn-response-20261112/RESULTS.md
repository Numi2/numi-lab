# Bounded turn response: inadequate

Two zero-turn controls reproduced the preceding 24-second run, and six turn lanes matched its first 9 seconds: 5,100 state samples and 81,600 contact substeps. All first episodes lasted 24 s without reported physical failure or post-4-s speed-cap hits. Direct body moments remained disabled and course obstacle collision was disabled.

Positive .15 heading demand at .55 m/s produced about -.173 to -.174 rad body-yaw change over 12–24 s and path angles -.126 to -.129 rad. Negative -.15 demand produced about +.080 rad yaw change but path angles -.056 to -.059 rad. At .95 m/s the corresponding positive/negative yaw changes were -.174 and +.081 rad, while path angles were -.131 and -.046 rad. The zero-turn controls also drifted negatively, so these are not an unambiguous turn response. Sampled maximum roll was at most .003 rad; turning the body or changing a path angle modestly is not completion of a left/right target maneuver.

The response is sign-inconsistent and inadequate for navigation; no controller is qualified or frozen. Different initial seed lanes and the pre-existing drift limit the strength of comparisons. Results are development-only and do not establish realistic aerodynamics, recovery, unseen-layout navigation or landing.
