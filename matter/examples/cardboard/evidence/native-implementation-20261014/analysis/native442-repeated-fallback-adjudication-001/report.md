# Native442 repeated-Cauchy adjudication

Status: **failed_accepted_prefix_only**.

The source-bound run reports 1 accepted control steps out of the 27-control protocol; completed coupon = false. An accepted prefix is not a completed coupon.

It retained 128 residual-Armijo events, 64 Cauchy fallback records, and 4096 true-residual cycle events. Primary Newton records 0–47 and the first Newton-48 Cauchy trial are compared field-for-field to native435.

## Fallback trials

| Newton | Env | Accepted | Alpha | Base L2 | Trial L2 | Ratio | Outcome |
|---:|---:|:---:|---:|---:|---:|---:|---|
| 0 | 0 | True | 1 | 0.0048547294 | 0.00128042756 | 0.2637484923464529 | selected_trial_accepted |
| 0 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 1 | 0 | True | 0.5 | 0.00128042756 | 0.00080283999 | 0.6270093014867628 | selected_trial_accepted |
| 1 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 2 | 0 | True | 0.5 | 0.00080283999 | 0.000693773909 | 0.8641496657384992 | selected_trial_accepted |
| 2 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 3 | 0 | True | 0.25 | 0.000693773909 | 0.000638395315 | 0.9201777505876197 | selected_trial_accepted |
| 3 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 4 | 0 | True | 0.25 | 0.000638395315 | 0.000621670973 | 0.9738025301767761 | selected_trial_accepted |
| 4 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 5 | 0 | True | 0.125 | 0.000621670973 | 0.000572416349 | 0.9207705906513348 | selected_trial_accepted |
| 5 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 6 | 0 | True | 0.03125 | 0.000572416349 | 0.000566714909 | 0.990039697485999 | selected_trial_accepted |
| 6 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 7 | 0 | True | 0.25 | 0.000566714909 | 0.000492680701 | 0.8693625192768663 | selected_trial_accepted |
| 7 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 8 | 0 | True | 0.0625 | 0.000492680701 | 0.000477390917 | 0.9689661397960866 | selected_trial_accepted |
| 8 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 9 | 0 | True | 0.25 | 0.000477390917 | 0.00043332862 | 0.9077018530706565 | selected_trial_accepted |
| 9 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 10 | 0 | True | 0.0625 | 0.00043332862 | 0.00041729206 | 0.9629921513146306 | selected_trial_accepted |
| 10 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 11 | 0 | True | 0.5 | 0.00041729206 | 0.000408981898 | 0.9800855017466663 | selected_trial_accepted |
| 11 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 12 | 0 | True | 0.125 | 0.000408981898 | 0.000388711749 | 0.9504375399030498 | selected_trial_accepted |
| 12 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 13 | 0 | True | 0.125 | 0.000388711749 | 0.000372060837 | 0.9571638571696477 | selected_trial_accepted |
| 13 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 14 | 0 | True | 0.5 | 0.000372060837 | 0.000370822847 | 0.9966726140542441 | selected_trial_accepted |
| 14 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 15 | 0 | True | 0.0625 | 0.000370822847 | 0.000367044442 | 0.9898107545676655 | selected_trial_accepted |
| 15 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 16 | 0 | True | 0.125 | 0.000367044442 | 0.000358161575 | 0.9757989333618625 | selected_trial_accepted |
| 16 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 17 | 0 | True | 0.25 | 0.000358161575 | 0.000318064936 | 0.8880487416887197 | selected_trial_accepted |
| 17 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 18 | 0 | True | 0.0625 | 0.000318064936 | 0.000313634082 | 0.9860693415133318 | selected_trial_accepted |
| 18 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 19 | 0 | True | 0.25 | 0.000313634082 | 0.000291596923 | 0.9297360833380345 | selected_trial_accepted |
| 19 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 20 | 0 | True | 0.03125 | 0.000291596923 | 0.000287910225 | 0.987356869331574 | selected_trial_accepted |
| 20 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 21 | 0 | True | 0.0625 | 0.000287910225 | 0.00028732809 | 0.9979780676424396 | selected_trial_accepted |
| 21 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 22 | 0 | True | 1 | 0.00028732809 | 0.000233088213 | 0.8112266816655482 | selected_trial_accepted |
| 22 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 23 | 0 | True | 0.5 | 0.000233088213 | 0.000212349303 | 0.9110254880198512 | selected_trial_accepted |
| 23 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 24 | 0 | True | 0.0078125 | 0.000212349303 | 0.000211914841 | 0.9979540220106115 | selected_trial_accepted |
| 24 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 25 | 0 | True | 0.25 | 0.000211914841 | 0.000204440032 | 0.9647272981697398 | selected_trial_accepted |
| 25 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 26 | 0 | True | 0.0078125 | 0.000204440032 | 0.000200749811 | 0.9819496164038949 | selected_trial_accepted |
| 26 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 27 | 0 | True | 0.0625 | 0.000200749811 | 0.000185957761 | 0.9263159953859184 | selected_trial_accepted |
| 27 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 28 | 0 | True | 0.0625 | 0.000185957761 | 0.000184803532 | 0.9937930581988456 | selected_trial_accepted |
| 28 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 29 | 0 | True | 0.001953125 | 0.000184803532 | 0.000184628312 | 0.999051857948256 | selected_trial_accepted |
| 29 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 30 | 0 | True | 0.001953125 | 0.000184628312 | 0.000184509176 | 0.9993547251842935 | selected_trial_accepted |
| 30 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 31 | 0 | True | 0.001953125 | 0.000184509176 | 0.000184505159 | 0.9999782287250581 | selected_trial_accepted |
| 31 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 32 | 0 | True | 0.0009765625 | 0.000184505159 | 0.000184503864 | 0.9999929812260696 | selected_trial_accepted |
| 32 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 33 | 0 | True | 0.000244140625 | 0.000184503864 | 0.0001844816 | 0.9998793304404725 | selected_trial_accepted |
| 33 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 34 | 0 | True | 0.000244140625 | 0.0001844816 | 0.000184468066 | 0.9999266376700983 | selected_trial_accepted |
| 34 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 35 | 0 | True | 0.000122070312 | 0.000184468066 | 0.000184457909 | 0.9999449389793028 | selected_trial_accepted |
| 35 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 36 | 0 | True | 0.000122070312 | 0.000184457909 | 0.000184453136 | 0.9999741241781072 | selected_trial_accepted |
| 36 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 37 | 0 | True | 6.10351562e-05 | 0.000184453136 | 0.00018444944 | 0.9999799623900132 | selected_trial_accepted |
| 37 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 38 | 0 | True | 3.05175781e-05 | 0.00018444944 | 0.000184446864 | 0.9999860341131965 | selected_trial_accepted |
| 38 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 39 | 0 | True | 3.05175781e-05 | 0.000184446864 | 0.000184446093 | 0.9999958199343526 | selected_trial_accepted |
| 39 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 40 | 0 | True | 1.52587891e-05 | 0.000184446093 | 0.00018444522 | 0.9999952669097739 | selected_trial_accepted |
| 40 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 41 | 0 | True | 7.62939453e-06 | 0.00018444522 | 0.000184444565 | 0.9999964488101127 | selected_trial_accepted |
| 41 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 42 | 0 | True | 7.62939453e-06 | 0.000184444565 | 0.000184444332 | 0.9999987367478137 | selected_trial_accepted |
| 42 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 43 | 0 | True | 1.90734863e-06 | 0.000184444332 | 0.000184444187 | 0.9999992138549425 | selected_trial_accepted |
| 43 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 44 | 0 | True | 2.38418579e-07 | 0.000184444187 | 0.000184444041 | 0.99999920843263 | selected_trial_accepted |
| 44 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 45 | 0 | True | 1.90734863e-06 | 0.000184444041 | 0.000184444012 | 0.9999998427707405 | selected_trial_accepted |
| 45 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 46 | 0 | True | 9.53674316e-07 | 0.000184444012 | 0.000184443794 | 0.9999988180695181 | selected_trial_accepted |
| 46 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 47 | 0 | True | 9.53674316e-07 | 0.000184443794 | 0.00018444375 | 0.9999997614449417 | selected_trial_accepted |
| 47 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 48 | 0 | True | 1 | 0.00018444375 | 0.000160052223 | 0.8677562834197418 | selected_trial_accepted |
| 48 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 49 | 0 | True | 0.03125 | 0.000160052223 | 0.000155886693 | 0.9739739322458519 | selected_trial_accepted |
| 49 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 50 | 0 | True | 0.000244140625 | 0.000155886693 | 0.000155870934 | 0.9998989073429122 | selected_trial_accepted |
| 50 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 51 | 0 | True | 3.81469727e-06 | 0.000155870934 | 0.000155870628 | 0.9999980368373234 | selected_trial_accepted |
| 51 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 52 | 0 | True | 1.1920929e-07 | 0.000155870628 | 0.000155870541 | 0.9999994418448099 | selected_trial_accepted |
| 52 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 53 | 0 | True | 1 | 0.000155870541 | 0.000142245088 | 0.9125848097236027 | selected_trial_accepted |
| 53 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 54 | 0 | True | 0.0078125 | 0.000142245088 | 0.000142086239 | 0.9988832725106122 | selected_trial_accepted |
| 54 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 55 | 0 | True | 0.015625 | 0.000142086239 | 0.000141300494 | 0.9944699430041216 | selected_trial_accepted |
| 55 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 56 | 0 | True | 0.00390625 | 0.000141300494 | 0.000141099939 | 0.9985806489820198 | selected_trial_accepted |
| 56 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 57 | 0 | True | 0.0009765625 | 0.000141099939 | 0.000141036726 | 0.9995519983888868 | selected_trial_accepted |
| 57 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 58 | 0 | True | 0.00048828125 | 0.000141036726 | 0.000141008379 | 0.9997990097983415 | selected_trial_accepted |
| 58 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 59 | 0 | True | 0.000244140625 | 0.000141008379 | 0.000140997567 | 0.9999233237054658 | selected_trial_accepted |
| 59 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 60 | 0 | True | 0.000122070312 | 0.000140997567 | 0.000140993507 | 0.9999712051768949 | selected_trial_accepted |
| 60 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 61 | 0 | True | 3.05175781e-05 | 0.000140993507 | 0.000140990873 | 0.99998131828865 | selected_trial_accepted |
| 61 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 62 | 0 | True | 3.05175781e-05 | 0.000140990873 | 0.000140989912 | 0.9999931839559572 | selected_trial_accepted |
| 62 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |
| 63 | 0 | True | 3.05175781e-05 | 0.000140989912 | 0.000140989199 | 0.9999949429005957 | selected_trial_accepted |
| 63 | 1 | True | 1 | 6.78000447e-08 | 6.78000447e-08 | 1.0 | certified_noop |

## Limits

- No physical calibration or convergence claim follows unless all 27 controls complete and terminal acceptance is present.
- Status-gated post-failure residual sentinels are not measurements.
- A Newton-direction true-residual certificate does not certify the Cauchy fallback; record each fallback Armijo result and rely on the final transaction certificate for a committed step.
- The coupled rich capture is explicitly skipped for this two-environment run; no coupled operator audit is claimed.
- The source137 runtime messages contain stale trace-cap labels; selected all scope is 512 as encoded in frozen source.
