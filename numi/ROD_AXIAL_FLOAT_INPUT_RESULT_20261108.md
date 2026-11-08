# Rounded-input axial reference

The [rounded-input comparison](ROD_AXIAL_FLOAT_INPUT_PLAN_20261108.md) keeps the original model intact for the owner and evaluates the independent FP64 reference using copies rounded to the FP32 values supplied to the device. At 16 m, the 5- and 17-node cases now satisfy the stated gates, while the 128-node case still publishes success with a 0.00248206 N maximum support-force error against approximately 0.09844 N, exceeding the 1% gate. Its position and velocity errors are just below the 1 micrometre and 0.001 m/s limits.

This isolates an inaccurate accepted support force even when the reference starts from rounded inputs. It does not separate FP32 solve arithmetic, final-state rounding and reaction arithmetic. The earlier unrounded-source comparison remains the relevant application-level observation for requested geometry; neither run warrants promotion.

The initial version of this diagnostic failed before GPU dispatch because changing the actual model's rest lengths violated its rest-geometry validation. That stderr and binary/source identity are [retained with the corrected run](measurements/evidence/rod-axial-float-input-20261108); no result is attributed to the failed attempt. The corrected source is metal_rod_axial_float_input_v2.mm at commit 3ac394e8b, using the unchanged candidate shader.
