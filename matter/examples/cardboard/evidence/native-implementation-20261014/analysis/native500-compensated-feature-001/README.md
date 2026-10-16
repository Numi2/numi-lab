# native500 compensated-feature audit package

Remote package path:
/Users/n/cardboard-implementation-20261014-host-macmini/analysis/native500-compensated-feature-001

Read-only audit of the already-retained native500 Metal replay. This package does not contain a production source patch, does not launch Metal, and does not run a Matter transaction. It compares exact captured FP32 endpoint words using:
- a C++ two-FP32-word closest-point routine transcribed from the source176 VT branch and using the source176 compensated geometry primitives;
- an independent 90-digit Decimal closest-point and IPC barrier-energy replay;
- center-gradient dot products against each captured rounded endpoint jump.

The slot7 replay has eight alphas: 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128. All 24 endpoints (current, contact-limiter, FEM-applied) remain on edge_bc in both two-word and Decimal evaluation. Across them, maximum pair-distance error versus Decimal is 3.222e-18 m; native FP32 separation error reaches 3.0835e-11 m. Corrected exact barrier energy still increases for all eight limiter endpoints. The exact FEM-applied endpoints have energy increases at the first three alphas and decreases at 1/8 through 1/128. See report.md for the full table and limits.

## Bound inputs and source

Source snapshot:
/Users/n/cardboard-implementation-20261014-host-macmini/snapshots/source-full-contact-private-scratch-176-003
Frozen source manifest SHA-256:
3bb43be51637a47494e33ead26c33702060cb09ab1f12c90f33602974bca6d92

Source file SHA-256:
- include/metalrobo/compensated_geometry_gpu.h: 1bf85c7e7b8b2102c5e0e6f3fbba88e2b30530e6b5fbb33375fb581d5261d84e
- include/metalrobo/compensated_translation_gpu.h: a04166e5c8ef76db05271c8c9b084f0b2edc9bb4a7e6bfe1986527a4f8c09097
- matter/src/metal/contact.metalinc: b1dc8d0e6843d5aed3a8c6d28449dae8c4d3a60488e644195e9fed5ed121d346
- matter/src/metal/common.metalinc: 61ace798a14c1ca0ba1f2cfb5c8161ec43c9e7a0f8a6b05bb1ef0e9f9ca3380
- matter/src/metal/fem.metalinc: 33edccfec4a2faee1b0bfd7a59aa7847ac707ac3c56baa9a0030e858eeba7170

Captured case manifest SHA-256:
dc62eb0462fbf4af9f495363c63bb5ef692213643da0f318602eba3b91c08370

Native500 replay receipt:
/Users/n/cardboard-implementation-20261014-host-macmini/native/contact-barrier-energy-replay-500/receipt.json
Receipt identifies source closure SHA 8427a7f247282ac8488061975174830b9ccfaaba4918dff626a2b9673b03ca41, host SHA 97c0b4953a9e8b60058f62227a8febe0d9bc2cc843c1bf63a948b93377b5b4f4, metallib SHA 5e91e698eaa76a28dee887e884f29df7bf412e3f3241bd1d6a489d72d44a9f9e, and output SHA cb1f014b70d5bd21e08f385e4fbea2c6d857328b2fd4a9c0506d64cb18ff0237. It records zero physical transactions and ccd_qualification=false.

## Reproduction commands

On macmini, from /Users/n:

Q=/Users/n/cardboard-implementation-20261014-host-macmini
SRC="$Q/snapshots/source-full-contact-private-scratch-176-003"
P="$Q/analysis/native500-compensated-feature-001"
"$Q/venvs/cardboard-reference-185/bin/python3" "$P/decimal_reference.py" > "$P/decimal-reference.jsonl"
"$Q/venvs/cardboard-reference-185/bin/python3" "$P/energy_armijo_analysis.py" > "$P/energy-armijo.json"
"$Q/venvs/cardboard-reference-185/bin/python3" "$P/add_slope.py"
"$Q/venvs/cardboard-reference-185/bin/python3" "$P/realized_gradient.py" > "$P/realized-gradient-summary.json"
clang++ -std=c++17 -O2 -fno-fast-math -Wall -Wextra -Werror -I "$SRC/include" "$P/closest_vt_dd.cpp" -o "$P/closest_vt_dd.rebuilt"
"$P/closest_vt_dd.rebuilt" "$Q/native/contact-barrier-energy-replay-500/output.bin" > "$P/dd-output.rebuilt.jsonl"
cmp "$P/dd-output.jsonl" "$P/dd-output.rebuilt.jsonl"

The C++ replay uses mrCompensatedVectorAdd, mrCompensatedVectorScale, mrCompensatedVectorDot, mrCompensatedDivide, and mrCompensatedSqrt. The independent Decimal reference uses the exact binary32 endpoint words from native500 output and evaluates the same edge/face branch order. The energy replay uses the exact binary32 thickness and stiffness from the captured contact record; its exact Decimal energies are diagnostic comparisons, not values produced by the native FP32 kernel.

## Evidence limits

The corrected separation is accurate for these exact captured endpoint words and the tested edge_bc branch. This does not prove other VT branches, EE features, the production gradient chain, general numerical error bounds, or CCD safety. The reported barrier Armijo table uses the native recorded directional derivative with the source Armijo coefficient and does not alter them. No conclusion here qualifies a full contact trajectory or justifies an acceptance-tolerance change.

The next candidate must unify the FEM position update order used by contact line-search trials and by FEM apply. A CCD sweep from the current candidate to each exact helper-produced endpoint still needs to be certified; this package does not claim that endpoint replacement alone proves the sweep conservative.
