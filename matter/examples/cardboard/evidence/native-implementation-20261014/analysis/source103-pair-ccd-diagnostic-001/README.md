# Source-103 pair-specific CCD diagnostic

This package prepares and compiles an isolated diagnostic for the exact two
captured states involved in the first fatal private Armijo CCD subcall. It
does not edit source 103, create a Matter world, or execute a physical
transaction. The only Metal dispatch is the standalone pair kernel, which
directly invokes the frozen source's production `vertexTriangleCCD` and
`edgeEdgeCCD` routines. Root owns that dispatch and will run it after the
currently queued native transaction.

The prepared set has 30 cases: 12 vertex-triangle calls (both directions for
each of three vertices) and 18 edge-edge calls (all 3 by 3 edge pairs). It
uses primitive IDs 92 and 127, candidate slot 283, Newton root 1201 / Newton
iteration 7, from center capture 359 and private trial 23 of capture 358.
Endpoints use the exact uploaded FP32 node state and source expression
`finish - dt * velocity`, with captured FP32 timestep. All nine float4 input
records per case are serialized little-endian and carry component bit
patterns in `prepared/prepared.json`.

The standalone kernel calls the production routine and a diagnostic trace
clone. The clone retains the production calculations and writes its actual
conservative-advance iterations, effective thickness, and speed bound. The
host compares clone summary values bitwise to the direct production result.
An uncertified result remains an observed failure; the diagnostic does not
reinterpret it as a collision or a miss. The independent CPU FP64 geometry
and conservative-advance values are comparative only, not formal CCD
certificates.

The current source speed bound is translation-dependent. For exact captured
endpoint displacements, a mathematical Galilean-relative Lipschitz bound is
also reported:

* VT: `max_i |d_vertex - d_triangle_i|`, from convexity of barycentric
  combinations.
* EE: `max_ij |d_edgeA_i - d_edgeB_j|`, from convexity of segment parameters.

Across these 30 calls the current absolute-endpoint-sum bound is 1.68–6.71
times the relative bound for VT and 1.82–2.54 times for EE. This is a
diagnostic hypothesis, not a production repair or safety argument. A future
replacement would need an outward FP32 rounding/error proof for interpolation,
closest-feature geometry, and the final distance test; this package changes
no hit math, effective thickness, or iteration count.

## CPU preparation and checks

From this directory:

```sh
python3 prepare_pair_inputs.py
python3 test_package_contract.py
python3 verify_pair_package.py --output cpu-audit.json
./build.sh
```

`build.sh` compiles the diagnostic Metal source and host program. It does not
run the host executable. The compiler emitted two warnings from untouched
source-103 `fgmres.metalinc` (`sharedValid` and `refinementValid`, existing
threadgroup initialization diagnostics); the standalone entrypoint compiled
and linked successfully. These warnings are retained in `compile/build.log`.

## Root-owned native diagnostic command

After the root schedules it, the exact command is:

```sh
compile/pair_ccd_native_check \
  prepared/prepared.json prepared/cases.bin \
  compile/diagnostic_pair.metallib native-results.json
python3 verify_pair_package.py \
  --native-result native-results.json --output native-audit.json
```

The native output must report zero physical transactions. Result files retain
per-subcall certification, TOI, separation, normal, witness parameters,
trace rows, and raw float bits. Clone/direct-production disagreement is a
diagnostic failure. There is no pass criterion requiring all cases to be
certified; the purpose is to identify the exact subcall and its trace.

## Bound evidence and limits

The input manifest binds source-103 build receipt and all 778 source paths in
that receipt, plus the capture manifests/payloads and their array hashes. The
CPU verifier rechecks those hashes. It also checks binary reconstruction,
case ordering, captured timestep/primitive/candidate identity, and the
absolute-versus-relative bound calculations.

No GPU dispatch has been performed by this package as sealed. No source,
threshold, transaction, or solver state was changed. The CPU FP64 CCD clone
is not an exact arithmetic proof; a noncertified FP64 trace is not grounds to
modify the production contact rules.
