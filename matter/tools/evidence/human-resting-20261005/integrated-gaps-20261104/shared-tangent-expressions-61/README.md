# Exact read-only tangent bundle sharing (61)

The native material source generator now value-numbers exactly identical
expressions across nine read-only tangent outputs. No algebraic identities,
operand reordering, precision reduction or material approximation is used.
Integer powers reuse the identical multiplication loop. Domain guards,
finite flags, output order and partial-output failure semantics are retained.
Outputs with cache stores, validity-changing fiber helpers, state or unsupported
operations keep the scalar path. Cache layout and consumers remain unchanged.

The two actual Human families emit 313/473 scalar values and 2/1 power loops
instead of 2,607/4,326 values and 18/9 loops. These are source work counts;
LLVM may already remove repeated instructions. They do not forecast GPU gain.

Apple M6 Metal exact bundle parity passes 8,192 cases (6,798 valid, 1,394
rejected, 805 nonfinite), including fixed-seed random directions and poisoned
inputs. Scalar Metal parity also passes all 25,344 cases. All 198 scalar
functions retain their arithmetic; 18 regenerated functions differ only by
removing consecutive duplicate finite checks with nothing between them.
The complete FGMRES suite passes 2,743,464 checks and the full material/frame,
plastic-history and rollback suite passes. Generator fixtures verify pure
admission and cache-writing/stateful fallback.

Human trial 307 accepts 32 roots and reproduces all 1,798,453 endpoint records
and numerical traces of control 306 exactly. Solves/columns/exhaustions remain
144/2,304/62. Body time is 225.036409 s versus 227.225529 s, approximately
0.96% less time in one pair. No useful or sustained performance gain is
established by this comparison. Body rate is 0.00000222186x, far below 0.1x.
GPU allocation remains about 7.86 GB.

This is 0.5 ms of unsupported release motion, not supported rest, full breathing,
physiological plausibility, presentation qualification or completion.
The source-only preparation exited 1 after successfully generating source;
it executed no physics and has no performance result.
A premature oracle invocation without the installed fused source and an initial
material-check invocation without its required asset are retained as failures.

Reproduce bundle parity with:
python3 matter/tools/check_native_finite_guards_metal.py BEFORE_GENERATED \
 matter/src/metal/generated_stateless_tangent.metalinc --include matter/include \
 --output OUTPUT_DIRECTORY --bundles

Compile the small codegen check against the existing compiler library, then
pass an output directory. It uses the existing cardboard-expression fixture.
No new runtime option, persistent buffer, simulator or artifact format is added.
