# Exact finite-check lowering (60)

Native float material programs accumulate finite flags within straight-line
blocks, publishing them before every domain guard, expression-cache write,
fiber helper and return. Every scalar operation retains its order and precision.
Intermediate nonfinite values still reject even if a later min/select masks them.
No material, solver tolerance, physical acceptance or runtime flag changes.

All 198 actual generated scalar programs pass exact C++ and Apple M6 Metal
parity on 25,344 cases: 20,438 valid, 4,906 rejected, 1,541 nonfinite.
The actual old/new source generator passes two manufactured expression-cache
fixtures (2,304 cases). Return bits, flags and partial cache writes agree.
The complete FGMRES contract passes 2,743,464 checks.

Full Human trial 306 accepts 32 roots / 0.5 ms, reproducing all 318,974 node
and 1,479,479 tetrahedron endpoint records and numerical traces exactly.
144 active solves, 2,304 columns and 62 budget exhaustions are unchanged.
Body time is 227.225529 s versus 231.373590 s for matched control 304b:
1.79% less time / 1.83% higher throughput in one pair only.
This is provisional, not a repeated or sustained speedup estimate.
Body rate is 0.00000220046x, far below 0.1x. This is an unsupported release
transient, not supported rest, breathing qualification or native presentation.

The original generated source and generator are identified by the scoped diff
and parent commit. Full execution pins and dirty-source patch are retained here;
the full compressed endpoint remains on Numi. No physics check was relaxed.

Reproduce the bounded Metal oracle:
python3 matter/tools/check_native_finite_guards_metal.py BEFORE_GENERATED \
 matter/src/metal/generated_stateless_tangent.metalinc --include matter/include \
 --output OUTPUT_DIRECTORY

The tool discovers the installed standalone Metal toolchain; full Xcode is
not required. Offline scalar/generator tests do not replace native integration.
