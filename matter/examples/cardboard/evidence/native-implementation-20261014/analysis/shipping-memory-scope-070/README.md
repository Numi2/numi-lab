# Shipping sparse free-BOX memory-scope preflight (070)

This CPU-only preflight corrects the sparse private-contact model for the bounded native mode added to Matter: one non-articulated free BOX with six generalized rows, plus otherwise eligible proxy geometry. It separately verifies that a generic six-row free-rigid request remains rejected. The shipping scenario labels the bounded scope as an assumption; actual runtime admission must still validate the compiled proxy/body layout and borrowed buffers.

The generated geometry contains 453,378 base nodes and 1,477,728 tetrahedra (528,444 nodes after starch-interface duplication and 260,835 possible cohesive faces). The bounded sparse helper admits its modeled 65,536-candidate rigid arena, but the complete shipping private residual-descent arenas and native count limits remain rejected. This is not a full-box memory admission, native runtime measurement, solver step, or physical qualification. The 16 GiB value is compared against a modeled allocation envelope; host/device peak and full shipping Metal allocation were not measured.

Validation performed:

- `python3 matter/tools/cardboard_shipping_allocation_preflight.py --output <this-directory>/preflight.json` compiled and ran the CPU geometry/allocation preflight without initializing Metal.
- `clang++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -Iinclude -Imatter/include -Imatter/tools matter/tools/cardboard_shipping_memory_check.cpp -o /tmp/cardboard_shipping_memory_check-070 && /tmp/cardboard_shipping_memory_check-070` passed with `shipping memory preflight checks passed`.

`preflight.json` records the exact non-system translation dependency hashes, including the preflight source and runtime admission helper. It is the source-bound machine-readable result; this README is explanatory.
