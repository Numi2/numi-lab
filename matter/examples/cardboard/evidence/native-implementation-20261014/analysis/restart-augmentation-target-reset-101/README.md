# Restart target-scale reset-order repair

Against frozen source100, the per-Newton setup captured the base-RAP target scale, then dispatched `nm_fgmres_regional_restart_reset_metadata` later in the same encode, zeroing that target before the restart-mode kernel consumed it. Native case 341 records a nonzero raw mode norm but `target_scale=0`, `mode_scale=0`, `mode_valid=0`, and cleared q/Aq. An independent check of that capture's base RAP found finite, nonzero coarse diagonals; the zero target is therefore explained by dispatch ordering, not a missing diagonal.

The patch moves the existing reset dispatch, with the same seven buffers and same initialization marker, to precede regional basis projection, base RAP construction, and target-scale capture. It changes no RAP formula, rank gate, scale threshold, physical operator, or solver tolerance. The patched `runtime.mm` hash is also the runtime hash in frozen source101.

Run the source-order regression with:

```sh
python -m unittest -v /Users/home/cardboard-implementation-20261014/analysis/restart-augmentation-target-reset-101/test_reset_order.py
```

The tests verify the exact source100 hash fails the required order and the patched runtime hash passes `reset -> project basis -> build RAP -> capture target scale`, including all seven reset-buffer bindings. These are static source checks only. Root has build-checked source101; the matched native rerun is required to determine whether the mode becomes admissible. Case 341 remains preserved as the pre-fix failed native result.
