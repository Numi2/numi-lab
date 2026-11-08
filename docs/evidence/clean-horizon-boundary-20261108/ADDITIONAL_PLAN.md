# Reset and repeat boundaries

With two environments, scheduled reset for environment 1 occurs at zero-based step 220. For 220 steps both environments should be clean; for 221 steps only environment 0 should be clean. Test chunks 1 and 8. For one environment with no state trace, compare 128 steps x 2 repeats (the timeout is exactly the final requested step, clean 1) with 256 steps x 2 repeats (the timeout occurs before the end, clean 0). Record counts and require zero physics failures.
