# Status-5 Armijo-alpha lane correction (diagnostic-only overlay)

Base: `/Users/home/cardboard-implementation-20261014/snapshots/source-status5-vtccd-diagnostics-125`, bound by `frozen-source.json` SHA-256 `dc354c1d0c9619b07d33de0c867239dc260f7fe572372769d6e5e9dde7b023e5`.

The source-125 observer passed `femLineSearch` to `nm_fem_status5_capture`; its `.x` is private scratch and remained zero while a trial was being evaluated. The kernel then recorded that zero as if it were an available alpha. The actual private Armijo trial alpha is written to `femArmijoMetrics.z` by `nm_fem_armijo_begin_trial` and `nm_fem_armijo_begin_bounded_trial`.

This overlay changes only the capture helper/kernel argument name and source, and the four private status-capture call sites. Stages 30, 31, 40, and 41 now bind `state.femArmijoMetrics`; the kernel records `.z`. Global stages continue passing `nil`, so alpha remains explicitly unavailable there. ABI sizes, field offsets, physics, status production, and acceptance are unchanged. Source 125 is unmodified.

Validation:

- `verify_alpha_lane.py`: passed; binds the two frozen source hashes, all four private call sites, all four global unavailable call sites, and the kernel's `.z` read.
- Objective-C++ syntax check: passed with Apple clang using source-125 headers and the copied overlay runtime.
- Metal compile and metallib link: passed for `matter/src/matter_kernels.metal` with the candidate `fem.metalinc` and all other source-125 includes.
- `patch --dry-run -p1` against source 125: passed.
- No native GPU run was made; this overlay is not natively qualified.

The compile used a temporary lightweight source mirror, symlinking unchanged source files to frozen 125 and copying both changed files as regular files. That temporary mirror is not part of this package. To repeat syntax validation, use the compiler command recorded in `manifest.json`; to repeat Metal validation, build a temporary source mirror the same way and run the two Metal commands recorded there, writing outputs outside source 125.
