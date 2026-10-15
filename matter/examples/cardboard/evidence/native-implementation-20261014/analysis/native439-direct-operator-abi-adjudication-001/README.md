# Native439 direct-operator ABI adjudication

This is a CPU-side integrity and result audit of the already completed native439 suite. It does not launch Metal work or alter the retained run. `report.json` records each case receipt, stdout/stderr counts and hashes, build identity, and scope. `verify.py` rechecks the immutable suite manifest, per-case receipts, source-build identity, and executable hashes.

All five native checks exited 0 without timeout and each run reported unchanged source inputs. This is operator/material-point regression evidence only; it is not a full coupon or physical box qualification.

The standalone `fem_regional_material` case took 214.518 seconds, within the recorded 240-second bound. Available historical regional-material CTest entries were CPU-only, so they cannot establish a comparable native Metal baseline.

Direct-tool cache coverage is mode 0 in four tools and modes 0 plus 1 in `fem_material_frame`; this suite does not directly select mode 2. Source review found no mode 0/1/2 change or unbound buffer 26.
