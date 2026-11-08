# Reproduce the isolated experiment

From the repository root on the Apple host, with its recorded integration Python environment:

```sh
export OPENBLAS_NUM_THREADS=1
NUMI_PYTHON=/Users/numi/numi-window-integration-venv-20261107/bin/python
$NUMI_PYTHON docs/evidence/rod-scaled-primal-20261108/scaled_primal.py
$NUMI_PYTHON docs/evidence/rod-scaled-primal-20261108/check_block_solve.py
$NUMI_PYTHON docs/evidence/rod-scaled-primal-20261108/energy_audit.py
xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math -c docs/evidence/rod-scaled-primal-20261108/rod_block_prototype.metal -o /tmp/numi-rod-block.air
xcrun -sdk macosx metallib /tmp/numi-rod-block.air -o /tmp/numi-rod-block.metallib
xcrun clang++ -std=c++20 -O2 -fobjc-arc -framework Foundation -framework Metal docs/evidence/rod-scaled-primal-20261108/rod_block_host.mm -o /tmp/numi-rod-block-host
$NUMI_PYTHON docs/evidence/rod-scaled-primal-20261108/metal_fixture.py
/tmp/numi-rod-block-host /tmp/numi-rod-block.metallib docs/evidence/rod-scaled-primal-20261108/metal-input.bin docs/evidence/rod-scaled-primal-20261108/metal-output.bin
$NUMI_PYTHON docs/evidence/rod-scaled-primal-20261108/metal_analyze.py
$NUMI_PYTHON docs/evidence/rod-scaled-primal-20261108/metal_heldout.py generate
/tmp/numi-rod-block-host /tmp/numi-rod-block.metallib docs/evidence/rod-scaled-primal-20261108/metal-heldout-input.bin docs/evidence/rod-scaled-primal-20261108/metal-heldout-output.bin
$NUMI_PYTHON docs/evidence/rod-scaled-primal-20261108/metal_heldout.py analyze
$NUMI_PYTHON docs/evidence/rod-scaled-primal-20261108/metal_negative.py generate
/tmp/numi-rod-block-host /tmp/numi-rod-block.metallib docs/evidence/rod-scaled-primal-20261108/metal-negative-input.bin docs/evidence/rod-scaled-primal-20261108/metal-negative-output.bin
$NUMI_PYTHON docs/evidence/rod-scaled-primal-20261108/metal_negative.py analyze
$NUMI_PYTHON docs/evidence/rod-scaled-primal-20261108/compression_control.py generate
/tmp/numi-rod-block-host /tmp/numi-rod-block.metallib docs/evidence/rod-scaled-primal-20261108/metal-compression-input.bin docs/evidence/rod-scaled-primal-20261108/metal-compression-output.bin
$NUMI_PYTHON docs/evidence/rod-scaled-primal-20261108/compression_control.py analyze
$NUMI_PYTHON docs/evidence/rod-scaled-primal-20261108/metal_trajectory.py
```

These commands overwrite generated input/output paths in the current checkout. Run them in a copy when preserving the committed bytes. The trajectory script invokes the exact `/tmp/numi-rod-block-host` and `/tmp/numi-rod-block.metallib` paths built above. Reproduction requires Metal and NumPy; CPU screen scripts with deliberately failed gates return nonzero. Review `RESULT.md` and each predeclared plan for the restricted physical model and acceptance gates.
