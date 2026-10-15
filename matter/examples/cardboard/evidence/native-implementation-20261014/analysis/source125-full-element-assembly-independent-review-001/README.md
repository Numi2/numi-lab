# Independent review of 150-002 element assembly

This read-only review independently checked the 150-002 input/file hashes and reassembled the captured environment-0 tangent using scalar Python math.fsum loops. Its small NPY/NPZ reader is standard-library-only; the 1920 local 12x12 matrices and all 2538 assembled columns were recomputed without NumPy, BLAS, SciPy, or the source assembly helper. The captured cached tangent matrices were also compared to the 419 JSONL records after FP32 conversion for all 1920 tetrahedra.

The independent values reproduce the published maximum element-column relative error (3.859290877495561e-5) and maximum assembled-column relative error (5.113862904090013e-8) to within 5.3e-17 absolute. The assembled matrix relative difference is 1.4565263069886152e-8; independent scalar assemblies agree with their retained sparse matrices at 1.46e-16 and 1.60e-16 relative. No warnings or floating-point exceptions occurred. The separate mass diagonal is included unchanged.

The 150-002 report is supported as an analytic same-center element/frame/assembly comparison. It does not establish a fresh native all-column action sweep, native finite-difference noise floor, solver convergence, cohesive-interface behavior, material calibration, or physical qualification. The capture is one environment/control/Newton center; the fixture has finite elastic starch geometry with its original substrate connectivity and no independent cohesive interfaces. The inherited 150-001 “untied” phrase is corrected in the 150-002 record and is not interpreted here as an unbonded limiting control.

Validation command:

    python3 analysis/source125-full-element-assembly-independent-review-001/review.py
    python3 analysis/source125-full-element-assembly-independent-review-001/seal.py

Three reviewer-script attempts are retained as logs. The first found that the active system Python lacked NumPy/SciPy; the review was rewritten to use a standard-library-only decoder. The second caught an indexing error in the reviewer's own B-matrix construction by producing zero reference columns, before any report was emitted; it was corrected to the documented delta(a, displacement) convention. A third stopped on a report-schema lookup before output. The final scalar reconstruction passed all hash, input-identity, cache-to-capture, matrix, and gate checks.
