# Catalog input binding

The selector writes a versioned catalog identity containing its evaluation
arguments and hashes of the evaluator and supported self-contained file
inputs. The shared Swift catalog and guided auto-load decision verify those
files at the time of use. Old metadata and unresolved multi-file dependency
closures yield unknown and do not auto-load; manual research selection remains
independent. The selector continues to bypass evidence reuse for unresolved
inputs. These checks are fail-closed and are not a complete reproducibility,
physical qualification, or authenticated provenance guarantee.

`producer-consumer.py` exercises actual Python `_evaluate` metadata and the
compiled Swift decision with a synthetic CPU evaluator. The retained corrected
output confirms the same-path replacement, deletion and unsupported URDF
closure cases. Its first two failures, including a macOS path normalization
mismatch, are retained in `initial-failures.txt`. The separate Swift test
covers old and missing metadata, missing listed inputs, an altered evaluator,
policy/evidence changes, and existing selection roles; Python selector tests
also pass. None of this is a native physics or UI test.

The identity does not cover dynamic-library contents, process environment,
operating system, all external dependencies, or mutations during evaluation.
The decision checks current files, so moving or deleting a previously valid
input yields unknown until another supported evaluation records valid paths.
