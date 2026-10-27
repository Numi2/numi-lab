# Selected-axis SAT experiment package

This package captures the tested 638 selector/adapter and the 697-pair construction trial. It is not a physical or native qualification.

## Outcomes

- The finite-axis selector chose 680 of the 697 typed skin-face/bone-triangle crossing pairs. Its report is box-only selection evidence, not a collision certificate.
- The fit solved successfully and the existing 582 retention plus full checks completed on `dd855a4792591b0acc34ee5c1926a75f7fdaa9309f269393871d0c1c19a02688`: 725 surface pairs, 0 contained node-bank pairs, 0 self pairs, complete 185-bank coverage. This regressed the 697-pair baseline and remains preserved as a negative result.
- A separate component-merit extraction from the raw fit produced `0c84cfa08c1da9c944b6f0922d69ee3e83f0e41faf1ade705867ee4ef4487b48`, with 695 surface pairs, 0 contained, 0 self and all 185 banks. The only bankwise change was -2 in bank 114/117. This is an unadmitted construction candidate.
- All candidates have `accepted_physical_steps=0`; native admission is false.

## Verification

The focused tests were rerun with `/Users/n/numi-human-prep-venv-20261005/bin/python -m pytest -q test_connected_patch_sat_selector638.py test_sat_axis_map_adapter638.py`: 20 passed. The suite includes an actual 697-pair frozen-637 row-builder smoke with the adapter installed; it is diagnostic and does not solve or emit geometry. Python compilation passed. The failed initial default-Python test invocation is retained.

`manifest.json` records SHA-256 and size for every staged item. The 52 MB full fit report is gzip-compressed and its restored stream was hash-verified against the source. Most staged evidence is a same-filesystem hardlink to its frozen source, not another disk copy.

The first selector and fit launch failures remain in the package beside the successful retries.
