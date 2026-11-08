# Integration verification

The integrated source at merge commit `c514358dd023fd1ae5553507ec8e09bbc103ada8` combines the main repair `5eb4e81f1` with the newer Window and training integration. On the Apple M6, the integrated build passed the `numi_human_costal_binding` and `numi_human_costal_cook_ownership` CTests (2/2). The integrated real-owner probe separately passed its forced rejection/restoration and initial-state replay checks; their full outputs are retained alongside this file.

The fixture uses synthetic two-body ownership. These checks do not establish anatomical ownership, sustained physiological motion, or calibrated forces. See RESULT.md for the baseline failure, scope and main-branch evidence.
