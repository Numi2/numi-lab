# Restore source required by the main build

A fresh CMake configure of main at `910aa100b10de9e95115bfffd0b45456a432e5ad` failed because `CMakeLists.txt` names `apps/NumiHumanConformingTissueRefinementCheck.cpp`, which was absent from that revision. Restore the exact file from existing commit `6bd04d58d731c07f95be4d417df9f1034aad7931`. Its accompanying refinement header has the same Git blob on both revisions (`71ef839ecf2a660879934744cc1e892aef1d9a49`).

On Apple M6, fresh CMake configure succeeds after restoration. The file compiled independently with clang++ C++20, `-Wall -Wextra -Wpedantic -Werror -UNDEBUG`, and the executable reported: `conforming tissue refinement synthetic shared-edge and rejection checks passed`. This verifies the offline synthetic check only; it does not validate anatomical assets, runtime mechanics or loaded physiology.
