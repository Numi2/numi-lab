# Fix unpaired skin/fat basis capacity accounting

The source owner compared stable face-entry count with a limit on emitted basis regions. A closed synthetic shell with435 stable child identities and one emitted basis region was incorrectly rejected by the170-region limit. Count distinct emitted label values instead; donor assignment, partitioning, region limit and physical ownership remain unchanged.

The actual owner API regression is apps/NumiHumanUnpairedBasisRegionCapacityCheck.cpp. The unmodified parent header reproduces the false rejection. The corrected header accepts one region and still rejects a genuine output above170 regions when maximumFacesPerBasisRegion=1. Assertions are enabled with-UNDEBUG. Independent replay passes.

Build from the repository root:
clang++ -std=c++20 -O0 -Wall -Wextra -UNDEBUG -Iapps -Iinclude -Imatter/include apps/NumiHumanUnpairedBasisRegionCapacityCheck.cpp -o /tmp/numi-human-basis-capacity-check
/tmp/numi-human-basis-capacity-check

This compiler fix does not activate a new scene or establish anatomy, native-step convergence or performance. The compact036 evidence records exact28-file main-source closure and all49 local preparation artifacts; unchanged headers and test binaries are hash-pinned rather than duplicated.
