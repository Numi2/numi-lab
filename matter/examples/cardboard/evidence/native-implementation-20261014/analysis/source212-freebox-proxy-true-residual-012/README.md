# Source 212 overlay 012 — preserve one-free-BOX true-residual scope

This isolated candidate starts from immutable source207-buildfix-002 and fixes the 011 admission regression. It admits the previously supported single free dynamic BOX without capsule endpoint state, while retaining the same exact BOX, FEM-pair, current-body, and solver-layout gates. Prescribed capsules remain optional; when present, endpoint inputs and at least one FEM-pair per required shape are checked. Physical operator, convergence thresholds, iteration limits, and other unsupported-body rejection are unchanged.

Validation: 13 Python source contracts pass; the extracted C++ admission checker passes 20 cases at O0 and O3 with byte-identical stdout. No GPU dispatch or physical transaction occurred. The 011 build is not a build of this 012 source; rebuild from this exact frozen manifest before any native claim.
