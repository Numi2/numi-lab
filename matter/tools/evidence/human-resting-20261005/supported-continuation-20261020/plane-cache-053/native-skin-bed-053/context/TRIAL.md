# Plane arithmetic and cache-only block assembly 053

SSH Numi only, same source pose, masses/materials/physical tolerances, 34 registered ligament rows. No visual verification.
Fix fixed-plane arithmetic scale to the signed-distance dot products; tangential coordinates do not enter this distance. The 052 diagnostic collision thickness was numerical inflation, not proof of the authored clearance. Preserve the 052 failed run and confirm the compiled behavior before changing geometry.
Selector6/cacheMode2 block assembly uses cached-only pass, marks due-tet cache misses in existing inverse-block scratch, then existing AD fallback only for those rows. No extra physical state or altered due mask; other selectors retain old path.
Verify CPU contact arithmetic/compiler/support, native material-frame cache parity, nonzero coupled fields, due/non-due two-object block fixture with bitwise cached block parity. On first failure retain witness and repair before broader checks.
Then native scalar125us control and sole FEM-skin bed125us. If accepted, eight roots and two-root forced-late-reject replay. Optional mode6 trial is bounded180wall seconds and is not a physical qualification if interrupted. Test larger timestep only after a passing control. Target0.1x is not yet met.
