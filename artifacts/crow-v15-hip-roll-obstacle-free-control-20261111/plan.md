# Explicit obstacle-free 35-body control

For flight diagnosis only, build the same hip-roll host with course-object collisions disabled but ground retained; this is never a navigation or landing result. Keep the frozen flight law untouched. First rebind a zero diagnostic policy to the changed world and confirm native contract, then assess the flight envelope against independent collision-enabled controls.
