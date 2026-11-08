# Build identity and reproduction limits

The preliminary passing stress used helper source commit fd09e20dd and source-pinned stress fixture inherited from the axial branch. It compiled with:

    xcrun clang++ -std=c++20 -O2 -I<worktree>/include -I<worktree>/numi/measurements <worktree>/numi/measurements/rod_local_frame_stress.mm -L<owner-build>/lib -lmetalrobo -Wl,-rpath,<owner-build>/lib -o <run>/probe

It ran against the pinned multi-anchor shader and existing standalone owner dylib. The modified rod shader was compiled separately with:

    xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math -I <multi-anchor-worktree>/include -c <multi-anchor-worktree>/src/metal/DiscreteElasticRod.metal -o <run>/rod.air

The metallib was linked from the exact AIR list in linked-air-hashes.txt, substituting the multi-anchor rod AIR for the earlier task-local rod.air entry. The substituted AIR and resulting metallib hashes are retained in the multi-anchor evidence; the owner dylib and metallib hashes are in runtime-hashes.txt. The exact tool versions in this directory describe the host used for this record. This evidence retains hashes and source, not the actual executable, dylib or all AIR binaries; an independent clean rebuild and hash reproduction have not been demonstrated.
