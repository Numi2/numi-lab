# Registered skin/fat donor-frame correction

Vertices in a registered shell can have different donor-bone frames. The outer offset already maps a world normal into each donor frame. The inner fat offset incorrectly triangulated those mutually incompatible local coordinates. It now maps its world normal through the same per-node frame.

The isolated nine-node planar fixture uses alternating donor rotations and translations. It checks each fat offset against its world displacement mapped into the node's donor frame. Before the fix it fails with a 2.64766 mm discrepancy; afterward the maximum discrepancy is 1.30104e-18 m.

This verifies loading/preparation coordinates, not whole-body contact stability or anatomical qualification. The rejected whole-body reduction candidates remain outside production.

Run on SSH Numi (Apple M6), without visual inspection:

    clang++ -O2 -std=c++23 -Iapps -Imatter/include -Iinclude tests/human_skin_donor_frame_normals.cpp -o /tmp/human_skin_donor_frame_normals
    /tmp/human_skin_donor_frame_normals
