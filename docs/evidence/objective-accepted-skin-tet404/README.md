# Objective accepted-state skin embedding

Detailed skin vertices may now be embedded in a physical tetrahedron using four prepared affine weights. The Metal presentation pass gathers absolute accepted volume coordinates. It reuses the accepted presentation buffers and applies the same deformation gradient to normals and tangents. The existing three-node and exact one-hot paths remain available.

This addresses a mathematical defect in adding a fixed off-surface reference residual to a three-node displacement interpolation: that residual would not rotate with the body. The four-node embedding reproduces translations, rigid rotations and affine deformation, including nonzero offsets. Preparation must still certify source ancestry, reference reconstruction, region, conditioning and extrapolation bounds; this rendering helper supplies none of those certificates by itself.

The Apple M6 GPU check passed zero motion, translation, a 90-degree rotation and affine deformation, plus invalid-update immutability and invalid-row rejection. Maximum position error was 4.6242e-8 m and maximum normal/tangent error 6.48327e-8. Existing exact one-hot Float32 bit checks and legacy displacement tests also passed. The complete numi-human-native target compiled and linked on SSH Numi.

This increment does not admit the repaired Human physical mesh or qualify motion. Connected coarse boundary fitting, conservative mass transfer and native accepted physical steps remain separate gates. This code performs presentation only; it does not integrate physical state.

Commands:

    /opt/homebrew/bin/ninja -C /Users/numi/Work/skin-fat-native-integration-209/build-native numi-human-accepted-skin-direct-position-check numi-human-native -j1
    /Users/numi/Work/skin-fat-native-integration-209/build-native/numi-human-accepted-skin-direct-position-check

The enclosed source manifest pins the tested working files; the GPU receipt pins the check binary. Remote evidence remains under physical-transfer-209/native-tet-skin-transfer-404/.
