# Source identities, transform, and preparation commands

This record distinguishes exact anatomical identities, source hashes, and the registration hash. None is substituted for another.

## Registered source mapping

- Active NHA payload SHA-256: `3b0acb6146358f3302f2b85e2214bafa9f2fae7b0e51af70fd0838d6609da4a4`
- Active anatomy receipt SHA-256: `76c6651a86e9353af9007ba6433beb2a7605b607d9c261def9537ab447d43a51`
- BodyParts3D registration recipe SHA-256: `31c77660a16e8688fe68489eadff9d3861ceb27ed3b73d0288bda330d2a13b8c`
- NHRIGID2 rest-pose file SHA-256: `2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4`
- BodyParts3D v4.0 IS-A archive SHA-256: `40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e`
- Part-Of archive SHA-256: `9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97`
- IS-A table SHA-256: `a3de74423f943b0d724ae8f59b3a817f87c423a544f8db98113b1980817cbeaf`

Anatomical identities are BodyParts3D members FJ1772 / FMA50878 (left optic nerve) and FJ1819 / FMA50875 (right optic nerve), each classified by the IS-A relation. OBJ SHA-256 values are FJ1772 `314b4a735672c1ce6690d29199856569fa3db4864f2d94690e091432e085e057` and FJ1819 `3fde0d489978522bb0c916db61287c42aedfb4c6660b6f8016a148ffccef89f4`.

## Exact source frame and seam handling

The transform is the registered `global_source_mm_to_myosim_world_m` matrix followed by inverse NHRIGID2 body-index-23 rest pose. The recipe was checked against unchanged BodyParts3D FJ1730, mapped to active anatomy stable ID 322 with identical ordered triangle topology: vertex RMS error 5.86331065e-8 m; maximum error 6.11365981e-8 m. No arbitrary registration/refit was used.

Original OBJ seams were welded only where source coordinates were exactly equal. No faces collapsed; the 1,190 / 1,191 canonical vertices retain all OBJ aliases in JSON. All 2,376 / 2,378 source faces were preserved; the seams leave one connected closed component per nerve with zero boundary edges, non-manifold edges, or winding conflicts. No caps, synthesized geometry, or geometric defect repair were added.

## TetGen preparation

TetGen 1.6 (August 2020), external binary SHA-256 `6de4b560676dc348115208c1e75568638a92d2d498213b17cc61516ba4267dc6`, was used only during offline asset preparation and is not included in this bundle. The retained preparation script ran:

```sh
TetGen -pd FJ1772.poly
TetGen -pYq1.5a1e-9 FJ1772.poly
TetGen -pd FJ1819.poly
TetGen -pYq1.5a1e-9 FJ1819.poly
```

The intersection-only runs exited 0 before meshing. The quality runs exited 0. `-Y` preserves source boundary triangles and face markers; `.node` point markers retain canonical source vertex ancestry and alias JSON records all welded OBJ vertex IDs. The asset test reverses TetGen's inward output boundary winding to outward while preserving the source-face marker.

The exact per-file SHA-256 values of bundled input data are in `ASSETS.sha256`. The CPU runner checks them before use. Evidence reports record positive finite double and Float32 tet volumes, source/tet volume closure and source-face marker retention.

The source-preparation pass was run as `python3 /Users/n/numi-human-separation-20261018/C/nerve-volume-240/prepare_optic_tetgen.py` on numi against the hashes above. The CPU-only adapter pass is rerunnable from this bundle via `./tests/run_tetgen_organic_fem_cpu.sh`; the test fixture uses identity coordinates only and is not a runtime pose registration.
