# Source attribution and limitations — native resting scene

This file accompanies the scene and movie from the 8 October 2026 native
preflight at final-native-scene-preflight-936/skin-927-lung-924-viewer-018-v014-final-attempt3/.
Keep it with copies of the scene folder, native-viewer.mov, and extracted
frames. It records upstream credits and limits; it does not grant rights beyond
the source licenses.

## Exact preflight identity

The retained invocation and run metadata bind this attribution record to the
following outputs and composed source inputs:

- Accepted horizon: 10,000 roots at nominal 2 ms (20 s); the native terminal log reports 20.000000949949026 s after Float32 step accumulation and exit code 0.
- Native viewer movie SHA-256: 46c72bc48cbb49e65bd521554d8fdd0f9f78ab5464664d22e39a746c7d06b5e8.
- Invocation SHA-256: 98e4ec56d5fededb8794e515d8c3cd27a8a794a50a8a4778653bb1157ca7e2c9.
- Composed anatomy receipt SHA-256: fca3c96b1c71f1bb2c4e14ed4a924dff2b9fe436f3f81c8ce6a321506585a84e.
- Scene manifest SHA-256: 6bb0ee85bf5a228190ee80f471d87c23e40d359bca278bccdf66a7ab75c82c01.
- NHSKIN 927 source manifest SHA-256: 120b78369a8b74a4719eccc9ef416a44086574dc5ddcddbc55b721e2d06ccf99.

The 936 scene manifest and run metadata remain the authority for the exact
composed invocation; the payload fingerprints below identify the source assets.

## Direct BodyParts3D 4.0 material

The scene loads direct BodyParts3D 4.0 meshes and atlas identities through the
is-a and part-of archives. The official license is CC Attribution 4.0
International and requires this credit:

> BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International

- License and download: https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html and https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html
- isa_BP3D_4.0_obj_99.zip: SHA-256 40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e.
- partof_BP3D_4.0_obj_99.zip: SHA-256 9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97.

This is the direct 4.0 database credit. It is separate from the credit for
BodyParts3D content incorporated into the Z-Anatomy package, whose pinned
license names CC BY-SA 2.1 Japan.

## Z-Anatomy source package

The source catalog declares Z-Anatomy repository revision
9a2ef22cc8443d14f9aa18ba91246416b1c6ab7a, bound to the archive and Blender
scene below. The raw export does not itself contain a Git revision field; the
revision is catalog-declared and linked by the matching archive/scene hashes.

- Project: https://github.com/Z-Anatomy/Models-of-human-anatomy
- Pinned license: https://github.com/Z-Anatomy/Models-of-human-anatomy/blob/9a2ef22cc8443d14f9aa18ba91246416b1c6ab7a/License.txt
- Z-Anatomy.zip: SHA-256 e029688545627bd0214b269e1063143abb580aad72b2c2445d6d8a9a0d9da736.
- Z-Anatomy/Startup.blend: SHA-256 9f08a17ea0115fed80b2a73ecdf0a1bc2ab2f6956f37c593ce23d513ea35afcd.
- License: CC BY-SA 4.0 for Z-Anatomy source content, with attribution and ShareAlike requirements.
- Required attribution from that revision: Z-Anatomy - The libre 3D atlas of anatomy - CC-BY-SA 4.0.
- The same license separately requires the included-source credit BodyParts3D - The Database Center for Life Science - CC-BY-SA 2.1 Japan.

The 924 anatomy receipt identifies Z-Anatomy-derived scene inputs as stable
IDs 305–311: five lung-lobe meshes (305–309), a pleura proxy (310), and
source-derived diaphragm geometry (311). Stable IDs 14–21 are inferred liver
exterior display patches whose names and FMA crosswalks use Z-Anatomy segment
guides; their rendered geometry is one closed aggregate shell, not eight
source segment meshes, internal Couinaud partitions, or independent segment
volumes. The selected Z-derived scene set does not include the separately credited cranial-nerve, inner-ear,
or kidney reference components from the upstream license. Its inner-ear and
kidney components carry additional non-commercial terms.

The pleura is an external lung-union proxy. It is not a parietal pleural layer,
interlobar lining, fluid domain, or pleural mechanics model. The lung meshes are
registered atlas envelopes, not parenchymal tissue, measured gas volume, or
clinical registration.

The historical stable-ID-22 row in the packed NHA payload is traced in retained
lineage to BodyParts3D member FJ2824 (source SHA-256
2f5e34fceea6c8b718bd3f50a125bf24c434ae7eb3d925b9795b0585272c1349). The
001 and 007 payloads preserve its 11,459 vertices, 21,592 faces, and exact face
topology; the later neutral-pose rebase carries it from body 7 to torso body 20
with unchanged face topology and a recorded selected-set maximum world-position
error of 1.60e-8 m. The 924 receipt intentionally retires ID 22 as an alias to
aggregate liver display patch 21 and excludes it from the active source-ID map.
The packed legacy row remains present; the retirement records that it is not a
separately identified liver segment or separately defined volume. This lineage
is an inferred registration/rebase, not a new source measurement.

## MyoSim articulated model

The active body and muscle model is based on MyoSim myofullbody, pinned at
commit 33c89c2bde282553dde3f526768eb3bdcfaa7649 from
https://github.com/MyoHub/myo_sim. The pinned upstream license is
https://github.com/MyoHub/myo_sim/blob/33c89c2bde282553dde3f526768eb3bdcfaa7649/LICENSE.
The retained source manifest identifies
the archive SHA-256 as
280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975 and the
license as Apache-2.0. The source manifest also records local knee-coordinate
overlays; this preflight uses those compiled source-derived payloads.

MyoSim supplies articulated body/joint/muscle definitions and source tendon
sites. It is distinct from the anatomical surface providers listed above.
The local Human notices and manifests retain the Apache notice; model rights
remain bounded by upstream terms and any separately identified source models.

## Loaded scene payloads and inferred content

The invocation binds the following source-bearing payloads and scene supports:

It also loads the Numi-format reference-circulation configuration
resting_reference_lv15.native.v3.json (SHA-256
43ff6b49daf1d42cf9e88d85d8577f70e112dfcc3756ec43954e544a3e4bc0dc) and the
resting-reference-respiration.json file (SHA-256
c518926bf47fba945cef52bb952ed6c559d604508988082c5b641b720bda503d). These
parameter/configuration files are not 3D anatomy assets; their scientific
source lineage is recorded with the circulation reference and native study.

| Payload | SHA-256 | Source / role |
| --- | --- | --- |
| NHA anatomy, 924 | 3c444be7736c066a992988cc32b687917e1c4c5c3968a16b4d5f0106d5b5024e | BodyParts3D 4.0 and selected Z-Anatomy geometry/identity references |
| NHSKIN, 927 | bd4bfbbf0e071e24a1bb9eea7b9cd9f34ef862ef20417b5891cd24cec1b009d1 | BodyParts3D FJ2810 shell with inferred source-position correction |
| NHTISS4 | b3d0381f73e05b04ed7ea1ba23eb6559073ad20aa0d9e2ad0f97a18fab3359bd | 148 passive muscle surfaces and 2 tendon surfaces; mixed BodyParts3D/MyoSim reference geometry |
| NHTENDON3 | 49daaf61421254cb18f3aa32f1d332b1810537afd4c4ef813abd2f0d118dcecc | 832 attachment endpoints with inferred surface envelopes and explicit point fallbacks |
| NHBONES3 | 7d94e2f28ac6cea30c6c11a79919077acc40281636eb20b11b50b50aebd3ec49 | 185 BodyParts3D 4.0 registered bone surfaces |
| NHRIGID | 2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4 | MyoSim articulated rigid-body source model |
| NHMYO | e5bb8a8168706bb3b23cf42b1d849659569ba2e6027bcc85729c3110fdbb8c4b | MyoSim-derived authored muscle records and parameters |
| NHEQ | 0e3c593792a8c4ccd619f1999f6685b68e6ff0fd5ae5f6f0bca66b594a0e9b88 | Compiled joint-equality constraints for the MyoSim articulated model |
| NHCNT bed support | bcfece8e5da553b98694b724644234407fa4c38383b1e18d3c24d7caefa28927 | 32-region point-support definition derived from registered geometry |

The NHSKIN correction changes referenced positions and regenerates their rest-world normals: its manifest reports the canonical
86 bindings, per-vertex influence records, weights, triangle indices, and
registration fingerprint preserved. The 0.25 mm directional target used by the
offline clearance solver is an engineering constraint, not a measured skin
thickness. Runtime bed support is a reduced 32-region point-witness method
whose selected support vertex is chosen from the complete skin at each step;
it is not a triangle-mesh collision simulation.

NHTISS4 provides passive inspection surfaces, not a force-transmitting tissue
continuum, calibrated constitutive law, or collision response. NHTENDON3
contains inferred attachment envelopes and point fallbacks; its endpoints are
not source-authored or measured human entheses. The reduced circulation model
owns chamber blood volumes and flow accounting; anatomical heart surfaces are
visual references rather than independent blood-volume compartments.

## Evidence boundary

The linked native-viewer.mov is from a 10,000-step, 2 ms (20 simulated second)
native integration preflight using the hashes above. It is not the separate
310-second paired study, a full breathing-cycle anatomy qualification, a
measured-person registration, or clinical validation. A later NHA/NHSKIN hash
change requires updating this source fingerprint block before reusing the
sidecar. Keep the upstream credits with any distributed derivative.

## Viewer018 display correction

Native inspection now masks only the receipt-retired organ identity (semantic 51010, stable ID 22). Active liver patches and bone/muscle identities are unchanged. The legacy geometry remains in exported MRVPacks. This attribution derives from retained 944; only preflight identities and this display note changed.
