# Current working anatomy, skin and fat hidden

This renders the latest installed anatomy assembly directly from its prepared source pack on SSH Numi, Apple M6, on 2026-11-04. Model selection follows the repaired source lineage, independently of whether a physical trajectory was accepted. It does not select an older checkpoint by acceptance status.

The selected assembly includes the zero-failure skin/fat source lineage 1020, current433 physical volume and mass mapping, regional1061 preparation, and dominant-frame042 registration. The registered outer skin has 76,193 vertices; the inner skin and deep-fat surfaces have 118,197 and 103,509 vertices. The current source-domain audit and exact typed asset SHA-256 identities are in preview-verification.json. All 108 pinned runtime/input paths were reread: zero missing or differing hashes.

The source file is native-scene-view-229/install-motion/resting-human.mrvpack, freshly assembled in this session from those inputs. Geometry is read from that prepared source file, not an advanced accepted geometry packet. The existing time-zero receipt supplies the complete 157 initial body frames; its 86 source owner frames match the original anatomical registration exactly. Nonzero-time receipts and mismatched source packs are rejected.

Only visibility and rigid placement are changed for inspection: the outer skin, inner skin and deep fat are hidden, together with four duplicate ocular presentation instances. The 5,725,951 source triangles retain their coordinates and identities. This is not a physical cut. Bones, muscles and vascular anatomy are retained; deeper organs may be occluded by superficial muscles.

The existing Metal renderer produced three 1536-square PNGs. The original command, renderer/binary hashes, output hashes and native device are retained. Derived preview packet caches are removed after rendering; original source packs and receipts remain. No physical equation, material, mass or solver was changed.

Validation: native build and all three source renders passed; the existing accepted-replay path also passed a native regression render. Four negative cases reject advanced source poses, mismatched source geometry, incompatible layer modes and missing receipts.

Remaining gaps:

- Visible vascular-looking routes beside both forearms lie outside the surrounding muscle envelope. Their exact source IDs, routing and cross-structure clearance need measurement and repair. A hidden skin layer does not resolve this.
- Source-level zero skin/fat quality failures and audited skin-to-bone crossing counts do not establish zero intersections among every tissue in every native pose. The complete scene still needs motion qualification.
- Fresh physical trial 229 reproduced all ten baseline traces byte for byte, but advanced only 0.000500000024 simulated seconds in 229.663 seconds of simulation-loop wall time. Initialization-inclusive time was 436.224 seconds. This release transient is not settled supported rest; it does not demonstrate a full breath, heartbeat or five-minute integrated run.
- Sustained dynamics, solver throughput, full tissue/cardiovascular/respiratory coupling and continuous native-viewer recording remain qualification work.

The red structures in these views are anatomical meshes. The trial configuration disables blood MPM and bleeding; visible red geometry is not demonstrated blood leakage. The assembly remains a mixed-source reference adult, not measurements of one person.

Storage: the verified historical-evidence cleanup reached 30.6 GiB free on Numi. The private GitHub release https://github.com/Numi2/macmini-evidence-archive/releases/tag/numi-20261104-cold-human-evidence contains 29 archives, file manifests, independent recovery checks and RESTORE.py instructions. 1,788 retired historical paths are recoverable; current sources, inputs and run packets were retained. Storage verification reports are included here.
