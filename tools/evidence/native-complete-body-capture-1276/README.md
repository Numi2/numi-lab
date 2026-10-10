# Complete accepted body pose capture

The native resting geometry receipt previously exported only bodies used by the skin registration. That omitted two body owners used by passive muscle anatomy, including the abdomen. The opt-in geometry export now adds accepted_body_pose_count and accepted_body_poses for every source body, read from the same already completed and hashed presentation buffer. The existing accepted_registered_body_poses field is preserved.

Native source revision 015ff2e was built and run on the SSH Mac mini (M4 Pro, 24 GB, macOS26.6) for 8,000 accepted 2 ms steps with the integrated resting human. At eight captures:
- all 157 body indices are present in order and every tissue binding owner is covered;
- all 86 legacy registered poses equal the prior capture output;
- full body and respiratory state hashes exactly match the preceding native run;
- 128 referenced vertices per abdominal surface 67/68 per capture forward to within 1 micrometre of actual native geometry.

All 1,000 rows and 60 coupled trace columns exactly match the retained baseline prefix. No forces, controllers, stepping or buffer ownership changed. This only extends existing opt-in capture serialization. The 15 existing accepted-MRV-pack audit tests also pass.

The [Human twelve-surface evidence](https://github.com/Numi2/numilab-human/tree/main/tools/evidence/native-anatomy-twelve-surfaces-1276) retains the exact launch declaration, native recording, geometry checks and all source/asset hashes. Local paths in the verifier require retained Mini assets. This verifies capture coverage, not whole-body anatomy or five-minute qualification.
