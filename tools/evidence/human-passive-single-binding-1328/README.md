# One-body passive anatomy reader

A passive source surface bound to one skull body was rejected by the generic native tissue loader, although the existing blend/render path handles one positive binding. The loader now accepts one or more bindings for all supported passive surface layers. Duplicate bodies, malformed transforms/weights, invalid topology indices, missing bindings and duplicate identities remain rejected. Active FEM selection retains its separate two-body endpoint requirement.

The narrow reader change at 1c68fc9 was compiled and exercised by a native 16-second/8,000-step integrated resting-human run on SSH macmini (Apple M4 Pro). Both superficial masseters loaded and passed eight actual-capture self/topology checks and scans of 862 surrounding surfaces. The other 861 surfaces, all 157 accepted body poses and 1,000 x 60 coupled physiology values matched the parent exactly. The complete source/asset/recording evidence is in Numi2/numilab-human/tools/evidence/native-anatomy-masseters-1328.

The original run's literal DYLD_LIBRARY_PATH check is false because the new build uses symlinks to the retained runtime. The included native verification records the mismatch, proves canonical directory equality, and verifies all other invocation fields and input hashes. No physiology or force owner changed.

The same change was compiled on main d07c088, preserving the intervening opt-in MTP mechanics addition. Native evidence is pinned to 1c68fc9, not that later compile. Full anatomy and final five-minute qualification remain incomplete.
