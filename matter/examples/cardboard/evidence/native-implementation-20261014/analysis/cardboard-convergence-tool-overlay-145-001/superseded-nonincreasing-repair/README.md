# Superseded comparator attempt

This retained artifact proposed changing the resolution triple gate from strict `fine < coarse` to `fine <= coarse`. It was rejected after the plan language was rechecked: the required differences must decrease, so equality must fail. The patch is not included in the active package patch, is not applied to source143, and must not be used. The active source overlay preserves the strict comparator and has an explicit equality-rejection test.
