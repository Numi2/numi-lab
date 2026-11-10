# Failed capture attempt

The exact used-J descent was rerun with the built-in protected CCD-capture flag enabled and reproduced the same failure, but no snapshot was written. Source inspection showed this failure is a pre-advance event-queue overflow, which the protected capture path does not currently handle. A separate passive host diagnostic obtained the event counts. No nonexistent snapshot is represented as evidence.
