# Reacting-slider anchor admission: work in progress

The explicitly requested one-axis midpoint owner now rejects nonzero parent or
child joint anchors before invoking the device-physics callback. Its existing
fixture uses zero anchors. A later compensated implementation must either
retain this restriction or qualify a low-word projection through anchor and
proxy offsets; the restriction alone does not implement compensated dynamics.

The source and native-probe changes passed Objective-C++ syntax checks. The
probe adds one rejected case for each axis of each anchor, but it has **not
been run natively**. The complete reacting-body trajectory, physical force,
work and replay requirements remain unqualified. The production admission
remains off. Neither the earlier collision-search option nor the AL contact
option is enabled by this change.
