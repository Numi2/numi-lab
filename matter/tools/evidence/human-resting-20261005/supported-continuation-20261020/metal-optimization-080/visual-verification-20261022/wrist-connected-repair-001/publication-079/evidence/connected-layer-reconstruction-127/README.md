# Connected layer reconstruction 127

This helper creates a construction-only inner/deep surface proposal from a supplied outer surface, per-node or scalar positive skin/fat depth fields, and exact fixed-boundary coordinates. It solves only anchored connected components in the induced active surface graph. The caller owns and must include the patch interface boundary. Vertices outside the active patch retain their original layer offsets exactly.

The focused 20-test suite passes. This does not create volume cells, transfer source ancestry or material mass, or qualify native or physical behavior.
