# Check every native rendered triangle

Previously the native volume checks covered the 12 functional respiratory and cardiac surfaces. A separate GPU reduction now checks every submitted triangle for zero or nonfinite binary32 area in the same presentation command. It reads the existing index and deformed vertex buffers, uses 64 bounded workgroups, and returns only aggregate counts and one exact failure witness. It does not change geometry, mechanical state, or controller history.

The existing surface trace and accepted receipt now include the total triangle count and invalid counts. A failed passive surface is identified by its actual primitive, semantic, stable anatomical identity, local triangle and binary32 coordinates. It is rejected before accepted geometry export or presentation. No artificial volume error is assigned to passive surfaces: that diagnostic uses the existing "NaN" sentinel and is explicitly inapplicable. A nonfinite mesh retains its compact failure receipt without attempting to serialize an invalid full geometry pack.

The compiled Metal regression passes valid, coincident, collinear and nonfinite fixtures across multiple workgroups and beyond one grid stride. It also verifies clean-buffer reuse, a single triangle, an empty input, exact witness positions and unchanged input geometry. The existing surface failure diagnostic test passes.

On the actual integrated 531 asset bundle, the new gate finds exactly nine collapsed triangles among 4,914,282 triangles, with zero nonfinite areas. It rejects source ID387, semantic51010, triangle568 at accepted time zero before the physical horizon begins. This matches the independent geometry census. The entire 2,833,965-vertex binary32 initial buffer is identical to the previous renderer; the diagnostic did not deform it.

This is the expected negative regression, not a completed human demonstration. The source eye repair remains necessary, and tendon self intersections and respiratory conditioning remain separate open work. The triangle-area check does not detect self intersections, validate anatomical relationships, or establish clinical plausibility. It runs at every presented accepted frame; the existing physical and physiological transaction gates remain responsible for each physical timestep.

The first retained native negative run reported zero for an inapplicable passive volume error. The final retained run corrects that receipt field to the existing NaN sentinel with applicability false. Both runs remain in the Mac mini evidence folder.
