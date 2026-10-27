# Numi Human P187: corrected cap topology diagnostic

Construction diagnostic on SSH Numi continuing P186 commit 189c931242a64cc167b40c3fc006b16018d93914. The best full audit remains 653 comparable parent skin/bone intersections, 125 sampled inside node/bank pairs, zero skin self-intersections and 45 skin boundary edges. Nothing here is physically admitted.

The corrected bank 74/104 bone clipping uses canonical endpoint tokens and fixes the prior cyclic polygon insertion bug. Both plane sides now produce one connected component with all 25 contour edges and valid incidence, but neither cap passes projection/ancestry checks inside the original skin source face. The positive side projects to 2.5221693895 barycentric square units and the negative to 1.4072532353, versus contour area 0.3214222488; both also fail consistent orientation, centroid containment and source-barycentric ancestry checks. This rules out the tested single-face cap construction, not a larger multi-face reconstruction. No 3D candidate or physical-step test was produced.

Source and mixed-reference attribution are retained in P182/P183.

A bounded 43/16 boundary expansion retained a simple loop with no exact target-bone edge hits and no inside boundary nodes. Minimum edge-to-bone distances were 0.014878 mm at the initial boundary, 0.027003 mm after one ring and 0.038572 mm after two. No physical thickness or clearance threshold was claimed; these narrow gaps require further anatomical assessment. No skin vertex moved.
