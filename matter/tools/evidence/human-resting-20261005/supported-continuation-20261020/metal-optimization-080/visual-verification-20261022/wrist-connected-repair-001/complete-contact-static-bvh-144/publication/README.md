# Complete per-body GPU bone-contact BVH
This increment adds deterministic load-time construction and validation of complete per-body BVHs. GPU closest-triangle queries traverse the full supplied bank rather than a cached rest-pose triangle neighborhood. Buffers stay GPU-resident; existing contact force, reaction, tangent and accept/reject owners are retained.

The full CPU/Metal/GPU coupon suite passed in full-coupon-run-004 on SSH numi. Coverage includes grouped multiple moving masters, force/wrench parity, coupled tangent finite differences, sub-ULP nearest-face ranking for both BVH and legacy ranges, and far exterior inactive versus interior fail-closed behavior. Earlier failed test binding attempts are retained.

This library increment does not create missing slave/body pairs. It does not establish anatomical clearance, arbitrary joint motion, sustained simulation or integrated performance. The private complete-Human producer's initial source preflight rejected its supplied transformed bank; investigation is retained separately in global-bone-static-bvh-147. No physical step was accepted there. Production anatomy is not qualified by these coupons.

The original worker publication map had stale hashes under /Users/n aliases alongside current /Users/numi aliases for three files. The correction receipt records these mismatches. Publication uses canonical paths, checked against the frozen final source hashes and run004 receipt.
