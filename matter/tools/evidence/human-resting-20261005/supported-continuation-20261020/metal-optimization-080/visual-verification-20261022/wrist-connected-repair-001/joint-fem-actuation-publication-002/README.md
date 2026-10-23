# Shared-owner actuation app patch - alias-safe revision 002

Narrow follow-up to publication-001 at ../joint-fem-actuation-publication-001/README.md. It fixes a C++ range-aliasing hazard in one-time ground-aligned release placement: an explicit root-pose path can make poseQ span overriddenPoseQ, so assigning that range back into the same vector is unsafe. The corrected code copies into an independent vector before changing root translation and rebinds the span after the move. Physics, gates, and placement values are unchanged.

Apply patches/alias-safe-root-placement.patch after the publication-001 composed app patch. patches/composed-clean-main-v2.patch is the one-shot patch from base commit 75bd5ba2ff33d3a4875b5303395755ab817faa0f; that app file is byte-identical at remote main eb31b83f58941b3aa3ba22047afb7a5f2f18979a. Both patch forms passed isolated apply checks.

The corrected app translation unit compiled and linked with -Werror against frozen ABI15 inputs. All 243 tracked C/C++ header entries match current remote main. The linked library source manifest differs in one unrelated non-header Metal source file, so this is not a full clean-main core rebuild. No GPU or physics run was performed for this source-only fix.

Publication-001 shared-scene tests remain failed at zero accepted steps; exact receipts and limits are preserved there and are not repeated or promoted by this patch. See preflight/span-alias-source-preflight.json, preflight/abi15-main-source-comparison.json, and preflight/build-result-abi15-alias-safe.json.
