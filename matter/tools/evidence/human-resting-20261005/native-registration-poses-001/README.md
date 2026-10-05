# Native anatomy registration poses

The optional accepted-MRVPack receipt now includes the exact body poses for
the relevant thorax/abdomen/pelvis owners from the already completed and hashed
presentation buffer. Initial source-organ bounds and body poses are calculated
once from the existing native source pack and `initialBodies`. Offline anatomy
tools can therefore use the real registration rather than fitting a transform
from a rendered mesh. This adds no GPU state or per-step readback.

The SSH Mac mini built and ran 32 native steps at 2 ms, exporting steps 0 and
31. The first coupled CSV row is byte-for-byte equal to the corresponding row
from the previous six-second run. Step-zero captured body poses equal the
initial registration poses exactly. All runtime inputs had unchanged hashes
after execution. The initial compile rejected a GNU shorthand conditional
under the existing warnings-as-errors policy; the corrected standard C++
conditional built successfully. Both results remain in the build log.

The upper visceral anchor group's most caudal coordinate is −0.259147108 m
(right kidney, stable ID 4); the bladder/prostate group's most cranial
coordinate is −0.397011578 m (bladder, stable ID 462). These are source geometry
bounds in native torso-local coordinates, not measured ligament positions.
The subsequent passive attachment candidate must declare that distinction.

A separate negative admission run deliberately increased the physiological
diaphragm area by 1%. The native loader rejected its mismatch with the
source-derived area in 0.488 seconds, before any physical step started.

These are metadata and fail-closed admission checks. Known anatomical
interface defects remain; this is not anatomical acceptance or another
physiology endurance run. Large packs and the short native recording remain
at the corresponding original paths under
`/Users/n/numi-human-resting-evidence-20261005/` on the Mini.
