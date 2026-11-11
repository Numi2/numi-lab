# Greater knee swing: increased motion but failed support gate

The original physical knee drives and frozen flight controller were retained; the walking-only knee swing coefficient was increased from 0.40 to 0.60. The two previously used development seeds had eight lanes apiece. Before the walking change, the logged joint states, action output and contact-substep summaries matched the reference through step 199 for every lane. Both 300-step native runs returned success with zero reported simulator errors. See the prefix comparisons and run logs for the limits of parity.

Mean bilateral knee span increased by more than 15% in all 16 lanes (typically from about 0.233 to 0.354 rad). But the number of physics substeps with neither foot reporting upward support rose from 33 to 2,112 out of 25,600 during the compared walking windows. Both arms had no observed nonfoot contact and all lanes moved forward in aggregate. The preregistered support gate therefore failed: this controller is not promoted. Greater motion was accompanied by substantially more lost support; no claim is made that every unsupported substep was a fall.

The torso is a free rigid body, not an internally articulated trunk. Kinematic body responses changed in the candidate (see `body-comparison.json`) but are not validated as natural. No full journey, landing, or unused-layout outcome is established. Knee stiffness alone was tested separately and also failed its support criterion. A coordinated contact-aware gait is needed before promoting a larger swing.

`feedback.patch`, `feedback.metal`, compressed compiled library and `hashes.json` identify the candidate. All replay and contact recordings are preserved, including the negative outcome.
