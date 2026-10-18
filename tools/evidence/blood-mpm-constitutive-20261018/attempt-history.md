# Retained test-attempt history

The final CPU receipt records the passing integrated numi-matter-stateful-check run. Earlier attempts were not promoted:
- Standalone attempt 002 failed compilation because two diagnostic newline character literals were accidentally written as multi-character literals. Corrected in the isolated staging source.
- Standalone attempt 004 compiled but failed its fixture admission because it enabled adaptive MPM without a valid rigid-proxy binding. The CPU constitutive fixture was changed to non-adaptive; no solver admission rule was weakened.
- The first isolated integrated-source link attempt failed because the fresh overlay lacked its output bin/ directory. The build script now creates that directory before linking; source and test were unchanged.

The original logs and JSON receipts remain in /Users/n/numi-human-separation-20261018/C/bleeding-liquid-205/.
