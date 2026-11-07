# Earned synthetic-skin checkpoint restore investigation — 2026-11-07

The source of this investigation is the independent
`codex/synthetic-skin-suture-20260930` lineage at commit
`8ce44aca14f850dc05259c93d5d723e255ae3237`, not the later
`tissue-receiver-bridge-start` phase. The documented first-bite checkpoint is
a qualified bounded puncture state, not a completed stitch.

From the [skin-suture-history release](https://github.com/Numi2/numi-lab/releases/tag/skin-suture-history-20261014),
the downloaded Mac mini archive matched receipt SHA-256
`4818d1f9f35d6e77aad520fd35c02f20710734a796ef14405c0446e870ec334c`.
Three independently extracted input hashes matched its manifest:

| Input | SHA-256 |
| --- | --- |
| step-1870 TSV | `513f41761ef3cd42fd9c62a0b8f6542a4e9987a57abd3e281674ce0ddb6e9faf` |
| Matter sidecar | `a5a0fcf0e9f347998152e3919ecea0ef25ddc744716400c3f7297cc6f6e371a4` |
| rigid-contact cache | `e48012d195acc63cf2e48ad4f9d62128c2bef5a49f7c05300bf29d558909ca02` |

The branch's CPU Matter reader accepted the sidecar and reproduced the
TSV's recorded content hash `4745382909208999294`, payload size
`30316752`, physics fingerprint `17338282561063946900`, and program
fingerprint `17848984240238310200`. The branch's rigid-cache reader
accepted the exact TSV and step, reporting 16 manifolds and multiplier 1.

On Apple M6, a separately rebuilt branch binary attempted the v3
`--tissue-checkpoint-restore-only --synthetic-skin` command and rejected
the checkpoint: **Matter snapshot does not match the initialized device
program**. The conditional four-microstep continuation did not run. The
checkpoint was not modified. This is a failed compatibility check, not a
physical trajectory result. The retained local evidence is
`/Users/numi/suture-archive-search-20261107/step1870-native-bpcnxwhs/`.
That run's executable and Matter shader SHA-256 values were
`30bc08781d8dfebf64ae9f690cdcaee7044f69175afc3510e599d6f2cc14b341`
and `a983c833a26fbd3014f1a57079cf0f3a5b389786a3f16ad66c5794dab5831568`.

The subsequent CPU-only cook diagnostic, which returns before Matter
initialization, reported world fingerprint `10068538913772727962`,
physics fingerprint `17338282561063946900`, and material state stride 0.
The latter two match the snapshot. The device-program fingerprint includes
the exact Matter metallib bytes, ABI, cooked world and configuration. A
source-level reconstruction of that fingerprint tested the three distinct
Matter metallibs in the same release archive and 37 locally visible Matter
metallibs; none yielded the archived program fingerprint. This calculation
is a search aid, not a native compatible restore. Branches ignore
`*.metallib`, so a source commit alone does not retain the original image.

The source-level search can be repeated with
`tests/surgery/program_fingerprint_match.cpp`, an input list of candidate
paths on stdin, and the recorded cook/configuration. The code is deliberately
specific to this checkpoint and prints every calculated fingerprint. Any
future candidate must also pass the native restore, cache identity and physical
certificates; its presence or a matching reconstructed number alone is
insufficient.

No fingerprint check was overridden. A possible future versioned migration
would need a complete mapping of all authoritative Matter, rigid-contact,
rod, articulation and topology state, an immutable receipt binding both
programs and source state, and independent checks of the discrete equations,
forces, clearance, rollback and replay. A direct fingerprint rewrite would
establish none of those. With no old compatible image to replay as an oracle,
deterministic equivalence and safe promotion remain unestablished.

## Additional release inventory check

A read-only GitHub release inventory on 2026-11-07 listed eight releases. The
Numi and Mac mini 2026-11-06 closure manifests contain no `.metallib` or `.air`
entries. The five 2026-10-21 Human replay manifests list `.mrvpack` replays and
no shader files; the cardboard release index describes four cardboard evidence
roots and does not list a shader. These metadata observations do not establish
the contents of any archive that lacks a complete file manifest.

The three plastic-bag release tarballs (110.8 MB, 154.3 MB and 526.9 MB), and
one Human solver/contact-memory tarball (373.4 MB), remain uninspected. Their
release dates and subjects differ from the surgical checkpoint and offer no
specific evidence that they contain its exact device program. They cannot be
ruled out by filename. No large archive was downloaded for this inventory.
The required device-program fingerprint remains `17848984240238310200`;
exact native restore remains the final compatibility test.
