Integrity incident record

The original package 001 was sealed with manifest SHA-256 75e1326759f2089a6f8c04cb94cf27d97091465e12d17f6d1105d338d7d8504e, patch SHA-256 1dad9db30931b963497c9a0153dd5fb3c04c812f7f1678b59fc33e69845f5598, and expected candidate SHA-256 2727109d414ed75662d2f135002d4b04bfa6ed1a095459008362c8a7ab5a065f.
A later edit changed candidate bytes after sealing. The observed post-seal bytes are preserved in post-seal-001-candidate-preserved.mm with SHA-256 4d1aca6a92f2bb28c0c3aa15f37e32bf531dfb091d71cef3792c2ecc9a375d9e. Root detected the mismatch before source131 was cloned or patched; no integration or build used those bytes.
The original candidate is restored from the unchanged package-001 patch against the frozen source-129 base, then verified against the expected candidate hash. Package 002 is a separate expanded diagnostic candidate; it does not revise package 001 evidence.
