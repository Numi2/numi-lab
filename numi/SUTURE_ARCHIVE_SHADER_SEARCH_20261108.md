# Earned suture checkpoint: additional archive search

The retained step-1870 synthetic-skin checkpoint needs device-program
fingerprint 17848984240238310200. Its exact native restore remains blocked.
The earlier investigation and input hashes are retained on branch
codex/skin-suture-fingerprint-diagnostic, commit 9cc78d22. This search does
not alter the checkpoint or weaken its program check.

Four previously uninspected release archives were downloaded under
/Users/numi/suture-archive-search-20261107/unmanifested-archives. The three
plastic-bag archives matched their release SHA256SUMS for the archive itself;
the Human archive matched the release asset's GitHub SHA-256 digest
a2eaa4f28cc7dffecfb96c1929fa4619320d8270ae9a9099ec5be910472276db.
All archive directories were scanned for .metallib and .air entries.

The three plastic-bag releases each contained one PlasticBag.metallib. The
existing checkpoint-specific fingerprint helper computed the following values:

| Release | Shader SHA-256 | Computed program fingerprint |
| --- | --- | --- |
| approved | e3dd0f103b461ec642bfd4b4c619b1dd98ad62166e7f89cf0f3f9c767e4d85e | 4414940737996868035 |
| empty | dc6f0d7f8da12c1a4bb621069d64385c1dd75287a522b952cee14fd9082a6c43 | 326368869255105401 |
| handoff | 72c610246d38d3421ff805bbaa35176378cb1eb3d1fa7e2a37fb66b97b2554e4 | 17370286549973192246 |

None matches the required fingerprint. The Human solver/contact-memory archive
listed no .metallib or .air entries. The helper is a search aid; a calculated
match would still require an actual native restore, identity and physical
validation. No matching shader or successful suture continuation was found.
The downloaded originals and extracted shader files remain on the host.
