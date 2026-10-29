# Source face / donor partition regression

One registered anatomical source face may span several mechanical donors. The unpaired reduction-label builder previously rejected this despite valid per-cell dominant-parent/donor provenance. It now partitions basis regions using `(stable source face, donor)` while retaining the original source identity. All parent material, donor, ancestry and region-capacity checks remain.

The new fixture reproduces the old rejection, accepts two valid donors under one source identity with separate basis labels, and rejects a child whose donor disagrees with its dominant parent. Assertions-enabled unpaired-volume and reduction-label checks pass on SSH Numi (Apple M6). This is label/transfer software evidence; it does not qualify a repaired physical volume or motion.
