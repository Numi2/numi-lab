# Storage recovery for midpoint qualification

Three unindexed temporary Git pack fragments, 3,439,165,438 bytes total, were retired only after archival to the private `Numi2/workstation-checkpoint-archive` release `numi-ccd-pack-fragments-20261109`, independent download into a new destination, and byte, size, mode and nanosecond mtime verification. Source identities, hashes, lack of open handles, remote digests and retirement receipts are retained beside this note and in the release. `restore_fragment.py` can download an asset and restore its original bytes and metadata to a new destination, using its corresponding source manifest.

These fragments are preserved as opaque bytes: pack validity and their usefulness for Git history were not established. No Human data was selected. The shared NumiLab Git store passed no-lazy-fetch connectivity-only fsck before and after each retirement. A full fsck was stopped when observed resident memory reached 5,755,616 KiB; it did not complete and is not reported as passed.

The three retirements recovered approximately 3.2 GiB. This creates operating headroom; it does not guarantee future availability. The 3.1 GiB simulation free-space guard remains unchanged, and all failed or interrupted studies remain separate evidence.
