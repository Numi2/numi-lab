# Cold completed output recovery bundle

This bundle preserves 49 cold, completed reports, logs, target JSONL files, and one nearest-union report from historical `contained-clearance-545` work. These paths were outside active 575+ and round-033 work. Each payload has an individual Zstandard archive under `payloads/`.

`receipt.json` records the exact P142/P143 immutable GitHub commits and manifest hashes, each published member SHA-256/byte count, local source and publication-stage paths, original mode and nanosecond mtime, compression hashes/sizes, restore checks, and retired paths. Fresh streamed downloads from GitHub raw URLs matched all 49 remote manifest entries (718,007,530 bytes total). Both remote manifests are retained under `remote-manifests/`.

To restore selected files into a new tree, run:

```sh
python3 restore_cold_outputs.py --receipt receipt.json --destination-root /path/to/empty-root --only contained-clearance-545/final-one-021/fit/report.json
```

Repeat `--only` for additional members. To restore every listed source and publication-stage copy:

```sh
python3 restore_cold_outputs.py --receipt receipt.json --destination-root /path/to/empty-root --restore-all
```

The destination root receives `D/contained-clearance-545/...`, `publication-142/...`, and `publication-143/...` paths. Restoration checks archive SHA-256, restored content SHA-256 and bytes, mode, and mtime. This recovery bundle does not change any geometry or qualification status.
