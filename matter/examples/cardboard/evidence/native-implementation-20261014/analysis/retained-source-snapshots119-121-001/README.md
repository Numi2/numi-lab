Exact source119, source120 and source121 restoration from fixed Git base7ebc64a666c4bd5e1ff61b3684655979ab80dc60 and hash-checked gzip blobs.

Run `python3 restore.py --repository REPO --number 121 --output NEW_DIRECTORY`. The output must not exist. Each snapshot retains its actual build receipt hash; restoration does not reclassify any failed physical run.
